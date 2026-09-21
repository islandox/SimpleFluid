#include <gtest/gtest.h>
#include "materials/MaterialProvider.hh"

#include <atomic>
#include <cstring>

namespace
{
using Provider = SimpleFluid::MaterialProvider;
using Input = Provider::Input;
using Properties = Provider::Properties;

struct Context
{
    mutable std::atomic<size_t> calls{0};
    int status = SF_MATERIAL_SUCCESS;
    int invalid_field = -1;
    size_t invalid_index = 0;
    double invalid_value = std::numeric_limits<double>::quiet_NaN();
    bool omit_derivative = false;
    bool unterminated_error = false;
};

int evaluate(const void* opaque, size_t count, const Input* inputs, Properties* outputs,
             char* error, size_t capacity) noexcept
{
    const auto& context = *static_cast<const Context*>(opaque);
    ++context.calls;
    if (capacity) error[0] = '\0';
    for (size_t i = 0; i < count; ++i)
    {
        outputs[i].density = 1200 - 0.4 * (inputs[i].temperature_kelvin - 300)
                           + (inputs[i].composition_count ? inputs[i].composition[0] : 0);
        outputs[i].specific_heat_capacity = 4000;
        outputs[i].dynamic_viscosity = 0.001;
        outputs[i].thermal_conductivity = 0.6;
        if (!context.omit_derivative) outputs[i].density_temperature_derivative = -0.4;
        if (i == context.invalid_index)
        {
            const std::array<double Properties::*, 5> fields{&Properties::density,
                &Properties::specific_heat_capacity, &Properties::dynamic_viscosity,
                &Properties::thermal_conductivity, &Properties::density_temperature_derivative};
            if (context.invalid_field >= 0) outputs[i].*fields[context.invalid_field] = context.invalid_value;
        }
        if (context.status != SF_MATERIAL_SUCCESS)
        {
            if (capacity)
            {
                if (context.unterminated_error) std::memset(error, 'x', capacity);
                else
                {
                    constexpr char message[] = "fixture correlation failed";
                    const auto length = std::min(capacity - 1, sizeof(message) - 1);
                    std::memcpy(error, message, length);
                    error[length] = '\0';
                }
            }
            return context.status; // Deliberately leave the rest of the batch unwritten.
        }
    }
    return SF_MATERIAL_SUCCESS;
}

sf_material_provider_v1 descriptor(const Context& context, size_t composition_count = 0)
{
    return {SF_MATERIAL_ABI_VERSION_1, sizeof(sf_material_provider_v1), "test material", composition_count,
        SF_MATERIAL_CAP_AFFINE_DENSITY_TEMPERATURE | SF_MATERIAL_CAP_CONSTANT_CP_MU_K, &context, evaluate};
}

Input state(double temperature = 300)
{
    return {temperature, 101325, nullptr, 0};
}

void expect_sentinel(const Properties& properties)
{
    EXPECT_EQ(properties.density, -17);
    EXPECT_EQ(properties.specific_heat_capacity, -17);
    EXPECT_EQ(properties.dynamic_viscosity, -17);
    EXPECT_EQ(properties.thermal_conductivity, -17);
    EXPECT_EQ(properties.density_temperature_derivative, -17);
}
} // namespace

TEST(MaterialProviderTest, CopiesBorrowedDescriptorAndEvaluatesCompositionAndExpansion)
{
    Context context;
    auto description = descriptor(context, 2);
    const Provider provider(description);
    description.name = "changed descriptor";
    description.context = nullptr;
    description.evaluate = nullptr;
    EXPECT_EQ(provider.name(), "test material");
    EXPECT_EQ(provider.composition_count(), 2U);
    EXPECT_TRUE(provider.has_capability(SF_MATERIAL_CAP_AFFINE_DENSITY_TEMPERATURE));
    EXPECT_TRUE(provider.has_capability(SF_MATERIAL_CAP_CONSTANT_CP_MU_K));
    EXPECT_FALSE(provider.has_capability(UINT64_C(4)));
    EXPECT_EQ(provider.descriptor().context, &context);
    const std::array<double, 2> composition{2, 0.5};
    const auto properties = provider.evaluate(Input{310, 101325, composition.data(), composition.size()});
    EXPECT_EQ(properties.density, 1198);
    EXPECT_EQ(properties.specific_heat_capacity, 4000);
    EXPECT_EQ(properties.dynamic_viscosity, 0.001);
    EXPECT_EQ(properties.thermal_conductivity, 0.6);
    EXPECT_EQ(properties.density_temperature_derivative, -0.4);
    EXPECT_DOUBLE_EQ(Provider::beta(properties), 0.4 / 1198);
    EXPECT_EQ(context.calls.load(), 1U);
}

TEST(MaterialProviderTest, RejectsMalformedDescriptorsAndAcceptsAppendedVersionOneFields)
{
    Context context;
    auto candidate = descriptor(context);
    candidate.abi_version = 2;
    EXPECT_THROW((Provider(candidate)), std::invalid_argument);
    candidate = descriptor(context); --candidate.struct_size;
    EXPECT_THROW((Provider(candidate)), std::invalid_argument);
    candidate = descriptor(context); candidate.name = nullptr;
    EXPECT_THROW((Provider(candidate)), std::invalid_argument);
    candidate = descriptor(context); candidate.name = "";
    EXPECT_THROW((Provider(candidate)), std::invalid_argument);
    candidate = descriptor(context); candidate.evaluate = nullptr;
    EXPECT_THROW((Provider(candidate)), std::invalid_argument);
    candidate = descriptor(context); candidate.struct_size += 16;
    EXPECT_NO_THROW((Provider(candidate)));
    EXPECT_EQ(context.calls.load(), 0U);
}

TEST(MaterialProviderTest, ValidatesEveryInputBeforeCallingTheProvider)
{
    Context context;
    const Provider provider(descriptor(context));
    const std::array<double, 4> invalid_values{0, -1,
        std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
    for (const auto value : invalid_values)
    {
        std::array<Input, 2> inputs{state(), state(value)};
        EXPECT_THROW(provider.evaluate(inputs), std::invalid_argument);
        inputs[1] = state(); inputs[1].absolute_pressure_pascal = value;
        EXPECT_THROW(provider.evaluate(inputs), std::invalid_argument);
    }
    const Provider mixture(descriptor(context, 2));
    EXPECT_THROW(mixture.evaluate(state()), std::invalid_argument);
    auto missing = state(); missing.composition_count = 2;
    EXPECT_THROW(mixture.evaluate(missing), std::invalid_argument);
    const std::array<double, 2> nonfinite{1, std::numeric_limits<double>::quiet_NaN()};
    EXPECT_THROW(mixture.evaluate(Input{300, 101325, nonfinite.data(), nonfinite.size()}), std::invalid_argument);
    EXPECT_EQ(context.calls.load(), 0U);
}

TEST(MaterialProviderTest, DoesNotPublishPartialFailuresOrInvalidResults)
{
    Context context;
    const Provider provider(descriptor(context));
    const std::array<Input, 3> inputs{state(), state(310), state(320)};
    std::array<Properties, 3> outputs;
    outputs.fill({-17, -17, -17, -17, -17});
    context.status = SF_MATERIAL_EVALUATION_FAILED;
    EXPECT_THROW(provider.evaluate(inputs, outputs), std::runtime_error);
    for (const auto& output : outputs) expect_sentinel(output);
    context.status = SF_MATERIAL_SUCCESS;
    context.invalid_field = 0; context.invalid_index = 1;
    EXPECT_THROW(provider.evaluate(inputs, outputs), std::runtime_error);
    for (const auto& output : outputs) expect_sentinel(output);
    context.invalid_field = -1;
    provider.evaluate(inputs, outputs);
    EXPECT_EQ(outputs[0].density, 1200);
    EXPECT_EQ(outputs[1].density, 1196);
    EXPECT_EQ(outputs[2].density, 1192);
}

TEST(MaterialProviderTest, RejectsEachInvalidOrUnwrittenProperty)
{
    Context context;
    const Provider provider(descriptor(context));
    for (int field = 0; field < 5; ++field)
    {
        context.invalid_field = field;
        for (const auto value : {std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::infinity()})
        {
            context.invalid_value = value;
            EXPECT_THROW(provider.evaluate(state()), std::runtime_error);
        }
        if (field < 4)
        {
            context.invalid_value = field == 2 ? -1 : 0;
            EXPECT_THROW(provider.evaluate(state()), std::runtime_error);
        }
    }
    context.invalid_field = 2; context.invalid_value = 0;
    EXPECT_EQ(provider.evaluate(state()).dynamic_viscosity, 0);
    context.invalid_field = -1; context.omit_derivative = true;
    EXPECT_THROW(provider.evaluate(state()), std::runtime_error);
}

TEST(MaterialProviderTest, ReportsProviderNameStatusAndBoundedDiagnostics)
{
    Context context;
    const Provider provider(descriptor(context));
    for (const int status : std::array<int, 4>{SF_MATERIAL_INVALID_ARGUMENT, SF_MATERIAL_UNSUPPORTED_ABI,
                                               SF_MATERIAL_EVALUATION_FAILED, 99})
    {
        context.status = status;
        try { static_cast<void>(provider.evaluate(state())); FAIL() << "Expected a provider error"; }
        catch (const std::runtime_error& error)
        {
            const std::string message = error.what();
            EXPECT_NE(message.find("test material"), std::string::npos);
            EXPECT_NE(message.find(std::to_string(status)), std::string::npos);
            EXPECT_NE(message.find("fixture correlation failed"), std::string::npos);
        }
    }
    context.unterminated_error = true;
    try { static_cast<void>(provider.evaluate(state())); FAIL() << "Expected a provider error"; }
    catch (const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_LT(message.size(), 1024U);
        EXPECT_NE(message.find("xxxxx"), std::string::npos);
    }
}

TEST(MaterialProviderTest, ContainsCallbackExceptionsBeforePublishingOutputs)
{
    Context context;
    auto description = descriptor(context);
    description.evaluate = [](const void*, size_t, const Input*, Properties*, char*, size_t) -> int
    { throw std::runtime_error("unexpected provider exception"); };
    const Provider provider(description);
    const std::array<Input, 1> inputs{state()};
    std::array<Properties, 1> outputs{{{-17, -17, -17, -17, -17}}};
    try { provider.evaluate(inputs, outputs); FAIL() << "Expected a provider error"; }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("test material"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("no-exception contract"), std::string::npos);
    }
    expect_sentinel(outputs[0]);
}

TEST(MaterialProviderTest, HandlesEmptyBatchesWithoutCallbackAndRejectsMismatchedOutputCounts)
{
    Context context;
    const Provider provider(descriptor(context));
    EXPECT_TRUE(provider.evaluate(std::span<const Input>{}).empty());
    provider.evaluate(std::span<const Input>{}, std::span<Properties>{});
    std::array<Properties, 1> outputs{};
    EXPECT_THROW(provider.evaluate(std::span<const Input>{}, outputs), std::invalid_argument);
    EXPECT_EQ(context.calls.load(), 0U);
}

TEST(MaterialProviderTest, ExpansionHelperRetainsSignAndRejectsNonfiniteResults)
{
    Properties properties{1000, 4000, 0.001, 0.6, 1};
    EXPECT_DOUBLE_EQ(Provider::beta(properties), -0.001);
    properties.density = 0;
    EXPECT_THROW(Provider::beta(properties), std::invalid_argument);
    properties.density = 1000;
    properties.density_temperature_derivative = std::numeric_limits<double>::infinity();
    EXPECT_THROW(Provider::beta(properties), std::invalid_argument);
    properties.density = std::numeric_limits<double>::denorm_min();
    properties.density_temperature_derivative = std::numeric_limits<double>::max();
    EXPECT_THROW(Provider::beta(properties), std::invalid_argument);
}
