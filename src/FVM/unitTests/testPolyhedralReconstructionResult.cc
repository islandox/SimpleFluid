/** @file testPolyhedralReconstructionResult.cc
 * @brief Complete mode-aware production PLIC records and malformed input oracles.
 */
#include "FVM/details/GeneralVoF/Wire.hh"
#include "geometry/PLIC/PolyhedralInterface.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "geometry/PLIC/ScalarPacket.hh"
#include "plic_result_wire_fixture.hh"
#include <array>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::PLIC;
namespace codec = SimpleFluid::VoF::general_vof_detail;
using Bytes = std::vector<char>;

// Header placement follows the public natural-layout ABI. Body offsets remain
// independent schema oracles: changing field order or dropping a field must fail.
using Header = codec::ReconstructionPacketHeader;
constexpr std::size_t header_size = sizeof(Header);
constexpr std::size_t family_offset = offsetof(Header, family);
constexpr std::size_t version_offset = offsetof(Header, version);
constexpr std::size_t kind_offset = offsetof(Header, kind);
constexpr std::size_t mode_offset = offsetof(Header, mode);
constexpr std::size_t profile_offset = offsetof(Header, profile);
constexpr std::size_t length_offset = offsetof(Header, body_length);
constexpr auto header_fields = []
{
    std::array<bool, header_size> fields{};
    for (const auto offset : {family_offset, version_offset, kind_offset, mode_offset, profile_offset})
        fields[offset] = true;
    for (std::size_t i = 0; i < sizeof(Header::body_length); ++i)
        fields[length_offset + i] = true;
    return fields;
}();
constexpr std::size_t scalar_size = 8 * (2 + (std::numeric_limits<long double>::digits + 31) / 32);
constexpr std::size_t iterations_offset = header_size + 95 + scalar_size;
constexpr std::size_t resolution_offset = header_size + 103 + scalar_size;
constexpr std::size_t difference_offset = header_size + 185 + scalar_size;

void put_word(Bytes& bytes, std::size_t offset, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        bytes.at(offset + i) = char(value >> (8 * i));
}
std::uint64_t get_word(const Bytes& bytes, std::size_t offset)
{
    std::uint64_t result = 0;
    for (unsigned i = 0; i < 8; ++i)
        result |= std::uint64_t(static_cast<unsigned char>(bytes.at(offset + i))) << (8 * i);
    return result;
}
Bytes unhex(std::string_view text)
{
    const auto digit = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    Bytes result;
    for (std::size_t i = 0; i < text.size(); i += 2)
        result.push_back(char(16 * digit(text[i]) + digit(text[i + 1])));
    return result;
}
Bytes encode(const DynamicPolyhedralReconstructionResult& result)
{
    codec::wire::Writer writer;
    codec::plane(writer, result);
    return std::move(writer.data);
}
DynamicPolyhedralReconstructionResult decode(const Bytes& bytes)
{
    codec::wire::Reader reader{bytes};
    auto result = codec::plane(reader);
    reader.done();
    return result;
}
void expect_rejected(const Bytes& bytes)
{
    EXPECT_ANY_THROW((void) decode(bytes));
}
template<ArithmeticMode Mode> PolyhedralReconstructionResult<Mode> full_record()
{
    PolyhedralReconstructionResult<Mode> result;
    result.geometry.gas = {{.6, -.8, 0}, {-2, 3.5, 4.25}, -.375};
    // A reported scalar and the signed minority residual have independent
    // contracts; a transport codec must not derive either from the other.
    result.target_volume_residual = .03125;
    result.stencil_volume_residual = .125;
    result.stencil_volume_residual_measured = true;
    result.fit_converged = true;
    result.orientation_method = PolyhedralOrientationMethod::AngularVolumeFit;
    result.model_policy = PlicModelPolicy::RequireModelAgreement;
    result.model_quality = PlicModelQuality::WithinThreshold;
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        result.target_volume_difference = {-.125L};
        result.target_volume_numerical_error_estimate = 0x1p-60L;
    }
    else if constexpr (Mode == ArithmeticMode::BoundedCertification)
        result.target_volume_difference = {-.125L, 0x1p-70L};
    else
        result.target_volume_difference = {-.125L, 0x1p-70L, true};
    // Even a field that this mode does not use numerically is concrete public
    // record data. Transport must preserve it rather than infer a replacement.
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
        result.target_volume_numerical_error_estimate = 0x1p-61L;
    else if constexpr (Mode == ArithmeticMode::Exact)
        result.target_volume_numerical_error_estimate = 0x1p-62L;
    result.stencil_condition = 6.25;
    result.orientation_step = .002;
    result.fit_iterations = std::numeric_limits<unsigned>::max();
    result.resolution.reason = PolyhedralResolutionAssessment::Reason::IllConditionedAngularSupport;
    result.resolution.local_cell_span = .25;
    result.resolution.requested_cell_span = .5;
    result.resolution.gas_support_fraction = .75;
    result.resolution.carrier_support_fraction = .875;
    result.resolution.stencil_condition = 3.25;
    result.resolution.directional_strength = .625;
    result.resolution.directional_uncertainty = 0x1p-20;
    result.resolution.angular_support_condition = std::numeric_limits<double>::infinity();
    result.resolution.angular_support_cells = std::numeric_limits<unsigned>::max();
    result.resolution.angular_fit_eligible = true;
    result.resolution.stencil_cells = 513;
    return result;
}
template<ArithmeticMode Mode> void expect_complete_roundtrip()
{
    const auto packet = encode(to_dynamic(full_record<Mode>()));
    // Fixed tags and independent expected length, including the Exact enclosure.
    ASSERT_GE(packet.size(), header_size);
    EXPECT_EQ(static_cast<unsigned char>(packet[family_offset]), 15u);
    EXPECT_EQ(static_cast<unsigned char>(packet[version_offset]), 3u);
    EXPECT_EQ(static_cast<unsigned char>(packet[kind_offset]), 1u);
    EXPECT_EQ(static_cast<unsigned char>(packet[mode_offset]), static_cast<unsigned>(Mode));
    constexpr unsigned profile = std::numeric_limits<long double>::digits == 53 ? 1 :
                                 std::numeric_limits<long double>::digits == 64 ? 2 : 3;
    EXPECT_EQ(static_cast<unsigned char>(packet[profile_offset]), profile);
    for (std::size_t i = 0; i < header_size; ++i)
        if (!header_fields[i])
            EXPECT_EQ(packet[i], 0);
    constexpr auto body_size = 185 + 2 * scalar_size +
                               (Mode == ArithmeticMode::Ordinary ? 0 : scalar_size) +
                               (Mode == ArithmeticMode::Exact ? 1 : 0);
    EXPECT_EQ(get_word(packet, length_offset), body_size);
    EXPECT_EQ(packet.size(), header_size + body_size);
    auto decoded = decode(packet);
    ASSERT_TRUE(std::holds_alternative<PolyhedralReconstructionResult<Mode>>(decoded));
    const auto& result = std::get<PolyhedralReconstructionResult<Mode>>(decoded);
    EXPECT_EQ(result.geometry.gas.normal, (GeometryPoint{.6, -.8, 0}));
    EXPECT_EQ(result.geometry.gas.anchor, (GeometryPoint{-2, 3.5, 4.25}));
    EXPECT_EQ(result.geometry.gas.offset, -.375);
    EXPECT_FALSE(result.geometry.pure_gas);
    EXPECT_FALSE(result.geometry.pure_carrier);
    EXPECT_EQ(result.target_volume_residual, .03125);
    EXPECT_EQ(result.target_volume_difference.value, -.125L);
    if constexpr (Mode != ArithmeticMode::Ordinary)
        EXPECT_EQ(result.target_volume_difference.absolute_error, 0x1p-70L);
    if constexpr (Mode == ArithmeticMode::Exact)
        EXPECT_TRUE(result.target_volume_difference.exact_nonzero);
    constexpr auto expected_numerical_error = Mode == ArithmeticMode::Ordinary ? 0x1p-60L :
                                              Mode == ArithmeticMode::BoundedCertification ? 0x1p-61L : 0x1p-62L;
    EXPECT_EQ(result.target_volume_numerical_error_estimate, expected_numerical_error);
    EXPECT_EQ(result.stencil_volume_residual, .125);
    EXPECT_TRUE(result.stencil_volume_residual_measured);
    EXPECT_TRUE(result.fit_converged);
    EXPECT_EQ(result.orientation_method, PolyhedralOrientationMethod::AngularVolumeFit);
    EXPECT_EQ(result.model_policy, PlicModelPolicy::RequireModelAgreement);
    EXPECT_EQ(result.model_quality, PlicModelQuality::WithinThreshold);
    EXPECT_EQ(result.stencil_condition, 6.25);
    EXPECT_EQ(result.orientation_step, .002);
    EXPECT_EQ(result.fit_iterations, std::numeric_limits<unsigned>::max());
    EXPECT_EQ(result.resolution.reason, PolyhedralResolutionAssessment::Reason::IllConditionedAngularSupport);
    EXPECT_EQ(result.resolution.local_cell_span, .25);
    EXPECT_EQ(result.resolution.requested_cell_span, .5);
    EXPECT_EQ(result.resolution.gas_support_fraction, .75);
    EXPECT_EQ(result.resolution.carrier_support_fraction, .875);
    EXPECT_EQ(result.resolution.stencil_condition, 3.25);
    EXPECT_EQ(result.resolution.directional_strength, .625);
    EXPECT_EQ(result.resolution.directional_uncertainty, 0x1p-20);
    EXPECT_EQ(result.resolution.angular_support_condition, std::numeric_limits<double>::infinity());
    EXPECT_EQ(result.resolution.angular_support_cells, std::numeric_limits<unsigned>::max());
    EXPECT_TRUE(result.resolution.angular_fit_eligible);
    EXPECT_EQ(result.resolution.stencil_cells, 513u);
    EXPECT_EQ(encode(decoded), packet);
}
template<class T> concept HasFitMetadata = requires(T value) { value.fit_iterations; };
template<class T> concept HasModelMetadata = requires(T value) { value.model_policy; };
static_assert(!HasFitMetadata<PolyhedralInterface> && !HasModelMetadata<PolyhedralInterface>);
static_assert(std::is_trivially_copyable_v<PolyhedralInterface>);
static_assert(decltype(PolyhedralReconstructionResult<ArithmeticMode::Ordinary>{}.target_volume_difference)::
                  arithmetic_mode == ArithmeticMode::Ordinary);
static_assert(!std::is_convertible_v<PolyhedralReconstructionResult<ArithmeticMode::Ordinary>,
    PolyhedralReconstructionResult<ArithmeticMode::Exact>>);
static_assert(std::variant_size_v<DynamicPolyhedralReconstructionResult> == 3);

TEST(PolyhedralReconstructionResultTest, GeometryIsIndependentOfReconstructionMetadata)
{
    PolyhedralInterface shape{{{1, 0, 0}, {2, 3, 4}, .375}, false, false};
    PolyhedralReconstructionResult<ArithmeticMode::Ordinary> result;
    result.geometry = shape;
    result.fit_iterations = 123;
    EXPECT_EQ(result.geometry.gas.normal, shape.gas.normal);
    EXPECT_EQ(result.geometry.gas.offset, .375);
    EXPECT_FALSE(result.geometry.pure_gas);
}
TEST(PolyhedralReconstructionResultTest, OrdinaryCompleteRecord)
{
    expect_complete_roundtrip<ArithmeticMode::Ordinary>();
}
TEST(PolyhedralReconstructionResultTest, BoundedCompleteRecord)
{
    expect_complete_roundtrip<ArithmeticMode::BoundedCertification>();
}
TEST(PolyhedralReconstructionResultTest, ExactCompleteRecord)
{
    expect_complete_roundtrip<ArithmeticMode::Exact>();
}
TEST(PolyhedralReconstructionResultTest, PureAndModelProvenanceRemainIndependentOfResidualSign)
{
    auto approximation = full_record<ArithmeticMode::Ordinary>();
    approximation.model_policy = PlicModelPolicy::ConservativeApproximation;
    approximation.model_quality = PlicModelQuality::AboveThreshold;
    approximation.orientation_method = PolyhedralOrientationMethod::Gradient;
    approximation.fit_converged = false;
    approximation.fit_iterations = 0;
    const auto decoded = decode(encode(to_dynamic(approximation)));
    const auto& gradient = std::get<PolyhedralReconstructionResult<ArithmeticMode::Ordinary>>(decoded);
    EXPECT_EQ(gradient.model_policy, PlicModelPolicy::ConservativeApproximation);
    EXPECT_EQ(gradient.model_quality, PlicModelQuality::AboveThreshold);
    EXPECT_EQ(gradient.orientation_method, PolyhedralOrientationMethod::Gradient);
    EXPECT_FALSE(gradient.fit_converged);
    for (bool gas : {false, true})
    {
        PolyhedralReconstructionResult<ArithmeticMode::Exact> pure;
        pure.geometry.gas = {{1, 0, 0}, {2, 3, 4}, .375};
        pure.geometry.pure_gas = gas;
        pure.geometry.pure_carrier = !gas;
        pure.target_volume_residual = .125;
        pure.target_volume_difference = {-.125L, 0, true};
        const auto restored = decode(encode(to_dynamic(pure)));
        const auto& p = std::get<PolyhedralReconstructionResult<ArithmeticMode::Exact>>(restored);
        EXPECT_EQ(p.geometry.pure_gas, gas);
        EXPECT_EQ(p.geometry.pure_carrier, !gas);
        EXPECT_EQ(p.target_volume_residual, .125);
        EXPECT_EQ(p.target_volume_difference.value, -.125L);
        EXPECT_EQ(p.model_quality, PlicModelQuality::NotMeasured);
        EXPECT_FALSE(p.stencil_volume_residual_measured);
        EXPECT_FALSE(p.fit_converged);
        EXPECT_EQ(p.orientation_method, PolyhedralOrientationMethod::SuppliedNormal);
    }
}
template<ArithmeticMode Mode> void expect_scalar_extrema()
{
    for (const long double value : {std::numeric_limits<long double>::max(),
             std::numeric_limits<long double>::lowest(), std::numeric_limits<long double>::min(),
             std::numeric_limits<long double>::denorm_min(), std::nextafter(1.L, 2.L), -0.L, 0.L})
    {
        auto original = full_record<Mode>();
        original.target_volume_difference.value = value;
        if constexpr (Mode == ArithmeticMode::Exact)
            original.target_volume_difference.exact_nonzero = value != 0;
        const auto restored = decode(encode(to_dynamic(original)));
        const auto& result = std::get<PolyhedralReconstructionResult<Mode>>(restored);
        EXPECT_EQ(result.target_volume_difference.value, value);
        EXPECT_EQ(std::signbit(result.target_volume_difference.value), std::signbit(value));
    }
}
TEST(PolyhedralReconstructionResultTest, NativeExtremaSubnormalAndLowestPrecisionBitAreLossless)
{
    expect_scalar_extrema<ArithmeticMode::Ordinary>();
    expect_scalar_extrema<ArithmeticMode::BoundedCertification>();
    expect_scalar_extrema<ArithmeticMode::Exact>();
}
TEST(PolyhedralReconstructionResultTest, SignedZeroSurvivesDoubleAndNativeFields)
{
    auto original = full_record<ArithmeticMode::Exact>();
    original.geometry.gas.normal = {1, -0., 0.};
    original.geometry.gas.anchor = {-0., 0., -0.};
    original.geometry.gas.offset = -0.;
    original.target_volume_residual = -0.;
    original.target_volume_difference = {-0.L, -0.L, false};
    original.target_volume_numerical_error_estimate = -0.L;
    original.orientation_step = -0.;
    original.resolution.directional_uncertainty = -0.;
    const auto restored = decode(encode(to_dynamic(original)));
    const auto& result = std::get<PolyhedralReconstructionResult<ArithmeticMode::Exact>>(restored);
    EXPECT_TRUE(std::signbit(result.geometry.gas.normal[1]));
    EXPECT_FALSE(std::signbit(result.geometry.gas.normal[2]));
    EXPECT_TRUE(std::signbit(result.geometry.gas.anchor[0]));
    EXPECT_FALSE(std::signbit(result.geometry.gas.anchor[1]));
    EXPECT_TRUE(std::signbit(result.geometry.gas.offset));
    EXPECT_TRUE(std::signbit(result.target_volume_residual));
    EXPECT_TRUE(std::signbit(result.target_volume_difference.value));
    EXPECT_TRUE(std::signbit(result.target_volume_difference.absolute_error));
    EXPECT_FALSE(result.target_volume_difference.exact_nonzero);
    EXPECT_TRUE(std::signbit(result.target_volume_numerical_error_estimate));
    EXPECT_TRUE(std::signbit(result.orientation_step));
    EXPECT_TRUE(std::signbit(result.resolution.directional_uncertainty));
}
template<ArithmeticMode Mode> void expect_exact_transfer_core_validation()
{
    const auto valid = encode(to_dynamic(full_record<Mode>()));
    codec::wire::Reader reader{valid};
    const auto decoded = codec::plane_for_exact_transfer(reader);
    EXPECT_NO_THROW(reader.done());
    ASSERT_TRUE(std::holds_alternative<PolyhedralReconstructionResult<Mode>>(decoded));
    const auto& result = std::get<PolyhedralReconstructionResult<Mode>>(decoded);
    EXPECT_EQ(result.resolution.reason, PolyhedralResolutionAssessment::Reason::IllConditionedAngularSupport);
    EXPECT_EQ(result.resolution.angular_support_condition, std::numeric_limits<double>::infinity());
    // The Exact-transfer boundary historically validated these core binary64
    // fields for every result mode. Resolution diagnostics have a separate
    // contract and can legitimately retain an infinite condition indicator.
    constexpr std::array<std::size_t, 10> core_offsets{
        0, 8, 16, 24, 32, 40, 48, 58, 79 + scalar_size, 87 + scalar_size};
    for (const auto offset : core_offsets)
        for (const double value : {std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()})
        {
            SCOPED_TRACE(offset);
            SCOPED_TRACE(value);
            auto malformed = valid;
            put_word(malformed, header_size + offset, std::bit_cast<std::uint64_t>(value));
            codec::wire::Reader rejected{malformed};
            EXPECT_ANY_THROW((void) codec::plane_for_exact_transfer(rejected));
        }
}
TEST(PolyhedralReconstructionResultTest, ExactTransferBoundaryValidatesCoreValuesInEveryMode)
{
    expect_exact_transfer_core_validation<ArithmeticMode::Ordinary>();
    expect_exact_transfer_core_validation<ArithmeticMode::BoundedCertification>();
    expect_exact_transfer_core_validation<ArithmeticMode::Exact>();
}
TEST(PolyhedralReconstructionResultTest, RejectsLegacyGeneralAndExactSchemas)
{
    constexpr std::array<std::size_t, 5> old_sizes{221, 221, 165, 221, 165};
    for (std::size_t i = 0; i < old_sizes.size(); ++i)
    {
        const auto bytes = unhex(reconstruction_result_test::general_packets[i]);
        EXPECT_EQ(bytes.size(), old_sizes[i]);
        expect_rejected(bytes);
    }
    for (const auto packet : reconstruction_result_test::exact_packets)
    {
        const auto bytes = unhex(packet);
        EXPECT_EQ(bytes.size(), 194u);
        expect_rejected(bytes);
    }
}
TEST(PolyhedralReconstructionResultTest, RejectsForeignHeaderTagsProfilesPaddingAndLengths)
{
    const auto valid = encode(to_dynamic(full_record<ArithmeticMode::Exact>()));
    for (const auto offset : {family_offset, version_offset, kind_offset, mode_offset, profile_offset})
    {
        auto malformed = valid;
        malformed[offset] = char(0xff);
        SCOPED_TRACE(offset);
        expect_rejected(malformed);
    }
    for (std::size_t offset = 0; offset < header_size; ++offset)
        if (!header_fields[offset])
        {
            auto malformed = valid;
            malformed[offset] = char(1);
            SCOPED_TRACE(offset);
            expect_rejected(malformed);
        }
    auto rational_profile = valid;
    rational_profile[profile_offset] = 0; // ExactRational is not a native scalar profile.
    expect_rejected(rational_profile);
    auto foreign_profile = valid;
    foreign_profile[profile_offset] = valid[profile_offset] == 1 ? 2 : 1;
    expect_rejected(foreign_profile);
    for (const auto length : {std::uint64_t(0), get_word(valid, length_offset) - 1, get_word(valid, length_offset) + 1,
             std::numeric_limits<std::uint64_t>::max()})
    {
        auto malformed = valid;
        put_word(malformed, length_offset, length);
        expect_rejected(malformed);
    }
}
TEST(PolyhedralReconstructionResultTest, ExpectedModeRejectsEveryMismatchBeforeReadingBody)
{
    const std::array records{to_dynamic(full_record<ArithmeticMode::Ordinary>()),
        to_dynamic(full_record<ArithmeticMode::BoundedCertification>()),
        to_dynamic(full_record<ArithmeticMode::Exact>())};
    for (unsigned actual = 0; actual < records.size(); ++actual)
        for (unsigned expected = 0; expected < 3; ++expected)
        {
            const auto packet = encode(records[actual]);
            codec::wire::Reader reader{packet};
            if (actual == expected)
            {
                EXPECT_NO_THROW((void) codec::plane(reader, ArithmeticMode(expected)));
                EXPECT_NO_THROW(reader.done());
            }
            else
            {
                EXPECT_ANY_THROW((void) codec::plane(reader, ArithmeticMode(expected)));
                EXPECT_EQ(reader.offset, header_size);
            }
        }
}
TEST(PolyhedralReconstructionResultTest, RejectsEveryTruncationAndTrailingBytes)
{
    const auto valid = encode(to_dynamic(full_record<ArithmeticMode::BoundedCertification>()));
    for (std::size_t size = 0; size < valid.size(); ++size)
    {
        SCOPED_TRACE(size);
        expect_rejected(Bytes(valid.begin(), valid.begin() + size));
    }
    auto trailing = valid;
    trailing.push_back('x');
    codec::wire::Reader reader{trailing};
    EXPECT_NO_THROW((void) codec::plane(reader));
    EXPECT_ANY_THROW(reader.done());
}
TEST(PolyhedralReconstructionResultTest, RejectsNoncanonicalScalarWords)
{
    const auto valid = encode(to_dynamic(full_record<ArithmeticMode::Ordinary>()));
    for (const auto [word, value] : std::array<std::pair<unsigned, std::uint64_t>, 6>{{
             {0, 2}, // Invalid sign.
             {1, std::uint64_t(std::numeric_limits<std::int64_t>::min())},
             {1, std::uint64_t(std::numeric_limits<long double>::max_exponent + 1)},
             {2, 0x100000000ULL}, // A limb must fit uint32_t.
             {2, 0}, // Unnormalized significand.
             {2, 0x40000000ULL}}}) // Same type of unnormalized alternate representation.
    {
        auto malformed = valid;
        put_word(malformed, difference_offset + 8 * word, value);
        expect_rejected(malformed);
    }
    auto alternate = valid;
    put_word(alternate, difference_offset + 8, std::uint64_t(std::int64_t(-1)));
    put_word(alternate, difference_offset + 16, 0x40000000ULL); // Same -.125 via a noncanonical exponent.
    expect_rejected(alternate);
    auto noncanonical_zero = valid;
    for (std::size_t i = 16; i < scalar_size; i += 8)
        put_word(noncanonical_zero, difference_offset + i, 0);
    // Zero must have exponent zero, even when the sign preserves negative zero.
    expect_rejected(noncanonical_zero);
    constexpr auto unused_bits = (32 - std::numeric_limits<long double>::digits % 32) % 32;
    if constexpr (unused_bits != 0)
    {
        auto rounded_bits = valid;
        put_word(rounded_bits, difference_offset + scalar_size - 8, 1);
        expect_rejected(rounded_bits);
    }
}
TEST(PolyhedralReconstructionResultTest, RejectsNegativeEnclosureRadiiAndOverflowingIterationCounts)
{
    for (const auto record : {to_dynamic(full_record<ArithmeticMode::BoundedCertification>()),
             to_dynamic(full_record<ArithmeticMode::Exact>())})
    {
        auto negative_radius = encode(record);
        put_word(negative_radius, difference_offset + scalar_size, 1);
        expect_rejected(negative_radius);
        auto invalid = record;
        std::visit([](auto& typed)
        {
            if constexpr (decltype(typed.target_volume_difference)::arithmetic_mode != ArithmeticMode::Ordinary)
                typed.target_volume_difference.absolute_error = -0x1p-70L;
        }, invalid);
        EXPECT_ANY_THROW((void) encode(invalid));
    }
    auto overflow = encode(to_dynamic(full_record<ArithmeticMode::Ordinary>()));
    put_word(overflow, iterations_offset, std::uint64_t(std::numeric_limits<unsigned>::max()) + 1);
    expect_rejected(overflow);
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t))
    {
        for (const auto offset : {resolution_offset + 65, resolution_offset + 74})
        {
            auto count_overflow = encode(to_dynamic(full_record<ArithmeticMode::Ordinary>()));
            put_word(count_overflow, offset, std::uint64_t(std::numeric_limits<std::size_t>::max()) + 1);
            expect_rejected(count_overflow);
        }
    }
}
TEST(PolyhedralReconstructionResultTest, RejectsInvalidBooleansEnumsAndModelProvenance)
{
    const auto valid = encode(to_dynamic(full_record<ArithmeticMode::Ordinary>()));
    for (const auto offset : {std::size_t(56), std::size_t(57), std::size_t(74), std::size_t(75),
             std::size_t(76), std::size_t(77), std::size_t(78), resolution_offset - header_size,
             resolution_offset - header_size + 73})
    {
        auto malformed = valid;
        malformed[header_size + offset] = char(0xff);
        expect_rejected(malformed);
    }
    auto unmeasured_quality = valid;
    unmeasured_quality[header_size + 74] = 0; // A measured quality cannot survive an unmeasured marker.
    expect_rejected(unmeasured_quality);
    auto foreign_policy = encode(to_dynamic(full_record<ArithmeticMode::Exact>()));
    foreign_policy[header_size + 77] = 1; // ConservativeApproximation is Ordinary-only.
    expect_rejected(foreign_policy);
    auto invalid = full_record<ArithmeticMode::BoundedCertification>();
    invalid.model_policy = PlicModelPolicy::ConservativeApproximation;
    EXPECT_ANY_THROW((void) encode(to_dynamic(invalid)));
    auto both_pure = full_record<ArithmeticMode::Exact>();
    both_pure.geometry.pure_gas = both_pure.geometry.pure_carrier = true;
    EXPECT_ANY_THROW((void) encode(to_dynamic(both_pure)));
}
} // namespace
