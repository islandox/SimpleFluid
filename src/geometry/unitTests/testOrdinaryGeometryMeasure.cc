/** Ordinary retained-measure wire precision, grammar and admission contracts. */
#include "geometry/PLIC/GeometryCodecTags.hh"
#include "geometry/PLIC/GeometryMeasure.hh"
#include <array>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <locale>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace P = SimpleFluid::PLIC;
using Measure = P::GeometryMeasure<SimpleFluid::ArithmeticMode::Ordinary>;

std::vector<char> packet(std::string_view text)
{
    std::vector<char> result{static_cast<char>(P::GeometryTag::OrdinaryMeasure), 1};
    result.insert(result.end(), text.begin(), text.end());
    return result;
}

class CommaDecimal : public std::numpunct<char>
{
protected:
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
};

class GlobalLocaleGuard
{
public:
    GlobalLocaleGuard() : saved_(std::locale())
    {
        // This unnamed locale changes C++ stream facets without selecting a C locale.
        std::locale::global(std::locale(std::locale::classic(), new CommaDecimal));
    }
    ~GlobalLocaleGuard() { std::locale::global(saved_); }
    GlobalLocaleGuard(const GlobalLocaleGuard&) = delete;
    GlobalLocaleGuard& operator=(const GlobalLocaleGuard&) = delete;

private:
    std::locale saved_;
};

class FloatingEnvironmentGuard
{
public:
    FloatingEnvironmentGuard()
    {
        if (std::fegetenv(&saved_) != 0)
            throw std::runtime_error("Cannot capture test floating-point environment.");
    }
    ~FloatingEnvironmentGuard() { std::fesetenv(&saved_); }
    FloatingEnvironmentGuard(const FloatingEnvironmentGuard&) = delete;
    FloatingEnvironmentGuard& operator=(const FloatingEnvironmentGuard&) = delete;

private:
    std::fenv_t saved_{};
};
} // namespace

TEST(OrdinaryGeometryMeasure, VersionOneBytesPreserveFiniteLongDoubleValuesExactly)
{
    const auto one = Measure::from_nominal(1).encode();
    std::vector<char> expected{0x60, 1};
    const std::string body = "1." + std::string(std::numeric_limits<long double>::max_digits10, '0') + "e+00";
    expected.insert(expected.end(), body.begin(), body.end());
    EXPECT_EQ(one, expected);

    const std::array values{0.0L, -0.0L, 1.0L + std::numeric_limits<long double>::epsilon(), -42.5L,
        std::numeric_limits<long double>::min(), std::numeric_limits<long double>::denorm_min(),
        std::numeric_limits<long double>::max()};
    for (const auto value : values)
    {
        SCOPED_TRACE(value);
        const auto encoded = Measure::from_nominal(value).encode();
        const auto decoded = Measure::decode(encoded).nominal();
        EXPECT_EQ(decoded, value);
        if (value == 0)
            EXPECT_EQ(std::signbit(decoded), std::signbit(value));
    }
}

TEST(OrdinaryGeometryMeasure, ScientificGrammarAllowsDecimalVariantsAndLongBodies)
{
    struct Case
    {
        std::string_view text;
        long double value;
    };
    const std::array cases{Case{"1e0", 1}, Case{"1.e+0", 1}, Case{".5E-1", .05L}, Case{"-1.25e+1", -12.5L}};
    for (const auto& entry : cases)
    {
        SCOPED_TRACE(entry.text);
        EXPECT_EQ(Measure::decode(packet(entry.text)).nominal(), entry.value);
    }
    const std::string long_body = std::string(160, '0') + "1e0";
    ASSERT_GT(long_body.size(), 128u);
    EXPECT_EQ(Measure::decode(packet(long_body)).nominal(), 1);
}

TEST(OrdinaryGeometryMeasure, MalformedAndUnrepresentableBodiesReject)
{
    const std::array<std::string_view, 30> invalid{"", "1", "1.", ".", "e0", "-e0", ".e0", "1e", "1e+", "1e-", "+1e0",
        " 1e0", "1e0 ", "\t1e0", "1e0\n", "0x1p0", "0x1e0", "1,5e0", "nan", "NaN", "nan(payload)", "inf", "-inf",
        "infinity", "1e0junk", "1e0e0", "--1e0", "1..0e0", "1e999999", "1e-999999"};
    for (const auto text : invalid)
    {
        SCOPED_TRACE(text);
        EXPECT_THROW(static_cast<void>(Measure::decode(packet(text))), std::invalid_argument);
    }
    const std::string embedded_nul("1e0\0", 4);
    EXPECT_THROW(static_cast<void>(Measure::decode(packet(embedded_nul))), std::invalid_argument);
}

TEST(OrdinaryGeometryMeasure, PacketHeaderAndExactByteBudgetsAreEnforced)
{
    const auto measure = Measure::from_nominal(-1.25L);
    const auto encoded = measure.encode();
    ASSERT_GT(encoded.size(), 2u);
    EXPECT_EQ(measure.encode(encoded.size()), encoded);
    EXPECT_EQ(Measure::decode(encoded, encoded.size()).nominal(), -1.25L);
    EXPECT_THROW(static_cast<void>(measure.encode(encoded.size() - 1)), std::length_error);
    EXPECT_THROW(static_cast<void>(Measure::decode(encoded, encoded.size() - 1)), std::invalid_argument);
    for (std::size_t length = 0; length <= 2; ++length)
        EXPECT_THROW(
            static_cast<void>(Measure::decode(std::span<const char>(encoded.data(), length))), std::invalid_argument);
    auto wrong_tag = encoded;
    wrong_tag[0] = static_cast<char>(P::GeometryTag::ExactMeasure);
    EXPECT_THROW(static_cast<void>(Measure::decode(wrong_tag)), std::invalid_argument);
    auto wrong_version = encoded;
    wrong_version[1] = 2;
    EXPECT_THROW(static_cast<void>(Measure::decode(wrong_version)), std::invalid_argument);
}

TEST(OrdinaryGeometryMeasure, GlobalCommaDecimalLocaleLeavesWireBytesAndValuesUnchanged)
{
    const std::array values{1.25L, -42.5L, 1.0L + std::numeric_limits<long double>::epsilon(),
        std::numeric_limits<long double>::min(), std::numeric_limits<long double>::max()};
    std::vector<std::vector<char>> baseline;
    for (const auto value : values)
        baseline.push_back(Measure::from_nominal(value).encode());
    const GlobalLocaleGuard restore;
    ASSERT_EQ(std::use_facet<std::numpunct<char>>(std::locale()).decimal_point(), ',');
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        SCOPED_TRACE(index);
        EXPECT_EQ(Measure::from_nominal(values[index]).encode(), baseline[index]);
        EXPECT_EQ(Measure::decode(baseline[index]).nominal(), values[index]);
    }
    EXPECT_EQ(Measure::decode(packet(".5E-1")).nominal(), .05L);
    EXPECT_THROW(static_cast<void>(Measure::decode(packet("1,25e0"))), std::invalid_argument);
}

TEST(OrdinaryGeometryMeasure, DirectedRoundingPreservesNearestCodecAndCallerEnvironment)
{
    const FloatingEnvironmentGuard restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    const std::array values{0.0L, -0.0L, .1L, -.1L, 1.0L + std::numeric_limits<long double>::epsilon(),
        std::numeric_limits<long double>::denorm_min(), std::numeric_limits<long double>::max()};
    std::vector<std::vector<char>> baseline;
    for (const auto value : values)
        baseline.push_back(Measure::from_nominal(value).encode());
    const std::array tiny{packet("1e-999999"), packet("-1e-999999")};
    std::size_t supported = 0;
    for (const int mode : {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
    {
        if (std::fesetround(mode) != 0)
            continue;
        ++supported;
        SCOPED_TRACE(mode);
        const auto check_environment = [mode]
        {
            const int rounding = std::fegetround(), flags = std::fetestexcept(FE_ALL_EXCEPT);
            EXPECT_EQ(rounding, mode);
#if defined(_LIBCPP_VERSION)
            EXPECT_EQ(flags, FE_DIVBYZERO);
#else
            // The existing charconv path may raise conversion exceptions.
            EXPECT_EQ(flags & FE_DIVBYZERO, FE_DIVBYZERO);
#endif
        };
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            SCOPED_TRACE(index);
            ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
            ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
            const auto encoded = Measure::from_nominal(values[index]).encode();
            check_environment();
            EXPECT_EQ(encoded, baseline[index]);
            const auto decoded = Measure::decode(baseline[index]).nominal();
            check_environment();
            EXPECT_EQ(decoded, values[index]);
            if (values[index] == 0)
                EXPECT_EQ(std::signbit(decoded), std::signbit(values[index]));
            EXPECT_THROW(
                static_cast<void>(Measure::from_nominal(values[index]).encode(encoded.size() - 1)), std::length_error);
            check_environment();
        }
        for (const auto& encoded : tiny)
        {
            ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
            ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
            EXPECT_THROW(static_cast<void>(Measure::decode(encoded)), std::invalid_argument);
            check_environment();
        }
    }
    if (!supported)
        GTEST_SKIP() << "No directed floating-point rounding mode is supported.";
}
