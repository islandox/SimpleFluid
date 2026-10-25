/** @file testPLICGeometry.cc
 * @brief Typed PLIC geometry, arithmetic separation, and checked publication contracts.
 */
#include "geometry/PLIC/ConvexGeometry.hh"
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace
{
using SimpleFluid::ArithmeticMode;
namespace PLIC = SimpleFluid::PLIC;
using OrdinaryValue = PLIC::GeometryValue<ArithmeticMode::Ordinary>;
using BoundedValue = PLIC::GeometryValue<ArithmeticMode::BoundedCertification>;
using OrdinaryGeometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
using BoundedGeometry = PLIC::ConvexCellGeometry<ArithmeticMode::BoundedCertification>;

template<class T>
concept HasErrorField = requires(T value) { value.absolute_error; };
template<class T>
concept HasEnclosure = requires(T value) {
    value.lower();
    value.upper();
    value.relative_error_at_most(1.L, 1e-12);
};

static_assert(sizeof(OrdinaryValue) == sizeof(long double));
static_assert(!HasErrorField<OrdinaryValue>);
static_assert(!HasEnclosure<OrdinaryValue>);
static_assert(HasErrorField<BoundedValue> && HasEnclosure<BoundedValue>);
static_assert(HasErrorField<SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact>>);
static_assert(!std::is_convertible_v<OrdinaryValue, BoundedValue>);
static_assert(!std::is_convertible_v<BoundedValue, OrdinaryValue>);
static_assert(
    !std::is_convertible_v<SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact>, OrdinaryValue>);
static_assert(std::same_as<decltype(PLIC::GeometryMoments<ArithmeticMode::Ordinary>{}.volume), OrdinaryValue>);
static_assert(
    std::same_as<decltype(PLIC::GeometryMoments<ArithmeticMode::BoundedCertification>{}.volume), BoundedValue>);

TEST(PLICGeometryTest, OrdinaryPublicationRejectsNonfiniteAndLostNonzeroValues)
{
    EXPECT_DOUBLE_EQ(OrdinaryValue{.5L}.as_double(), .5);
    EXPECT_EQ(OrdinaryValue{.5L}.binary64_absolute_error(), 0);
    EXPECT_THROW((void) OrdinaryValue{std::numeric_limits<long double>::infinity()}.as_double(), std::invalid_argument);
    EXPECT_THROW(
        (void) OrdinaryValue{std::numeric_limits<long double>::quiet_NaN()}.as_double(), std::invalid_argument);
    const long double below_binary64 = static_cast<long double>(std::numeric_limits<double>::denorm_min()) / 2;
    if (below_binary64 > 0)
        EXPECT_THROW((void) OrdinaryValue{below_binary64}.as_double(), std::invalid_argument);
}

TEST(PLICGeometryTest, BoundedPublicationRejectsAnUncertainZero)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    EXPECT_THROW(
        (void) (BoundedValue{0, std::numeric_limits<double>::denorm_min()}.as_double()), std::invalid_argument);
}

TEST(PLICGeometryTest, ExactPublicationRetainsConversionUncertaintyAndNonzeroEvidence)
{
    const SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact> enclosed{1, 1e-12L, true};
    EXPECT_DOUBLE_EQ(enclosed.as_double(), 1);
    EXPECT_GE(enclosed.binary64_absolute_error(), enclosed.absolute_error);
    const SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact> underflowed{
        0, std::numeric_limits<long double>::denorm_min(), true};
    EXPECT_THROW((void) underflowed.as_double(), std::runtime_error);
    EXPECT_THROW((void) underflowed.binary64_absolute_error(), std::runtime_error);
    const SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact> overflowed{
        std::numeric_limits<long double>::infinity(), 0, true};
    EXPECT_THROW((void) overflowed.as_double(), std::runtime_error);
}

TEST(PLICGeometryTest, PublicationAccountsForAdditionalBinary64Rounding)
{
    if constexpr (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
        GTEST_SKIP() << "A wider long double is required to exercise additional binary64 rounding.";
    const long double halfway = 1.L + std::ldexp(1.L, -std::numeric_limits<double>::digits);
    const OrdinaryValue ordinary{halfway};
    const auto rounded = ordinary.as_double();
    const auto difference = std::abs(halfway - static_cast<long double>(rounded));
    ASSERT_GT(difference, 0);
    EXPECT_GE(ordinary.binary64_absolute_error(), difference);
    const SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact> exact{halfway, difference, true};
    EXPECT_GE(exact.binary64_absolute_error(), 2 * difference);
}

template<class Left, class Right>
concept CanIntersect = requires(const Left& a, const Right& b) { a.intersect(b); };
static_assert(CanIntersect<OrdinaryGeometry, OrdinaryGeometry>);
static_assert(!CanIntersect<OrdinaryGeometry, BoundedGeometry>);
static_assert(!CanIntersect<BoundedGeometry, OrdinaryGeometry>);
static_assert(
    !std::is_convertible_v<PLIC::Point<ArithmeticMode::Ordinary>, PLIC::Point<ArithmeticMode::BoundedCertification>>);
static_assert(!HasErrorField<PLIC::Point<ArithmeticMode::Ordinary>>);
static_assert(HasErrorField<PLIC::Point<ArithmeticMode::BoundedCertification>>);
static_assert(std::same_as<decltype(OrdinaryGeometry::box({}, {})), PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>>);
static_assert(std::same_as<decltype(std::declval<const OrdinaryGeometry&>().clip({})),
    PLIC::ClipResult<ArithmeticMode::Ordinary>>);
static_assert(std::same_as<decltype(std::declval<const OrdinaryGeometry&>().volume()), OrdinaryValue>);
static_assert(std::same_as<decltype(std::declval<const BoundedGeometry&>().volume()), BoundedValue>);
static_assert(std::same_as<
    decltype(std::declval<const SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>&>().clip({})),
    PLIC::ClipResult<ArithmeticMode::Exact>>);
static_assert(std::same_as<
    decltype(std::declval<const SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>&>().volume()),
    SimpleFluid::PLIC::GeometryValue<SimpleFluid::ArithmeticMode::Exact>>);

TEST(PLICGeometryTest, OrdinaryClipPreservesTypedPhasesAndAnalyticMoments)
{
    PLIC::reset_geometry_statistics();
    const auto cube = OrdinaryGeometry::box({0, 0, 0}, {1, 1, 1});
    ASSERT_TRUE(cube);
    EXPECT_EQ(cube->volume().value, 1);
    const PLIC::GeometryHalfspace cut{{1, 0, 0}, {0, 0, 0}, .25};
    const auto clipped = cube->clip(cut);
    ASSERT_TRUE(clipped.inside);
    ASSERT_TRUE(clipped.outside);
    EXPECT_NEAR(clipped.inside->volume().value, .25L, 1e-15L);
    EXPECT_NEAR(clipped.outside->volume().value, .75L, 1e-15L);
    EXPECT_NEAR(clipped.inside->volume().value + clipped.outside->volume().value, cube->volume().value, 1e-15L);
    const auto moments = clipped.inside->moments({0, 0, 0});
    EXPECT_NEAR(moments.first_moment[0].value, .03125L, 1e-15L);
    EXPECT_NEAR(moments.first_moment[1].value, .125L, 1e-15L);
    EXPECT_NEAR(moments.first_moment[2].value, .125L, 1e-15L);
    const auto phases = cube->phases(cut);
    EXPECT_NEAR(phases.gas.value, .25L, 1e-15L);
    EXPECT_NEAR(phases.carrier.value, .75L, 1e-15L);
    EXPECT_EQ(PLIC::geometry_statistics().interval_operations, 0u);
}

TEST(PLICGeometryTest, TypedEntryRejectsForeignRuntimeModeAndReportsOrdinaryFailures)
{
    const auto cube = OrdinaryGeometry::box({0, 0, 0}, {1, 1, 1});
    EXPECT_THROW((void) BoundedGeometry::from_dynamic(cube->dynamic_geometry()), std::invalid_argument);
    try
    {
        (void) OrdinaryGeometry::box({0, 0, 0}, {0, 1, 1});
        FAIL() << "A zero-width box must reject.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_TRUE(std::string(error.what()).starts_with("Ordinary FP geometry: ")) << error.what();
    }
}

TEST(PLICGeometryTest, BoundedClipEnclosesIndependentAnalyticVolumes)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    const auto cube = BoundedGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto cut = cube->clip({{1, 0, 0}, {0, 0, 0}, .25});
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(cut.outside);
    const auto inside = cut.inside->volume();
    const auto outside = cut.outside->volume();
    EXPECT_LE(inside.lower(), .25L);
    EXPECT_GE(inside.upper(), .25L);
    EXPECT_LE(outside.lower(), .75L);
    EXPECT_GE(outside.upper(), .75L);
    EXPECT_LE(inside.lower() + outside.lower(), 1.L);
    EXPECT_GE(inside.upper() + outside.upper(), 1.L);
    EXPECT_TRUE(inside.relative_error_at_most(.25L, 1e-12));
}

TEST(PLICGeometryTest, BoundedPublicationIncludesAdditionalBinary64Rounding)
{
    if (!PLIC::certification_environment())
        GTEST_SKIP() << "Bounded wider floating-point environment unavailable.";
    if constexpr (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
        GTEST_SKIP() << "A wider long double is required to exercise additional binary64 rounding.";
    const long double halfway = 1.L + std::ldexp(1.L, -std::numeric_limits<double>::digits);
    const auto difference = std::abs(halfway - static_cast<long double>(static_cast<double>(halfway)));
    ASSERT_GT(difference, 0);
    const BoundedValue value{halfway, difference};
    EXPECT_GE(value.binary64_absolute_error(), 2 * difference);
}

TEST(PLICGeometryTest, TypedPacketsRetainModeAndRejectStaleTransactionContext)
{
    const auto cube = OrdinaryGeometry::box({0, 0, 0}, {1, 1, 1});
    const PLIC::GeometryPacketContext context{{17, 23, 31, 47}};
    const std::array sources{cube};
    const auto cut = cube->clip({{1, 0, 0}, {0, 0, 0}, .25});
    ASSERT_TRUE(cut.inside);
    const auto packet = cut.inside->encode(context);
    const auto restored = OrdinaryGeometry::decode(packet, context, sources);
    ASSERT_TRUE(restored);
    EXPECT_DOUBLE_EQ(restored->volume().as_double(), .25);
    auto stale = context;
    ++stale.publication.back();
    EXPECT_THROW((void) OrdinaryGeometry::decode(packet, stale, sources), std::invalid_argument);
    EXPECT_DOUBLE_EQ(cube->volume().as_double(), 1);
}

TEST(PLICGeometryTest, TypedInputConversionHonorsConstructionCaps)
{
    using Point = PLIC::Point<ArithmeticMode::Ordinary>;
    const std::array vertices{Point({0, 0, 0}), Point({1, 0, 0}), Point({0, 1, 0}), Point({0, 0, 1})};
    const std::vector<std::vector<std::size_t>> faces{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    PLIC::GeometryLimits limits;
    limits.maximum_vertices = 3;
    PLIC::reset_geometry_statistics();
    EXPECT_THROW((void) OrdinaryGeometry::make_points(vertices, faces, limits), std::invalid_argument);
    EXPECT_EQ(PLIC::geometry_statistics().constructions, 0u);
    limits.maximum_vertices = 4;
    limits.byte_budget = 1;
    EXPECT_THROW((void) OrdinaryGeometry::make_points(vertices, faces, limits), std::invalid_argument);
    EXPECT_EQ(PLIC::geometry_statistics().constructions, 0u);
    EXPECT_NEAR(OrdinaryGeometry::make_points(vertices, faces)->volume().value, 1.L / 6, 1e-15L);
}
} // namespace
