/** @file testNativeGeometryIdentity.cc
 * @brief Independent arithmetic and resource contracts for native identities.
 */
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <new>
#include <optional>

namespace
{
thread_local bool count_identity_allocations = false;
thread_local std::size_t identity_allocations = 0;
struct IdentityAllocationScope
{
    IdentityAllocationScope() { count_identity_allocations = true; }
    ~IdentityAllocationScope() { count_identity_allocations = false; }
};
} // namespace
void* operator new(std::size_t size)
{
    if (count_identity_allocations)
        ++identity_allocations;
    if (auto* value = std::malloc(size ? size : 1))
        return value;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size)
{
    return ::operator new(size);
}
void operator delete(void* value) noexcept
{
    std::free(value);
}
void operator delete(void* value, std::size_t) noexcept
{
    std::free(value);
}
void operator delete[](void* value) noexcept
{
    std::free(value);
}
void operator delete[](void* value, std::size_t) noexcept
{
    std::free(value);
}

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::PLIC;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr auto exact = ArithmeticMode::Exact;
template<ArithmeticMode Mode> PointIdentity<Mode> at(GeometryPoint value)
{
    const std::array<GeometryPoint, 3> vertices{
        value, GeometryPoint{value[0], value[1] + 1, value[2]}, GeometryPoint{value[0], value[1], value[2] + 1}};
    return PlanarFace<Mode>::make(vertices)->vertex_identities().front();
}
PointIdentity<exact> clipped_top_point(double height)
{
    const std::array<GeometryPoint, 4> square{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
    const std::array<GeometryPoint, 3> triangle{{{0, 0, 0}, {1, 0, 0}, {0, height, 0}}};
    const auto clipped = PlanarFace<exact>::make(square)->intersection(*PlanarFace<exact>::make(triangle));
    if (clipped)
        for (const auto& point : clipped->vertex_identities())
        {
            const auto display = point_display<exact>(point);
            if (display[1] == 1 && display[0] > 0)
                return point;
        }
    throw std::runtime_error("Independent rational intersection fixture has no expected top point.");
}
TEST(NativeGeometryIdentityTest, FiniteArithmeticAndKeyQueriesDoNotAllocate)
{
    const auto point = at<ordinary>({.25, -.5, .125});
    const auto shift = displacement<ordinary>({}, {.5, .25, -.125});
    const auto expected = at<ordinary>({.75, -.25, 0});
    const auto expected_hash = Meshes::AMR::adaptive_route_hash(canonical_key(expected));
    const std::array<GeometryPoint, 3> triangle{{{0, 0, 0}, {41, 17, 0}, {0, 0, 1}}};
    const auto line = PlanarFace<ordinary>::make(triangle)->edge_support_identities().front();
    const auto line_hash = Meshes::AMR::adaptive_route_hash(canonical_key(line));
    bool correct = true;
    identity_allocations = 0;
    {
        IdentityAllocationScope scope;
        for (unsigned i = 0; i < 20000; ++i)
        {
            const auto translated = point_translated<ordinary>(point, shift);
            const auto cancelled = displacement_difference<ordinary>(shift, shift);
            correct = correct && translated == expected && (translated <=> expected) == 0 && cancelled.is_zero();
            correct = correct && canonical_key_hash(translated) == expected_hash && canonical_key_size(translated) > 0;
            correct = correct && canonical_key_hash(line) == line_hash;
        }
    }
    EXPECT_TRUE(correct);
    EXPECT_EQ(identity_allocations, 0u);
    EXPECT_EQ(point.payload_bytes(), 0u);
    EXPECT_EQ(shift.payload_bytes(), 0u);
    EXPECT_EQ(line.payload_bytes(), 0u);
}
TEST(NativeGeometryIdentityTest, FiniteSubnormalAndWideValuesSurviveUntilPublication)
{
    using Wide = PlanarFace<ordinary>::WidePoint;
    const auto tiny = std::numeric_limits<long double>::denorm_min();
    const std::array<Wide, 3> vertices{{{tiny, 0, 0}, {tiny, 1, 0}, {tiny, 0, 1}}};
    const auto point = PlanarFace<ordinary>::make_wide(vertices)->vertex_identities().front();
    const auto key = canonical_key(point);
    EXPECT_EQ(numerical_point<ordinary>(point)[0].value, tiny);
    EXPECT_EQ(point_translated<ordinary>(point, {}), point);
    EXPECT_EQ(canonical_key(decode_identity<PointIdentity<ordinary>>(encode_identity(point))), key);
    if constexpr (std::numeric_limits<long double>::min_exponent < std::numeric_limits<double>::min_exponent)
        EXPECT_THROW(point_display<ordinary>(point), std::invalid_argument);
    const auto high = std::numeric_limits<double>::max();
    const auto wide = at<ordinary>({high, 0, 0});
    if constexpr (std::numeric_limits<long double>::max_exponent > std::numeric_limits<double>::max_exponent)
    {
        const auto enlarged = point_translated<ordinary>(wide, displacement<ordinary>({}, {high, 0, 0}));
        EXPECT_EQ(numerical_point<ordinary>(enlarged)[0].value, 2.L * static_cast<long double>(high));
        EXPECT_THROW(point_display<ordinary>(enlarged), std::invalid_argument);
    }
    else
        EXPECT_THROW(point_translated<ordinary>(wide, displacement<ordinary>({}, {high, 0, 0})), std::invalid_argument);
    EXPECT_EQ(numerical_point<ordinary>(wide)[0].value, static_cast<long double>(high));
}
TEST(NativeGeometryIdentityTest, ZeroSignsAndBackendPeriodicRulesRemainDistinct)
{
    const auto zero = at<ordinary>({0, 0, 0});
    const auto signed_zero = at<ordinary>({-0., -0., -0.});
    EXPECT_EQ(signed_zero, zero);
    EXPECT_FALSE(std::signbit(point_display<ordinary>(signed_zero)[0]));
    EXPECT_TRUE(displacement<ordinary>({-0., 0, 0}, {0, -0., 0}).is_zero());
    const std::array<bool, 3> periodic{true, false, false};
    EXPECT_EQ(point_display<ordinary>(periodic_point<ordinary>(at<ordinary>({2.5, 0, 0}), {}, {1, 1, 1}, periodic)),
        (GeometryPoint{.5, 0, 0}));
    // Exact authority normalization only identifies the represented upper seam;
    // it does not apply Ordinary's arbitrary-period wrapping rule.
    EXPECT_EQ(point_display<exact>(periodic_point<exact>(at<exact>({2.5, 0, 0}), {}, {1, 1, 1}, periodic)),
        (GeometryPoint{2.5, 0, 0}));
    EXPECT_EQ(point_display<exact>(periodic_point<exact>(at<exact>({1, 0, 0}), {}, {1, 1, 1}, periodic)),
        (GeometryPoint{0, 0, 0}));
    EXPECT_THROW(periodic_point<ordinary>(zero, {}, {}, periodic), std::invalid_argument);
    EXPECT_THROW(periodic_point<exact>(at<exact>({0, 0, 0}), {}, {}, periodic), std::invalid_argument);
}
TEST(NativeGeometryIdentityTest, NondyadicExactArithmeticRetainsIndependentRationalResultsAndLifetime)
{
    using Measure = GeometryMeasure<exact>;
    // x + y/1.5 = 1 at y=1 gives x=1/3, independently of identity arithmetic.
    const auto third = clipped_top_point(1.5);
    const auto before = canonical_key(third);
    const auto one_third = Measure::from_value(1).divided_by(Measure::from_value(3));
    EXPECT_EQ(point_projection<exact>(third, {1, 0, 0}, {}).compare(one_third), 0);
    const auto moved = point_translated<exact>(third, displacement<exact>({}, {.5, -.5, .25}));
    const auto five_sixths = Measure::from_value(5).divided_by(Measure::from_value(6));
    EXPECT_EQ(point_projection<exact>(moved, {1, 0, 0}, {}).compare(five_sixths), 0);
    EXPECT_EQ(point_projection<exact>(moved, {3, 2, 4}, {}).compare(Measure::from_value(4.5)), 0);
    EXPECT_EQ(canonical_key(third), before);
    const auto decoded = decode_identity<PointIdentity<exact>>(encode_identity(moved));
    EXPECT_EQ(decoded, moved);
    EXPECT_EQ(point_projection<exact>(decoded, {1, 0, 0}, {}).compare(five_sixths), 0);
    EXPECT_EQ(point_projection<exact>(decoded, {3, 2, 4}, {}).compare(Measure::from_value(4.5)), 0);
}
TEST(NativeGeometryIdentityTest, EdgeIndicesKeepTheirExistingBackendOrderingAndClosedEndpoints)
{
    const auto third = clipped_top_point(1.5);
    const auto two_thirds = clipped_top_point(3);
    const std::array<PointIdentity<exact>, 4> exact_points{
        two_thirds, at<exact>({0, 1, 0}), at<exact>({1, 1, 0}), third};
    EdgePointIndex<exact> rational(exact_points);
    EXPECT_EQ(rational.closed_segment(two_thirds, third), (std::vector<PointIdentity<exact>>{third, two_thirds}));
    const auto first = at<ordinary>({.25, 1, 0}), second = at<ordinary>({.5, 1, 0});
    const std::array<PointIdentity<ordinary>, 4> finite_points{
        second, at<ordinary>({0, 1, 0}), at<ordinary>({1, 1, 0}), first};
    EdgePointIndex<ordinary> finite(finite_points);
    EXPECT_EQ(finite.closed_segment(second, first), (std::vector<PointIdentity<ordinary>>{second, first}));
    EXPECT_THROW(rational.closed_segment(third, third), std::invalid_argument);
    EXPECT_THROW(finite.closed_segment(first, first), std::invalid_argument);
}
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
using LegacyIdentityLess = LegacyCanonicalKeyLess;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
TEST(NativeGeometryIdentityTest, NativeOrderingUsesValuesAndLegacyAdapterIsExplicit)
{
    const auto a = std::ldexp(1., 62), b = std::ldexp(1., 63);
    EXPECT_LT(at<ordinary>({a, 0, 0}), at<ordinary>({b, 0, 0}));
    const auto negative = displacement<ordinary>({}, {-1, 0, 0});
    const DisplacementIdentity<ordinary> zero;
    const auto positive = displacement<ordinary>({}, {1, 0, 0});
    EXPECT_LT(negative, zero);
    EXPECT_LT(zero, positive);
    const LegacyIdentityLess legacy;
    EXPECT_FALSE(legacy(negative, zero)); // The former empty-key-first policy differs.
    EXPECT_EQ(legacy(negative, zero), canonical_key(negative) < canonical_key(zero));
    const auto line = [](double slope)
    {
        const std::array<GeometryPoint, 3> points{{{0, 0, 0}, {1, slope, 0}, {0, 0, 1}}};
        return PlanarFace<ordinary>::make(points)->edge_support_identities().front();
    };
    EXPECT_LT(line(std::ldexp(1., -66)), line(std::ldexp(1., -65)));
}
} // namespace
