/** @file testOrdinarySupportPredicates.cc
 * @brief Independent constraints for the bounded indirect Ordinary predicate.
 */
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace
{
namespace PLIC = SimpleFluid::PLIC;
namespace predicate = PLIC::details::ordinary_predicates;
using Row = predicate::SupportRow;
using Source = std::array<predicate::FramedPoint, 4>;

Row plane(predicate::Point normal, long double offset = 0)
{
    Row result;
    result.normal = normal;
    result.offset = offset;
    return result;
}
Source unit_source()
{
    return {{{{0, 0, 0}, {}, 0}, {{1, 0, 0}, {}, 0}, {{0, 1, 0}, {}, 0}, {{0, 0, 1}, {}, 0}}};
}
void expect_distance(const predicate::SupportEvaluation& result, long double expected)
{
    ASSERT_FALSE(result.singular);
    EXPECT_EQ(static_cast<int>(result.distance.sign), expected > 0 ? 1 : expected < 0 ? -1 : 0);
    EXPECT_LE(std::abs(result.distance.value - expected), result.distance.absolute_error);
    if (expected == 0)
    {
        EXPECT_EQ(result.distance.value, 0);
        EXPECT_EQ(result.distance.absolute_error, 0);
    }
}
TEST(OrdinarySupportPredicates, SourceFacesDefineKnownBarycentricCoordinates)
{
    std::array<Row, 3> basis;
    basis[0].source_face = 1;
    basis[1].source_face = 2;
    basis[2].source_face = 3;
    Row query;
    query.source_face = 0;
    expect_distance(predicate::support_intersection_distance(unit_source(), basis, query), 1);
    query.source_face = 1;
    expect_distance(predicate::support_intersection_distance(unit_source(), basis, query), 0);
}
TEST(OrdinarySupportPredicates, CorrelatedSkewInputCancellationRetainsContactAndBothNextafterSides)
{
    const auto e = std::ldexp(1.L, -40);
    // P=(1-e)x-y. At source vertex 1 its exact coefficient is -4e^2,
    // although the first product and subtraction round to zero directly.
    // The constrained point is x=1+e, y=1-e^2, z=1/4. It need never be
    // materialized: Q=P+z-1/4 is identically zero there.
    const Source source{{{{0, 0, 0}, {}, 0}, {{4 + 4 * e, 4, 0}, {}, 0}, {{0, -4, 0}, {}, 0}, {{0, 0, 4}, {}, 0}}};
    const std::array<Row, 3> basis{plane({1 - e, -1, 0}), plane({1, 0, 0}, 1 + e), plane({0, 0, 1}, .25L)};
    auto query = plane({1 - e, -1, 1}, .25L);
    expect_distance(predicate::support_intersection_distance(source, basis, query), 0);
    for (const auto direction : {-1.L, 1.L})
    {
        query.offset = std::nextafter(.25L, direction);
        expect_distance(predicate::support_intersection_distance(source, basis, query), .25L - query.offset);
    }
}
TEST(OrdinarySupportPredicates, BasisPermutationAndOrientationCancelWhileQueryComplementReversesMetric)
{
    const auto source = unit_source();
    const std::array<Row, 3> original{plane({1, 0, 0}, .25L), plane({0, 1, 0}, .125L), plane({0, 0, 1}, .0625L)};
    std::array<unsigned, 3> order{0, 1, 2};
    do
    {
        for (unsigned signs = 0; signs < 8; ++signs)
        {
            auto basis = original;
            for (unsigned j = 0; j < 3; ++j)
            {
                basis[j] = original[order[j]];
                basis[j].orientation = (signs & (1u << j)) ? -1 : 1;
            }
            auto query = plane({3, -2, 4}, .5L);
            expect_distance(predicate::support_intersection_distance(source, basis, query), .25L);
            query.orientation = -1;
            expect_distance(predicate::support_intersection_distance(source, basis, query), -.25L);
        }
    } while (std::next_permutation(order.begin(), order.end()));
}
TEST(OrdinarySupportPredicates, OriginalSourceIncidenceOverridesRoundedPlaneResidual)
{
    std::array<Row, 3> basis;
    basis[0].source_face = 1;
    basis[1].source_face = 2;
    basis[2].source_face = 3;
    auto query = plane({1, 2, 3}, std::ldexp(1.L, -80));
    // The authoritative source incidence can differ from the represented
    // coefficient expression of an already constructed native plane.
    query.source_incidence = 1;
    expect_distance(predicate::support_intersection_distance(unit_source(), basis, query), 0);
    query.source_incidence = 0;
    expect_distance(predicate::support_intersection_distance(unit_source(), basis, query), -query.offset);
}
TEST(OrdinarySupportPredicates, ExactSingularityIsSeparateFromQueryContact)
{
    std::array<Row, 3> basis{plane({1, 0, 0}), plane({2, 0, 0}), plane({0, 0, 1})};
    const auto singular = predicate::support_intersection_distance(unit_source(), basis, plane({0, 1, 0}));
    EXPECT_TRUE(singular.singular);
    basis[1] = plane({0, 1, 0});
    expect_distance(predicate::support_intersection_distance(unit_source(), basis, plane({0, 0, 1})), 0);
}
TEST(OrdinarySupportPredicates, QueryComplementPreservesAnExactlyRepresentableSubnormalOffset)
{
    const std::array<Row, 3> basis{plane({1, 0, 0}), plane({0, 1, 0}), plane({0, 0, 1})};
    const auto tiny = std::numeric_limits<long double>::denorm_min();
    auto query = plane({}, -tiny);
    expect_distance(predicate::support_intersection_distance(unit_source(), basis, query), tiny);
    query.orientation = -1;
    expect_distance(predicate::support_intersection_distance(unit_source(), basis, query), -tiny);
}
TEST(OrdinarySupportPredicates, TranslationsAndPowerOfTwoFramesRetainAnchorUnitDistances)
{
    for (const auto exponent : {-40, 0, 40})
        for (const auto translation : {0., 1024.})
        {
            auto source = unit_source();
            std::array<Row, 3> basis{plane({1, 0, 0}, .25L), plane({0, 1, 0}, .125L), plane({0, 0, 1}, .0625L)};
            auto query = plane({3, -2, 4}, .5L);
            for (auto& point : source)
            {
                point.origin = {translation, translation, translation};
                point.exponent = exponent;
            }
            for (auto& row : basis)
            {
                row.anchor.origin = source[0].origin;
                row.anchor.exponent = exponent;
            }
            query.anchor = basis[0].anchor;
            expect_distance(predicate::support_intersection_distance(source, basis, query), .25L);
        }
}
TEST(OrdinarySupportPredicates, NonbinaryMetricCarriesResidualError)
{
    const std::array<Row, 3> basis{plane({3, 0, 0}, 1), plane({0, 1, 0}), plane({0, 0, 1})};
    const auto result = predicate::support_intersection_distance(unit_source(), basis, plane({1, 0, 0}));
    ASSERT_FALSE(result.singular);
    EXPECT_EQ(result.distance.sign, predicate::Sign::Positive);
    EXPECT_GT(result.distance.absolute_error, 0);
    // Independent residual for the analytic rational x=1/3. A fused
    // subtraction retains the error hidden by rounded value*3 == 1.
    EXPECT_LE(std::abs(std::fma(result.distance.value, 3.L, -1.L)), 3 * result.distance.absolute_error);
}
TEST(OrdinarySupportPredicates, RangeAndResourceFailuresPreserveCallerEnvironment)
{
    struct Restore
    {
        std::fenv_t saved{};
        Restore() { std::fegetenv(&saved); }
        ~Restore() { std::fesetenv(&saved); }
    } restore;
    const std::array<Row, 3> basis{plane({1, 0, 0}, .25L), plane({0, 1, 0}, .125L), plane({0, 0, 1}, .0625L)};
    const auto reference = predicate::support_intersection_distance(unit_source(), basis, plane({1, 1, 1}));
    for (const int rounding : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    {
        ASSERT_EQ(std::fesetround(rounding), 0);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_INVALID);
        const auto result = predicate::support_intersection_distance(unit_source(), basis, plane({1, 1, 1}));
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
        EXPECT_EQ(result.distance.value, reference.distance.value);
        EXPECT_EQ(result.distance.absolute_error, reference.distance.absolute_error);
        auto source = unit_source();
        source[1].local[0] = std::numeric_limits<long double>::max();
        source[1].exponent = 1;
        try
        {
            (void) predicate::support_intersection_distance(source, basis, plane({1, 1, 1}));
            FAIL();
        }
        catch (const PLIC::GeometryError& error)
        {
            EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::UnresolvedArithmetic);
        }
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
        for (const bool scalar_limit : {false, true})
        {
            PLIC::GeometryLimits limits;
            limits.arithmetic_mode = SimpleFluid::ArithmeticMode::Ordinary;
            limits.maximum_operations = scalar_limit ? 1 : 1000000;
            limits.byte_budget = scalar_limit ? 1000000 : 1;
            PLIC::details::Operation operation(limits);
            try
            {
                (void) predicate::support_intersection_distance(unit_source(), basis, plane({1, 1, 1}));
                FAIL();
            }
            catch (const PLIC::GeometryError& error)
            {
                EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
                EXPECT_FALSE(error.recoverable_trial());
            }
            EXPECT_EQ(std::fegetround(), rounding);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
        }
    }
}
TEST(OrdinarySupportPredicates, UnderflowLatticeNeverBecomesContact)
{
    auto source = unit_source();
    source[1].local[0] = std::numeric_limits<long double>::denorm_min();
    const std::array<Row, 3> basis{plane({.5L, 0, 0}), plane({0, 1, 0}), plane({0, 0, 1})};
    try
    {
        (void) predicate::support_intersection_distance(source, basis, plane({1, 1, 1}));
        FAIL();
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::UnresolvedArithmetic);
    }
}
TEST(OrdinarySupportPredicates, CapturedFace20DefiningSupportsProveContactAndBothPerturbations)
{
    using predicate::FramedPoint;
    using predicate::SupportRow;
    // Minimal captured forward contact for ordinary_predicates.
    // Original source and support expressions; exact hexadecimal literals from raw payloads.
    std::array<FramedPoint, 4> captured_source{};
    captured_source[0].local = {0xe6666666666664cdp-65L, 0x8000000000000000p-65L, 0x8000000000000000p-64L};
    captured_source[1].local = {0xe661cd3ebeb43883p-65L, 0x80016a2e733017e1p-65L, 0xffffdb9e8eeedf9ep-65L};
    captured_source[2].local = {0xffef0e68880ebe89p-66L, 0x0p+0L, 0xfffff27cff5b0d57p-65L};
    captured_source[3].local = {0xb3343b01ba0576e4p-65L, 0x8003a6c05f50f6e8p-64L, 0x80000117e0e469cep-64L};
    std::array<SupportRow, 3> captured_basis{};
    captured_basis[0].source_face = 0;
    SupportRow captured_query{};
    captured_basis[1].normal = {0x0p+0L, 0xfffffffffffff800p-67L, 0x0p+0L};
    captured_basis[1].anchor.local = {0xb333333333333000p-65L, 0x8000000000000000p-64L, 0x0p+0L};
    captured_basis[1].anchor.origin = {0x0p-1074, 0x0p-1074, 0x0p-1074};
    captured_basis[1].anchor.exponent = 0;
    captured_basis[1].offset = 0x0p+0L;
    captured_basis[1].orientation = 1;
    captured_basis[2].normal = {0x8000000000000000p-65L, -0xccccccccccccc000p-68L, 0x0p+0L};
    captured_basis[2].anchor.local = {0x8000000000000000p-65L, 0x0p+0L, 0x0p+0L};
    captured_basis[2].anchor.origin = {0x0p-1074, 0x0p-1074, 0x0p-1074};
    captured_basis[2].anchor.exponent = 0;
    captured_basis[2].offset = 0x0p+0L;
    captured_basis[2].orientation = 1;
    captured_query.normal = {0x8000000000000000p-65L, -0xcccccccccccce000p-68L, 0x0p+0L};
    captured_query.anchor.local = {0xfffffffffffff800p-66L, 0x0p+0L, 0x0p+0L};
    captured_query.anchor.origin = {0x1999999999999ap-56, 0x10000000000000p-53, 0x10000000000000p-53};
    captured_query.anchor.exponent = 0;
    captured_query.offset = 0x0p+0L;
    captured_query.orientation = 1;

    // Captured source12, forward donor29/cut3, complementary witness10.
    // Two skew native lines meet at represented x=.35, y=.5; lambda0=0
    // fixes the vertex on the source tetrahedron face. The query contains
    // that edge, independently of the rounded barycentric coordinates.
    const auto cached = predicate::plane_distance(
        captured_source[1], captured_query.anchor, captured_query.normal, captured_query.offset);
    EXPECT_GT(cached.absolute_error, 0);
    for (unsigned order = 0; order < 2; ++order)
    {
        for (int orientation : {-1, 1})
        {
            captured_query.orientation = orientation;
            captured_query.offset = 0;
            expect_distance(
                predicate::support_intersection_distance(captured_source, captured_basis, captured_query), 0);
            for (const auto delta : {-0x1p-70L, 0x1p-70L})
            {
                captured_query.offset = delta;
                expect_distance(
                    predicate::support_intersection_distance(captured_source, captured_basis, captured_query),
                    -orientation * delta);
            }
        }
        std::reverse(captured_basis.begin(), captured_basis.end());
    }
}
TEST(OrdinarySupportPredicates, DifferentOriginalFramesCancelBeforeWorldCoordinateRounding)
{
    auto source = unit_source();
    for (auto& point : source)
    {
        point.origin = {1024, -2048, 512};
        point.exponent = 1;
    }
    std::array<Row, 3> basis{plane({1, 0, 0}, .25L), plane({0, 1, 0}, .125L), plane({0, 0, 1}, .0625L)};
    for (auto& row : basis)
    {
        row.anchor.origin = source[0].origin;
        row.anchor.exponent = 1;
    }
    auto query = plane({3, -2, 4}, .5L);
    query.anchor.origin = {1025, -2049, 513};
    query.anchor.exponent = -1;
    // In query-frame units the intersection is (-1, 2.5, -1.75).
    expect_distance(predicate::support_intersection_distance(source, basis, query), -15.5L);
    query.orientation = -1;
    expect_distance(predicate::support_intersection_distance(source, basis, query), 15.5L);
}
} // namespace
