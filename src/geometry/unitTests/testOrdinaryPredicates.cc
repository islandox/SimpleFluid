/** @file testOrdinaryPredicates.cc
 * @brief Independent finite analytic predicates and strict native planar admission.
 */
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
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
using Point = predicate::Point;
using Face = PLIC::PlanarFace<SimpleFluid::ArithmeticMode::Ordinary>;

int parity(const std::array<unsigned, 4>& order)
{
    unsigned inversions = 0;
    for (unsigned i = 0; i < 4; ++i)
        for (unsigned j = i + 1; j < 4; ++j)
            inversions += order[i] > order[j];
    return inversions % 2 ? -1 : 1;
}
void expect_analytic(const predicate::Result& result, long double expected)
{
    EXPECT_EQ(predicate::require_sign(result), expected > 0 ? 1 : expected < 0 ? -1 : 0);
    // The oracle consists only of representable dyadic inputs and a known
    // affine determinant. No production clipping or rational reference runs.
    EXPECT_LE(std::abs(result.value - expected), result.absolute_error);
}
TEST(OrdinaryPredicates, AnalyticTetrahedronRetainsPermutationParityAndPowerOfTwoUnits)
{
    const auto epsilon = std::ldexp(1.L, -40);
    const std::array<Point, 4> original{{{0, 0, 0}, {1, 1, 1}, {1, 1 + epsilon, 1}, {1, 1, 1 + epsilon}}};
    for (int exponent : {-40, 0, 40})
        for (long double translation : {0.L, 1024.L})
        {
            auto points = original;
            for (auto& point : points)
                for (auto& x : point)
                    x = std::ldexp(x + translation, exponent);
            const auto canonical = predicate::orientation(points);
            std::array<unsigned, 4> order{0, 1, 2, 3};
            do
            {
                SCOPED_TRACE(::testing::Message()
                             << exponent << ',' << translation << ',' << order[0] << order[1] << order[2] << order[3]);
                const auto result =
                    predicate::orientation({points[order[0]], points[order[1]], points[order[2]], points[order[3]]});
                expect_analytic(result, parity(order) * std::ldexp(1.L, 3 * exponent - 80));
                EXPECT_EQ(result.value, parity(order) * canonical.value);
                EXPECT_EQ(result.absolute_error, canonical.absolute_error);
            } while (std::next_permutation(order.begin(), order.end()));
        }
}
TEST(OrdinaryPredicates, PositiveGrazingCutStaysDistinctFromContactAndReversedWinding)
{
    for (int exponent : {-20, -60, -100})
    {
        const auto delta = std::ldexp(1.L, exponent);
        std::array<Point, 4> points{{{0, 0, 0}, {delta, 0, 0}, {0, 2 * delta, 0}, {0, 0, 4 * delta}}};
        expect_analytic(predicate::orientation(points), std::ldexp(1.L, 3 * exponent + 3));
        std::swap(points[1], points[2]);
        expect_analytic(predicate::orientation(points), -std::ldexp(1.L, 3 * exponent + 3));
        points[3] = {delta, delta, 0};
        expect_analytic(predicate::orientation(points), 0);
    }
}
TEST(OrdinaryPredicates, ProjectedTriangleRetainsCyclicAndReversedFaceWinding)
{
    const std::array<Point, 3> points{{{0, 0, 0}, {1, 2, 3}, {2, 1, 4}}};
    const std::array<long double, 3> components{5, 2, -3};
    std::array<unsigned, 3> order{0, 1, 2};
    do
    {
        unsigned inversions = 0;
        for (unsigned i = 0; i < 3; ++i)
            for (unsigned j = i + 1; j < 3; ++j)
                inversions += order[i] > order[j];
        for (unsigned drop = 0; drop < 3; ++drop)
            expect_analytic(
                predicate::projected_orientation(points[order[0]], points[order[1]], points[order[2]], drop),
                (inversions % 2 ? -1 : 1) * components[drop]);
    } while (std::next_permutation(order.begin(), order.end()));
}
TEST(OrdinaryPredicates, PlaneDistanceRetainsCanceledProductsAndRepresentedOffset)
{
    const auto epsilon = std::ldexp(1.L, -40);
    // (1+e)(1-e)-1 = -e^2; direct rounded products can erase it.
    expect_analytic(predicate::plane_distance(Point{1 + epsilon, 1, 0}, Point{1 - epsilon, -1, 0}, Point{}, 0),
        -std::ldexp(1.L, -80));
    expect_analytic(predicate::plane_distance(Point{1, 2, 3}, Point{2, 3, 4}, Point{1, 1, 1}, 11), 0);
}
TEST(OrdinaryPredicates, CrossFrameDistanceRetainsCancellationBeforeWorldRounding)
{
    const auto epsilon = std::ldexp(1.L, -40);
    const predicate::FramedPoint point{{-1024.L + epsilon, 0, 0}, {1024. + std::ldexp(1., -10), 0, 0}, -20};
    const predicate::FramedPoint anchor{{}, {1024, 0, 0}, -20};
    // The physical residual is 2^-60 at a world origin 1024; retaining it must
    // not depend on constructing and subtracting those rounded world points.
    expect_analytic(predicate::plane_distance(point, anchor, {1, 0, 0}, 0), epsilon);
    expect_analytic(predicate::plane_distance(point, anchor, {-1, 0, 0}, 0), -epsilon);
}
TEST(OrdinaryPredicates, CrossFrameDistanceRetainsTranslationAndPowerOfTwoUnits)
{
    for (int unit : {-40, 0, 40})
        for (double translation : {0., 1024.})
        {
            predicate::FramedPoint point{{.25L, .5L, 1.L}, {1, 2, 3}, 2 + unit};
            predicate::FramedPoint anchor{{1, 2, 4}, {.5, 1, 1.5}, -1 + unit};
            for (unsigned a = 0; a < 3; ++a)
            {
                point.origin[a] = std::ldexp(point.origin[a] + translation, unit);
                anchor.origin[a] = std::ldexp(anchor.origin[a] + translation, unit);
            }
            expect_analytic(predicate::plane_distance(point, anchor, {2, -1, .5}, .5), 3);
        }
}
TEST(OrdinaryPredicates, CrossFrameRangeFailureIsTypedAndNeverInventsContact)
{
    const predicate::FramedPoint point{{std::numeric_limits<long double>::max(), 0, 0}, {}, 1};
    try
    {
        (void) predicate::plane_distance(point, {}, {1, 0, 0}, 0);
        FAIL();
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::UnresolvedArithmetic);
        EXPECT_TRUE(error.recoverable_trial());
    }
}
TEST(OrdinaryPredicates, UnderflowIsUnresolvedRatherThanAZeroContact)
{
    using L = std::numeric_limits<long double>;
    const auto delta = std::ldexp(1.L, (L::min_exponent - L::digits) / 3 - 4);
    const auto result = predicate::orientation({Point{}, Point{delta, 0, 0}, Point{0, delta, 0}, Point{0, 0, delta}});
    EXPECT_EQ(result.sign, predicate::Sign::Unresolved);
    EXPECT_GT(result.absolute_error, 0);
    try
    {
        (void) predicate::require_sign(result);
        FAIL() << "An unrepresentable positive determinant became a contact.";
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::UnresolvedArithmetic);
        EXPECT_TRUE(error.recoverable_trial());
    }
}
TEST(OrdinaryPredicates, InvalidInputsAndResourceExhaustionAreNotRecoverableTrials)
{
    try
    {
        (void) predicate::orientation(
            {Point{}, Point{1, 0, 0}, Point{0, 1, 0}, Point{0, 0, std::numeric_limits<long double>::infinity()}});
        FAIL();
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::InvalidInput);
        EXPECT_FALSE(error.recoverable_trial());
    }
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = SimpleFluid::ArithmeticMode::Ordinary;
    limits.maximum_operations = 1;
    PLIC::details::Operation operation(limits);
    try
    {
        (void) predicate::orientation({Point{}, Point{1, 0, 0}, Point{0, 1, 0}, Point{0, 0, 1}});
        FAIL();
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
        EXPECT_FALSE(error.recoverable_trial());
    }
}
TEST(OrdinaryPredicates, DirectedCallerRoundingIsRestoredAfterResultsAndFailures)
{
    struct Restore
    {
        int rounding = std::fegetround();
        ~Restore() { std::fesetround(rounding); }
    } restore;
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    const std::array<Point, 4> points{{{0, 0, 0}, {1, 1, 1}, {1, 1.1L, 1}, {1, 1, 1.3L}}};
    const auto reference = predicate::orientation(points);
    for (int rounding : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    {
        ASSERT_EQ(std::fesetround(rounding), 0);
        const auto directed = predicate::orientation(points);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(directed.value, reference.value);
        EXPECT_EQ(directed.absolute_error, reference.absolute_error);
        EXPECT_EQ(directed.sign, reference.sign);
        predicate::ResultAccumulator sum;
        sum.add(directed);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(sum.result().value, reference.value);
        EXPECT_EQ(std::fegetround(), rounding);
    }
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    try
    {
        (void) predicate::orientation(
            {Point{}, Point{1, 0, 0}, Point{0, 1, 0}, Point{0, 0, std::numeric_limits<long double>::infinity()}});
        FAIL();
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::InvalidInput);
        EXPECT_FALSE(error.recoverable_trial());
    }
    EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = SimpleFluid::ArithmeticMode::Ordinary;
    limits.maximum_operations = 1;
    PLIC::details::Operation operation(limits);
    try
    {
        (void) predicate::orientation(points);
        FAIL();
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
        EXPECT_FALSE(error.recoverable_trial());
    }
    EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
}
TEST(OrdinaryPredicates, AggregateRetainsUnresolvedSignedTermsAndClassifiesOnlyTheTotal)
{
    const auto epsilon = std::ldexp(1.L, -40);
    predicate::ResultAccumulator sum;
    sum.add({1, 0, predicate::Sign::Positive});
    sum.add({epsilon, 2 * epsilon, predicate::Sign::Unresolved});
    sum.add({-epsilon, 2 * epsilon, predicate::Sign::Unresolved});
    const auto total = sum.result(3);
    EXPECT_EQ(total.value, 8);
    EXPECT_EQ(total.sign, predicate::Sign::Positive);
    EXPECT_GE(total.absolute_error, 32 * epsilon);
    // The input errors must survive even when the signed nominal sum cancels.
    sum.add({-1, 0, predicate::Sign::Negative});
    const auto contact = sum.result();
    EXPECT_EQ(contact.value, 0);
    EXPECT_GT(contact.absolute_error, 0);
    EXPECT_EQ(contact.sign, predicate::Sign::Unresolved);
}
TEST(OrdinaryPredicates, AggregateCompensatesSignedCancellationAndRetainsCaps)
{
    const auto epsilon = std::ldexp(1.L, -std::numeric_limits<long double>::digits - 4);
    predicate::ResultAccumulator sum;
    sum.add({1, 0, predicate::Sign::Positive});
    sum.add({epsilon, 0, predicate::Sign::Positive});
    sum.add({-1, 0, predicate::Sign::Negative});
    expect_analytic(sum.result(), epsilon);
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = SimpleFluid::ArithmeticMode::Ordinary;
    limits.maximum_operations = 1;
    PLIC::details::Operation operation(limits);
    try
    {
        sum.add({1, 0, predicate::Sign::Positive});
        FAIL();
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::ResourceLimit);
        EXPECT_FALSE(error.recoverable_trial());
    }
}
TEST(OrdinaryPredicates, DefiningObliqueTriangleIsStructurallyPlanar)
{
    const std::array<PLIC::GeometryPoint, 3> points{{{0, 0, 0}, {.25, .5, .1}, {.5, .1, .3}}};
    const auto face = Face::make(points);
    ASSERT_TRUE(face);
    EXPECT_GT(face->area().value, 0);
    const auto reversed = face->reversed();
    EXPECT_NEAR(face->area().value, reversed->area().value, face->area().value * 1e-15L);
    for (const auto& point : face->vertex_identities())
    {
        EXPECT_TRUE(face->contains_point(point));
        EXPECT_TRUE(reversed->contains_point(point));
    }
    EXPECT_NEAR(face->coplanar_area_ratio(*reversed).as_double(), -1, 1e-15);
}
TEST(OrdinaryPredicates, AdditionalPolygonVerticesRequirePlanarityAndConvexity)
{
    const std::array<PLIC::GeometryPoint, 4> planar{{{0, 0, 0}, {1, 0, .5}, {1, 1, .75}, {0, 1, .25}}};
    EXPECT_NO_THROW((void) Face::make(planar));
    auto nonplanar = planar;
    nonplanar[3][2] = std::nextafter(nonplanar[3][2], 1.);
    try
    {
        (void) Face::make(nonplanar);
        FAIL() << "A genuinely nonplanar fourth vertex was admitted.";
    }
    catch (const PLIC::GeometryError& error)
    {
        EXPECT_EQ(error.code(), PLIC::GeometryErrorCode::InvalidTopology);
        EXPECT_FALSE(error.recoverable_trial());
    }
    const std::array<PLIC::GeometryPoint, 5> concave{{{0, 0, 0}, {1, 0, 0}, {.25, .25, 0}, {1, 1, 0}, {0, 1, 0}}};
    EXPECT_THROW((void) Face::make(concave), PLIC::GeometryError);
}
TEST(OrdinaryPredicates, FinitePrimitiveWorkNeverInvokesTheRationalBackend)
{
    PLIC::reset_geometry_statistics();
    expect_analytic(predicate::orientation({Point{}, Point{1, 0, 0}, Point{0, 1, 0}, Point{0, 0, 1}}), 1);
    const std::array<PLIC::GeometryPoint, 3> triangle{{{0, 0, 0}, {.25, .5, .1}, {.5, .1, .3}}};
    EXPECT_NO_THROW((void) Face::make(triangle));
    const auto stats = PLIC::geometry_statistics();
    EXPECT_GT(stats.floating_operations, 0u);
    EXPECT_EQ(stats.exact_predicates, 0u);
    EXPECT_EQ(stats.rational_constructions, 0u);
    EXPECT_EQ(stats.rational_measures, 0u);
    EXPECT_EQ(stats.rational_ledgers, 0u);
}
} // namespace
