/** @file testBoundedPlanarFace.cc
 * @brief Independent projected face-overlap, incidence and authority checks.
 */
#include "geometry/BoundedPlanarFace.hh"
#include "geometry/ExactGeometryInstrumentation.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <gtest/gtest.h>
namespace
{
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::vector<BoundedPoint> face_on_support(const BoundedConvexGeometry& geometry, const BoundedPlaneAuthority& support)
{
    for (const auto& face : geometry.faces())
    {
        std::vector<BoundedPoint> loop;
        for (auto index : face)
            loop.push_back(geometry.points()[index]);
        if (std::all_of(loop.begin(), loop.end(), [&](const auto& p)
                { return std::find(p.supports().begin(), p.supports().end(), support) != p.supports().end(); }))
            return loop;
    }
    throw std::runtime_error("Fixture has no face on the expected source support.");
}
std::vector<BoundedPoint> bottom(const BoundedConvexGeometry& geometry)
{
    for (const auto& face : geometry.faces())
    {
        std::vector<BoundedPoint> loop;
        for (auto index : face)
            loop.push_back(geometry.points()[index]);
        if (std::all_of(
                loop.begin(), loop.end(), [](const auto& p) { return p.value[2] == 0 && p.absolute_error[2] == 0; }))
            return loop;
    }
    throw std::runtime_error("Fixture has no lower face.");
}
void encloses(const BoundedGeometryValue& value, long double expected)
{
    EXPECT_LE(value.lower(), expected);
    EXPECT_GE(value.upper(), expected);
}
TEST(BoundedPlanarFaceTest, AnalyticOverlapAndOppositeWindingRetainSignedTransferRatiosWithoutExactWork)
{
    const auto first = BoundedConvexGeometry::from_geometry(*ConvexCellGeometry::box({0, 0, 0}, {2, 2, 1}));
    const auto second = BoundedConvexGeometry::from_geometry(*ConvexCellGeometry::box({1, .5, 0}, {3, 1.5, 1}));
    const auto a = bottom(*first);
    auto b = bottom(*second);
    const auto common = bounded_common_supports(a);
    ASSERT_FALSE(common.empty());
    const auto support = common.front();
    enable_exact_geometry_statistics(true);
    reset_exact_geometry_statistics();
    struct DisableOnExit
    {
        ~DisableOnExit() { enable_exact_geometry_statistics(false); }
    } disable;
    const auto result = bounded_planar_face_overlap(a, b, support);
    EXPECT_EQ(result.orientation, 1);
    encloses(result.first_projected_area, 4);
    encloses(result.second_projected_area, 2);
    encloses(result.overlap_projected_area, 1);
    encloses(result.overlap_over_first, .25L);
    encloses(result.overlap_over_second, .5L);
    ASSERT_GE(result.overlap_vertices.size(), 3);
    const auto repeated = bounded_planar_face_overlap(result.overlap_vertices, a, support);
    encloses(repeated.overlap_over_first, 1);
    encloses(repeated.overlap_projected_area, 1);
    std::reverse(b.begin(), b.end());
    const auto reversed = bounded_planar_face_overlap(a, b, support);
    EXPECT_EQ(reversed.orientation, -1);
    encloses(reversed.overlap_projected_area, 1);
    encloses(reversed.overlap_over_first, -.25L);
    encloses(reversed.overlap_over_second, -.5L);
    const auto self = bounded_planar_face_overlap(a, a, support);
    encloses(self.overlap_over_first, 1);
    const auto statistics = exact_geometry_statistics();
    EXPECT_EQ(statistics.measure_values, 0);
    EXPECT_EQ(statistics.measure_arithmetic, 0);
    EXPECT_EQ(statistics.volume_evaluations, 0);
    EXPECT_EQ(statistics.moment_evaluations, 0);
    EXPECT_EQ(statistics.clip_constructions, 0);
    EXPECT_EQ(statistics.intersection_constructions, 0);
}
TEST(BoundedPlanarFaceTest, CoplanarDisjointEdgeAndPointContactsHaveProvenZeroArea)
{
    const auto parent = BoundedConvexGeometry::from_geometry(*ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1}));
    const auto a = bottom(*parent);
    const auto support = bounded_common_supports(a).front();
    for (const auto low : {GeometryPoint{2, 0, 0}, GeometryPoint{1, 0, 0}, GeometryPoint{1, 1, 0}})
    {
        const auto other =
            BoundedConvexGeometry::from_geometry(*ConvexCellGeometry::box(low, {low[0] + 1, low[1] + 1, 1}));
        const auto result = bounded_planar_face_overlap(a, bottom(*other), support);
        EXPECT_EQ(result.overlap_projected_area.value, 0);
        EXPECT_EQ(result.overlap_projected_area.absolute_error, 0);
        EXPECT_EQ(result.overlap_over_first.value, 0);
        EXPECT_EQ(result.overlap_over_second.value, 0);
    }
}
TEST(BoundedPlanarFaceTest, ObliqueRationalSourceCutsUseOriginalSupportAndIndependentAreaOracle)
{
    const std::array<GeometryPoint, 4> vertices{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const auto exact = ConvexCellGeometry::make(vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    const auto first = BoundedConvexGeometry::from_geometry(*exact);
    std::vector<BoundedPoint> a;
    for (const auto& face : first->faces())
    {
        std::vector<BoundedPoint> loop;
        for (auto index : face)
            loop.push_back(first->points()[index]);
        if (std::all_of(
                loop.begin(), loop.end(), [](const auto& p) { return p.value[0] + p.value[1] + p.value[2] == 1; }))
        {
            a = std::move(loop);
            break;
        }
    }
    ASSERT_EQ(a.size(), 3);
    const auto support = bounded_common_supports(a).front();
    const double cut = 1. / 3;
    const auto split = exact->clip({{1, 0, 0}, {0, 0, 0}, cut});
    ASSERT_TRUE(split.inside);
    const auto second = BoundedConvexGeometry::from_geometry(*split.inside);
    const auto b = face_on_support(*second, support);
    const auto result = bounded_planar_face_overlap(a, b, support);
    const long double retained = 2 * static_cast<long double>(cut) - static_cast<long double>(cut) * cut;
    encloses(result.overlap_over_first, retained);
    encloses(result.overlap_over_second, 1);
    const auto repeated = bounded_planar_face_overlap(result.overlap_vertices, a, support);
    encloses(repeated.overlap_over_first, 1);
}
TEST(BoundedPlanarFaceTest, ForeignSupportTamperedPointsNonconvexLoopsAndResourceCapsReject)
{
    const auto box = BoundedConvexGeometry::from_geometry(*ConvexCellGeometry::box({0, 0, 0}, {1, 1, 1}));
    const auto a = bottom(*box);
    const auto support = bounded_common_supports(a).front();
    auto bad = a;
    bad[0].value[0] += .01;
    EXPECT_THROW((void) bounded_planar_face_overlap(a, bad, support), std::invalid_argument);
    auto foreign = support;
    foreign.key.push_back(13);
    EXPECT_THROW((void) bounded_planar_face_overlap(a, a, foreign), std::invalid_argument);
    bad = a;
    std::swap(bad[1], bad[2]);
    EXPECT_THROW((void) bounded_planar_face_overlap(a, bad, support), std::invalid_argument);
    BoundedGeometryLimits limits;
    limits.maximum_vertices = 3;
    EXPECT_THROW((void) bounded_planar_face_overlap(a, a, support, limits), std::invalid_argument);
    limits = {};
    limits.maximum_operations = 1;
    EXPECT_THROW((void) bounded_planar_face_overlap(a, a, support, limits), std::invalid_argument);
    limits = {};
    limits.byte_budget = 1;
    EXPECT_THROW((void) bounded_planar_face_overlap(a, a, support, limits), std::invalid_argument);
}
} // namespace
