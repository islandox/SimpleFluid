/** @file testFilteredSurfaceGeometry.cc
 * @brief Analytic exact-level scalar tetrahedron and canonical facet cases.
 */
#include "FVM/details/GeneralVoF/FilteredSurfaceGeometry.hh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid::VoF;
using namespace SimpleFluid::VoF::general_vof_detail::filtered_surface_kernel;

Tetra unit_tetra(std::array<long double, 4> scalar)
{
    Tetra result;
    const std::array<Wide, 4> coordinates{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    for (unsigned i = 0; i < 4; ++i)
        result.v[i] = {{0, i}, coordinates[i], scalar[i]};
    return result;
}
std::vector<CutId> identities(const std::vector<Cut>& cuts)
{
    std::vector<CutId> result;
    for (const auto& cut : cuts)
        result.push_back(cut.id);
    return result;
}
} // namespace

TEST(FilteredSurfaceGeometryTest, ExactEdgeLevelHasAnalyticHalfTetraVolumeAndTriangle)
{
    // q=1-2x-y-z on the unit tetrahedron. The gas tetrahedron has vertices
    // (0,0,0), (1/2,0,0), (0,1,0), (0,0,1): volume 1/12.
    const auto tetra = unit_tetra({1, -1, 0, 0});
    EXPECT_DOUBLE_EQ(static_cast<double>(clipped_volume(tetra, 0)), 1.0 / 12);
    Wide normal{};
    const auto cuts = section(tetra, 0, normal);
    ASSERT_EQ(cuts.size(), 3);
    std::map<CutId, Wide> points;
    for (const auto& cut : cuts)
        points.emplace(cut.id, cut.p);
    EXPECT_EQ(points.at(CutId{NodeId{0, 0}, NodeId{0, 1}}), (Wide{.5L, 0, 0}));
    EXPECT_EQ(points.at(CutId{NodeId{0, 2}, NodeId{0, 2}}), (Wide{0, 1, 0}));
    EXPECT_EQ(points.at(CutId{NodeId{0, 3}, NodeId{0, 3}}), (Wide{0, 0, 1}));
    const auto area = norm(cross(sub(cuts[1].p, cuts[0].p), sub(cuts[2].p, cuts[0].p))) / 2;
    EXPECT_NEAR(static_cast<double>(area), std::sqrt(6.0) / 4, 1e-14);
    EXPECT_NEAR(static_cast<double>(dot(normal, Wide{2, 1, 1})), std::sqrt(6.0), 1e-14);
}

TEST(FilteredSurfaceGeometryTest, ExactFullFacetComesOnlyFromGasSide)
{
    // q=1-x-y-z is positive throughout the interior; its zero face is the
    // complete plane x+y+z=1. Reversing q leaves no gas volume or gas-side face.
    Wide normal{};
    const auto gas = unit_tetra({1, 0, 0, 0});
    EXPECT_DOUBLE_EQ(static_cast<double>(clipped_volume(gas, 0)), 1.0 / 6);
    const auto cuts = section(gas, 0, normal);
    ASSERT_EQ(cuts.size(), 3);
    for (const auto& cut : cuts)
    {
        EXPECT_EQ(cut.id.first, cut.id.second);
        EXPECT_NE(cut.id.first.id, 0);
    }
    EXPECT_NEAR(static_cast<double>(dot(normal, Wide{1, 1, 1})), std::sqrt(3.0), 1e-14);
    const auto carrier = unit_tetra({-1, 0, 0, 0});
    EXPECT_EQ(clipped_volume(carrier, 0), 0);
    EXPECT_TRUE(section(carrier, 0, normal).empty());
}

TEST(FilteredSurfaceGeometryTest, OnlyStructuralPointAndEdgeSectionsAreEmpty)
{
    Wide normal{};
    for (const auto scalar : {std::array<long double, 4>{1, 1, 1, 0}, std::array<long double, 4>{1, 1, 0, 0}})
    {
        const auto tetra = unit_tetra(scalar);
        EXPECT_DOUBLE_EQ(static_cast<double>(clipped_volume(tetra, 0)), 1.0 / 6);
        EXPECT_TRUE(section(tetra, 0, normal).empty());
    }
    try
    {
        (void) section(unit_tetra({0, 0, 0, 0}), 0, normal);
        FAIL() << "A full tetrahedral plateau must reject.";
    }
    catch (const GeneralVoFSurfaceResolutionError& error)
    {
        EXPECT_EQ(error.reason(), GeneralVoFSurfaceResolutionReason::AmbiguousTopology);
    }
}

TEST(FilteredSurfaceGeometryTest, FullFacetCancellationUsesIdentityAndOppositeOrientation)
{
    Wide normal{};
    auto ids = identities(section(unit_tetra({1, 0, 0, 0}), 0, normal));
    FullFacetMap facets;
    EXPECT_FALSE(match_full_facet(facets, ids, 7).has_value());
    auto opposite = ids;
    std::reverse(opposite.begin(), opposite.end());
    const auto match = match_full_facet(facets, opposite, 8);
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(*match, 7);
    EXPECT_THROW((void) match_full_facet(facets, ids, 9), GeneralVoFSurfaceResolutionError);
    FullFacetMap duplicate;
    EXPECT_FALSE(match_full_facet(duplicate, ids, 7).has_value());
    EXPECT_THROW((void) match_full_facet(duplicate, ids, 8), GeneralVoFSurfaceResolutionError);
    auto unrelated = ids;
    unrelated[0] = {{1, unrelated[0].first.id}, {1, unrelated[0].second.id}};
    EXPECT_FALSE(match_full_facet(duplicate, unrelated, 10).has_value());
    EXPECT_EQ(duplicate.size(), 2);
}

TEST(FilteredSurfaceGeometryTest, PublishedVolumeGateRejectsPositiveTranslatedRoundingDefect)
{
    const auto published_tetra = [](long double offset)
    {
        std::array<Point, 4> points;
        for (auto& point : points)
            point = {static_cast<double>(offset), static_cast<double>(offset), static_cast<double>(offset)};
        for (unsigned i = 1; i < 4; ++i)
            points[i][i - 1] = static_cast<double>(offset + 3.0L / 8);
        const std::array<std::array<unsigned, 3>, 4> outward{{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}};
        std::array<GeneralVoFSurfacePatch, 4> patches;
        for (unsigned i = 0; i < 4; ++i)
            for (auto vertex : outward[i])
                patches[i].vertices.push_back(points[vertex]);
        return patches;
    };
    // The wide source tetra has edge3/8 and exact volume9/1024. At2^50,
    // binary64 rounds each offset+3/8 to offset+1/2, leaving a positive but
    // incorrect published volume1/48. Pre-publication volume alone misses it.
    constexpr long double target = 9.0L / 1024, allowance = 8e-15L * target;
    const auto centered = published_tetra(0);
    EXPECT_EQ(admit_published_volume(centered, {0, 0, 0}, target, allowance), target);
    const auto offset = std::ldexp(1.0L, 50);
    const auto translated = published_tetra(offset);
    const Point origin{static_cast<double>(offset), static_cast<double>(offset), static_cast<double>(offset)};
    EXPECT_EQ(translated[1].vertices[1][0] - origin[0], .5);
    EXPECT_DOUBLE_EQ(static_cast<double>(admit_published_volume(translated, origin, 1.0L / 48, 8e-15L / 48)), 1.0 / 48);
    try
    {
        (void) admit_published_volume(translated, origin, target, allowance);
        FAIL() << "Positive published volume must satisfy the original geometric target.";
    }
    catch (const GeneralVoFSurfaceResolutionError& error)
    {
        EXPECT_EQ(error.reason(), GeneralVoFSurfaceResolutionReason::VolumeConstraint);
    }
}
