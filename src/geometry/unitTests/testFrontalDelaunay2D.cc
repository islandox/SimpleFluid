/**
 * @file testFrontalDelaunay2D.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for advancing-front XY point placement and Delaunay topology.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/mesh/FrontalDelaunay2D.hh"
#include "geometry/mesh/PredicateSigns2D.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <unordered_set>

namespace
{

using Mesher = SimpleFluid::Meshes::FrontalDelaunay2D;
using Vec3 = Mesher::Vec3;
using SimpleFluid::Meshes::detail::PredicateSign;
using SimpleFluid::Meshes::detail::orient2d_sign;
using SimpleFluid::Meshes::detail::incircle_sign;

long double orient2d(const Vec3& a, const Vec3& b, const Vec3& c)
{
    return (static_cast<long double>(b.x) - a.x)
         * (static_cast<long double>(c.y) - a.y)
         - (static_cast<long double>(b.y) - a.y)
         * (static_cast<long double>(c.x) - a.x);
}

bool opposite_strict(PredicateSign a, PredicateSign b)
{
    return (a == PredicateSign::Positive && b == PredicateSign::Negative)
        || (a == PredicateSign::Negative && b == PredicateSign::Positive);
}

unsigned opposite_vertex(const Mesher::Triangle& triangle,
                         const std::pair<unsigned, unsigned>& edge)
{
    for (unsigned vertex : triangle)
        if (vertex != edge.first && vertex != edge.second) return vertex;
    throw std::runtime_error("Triangle has no vertex opposite an interior edge.");
}

void expect_local_legality(const Mesher::Result& mesh,
                           const std::pair<unsigned, unsigned>& edge,
                           const Mesher::Triangle& first,
                           const Mesher::Triangle& second)
{
    const auto c = opposite_vertex(first, edge);
    const auto d = opposite_vertex(second, edge);
    const auto& u = mesh.nodes[edge.first];
    const auto& v = mesh.nodes[edge.second];
    if (!opposite_strict(orient2d_sign(u, v, mesh.nodes[c]),
                         orient2d_sign(u, v, mesh.nodes[d]))
        || !opposite_strict(orient2d_sign(mesh.nodes[c], mesh.nodes[d], u),
                            orient2d_sign(mesh.nodes[c], mesh.nodes[d], v)))
        return;
    EXPECT_NE(incircle_sign(mesh.nodes[first[0]], mesh.nodes[first[1]],
                           mesh.nodes[first[2]], mesh.nodes[d]),
              PredicateSign::Positive);
}

void expect_complete_planar_triangulation(const Mesher::Result& mesh)
{
    ASSERT_GE(mesh.nodes.size(), 3U);
    ASSERT_GE(mesh.boundary_edges.size(), 3U);
    EXPECT_EQ(mesh.triangles.size(),
              2U * mesh.nodes.size() - 2U - mesh.boundary_edges.size());

    std::unordered_set<unsigned> used_nodes;
    std::map<std::pair<unsigned, unsigned>, SimpleFluid::Arr<size_t>> adjacency;
    for (size_t t = 0; t < mesh.triangles.size(); ++t)
    {
        const auto& triangle = mesh.triangles[t];
        EXPECT_EQ(orient2d_sign(mesh.nodes[triangle[0]],
                                mesh.nodes[triangle[1]],
                                mesh.nodes[triangle[2]]), PredicateSign::Positive);
        used_nodes.insert(triangle[0]);
        used_nodes.insert(triangle[1]);
        used_nodes.insert(triangle[2]);
        for (unsigned side = 0; side < 3; ++side)
        {
            auto node0 = triangle[side];
            auto node1 = triangle[(side + 1U) % 3U];
            if (node1 < node0) std::swap(node0, node1);
            adjacency[{node0, node1}].push_back(t);
        }
    }
    EXPECT_EQ(used_nodes.size(), mesh.nodes.size());
    size_t exterior_edge_count = 0;
    size_t nonmanifold_edge_count = 0;
    for (const auto& [edge, adjacent] : adjacency)
    {
        exterior_edge_count += adjacent.size() == 1U;
        nonmanifold_edge_count += adjacent.size() > 2U;
        if (adjacent.size() == 2U)
            expect_local_legality(mesh, edge, mesh.triangles[adjacent[0]],
                                  mesh.triangles[adjacent[1]]);
    }
    EXPECT_EQ(exterior_edge_count, mesh.boundary_edges.size());
    EXPECT_EQ(nonmanifold_edge_count, 0U);
    for (const auto& boundary : mesh.boundary_edges)
        EXPECT_EQ(adjacency[std::minmax(boundary.node0, boundary.node1)].size(), 1U);
}

SimpleFluid::Arr<Vec3> regular_loop(unsigned count, double radius)
{
    SimpleFluid::Arr<Vec3> result;
    for (unsigned i = 0; i < count; ++i)
    {
        const auto angle = 2.0 * std::numbers::pi * i / count;
        result.push_back({radius * std::cos(angle), radius * std::sin(angle), 7.0});
    }
    return result;
}

long double polygon_twice_area(const SimpleFluid::Arr<Vec3>& polygon)
{
    long double area = 0.0L;
    for (size_t i = 0; i < polygon.size(); ++i)
        area += orient2d(polygon.front(), polygon[i], polygon[(i + 1) % polygon.size()]);
    return area;
}

void expect_complete_annular_triangulation(
    const Mesher::Result& mesh,
    const SimpleFluid::Arr<Vec3>& outer,
    const SimpleFluid::Arr<Vec3>& inner)
{
    const auto boundary_count = outer.size() + inner.size();
    ASSERT_EQ(mesh.boundary_edges.size(), boundary_count);
    ASSERT_GE(mesh.nodes.size(), boundary_count);
    EXPECT_EQ(mesh.triangles.size(), 2U * mesh.nodes.size() - boundary_count);
    size_t node_id = 0;
    for (const auto* loop : {&outer, &inner})
    {
        for (const auto& node : *loop)
        {
            EXPECT_DOUBLE_EQ(mesh.nodes[node_id].x, node.x);
            EXPECT_DOUBLE_EQ(mesh.nodes[node_id].y, node.y);
            EXPECT_DOUBLE_EQ(mesh.nodes[node_id].z, 0.0);
            ++node_id;
        }
    }
    std::map<std::pair<unsigned, unsigned>, SimpleFluid::Arr<size_t>> adjacency;
    std::unordered_set<unsigned> used;
    long double area = 0.0L;
    for (size_t t = 0; t < mesh.triangles.size(); ++t)
    {
        const auto& triangle = mesh.triangles[t];
        const auto triangle_area = orient2d(mesh.nodes[triangle[0]],
            mesh.nodes[triangle[1]], mesh.nodes[triangle[2]]);
        EXPECT_EQ(orient2d_sign(mesh.nodes[triangle[0]],
                                mesh.nodes[triangle[1]], mesh.nodes[triangle[2]]),
                  PredicateSign::Positive);
        area += triangle_area;
        for (unsigned side = 0; side < 3; ++side)
        {
            const auto a = triangle[side];
            const auto b = triangle[(side + 1U) % 3U];
            ASSERT_LT(a, mesh.nodes.size());
            used.insert(a);
            adjacency[std::minmax(a, b)].push_back(t);
        }
        // Every triangle wholly on the strictly convex inner loop is in the hole.
        EXPECT_FALSE(std::all_of(triangle.begin(), triangle.end(),
            [&](unsigned node) { return node >= outer.size()
                && node < outer.size() + inner.size(); }));
    }
    EXPECT_EQ(used.size(), mesh.nodes.size());
    EXPECT_NEAR(area, polygon_twice_area(outer) - polygon_twice_area(inner), 1.0e-12L);
    size_t exterior_edges = 0;
    for (const auto& [edge, adjacent] : adjacency)
    {
        EXPECT_LE(adjacent.size(), 2U);
        exterior_edges += adjacent.size() == 1;
        if (adjacent.size() != 2) continue;
        // The constrained perimeter is exterior; only flippable free interior
        // edges are subject to the local Delaunay legality check.
        expect_local_legality(mesh, edge, mesh.triangles[adjacent[0]],
                              mesh.triangles[adjacent[1]]);
    }
    EXPECT_EQ(exterior_edges, boundary_count);
    for (const auto& boundary : mesh.boundary_edges)
    {
        EXPECT_EQ(adjacency[std::minmax(boundary.node0, boundary.node1)].size(), 1U);
        const auto first_inner = static_cast<unsigned>(outer.size());
        if (boundary.node0 < first_inner)
        {
            EXPECT_EQ(boundary.node1, (boundary.node0 + 1U) % first_inner);
            EXPECT_EQ(boundary.batch_name, "outer-wall");
        }
        else
        {
            EXPECT_EQ(boundary.node0,
                first_inner + (boundary.node1 - first_inner + 1U) % inner.size());
            EXPECT_EQ(boundary.batch_name, "inner-wall");
        }
    }
}

} // namespace

/** @brief Verifies malformed fronts and invalid sizing parameters are rejected. */
TEST(FrontalDelaunay2DTest, RejectsInvalidPolygonAndSizing)
{
    EXPECT_THROW(
        Mesher::triangulate(
            {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}},
            0.25),
        std::invalid_argument);
    EXPECT_THROW(
        Mesher::triangulate(
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
             {0.25, 0.25, 0.0}, {0.0, 1.0, 0.0}},
            0.25),
        std::invalid_argument);
    EXPECT_THROW(
        Mesher::triangulate(
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
            0.0),
        std::invalid_argument);
    EXPECT_THROW(Mesher::triangulate(
        {{0, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}}, 0.5),
        std::invalid_argument);
    // The former coordinate-scale-squared tolerance swallowed this real notch.
    constexpr double origin = 1.0e8;
    EXPECT_THROW(Mesher::triangulate(
        {{origin, origin, 0}, {origin + 1, origin, 0},
         {origin + 1, origin + 1, 0},
         {origin + 0.5, origin + 0.75, 0},
         {origin, origin + 1, 0}}, 0.5),
        std::invalid_argument);
}

TEST(FrontalDelaunay2DTest, SmallScaleAndCollinearBoundaryAreConforming)
{
    for (double h : {1.0e-2, 1.0e-4, 1.0e-5})
    {
        const auto mesh = Mesher::triangulate(
            {{0, 0, 0}, {h, 0, 0}, {0, h, 0}}, h / 2);
        expect_complete_planar_triangulation(mesh);
    }
    const auto mesh = Mesher::triangulate(
        {{0, 0, 0}, {0.5, 0, 0}, {1, 0, 0},
         {1, 1, 0}, {0, 1, 0}}, 1.0);
    expect_complete_planar_triangulation(mesh);
    const auto repeat = Mesher::triangulate(
        {{0, 0, 0}, {0.5, 0, 0}, {1, 0, 0},
         {1, 1, 0}, {0, 1, 0}}, 1.0);
    EXPECT_EQ(mesh.triangles, repeat.triangles);
}

TEST(FrontalDelaunay2DTest, RoundedObliqueBoundarySubdivisionsStayConforming)
{
    const auto mesh = Mesher::triangulate(
        {{0, 0, 0}, {1, 0.3, 0}, {1.4, 1.3, 0}, {0.4, 1, 0}}, 0.23);
    expect_complete_planar_triangulation(mesh);
}

/**
 * @brief Verifies advancing-front triangulation covers a convex polygon with
 * consistently oriented, boundary-conforming triangles.
 */
TEST(FrontalDelaunay2DTest, AdvancesFrontAcrossConvexPolygon)
{
    const auto mesh = Mesher::triangulate(
        {{0.0, 0.0, 7.0}, {1.0, 0.0, 7.0},
         {1.0, 1.0, 7.0}, {0.0, 1.0, 7.0}},
        0.32, "wall");

    expect_complete_planar_triangulation(mesh);
    EXPECT_GT(mesh.nodes.size(), mesh.boundary_edges.size());
    for (const auto& node : mesh.nodes)
    {
        EXPECT_GE(node.x, -1.0e-12);
        EXPECT_LE(node.x, 1.0 + 1.0e-12);
        EXPECT_GE(node.y, -1.0e-12);
        EXPECT_LE(node.y, 1.0 + 1.0e-12);
        EXPECT_DOUBLE_EQ(node.z, 0.0);
    }
    for (const auto& edge : mesh.boundary_edges)
    {
        EXPECT_EQ(edge.batch_name, "wall");
        EXPECT_LE(std::hypot(mesh.nodes[edge.node1].x - mesh.nodes[edge.node0].x,
                             mesh.nodes[edge.node1].y - mesh.nodes[edge.node0].y),
                  0.32 + 1.0e-12);
    }
    for (const auto target : {0.5, 1.0})
    {
        const auto symmetric_mesh = Mesher::triangulate(
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
             {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}},
            target);
        expect_complete_planar_triangulation(symmetric_mesh);
    }
}

/**
 * @brief Verifies prescribed circular fronts produce valid annular and disk
 * triangulations with the requested boundary markers.
 */
TEST(FrontalDelaunay2DTest, TriangulatesPrescribedCircularFronts)
{
    const auto mesh = Mesher::triangulate_disk(
        {0.0, 0.6, 1.0}, 0.4, "radial");

    expect_complete_planar_triangulation(mesh);
    for (const auto& point : mesh.nodes)
    {
        EXPECT_LE(std::hypot(point.x, point.y), 1.0 + 1.0e-12);
    }
    for (const auto& edge : mesh.boundary_edges)
    {
        EXPECT_EQ(edge.batch_name, "radial");
        EXPECT_NEAR(std::hypot(mesh.nodes[edge.node0].x,
                               mesh.nodes[edge.node0].y),
                    1.0, 1.0e-12);
    }

    // Regression for a center point that falls on an existing Delaunay edge.
    const auto offset_front_mesh = Mesher::triangulate_disk(
        {0.0, 0.9, 1.0}, 0.5, "radial");
    expect_complete_planar_triangulation(offset_front_mesh);
}

/** @brief Fixed annular interfaces retain their points, edges, and hole. */
TEST(FrontalDelaunay2DTest, AdvancesAnnularFrontsWithoutSplittingInterfaces)
{
    const auto outer = regular_loop(48, 0.21);
    const auto inner = regular_loop(48, 0.078);
    const auto mesh = Mesher::triangulate_annulus(outer, inner, 0.025,
                                                 "outer-wall", "inner-wall");
    expect_complete_annular_triangulation(mesh, outer, inner);
    EXPECT_GT(mesh.nodes.size(), outer.size() + inner.size());
    const auto repeat = Mesher::triangulate_annulus(outer, inner, 0.025,
                                                   "outer-wall", "inner-wall");
    EXPECT_EQ(mesh.triangles, repeat.triangles);
    ASSERT_EQ(mesh.nodes.size(), repeat.nodes.size());
    for (size_t node = 0; node < mesh.nodes.size(); ++node)
    {
        EXPECT_DOUBLE_EQ(mesh.nodes[node].x, repeat.nodes[node].x);
        EXPECT_DOUBLE_EQ(mesh.nodes[node].y, repeat.nodes[node].y);
    }
}

/** @brief Inner and outer interfaces may have independent angular counts. */
TEST(FrontalDelaunay2DTest, PreservesUnequalAnnularInterfaceCounts)
{
    const auto outer = regular_loop(32, 0.21);
    const auto inner = regular_loop(12, 0.078);
    const auto mesh = Mesher::triangulate_annulus(outer, inner, 0.025,
                                                 "outer-wall", "inner-wall");
    expect_complete_annular_triangulation(mesh, outer, inner);
    EXPECT_GT(mesh.nodes.size(), outer.size() + inner.size());
    EXPECT_EQ(std::count_if(mesh.boundary_edges.begin(), mesh.boundary_edges.end(),
        [](const auto& edge) { return edge.batch_name == "outer-wall"; }), 32);
    EXPECT_EQ(std::count_if(mesh.boundary_edges.begin(), mesh.boundary_edges.end(),
        [](const auto& edge) { return edge.batch_name == "inner-wall"; }), 12);
}

/** @brief Long fixed inner edges remain constrained despite nearby front points. */
TEST(FrontalDelaunay2DTest, RecoversNonCircularHoleSegments)
{
    const SimpleFluid::Arr<Vec3> outer{
        {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0},
        {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0}};
    const SimpleFluid::Arr<Vec3> inner{
        {-0.4, -0.15, 0.0}, {0.6, -0.15, 0.0},
        {0.6, 0.25, 0.0}, {-0.4, 0.25, 0.0}};
    const auto mesh = Mesher::triangulate_annulus(outer, inner, 0.21,
                                                 "outer-wall", "inner-wall");
    expect_complete_annular_triangulation(mesh, outer, inner);
    EXPECT_GT(mesh.nodes.size(), outer.size() + inner.size());
}

/** @brief Reject invalid loop topology and sizing before placing any nodes. */
TEST(FrontalDelaunay2DTest, RejectsInvalidFixedAnnularLoops)
{
    const auto outer = regular_loop(12, 1.0);
    const auto inner = regular_loop(8, 0.3);
    auto clockwise = inner;
    std::reverse(clockwise.begin(), clockwise.end());
    EXPECT_THROW(Mesher::triangulate_annulus(outer, clockwise, 0.2), std::invalid_argument);
    auto closed = inner;
    closed.push_back(closed.front());
    EXPECT_THROW(Mesher::triangulate_annulus(outer, closed, 0.2), std::invalid_argument);
    EXPECT_THROW(Mesher::triangulate_annulus(inner, outer, 0.2), std::invalid_argument);
    EXPECT_THROW(Mesher::triangulate_annulus(outer, outer, 0.2), std::invalid_argument);
    EXPECT_THROW(Mesher::triangulate_annulus(outer, inner, 0.0), std::invalid_argument);
    EXPECT_THROW(Mesher::triangulate_annulus(outer, inner, 0.2, ""), std::invalid_argument);
    EXPECT_THROW(Mesher::triangulate_annulus(outer, inner, 0.2, "outer", ""), std::invalid_argument);
    EXPECT_THROW(Mesher::triangulate_annulus(outer, inner, 1.0e-100), std::overflow_error);
    auto nonfinite = inner;
    nonfinite[0].x = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(Mesher::triangulate_annulus(outer, nonfinite, 0.2), std::invalid_argument);
}
