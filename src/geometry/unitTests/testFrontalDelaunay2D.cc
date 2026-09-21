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

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <unordered_set>

namespace
{

using Mesher = SimpleFluid::Meshes::FrontalDelaunay2D;
using Vec3 = Mesher::Vec3;

long double orient2d(const Vec3& a, const Vec3& b, const Vec3& c)
{
    return (static_cast<long double>(b.x) - a.x)
         * (static_cast<long double>(c.y) - a.y)
         - (static_cast<long double>(b.y) - a.y)
         * (static_cast<long double>(c.x) - a.x);
}

long double circumcircle_determinant(
    const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& point)
{
    const long double ax = static_cast<long double>(a.x) - point.x;
    const long double ay = static_cast<long double>(a.y) - point.y;
    const long double bx = static_cast<long double>(b.x) - point.x;
    const long double by = static_cast<long double>(b.y) - point.y;
    const long double cx = static_cast<long double>(c.x) - point.x;
    const long double cy = static_cast<long double>(c.y) - point.y;
    return (ax * ax + ay * ay) * (bx * cy - cx * by)
         - (bx * bx + by * by) * (ax * cy - cx * ay)
         + (cx * cx + cy * cy) * (ax * by - bx * ay);
}

void expect_complete_planar_triangulation(const Mesher::Result& mesh)
{
    ASSERT_GE(mesh.nodes.size(), 3U);
    ASSERT_GE(mesh.boundary_edges.size(), 3U);
    EXPECT_EQ(mesh.triangles.size(),
              2U * mesh.nodes.size() - 2U - mesh.boundary_edges.size());

    std::unordered_set<unsigned> used_nodes;
    std::map<std::pair<unsigned, unsigned>, unsigned> edge_counts;
    for (const auto& triangle : mesh.triangles)
    {
        EXPECT_GT(orient2d(mesh.nodes[triangle[0]],
                           mesh.nodes[triangle[1]],
                           mesh.nodes[triangle[2]]), 0.0L);
        used_nodes.insert(triangle[0]);
        used_nodes.insert(triangle[1]);
        used_nodes.insert(triangle[2]);
        for (unsigned side = 0; side < 3; ++side)
        {
            auto node0 = triangle[side];
            auto node1 = triangle[(side + 1U) % 3U];
            if (node1 < node0) std::swap(node0, node1);
            ++edge_counts[{node0, node1}];
        }

        for (unsigned point = 0; point < mesh.nodes.size(); ++point)
        {
            if (point == triangle[0]
                || point == triangle[1]
                || point == triangle[2])
            {
                continue;
            }
            EXPECT_LE(circumcircle_determinant(
                          mesh.nodes[triangle[0]], mesh.nodes[triangle[1]],
                          mesh.nodes[triangle[2]], mesh.nodes[point]),
                      1.0e-10L);
        }
    }
    EXPECT_EQ(used_nodes.size(), mesh.nodes.size());
    size_t exterior_edge_count = 0;
    size_t nonmanifold_edge_count = 0;
    for (const auto& [edge, count] : edge_counts)
    {
        (void)edge;
        exterior_edge_count += count == 1U;
        nonmanifold_edge_count += count > 2U;
    }
    EXPECT_EQ(exterior_edge_count, mesh.boundary_edges.size());
    EXPECT_EQ(nonmanifold_edge_count, 0U);
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
        EXPECT_GT(triangle_area, 0.0L);
        area += triangle_area;
        Vec3 centroid{};
        for (unsigned side = 0; side < 3; ++side)
        {
            const auto a = triangle[side];
            const auto b = triangle[(side + 1U) % 3U];
            ASSERT_LT(a, mesh.nodes.size());
            used.insert(a);
            adjacency[std::minmax(a, b)].push_back(t);
            centroid.x += mesh.nodes[a].x / 3.0;
            centroid.y += mesh.nodes[a].y / 3.0;
        }
        bool outside_hole = false;
        for (size_t edge = 0; edge < inner.size(); ++edge)
            outside_hole = outside_hole
                || orient2d(inner[edge], inner[(edge + 1) % inner.size()], centroid) < 0.0L;
        EXPECT_TRUE(outside_hole);
    }
    EXPECT_EQ(used.size(), mesh.nodes.size());
    EXPECT_NEAR(area, polygon_twice_area(outer) - polygon_twice_area(inner), 1.0e-12L);
    size_t exterior_edges = 0;
    for (const auto& [edge, adjacent] : adjacency)
    {
        EXPECT_LE(adjacent.size(), 2U);
        exterior_edges += adjacent.size() == 1;
        if (adjacent.size() != 2) continue;
        const auto& first = mesh.triangles[adjacent[0]];
        const auto& second = mesh.triangles[adjacent[1]];
        for (const auto opposite : second)
        {
            if (opposite == edge.first || opposite == edge.second) continue;
            // Every retained interior edge is unconstrained and locally Delaunay.
            EXPECT_LE(circumcircle_determinant(mesh.nodes[first[0]],
                mesh.nodes[first[1]], mesh.nodes[first[2]], mesh.nodes[opposite]),
                1.0e-12L);
        }
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
