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

#include <cmath>
#include <map>
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
