/**
 * @file FrontalDelaunay2D.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Frontal-Delaunay XY mesh generation implementation.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/FrontalDelaunay2D.hh"

#include <algorithm>
#include <cmath>
#include <compare>
#include <limits>
#include <map>
#include <numbers>
#include <queue>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes
{
namespace
{

using Vec3 = FrontalDelaunay2D::Vec3;
using Triangle = FrontalDelaunay2D::Triangle;

/** @brief Canonically ordered node pair used as an undirected edge key. */
struct Edge
{
    unsigned node0{};
    unsigned node1{};

    friend auto operator<=>(const Edge&, const Edge&) = default;
};

/**
 * @brief Order two node IDs into a canonical undirected edge.
 * @param node0 First node ID.
 * @param node1 Second node ID.
 * @return Edge with its smaller node ID first.
 */
Edge normalized_edge(unsigned node0, unsigned node1)
{
    return node0 < node1 ? Edge{node0, node1} : Edge{node1, node0};
}

/**
 * @brief Evaluate the signed twice-area orientation predicate in XY.
 * @param a First point.
 * @param b Second point.
 * @param c Third point.
 * @return Positive for counter-clockwise order and negative for clockwise.
 */
long double orient2d(const Vec3& a, const Vec3& b, const Vec3& c)
{
    return (static_cast<long double>(b.x) - a.x)
         * (static_cast<long double>(c.y) - a.y)
         - (static_cast<long double>(b.y) - a.y)
         * (static_cast<long double>(c.x) - a.x);
}

/**
 * @brief Compute squared XY distance between two points.
 * @param a First point.
 * @param b Second point.
 * @return Squared planar distance.
 */
real_t squared_distance(const Vec3& a, const Vec3& b)
{
    const auto dx = a.x - b.x;
    const auto dy = a.y - b.y;
    return dx * dx + dy * dy;
}

/**
 * @brief Test whether both planar coordinates are finite.
 * @param point Point to inspect.
 * @return True when x and y are finite.
 */
bool finite_xy(const Vec3& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y);
}

/**
 * @brief Test whether a point lies inside or on a convex CCW polygon.
 * @param polygon Convex counter-clockwise boundary vertices.
 * @param point Point to classify.
 * @return True when the point lies inside or on the boundary.
 */
bool inside_convex_polygon(const Arr<Vec3>& polygon, const Vec3& point)
{
    const auto scale = std::max<real_t>(
        1.0, std::max(std::abs(point.x), std::abs(point.y)));
    const auto tolerance = 128.0L
                         * std::numeric_limits<real_t>::epsilon()
                         * scale * scale;
    for (size_t edge = 0; edge < polygon.size(); ++edge)
    {
        if (orient2d(polygon[edge],
                     polygon[(edge + 1) % polygon.size()], point)
            < -tolerance)
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Return true when p is strictly inside a CCW triangle circumcircle.
 * @param a First triangle vertex.
 * @param b Second triangle vertex.
 * @param c Third triangle vertex.
 * @param point Point to classify.
 * @return True when @p point is strictly inside the circumcircle.
 */
bool in_circumcircle(const Vec3& a,
                     const Vec3& b,
                     const Vec3& c,
                     const Vec3& point)
{
    const long double ax = static_cast<long double>(a.x) - point.x;
    const long double ay = static_cast<long double>(a.y) - point.y;
    const long double bx = static_cast<long double>(b.x) - point.x;
    const long double by = static_cast<long double>(b.y) - point.y;
    const long double cx = static_cast<long double>(c.x) - point.x;
    const long double cy = static_cast<long double>(c.y) - point.y;

    const long double determinant =
        (ax * ax + ay * ay) * (bx * cy - cx * by)
      - (bx * bx + by * by) * (ax * cy - cx * ay)
      + (cx * cx + cy * cy) * (ax * by - bx * ay);

    const auto orientation = orient2d(a, b, c);
    const long double coordinate_scale =
        std::max({1.0L, std::abs(ax), std::abs(ay), std::abs(bx),
                  std::abs(by), std::abs(cx), std::abs(cy)});
    const auto tolerance = 256.0L
                         * std::numeric_limits<long double>::epsilon()
                         * coordinate_scale * coordinate_scale
                         * coordinate_scale * coordinate_scale;
    return orientation > 0.0L
         ? determinant > tolerance
         : determinant < -tolerance;
}

/**
 * @brief Test whether a point lies inside or on a counter-clockwise triangle.
 * @param a First triangle vertex.
 * @param b Second triangle vertex.
 * @param c Third triangle vertex.
 * @param point Point to classify.
 * @return True when @p point is inside or on the triangle.
 */
bool in_triangle(const Vec3& a,
                 const Vec3& b,
                 const Vec3& c,
                 const Vec3& point)
{
    const long double scale = std::max(
        {1.0L, std::abs(static_cast<long double>(a.x)),
         std::abs(static_cast<long double>(a.y)),
         std::abs(static_cast<long double>(b.x)),
         std::abs(static_cast<long double>(b.y)),
         std::abs(static_cast<long double>(c.x)),
         std::abs(static_cast<long double>(c.y))});
    const auto tolerance = 256.0L
                         * std::numeric_limits<real_t>::epsilon()
                         * scale * scale;
    return orient2d(a, b, point) >= -tolerance
        && orient2d(b, c, point) >= -tolerance
        && orient2d(c, a, point) >= -tolerance;
}

/**
 * @brief Bowyer-Watson triangulation of a unique planar point set.
 * @param input_nodes Unique planar points to triangulate.
 * @return Counter-clockwise triangle connectivity over @p input_nodes.
 * @throws std::invalid_argument If fewer than three points are supplied or
 *         their planar extent is not positive and finite.
 * @throws std::overflow_error If node IDs exceed the connectivity type.
 * @throws std::runtime_error If a point cannot be inserted or no triangle is
 *         produced.
 */
Arr<Triangle> delaunay_triangulate(const Arr<Vec3>& input_nodes)
{
    if (input_nodes.size() < 3)
    {
        throw std::invalid_argument(
            "FrontalDelaunay2D requires at least three distinct points.");
    }
    if (input_nodes.size()
        > static_cast<size_t>(std::numeric_limits<unsigned>::max() - 3U))
    {
        throw std::overflow_error(
            "FrontalDelaunay2D node count exceeds its connectivity type.");
    }

    real_t xmin = input_nodes.front().x;
    real_t xmax = xmin;
    real_t ymin = input_nodes.front().y;
    real_t ymax = ymin;
    for (const auto& point : input_nodes)
    {
        xmin = std::min(xmin, point.x);
        xmax = std::max(xmax, point.x);
        ymin = std::min(ymin, point.y);
        ymax = std::max(ymax, point.y);
    }

    const auto extent = std::max(xmax - xmin, ymax - ymin);
    if (!(extent > 0.0) || !std::isfinite(extent))
    {
        throw std::invalid_argument(
            "FrontalDelaunay2D point extent must be positive and finite.");
    }

    Arr<Vec3> nodes = input_nodes;
    const auto center_x = 0.5 * (xmin + xmax);
    const auto center_y = 0.5 * (ymin + ymax);
    const auto super0 = static_cast<unsigned>(nodes.size());
    nodes.push_back({center_x - 32.0 * extent,
                     center_y - 16.0 * extent, 0.0});
    const auto super1 = static_cast<unsigned>(nodes.size());
    nodes.push_back({center_x + 32.0 * extent,
                     center_y - 16.0 * extent, 0.0});
    const auto super2 = static_cast<unsigned>(nodes.size());
    nodes.push_back({center_x, center_y + 32.0 * extent, 0.0});

    Arr<Triangle> triangles{{super0, super1, super2}};
    for (unsigned point_id = 0; point_id < input_nodes.size(); ++point_id)
    {
        Arr<bool> remove(triangles.size(), false);
        std::map<Edge, unsigned> cavity_edges;
        for (size_t triangle_id = 0;
             triangle_id < triangles.size(); ++triangle_id)
        {
            const auto& triangle = triangles[triangle_id];
            if (!in_circumcircle(nodes[triangle[0]], nodes[triangle[1]],
                                 nodes[triangle[2]], nodes[point_id]))
            {
                continue;
            }

            remove[triangle_id] = true;
            ++cavity_edges[normalized_edge(triangle[0], triangle[1])];
            ++cavity_edges[normalized_edge(triangle[1], triangle[2])];
            ++cavity_edges[normalized_edge(triangle[2], triangle[0])];
        }

        // A point can lie exactly on an existing edge when circular fronts
        // are co-circular.  Include every containing triangle even when the
        // circumcircle test selected only one side of that edge; otherwise a
        // hanging node and an untriangulated half-edge can remain.
        for (size_t triangle_id = 0;
             triangle_id < triangles.size(); ++triangle_id)
        {
            if (remove[triangle_id])
            {
                continue;
            }
            const auto& triangle = triangles[triangle_id];
            if (!in_triangle(nodes[triangle[0]], nodes[triangle[1]],
                             nodes[triangle[2]], nodes[point_id]))
            {
                continue;
            }
            remove[triangle_id] = true;
            ++cavity_edges[normalized_edge(triangle[0], triangle[1])];
            ++cavity_edges[normalized_edge(triangle[1], triangle[2])];
            ++cavity_edges[normalized_edge(triangle[2], triangle[0])];
        }
        if (cavity_edges.empty())
        {
            throw std::runtime_error(
                "FrontalDelaunay2D could not insert a generated point.");
        }

        Arr<Triangle> next;
        next.reserve(triangles.size() + cavity_edges.size());
        for (size_t triangle_id = 0;
             triangle_id < triangles.size(); ++triangle_id)
        {
            if (!remove[triangle_id])
            {
                next.push_back(triangles[triangle_id]);
            }
        }
        for (const auto& [edge, count] : cavity_edges)
        {
            if (count != 1)
            {
                continue;
            }
            Triangle triangle{edge.node0, edge.node1, point_id};
            if (orient2d(nodes[triangle[0]], nodes[triangle[1]],
                         nodes[triangle[2]]) < 0.0L)
            {
                std::swap(triangle[0], triangle[1]);
            }
            next.push_back(triangle);
        }
        triangles = std::move(next);
    }

    Arr<Triangle> result;
    result.reserve(triangles.size());
    for (auto triangle : triangles)
    {
        if (triangle[0] >= input_nodes.size()
            || triangle[1] >= input_nodes.size()
            || triangle[2] >= input_nodes.size())
        {
            continue;
        }
        if (orient2d(nodes[triangle[0]], nodes[triangle[1]],
                     nodes[triangle[2]]) <= 0.0L)
        {
            continue;
        }
        result.push_back(triangle);
    }
    if (result.empty())
    {
        throw std::runtime_error(
            "FrontalDelaunay2D failed to construct any triangles.");
    }
    return result;
}

/**
 * @brief Require a non-empty boundary batch name.
 * @param boundary_name Name to validate.
 * @throws std::invalid_argument If @p boundary_name is empty.
 */
void validate_boundary_name(const std::string& boundary_name)
{
    if (boundary_name.empty())
    {
        throw std::invalid_argument(
            "FrontalDelaunay2D boundary name cannot be empty.");
    }
}

/**
 * @brief Require a positive finite target edge length.
 * @param target_edge_length Requested spacing to validate.
 * @throws std::invalid_argument If the spacing is not positive and finite.
 */
void validate_target_length(real_t target_edge_length)
{
    if (!(target_edge_length > 0.0)
        || !std::isfinite(target_edge_length))
    {
        throw std::invalid_argument(
            "FrontalDelaunay2D target edge length must be positive and finite.");
    }
}

/**
 * @brief Validate boundary preservation and edge-manifold invariants.
 * @param mesh Generated mesh to inspect.
 * @throws std::runtime_error If nodes are orphaned, boundary edges are lost,
 *         or the triangulation contains holes or non-manifold edges.
 */
void verify_boundary_edges(const FrontalDelaunay2D::Result& mesh)
{
    std::map<Edge, unsigned> mesh_edges;
    ArrBool used_nodes(mesh.nodes.size(), false);
    for (const auto& triangle : mesh.triangles)
    {
        used_nodes[triangle[0]] = true;
        used_nodes[triangle[1]] = true;
        used_nodes[triangle[2]] = true;
        ++mesh_edges[normalized_edge(triangle[0], triangle[1])];
        ++mesh_edges[normalized_edge(triangle[1], triangle[2])];
        ++mesh_edges[normalized_edge(triangle[2], triangle[0])];
    }

    if (std::find(used_nodes.begin(), used_nodes.end(), false)
        != used_nodes.end())
    {
        throw std::runtime_error(
            "FrontalDelaunay2D generated an orphan node.");
    }

    std::map<Edge, bool> declared_boundary_edges;
    for (const auto& boundary : mesh.boundary_edges)
    {
        const auto edge = normalized_edge(boundary.node0, boundary.node1);
        declared_boundary_edges.emplace(edge, true);
        const auto iter = mesh_edges.find(edge);
        if (iter == mesh_edges.end() || iter->second != 1)
        {
            throw std::runtime_error(
                "FrontalDelaunay2D failed to preserve a boundary edge.");
        }
    }

    for (const auto& [edge, count] : mesh_edges)
    {
        if (count > 2)
        {
            throw std::runtime_error(
                "FrontalDelaunay2D generated a non-manifold edge.");
        }
        if (count == 1 && !declared_boundary_edges.contains(edge))
        {
            throw std::runtime_error(
                "FrontalDelaunay2D generated an interior hole or hanging edge.");
        }
    }
}

} // namespace

FrontalDelaunay2D::Result FrontalDelaunay2D::triangulate(
    const Arr<Vec3>& supplied_boundary,
    real_t target_edge_length,
    const std::string& boundary_name)
{
    validate_target_length(target_edge_length);
    validate_boundary_name(boundary_name);

    Arr<Vec3> boundary = supplied_boundary;
    if (boundary.size() > 1
        && squared_distance(boundary.front(), boundary.back()) == 0.0)
    {
        boundary.pop_back();
    }
    if (boundary.size() < 3)
    {
        throw std::invalid_argument(
            "FrontalDelaunay2D polygon requires at least three vertices.");
    }
    for (auto& point : boundary)
    {
        if (!finite_xy(point))
        {
            throw std::invalid_argument(
                "FrontalDelaunay2D polygon contains a non-finite coordinate.");
        }
        point.z = 0.0;
    }

    long double twice_area = 0.0L;
    bool saw_strict_corner = false;
    long double coordinate_scale = 1.0L;
    for (const auto& point : boundary)
    {
        coordinate_scale = std::max(
            {coordinate_scale,
             std::abs(static_cast<long double>(point.x)),
             std::abs(static_cast<long double>(point.y))});
    }
    const auto convexity_tolerance = 128.0L
                                     * std::numeric_limits<real_t>::epsilon()
                                     * coordinate_scale * coordinate_scale;
    for (size_t vertex = 0; vertex < boundary.size(); ++vertex)
    {
        const auto& current = boundary[vertex];
        const auto& next = boundary[(vertex + 1) % boundary.size()];
        const auto length_squared = squared_distance(current, next);
        if (!(length_squared > 0.0))
        {
            throw std::invalid_argument(
                "FrontalDelaunay2D polygon contains a zero-length edge.");
        }
        twice_area += static_cast<long double>(current.x) * next.y
                    - static_cast<long double>(current.y) * next.x;

        const auto turn = orient2d(
            current, next, boundary[(vertex + 2) % boundary.size()]);
        if (turn < -convexity_tolerance)
        {
            throw std::invalid_argument(
                "FrontalDelaunay2D currently requires a convex polygon.");
        }
        saw_strict_corner = saw_strict_corner || turn > 0.0L;
    }
    for (size_t edge = 0; edge < boundary.size(); ++edge)
    {
        for (const auto& point : boundary)
        {
            if (orient2d(boundary[edge],
                         boundary[(edge + 1) % boundary.size()], point)
                < -convexity_tolerance)
            {
                throw std::invalid_argument(
                    "FrontalDelaunay2D currently requires a convex polygon.");
            }
        }
    }
    if (!(twice_area > 0.0L) || !saw_strict_corner)
    {
        throw std::invalid_argument(
            "FrontalDelaunay2D polygon must be counter-clockwise and non-degenerate.");
    }

    Result result;
    for (size_t edge = 0; edge < boundary.size(); ++edge)
    {
        const auto& point0 = boundary[edge];
        const auto& point1 = boundary[(edge + 1) % boundary.size()];
        const auto length = std::sqrt(squared_distance(point0, point1));
        const auto segments = std::max<size_t>(
            1, static_cast<size_t>(std::ceil(length / target_edge_length)));
        for (size_t segment = 0; segment < segments; ++segment)
        {
            const auto fraction = static_cast<real_t>(segment)
                                / static_cast<real_t>(segments);
            result.nodes.push_back({
                point0.x + fraction * (point1.x - point0.x),
                point0.y + fraction * (point1.y - point0.y), 0.0});
        }
    }

    const auto boundary_node_count = static_cast<unsigned>(result.nodes.size());
    result.boundary_edges.reserve(boundary_node_count);
    for (unsigned node = 0; node < boundary_node_count; ++node)
    {
        result.boundary_edges.push_back(
            {node, (node + 1U) % boundary_node_count, boundary_name});
    }

    // Seed one inward equilateral point per boundary segment, then advance
    // accepted points in approximately hexagonal fronts.
    std::queue<unsigned> active_front;
    const auto minimum_spacing = 0.58 * target_edge_length;
    const auto minimum_spacing_squared = minimum_spacing * minimum_spacing;
    auto try_insert = [&](const Vec3& candidate) -> bool
    {
        if (!inside_convex_polygon(boundary, candidate))
        {
            return false;
        }
        for (const auto& node : result.nodes)
        {
            if (squared_distance(candidate, node)
                < minimum_spacing_squared)
            {
                return false;
            }
        }
        result.nodes.push_back(candidate);
        active_front.push(static_cast<unsigned>(result.nodes.size() - 1));
        return true;
    };

    for (unsigned node = 0; node < boundary_node_count; ++node)
    {
        const auto& point0 = result.nodes[node];
        const auto& point1 = result.nodes[(node + 1U) % boundary_node_count];
        const auto dx = point1.x - point0.x;
        const auto dy = point1.y - point0.y;
        try_insert({0.5 * (point0.x + point1.x)
                        - 0.5 * std::sqrt(3.0) * dy,
                    0.5 * (point0.y + point1.y)
                        + 0.5 * std::sqrt(3.0) * dx,
                    0.0});
    }

    const auto maximum_nodes = std::max<size_t>(
        boundary_node_count + 1U,
        boundary_node_count
            + 16U * static_cast<size_t>(
                std::ceil(static_cast<double>(twice_area)
                        / (target_edge_length * target_edge_length))));
    while (!active_front.empty() && result.nodes.size() < maximum_nodes)
    {
        const auto point = result.nodes[active_front.front()];
        active_front.pop();
        for (unsigned direction = 0; direction < 6; ++direction)
        {
            const auto angle = std::numbers::pi_v<real_t>
                             * static_cast<real_t>(direction) / 3.0;
            try_insert({point.x + target_edge_length * std::cos(angle),
                        point.y + target_edge_length * std::sin(angle),
                        0.0});
        }
    }

    result.triangles = delaunay_triangulate(result.nodes);
    verify_boundary_edges(result);
    return result;
}

FrontalDelaunay2D::Result FrontalDelaunay2D::triangulate_disk(
    const ArrReal& radial_edges,
    real_t target_edge_length,
    const std::string& boundary_name)
{
    validate_target_length(target_edge_length);
    validate_boundary_name(boundary_name);
    if (radial_edges.size() < 2 || radial_edges.front() != 0.0)
    {
        throw std::invalid_argument(
            "FrontalDelaunay2D disk radii must start at zero and contain a positive radius.");
    }
    for (size_t radius_id = 0; radius_id < radial_edges.size(); ++radius_id)
    {
        if (!std::isfinite(radial_edges[radius_id])
            || (radius_id > 0
                && radial_edges[radius_id] <= radial_edges[radius_id - 1]))
        {
            throw std::invalid_argument(
                "FrontalDelaunay2D disk radii must be finite and strictly increasing.");
        }
    }

    Result result;
    result.nodes.push_back({0.0, 0.0, 0.0});
    unsigned outer_node_count = 0;
    unsigned outer_node_offset = 0;
    // Insert circular fronts from the center outward.  This ordering keeps
    // the center as an explicit vertex instead of placing it later on an
    // arbitrary diameter edge in a co-circular intermediate triangulation.
    // Alternating phase avoids aligned spokes between adjacent fronts.
    for (size_t ring = 1; ring < radial_edges.size(); ++ring)
    {
        const auto radius = radial_edges[ring];
        const auto node_count = std::max<size_t>(
            6, static_cast<size_t>(std::ceil(
                2.0 * std::numbers::pi_v<real_t> * radius
              / target_edge_length)));
        if (ring + 1 == radial_edges.size())
        {
            outer_node_count = static_cast<unsigned>(node_count);
            outer_node_offset = static_cast<unsigned>(result.nodes.size());
        }
        const auto phase = (ring % 2U)
                         * std::numbers::pi_v<real_t>
                         / static_cast<real_t>(node_count);
        for (size_t node = 0; node < node_count; ++node)
        {
            const auto angle = phase
                             + 2.0 * std::numbers::pi_v<real_t>
                             * static_cast<real_t>(node)
                             / static_cast<real_t>(node_count);
            result.nodes.push_back(
                {radius * std::cos(angle), radius * std::sin(angle), 0.0});
        }
    }

    result.boundary_edges.reserve(outer_node_count);
    for (unsigned node = 0; node < outer_node_count; ++node)
    {
        result.boundary_edges.push_back(
            {outer_node_offset + node,
             outer_node_offset + (node + 1U) % outer_node_count,
             boundary_name});
    }
    result.triangles = delaunay_triangulate(result.nodes);
    verify_boundary_edges(result);
    return result;
}

} // namespace SimpleFluid::Meshes
