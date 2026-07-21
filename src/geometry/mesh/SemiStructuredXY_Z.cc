/**
 * @file SemiStructuredXY_Z.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief SemiStructuredXY_Z setup and XY topology construction.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "geometry/mesh/FrontalDelaunay2D.hh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes
{
namespace
{

/**
 * @brief Validate that an entity count fits the mesh ID type.
 * @param count Entity count to validate.
 * @param quantity Entity label used in diagnostics.
 * @throws std::overflow_error If @p count exceeds the unsigned ID range.
 */
void validate_count(size_t count, const char* quantity)
{
    if (count > static_cast<size_t>(
                    std::numeric_limits<unsigned>::max()))
    {
        throw std::overflow_error(
            std::string("SemiStructuredXY_Z ") + quantity
            + " count exceeds its ID type.");
    }
}

/**
 * @brief Validate axial extrusion edge coordinates.
 * @param edges Z coordinates to validate.
 * @throws std::invalid_argument If fewer than two edges are supplied or they
 *         are non-finite or not strictly increasing.
 * @throws std::overflow_error If the edge count exceeds the ID type.
 */
void validate_z_edges(const Arr<real_t>& edges)
{
    if (edges.size() < 2)
    {
        throw std::invalid_argument(
            "SemiStructuredXY_Z requires at least two Z edge coordinates.");
    }
    validate_count(edges.size(), "Z edge");

    for (size_t edge = 0; edge < edges.size(); ++edge)
    {
        if (!std::isfinite(edges[edge]))
        {
            throw std::invalid_argument(
                "SemiStructuredXY_Z contains a non-finite Z coordinate.");
        }
        if (edge > 0 && edges[edge] <= edges[edge - 1])
        {
            throw std::invalid_argument(
                "SemiStructuredXY_Z Z coordinates must be strictly increasing.");
        }
    }
}

} // namespace

/**
 * @brief Construct an extruded polygonal finite-volume mesh.
 * @param xy_nodes Base-topology XY node coordinates.
 * @param xy_cell_nodes Counter-clockwise node loops for base cells.
 * @param z_edges Strictly increasing axial edge coordinates.
 * @param boundary_edges Named exterior edges in the base topology.
 * @throws std::invalid_argument If geometry or topology is invalid.
 * @throws std::overflow_error If entity counts exceed supported ID ranges.
 */
SemiStructuredXY_Z::SemiStructuredXY_Z(
    const Arr<Vec3>& xy_nodes,
    const Arr<Arr<unsigned>>& xy_cell_nodes,
    const Arr<real_t>& z_edges,
    const Arr<BoundaryEdge>& boundary_edges)
    : d_xy_nodes(xy_nodes),
      d_xy_cell_nodes(xy_cell_nodes),
      d_z_edges(z_edges)
{
    if (d_xy_nodes.empty())
    {
        throw std::invalid_argument(
            "SemiStructuredXY_Z requires at least one XY node.");
    }
    if (d_xy_cell_nodes.empty())
    {
        throw std::invalid_argument(
            "SemiStructuredXY_Z requires at least one XY cell.");
    }
    validate_count(d_xy_nodes.size(), "XY node");
    validate_count(d_xy_cell_nodes.size(), "XY cell");
    validate_z_edges(d_z_edges);

    for (const auto& node : d_xy_nodes)
    {
        if (!std::isfinite(node.x) || !std::isfinite(node.y))
        {
            throw std::invalid_argument(
                "SemiStructuredXY_Z contains a non-finite XY coordinate.");
        }
    }

    d_z_widths = MeshUtils::consec_diff(d_z_edges);
    d_z_midpoints = MeshUtils::consec_mid(d_z_edges);

    d_topology = Topology(
        static_cast<unsigned>(d_xy_nodes.size()),
        d_xy_cell_nodes,
        static_cast<unsigned>(d_z_widths.size()),
        boundary_edges);
    initialize_xy_geometry();

    const auto& indexer = d_topology.indexer();
    Base::d_num_cells = indexer.total_cells();
    Base::d_num_local_cells = Base::d_num_cells;
    Base::d_num_owned_cells = Base::d_num_cells;
    Base::d_num_faces = indexer.total_faces();
    Base::d_num_owned_faces = Base::d_num_faces;
    Base::d_num_nodes = indexer.total_nodes();
}

/**
 * @brief Triangulate a convex XY polygon and extrude it through Z.
 * @param xy_boundary Convex counter-clockwise polygon vertices.
 * @param z_edges Strictly increasing axial edge coordinates.
 * @param target_edge_length Requested XY mesh spacing.
 * @param side_batch_name Boundary name assigned to polygon sides.
 * @return Constructed triangular-prism mesh.
 * @throws std::invalid_argument If source geometry or spacing is invalid.
 * @throws std::overflow_error If generated connectivity exceeds its ID type.
 * @throws std::runtime_error If triangulation invariants cannot be satisfied.
 */
SemiStructuredXY_Z SemiStructuredXY_Z::from_frontal_delaunay(
    const Arr<Vec3>& xy_boundary,
    const Arr<real_t>& z_edges,
    real_t target_edge_length,
    const std::string& side_batch_name)
{
    auto xy_mesh = FrontalDelaunay2D::triangulate(
        xy_boundary, target_edge_length, side_batch_name);

    Arr<Arr<unsigned>> cells;
    cells.reserve(xy_mesh.triangles.size());
    for (const auto& triangle : xy_mesh.triangles)
    {
        cells.push_back({triangle[0], triangle[1], triangle[2]});
    }

    Arr<BoundaryEdge> boundary_edges;
    boundary_edges.reserve(xy_mesh.boundary_edges.size());
    for (auto& edge : xy_mesh.boundary_edges)
    {
        boundary_edges.push_back(
            {edge.node0, edge.node1, std::move(edge.batch_name)});
    }

    return SemiStructuredXY_Z(
        xy_mesh.nodes, cells, z_edges, boundary_edges);
}

/**
 * @brief Compute base-cell and base-edge geometry from XY connectivity.
 * @throws std::invalid_argument If a cell is not counter-clockwise and
 *         non-degenerate or an edge has non-positive length.
 */
void SemiStructuredXY_Z::initialize_xy_geometry()
{
    d_xy_cell_areas.resize(d_xy_cell_nodes.size());
    d_xy_cell_centroids.resize(d_xy_cell_nodes.size());

    for (unsigned cell = 0; cell < d_xy_cell_nodes.size(); ++cell)
    {
        const auto& nodes = d_xy_cell_nodes[cell];
        real_t twice_area = 0.0;
        real_t centroid_x_numerator = 0.0;
        real_t centroid_y_numerator = 0.0;

        for (size_t side = 0; side < nodes.size(); ++side)
        {
            const auto node0 = nodes[side];
            const auto node1 = nodes[(side + 1) % nodes.size()];
            const auto& point0 = d_xy_nodes[node0];
            const auto& point1 = d_xy_nodes[node1];
            const auto cross =
                point0.x * point1.y - point1.x * point0.y;
            twice_area += cross;
            centroid_x_numerator +=
                (point0.x + point1.x) * cross;
            centroid_y_numerator +=
                (point0.y + point1.y) * cross;
        }

        if (!std::isfinite(twice_area) || twice_area <= 0.0)
        {
            throw std::invalid_argument(
                "SemiStructuredXY_Z XY cells must be non-degenerate and "
                "counter-clockwise.");
        }
        d_xy_cell_areas[cell] = 0.5 * twice_area;
        d_xy_cell_centroids[cell] = {
            centroid_x_numerator / (3.0 * twice_area),
            centroid_y_numerator / (3.0 * twice_area),
            0.0};
    }

    const auto& side_faces = d_topology.side_faces();
    d_xy_edge_lengths.resize(side_faces.size());
    d_xy_edge_centroids.resize(side_faces.size());
    d_xy_edge_normals.resize(side_faces.size());

    for (unsigned edge_id = 0;
         edge_id < side_faces.size();
         ++edge_id)
    {
        const auto& edge = side_faces[edge_id];
        const auto& point0 = d_xy_nodes[edge.nodes[0]];
        const auto& point1 = d_xy_nodes[edge.nodes[1]];
        const auto dx = point1.x - point0.x;
        const auto dy = point1.y - point0.y;
        const auto length = std::hypot(dx, dy);
        if (!(length > 0.0) || !std::isfinite(length))
        {
            throw std::invalid_argument(
                "SemiStructuredXY_Z XY edges must have positive length.");
        }

        d_xy_edge_lengths[edge_id] = length;
        d_xy_edge_centroids[edge_id] = {
            0.5 * (point0.x + point1.x),
            0.5 * (point0.y + point1.y),
            0.0};
        d_xy_edge_normals[edge_id] = {
            dy / length,
            -dx / length,
            0.0};
    }
}

} // namespace SimpleFluid::Meshes
