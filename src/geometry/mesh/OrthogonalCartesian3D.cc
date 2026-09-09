/**
 * @file OrthogonalCartesian3D.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief OrthogonalCartesian3D implementation.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/OrthogonalCartesian3D.hh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace SimpleFluid::Meshes
{
namespace
{

/**
 * @brief Validate one Cartesian coordinate-edge array.
 * @param edges Edge coordinates to validate.
 * @param axis_name Axis label used in diagnostics.
 * @throws std::invalid_argument If the array is too short, non-finite, or not
 *         strictly increasing.
 * @throws std::overflow_error If its cell count exceeds the mesh ID type.
 */
void validate_edges(const Arr<real_t>& edges, const char* axis_name)
{
    if (edges.size() < 2)
    {
        throw std::invalid_argument(
            std::string("OrthogonalCartesian3D axis ") + axis_name
            + " requires at least two edge coordinates.");
    }
    if (edges.size() - 1
        > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            std::string("OrthogonalCartesian3D axis ") + axis_name
            + " has too many cells for its ID type.");
    }

    for (size_t edge = 0; edge < edges.size(); ++edge)
    {
        if (!std::isfinite(edges[edge]))
        {
            throw std::invalid_argument(
                std::string("OrthogonalCartesian3D axis ") + axis_name
                + " contains a non-finite edge coordinate.");
        }
        if (edge > 0 && edges[edge] <= edges[edge - 1])
        {
            throw std::invalid_argument(
                std::string("OrthogonalCartesian3D axis ") + axis_name
                + " edge coordinates must be strictly increasing.");
        }
    }
}

/**
 * @brief Narrow a size-based coordinate count to the mesh index type.
 * @param index Value to convert.
 * @return Converted non-negative index.
 * @throws std::overflow_error If @p index exceeds the integer ID type.
 */
int checked_index(size_t index)
{
    if (index > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            "OrthogonalCartesian3D local index exceeds its ID type.");
    }
    return static_cast<int>(index);
}

} // namespace

/**
 * @brief Construct and precompute Cartesian geometry and topology.
 * @param cell_edges Strictly increasing X, Y, and Z edge coordinates.
 * @throws std::invalid_argument If any coordinate array is invalid.
 * @throws std::overflow_error If entity counts exceed supported ID ranges.
 */
OrthogonalCartesian3D::OrthogonalCartesian3D(
    const Vec3D<Arr<real_t>>& cell_edges)
    : d_cell_edges(cell_edges)
{
    validate_edges(d_cell_edges[X], "x");
    validate_edges(d_cell_edges[Y], "y");
    validate_edges(d_cell_edges[Z], "z");

    const auto nx = d_cell_edges[X].size() - 1;
    const auto ny = d_cell_edges[Y].size() - 1;
    const auto nz = d_cell_edges[Z].size() - 1;
    CHECK_PRODUCT_OVERFLOW(nx + 1, ny + 1, nz + 1);

    d_indexer = Indexer(checked_index(nx), checked_index(ny), checked_index(nz));

    for (size_t dim = 0; dim < 3; ++dim)
    {
        d_cell_centroids[dim] = MeshUtils::consec_mid(d_cell_edges[dim]);
        d_cell_widths[dim] = MeshUtils::consec_diff(d_cell_edges[dim]);
    }

    Base::d_num_cells = nx * ny * nz;

    Base::d_num_local_cells = Base::d_num_cells;
    Base::d_num_owned_cells = Base::d_num_cells;

    const auto x_faces = (nx + 1) * ny * nz;
    const auto y_faces = nx * (ny + 1) * nz;
    const auto z_faces = nx * ny * (nz + 1);

    CHECK_SUM_OVERFLOW(x_faces, y_faces, z_faces);

    Base::d_num_faces = x_faces + y_faces + z_faces;
    Base::d_num_owned_faces = Base::d_num_faces;

    Base::d_num_nodes = (nx + 1) * (ny + 1) * (nz + 1);

    d_topology = OrthoMeshTopo(
        d_indexer,
        {{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}});
}

/**
 * @brief Replace one coordinate direction without changing structured topology.
 * @param axis Cartesian axis index.
 * @param edges Strictly increasing replacement edges with the existing size.
 * @throws std::invalid_argument If the axis or replacement layout is invalid.
 */
void OrthogonalCartesian3D::replace_axis_edges_fixed_topology(size_t axis, Arr<real_t> edges)
{
    if (d_geometry_state.epoch == std::numeric_limits<std::uint64_t>::max())
    {
        throw std::overflow_error("OrthogonalCartesian3D geometry epoch overflow.");
    }
    if (axis >= d_cell_edges.size())
    {
        throw std::invalid_argument("OrthogonalCartesian3D fixed-topology update axis is invalid.");
    }
    if (edges.size() != d_cell_edges[axis].size())
    {
        throw std::invalid_argument("OrthogonalCartesian3D fixed-topology update cannot change "
                                    "the edge count.");
    }

    constexpr std::array<const char*, 3> axis_names{{"x", "y", "z"}};
    validate_edges(edges, axis_names[axis]);
    auto centroids = MeshUtils::consec_mid(edges);
    auto widths = MeshUtils::consec_diff(edges);

    d_cell_edges[axis] = std::move(edges);
    d_cell_centroids[axis] = std::move(centroids);
    d_cell_widths[axis] = std::move(widths);
    ++d_geometry_state.epoch;
}

/**
 * @brief Return a physical boundary batch name.
 * @param batch_id Boundary batch identifier.
 * @return Configured lower or upper axis name.
 * @throws std::out_of_range If @p batch_id is invalid.
 */
const std::string&
OrthogonalCartesian3D::boundary_batch_name_impl(int batch_id) const
{
    return d_topology.boundary_batch_name(batch_id);
}

} // namespace SimpleFluid::Meshes
