/**
 * @file OrthogonalCartesian3D.cc
 * @brief OrthogonalCartesian3D implementation.
 */

#include "geometry/mesh/OrthogonalCartesian3D.hh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace SimpleFluid::Mesh
{
namespace
{

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

const std::string&
OrthogonalCartesian3D::boundary_patch_name_impl(int patch_id) const
{
    return d_topology.boundary_patch_name(patch_id);
}

const OrthogonalCartesian3D::BoundaryFacePatch&
OrthogonalCartesian3D::boundary_face_patch_impl(int patch_id) const
{
    return d_topology.boundary_face_patch(patch_id);
}

} // namespace SimpleFluid
