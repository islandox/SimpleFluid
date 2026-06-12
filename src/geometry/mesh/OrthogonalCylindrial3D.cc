/**
 * @file OrthogonalCylindrial3D.cc
 * @brief OrthogonalCylindrial3D setup and indexing implementation.
 */

#include "geometry/mesh/OrthogonalCylindrial3D.hh"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace SimpleFluid::Meshes
{
namespace
{

void validate_edges(
    const Arr<real_t>& edges,
    const char* coordinate_name)
{
    if (edges.size() < 2)
    {
        throw std::invalid_argument(
            std::string("OrthogonalCylindrial3D coordinate ")
            + coordinate_name
            + " requires at least two edge coordinates.");
    }
    if (edges.size() - 1
        > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            std::string("OrthogonalCylindrial3D coordinate ")
            + coordinate_name
            + " has too many cells for its ID type.");
    }

    for (size_t edge = 0; edge < edges.size(); ++edge)
    {
        if (!std::isfinite(edges[edge]))
        {
            throw std::invalid_argument(
                std::string("OrthogonalCylindrial3D coordinate ")
                + coordinate_name
                + " contains a non-finite edge coordinate.");
        }
        if (edge > 0 && edges[edge] <= edges[edge - 1])
        {
            throw std::invalid_argument(
                std::string("OrthogonalCylindrial3D coordinate ")
                + coordinate_name
                + " edge coordinates must be strictly increasing.");
        }
    }
}

} // namespace

OrthogonalCylindrial3D::OrthogonalCylindrial3D(
    const Vec3D<Arr<real_t>>& cell_edges)
    : d_cell_edges(cell_edges)
{
    validate_edges(d_cell_edges[R], "r");
    validate_edges(d_cell_edges[THETA], "theta");
    validate_edges(d_cell_edges[AXIAL], "z");

    if (d_cell_edges[R].front() <= 0.0)
    {
        throw std::invalid_argument(
            "OrthogonalCylindrial3D requires a positive inner radius.");
    }

    const auto theta_span =
        d_cell_edges[THETA].back() - d_cell_edges[THETA].front();
    constexpr auto two_pi = 2.0 * std::numbers::pi_v<real_t>;
    constexpr auto angular_tolerance =
        64.0 * std::numeric_limits<real_t>::epsilon() * two_pi;
    if (theta_span > two_pi + angular_tolerance)
    {
        throw std::invalid_argument(
            "OrthogonalCylindrial3D theta span cannot exceed 2 pi.");
    }
    auto theta_periodic =
        std::abs(theta_span - two_pi) <= angular_tolerance;
    if (theta_periodic && d_cell_edges[THETA].size() < 3)
    {
        throw std::invalid_argument(
            "Periodic OrthogonalCylindrial3D requires at least "
            "two theta cells.");
    }

    const auto nr = d_cell_edges[R].size() - 1;
    const auto nt = d_cell_edges[THETA].size() - 1;
    const auto nz = d_cell_edges[AXIAL].size() - 1;
    CHECK_PRODUCT_OVERFLOW(nr + 1, nt + 1, nz + 1);

    d_indexer = Indexer(
        static_cast<unsigned>(nr),
        static_cast<unsigned>(nt),
        static_cast<unsigned>(nz),
        false,
        theta_periodic,
        false);

    for (size_t dim = 0; dim < 3; ++dim)
    {
        d_cell_midpoints[dim] = MeshUtils::consec_mid(d_cell_edges[dim]);
        d_cell_widths[dim] = MeshUtils::consec_diff(d_cell_edges[dim]);
    }

    d_cell_Dr2.reserve(d_indexer.num_cells_per_dim[R]);
    for (unsigned i = 0; i < d_indexer.num_cells_per_dim[R]; ++i)
    {
        auto r0 = d_cell_edges[R][i];
        auto r1 = d_cell_edges[R][i+1];
        d_cell_Dr2.push_back((r1 + r0) * (r1 - r0));
    }

    for (auto orientation = 0; orientation < 3; ++orientation)
    {
        for (auto dim = 0; dim < 3; ++dim)
        {
            auto& face_area_magnitudes = d_face_area_magnitudes[orientation][dim];
            auto n_face = d_indexer.num_faces_per_dim_per_orientation[orientation][dim];
            face_area_magnitudes.resize(n_face);
            for (unsigned i = 0; i < n_face; ++i)
            {
                if (orientation != dim)
                {
                    face_area_magnitudes[i] = d_cell_widths[dim][i];

                    if (orientation == Z_FACE && dim == R)
                        face_area_magnitudes[i] = 0.5 * d_cell_Dr2[i];
                }

                if (orientation == dim)
                {
                    face_area_magnitudes[i] = 1;

                    if (orientation == R_FACE && dim == R)
                        face_area_magnitudes[i] = d_cell_edges[dim][i];
                }
            }
        }
    }

    Base::d_num_cells = d_indexer.total_cells();
    Base::d_num_local_cells = Base::d_num_cells;
    Base::d_num_owned_cells = Base::d_num_cells;

    Base::d_num_faces = d_indexer.total_faces();
    Base::d_num_owned_faces = Base::d_num_faces;

    Base::d_num_nodes = d_indexer.total_nodes();

    d_topology = OrthoMeshTopo(
        d_indexer,
        {{"rmin", "rmax", "thetamin", "thetamax", "zmin", "zmax"}});
}

const std::string&
OrthogonalCylindrial3D::boundary_batch_name_impl(int batch_id) const
{
    return d_topology.boundary_batch_name(batch_id);
}

} // namespace SimpleFluid::Meshes
