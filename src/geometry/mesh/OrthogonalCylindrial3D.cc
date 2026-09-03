/**
 * @file OrthogonalCylindrial3D.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief OrthogonalCylindrial3D setup and indexing implementation.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
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

/**
 * @brief Validate one cylindrical coordinate-edge array.
 * @param edges Edge coordinates to validate.
 * @param coordinate_name Coordinate label used in diagnostics.
 * @throws std::invalid_argument If the array is too short, non-finite, or not
 *         strictly increasing.
 * @throws std::overflow_error If its cell count exceeds the mesh ID type.
 */
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

/**
 * @brief Construct and precompute cylindrical geometry and topology.
 * @param cell_edges Strictly increasing radial, angular, and axial edges.
 * @throws std::invalid_argument If coordinates or the angular domain are invalid.
 * @throws std::overflow_error If entity counts exceed supported ID ranges.
 */
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

/**
 * @brief Replace axial coordinates without changing cylindrical topology.
 * @param edges Strictly increasing axial edges with the existing size.
 * @throws std::invalid_argument If the replacement layout is invalid.
 */
void OrthogonalCylindrial3D::replace_axial_edges_fixed_topology(Arr<real_t> edges)
{
    if (d_geometry_state.epoch == std::numeric_limits<std::uint64_t>::max())
    {
        throw std::overflow_error("OrthogonalCylindrial3D geometry epoch overflow.");
    }
    if (edges.size() != d_cell_edges[AXIAL].size())
    {
        throw std::invalid_argument("OrthogonalCylindrial3D fixed-topology update cannot change "
                                    "the axial edge count.");
    }

    validate_edges(edges, "z");
    auto midpoints = MeshUtils::consec_mid(edges);
    auto widths = MeshUtils::consec_diff(edges);
    auto radial_face_widths = widths;
    auto theta_face_widths = widths;

    d_cell_edges[AXIAL] = std::move(edges);
    d_cell_midpoints[AXIAL] = std::move(midpoints);
    d_cell_widths[AXIAL] = std::move(widths);
    d_face_area_magnitudes[R_FACE][AXIAL] = std::move(radial_face_widths);
    d_face_area_magnitudes[THETA_FACE][AXIAL] = std::move(theta_face_widths);
    ++d_geometry_state.epoch;
}

/**
 * @brief Return a physical boundary batch name.
 * @param batch_id Boundary batch identifier.
 * @return Configured radial, angular, or axial boundary name.
 * @throws std::out_of_range If @p batch_id is invalid.
 */
const std::string&
OrthogonalCylindrial3D::boundary_batch_name_impl(int batch_id) const
{
    return d_topology.boundary_batch_name(batch_id);
}

} // namespace SimpleFluid::Meshes
