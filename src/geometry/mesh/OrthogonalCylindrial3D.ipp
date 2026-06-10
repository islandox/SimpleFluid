/**
 * @file OrthogonalCylindrial3D.ipp
 * @brief Inline entity-query implementations for OrthogonalCylindrial3D.
 */

#pragma once

#include "utils/debug_check.hh"

namespace SimpleFluid::Mesh
{

inline void OrthogonalCylindrial3D::check_cell_id(cell_id_t id) const
{
    CHECK_BOUNDS(id.i, 0, d_indexer.num_cells_per_dim[R]);
    CHECK_BOUNDS(id.j, 0, d_indexer.num_cells_per_dim[THETA]);
    CHECK_BOUNDS(id.k, 0, d_indexer.num_cells_per_dim[AXIAL]);
}

inline void OrthogonalCylindrial3D::check_face_id(face_id_t id) const
{
    CHECK(id.orientation == R_FACE || id.orientation == THETA_FACE || id.orientation == Z_FACE);
    CHECK_BOUNDS(
        id.i,
        0,
        d_indexer.num_faces_per_dim_per_orientation[id.orientation][R]);
    CHECK_BOUNDS(
        id.j,
        0,
        d_indexer.num_faces_per_dim_per_orientation[id.orientation][THETA]);
    CHECK_BOUNDS(
        id.k,
        0,
        d_indexer.num_faces_per_dim_per_orientation[id.orientation][AXIAL]);
}

inline void OrthogonalCylindrial3D::check_node_id(node_id_t id) const
{
    CHECK_BOUNDS(id.i, 0, d_indexer.num_nodes_per_dim[R]);
    CHECK_BOUNDS(id.j, 0, d_indexer.num_nodes_per_dim[THETA]);
    CHECK_BOUNDS(id.k, 0, d_indexer.num_nodes_per_dim[AXIAL]);
}

inline auto OrthogonalCylindrial3D::cylindrical_point(
    real_t radius,
    real_t theta,
    real_t z) const -> Vec3
{
    return {radius * std::cos(theta), radius * std::sin(theta), z};
}

inline real_t OrthogonalCylindrial3D::sector_centroid_radius(
    size_t radial_cell,
    size_t theta_cell) const
{
    const auto r0 = d_cell_edges[R][radial_cell];
    const auto r1 = d_cell_edges[R][radial_cell + 1];
    const auto delta_theta = d_cell_widths[THETA][theta_cell];
    const auto radial_average =
        (2.0 / 3.0) * (r1 * r1 * r1 - r0 * r0 * r0)
      / (r1 * r1 - r0 * r0);
    const auto half_angle = 0.5 * delta_theta;
    return radial_average * std::sin(half_angle) / half_angle;
}

inline real_t
OrthogonalCylindrial3D::cell_volume_impl(cell_id_t id) const
{
    const auto i = id.i;
    const auto j = id.j;
    const auto k = id.k;
    return 0.5 * d_cell_Dr2[i]
         * d_cell_widths[THETA][j]
         * d_cell_widths[AXIAL][k];
}

inline auto
OrthogonalCylindrial3D::cell_centroid_impl(cell_id_t id) const -> Vec3
{
    const auto i = id.i;
    const auto j = id.j;
    const auto k = id.k;
    return cylindrical_point(
        sector_centroid_radius(i, j),
        d_cell_midpoints[THETA][j],
        d_cell_midpoints[AXIAL][k]);
}

inline auto
OrthogonalCylindrial3D::cell_faces_impl(cell_id_t id) const
    -> std::array<face_id_t, 6>
{
    const auto upper_theta =
        is_theta_periodic()
        && static_cast<size_t>(id.j + 1)
            == d_indexer.num_cells_per_dim[THETA]
      ? 0
      : id.j + 1;

    return {{
        {id.i, id.j, id.k, R_FACE},
        {id.i + 1, id.j, id.k, R_FACE},
        {id.i, id.j, id.k, THETA_FACE},
        {id.i, upper_theta, id.k, THETA_FACE},
        {id.i, id.j, id.k, Z_FACE},
        {id.i, id.j, id.k + 1, Z_FACE}}};
}

inline auto
OrthogonalCylindrial3D::owner_cell_impl(face_id_t id) const -> cell_id_t
{
    return d_topology.owner_cell(id);
}

inline auto
OrthogonalCylindrial3D::neighbor_cell_impl(face_id_t id) const -> cell_id_t
{
    return d_topology.neighbor_cell(id);
}

inline real_t
OrthogonalCylindrial3D::face_area_impl(face_id_t id) const
{
    const auto i = id.i;
    const auto j = id.j;
    const auto k = id.k;
    const auto O = static_cast<size_t>(id.orientation);

    return d_face_area_magnitudes[O][R][i]
         * d_face_area_magnitudes[O][THETA][j]
         * d_face_area_magnitudes[O][AXIAL][k];
}

inline auto
OrthogonalCylindrial3D::face_centroid_impl(face_id_t id) const -> Vec3
{
    const auto i = id.i;
    const auto j = id.j;
    const auto k = id.k;

    if (id.orientation == R_FACE)
    {
        const auto half_angle = 0.5 * d_cell_widths[THETA][j];
        const auto centroid_radius =
            d_cell_edges[R][i] * std::sin(half_angle) / half_angle;
        return cylindrical_point(
            centroid_radius,
            d_cell_midpoints[THETA][j],
            d_cell_midpoints[AXIAL][k]);
    }
    if (id.orientation == THETA_FACE)
    {
        return cylindrical_point(
            d_cell_midpoints[R][i],
            d_cell_edges[THETA][j],
            d_cell_midpoints[AXIAL][k]);
    }
    return cylindrical_point(
        sector_centroid_radius(i, j),
        d_cell_midpoints[THETA][j],
        d_cell_edges[AXIAL][k]);
}

inline auto
OrthogonalCylindrial3D::face_normal_impl(face_id_t id) const -> Vec3
{
    if (id.orientation == R_FACE)
    {
        const auto theta =
            d_cell_midpoints[THETA][id.j];
        const auto sign = id.i == 0 ? -1.0 : 1.0;
        return {sign * std::cos(theta), sign * std::sin(theta), 0.0};
    }
    if (id.orientation == THETA_FACE)
    {
        const auto theta =
            d_cell_edges[THETA][id.j];
        const auto sign = !is_theta_periodic() && id.j == 0 ? -1.0 : 1.0;
        return {
            sign * -std::sin(theta),
            sign * std::cos(theta),
            0.0};
    }
    return {0.0, 0.0, id.k == 0 ? -1.0 : 1.0};
}

inline auto
OrthogonalCylindrial3D::node_coordinates_impl(node_id_t id) const -> Vec3
{
    return cylindrical_point(
        d_cell_edges[R][id.i],
        d_cell_edges[THETA][id.j],
        d_cell_edges[AXIAL][id.k]);
}

inline int OrthogonalCylindrial3D::boundary_id_impl(face_id_t id) const
{
    return d_topology.boundary_id(id);
}

} // namespace SimpleFluid
