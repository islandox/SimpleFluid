/**
 * @file OrthogonalCartesian3D.ipp
 * @brief Inline entity-query implementations for OrthogonalCartesian3D.
 */

#pragma once

#include "utils/debug_check.hh"

namespace SimpleFluid::Meshes
{

inline void OrthogonalCartesian3D::check_cell_id(cell_id_t id) const
{
    CHECK_BOUNDS(id.i, 0, d_indexer.num_cells_per_dim[X]);
    CHECK_BOUNDS(id.j, 0, d_indexer.num_cells_per_dim[Y]);
    CHECK_BOUNDS(id.k, 0, d_indexer.num_cells_per_dim[Z]);
}

inline void OrthogonalCartesian3D::check_face_id(face_id_t id) const
{
    CHECK(id.orientation == X_FACE || id.orientation == Y_FACE || id.orientation == Z_FACE);
    CHECK_BOUNDS(id.i, 0, id.orientation == X_FACE ?
                            d_indexer.num_cells_per_dim[X] + 1 : d_indexer.num_cells_per_dim[X]);
    CHECK_BOUNDS(id.j, 0, id.orientation == Y_FACE ? 
                            d_indexer.num_cells_per_dim[Y] + 1 : d_indexer.num_cells_per_dim[Y]);
    CHECK_BOUNDS(id.k, 0, id.orientation == Z_FACE ? 
                            d_indexer.num_cells_per_dim[Z] + 1 : d_indexer.num_cells_per_dim[Z]);
}

inline void OrthogonalCartesian3D::check_node_id(node_id_t id) const
{
    CHECK_BOUNDS(id.i, 0, d_indexer.num_cells_per_dim[X] + 1);
    CHECK_BOUNDS(id.j, 0, d_indexer.num_cells_per_dim[Y] + 1);
    CHECK_BOUNDS(id.k, 0, d_indexer.num_cells_per_dim[Z] + 1);
}

inline real_t
OrthogonalCartesian3D::cell_volume_impl(cell_id_t id) const
{
    return d_cell_widths[X][id.i]
         * d_cell_widths[Y][id.j]
         * d_cell_widths[Z][id.k];
}

inline auto
OrthogonalCartesian3D::cell_centroid_impl(cell_id_t id) const -> Vec3
{
    return {
        d_cell_centroids[X][id.i],
        d_cell_centroids[Y][id.j],
        d_cell_centroids[Z][id.k]};
}

inline auto
OrthogonalCartesian3D::cell_faces_impl(cell_id_t id) const
    -> std::array<face_id_t, 6>
{
    return {{
        {id.i, id.j, id.k, X_FACE},
        {id.i + 1, id.j, id.k, X_FACE},
        {id.i, id.j, id.k, Y_FACE},
        {id.i, id.j + 1, id.k, Y_FACE},
        {id.i, id.j, id.k, Z_FACE},
        {id.i, id.j, id.k + 1, Z_FACE}}};
}

inline auto
OrthogonalCartesian3D::owner_cell_impl(face_id_t id) const -> cell_id_t
{
    return d_topology.owner_cell(id);
}

inline auto
OrthogonalCartesian3D::neighbor_cell_impl(face_id_t id) const -> cell_id_t
{
    return d_topology.neighbor_cell(id);
}

inline real_t
OrthogonalCartesian3D::face_area_impl(face_id_t id) const
{
    if (id.orientation == X_FACE)
    {
        return d_cell_widths[Y][id.j]
             * d_cell_widths[Z][id.k];
    }
    if (id.orientation == Y_FACE)
    {
        return d_cell_widths[X][id.i]
             * d_cell_widths[Z][id.k];
    }
    return d_cell_widths[X][id.i]
         * d_cell_widths[Y][id.j];
}

inline auto
OrthogonalCartesian3D::face_centroid_impl(face_id_t id) const -> Vec3
{
    if (id.orientation == X_FACE)
    {
        return {
            d_cell_edges[X][id.i],
            d_cell_centroids[Y][id.j],
            d_cell_centroids[Z][id.k]};
    }
    if (id.orientation == Y_FACE)
    {
        return {
            d_cell_centroids[X][id.i],
            d_cell_edges[Y][id.j],
            d_cell_centroids[Z][id.k]};
    }
    return {
        d_cell_centroids[X][id.i],
        d_cell_centroids[Y][id.j],
        d_cell_edges[Z][id.k]};
}

inline auto
OrthogonalCartesian3D::face_normal_impl(face_id_t id) const -> Vec3
{
    const auto sign =
        (id.orientation == X_FACE && id.i == 0)
        || (id.orientation == Y_FACE && id.j == 0)
        || (id.orientation == Z_FACE && id.k == 0)
      ? -1.0
      : 1.0;

    Vec3 normal;
    normal.component(static_cast<size_t>(id.orientation)) = sign;
    return normal;
}

inline auto
OrthogonalCartesian3D::node_coordinates_impl(node_id_t id) const -> Vec3
{
    return {
        d_cell_edges[X][id.i],
        d_cell_edges[Y][id.j],
        d_cell_edges[Z][id.k]};
}

inline int OrthogonalCartesian3D::boundary_id_impl(face_id_t id) const
{
    return d_topology.boundary_id(id);
}

inline auto
OrthogonalCartesian3D::boundary_face_patch_impl(int patch_id) const
{
    return d_topology.boundary_face_patch(patch_id);
}

inline auto
OrthogonalCartesian3D::boundary_patch_ids_impl() const
{
    return d_topology.boundary_patch_ids();
}

inline int
OrthogonalCartesian3D::num_boundary_patches_impl() const noexcept
{
    return d_topology.num_boundary_patches();
}

} // namespace SimpleFluid
