/**
 * @file SemiStructuredXY_Z.ipp
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Inline entity-query implementations for SemiStructuredXY_Z.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "utils/debug_check.hh"

namespace SimpleFluid::Meshes
{

/**
 * @brief Check a base-cell ordinal and axial cell layer.
 * @param id Cell ID to validate.
 */
inline void SemiStructuredXY_Z::check_cell_id(cell_id_t id) const
{
    CHECK_BOUNDS(id.ij, 0, indexer().num_cells_per_layer);
    CHECK_BOUNDS(id.k, 0, indexer().num_layers);
}

/**
 * @brief Check an axial or side face ID against its orientation-specific bounds.
 * @param id Face ID to validate.
 */
inline void SemiStructuredXY_Z::check_face_id(face_id_t id) const
{
    CHECK(id.orientation == Z_FACE || id.orientation == SIDE_FACE);
    if (id.orientation == Z_FACE)
    {
        CHECK_BOUNDS(id.ij, 0, indexer().num_cells_per_layer);
        CHECK_BOUNDS(id.k, 0, indexer().num_node_layers);
        return;
    }

    CHECK_BOUNDS(id.ij, 0, indexer().num_side_faces_per_layer);
    CHECK_BOUNDS(id.k, 0, indexer().num_layers);
}

/**
 * @brief Check a base-node ordinal and axial node layer.
 * @param id Node ID to validate.
 */
inline void SemiStructuredXY_Z::check_node_id(node_id_t id) const
{
    CHECK_BOUNDS(id.ij, 0, indexer().num_nodes_per_layer);
    CHECK_BOUNDS(id.k, 0, indexer().num_node_layers);
}

/**
 * @brief Compute prism volume from base area and axial width.
 * @param id Cell identifier.
 * @return Cell volume.
 */
inline real_t SemiStructuredXY_Z::cell_volume_impl(cell_id_t id) const
{
    return d_xy_cell_areas[id.ij] * d_z_widths[id.k];
}

/**
 * @brief Combine a base-cell centroid with an axial midpoint.
 * @param id Cell identifier.
 * @return Cell centroid.
 */
inline auto SemiStructuredXY_Z::cell_centroid_impl(cell_id_t id) const
    -> Vec3
{
    auto centroid = d_xy_cell_centroids[id.ij];
    centroid.z = d_z_midpoints[id.k];
    return centroid;
}

inline auto SemiStructuredXY_Z::cell_faces_impl(cell_id_t id) const
    -> std::vector<face_id_t>
{
    return d_topology.cell_faces(id);
}

inline auto SemiStructuredXY_Z::owner_cell_impl(face_id_t id) const
    -> cell_id_t
{
    return d_topology.owner_cell(id);
}

inline auto SemiStructuredXY_Z::neighbor_cell_impl(face_id_t id) const
    -> cell_id_t
{
    return d_topology.neighbor_cell(id);
}

/**
 * @brief Compute the area of an axial polygon or extruded side face.
 * @param id Face identifier.
 * @return Face area.
 */
inline real_t SemiStructuredXY_Z::face_area_impl(face_id_t id) const
{
    if (id.orientation == Z_FACE)
    {
        return d_xy_cell_areas[id.ij];
    }
    return d_xy_edge_lengths[id.ij] * d_z_widths[id.k];
}

/**
 * @brief Compute an axial or side-face centroid.
 * @param id Face identifier.
 * @return Face centroid.
 */
inline auto SemiStructuredXY_Z::face_centroid_impl(face_id_t id) const
    -> Vec3
{
    if (id.orientation == Z_FACE)
    {
        auto centroid = d_xy_cell_centroids[id.ij];
        centroid.z = d_z_edges[id.k];
        return centroid;
    }

    auto centroid = d_xy_edge_centroids[id.ij];
    centroid.z = d_z_midpoints[id.k];
    return centroid;
}

/**
 * @brief Return the owner-oriented unit normal of an axial or side face.
 * @param id Face identifier.
 * @return Face unit normal.
 */
inline auto SemiStructuredXY_Z::face_normal_impl(face_id_t id) const
    -> Vec3
{
    if (id.orientation == Z_FACE)
    {
        return {0.0, 0.0, id.k == 0 ? -1.0 : 1.0};
    }
    return d_xy_edge_normals[id.ij];
}

/**
 * @brief Combine a base-node point with an axial edge coordinate.
 * @param id Node identifier.
 * @return Node coordinates.
 */
inline auto SemiStructuredXY_Z::node_coordinates_impl(node_id_t id) const
    -> Vec3
{
    auto point = d_xy_nodes[id.ij];
    point.z = d_z_edges[id.k];
    return point;
}

inline int SemiStructuredXY_Z::boundary_id_impl(face_id_t id) const
{
    return d_topology.boundary_id(id);
}

inline const std::string&
SemiStructuredXY_Z::boundary_batch_name_impl(int batch_id) const
{
    return d_topology.boundary_batch_name(batch_id);
}

inline auto
SemiStructuredXY_Z::boundary_face_batch_impl(int batch_id) const
{
    return d_topology.boundary_face_batch(batch_id);
}

inline auto
SemiStructuredXY_Z::boundary_batch_ids_impl() const
{
    return d_topology.boundary_batch_ids();
}

inline int
SemiStructuredXY_Z::num_boundary_batches_impl() const noexcept
{
    return d_topology.num_boundary_batches();
}

} // namespace SimpleFluid::Meshes
