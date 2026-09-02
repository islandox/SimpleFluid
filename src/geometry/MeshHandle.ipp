/**
 * @file MeshHandle.ipp
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Inline definitions for per-face and per-cell geometry queries
 *        on MeshHandle.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <limits>
#include <span>
#include <stdexcept>

namespace SimpleFluid
{

/// @name Per-cell geometry queries
/// @{

/// @brief Returns the volume of the cell identified by @p cell_lid.
/// @param cell_lid  Local index of the query cell.
/// @return The volume of the cell as a `real_t`.
/// @note Delegates to the underlying geometry mesh via `visit_geometry_cell`.
template<TpetraTypePack Pack>
inline real_t
MeshHandle<Pack>::cell_volume(local_ordinal_type cell_lid) const
{
    return visit_geometry_cell(
        cell_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.cell_volume(id);
        });
}

/// @brief Returns the centroid (Vec3) of the cell identified by @p cell_lid.
/// @param cell_lid  Local index of the query cell.
/// @return The centroid of the cell as a `Vec3`.
/// @note Delegates to the underlying geometry mesh via `visit_geometry_cell`.
template<TpetraTypePack Pack>
inline auto
MeshHandle<Pack>::cell_centroid(local_ordinal_type cell_lid) const -> Vec3
{
    return visit_geometry_cell(
        cell_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.cell_centroid(id);
        });
}

/// @brief Returns a span over the local-ids of the faces bounding the given cell.
/// @param cell_lid  Local index of the query cell.
/// @return A `std::span` over the local-ids of the bounding faces.
/// @note Legacy meshes return a zero-copy view of their existing connectivity
///       when face ordering is already owned-first.
template<TpetraTypePack Pack>
inline std::span<const typename MeshHandle<Pack>::local_ordinal_type>
MeshHandle<Pack>::faces(local_ordinal_type cell_lid) const
{
    check_cell(cell_lid);
    if (d_cell_face_offsets.empty())
    {
        if (const auto legacy = legacy_mesh())
        {
            const auto geometry_lid = geometry_cell_lid(cell_lid);
            const auto& face_lids = legacy->faces(checked_local(
                static_cast<size_t>(geometry_lid)));
            return {
                face_lids.empty() ? nullptr : &face_lids[0],
                face_lids.size()};
        }
    }
    const auto local = static_cast<size_t>(cell_lid);
    const auto begin = d_cell_face_offsets[local];
    const auto end   = d_cell_face_offsets[local + 1];
    return std::span<const local_ordinal_type>(d_cell_face_lids)
        .subspan(begin, end - begin);
}

/// @}

/// @name Per-face geometry queries
/// @{

/// @brief Returns the area of the face identified by @p face_lid.
/// @param face_lid  Local index of the query face.
/// @return The area of the face as a `real_t`.
/// @note Delegates to the underlying geometry mesh via `visit_geometry_face`.
template<TpetraTypePack Pack>
inline real_t
MeshHandle<Pack>::face_area(local_ordinal_type face_lid) const
{
    return visit_geometry_face(
        face_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.face_area(id);
        });
}

/// @brief Returns the centroid (Vec3) of the face identified by @p face_lid.
/// @param face_lid  Local index of the query face.
/// @return The centroid of the face as a `Vec3`.
/// @note Delegates to the underlying geometry mesh via `visit_geometry_face`.
template<TpetraTypePack Pack>
inline auto
MeshHandle<Pack>::face_centroid(local_ordinal_type face_lid) const -> Vec3
{
    return visit_geometry_face(
        face_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.face_centroid(id);
        });
}

/// @brief Returns the outward-pointing unit normal (Vec3) of the face.
/// @param face_lid  Local index of the query face.
/// @return The outward-pointing unit normal as a `Vec3`.
/// @note "Outward" is defined by the mesh convention (typically relative to
///       the owner cell). Delegates to the underlying geometry mesh.
template<TpetraTypePack Pack>
inline auto
MeshHandle<Pack>::face_normal(local_ordinal_type face_lid) const -> Vec3
{
    return visit_geometry_face(
        face_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.face_normal(id);
        });
}

/// @brief Returns the integer boundary-condition tag associated with the face.
/// @param face_lid  Local index of the query face.
/// @return The boundary-condition tag, or `invalid_boundary_id` for interior faces.
/// @note A value of `invalid_boundary_id` indicates an interior face.
template<TpetraTypePack Pack>
inline int
MeshHandle<Pack>::boundary_id(local_ordinal_type face_lid) const
{
    return visit_geometry_face(
        face_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.boundary_id(id);
        });
}

/// @}

/// @name Face-cell topology helpers
/// @{

/// @brief Returns the local id of the owner or neighbor cell adjacent to a face.
/// @param face_lid  Local index of the query face.
/// @param owner     If `true`, returns the owner cell; if `false`, returns the
///                  neighbor cell.
/// @return The local cell id, or `invalid_local_id()` if the cell does not
///         exist in the local partition or the face is on the boundary.
/// @note Performs a two-step lookup:
///       1. Use the geometry mesh to obtain the geometry-level cell id.
///       2. Map that geometry id back to a local cell id via the indexer.
template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::adjacent_cell(local_ordinal_type face_lid,
                                bool owner) const
{
    const auto geometry_lid = geometry_face_lid(face_lid);
    const auto geometry_cell = visit(
        [&](const auto& mesh) -> size_t
        {
            const auto face =
                mesh.face_id(static_cast<size_t>(geometry_lid));
            const auto cell = owner
                ? mesh.owner_cell(face)
                : mesh.neighbor_cell(face);
            // Different mesh backends encode "invalid cell" differently.
            if constexpr (std::is_same_v<
                              std::decay_t<decltype(mesh)>,
                              STKAdapter>)
            {
                if (cell == invalid_id<local_ordinal_type>())
                {
                    return std::numeric_limits<size_t>::max();
                }
            }
            else if (cell == std::decay_t<decltype(mesh)>::invalid_cell_id())
            {
                return std::numeric_limits<size_t>::max();
            }
            return static_cast<size_t>(mesh.cell_local_id(cell));
        });

    if (geometry_cell == std::numeric_limits<size_t>::max())
    {
        return invalid_local_id();
    }
    return geometry_to_local_cell(geometry_cell);
}

/// @}

/// @name Face-cell topology accessors
/// @{

/// @brief Returns the local id of the owner cell of the given face.
/// @param face_lid  Local index of the query face.
/// @return The local id of the owner cell.
/// @note Convenience wrapper around `adjacent_cell(face_lid, /*owner=*/true)`.
template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::owner_cell(local_ordinal_type face_lid) const
{
    return adjacent_cell(face_lid, true);
}

/// @brief Returns the local id of the neighbor cell of the given face.
/// @param face_lid  Local index of the query face.
/// @return The local id of the neighbor cell, or `invalid_local_id()` for
///         boundary faces.
/// @note Convenience wrapper around `adjacent_cell(face_lid, /*owner=*/false)`.
template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::neighbor_cell(local_ordinal_type face_lid) const
{
    return adjacent_cell(face_lid, false);
}

/// @brief Returns the face-area vector (normal * area) using the mesh-convention
///        outward direction (owner side).
/// @param face_lid  Local index of the query face.
/// @return The face-area vector as a `Vec3`.
/// @note Equivalent to `face_normal * face_area`.
template<TpetraTypePack Pack>
inline auto
MeshHandle<Pack>::face_area_vector(local_ordinal_type face_lid) const -> Vec3
{
    return face_normal(face_lid) * face_area(face_lid);
}

/// @brief Returns the face-area vector oriented outward from the specified
///        adjacent cell.
/// @param face_lid  Local index of the query face.
/// @param cell_lid  Local index of the adjacent cell.
/// @return The face-area vector as a `Vec3`.
/// @note The sign of the normal is flipped if @p cell_lid is the neighbor cell.
template<TpetraTypePack Pack>
inline auto
MeshHandle<Pack>::face_area_vector_outward(
    local_ordinal_type face_lid,
    local_ordinal_type cell_lid) const -> Vec3
{
    return face_normal_outward(face_lid, cell_lid) * face_area(face_lid);
}

/// @brief Euclidean distance between the face centroid and the given cell centroid.
/// @param face_lid  Local index of the query face.
/// @param cell_lid  Local index of the adjacent cell.
/// @return The Euclidean distance as a `real_t`.
template<TpetraTypePack Pack>
inline real_t
MeshHandle<Pack>::cell_to_face_distance(
    local_ordinal_type face_lid,
    local_ordinal_type cell_lid) const
{
    return (face_centroid(face_lid) - cell_centroid(cell_lid)).norm();
}

/// @brief Return whether the underlying geometry has no cell across a face.
/// @param face_lid Local index of the query face.
/// @return True for a physical exterior, independent of overlap depth.
template<TpetraTypePack Pack>
inline bool
MeshHandle<Pack>::is_geometry_exterior_face(
    local_ordinal_type face_lid) const
{
    const auto geometry_lid = geometry_face_lid(face_lid);
    return std::visit(
        [geometry_lid](const auto& mesh_pointer)
        {
            const auto& mesh = *mesh_pointer;
            const auto id = mesh.face_id(
                static_cast<size_t>(geometry_lid));
            const auto neighbor = mesh.neighbor_cell(id);
            if constexpr (std::is_same_v<
                              std::decay_t<decltype(mesh)>,
                              STKAdapter>)
            {
                return neighbor == invalid_id<local_ordinal_type>();
            }
            else
            {
                return neighbor
                    == std::decay_t<decltype(mesh)>::invalid_cell_id();
            }
        },
        d_mesh);
}

/// @brief Returns true if the face lies on the domain boundary (no neighbor cell).
/// @param face_lid  Local index of the query face.
/// @return `true` if the face is exterior (has no neighbor).
template<TpetraTypePack Pack>
inline bool
MeshHandle<Pack>::is_exterior_face(local_ordinal_type face_lid) const
{
    return neighbor_cell(face_lid) == invalid_local_id();
}

/// @brief Returns true if the face is interior (has both owner and neighbor cells).
/// @param face_lid  Local index of the query face.
/// @return `true` if the face is interior.
template<TpetraTypePack Pack>
inline bool
MeshHandle<Pack>::is_interior_face(local_ordinal_type face_lid) const
{
    return !is_exterior_face(face_lid);
}

/// @brief Returns true if the face is on the domain boundary *and* carries a
///        valid (non-default) boundary-condition identifier.
/// @param face_lid  Local index of the query face.
/// @return `true` if the face is a boundary face with a valid BC identifier.
template<TpetraTypePack Pack>
inline bool
MeshHandle<Pack>::is_boundary_face(local_ordinal_type face_lid) const
{
    return is_exterior_face(face_lid)
        && boundary_id(face_lid) != invalid_boundary_id;
}

/// @}

/// @name Derived face / cell queries
/// @{

/// @brief Given a face and one adjacent cell, returns the cell on the other side of
///        the face.
/// @param face_lid  Local index of the query face.
/// @param cell_lid  Local index of the adjacent cell.
/// @return The local id of the opposite cell.
/// @throws std::invalid_argument if @p cell_lid is not adjacent to the face.
template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::opposite_cell(local_ordinal_type face_lid,
                                local_ordinal_type cell_lid) const
{
    const auto owner   = owner_cell(face_lid);
    const auto neighbor = neighbor_cell(face_lid);
    if (cell_lid == owner)
    {
        return neighbor;
    }
    if (neighbor != invalid_local_id() && cell_lid == neighbor)
    {
        return owner;
    }
    throw std::invalid_argument(
        "Cell is not adjacent to requested face.");
}

/// @brief Placeholder for periodic-boundary support.
/// @param face_lid  Local index of the query face.
/// @param cell_lid  Local index of the adjacent cell.
/// @return The opposite cell id (same as `opposite_cell`).
/// @note Currently delegates to `opposite_cell` (non-periodic behaviour).
template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::opposite_or_periodic_neighbor_cell(
    local_ordinal_type face_lid,
    local_ordinal_type cell_lid) const
{
    return opposite_cell(face_lid, cell_lid);
}

/// @brief Returns the face normal oriented outward from the given adjacent cell.
/// @param face_lid  Local index of the query face.
/// @param cell_lid  Local index of the adjacent cell.
/// @return The outward-pointing unit normal as a `Vec3`.
/// @throws std::invalid_argument if @p cell_lid is not adjacent to the face.
/// @note Flips the sign of `face_normal` when @p cell_lid is the neighbor cell.
template<TpetraTypePack Pack>
inline auto
MeshHandle<Pack>::face_normal_outward(
    local_ordinal_type face_lid,
    local_ordinal_type cell_lid) const -> Vec3
{
    const auto owner = owner_cell(face_lid);
    if (cell_lid == owner)
    {
        return face_normal(face_lid);
    }
    if (cell_lid == neighbor_cell(face_lid))
    {
        return face_normal(face_lid) * -1.0;
    }
    throw std::invalid_argument(
        "Cell is not adjacent to requested face.");
}

/// @brief Distance between the centroids of the two cells straddling the face.
/// @param face_lid  Local index of the query face.
/// @return The Euclidean distance as a `real_t`, or 0.0 for exterior (boundary) faces.
/// @note Returns 0.0 for exterior (boundary) faces.
template<TpetraTypePack Pack>
inline real_t
MeshHandle<Pack>::face_cell_center_distance(
    local_ordinal_type face_lid) const
{
    const auto neighbor = neighbor_cell(face_lid);
    if (neighbor == invalid_local_id())
    {
        return 0.0;
    }
    return (cell_centroid(neighbor)
          - cell_centroid(owner_cell(face_lid))).norm();
}

/// @brief Vector from @p cell_lid centroid to the centroid of the opposite cell
///        across the given face.
/// @param face_lid  Local index of the query face.
/// @param cell_lid  Local index of the adjacent cell.
/// @return The vector from `cell_centroid(cell_lid)` to `cell_centroid(opposite_cell(...))`.
/// @throws std::invalid_argument for exterior faces (no opposite cell).
template<TpetraTypePack Pack>
inline auto
MeshHandle<Pack>::cell_center_vector(
    local_ordinal_type face_lid,
    local_ordinal_type cell_lid) const -> Vec3
{
    const auto other = opposite_cell(face_lid, cell_lid);
    if (other == invalid_local_id())
    {
        throw std::invalid_argument(
            "Exterior face does not have an opposite cell.");
    }
    return cell_centroid(other) - cell_centroid(cell_lid);
}

/// @}

/// @name Visit helpers (private)
/// @{

/// @brief Dispatches a callable to the active geometry mesh variant for a cell.
/// @tparam Function  Callable type (deduced).
/// @param cell_lid  Local index of the cell.
/// @param function  Callable to invoke with `(mesh, geometry_cell_id)`.
/// @return The result of the callable, with type deduced via `decltype(auto)`.
/// @note The callable receives the mesh reference and the geometry-level cell id
///       corresponding to the local cell id.
template<TpetraTypePack Pack>
template<class Function>
decltype(auto)
MeshHandle<Pack>::visit_geometry_cell(
    local_ordinal_type cell_lid,
    Function&& function) const
{
    const auto geometry_lid = geometry_cell_lid(cell_lid);
    return visit(
        [&](const auto& mesh) -> decltype(auto)
        {
            return std::forward<Function>(function)(
                mesh,
                mesh.cell_id(static_cast<size_t>(geometry_lid)));
        });
}

/// @brief Dispatches a callable to the active geometry mesh variant for a face.
/// @tparam Function  Callable type (deduced).
/// @param face_lid  Local index of the face.
/// @param function  Callable to invoke with `(mesh, geometry_face_id)`.
/// @return The result of the callable, with type deduced via `decltype(auto)`.
/// @note The callable receives the mesh reference and the geometry-level face id
///       corresponding to the local face id.
template<TpetraTypePack Pack>
template<class Function>
decltype(auto)
MeshHandle<Pack>::visit_geometry_face(
    local_ordinal_type face_lid,
    Function&& function) const
{
    const auto geometry_lid = geometry_face_lid(face_lid);
    return visit(
        [&](const auto& mesh) -> decltype(auto)
        {
            return std::forward<Function>(function)(
                mesh,
                mesh.face_id(static_cast<size_t>(geometry_lid)));
        });
}

/// @}

} // namespace SimpleFluid
