/**
 * @file MeshHandle.ipp
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Inline definitions for MeshHandle: borrowed-view forwarders, binding
 *        publication, and the allocation-free cell-face observation helpers.
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
#include <variant>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
inline const detail::MeshBinding<Pack>& MeshHandle<Pack>::binding() const
{
    if (!d_binding)
    {
        throw std::logic_error("MeshHandle binding is not published.");
    }
    d_binding->validate(d_layout_generation);
    return *d_binding;
}

template<TpetraTypePack Pack>
inline void MeshHandle<Pack>::publish_binding()
{
    auto binding = std::make_shared<detail::MeshBinding<Pack>>();
    binding->geometry_identity =
        std::visit([](const auto& mesh) noexcept -> const void* { return mesh.get(); }, d_mesh);
    binding->owned_cell_map = d_owned_cell_map.get();
    binding->overlap_cell_map = d_overlap_cell_map.get();
    binding->owned_face_map = d_owned_face_map.get();
    binding->overlap_face_map = d_overlap_face_map.get();
    binding->boundary_face_map = d_boundary_face_map.get();
    binding->num_owned_cells = num_owned_cells();
    binding->num_local_cells = num_local_cells();
    binding->num_owned_faces = num_owned_faces();
    binding->num_local_faces = num_faces();
    binding->owned_cell_map_size = d_owned_cell_map ? d_owned_cell_map->getLocalNumElements() : 0;
    binding->overlap_cell_map_size = d_overlap_cell_map ? d_overlap_cell_map->getLocalNumElements() : 0;
    binding->owned_face_map_size = d_owned_face_map ? d_owned_face_map->getLocalNumElements() : 0;
    binding->overlap_face_map_size = d_overlap_face_map ? d_overlap_face_map->getLocalNumElements() : 0;
    binding->boundary_face_map_size = d_boundary_face_map ? d_boundary_face_map->getLocalNumElements() : 0;
    binding->layout_generation = d_layout_generation;
    binding->cells_reordered = d_cells_reordered;
    binding->serial_identity = d_serial_identity;
    binding->validate();
    d_binding = std::move(binding);
}

template<TpetraTypePack Pack>
inline void MeshHandle<Pack>::bump_layout_generation()
{
    ++d_layout_generation;
    publish_binding();
}

/// @name Topology forwarders
/// @{

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::orthogonal_decomposition() const noexcept
    -> SP<const Meshes::OrthogonalMeshDecomposition>
{
    return topology().orthogonal_decomposition();
}

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::decomposed_orthogonal_topology() const noexcept
    -> SP<const Meshes::DecomposedOrthogonalTopology>
{
    return topology().decomposed_orthogonal_topology();
}

template<TpetraTypePack Pack>
inline const typename MeshHandle<Pack>::SemiStructuredDecomposition&
MeshHandle<Pack>::semi_structured_decomposition() const
{
    return topology().semi_structured_decomposition();
}

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::decomposed_semi_structured_topology() const noexcept
    -> SP<const Meshes::DecomposedSemiStructuredTopology>
{
    return topology().decomposed_semi_structured_topology();
}

template<TpetraTypePack Pack>
inline size_t MeshHandle<Pack>::num_owned_cells() const noexcept
{
    return topology().num_owned_cells();
}

template<TpetraTypePack Pack>
inline size_t MeshHandle<Pack>::num_local_cells() const noexcept
{
    return topology().num_local_cells();
}

template<TpetraTypePack Pack>
inline size_t MeshHandle<Pack>::num_owned_faces() const noexcept
{
    return topology().num_owned_faces();
}

template<TpetraTypePack Pack>
inline size_t MeshHandle<Pack>::num_faces() const noexcept
{
    return topology().num_faces();
}

template<TpetraTypePack Pack>
inline bool MeshHandle<Pack>::is_owned_cell(local_ordinal_type cell_lid) const
{
    return topology().is_owned_cell(cell_lid);
}

template<TpetraTypePack Pack>
inline bool MeshHandle<Pack>::is_owned_face(local_ordinal_type face_lid) const
{
    return topology().is_owned_face(face_lid);
}

template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::global_ordinal_type
MeshHandle<Pack>::cell_global_id(local_ordinal_type cell_lid) const
{
    return topology().cell_global_id(cell_lid);
}

template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::global_ordinal_type
MeshHandle<Pack>::cell_geometry_global_id(local_ordinal_type cell_lid) const
{
    return topology().cell_geometry_global_id(cell_lid);
}

template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::global_ordinal_type
MeshHandle<Pack>::face_global_id(local_ordinal_type face_lid) const
{
    return topology().face_global_id(face_lid);
}

template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::global_ordinal_type
MeshHandle<Pack>::face_geometry_global_id(local_ordinal_type face_lid) const
{
    return topology().face_geometry_global_id(face_lid);
}

template<TpetraTypePack Pack>
inline bool MeshHandle<Pack>::supports_region_execution() const noexcept
{
    return topology().supports_region_execution();
}

template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::region_cell_local_id(MultiRegion::ID cell) const
{
    return topology().region_cell_local_id(cell);
}

template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::region_face_local_id(MultiRegion::ID face) const
{
    return topology().region_face_local_id(face);
}

template<TpetraTypePack Pack>
inline bool MeshHandle<Pack>::has_reordered_cells() const noexcept
{
    return topology().has_reordered_cells();
}

template<TpetraTypePack Pack>
inline const std::string& MeshHandle<Pack>::boundary_batch_name(int batch_id) const
{
    return topology().boundary_batch_name(batch_id);
}

template<TpetraTypePack Pack>
inline const typename MeshHandle<Pack>::BoundaryFaceBatch&
MeshHandle<Pack>::boundary_face_batch(int batch_id) const
{
    return topology().boundary_face_batch(batch_id);
}

template<TpetraTypePack Pack>
inline const std::unordered_map<int, typename MeshHandle<Pack>::BoundaryFaceBatch>&
MeshHandle<Pack>::boundary_batches() const noexcept
{
    return topology().boundary_batches();
}

template<TpetraTypePack Pack>
inline Teuchos::RCP<const typename MeshHandle<Pack>::map_type> MeshHandle<Pack>::owned_cell_map() const
{
    return topology().owned_cell_map();
}

template<TpetraTypePack Pack>
inline Teuchos::RCP<const typename MeshHandle<Pack>::map_type> MeshHandle<Pack>::overlap_cell_map() const
{
    return topology().overlap_cell_map();
}

template<TpetraTypePack Pack>
inline Teuchos::RCP<const typename MeshHandle<Pack>::map_type> MeshHandle<Pack>::owned_face_map() const
{
    return topology().owned_face_map();
}

template<TpetraTypePack Pack>
inline Teuchos::RCP<const typename MeshHandle<Pack>::map_type> MeshHandle<Pack>::overlap_face_map() const
{
    return topology().overlap_face_map();
}

template<TpetraTypePack Pack>
inline Teuchos::RCP<const typename MeshHandle<Pack>::map_type> MeshHandle<Pack>::boundary_face_map() const
{
    return topology().boundary_face_map();
}

template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::owner_cell(local_ordinal_type face_lid) const
{
    return topology().owner_cell(face_lid);
}

template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::neighbor_cell(local_ordinal_type face_lid) const
{
    return topology().neighbor_cell(face_lid);
}

template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::opposite_cell(local_ordinal_type face_lid, local_ordinal_type cell_lid) const
{
    return topology().opposite_cell(face_lid, cell_lid);
}

template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::opposite_or_periodic_neighbor_cell(local_ordinal_type face_lid,
                                                     local_ordinal_type cell_lid) const
{
    return topology().opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
}

template<TpetraTypePack Pack>
inline bool MeshHandle<Pack>::is_geometry_exterior_face(local_ordinal_type face_lid) const
{
    return topology().is_geometry_exterior_face(face_lid);
}

template<TpetraTypePack Pack>
inline bool MeshHandle<Pack>::is_exterior_face(local_ordinal_type face_lid) const
{
    return topology().is_exterior_face(face_lid);
}

template<TpetraTypePack Pack>
inline bool MeshHandle<Pack>::is_interior_face(local_ordinal_type face_lid) const
{
    return topology().is_interior_face(face_lid);
}

template<TpetraTypePack Pack>
inline int MeshHandle<Pack>::boundary_id(local_ordinal_type face_lid) const
{
    return topology().boundary_id(face_lid);
}

template<TpetraTypePack Pack>
inline bool MeshHandle<Pack>::is_boundary_face(local_ordinal_type face_lid) const
{
    return topology().is_boundary_face(face_lid);
}

/// @}

/// @name Geometry forwarders
/// @{

template<TpetraTypePack Pack>
inline const typename MeshHandle<Pack>::variant_type& MeshHandle<Pack>::variant() const noexcept
{
    return geometry().variant();
}

template<TpetraTypePack Pack>
inline const void* MeshHandle<Pack>::geometry_identity() const noexcept
{
    return geometry().geometry_identity();
}

template<TpetraTypePack Pack>
template<class Visitor>
inline decltype(auto) MeshHandle<Pack>::visit(Visitor&& visitor) const
{
    return geometry().visit(std::forward<Visitor>(visitor));
}

template<TpetraTypePack Pack>
template<class Visitor>
inline decltype(auto) MeshHandle<Pack>::visit_mutable(Visitor&& visitor)
{
    return geometry().visit_mutable(std::forward<Visitor>(visitor));
}

template<TpetraTypePack Pack>
inline bool MeshHandle<Pack>::has_mutable_geometry() const noexcept
{
    return geometry().has_mutable_geometry();
}

template<TpetraTypePack Pack>
inline std::uint64_t MeshHandle<Pack>::geometry_epoch() const
{
    return geometry().geometry_epoch();
}

template<TpetraTypePack Pack>
inline real_t MeshHandle<Pack>::cell_volume(local_ordinal_type cell_lid) const
{
    return geometry().cell_volume(cell_lid);
}

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::cell_centroid(local_ordinal_type cell_lid) const -> Vec3
{
    return geometry().cell_centroid(cell_lid);
}

template<TpetraTypePack Pack>
inline real_t MeshHandle<Pack>::face_area(local_ordinal_type face_lid) const
{
    return geometry().face_area(face_lid);
}

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::face_centroid(local_ordinal_type face_lid) const -> Vec3
{
    return geometry().face_centroid(face_lid);
}

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::face_normal(local_ordinal_type face_lid) const -> Vec3
{
    return geometry().face_normal(face_lid);
}

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::face_area_vector(local_ordinal_type face_lid) const -> Vec3
{
    return geometry().face_area_vector(face_lid);
}

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::face_normal_outward(local_ordinal_type face_lid,
                                                  local_ordinal_type cell_lid) const -> Vec3
{
    return geometry().face_normal_outward(face_lid, cell_lid);
}

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::face_area_vector_outward(local_ordinal_type face_lid,
                                                       local_ordinal_type cell_lid) const -> Vec3
{
    return geometry().face_area_vector_outward(face_lid, cell_lid);
}

template<TpetraTypePack Pack>
inline real_t MeshHandle<Pack>::face_cell_center_distance(local_ordinal_type face_lid) const
{
    return geometry().face_cell_center_distance(face_lid);
}

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::face_center_vector(local_ordinal_type face_lid,
                                                 local_ordinal_type cell_lid) const -> Vec3
{
    return geometry().face_center_vector(face_lid, cell_lid);
}

template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::cell_center_vector(local_ordinal_type face_lid,
                                                 local_ordinal_type cell_lid) const -> Vec3
{
    return geometry().cell_center_vector(face_lid, cell_lid);
}

template<TpetraTypePack Pack>
inline real_t MeshHandle<Pack>::cell_to_face_distance(local_ordinal_type face_lid,
                                                      local_ordinal_type cell_lid) const
{
    return geometry().cell_to_face_distance(face_lid, cell_lid);
}

/// @}

/// @name Per-cell observation helpers
/// @{

/// @brief Observe cell-face IDs without persistent connectivity or scratch storage.
/// @param cell_lid Local cell ordinal, including supported halo cells.
/// @return Independent sized/indexable range; the handle must remain alive.
template<TpetraTypePack Pack>
inline auto MeshHandle<Pack>::faces(local_ordinal_type cell_lid) const -> CellFaceRange
{
    check_cell(cell_lid);
    const auto geometry = geometry_cell_lid(cell_lid);
    const size_t count = visit([&](const auto& mesh) -> size_t
    {
        const auto& native = mesh.faces(mesh.cell_id(static_cast<size_t>(geometry)));
        size_t result = 0;
        for (const auto face : native)
            result += geometry_to_local_face(mesh.face_local_id(face)) != invalid_local_id();
        return result;
    });
    return {this, static_cast<size_t>(cell_lid), count,
        [](const void* source, size_t cell, size_t entry) -> local_ordinal_type
        {
            const auto& handle = *static_cast<const MeshHandle*>(source);
            const auto geometry = handle.geometry_cell_lid(checked_local(cell));
            return handle.visit([&](const auto& mesh) -> local_ordinal_type
            {
                const auto& native = mesh.faces(mesh.cell_id(static_cast<size_t>(geometry)));
                for (const auto face : native)
                {
                    const auto local = handle.geometry_to_local_face(mesh.face_local_id(face));
                    if (local != invalid_local_id() && entry-- == 0) return local;
                }
                throw std::out_of_range("Cell-face range index out of bounds.");
            });
        }};
}

template<TpetraTypePack Pack>
template<class Visitor>
inline void MeshHandle<Pack>::visit_cell_faces(
    local_ordinal_type cell_lid, Visitor&& visitor) const
{
    const auto execution = acquire_execution_view();
    const auto geometry = geometry_cell_lid(cell_lid);
    visit([&](const auto& mesh)
    {
        const auto cell = mesh.cell_id(static_cast<size_t>(geometry));
        const auto to_local_cell = [&](auto native_cell) -> local_ordinal_type
        {
            using mesh_type = std::remove_cvref_t<decltype(mesh)>;
            if constexpr (std::is_same_v<mesh_type, STKAdapter>)
            {
                if (native_cell == invalid_id<local_ordinal_type>()) return invalid_local_id();
            }
            else if (native_cell == mesh_type::invalid_cell_id()) return invalid_local_id();
            return geometry_to_local_cell(mesh.cell_local_id(native_cell));
        };
        for (const auto face : mesh.faces(cell))
        {
            const auto local = geometry_to_local_face(mesh.face_local_id(face));
            if (local == invalid_local_id()) continue;
            if constexpr (std::is_invocable_v<Visitor&, local_ordinal_type,
                              decltype(face), decltype(cell), decltype(mesh), decltype(to_local_cell)>)
                visitor(local, face, cell, mesh, to_local_cell);
            else if constexpr (std::is_invocable_v<Visitor&, local_ordinal_type,
                              decltype(face), decltype(cell), decltype(mesh)>)
                visitor(local, face, cell, mesh);
            else
                visitor(local);
        }
    });
}

/// @}

/// @name Shared topology helpers
/// @{

/// @brief Returns the local id of the owner or neighbor cell adjacent to a face.
/// @param face_lid  Local index of the query face.
/// @param owner     If `true`, returns the owner cell; if `false`, returns the
///                  neighbor cell.
/// @return The local cell id, or `invalid_local_id()` if the cell does not
///         exist in the local partition or the face is on the boundary.
/// @note Both borrowed views and MeshHandle::owner_cell()/neighbor_cell() route
///       through this shared completion of the two-step lookup.
template<TpetraTypePack Pack>
inline typename MeshHandle<Pack>::local_ordinal_type
MeshHandle<Pack>::adjacent_cell(local_ordinal_type face_lid, bool owner) const
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
inline decltype(auto)
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
inline decltype(auto)
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

//----------------------------- borrowed view definitions ---------------------------------//
// The views store only a back-pointer and the validated binding snapshot. Their
// inline bodies intentionally follow the MeshHandle class definition so this
// fragment remains the single include point for MeshHandle consumers.
#include "geometry/MeshTopologyHandle.ipp"
#include "geometry/MeshGeometryHandle.ipp"

namespace SimpleFluid
{

template<TpetraTypePack Pack>
inline Meshes::MultiRegionMesh::ExecutionView MeshHandle<Pack>::acquire_execution_view() const
{
    return geometry().acquire_execution_view();
}

} // namespace SimpleFluid
