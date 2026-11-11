/**
 * @file MeshTopologyHandle.ipp
 * @brief Inline definitions for the borrowed topology view over MeshHandle.
 *
 * Included by geometry/MeshHandle.ipp after the MeshHandle class definition is
 * visible. The view stores no topology: each query forwards to the owner.
 *
 * @version 0.1
 * @date 2026-11-11
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <algorithm>
#include <variant>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
inline MeshTopologyHandle<Pack>::MeshTopologyHandle(const MeshHandle<Pack>& owner) noexcept
    : d_owner(&owner), d_binding(owner.binding_ptr())
{
}

template<TpetraTypePack Pack>
inline const detail::MeshBinding<Pack>& MeshTopologyHandle<Pack>::binding() const
{
    if (!d_binding)
        throw std::logic_error("MeshTopologyHandle has no published MeshBinding.");
    d_binding->validate(d_owner->d_layout_generation);
    return *d_binding;
}

template<TpetraTypePack Pack>
inline std::uint64_t MeshTopologyHandle<Pack>::layout_generation() const noexcept
{
    return d_owner->d_layout_generation;
}

template<TpetraTypePack Pack>
inline std::size_t MeshTopologyHandle<Pack>::num_owned_cells() const noexcept
{
    if (d_owner->d_orthogonal_topology)
        return d_owner->d_orthogonal_topology->num_owned_cells();
    if (d_owner->d_semi_structured_topology) return d_owner->d_semi_structured_topology->num_owned_cells();
    if (const auto legacy = d_owner->legacy_mesh())
    {
        return legacy->num_owned_cells();
    }
    return d_owner->d_serial_identity
               ? d_owner->d_serial_counts[0]
               : (d_owner->d_map_indexing ? d_owner->d_owned_cell_map->getLocalNumElements()
                                          : d_owner->d_indexer.num_owned_cells());
}

template<TpetraTypePack Pack>
inline std::size_t MeshTopologyHandle<Pack>::num_local_cells() const noexcept
{
    if (d_owner->d_orthogonal_topology)
        return d_owner->d_orthogonal_topology->num_local_cells();
    if (d_owner->d_semi_structured_topology) return d_owner->d_semi_structured_topology->num_local_cells();
    if (const auto legacy = d_owner->legacy_mesh())
    {
        return legacy->num_local_cells();
    }
    return d_owner->d_serial_identity
               ? d_owner->d_serial_counts[0]
               : (d_owner->d_map_indexing ? d_owner->d_overlap_cell_map->getLocalNumElements()
                                          : d_owner->d_indexer.num_local_cells());
}

template<TpetraTypePack Pack>
inline std::size_t MeshTopologyHandle<Pack>::num_cells() const noexcept
{
    return num_local_cells();
}

template<TpetraTypePack Pack>
inline std::size_t MeshTopologyHandle<Pack>::num_owned_faces() const noexcept
{
    if (d_owner->d_orthogonal_topology)
        return d_owner->d_orthogonal_topology->num_owned_faces();
    if (d_owner->d_semi_structured_topology) return d_owner->d_semi_structured_topology->num_owned_faces();
    if (const auto legacy = d_owner->legacy_mesh())
    {
        const auto map = legacy->owned_face_map();
        return map.is_null() ? 0 : map->getLocalNumElements();
    }
    return d_owner->d_serial_identity
               ? d_owner->d_serial_counts[1]
               : (d_owner->d_map_indexing ? d_owner->d_owned_face_map->getLocalNumElements()
                                          : d_owner->d_indexer.num_owned_faces());
}

template<TpetraTypePack Pack>
inline std::size_t MeshTopologyHandle<Pack>::num_faces() const noexcept
{
    if (d_owner->d_orthogonal_topology)
        return d_owner->d_orthogonal_topology->num_local_faces();
    if (d_owner->d_semi_structured_topology) return d_owner->d_semi_structured_topology->num_local_faces();
    if (const auto legacy = d_owner->legacy_mesh())
    {
        return legacy->num_faces();
    }
    return d_owner->d_serial_identity
               ? d_owner->d_serial_counts[1]
               : (d_owner->d_map_indexing ? d_owner->d_overlap_face_map->getLocalNumElements()
                                          : d_owner->d_indexer.num_local_faces());
}

template<TpetraTypePack Pack>
inline bool MeshTopologyHandle<Pack>::is_owned_cell(local_ordinal_type cell_lid) const
{
    d_owner->check_cell(cell_lid);
    if (const auto legacy = d_owner->legacy_mesh())
    {
        return legacy->is_owned_cell(cell_lid);
    }
    return static_cast<std::size_t>(cell_lid) < num_owned_cells();
}

template<TpetraTypePack Pack>
inline bool MeshTopologyHandle<Pack>::is_owned_face(local_ordinal_type face_lid) const
{
    d_owner->check_face(face_lid);
    if (const auto legacy = d_owner->legacy_mesh())
    {
        return legacy->is_owned_face(
            d_owner->checked_local(static_cast<std::size_t>(d_owner->geometry_face_lid(face_lid))));
    }
    return static_cast<std::size_t>(face_lid) < num_owned_faces();
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::global_ordinal_type
MeshTopologyHandle<Pack>::cell_global_id(local_ordinal_type cell_lid) const
{
    d_owner->check_cell(cell_lid);
    if (const auto legacy = d_owner->legacy_mesh())
    {
        const auto geometry_cell =
            d_owner->checked_local(static_cast<std::size_t>(d_owner->geometry_cell_lid(cell_lid)));
        return legacy->mesh_gid_to_tpetra_gid(legacy->cell_global_id(geometry_cell));
    }
    return d_owner->d_serial_identity
               ? static_cast<global_ordinal_type>(cell_lid)
               : (d_owner->d_map_indexing ? d_owner->d_overlap_cell_map->getGlobalElement(cell_lid)
                                          : d_owner->d_indexer.cell_global_id(cell_lid));
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::global_ordinal_type
MeshTopologyHandle<Pack>::cell_geometry_global_id(local_ordinal_type cell_lid) const
{
    d_owner->check_cell(cell_lid);
    if (const auto legacy = d_owner->legacy_mesh())
    {
        return legacy->cell_global_id(
            d_owner->checked_local(static_cast<std::size_t>(d_owner->geometry_cell_lid(cell_lid))));
    }
    if (d_owner->d_semi_structured_topology || d_owner->d_orthogonal_decomposition)
        return d_owner->geometry_cell_lid(cell_lid);
    return d_owner->d_serial_identity
               ? static_cast<global_ordinal_type>(cell_lid)
               : (d_owner->d_map_indexing ? d_owner->d_overlap_cell_map->getGlobalElement(cell_lid)
                                          : d_owner->d_indexer.cell_global_id(cell_lid));
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::global_ordinal_type
MeshTopologyHandle<Pack>::face_global_id(local_ordinal_type face_lid) const
{
    d_owner->check_face(face_lid);
    if (const auto legacy = d_owner->legacy_mesh())
    {
        return legacy->face_global_id(
            d_owner->checked_local(static_cast<std::size_t>(d_owner->geometry_face_lid(face_lid))));
    }
    return d_owner->d_serial_identity
               ? static_cast<global_ordinal_type>(face_lid)
               : (d_owner->d_map_indexing ? d_owner->d_overlap_face_map->getGlobalElement(face_lid)
                                          : d_owner->d_indexer.face_global_id(face_lid));
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::global_ordinal_type
MeshTopologyHandle<Pack>::face_geometry_global_id(local_ordinal_type face_lid) const
{
    d_owner->check_face(face_lid);
    if (const auto legacy = d_owner->legacy_mesh())
        return legacy->face_global_id(
            d_owner->checked_local(static_cast<std::size_t>(d_owner->geometry_face_lid(face_lid))));
    return (d_owner->d_semi_structured_topology || d_owner->d_orthogonal_decomposition)
               ? d_owner->geometry_face_lid(face_lid)
               : face_global_id(face_lid);
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::local_ordinal_type
MeshTopologyHandle<Pack>::owner_cell(local_ordinal_type face_lid) const
{
    return d_owner->adjacent_cell(face_lid, true);
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::local_ordinal_type
MeshTopologyHandle<Pack>::neighbor_cell(local_ordinal_type face_lid) const
{
    return d_owner->adjacent_cell(face_lid, false);
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::local_ordinal_type
MeshTopologyHandle<Pack>::opposite_cell(local_ordinal_type face_lid, local_ordinal_type cell_lid) const
{
    const auto owner = owner_cell(face_lid);
    const auto neighbor = neighbor_cell(face_lid);
    if (cell_lid == owner)
    {
        return neighbor;
    }
    if (neighbor != invalid_local_id() && cell_lid == neighbor)
    {
        return owner;
    }
    throw std::invalid_argument("Cell is not adjacent to requested face.");
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::local_ordinal_type
MeshTopologyHandle<Pack>::opposite_or_periodic_neighbor_cell(local_ordinal_type face_lid,
                                                             local_ordinal_type cell_lid) const
{
    return opposite_cell(face_lid, cell_lid);
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::CellFaceRange
MeshTopologyHandle<Pack>::faces(local_ordinal_type cell_lid) const
{
    return d_owner->faces(cell_lid);
}

template<TpetraTypePack Pack>
template<class Visitor>
inline void MeshTopologyHandle<Pack>::visit_cell_faces(local_ordinal_type cell_lid, Visitor&& visitor) const
{
    d_owner->visit_cell_faces(cell_lid, std::forward<Visitor>(visitor));
}

template<TpetraTypePack Pack>
inline bool MeshTopologyHandle<Pack>::is_geometry_exterior_face(local_ordinal_type face_lid) const
{
    const auto geometry_lid = d_owner->geometry_face_lid(face_lid);
    return std::visit(
        [geometry_lid](const auto& mesh_pointer)
        {
            const auto& mesh = *mesh_pointer;
            const auto id = mesh.face_id(static_cast<std::size_t>(geometry_lid));
            const auto neighbor = mesh.neighbor_cell(id);
            if constexpr (std::is_same_v<std::decay_t<decltype(mesh)>, typename MeshHandle<Pack>::STKAdapter>)
            {
                return neighbor == invalid_id<local_ordinal_type>();
            }
            else
            {
                return neighbor == std::decay_t<decltype(mesh)>::invalid_cell_id();
            }
        },
        d_owner->d_mesh);
}

template<TpetraTypePack Pack>
inline bool MeshTopologyHandle<Pack>::is_exterior_face(local_ordinal_type face_lid) const
{
    return neighbor_cell(face_lid) == invalid_local_id();
}

template<TpetraTypePack Pack>
inline bool MeshTopologyHandle<Pack>::is_interior_face(local_ordinal_type face_lid) const
{
    return !is_exterior_face(face_lid);
}

template<TpetraTypePack Pack>
inline int MeshTopologyHandle<Pack>::boundary_id(local_ordinal_type face_lid) const
{
    return d_owner->visit_geometry_face(
        face_lid,
        [](const auto& mesh, const auto id)
        {
            return mesh.boundary_id(id);
        });
}

template<TpetraTypePack Pack>
inline bool MeshTopologyHandle<Pack>::is_boundary_face(local_ordinal_type face_lid) const
{
    return is_exterior_face(face_lid) && boundary_id(face_lid) != invalid_boundary_id;
}

template<TpetraTypePack Pack>
inline std::vector<int> MeshTopologyHandle<Pack>::boundary_batch_ids() const
{
    std::vector<int> ids;
    ids.reserve(d_owner->d_boundary_batches.size());
    for (const auto& [id, batch] : d_owner->d_boundary_batches)
    {
        static_cast<void>(batch);
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

template<TpetraTypePack Pack>
inline const std::string& MeshTopologyHandle<Pack>::boundary_batch_name(int batch_id) const
{
    const auto iter = d_owner->d_boundary_names.find(batch_id);
    if (iter == d_owner->d_boundary_names.end())
    {
        throw std::out_of_range("Unknown boundary batch ID.");
    }
    return iter->second;
}

template<TpetraTypePack Pack>
inline const typename MeshTopologyHandle<Pack>::BoundaryFaceBatch&
MeshTopologyHandle<Pack>::boundary_face_batch(int batch_id) const
{
    return d_owner->d_boundary_batches.at(batch_id);
}

template<TpetraTypePack Pack>
inline const std::unordered_map<int, typename MeshTopologyHandle<Pack>::BoundaryFaceBatch>&
MeshTopologyHandle<Pack>::boundary_batches() const noexcept
{
    return d_owner->d_boundary_batches;
}

template<TpetraTypePack Pack>
inline Teuchos::RCP<const typename MeshTopologyHandle<Pack>::map_type>
MeshTopologyHandle<Pack>::owned_cell_map() const
{
    return d_owner->d_owned_cell_map;
}

template<TpetraTypePack Pack>
inline Teuchos::RCP<const typename MeshTopologyHandle<Pack>::map_type>
MeshTopologyHandle<Pack>::overlap_cell_map() const
{
    return d_owner->d_overlap_cell_map;
}

template<TpetraTypePack Pack>
inline Teuchos::RCP<const typename MeshTopologyHandle<Pack>::map_type>
MeshTopologyHandle<Pack>::owned_face_map() const
{
    return d_owner->d_owned_face_map;
}

template<TpetraTypePack Pack>
inline Teuchos::RCP<const typename MeshTopologyHandle<Pack>::map_type>
MeshTopologyHandle<Pack>::overlap_face_map() const
{
    return d_owner->d_overlap_face_map;
}

template<TpetraTypePack Pack>
inline Teuchos::RCP<const typename MeshTopologyHandle<Pack>::map_type>
MeshTopologyHandle<Pack>::boundary_face_map() const
{
    return d_owner->d_boundary_face_map;
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::local_ordinal_type
MeshTopologyHandle<Pack>::geometry_to_local_cell(std::size_t geometry_lid) const noexcept
{
    return d_owner->geometry_to_local_cell(geometry_lid);
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::local_ordinal_type
MeshTopologyHandle<Pack>::geometry_to_local_face(std::size_t geometry_lid) const noexcept
{
    return d_owner->geometry_to_local_face(geometry_lid);
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::local_ordinal_type
MeshTopologyHandle<Pack>::region_cell_local_id(std::uint64_t cell) const
{
    if (!std::holds_alternative<typename MeshHandle<Pack>::MultiRegionPtr>(d_owner->d_mesh))
        throw std::logic_error("Resolved region cells require a composite mesh.");
    return d_owner->geometry_to_local_cell(static_cast<std::size_t>(cell));
}

template<TpetraTypePack Pack>
inline typename MeshTopologyHandle<Pack>::local_ordinal_type
MeshTopologyHandle<Pack>::region_face_local_id(std::uint64_t face) const
{
    if (!std::holds_alternative<typename MeshHandle<Pack>::MultiRegionPtr>(d_owner->d_mesh))
        throw std::logic_error("Resolved region faces require a composite mesh.");
    return d_owner->geometry_to_local_face(static_cast<std::size_t>(face));
}

template<TpetraTypePack Pack>
inline SP<const Meshes::OrthogonalMeshDecomposition>
MeshTopologyHandle<Pack>::orthogonal_decomposition() const noexcept
{
    return d_owner->d_orthogonal_decomposition;
}

template<TpetraTypePack Pack>
inline SP<const Meshes::DecomposedOrthogonalTopology>
MeshTopologyHandle<Pack>::decomposed_orthogonal_topology() const noexcept
{
    return d_owner->d_orthogonal_topology;
}

template<TpetraTypePack Pack>
inline const typename MeshTopologyHandle<Pack>::SemiStructuredDecomposition&
MeshTopologyHandle<Pack>::semi_structured_decomposition() const
{
    if (!std::holds_alternative<typename MeshHandle<Pack>::SemiStructuredPtr>(d_owner->d_mesh))
        throw std::invalid_argument("XY/Z decomposition requires a SemiStructuredXY_Z handle.");
    return d_owner->d_semi_structured_decomposition;
}

template<TpetraTypePack Pack>
inline SP<const Meshes::DecomposedSemiStructuredTopology>
MeshTopologyHandle<Pack>::decomposed_semi_structured_topology() const noexcept
{
    return d_owner->d_semi_structured_topology;
}

template<TpetraTypePack Pack>
inline bool MeshTopologyHandle<Pack>::has_reordered_cells() const noexcept
{
    return d_owner->d_cells_reordered;
}

template<TpetraTypePack Pack>
inline bool MeshTopologyHandle<Pack>::supports_region_execution() const noexcept
{
    return std::holds_alternative<typename MeshHandle<Pack>::MultiRegionPtr>(d_owner->d_mesh) &&
           !d_owner->d_cells_reordered && d_owner->d_cell_geometry_lids.empty();
}

} // namespace SimpleFluid
