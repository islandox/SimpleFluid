/**
 * @file PartitionedMeshBase.ipp
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Inline method definitions for PartitionedMesh.
 * @version 0.1
 * @date 2026-06-21
 *
 * @details Contains lightweight geometry, ownership, and map accessors that
 * remain visible through PartitionedMeshBase.hh. Construction and distributed
 * map initialization are defined in PartitionedMeshBase.tcc.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

namespace SimpleFluid::Meshes
{

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline const GeometryMesh&
PartitionedMesh<GeometryMesh, Pack>::mesh() const noexcept
{
    return *d_mesh;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline SP<const GeometryMesh>
PartitionedMesh<GeometryMesh, Pack>::mesh_ptr() const noexcept
{
    return d_mesh;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline const typename PartitionedMesh<GeometryMesh, Pack>::indexer_type&
PartitionedMesh<GeometryMesh, Pack>::indexer() const noexcept
{
    return d_indexer;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_global_cells() const noexcept
{
    return d_global_cell_map->getGlobalNumElements();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_owned_cells() const noexcept
{
    return d_indexer.num_owned_cells();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_local_cells() const noexcept
{
    return d_indexer.num_local_cells();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_cells() const noexcept
{
    return num_local_cells();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_global_faces() const noexcept
{
    return d_global_face_map->getGlobalNumElements();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_owned_faces() const noexcept
{
    return d_indexer.num_owned_faces();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_local_faces() const noexcept
{
    return d_indexer.num_local_faces();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_faces() const noexcept
{
    return num_local_faces();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_global_nodes() const noexcept
{
    return d_global_node_map->getGlobalNumElements();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_owned_nodes() const noexcept
{
    return d_indexer.num_owned_nodes();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_local_nodes() const noexcept
{
    return d_indexer.num_local_nodes();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline size_t
PartitionedMesh<GeometryMesh, Pack>::num_nodes() const noexcept
{
    return num_local_nodes();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::global_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::cell_global_id(
    local_ordinal_type local_id) const
{
    return d_indexer.local_to_global_cell_ordinal(local_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::global_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::face_global_id(
    local_ordinal_type local_id) const
{
    return d_indexer.local_to_global_face_ordinal(local_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::global_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::node_global_id(
    local_ordinal_type local_id) const
{
    return d_indexer.local_to_global_node_ordinal(local_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::cell_local_id(
    const cell_id_t& global_id) const noexcept
{
    return d_indexer.cell_local_id(global_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::face_local_id(
    const face_id_t& global_id) const noexcept
{
    return d_indexer.face_local_id(global_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::node_local_id(
    const node_id_t& global_id) const noexcept
{
    return d_indexer.node_local_id(global_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_local_cell(
    local_ordinal_type local_id) const noexcept
{
    return valid_local(local_id, num_local_cells());
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_local_face(
    local_ordinal_type local_id) const noexcept
{
    return valid_local(local_id, num_local_faces());
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_local_node(
    local_ordinal_type local_id) const noexcept
{
    return valid_local(local_id, num_local_nodes());
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_owned_cell(
    local_ordinal_type local_id) const
{
    return d_indexer.is_owned_cell(local_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_owned_face(
    local_ordinal_type local_id) const
{
    return d_indexer.is_owned_face(local_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_owned_node(
    local_ordinal_type local_id) const
{
    return d_indexer.is_owned_node(local_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_local_cell_id(
    const cell_id_t& global_id) const noexcept
{
    return cell_local_id(global_id) != invalid_local_id();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_local_face_id(
    const face_id_t& global_id) const noexcept
{
    return face_local_id(global_id) != invalid_local_id();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_local_node_id(
    const node_id_t& global_id) const noexcept
{
    return node_local_id(global_id) != invalid_local_id();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_owned_global_cell(
    global_ordinal_type global_id) const
{
    return d_owned_cell_map->isNodeGlobalElement(global_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_owned_global_face(
    global_ordinal_type global_id) const
{
    return d_owned_face_map->isNodeGlobalElement(global_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_owned_global_node(
    global_ordinal_type global_id) const
{
    return d_owned_node_map->isNodeGlobalElement(global_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
constexpr typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::invalid_local_id() noexcept
{
    return indexer_type::invalid_local_id();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline real_t PartitionedMesh<GeometryMesh, Pack>::cell_volume(
    local_ordinal_type local_id) const
{
    return mesh().cell_volume(geometry_cell_id(local_id));
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::Vec3
PartitionedMesh<GeometryMesh, Pack>::cell_centroid(
    local_ordinal_type local_id) const
{
    return mesh().cell_centroid(geometry_cell_id(local_id));
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline real_t PartitionedMesh<GeometryMesh, Pack>::face_area(
    local_ordinal_type local_id) const
{
    return mesh().face_area(geometry_face_id(local_id));
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::Vec3
PartitionedMesh<GeometryMesh, Pack>::face_centroid(
    local_ordinal_type local_id) const
{
    return mesh().face_centroid(geometry_face_id(local_id));
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::Vec3
PartitionedMesh<GeometryMesh, Pack>::face_normal(
    local_ordinal_type local_id) const
{
    return mesh().face_normal(geometry_face_id(local_id));
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_exterior_face(
    local_ordinal_type face_local_id) const
{
    return neighbor_cell(face_local_id) == invalid_local_id();
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_interior_face(
    local_ordinal_type face_local_id) const
{
    return !is_exterior_face(face_local_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::is_boundary_face(
    local_ordinal_type face_local_id) const
{
    return is_exterior_face(face_local_id)
        && boundary_id(face_local_id) != mesh_type::invalid_boundary_id;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline int PartitionedMesh<GeometryMesh, Pack>::boundary_id(
    local_ordinal_type face_local_id) const
{
    return mesh().boundary_id(geometry_face_id(face_local_id));
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline const std::string&
PartitionedMesh<GeometryMesh, Pack>::boundary_batch_name(int batch_id) const
{
    return mesh().boundary_batch_name(batch_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline const typename PartitionedMesh<GeometryMesh, Pack>::BoundaryFaceBatch&
PartitionedMesh<GeometryMesh, Pack>::boundary_face_batch(int batch_id) const
{
    return d_boundary_batches.at(batch_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline const std::unordered_map<int,
    typename PartitionedMesh<GeometryMesh, Pack>::BoundaryFaceBatch>&
PartitionedMesh<GeometryMesh, Pack>::boundary_batches() const noexcept
{
    return d_boundary_batches;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::Vec3
PartitionedMesh<GeometryMesh, Pack>::node_coordinates(
    local_ordinal_type node_local_id) const
{
    return mesh().node_coordinates(geometry_node_id(node_local_id));
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::Vec3
PartitionedMesh<GeometryMesh, Pack>::node_coord(
    local_ordinal_type node_local_id) const
{
    return node_coordinates(node_local_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline Teuchos::RCP<const typename PartitionedMesh<GeometryMesh, Pack>::map_type>
PartitionedMesh<GeometryMesh, Pack>::owned_cell_map() const
{
    return d_owned_cell_map;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline Teuchos::RCP<const typename PartitionedMesh<GeometryMesh, Pack>::map_type>
PartitionedMesh<GeometryMesh, Pack>::overlap_cell_map() const
{
    return d_overlap_cell_map;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline Teuchos::RCP<const typename PartitionedMesh<GeometryMesh, Pack>::map_type>
PartitionedMesh<GeometryMesh, Pack>::owned_face_map() const
{
    return d_owned_face_map;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline Teuchos::RCP<const typename PartitionedMesh<GeometryMesh, Pack>::map_type>
PartitionedMesh<GeometryMesh, Pack>::overlap_face_map() const
{
    return d_overlap_face_map;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline Teuchos::RCP<const typename PartitionedMesh<GeometryMesh, Pack>::map_type>
PartitionedMesh<GeometryMesh, Pack>::boundary_face_map() const
{
    return d_boundary_face_map;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline Teuchos::RCP<const typename PartitionedMesh<GeometryMesh, Pack>::map_type>
PartitionedMesh<GeometryMesh, Pack>::owned_node_map() const
{
    return d_owned_node_map;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline Teuchos::RCP<const typename PartitionedMesh<GeometryMesh, Pack>::map_type>
PartitionedMesh<GeometryMesh, Pack>::overlap_node_map() const
{
    return d_overlap_node_map;
}

// Private helper methods

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline bool PartitionedMesh<GeometryMesh, Pack>::valid_local(
    local_ordinal_type local_id, size_t count)
{
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        if (local_id < 0)
        {
            return false;
        }
    }
    return static_cast<size_t>(local_id) < count;
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::local_ordinal_type
PartitionedMesh<GeometryMesh, Pack>::checked_local(size_t local_id)
{
    if (!std::in_range<local_ordinal_type>(local_id))
    {
        throw std::overflow_error(
            "Partitioned mesh local ordinal overflow.");
    }
    return static_cast<local_ordinal_type>(local_id);
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::cell_id_t
PartitionedMesh<GeometryMesh, Pack>::geometry_cell_id(
    local_ordinal_type local_id) const
{
    if (!is_local_cell(local_id))
    {
        throw std::out_of_range("Cell local ID is out of range.");
    }
    return mesh().cell_id(static_cast<size_t>(local_id));
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::face_id_t
PartitionedMesh<GeometryMesh, Pack>::geometry_face_id(
    local_ordinal_type local_id) const
{
    if (!is_local_face(local_id))
    {
        throw std::out_of_range("Face local ID is out of range.");
    }
    return mesh().face_id(static_cast<size_t>(local_id));
}

template<MeshClass GeometryMesh, TpetraTypePack Pack>
inline typename PartitionedMesh<GeometryMesh, Pack>::node_id_t
PartitionedMesh<GeometryMesh, Pack>::geometry_node_id(
    local_ordinal_type local_id) const
{
    if (!is_local_node(local_id))
    {
        throw std::out_of_range("Node local ID is out of range.");
    }
    return mesh().node_id(static_cast<size_t>(local_id));
}

} // namespace SimpleFluid::Meshes
