/**
 * @file UnstructuredMesh.ipp
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Inline entity-query implementations for UnstructuredMesh.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

namespace SimpleFluid::Meshes
{

/**
 * @brief Validate a compact cell ID.
 * @param id Cell ID to validate.
 * @throws std::out_of_range If @p id is outside local cells.
 */
inline void UnstructuredMesh::check_cell_id(CellID id) const
{
    if (id >= d_cells.size())
    {
        throw std::out_of_range(
            "UnstructuredMesh cell ID is out of range.");
    }
}

/**
 * @brief Validate a compact face ID.
 * @param id Face ID to validate.
 * @throws std::out_of_range If @p id is outside local faces.
 */
inline void UnstructuredMesh::check_face_id(FaceID id) const
{
    if (id >= d_faces.size())
    {
        throw std::out_of_range(
            "UnstructuredMesh face ID is out of range.");
    }
}

/**
 * @brief Validate a compact node ID.
 * @param id Node ID to validate.
 * @throws std::out_of_range If @p id is outside local nodes.
 */
inline void UnstructuredMesh::check_node_id(NodeID id) const
{
    if (id >= d_nodes.size())
    {
        throw std::out_of_range(
            "UnstructuredMesh node ID is out of range.");
    }
}

/**
 * @brief Return ordered node connectivity for a cell.
 * @param id Cell ID.
 * @return Cell node IDs.
 * @throws std::out_of_range If @p id is invalid.
 */
inline const Arr<UnstructuredMesh::NodeID>&
UnstructuredMesh::cell_nodes(CellID id) const
{
    check_cell_id(id);
    return d_cells[id].node_ids;
}

/**
 * @brief Return ordered node connectivity for a face.
 * @param id Face ID.
 * @return Face node IDs.
 * @throws std::out_of_range If @p id is invalid.
 */
inline const Arr<UnstructuredMesh::NodeID>&
UnstructuredMesh::face_nodes(FaceID id) const
{
    check_face_id(id);
    return d_faces[id].node_ids;
}

/**
 * @brief Return the topology type of a cell.
 * @param id Cell ID.
 * @return Cell topology type.
 * @throws std::out_of_range If @p id is invalid.
 */
inline auto UnstructuredMesh::cell_type(CellID id) const -> CellType
{
    check_cell_id(id);
    return d_cells[id].type;
}

inline real_t UnstructuredMesh::cell_volume_impl(CellID id) const
{
    return d_cells[id].volume;
}

inline auto UnstructuredMesh::cell_centroid_impl(CellID id) const
    -> Vec3
{
    return d_cells[id].center;
}

inline const Arr<UnstructuredMesh::FaceID>&
UnstructuredMesh::cell_faces_impl(CellID id) const
{
    return d_cells[id].face_ids;
}

inline auto UnstructuredMesh::owner_cell_impl(FaceID id) const
    -> CellID
{
    return d_faces[id].owner;
}

inline auto UnstructuredMesh::neighbor_cell_impl(FaceID id) const
    -> CellID
{
    return d_faces[id].neighbor;
}

inline real_t UnstructuredMesh::face_area_impl(FaceID id) const
{
    return d_faces[id].area;
}

inline auto UnstructuredMesh::face_centroid_impl(FaceID id) const
    -> Vec3
{
    return d_faces[id].center;
}

inline auto UnstructuredMesh::face_normal_impl(FaceID id) const
    -> Vec3
{
    return d_faces[id].normal;
}

inline auto UnstructuredMesh::node_coordinates_impl(NodeID id) const
    -> Vec3
{
    return d_nodes[id];
}

inline int UnstructuredMesh::boundary_id_impl(FaceID id) const
{
    return d_faces[id].boundary_id;
}

/**
 * @brief Return the name associated with a boundary batch.
 * @param batch_id Boundary batch ID.
 * @return Boundary name.
 * @throws std::out_of_range If @p batch_id is unavailable.
 */
inline const std::string&
UnstructuredMesh::boundary_batch_name_impl(int batch_id) const
{
    const auto iter = d_boundary_names.find(batch_id);
    if (iter == d_boundary_names.end())
    {
        throw std::out_of_range(
            "Requested boundary batch is not found.");
    }
    return iter->second;
}

/**
 * @brief Return locally visible faces in a boundary batch.
 * @param batch_id Boundary batch ID.
 * @return Boundary-face batch.
 * @throws std::out_of_range If @p batch_id is unavailable.
 */
inline const UnstructuredMesh::BoundaryFaceBatch&
UnstructuredMesh::boundary_face_batch_impl(int batch_id) const
{
    const auto iter = d_boundary_batches.find(batch_id);
    if (iter == d_boundary_batches.end())
    {
        throw std::out_of_range(
            "Requested boundary batch is not found.");
    }
    return iter->second;
}

/**
 * @brief Enumerate available boundary batch IDs.
 * @return Boundary IDs in the map's iteration order.
 */
inline std::vector<int> UnstructuredMesh::boundary_batch_ids_impl() const
{
    std::vector<int> ids;
    ids.reserve(d_boundary_batches.size());
    for (const auto& [id, batch] : d_boundary_batches)
    {
        static_cast<void>(batch);
        ids.push_back(id);
    }
    return ids;
}

inline int UnstructuredMesh::num_boundary_batches_impl() const noexcept
{
    return static_cast<int>(d_boundary_batches.size());
}

} // namespace SimpleFluid::Meshes
