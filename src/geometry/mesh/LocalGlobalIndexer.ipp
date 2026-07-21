/**
 * @file LocalGlobalIndexer.ipp
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Inline method definitions for LocalGlobalIndexer.
 * @version 0.1
 * @date 2026-06-21
 *
 * @details This file contains short accessors and forwarding operations that
 * remain visible to every translation unit including LocalGlobalIndexer.hh.
 * Construction, validation, and mapping-table population are defined in
 * LocalGlobalIndexer.tcc.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

namespace SimpleFluid::Meshes
{

/**
 * @brief Return the sentinel used for unavailable local entities.
 * @tparam Pack Mesh entity-ID and ordinal types.
 * @return Minus one for signed ordinals, otherwise the maximum value.
 */
template<MeshIndexTypePack Pack>
constexpr typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::invalid_local_id() noexcept
{
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        return static_cast<local_ordinal_type>(-1);
    }
    return std::numeric_limits<local_ordinal_type>::max();
}

/**
 * @brief Validate a local ordinal against an entity index.
 * @tparam Pack Mesh entity-ID and ordinal types.
 * @tparam LocalID Rank-local entity identifier type.
 * @tparam GlobalID Global entity identifier type.
 * @param index Entity index supplying the local count.
 * @param local_id Local ordinal to validate.
 * @param entity Entity label used in diagnostics.
 * @throws std::out_of_range If @p local_id is invalid.
 */
template<MeshIndexTypePack Pack>
template<class LocalID, class GlobalID>
inline void LocalGlobalIndexer<Pack>::check_local(
    const EntityIndex<LocalID, GlobalID>& index,
    local_ordinal_type local_id,
    std::string_view entity)
{
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        if (local_id < 0)
        {
            throw std::out_of_range(
                "Local/global indexer " + std::string(entity)
                + " local ID is out of range.");
        }
    }
    if (static_cast<size_t>(local_id) >= index.global_ids.size())
    {
        throw std::out_of_range(
            "Local/global indexer " + std::string(entity)
            + " local ID is out of range.");
    }
}

/**
 * @brief Resolve a validated local ordinal to its global ID.
 * @tparam Pack Mesh entity-ID and ordinal types.
 * @tparam LocalID Rank-local entity identifier type.
 * @tparam GlobalID Global entity identifier type.
 * @param index Entity index to query.
 * @param local_id Local ordinal.
 * @param entity Entity label used in diagnostics.
 * @return Global entity ID.
 * @throws std::out_of_range If @p local_id is invalid.
 */
template<MeshIndexTypePack Pack>
template<class LocalID, class GlobalID>
inline GlobalID
LocalGlobalIndexer<Pack>::global_id(
    const EntityIndex<LocalID, GlobalID>& index,
    local_ordinal_type local_id,
    std::string_view entity)
{
    check_local(index, local_id, entity);
    return index.global_ids[static_cast<size_t>(local_id)];
}

/**
 * @brief Find the local ordinal associated with a global ID.
 * @tparam Pack Mesh entity-ID and ordinal types.
 * @tparam LocalID Rank-local entity identifier type.
 * @tparam GlobalID Global entity identifier type.
 * @param index Entity index to query.
 * @param global_id Global ID to locate.
 * @return Local ordinal, or @ref invalid_local_id when unavailable.
 */
template<MeshIndexTypePack Pack>
template<class LocalID, class GlobalID>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::local_id(
    const EntityIndex<LocalID, GlobalID>& index,
    const GlobalID& global_id) noexcept
{
    const auto iter = index.ordinal_by_global_id.find(global_id);
    return iter == index.ordinal_by_global_id.end()
         ? invalid_local_id()
         : iter->second;
}

/**
 * @brief Resolve an explicit rank-local ID to its global ID.
 * @tparam Pack Mesh entity-ID and ordinal types.
 * @tparam LocalID Rank-local entity identifier type.
 * @tparam GlobalID Global entity identifier type.
 * @param index Entity index to query.
 * @param local_id Rank-local entity ID.
 * @param entity Entity label used in diagnostics.
 * @return Global entity ID.
 * @throws std::out_of_range If @p local_id is unavailable.
 */
template<MeshIndexTypePack Pack>
template<class LocalID, class GlobalID>
inline GlobalID
LocalGlobalIndexer<Pack>::mapped_global_id(
    const EntityIndex<LocalID, GlobalID>& index,
    const LocalID& local_id,
    std::string_view entity)
{
    const auto iter = index.ordinal_by_local_id.find(local_id);
    if (iter == index.ordinal_by_local_id.end())
    {
        throw std::out_of_range(
            "Local/global indexer " + std::string(entity)
            + " local ID is not available.");
    }
    return index.global_ids[static_cast<size_t>(iter->second)];
}

/**
 * @brief Resolve a global ID to its explicit rank-local entity ID.
 * @tparam Pack Mesh entity-ID and ordinal types.
 * @tparam LocalID Rank-local entity identifier type.
 * @tparam GlobalID Global entity identifier type.
 * @param index Entity index to query.
 * @param global_id Global entity ID.
 * @param entity Entity label used in diagnostics.
 * @return Rank-local entity ID.
 * @throws std::out_of_range If @p global_id is unavailable locally.
 */
template<MeshIndexTypePack Pack>
template<class LocalID, class GlobalID>
inline LocalID
LocalGlobalIndexer<Pack>::mapped_local_id(
    const EntityIndex<LocalID, GlobalID>& index,
    const GlobalID& global_id,
    std::string_view entity)
{
    const auto ordinal = local_id(index, global_id);
    if (ordinal == invalid_local_id())
    {
        throw std::out_of_range(
            "Local/global indexer " + std::string(entity)
            + " global ID is not available locally.");
    }
    return index.local_ids[static_cast<size_t>(ordinal)];
}

/**
 * @brief Resolve a local ordinal to its global ordinal.
 * @tparam Pack Mesh entity-ID and ordinal types.
 * @tparam LocalID Rank-local entity identifier type.
 * @tparam GlobalID Global entity identifier type.
 * @param index Entity index to query.
 * @param local_ordinal Local ordinal.
 * @param entity Entity label used in diagnostics.
 * @return Global ordinal.
 * @throws std::out_of_range If @p local_ordinal is invalid.
 */
template<MeshIndexTypePack Pack>
template<class LocalID, class GlobalID>
inline typename LocalGlobalIndexer<Pack>::global_ordinal_type
LocalGlobalIndexer<Pack>::global_ordinal(
    const EntityIndex<LocalID, GlobalID>& index,
    local_ordinal_type local_ordinal,
    std::string_view entity)
{
    check_local(index, local_ordinal, entity);
    return index.global_ordinals[
        static_cast<size_t>(local_ordinal)];
}

/**
 * @brief Find the local ordinal associated with a global ordinal.
 * @tparam Pack Mesh entity-ID and ordinal types.
 * @tparam LocalID Rank-local entity identifier type.
 * @tparam GlobalID Global entity identifier type.
 * @param index Entity index to query.
 * @param global_ordinal Global ordinal to locate.
 * @return Local ordinal, or @ref invalid_local_id when unavailable.
 */
template<MeshIndexTypePack Pack>
template<class LocalID, class GlobalID>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::local_ordinal(
    const EntityIndex<LocalID, GlobalID>& index,
    global_ordinal_type global_ordinal) noexcept
{
    const auto iter =
        index.local_by_global_ordinal.find(global_ordinal);
    return iter == index.local_by_global_ordinal.end()
         ? invalid_local_id()
         : iter->second;
}

template<MeshIndexTypePack Pack>
inline size_t LocalGlobalIndexer<Pack>::num_owned_cells() const noexcept
{
    return d_cells.owned_count;
}

template<MeshIndexTypePack Pack>
inline size_t LocalGlobalIndexer<Pack>::num_local_cells() const noexcept
{
    return d_cells.global_ids.size();
}

template<MeshIndexTypePack Pack>
inline size_t LocalGlobalIndexer<Pack>::num_owned_faces() const noexcept
{
    return d_faces.owned_count;
}

template<MeshIndexTypePack Pack>
inline size_t LocalGlobalIndexer<Pack>::num_local_faces() const noexcept
{
    return d_faces.global_ids.size();
}

template<MeshIndexTypePack Pack>
inline size_t LocalGlobalIndexer<Pack>::num_owned_nodes() const noexcept
{
    return d_nodes.owned_count;
}

template<MeshIndexTypePack Pack>
inline size_t LocalGlobalIndexer<Pack>::num_local_nodes() const noexcept
{
    return d_nodes.global_ids.size();
}

template<MeshIndexTypePack Pack>
inline bool LocalGlobalIndexer<Pack>::is_owned_cell(
    local_ordinal_type local_id) const
{
    check_local(d_cells, local_id, "cell");
    return static_cast<size_t>(local_id) < d_cells.owned_count;
}

template<MeshIndexTypePack Pack>
inline bool LocalGlobalIndexer<Pack>::is_owned_face(
    local_ordinal_type local_id) const
{
    check_local(d_faces, local_id, "face");
    return static_cast<size_t>(local_id) < d_faces.owned_count;
}

template<MeshIndexTypePack Pack>
inline bool LocalGlobalIndexer<Pack>::is_owned_node(
    local_ordinal_type local_id) const
{
    check_local(d_nodes, local_id, "node");
    return static_cast<size_t>(local_id) < d_nodes.owned_count;
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::cell_id_t
LocalGlobalIndexer<Pack>::cell_id(ordinal_t ordinal) const
{
    return global_id(d_cells, ordinal, "cell");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::face_id_t
LocalGlobalIndexer<Pack>::face_id(ordinal_t ordinal) const
{
    return global_id(d_faces, ordinal, "face");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::node_id_t
LocalGlobalIndexer<Pack>::node_id(ordinal_t ordinal) const
{
    return global_id(d_nodes, ordinal, "node");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::ordinal_t
LocalGlobalIndexer<Pack>::cell_ordinal(const cell_id_t& id) const noexcept
{
    return local_id(d_cells, id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::ordinal_t
LocalGlobalIndexer<Pack>::face_ordinal(const face_id_t& id) const noexcept
{
    return local_id(d_faces, id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::ordinal_t
LocalGlobalIndexer<Pack>::node_ordinal(const node_id_t& id) const noexcept
{
    return local_id(d_nodes, id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::global_cell_id_t
LocalGlobalIndexer<Pack>::local_to_global_cell_id(
    const local_cell_id_t& local_id) const
{
    return mapped_global_id(d_cells, local_id, "cell");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::global_face_id_t
LocalGlobalIndexer<Pack>::local_to_global_face_id(
    const local_face_id_t& local_id) const
{
    return mapped_global_id(d_faces, local_id, "face");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::global_node_id_t
LocalGlobalIndexer<Pack>::local_to_global_node_id(
    const local_node_id_t& local_id) const
{
    return mapped_global_id(d_nodes, local_id, "node");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_cell_id_t
LocalGlobalIndexer<Pack>::global_to_local_cell_id(
    const global_cell_id_t& global_id) const
{
    return mapped_local_id(d_cells, global_id, "cell");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_face_id_t
LocalGlobalIndexer<Pack>::global_to_local_face_id(
    const global_face_id_t& global_id) const
{
    return mapped_local_id(d_faces, global_id, "face");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_node_id_t
LocalGlobalIndexer<Pack>::global_to_local_node_id(
    const global_node_id_t& global_id) const
{
    return mapped_local_id(d_nodes, global_id, "node");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::global_ordinal_type
LocalGlobalIndexer<Pack>::local_to_global_cell_ordinal(
    local_ordinal_type local_ordinal) const
{
    return global_ordinal(d_cells, local_ordinal, "cell");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::global_ordinal_type
LocalGlobalIndexer<Pack>::local_to_global_face_ordinal(
    local_ordinal_type local_ordinal) const
{
    return global_ordinal(d_faces, local_ordinal, "face");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::global_ordinal_type
LocalGlobalIndexer<Pack>::local_to_global_node_ordinal(
    local_ordinal_type local_ordinal) const
{
    return global_ordinal(d_nodes, local_ordinal, "node");
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_cell_ordinal(
    global_ordinal_type global_ordinal) const noexcept
{
    return local_ordinal(d_cells, global_ordinal);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_face_ordinal(
    global_ordinal_type global_ordinal) const noexcept
{
    return local_ordinal(d_faces, global_ordinal);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_node_ordinal(
    global_ordinal_type global_ordinal) const noexcept
{
    return local_ordinal(d_nodes, global_ordinal);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::cell_id_t
LocalGlobalIndexer<Pack>::cell_global_id(local_ordinal_type local_id) const
{
    return cell_id(local_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::face_id_t
LocalGlobalIndexer<Pack>::face_global_id(local_ordinal_type local_id) const
{
    return face_id(local_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::node_id_t
LocalGlobalIndexer<Pack>::node_global_id(local_ordinal_type local_id) const
{
    return node_id(local_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::cell_local_id(const cell_id_t& global_id) const noexcept
{
    return cell_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::face_local_id(const face_id_t& global_id) const noexcept
{
    return face_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::node_local_id(const node_id_t& global_id) const noexcept
{
    return node_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::cell_id_t
LocalGlobalIndexer<Pack>::local_to_global_cell(
    local_ordinal_type local_id) const
{
    return cell_id(local_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::face_id_t
LocalGlobalIndexer<Pack>::local_to_global_face(
    local_ordinal_type local_id) const
{
    return face_id(local_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::node_id_t
LocalGlobalIndexer<Pack>::local_to_global_node(
    local_ordinal_type local_id) const
{
    return node_id(local_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_cell(
    const cell_id_t& global_id) const noexcept
{
    return cell_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_face(
    const face_id_t& global_id) const noexcept
{
    return face_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_node(
    const node_id_t& global_id) const noexcept
{
    return node_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
inline const std::vector<typename LocalGlobalIndexer<Pack>::cell_id_t>&
LocalGlobalIndexer<Pack>::cell_global_ids() const noexcept
{
    return d_cells.global_ids;
}

template<MeshIndexTypePack Pack>
inline const std::vector<typename LocalGlobalIndexer<Pack>::face_id_t>&
LocalGlobalIndexer<Pack>::face_global_ids() const noexcept
{
    return d_faces.global_ids;
}

template<MeshIndexTypePack Pack>
inline const std::vector<typename LocalGlobalIndexer<Pack>::node_id_t>&
LocalGlobalIndexer<Pack>::node_global_ids() const noexcept
{
    return d_nodes.global_ids;
}

template<MeshIndexTypePack Pack>
inline std::span<const typename LocalGlobalIndexer<Pack>::cell_id_t>
LocalGlobalIndexer<Pack>::owned_cell_global_ids() const noexcept
{
    return std::span<const cell_id_t>(d_cells.global_ids)
        .first(d_cells.owned_count);
}

template<MeshIndexTypePack Pack>
inline std::span<const typename LocalGlobalIndexer<Pack>::cell_id_t>
LocalGlobalIndexer<Pack>::ghost_cell_global_ids() const noexcept
{
    return std::span<const cell_id_t>(d_cells.global_ids)
        .subspan(d_cells.owned_count);
}

template<MeshIndexTypePack Pack>
inline std::span<const typename LocalGlobalIndexer<Pack>::face_id_t>
LocalGlobalIndexer<Pack>::owned_face_global_ids() const noexcept
{
    return std::span<const face_id_t>(d_faces.global_ids)
        .first(d_faces.owned_count);
}

template<MeshIndexTypePack Pack>
inline std::span<const typename LocalGlobalIndexer<Pack>::face_id_t>
LocalGlobalIndexer<Pack>::overlap_face_global_ids() const noexcept
{
    return std::span<const face_id_t>(d_faces.global_ids)
        .subspan(d_faces.owned_count);
}

template<MeshIndexTypePack Pack>
inline std::span<const typename LocalGlobalIndexer<Pack>::node_id_t>
LocalGlobalIndexer<Pack>::owned_node_global_ids() const noexcept
{
    return std::span<const node_id_t>(d_nodes.global_ids)
        .first(d_nodes.owned_count);
}

template<MeshIndexTypePack Pack>
inline std::span<const typename LocalGlobalIndexer<Pack>::node_id_t>
LocalGlobalIndexer<Pack>::overlap_node_global_ids() const noexcept
{
    return std::span<const node_id_t>(d_nodes.global_ids)
        .subspan(d_nodes.owned_count);
}

} // namespace SimpleFluid::Meshes
