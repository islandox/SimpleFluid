/**
 * @file LocalGlobalIndexer.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template method definitions for LocalGlobalIndexer.
 * @version 0.1
 * @date 2026-06-21
 *
 * @details LocalGlobalIndexer.cc includes this file to build the supported
 * explicit instantiations. A translation unit that uses a custom index pack
 * may include this file after LocalGlobalIndexer.hh to instantiate the same
 * implementation for that pack.
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/LocalGlobalIndexer.hh"

namespace SimpleFluid::Meshes
{

template<MeshIndexTypePack Pack>
LocalGlobalIndexer<Pack>::LocalGlobalIndexer(
    std::vector<cell_id_t> owned_cells,
    std::vector<cell_id_t> ghost_cells,
    std::vector<face_id_t> owned_faces,
    std::vector<face_id_t> overlap_faces,
    std::vector<node_id_t> nodes)
{
    set_cells(std::move(owned_cells), std::move(ghost_cells));
    set_faces(std::move(owned_faces), std::move(overlap_faces));
    set_nodes(std::move(nodes));
}

template<MeshIndexTypePack Pack>
LocalGlobalIndexer<Pack>::LocalGlobalIndexer(
    std::vector<cell_id_t> owned_cells,
    std::vector<cell_id_t> ghost_cells,
    std::vector<face_id_t> owned_faces,
    std::vector<face_id_t> overlap_faces,
    std::vector<node_id_t> owned_nodes,
    std::vector<node_id_t> overlap_nodes)
{
    set_cells(std::move(owned_cells), std::move(ghost_cells));
    set_faces(std::move(owned_faces), std::move(overlap_faces));
    set_nodes(std::move(owned_nodes), std::move(overlap_nodes));
}

template<MeshIndexTypePack Pack>
LocalGlobalIndexer<Pack>::LocalGlobalIndexer(
    std::vector<CellMapping> owned_cells,
    std::vector<CellMapping> ghost_cells,
    std::vector<FaceMapping> owned_faces,
    std::vector<FaceMapping> overlap_faces,
    std::vector<NodeMapping> nodes)
{
    set_cells(std::move(owned_cells), std::move(ghost_cells));
    set_faces(std::move(owned_faces), std::move(overlap_faces));
    set_nodes(std::move(nodes));
}

template<MeshIndexTypePack Pack>
LocalGlobalIndexer<Pack>::LocalGlobalIndexer(
    std::vector<CellMapping> owned_cells,
    std::vector<CellMapping> ghost_cells,
    std::vector<FaceMapping> owned_faces,
    std::vector<FaceMapping> overlap_faces,
    std::vector<NodeMapping> owned_nodes,
    std::vector<NodeMapping> overlap_nodes)
{
    set_cells(std::move(owned_cells), std::move(ghost_cells));
    set_faces(std::move(owned_faces), std::move(overlap_faces));
    set_nodes(std::move(owned_nodes), std::move(overlap_nodes));
}

template<MeshIndexTypePack Pack>
void LocalGlobalIndexer<Pack>::set_cells(
    std::vector<cell_id_t> owned,
    std::vector<cell_id_t> ghost)
{
    assign(d_cells, std::move(owned), std::move(ghost), "cell");
}

template<MeshIndexTypePack Pack>
void LocalGlobalIndexer<Pack>::set_faces(
    std::vector<face_id_t> owned,
    std::vector<face_id_t> overlap)
{
    assign(d_faces, std::move(owned), std::move(overlap), "face");
}

template<MeshIndexTypePack Pack>
void LocalGlobalIndexer<Pack>::set_nodes(std::vector<node_id_t> nodes)
{
    assign(d_nodes, std::move(nodes), {}, "node");
}

template<MeshIndexTypePack Pack>
void LocalGlobalIndexer<Pack>::set_nodes(
    std::vector<node_id_t> owned,
    std::vector<node_id_t> overlap)
{
    assign(d_nodes, std::move(owned), std::move(overlap), "node");
}

template<MeshIndexTypePack Pack>
void LocalGlobalIndexer<Pack>::set_cells(
    std::vector<CellMapping> owned,
    std::vector<CellMapping> ghost)
{
    assign_mappings(
        d_cells, std::move(owned), std::move(ghost), "cell");
}

template<MeshIndexTypePack Pack>
void LocalGlobalIndexer<Pack>::set_faces(
    std::vector<FaceMapping> owned,
    std::vector<FaceMapping> overlap)
{
    assign_mappings(
        d_faces, std::move(owned), std::move(overlap), "face");
}

template<MeshIndexTypePack Pack>
void LocalGlobalIndexer<Pack>::set_nodes(std::vector<NodeMapping> nodes)
{
    assign_mappings(d_nodes, std::move(nodes), {}, "node");
}

template<MeshIndexTypePack Pack>
void LocalGlobalIndexer<Pack>::set_nodes(
    std::vector<NodeMapping> owned,
    std::vector<NodeMapping> overlap)
{
    assign_mappings(
        d_nodes, std::move(owned), std::move(overlap), "node");
}

template<MeshIndexTypePack Pack>
template<class LocalID, class GlobalID>
void LocalGlobalIndexer<Pack>::assign(
    EntityIndex<LocalID, GlobalID>& index,
    std::vector<GlobalID> owned,
    std::vector<GlobalID> overlap,
    std::string_view entity)
{
    auto make_mappings = [](const std::vector<GlobalID>& ids,
                            size_t offset)
    {
        std::vector<EntityMapping<LocalID, GlobalID>> mappings;
        mappings.reserve(ids.size());
        for (size_t index = 0; index < ids.size(); ++index)
        {
            const auto ordinal = offset + index;
            mappings.push_back({
                infer_local_id<LocalID>(ids[index], ordinal),
                ids[index],
                infer_global_ordinal(ids[index], ordinal)});
        }
        return mappings;
    };

    const auto owned_count = owned.size();
    assign_mappings(
        index,
        make_mappings(owned, 0),
        make_mappings(overlap, owned_count),
        entity);
}

template<MeshIndexTypePack Pack>
template<class LocalID, class GlobalID>
void LocalGlobalIndexer<Pack>::assign_mappings(
    EntityIndex<LocalID, GlobalID>& index,
    std::vector<EntityMapping<LocalID, GlobalID>> owned,
    std::vector<EntityMapping<LocalID, GlobalID>> overlap,
    std::string_view entity)
{
    const auto count = owned.size() + overlap.size();
    if (count > static_cast<size_t>(
            std::numeric_limits<local_ordinal_type>::max()))
    {
        throw std::overflow_error(
            "Local/global indexer " + std::string(entity)
            + " count exceeds its local ordinal type.");
    }

    EntityIndex<LocalID, GlobalID> rebuilt;
    rebuilt.owned_count = owned.size();
    owned.insert(
        owned.end(), overlap.begin(), overlap.end());
    rebuilt.local_ids.reserve(count);
    rebuilt.global_ids.reserve(count);
    rebuilt.global_ordinals.reserve(count);
    for (size_t local = 0; local < owned.size(); ++local)
    {
        const auto local_ordinal =
            static_cast<local_ordinal_type>(local);
        auto& mapping = owned[local];
        if (!rebuilt.ordinal_by_local_id.emplace(
                mapping.local_id, local_ordinal).second)
        {
            throw std::invalid_argument(
                "Local/global indexer contains duplicate local "
                + std::string(entity) + " ID.");
        }
        if (!rebuilt.ordinal_by_global_id.emplace(
                mapping.global_id, local_ordinal).second)
        {
            throw std::invalid_argument(
                "Local/global indexer contains duplicate global "
                + std::string(entity) + " ID.");
        }
        if (!rebuilt.local_by_global_ordinal.emplace(
                mapping.global_ordinal, local_ordinal).second)
        {
            throw std::invalid_argument(
                "Local/global indexer contains duplicate global "
                + std::string(entity) + " ordinal.");
        }
        rebuilt.local_ids.push_back(std::move(mapping.local_id));
        rebuilt.global_ids.push_back(std::move(mapping.global_id));
        rebuilt.global_ordinals.push_back(mapping.global_ordinal);
    }
    index = std::move(rebuilt);
}

template<MeshIndexTypePack Pack>
template<class LocalID, class GlobalID>
LocalID
LocalGlobalIndexer<Pack>::infer_local_id(
    const GlobalID& global_id,
    size_t local_ordinal)
{
    if constexpr (std::integral<LocalID>)
    {
        if (!std::in_range<LocalID>(local_ordinal))
        {
            throw std::overflow_error(
                "Local/global indexer local ID exceeds its type.");
        }
        return static_cast<LocalID>(local_ordinal);
    }
    else
    {
        static_assert(std::same_as<LocalID, GlobalID>,
            "Distinct local and global entity ID types require explicit "
            "EntityMapping inputs.");
        return global_id;
    }
}

template<MeshIndexTypePack Pack>
template<class GlobalID>
typename LocalGlobalIndexer<Pack>::global_ordinal_type
LocalGlobalIndexer<Pack>::infer_global_ordinal(
    const GlobalID& global_id,
    size_t local_ordinal)
{
    if constexpr (std::integral<GlobalID>)
    {
        if (!std::in_range<global_ordinal_type>(global_id))
        {
            throw std::overflow_error(
                "Local/global indexer global ID exceeds its ordinal "
                "type.");
        }
        return static_cast<global_ordinal_type>(global_id);
    }
    else
    {
        static_assert(std::integral<GlobalID>,
            "Non-integral global entity IDs require explicit "
            "EntityMapping inputs with global ordinals.");
    }
}

} // namespace SimpleFluid::Meshes
