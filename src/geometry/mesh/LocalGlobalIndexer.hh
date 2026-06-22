/**
 * @file LocalGlobalIndexer.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Bidirectional local/global indexing for partitioned CRTP meshes.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "geometry/mesh/MeshIndexTypes.hh"

#include <concepts>
#include <cstddef>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes
{

/**
 * @brief Map compact partition-local ordinals to typed mesh-global IDs.
 *
 * Cells, faces, and nodes store owned entities before overlap entities.
 * Mapping-record constructors support distinct local/global entity ID types
 * and explicit global ordinals. Legacy constructors infer compact integral
 * local IDs and integral global ordinals from their input order and IDs.
 *
 * @par Template instantiation
 * SimpleFluid provides compiled definitions for the index packs instantiated
 * in `LocalGlobalIndexer.cc`. Code using another valid @p Pack must make the
 * definitions in `geometry/mesh/LocalGlobalIndexer.tcc` visible where they are
 * instantiated, or explicitly instantiate the specialization from that file
 * in one translation unit.
 *
 * Lightweight accessors are defined in `LocalGlobalIndexer.ipp`, which this
 * header includes automatically.
 *
 * @tparam Pack Mesh entity-ID and local/global ordinal types.
 */
template<MeshIndexTypePack Pack = MeshIndexTypes<
             global_index_t,
             global_index_t,
             global_index_t,
             local_index_t,
             global_index_t>>
class LocalGlobalIndexer
{
public:
    using index_type_pack = Pack;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using cell_id_t = typename Pack::cell_id_t;
    using face_id_t = typename Pack::face_id_t;
    using node_id_t = typename Pack::node_id_t;
    using local_cell_id_t = typename Pack::local_cell_id_t;
    using local_face_id_t = typename Pack::local_face_id_t;
    using local_node_id_t = typename Pack::local_node_id_t;
    using global_cell_id_t = typename Pack::global_cell_id_t;
    using global_face_id_t = typename Pack::global_face_id_t;
    using global_node_id_t = typename Pack::global_node_id_t;
    using ordinal_t = local_ordinal_type;

    template<class LocalID, class GlobalID>
    struct EntityMapping
    {
        LocalID local_id;
        GlobalID global_id;
        global_ordinal_type global_ordinal;
    };

    using CellMapping = EntityMapping<local_cell_id_t, global_cell_id_t>;
    using FaceMapping = EntityMapping<local_face_id_t, global_face_id_t>;
    using NodeMapping = EntityMapping<local_node_id_t, global_node_id_t>;

    static_assert(!std::same_as<local_ordinal_type, bool>);
    static_assert(!std::same_as<global_ordinal_type, bool>);
    static_assert(std::same_as<cell_id_t, global_cell_id_t>);
    static_assert(std::same_as<face_id_t, global_face_id_t>);
    static_assert(std::same_as<node_id_t, global_node_id_t>);

    LocalGlobalIndexer() = default;

    LocalGlobalIndexer(
        std::vector<cell_id_t> owned_cells,
        std::vector<cell_id_t> ghost_cells,
        std::vector<face_id_t> owned_faces,
        std::vector<face_id_t> overlap_faces,
        std::vector<node_id_t> nodes);

    LocalGlobalIndexer(
        std::vector<cell_id_t> owned_cells,
        std::vector<cell_id_t> ghost_cells,
        std::vector<face_id_t> owned_faces,
        std::vector<face_id_t> overlap_faces,
        std::vector<node_id_t> owned_nodes,
        std::vector<node_id_t> overlap_nodes);

    LocalGlobalIndexer(
        std::vector<CellMapping> owned_cells,
        std::vector<CellMapping> ghost_cells,
        std::vector<FaceMapping> owned_faces,
        std::vector<FaceMapping> overlap_faces,
        std::vector<NodeMapping> nodes);

    LocalGlobalIndexer(
        std::vector<CellMapping> owned_cells,
        std::vector<CellMapping> ghost_cells,
        std::vector<FaceMapping> owned_faces,
        std::vector<FaceMapping> overlap_faces,
        std::vector<NodeMapping> owned_nodes,
        std::vector<NodeMapping> overlap_nodes);

    static constexpr local_ordinal_type invalid_local_id() noexcept;

    void set_cells(
        std::vector<cell_id_t> owned,
        std::vector<cell_id_t> ghost);

    void set_faces(
        std::vector<face_id_t> owned,
        std::vector<face_id_t> overlap);

    void set_nodes(std::vector<node_id_t> nodes);

    void set_nodes(
        std::vector<node_id_t> owned,
        std::vector<node_id_t> overlap);

    void set_cells(
        std::vector<CellMapping> owned,
        std::vector<CellMapping> ghost);

    void set_faces(
        std::vector<FaceMapping> owned,
        std::vector<FaceMapping> overlap);

    void set_nodes(std::vector<NodeMapping> nodes);

    void set_nodes(
        std::vector<NodeMapping> owned,
        std::vector<NodeMapping> overlap);

    size_t num_owned_cells() const noexcept;
    size_t num_local_cells() const noexcept;
    size_t num_owned_faces() const noexcept;
    size_t num_local_faces() const noexcept;
    size_t num_owned_nodes() const noexcept;
    size_t num_local_nodes() const noexcept;

    bool is_owned_cell(local_ordinal_type local_id) const;
    bool is_owned_face(local_ordinal_type local_id) const;
    bool is_owned_node(local_ordinal_type local_id) const;

    cell_id_t cell_id(ordinal_t ordinal) const;
    face_id_t face_id(ordinal_t ordinal) const;
    node_id_t node_id(ordinal_t ordinal) const;

    ordinal_t cell_ordinal(const cell_id_t& id) const noexcept;
    ordinal_t face_ordinal(const face_id_t& id) const noexcept;
    ordinal_t node_ordinal(const node_id_t& id) const noexcept;

    global_cell_id_t local_to_global_cell_id(
        const local_cell_id_t& local_id) const;
    global_face_id_t local_to_global_face_id(
        const local_face_id_t& local_id) const;
    global_node_id_t local_to_global_node_id(
        const local_node_id_t& local_id) const;

    local_cell_id_t global_to_local_cell_id(
        const global_cell_id_t& global_id) const;
    local_face_id_t global_to_local_face_id(
        const global_face_id_t& global_id) const;
    local_node_id_t global_to_local_node_id(
        const global_node_id_t& global_id) const;

    global_ordinal_type local_to_global_cell_ordinal(
        local_ordinal_type local_ordinal) const;
    global_ordinal_type local_to_global_face_ordinal(
        local_ordinal_type local_ordinal) const;
    global_ordinal_type local_to_global_node_ordinal(
        local_ordinal_type local_ordinal) const;

    local_ordinal_type global_to_local_cell_ordinal(
        global_ordinal_type global_ordinal) const noexcept;
    local_ordinal_type global_to_local_face_ordinal(
        global_ordinal_type global_ordinal) const noexcept;
    local_ordinal_type global_to_local_node_ordinal(
        global_ordinal_type global_ordinal) const noexcept;

    cell_id_t cell_global_id(local_ordinal_type local_id) const;
    face_id_t face_global_id(local_ordinal_type local_id) const;
    node_id_t node_global_id(local_ordinal_type local_id) const;

    local_ordinal_type cell_local_id(const cell_id_t& global_id) const noexcept;
    local_ordinal_type face_local_id(const face_id_t& global_id) const noexcept;
    local_ordinal_type node_local_id(const node_id_t& global_id) const noexcept;

    cell_id_t local_to_global_cell(local_ordinal_type local_id) const;
    face_id_t local_to_global_face(local_ordinal_type local_id) const;
    node_id_t local_to_global_node(local_ordinal_type local_id) const;

    local_ordinal_type global_to_local_cell(
        const cell_id_t& global_id) const noexcept;
    local_ordinal_type global_to_local_face(
        const face_id_t& global_id) const noexcept;
    local_ordinal_type global_to_local_node(
        const node_id_t& global_id) const noexcept;

    const std::vector<cell_id_t>& cell_global_ids() const noexcept;
    const std::vector<face_id_t>& face_global_ids() const noexcept;
    const std::vector<node_id_t>& node_global_ids() const noexcept;

    std::span<const cell_id_t> owned_cell_global_ids() const noexcept;
    std::span<const cell_id_t> ghost_cell_global_ids() const noexcept;
    std::span<const face_id_t> owned_face_global_ids() const noexcept;
    std::span<const face_id_t> overlap_face_global_ids() const noexcept;
    std::span<const node_id_t> owned_node_global_ids() const noexcept;
    std::span<const node_id_t> overlap_node_global_ids() const noexcept;

private:
    template<class LocalID, class GlobalID>
    struct EntityIndex
    {
        size_t owned_count = 0;
        std::vector<LocalID> local_ids;
        std::vector<GlobalID> global_ids;
        std::vector<global_ordinal_type> global_ordinals;
        std::map<LocalID, local_ordinal_type> ordinal_by_local_id;
        std::map<GlobalID, local_ordinal_type> ordinal_by_global_id;
        std::map<global_ordinal_type, local_ordinal_type>
            local_by_global_ordinal;
    };

    template<class LocalID, class GlobalID>
    static void assign(
        EntityIndex<LocalID, GlobalID>& index,
        std::vector<GlobalID> owned,
        std::vector<GlobalID> overlap,
        std::string_view entity);

    template<class LocalID, class GlobalID>
    static void assign_mappings(
        EntityIndex<LocalID, GlobalID>& index,
        std::vector<EntityMapping<LocalID, GlobalID>> owned,
        std::vector<EntityMapping<LocalID, GlobalID>> overlap,
        std::string_view entity);

    template<class LocalID, class GlobalID>
    static void check_local(
        const EntityIndex<LocalID, GlobalID>& index,
        local_ordinal_type local_id,
        std::string_view entity);

    template<class LocalID, class GlobalID>
    static GlobalID global_id(
        const EntityIndex<LocalID, GlobalID>& index,
        local_ordinal_type local_id,
        std::string_view entity);

    template<class LocalID, class GlobalID>
    static local_ordinal_type local_id(
        const EntityIndex<LocalID, GlobalID>& index,
        const GlobalID& global_id) noexcept;

    template<class LocalID, class GlobalID>
    static GlobalID mapped_global_id(
        const EntityIndex<LocalID, GlobalID>& index,
        const LocalID& local_id,
        std::string_view entity);

    template<class LocalID, class GlobalID>
    static LocalID mapped_local_id(
        const EntityIndex<LocalID, GlobalID>& index,
        const GlobalID& global_id,
        std::string_view entity);

    template<class LocalID, class GlobalID>
    static global_ordinal_type global_ordinal(
        const EntityIndex<LocalID, GlobalID>& index,
        local_ordinal_type local_ordinal,
        std::string_view entity);

    template<class LocalID, class GlobalID>
    static local_ordinal_type local_ordinal(
        const EntityIndex<LocalID, GlobalID>& index,
        global_ordinal_type global_ordinal) noexcept;

    template<class LocalID, class GlobalID>
    static LocalID infer_local_id(
        const GlobalID& global_id,
        size_t local_ordinal);

    template<class GlobalID>
    static global_ordinal_type infer_global_ordinal(
        const GlobalID& global_id,
        size_t local_ordinal);

    EntityIndex<local_cell_id_t, global_cell_id_t> d_cells;
    EntityIndex<local_face_id_t, global_face_id_t> d_faces;
    EntityIndex<local_node_id_t, global_node_id_t> d_nodes;
};

} // namespace SimpleFluid::Meshes

#include "geometry/mesh/LocalGlobalIndexer.ipp"
