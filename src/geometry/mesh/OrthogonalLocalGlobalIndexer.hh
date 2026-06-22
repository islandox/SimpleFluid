/**
 * @file OrthogonalLocalGlobalIndexer.hh
 * @brief Arithmetic local/global indexing for orthogonal mesh blocks.
 */

#pragma once

#include "geometry/mesh/LocalGlobalIndexer.hh"
#include "geometry/mesh/OrthogonalIndexer.hh"

#include <algorithm>
#include <concepts>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace SimpleFluid::Meshes
{

/**
 * @brief Local/global indexer for one axis-aligned orthogonal mesh block.
 *
 * Each global dimension is divided into balanced contiguous blocks. The
 * selected 3D block is represented by a local OrthogonalIndexer, so entity
 * conversion is computed from block offsets without materialized ID maps.
 * All entities in the block are owned; ghost halos are intentionally outside
 * this specialization's current contract.
 *
 * @par Template instantiation
 * SimpleFluid provides compiled definitions for the ordinal combinations
 * instantiated in `OrthogonalLocalGlobalIndexer.cc`. Code using another
 * orthogonal @p Pack must make the definitions in
 * `geometry/mesh/OrthogonalLocalGlobalIndexer.tcc` visible where they are
 * instantiated, or explicitly instantiate the specialization from that file
 * in one translation unit.
 *
 * @tparam Pack Orthogonal entity-ID types and local/global ordinal types.
 */
template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
class LocalGlobalIndexer<Pack>
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
    using Indexer = OrthogonalIndexer;
    using Ordinal = Indexer::Ordinal;
    using BlockShape = Vec3D<size_t>;
    using BlockOrigin = Vec3D<Ordinal>;

    static_assert(!std::same_as<local_ordinal_type, bool>);
    static_assert(!std::same_as<global_ordinal_type, bool>);

    LocalGlobalIndexer() = default;

    explicit LocalGlobalIndexer(const Indexer& global_indexer)
        : LocalGlobalIndexer(
              global_indexer,
              BlockShape{1, 1, 1},
              BlockShape{0, 0, 0})
    {
    }

    LocalGlobalIndexer(
        const Indexer& global_indexer,
        BlockShape block_counts,
        BlockShape block_coordinates);

    const Indexer& local_indexer() const noexcept;
    const Indexer& global_indexer() const noexcept;
    const BlockShape& block_counts() const noexcept;
    const BlockShape& block_coordinates() const noexcept;
    const BlockOrigin& block_begin() const noexcept;

    static constexpr local_ordinal_type invalid_local_id() noexcept;

    size_t num_owned_cells() const noexcept;
    size_t num_local_cells() const noexcept;
    size_t num_owned_faces() const noexcept;
    size_t num_local_faces() const noexcept;
    size_t num_owned_nodes() const noexcept;
    size_t num_local_nodes() const noexcept;

    bool is_owned_cell(local_ordinal_type local_ordinal) const;
    bool is_owned_face(local_ordinal_type local_ordinal) const;
    bool is_owned_node(local_ordinal_type local_ordinal) const;

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

    cell_id_t cell_id(ordinal_t local_ordinal) const;
    face_id_t face_id(ordinal_t local_ordinal) const;
    node_id_t node_id(ordinal_t local_ordinal) const;

    ordinal_t cell_ordinal(const cell_id_t& global_id) const noexcept;
    ordinal_t face_ordinal(const face_id_t& global_id) const noexcept;
    ordinal_t node_ordinal(const node_id_t& global_id) const noexcept;

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

    cell_id_t cell_global_id(local_ordinal_type local_ordinal) const;
    face_id_t face_global_id(local_ordinal_type local_ordinal) const;
    node_id_t node_global_id(local_ordinal_type local_ordinal) const;

    local_ordinal_type cell_local_id(
        const cell_id_t& global_id) const noexcept;
    local_ordinal_type face_local_id(
        const face_id_t& global_id) const noexcept;
    local_ordinal_type node_local_id(
        const node_id_t& global_id) const noexcept;

    cell_id_t local_to_global_cell(local_ordinal_type local_ordinal) const;
    face_id_t local_to_global_face(local_ordinal_type local_ordinal) const;
    node_id_t local_to_global_node(local_ordinal_type local_ordinal) const;

    local_ordinal_type global_to_local_cell(
        const cell_id_t& global_id) const noexcept;
    local_ordinal_type global_to_local_face(
        const face_id_t& global_id) const noexcept;
    local_ordinal_type global_to_local_node(
        const node_id_t& global_id) const noexcept;

private:
    template<class ID>
    static Vec3D<Ordinal> coordinates(const ID& id) noexcept;

    static bool valid_coordinates(
        const Vec3D<Ordinal>& coordinates,
        const Vec3D<Ordinal>& dimensions) noexcept;

    static void check_cell_id(
        const Indexer& indexer,
        const cell_id_t& id,
        std::string_view scope);

    static void check_face_id(
        const Indexer& indexer,
        const face_id_t& id,
        std::string_view scope);

    static void check_node_id(
        const Indexer& indexer,
        const node_id_t& id,
        std::string_view scope);

    void wrap_global_face(global_face_id_t& id) const noexcept;
    void wrap_global_node(global_node_id_t& id) const noexcept;

    static bool unmap_coordinate(
        Ordinal global_coordinate,
        Ordinal begin,
        Ordinal local_extent,
        Ordinal global_extent,
        bool periodic,
        Ordinal& local_coordinate) noexcept;

    bool try_global_to_local_cell_id(
        const global_cell_id_t& global_id,
        local_cell_id_t& local_id) const noexcept;

    bool try_global_to_local_face_id(
        const global_face_id_t& global_id,
        local_face_id_t& local_id) const noexcept;

    bool try_global_to_local_node_id(
        const global_node_id_t& global_id,
        local_node_id_t& local_id) const noexcept;

    static void check_local_ordinal(
        local_ordinal_type local_ordinal,
        size_t count,
        std::string_view entity);

    static size_t checked_geometry_ordinal(
        local_ordinal_type local_ordinal,
        size_t count,
        std::string_view entity);

    static local_ordinal_type checked_local(size_t ordinal) noexcept;

    static global_ordinal_type checked_global(size_t ordinal);

    static std::optional<size_t> global_geometry_ordinal(
        global_ordinal_type ordinal,
        size_t count) noexcept;

    void validate_ordinal_capacity() const;

    Indexer d_local_indexer;
    Indexer d_global_indexer;
    BlockShape d_block_counts{1, 1, 1};
    BlockShape d_block_coordinates{0, 0, 0};
    BlockOrigin d_block_begin{0, 0, 0};
};

} // namespace SimpleFluid::Meshes

#include "geometry/mesh/OrthogonalLocalGlobalIndexer.ipp"
