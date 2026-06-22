/**
 * @file OrthogonalLocalGlobalIndexer.ipp
 * @brief Inline methods for the arithmetic LocalGlobalIndexer specialization.
 *
 * @details Contains inexpensive accessors, ordinal conversion wrappers, and
 * validation helpers included automatically by
 * OrthogonalLocalGlobalIndexer.hh. Block construction and coordinate mapping
 * are defined in OrthogonalLocalGlobalIndexer.tcc.
 */

#pragma once

namespace SimpleFluid::Meshes
{

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline const typename LocalGlobalIndexer<Pack>::Indexer&
LocalGlobalIndexer<Pack>::local_indexer() const noexcept
{
    return d_local_indexer;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline const typename LocalGlobalIndexer<Pack>::Indexer&
LocalGlobalIndexer<Pack>::global_indexer() const noexcept
{
    return d_global_indexer;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline const typename LocalGlobalIndexer<Pack>::BlockShape&
LocalGlobalIndexer<Pack>::block_counts() const noexcept
{
    return d_block_counts;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline const typename LocalGlobalIndexer<Pack>::BlockShape&
LocalGlobalIndexer<Pack>::block_coordinates() const noexcept
{
    return d_block_coordinates;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline const typename LocalGlobalIndexer<Pack>::BlockOrigin&
LocalGlobalIndexer<Pack>::block_begin() const noexcept
{
    return d_block_begin;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
constexpr typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::invalid_local_id() noexcept
{
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        return static_cast<local_ordinal_type>(-1);
    }
    return std::numeric_limits<local_ordinal_type>::max();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline size_t LocalGlobalIndexer<Pack>::num_owned_cells() const noexcept
{
    return d_local_indexer.total_cells();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline size_t LocalGlobalIndexer<Pack>::num_local_cells() const noexcept
{
    return d_local_indexer.total_cells();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline size_t LocalGlobalIndexer<Pack>::num_owned_faces() const noexcept
{
    return d_local_indexer.total_faces();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline size_t LocalGlobalIndexer<Pack>::num_local_faces() const noexcept
{
    return d_local_indexer.total_faces();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline size_t LocalGlobalIndexer<Pack>::num_owned_nodes() const noexcept
{
    return d_local_indexer.total_nodes();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline size_t LocalGlobalIndexer<Pack>::num_local_nodes() const noexcept
{
    return d_local_indexer.total_nodes();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline bool LocalGlobalIndexer<Pack>::is_owned_cell(
    local_ordinal_type local_ordinal) const
{
    check_local_ordinal(
        local_ordinal, num_local_cells(), "cell");
    return true;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline bool LocalGlobalIndexer<Pack>::is_owned_face(
    local_ordinal_type local_ordinal) const
{
    check_local_ordinal(
        local_ordinal, num_local_faces(), "face");
    return true;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline bool LocalGlobalIndexer<Pack>::is_owned_node(
    local_ordinal_type local_ordinal) const
{
    check_local_ordinal(
        local_ordinal, num_local_nodes(), "node");
    return true;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::global_ordinal_type
LocalGlobalIndexer<Pack>::local_to_global_cell_ordinal(
    local_ordinal_type local_ordinal) const
{
    return checked_global(
        d_global_indexer.cell_ordinal(cell_id(local_ordinal)));
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::global_ordinal_type
LocalGlobalIndexer<Pack>::local_to_global_face_ordinal(
    local_ordinal_type local_ordinal) const
{
    return checked_global(
        d_global_indexer.face_ordinal(face_id(local_ordinal)));
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::global_ordinal_type
LocalGlobalIndexer<Pack>::local_to_global_node_ordinal(
    local_ordinal_type local_ordinal) const
{
    return checked_global(
        d_global_indexer.node_ordinal(node_id(local_ordinal)));
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_cell_ordinal(
    global_ordinal_type global_ordinal) const noexcept
{
    const auto geometry_ordinal = global_geometry_ordinal(
        global_ordinal, d_global_indexer.total_cells());
    return geometry_ordinal.has_value()
         ? cell_ordinal(d_global_indexer.cell_id(*geometry_ordinal))
         : invalid_local_id();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_face_ordinal(
    global_ordinal_type global_ordinal) const noexcept
{
    const auto geometry_ordinal = global_geometry_ordinal(
        global_ordinal, d_global_indexer.total_faces());
    return geometry_ordinal.has_value()
         ? face_ordinal(d_global_indexer.face_id(*geometry_ordinal))
         : invalid_local_id();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_node_ordinal(
    global_ordinal_type global_ordinal) const noexcept
{
    const auto geometry_ordinal = global_geometry_ordinal(
        global_ordinal, d_global_indexer.total_nodes());
    return geometry_ordinal.has_value()
         ? node_ordinal(d_global_indexer.node_id(*geometry_ordinal))
         : invalid_local_id();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::cell_id_t
LocalGlobalIndexer<Pack>::cell_global_id(
    local_ordinal_type local_ordinal) const
{
    return cell_id(local_ordinal);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::face_id_t
LocalGlobalIndexer<Pack>::face_global_id(
    local_ordinal_type local_ordinal) const
{
    return face_id(local_ordinal);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::node_id_t
LocalGlobalIndexer<Pack>::node_global_id(
    local_ordinal_type local_ordinal) const
{
    return node_id(local_ordinal);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::cell_local_id(
    const cell_id_t& global_id) const noexcept
{
    return cell_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::face_local_id(
    const face_id_t& global_id) const noexcept
{
    return face_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::node_local_id(
    const node_id_t& global_id) const noexcept
{
    return node_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::cell_id_t
LocalGlobalIndexer<Pack>::local_to_global_cell(
    local_ordinal_type local_ordinal) const
{
    return cell_id(local_ordinal);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::face_id_t
LocalGlobalIndexer<Pack>::local_to_global_face(
    local_ordinal_type local_ordinal) const
{
    return face_id(local_ordinal);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::node_id_t
LocalGlobalIndexer<Pack>::local_to_global_node(
    local_ordinal_type local_ordinal) const
{
    return node_id(local_ordinal);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_cell(
    const cell_id_t& global_id) const noexcept
{
    return cell_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_face(
    const face_id_t& global_id) const noexcept
{
    return face_ordinal(global_id);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::global_to_local_node(
    const node_id_t& global_id) const noexcept
{
    return node_ordinal(global_id);
}

// Private helper methods

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
template<class ID>
inline Vec3D<typename LocalGlobalIndexer<Pack>::Ordinal>
LocalGlobalIndexer<Pack>::coordinates(const ID& id) noexcept
{
    return {id.i, id.j, id.k};
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline bool LocalGlobalIndexer<Pack>::valid_coordinates(
    const Vec3D<Ordinal>& coordinates,
    const Vec3D<Ordinal>& dimensions) noexcept
{
    for (size_t dimension = 0; dimension < 3; ++dimension)
    {
        if (coordinates[dimension] >= dimensions[dimension])
        {
            return false;
        }
    }
    return true;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline void LocalGlobalIndexer<Pack>::check_cell_id(
    const Indexer& indexer,
    const cell_id_t& id,
    std::string_view scope)
{
    if (!valid_coordinates(
            coordinates(id), indexer.num_cells_per_dim))
    {
        throw std::out_of_range(
            "Orthogonal " + std::string(scope)
            + " cell ID is out of range.");
    }
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline void LocalGlobalIndexer<Pack>::check_face_id(
    const Indexer& indexer,
    const face_id_t& id,
    std::string_view scope)
{
    if (id.orientation >= 3
        || !valid_coordinates(
            coordinates(id),
            indexer.num_faces_per_dim_per_orientation[
                id.orientation]))
    {
        throw std::out_of_range(
            "Orthogonal " + std::string(scope)
            + " face ID is out of range.");
    }
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline void LocalGlobalIndexer<Pack>::check_node_id(
    const Indexer& indexer,
    const node_id_t& id,
    std::string_view scope)
{
    if (!valid_coordinates(
            coordinates(id), indexer.num_nodes_per_dim))
    {
        throw std::out_of_range(
            "Orthogonal " + std::string(scope)
            + " node ID is out of range.");
    }
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline void LocalGlobalIndexer<Pack>::check_local_ordinal(
    local_ordinal_type local_ordinal,
    size_t count,
    std::string_view entity)
{
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        if (local_ordinal < 0)
        {
            throw std::out_of_range(
                "Orthogonal " + std::string(entity)
                + " local ordinal is out of range.");
        }
    }
    if (static_cast<size_t>(local_ordinal) >= count)
    {
        throw std::out_of_range(
            "Orthogonal " + std::string(entity)
            + " local ordinal is out of range.");
    }
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline size_t LocalGlobalIndexer<Pack>::checked_geometry_ordinal(
    local_ordinal_type local_ordinal,
    size_t count,
    std::string_view entity)
{
    check_local_ordinal(local_ordinal, count, entity);
    return static_cast<size_t>(local_ordinal);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::local_ordinal_type
LocalGlobalIndexer<Pack>::checked_local(size_t ordinal) noexcept
{
    return static_cast<local_ordinal_type>(ordinal);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline typename LocalGlobalIndexer<Pack>::global_ordinal_type
LocalGlobalIndexer<Pack>::checked_global(size_t ordinal)
{
    if (!std::in_range<global_ordinal_type>(ordinal))
    {
        throw std::overflow_error(
            "Orthogonal global ordinal exceeds its type.");
    }
    return static_cast<global_ordinal_type>(ordinal);
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
inline std::optional<size_t>
LocalGlobalIndexer<Pack>::global_geometry_ordinal(
    global_ordinal_type ordinal,
    size_t count) noexcept
{
    if constexpr (std::is_signed_v<global_ordinal_type>)
    {
        if (ordinal < 0)
        {
            return std::nullopt;
        }
    }
    if (!std::in_range<size_t>(ordinal)
        || static_cast<size_t>(ordinal) >= count)
    {
        return std::nullopt;
    }
    return static_cast<size_t>(ordinal);
}

} // namespace SimpleFluid::Meshes
