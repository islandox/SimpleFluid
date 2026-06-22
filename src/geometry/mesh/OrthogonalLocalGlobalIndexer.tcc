/**
 * @file OrthogonalLocalGlobalIndexer.tcc
 * @brief Out-of-line methods for the arithmetic indexer specialization.
 *
 * @details OrthogonalLocalGlobalIndexer.cc includes this file for the common
 * explicit instantiations. A translation unit using another orthogonal index
 * pack may include it to instantiate block construction and coordinate
 * mapping for that pack.
 */

#include "geometry/mesh/OrthogonalLocalGlobalIndexer.hh"

namespace SimpleFluid::Meshes
{

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
LocalGlobalIndexer<Pack>::LocalGlobalIndexer(
    const Indexer& global_indexer,
    BlockShape block_counts,
    BlockShape block_coordinates)
    : d_global_indexer(global_indexer),
      d_block_counts(block_counts),
      d_block_coordinates(block_coordinates)
{
    Vec3D<Ordinal> local_cells{};
    Vec3D<bool> local_periodic{};
    for (size_t dimension = 0; dimension < 3; ++dimension)
    {
        const auto global_cells = static_cast<size_t>(
            d_global_indexer.num_cells_per_dim[dimension]);
        const auto blocks = d_block_counts[dimension];
        const auto block = d_block_coordinates[dimension];
        if (global_cells == 0)
        {
            throw std::invalid_argument(
                "Orthogonal local/global indexer requires positive "
                "global dimensions.");
        }
        if (blocks == 0 || blocks > global_cells)
        {
            throw std::invalid_argument(
                "Orthogonal local/global indexer block count is "
                "invalid.");
        }
        if (block >= blocks)
        {
            throw std::out_of_range(
                "Orthogonal local/global indexer block coordinate is "
                "out of range.");
        }

        const auto base_size = global_cells / blocks;
        const auto remainder = global_cells % blocks;
        const auto begin =
            block * base_size + std::min(block, remainder);
        const auto size =
            base_size + (block < remainder ? 1 : 0);
        d_block_begin[dimension] = static_cast<Ordinal>(begin);
        local_cells[dimension] = static_cast<Ordinal>(size);
        local_periodic[dimension] =
            d_global_indexer.periodic_dimensions[dimension]
            && blocks == 1;
    }

    d_local_indexer = Indexer(
        local_cells[0],
        local_cells[1],
        local_cells[2],
        local_periodic[0],
        local_periodic[1],
        local_periodic[2]);
    validate_ordinal_capacity();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::global_cell_id_t
LocalGlobalIndexer<Pack>::local_to_global_cell_id(
    const local_cell_id_t& local_id) const
{
    check_cell_id(d_local_indexer, local_id, "local");
    return {
        static_cast<Ordinal>(local_id.i + d_block_begin[0]),
        static_cast<Ordinal>(local_id.j + d_block_begin[1]),
        static_cast<Ordinal>(local_id.k + d_block_begin[2])};
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::global_face_id_t
LocalGlobalIndexer<Pack>::local_to_global_face_id(
    const local_face_id_t& local_id) const
{
    check_face_id(d_local_indexer, local_id, "local");
    global_face_id_t global_id{
        static_cast<Ordinal>(local_id.i + d_block_begin[0]),
        static_cast<Ordinal>(local_id.j + d_block_begin[1]),
        static_cast<Ordinal>(local_id.k + d_block_begin[2]),
        local_id.orientation};
    wrap_global_face(global_id);
    return global_id;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::global_node_id_t
LocalGlobalIndexer<Pack>::local_to_global_node_id(
    const local_node_id_t& local_id) const
{
    check_node_id(d_local_indexer, local_id, "local");
    global_node_id_t global_id{
        static_cast<Ordinal>(local_id.i + d_block_begin[0]),
        static_cast<Ordinal>(local_id.j + d_block_begin[1]),
        static_cast<Ordinal>(local_id.k + d_block_begin[2])};
    wrap_global_node(global_id);
    return global_id;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::local_cell_id_t
LocalGlobalIndexer<Pack>::global_to_local_cell_id(
    const global_cell_id_t& global_id) const
{
    local_cell_id_t local_id;
    if (!try_global_to_local_cell_id(global_id, local_id))
    {
        throw std::out_of_range(
            "Global cell ID is outside the orthogonal block.");
    }
    return local_id;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::local_face_id_t
LocalGlobalIndexer<Pack>::global_to_local_face_id(
    const global_face_id_t& global_id) const
{
    local_face_id_t local_id;
    if (!try_global_to_local_face_id(global_id, local_id))
    {
        throw std::out_of_range(
            "Global face ID is outside the orthogonal block.");
    }
    return local_id;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::local_node_id_t
LocalGlobalIndexer<Pack>::global_to_local_node_id(
    const global_node_id_t& global_id) const
{
    local_node_id_t local_id;
    if (!try_global_to_local_node_id(global_id, local_id))
    {
        throw std::out_of_range(
            "Global node ID is outside the orthogonal block.");
    }
    return local_id;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::cell_id_t
LocalGlobalIndexer<Pack>::cell_id(ordinal_t local_ordinal) const
{
    return local_to_global_cell_id(d_local_indexer.cell_id(
        checked_geometry_ordinal(
            local_ordinal, num_local_cells(), "cell")));
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::face_id_t
LocalGlobalIndexer<Pack>::face_id(ordinal_t local_ordinal) const
{
    return local_to_global_face_id(d_local_indexer.face_id(
        checked_geometry_ordinal(
            local_ordinal, num_local_faces(), "face")));
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::node_id_t
LocalGlobalIndexer<Pack>::node_id(ordinal_t local_ordinal) const
{
    return local_to_global_node_id(d_local_indexer.node_id(
        checked_geometry_ordinal(
            local_ordinal, num_local_nodes(), "node")));
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::ordinal_t
LocalGlobalIndexer<Pack>::cell_ordinal(
    const cell_id_t& global_id) const noexcept
{
    local_cell_id_t local_id;
    return try_global_to_local_cell_id(global_id, local_id)
         ? checked_local(d_local_indexer.cell_ordinal(local_id))
         : invalid_local_id();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::ordinal_t
LocalGlobalIndexer<Pack>::face_ordinal(
    const face_id_t& global_id) const noexcept
{
    local_face_id_t local_id;
    return try_global_to_local_face_id(global_id, local_id)
         ? checked_local(d_local_indexer.face_ordinal(local_id))
         : invalid_local_id();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
typename LocalGlobalIndexer<Pack>::ordinal_t
LocalGlobalIndexer<Pack>::node_ordinal(
    const node_id_t& global_id) const noexcept
{
    local_node_id_t local_id;
    return try_global_to_local_node_id(global_id, local_id)
         ? checked_local(d_local_indexer.node_ordinal(local_id))
         : invalid_local_id();
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
void LocalGlobalIndexer<Pack>::wrap_global_face(
    global_face_id_t& id) const noexcept
{
    auto values = coordinates(id);
    const auto& dimensions =
        d_global_indexer.num_faces_per_dim_per_orientation[
            id.orientation];
    for (size_t dimension = 0; dimension < 3; ++dimension)
    {
        if (d_global_indexer.periodic_dimensions[dimension]
            && values[dimension] >= dimensions[dimension])
        {
            values[dimension] = static_cast<Ordinal>(
                values[dimension] % dimensions[dimension]);
        }
    }
    id.i = values[0];
    id.j = values[1];
    id.k = values[2];
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
void LocalGlobalIndexer<Pack>::wrap_global_node(
    global_node_id_t& id) const noexcept
{
    auto values = coordinates(id);
    for (size_t dimension = 0; dimension < 3; ++dimension)
    {
        const auto extent =
            d_global_indexer.num_nodes_per_dim[dimension];
        if (d_global_indexer.periodic_dimensions[dimension]
            && values[dimension] >= extent)
        {
            values[dimension] = static_cast<Ordinal>(
                values[dimension] % extent);
        }
    }
    id.i = values[0];
    id.j = values[1];
    id.k = values[2];
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
bool LocalGlobalIndexer<Pack>::unmap_coordinate(
    Ordinal global_coordinate,
    Ordinal begin,
    Ordinal local_extent,
    Ordinal global_extent,
    bool periodic,
    Ordinal& local_coordinate) noexcept
{
    auto unwrapped = static_cast<size_t>(global_coordinate);
    const auto block_begin = static_cast<size_t>(begin);
    if (periodic && unwrapped < block_begin)
    {
        unwrapped += global_extent;
    }
    if (unwrapped < block_begin
        || unwrapped >= block_begin + local_extent)
    {
        return false;
    }
    local_coordinate = static_cast<Ordinal>(unwrapped - block_begin);
    return true;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
bool LocalGlobalIndexer<Pack>::try_global_to_local_cell_id(
    const global_cell_id_t& global_id,
    local_cell_id_t& local_id) const noexcept
{
    if (!valid_coordinates(
            coordinates(global_id),
            d_global_indexer.num_cells_per_dim))
    {
        return false;
    }
    const auto global = coordinates(global_id);
    Vec3D<Ordinal> local{};
    for (size_t dimension = 0; dimension < 3; ++dimension)
    {
        if (!unmap_coordinate(
                global[dimension],
                d_block_begin[dimension],
                d_local_indexer.num_cells_per_dim[dimension],
                d_global_indexer.num_cells_per_dim[dimension],
                false,
                local[dimension]))
        {
            return false;
        }
    }
    local_id = {local[0], local[1], local[2]};
    return true;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
bool LocalGlobalIndexer<Pack>::try_global_to_local_face_id(
    const global_face_id_t& global_id,
    local_face_id_t& local_id) const noexcept
{
    if (global_id.orientation >= 3
        || !valid_coordinates(
            coordinates(global_id),
            d_global_indexer.num_faces_per_dim_per_orientation[
                global_id.orientation]))
    {
        return false;
    }
    local_id.orientation = global_id.orientation;
    const auto global = coordinates(global_id);
    Vec3D<Ordinal> local{};
    const auto& local_dimensions =
        d_local_indexer.num_faces_per_dim_per_orientation[
            global_id.orientation];
    const auto& global_dimensions =
        d_global_indexer.num_faces_per_dim_per_orientation[
            global_id.orientation];
    for (size_t dimension = 0; dimension < 3; ++dimension)
    {
        if (!unmap_coordinate(
                global[dimension],
                d_block_begin[dimension],
                local_dimensions[dimension],
                global_dimensions[dimension],
                d_global_indexer.periodic_dimensions[dimension],
                local[dimension]))
        {
            return false;
        }
    }
    local_id = {
        local[0], local[1], local[2], global_id.orientation};
    return true;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
bool LocalGlobalIndexer<Pack>::try_global_to_local_node_id(
    const global_node_id_t& global_id,
    local_node_id_t& local_id) const noexcept
{
    if (!valid_coordinates(
            coordinates(global_id),
            d_global_indexer.num_nodes_per_dim))
    {
        return false;
    }
    const auto global = coordinates(global_id);
    Vec3D<Ordinal> local{};
    for (size_t dimension = 0; dimension < 3; ++dimension)
    {
        if (!unmap_coordinate(
                global[dimension],
                d_block_begin[dimension],
                d_local_indexer.num_nodes_per_dim[dimension],
                d_global_indexer.num_nodes_per_dim[dimension],
                d_global_indexer.periodic_dimensions[dimension],
                local[dimension]))
        {
            return false;
        }
    }
    local_id = {local[0], local[1], local[2]};
    return true;
}

template<MeshIndexTypePack Pack>
    requires requires {
        typename Pack::orthogonal_index_type_pack_tag;
    }
void LocalGlobalIndexer<Pack>::validate_ordinal_capacity() const
{
    const auto local_max = static_cast<size_t>(
        std::numeric_limits<local_ordinal_type>::max());
    if (num_local_cells() > local_max
        || num_local_faces() > local_max
        || num_local_nodes() > local_max)
    {
        throw std::overflow_error(
            "Orthogonal block exceeds its local ordinal type.");
    }
    if (!std::in_range<global_ordinal_type>(
            d_global_indexer.total_cells())
        || !std::in_range<global_ordinal_type>(
            d_global_indexer.total_faces())
        || !std::in_range<global_ordinal_type>(
            d_global_indexer.total_nodes()))
    {
        throw std::overflow_error(
            "Orthogonal mesh exceeds its global ordinal type.");
    }
}

} // namespace SimpleFluid::Meshes
