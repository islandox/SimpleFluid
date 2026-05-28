/**
 * @file STKMesh.ipp
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief Inline method definitions for the STKMesh class.
 * @version 0.1
 * @date 2026-05-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

namespace SimpleFluid
{

/**
 * @brief Read the coordinates of an STK node entity.
 *
 * @tparam Pack Tpetra type pack.
 * @param node STK node entity.
 * @return 3D coordinate vector.
 * @throws std::runtime_error if coordinate data is unavailable.
 */
template<TpetraTypePack Pack>
inline auto STKMesh<Pack>::node_coord(stk::mesh::Entity node) const -> Vec3
{
    if (!node.is_local_offset_valid() || d_stk.coord_field == nullptr)
    {
        throw std::runtime_error("Cannot read node coordinates from the STK mesh.");
    }

    const double* coord = stk::mesh::field_data(*d_stk.coord_field, node);
    if (coord == nullptr)
    {
        throw std::runtime_error("Coordinate field has no data for node "
                               + std::to_string(d_stk.bulk->identifier(node)) + ".");
    }

    const auto dim = Base::spatial_dimension();
    return Vec3{coord[0], dim > 1 ? coord[1] : 0.0, dim > 2 ? coord[2] : 0.0};
}

/**
 * @brief Read the coordinates of an STK node by its global ID.
 *
 * @tparam Pack Tpetra type pack.
 * @param node_id STK node entity ID.
 * @return 3D coordinate vector.
 * @throws std::out_of_range if the node ID is not found.
 */
template<TpetraTypePack Pack>
inline auto STKMesh<Pack>::node_coord_by_id(EntityId node_id) const -> Vec3
{
    const auto node = d_stk.bulk->get_entity(stk::topology::NODE_RANK, node_id);
    if (!node.is_local_offset_valid())
    {
        throw std::out_of_range("Node id not found: " + std::to_string(node_id));
    }

    return node_coord(node);
}

/**
 * @brief Collect the coordinates of all nodes belonging to an STK element.
 *
 * @tparam Pack Tpetra type pack.
 * @param elem STK element entity.
 * @return Vector of 3D node coordinates.
 * @throws std::out_of_range if the element entity is invalid.
 */
template<TpetraTypePack Pack>
inline auto STKMesh<Pack>::element_node_coords(stk::mesh::Entity elem) const -> Arr<Vec3>
{
    if (!elem.is_local_offset_valid())
    {
        throw std::out_of_range("Invalid STK element entity.");
    }

    const auto num_nodes = d_stk.bulk->num_nodes(elem);
    const auto* nodes = d_stk.bulk->begin_nodes(elem);

    Arr<Vec3> coords;
    coords.reserve(num_nodes);

    for (unsigned i = 0; i < num_nodes; ++i)
    {
        coords.push_back(node_coord(nodes[i]));
    }

    return coords;
}

}