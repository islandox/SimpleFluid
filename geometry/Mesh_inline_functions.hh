/**
 * @file Mesh_inline_functions.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief
 * @version 0.1
 * @date 2026-05-25
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <Kokkos_Core.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace SimpleFluid
{

namespace detail
{

template <class Ordinal>
inline constexpr Ordinal invalid_local_ordinal()
{
    return std::numeric_limits<Ordinal>::max();
}

template <class Ordinal>
inline Ordinal checked_size_to_ordinal(std::size_t value, std::string_view label)
{
    if (value >= static_cast<std::size_t>(std::numeric_limits<Ordinal>::max()))
    {
        throw std::overflow_error("Value for " + std::string(label)
                                + " exceeds maximum representable by ordinal type.");
    }

    return static_cast<Ordinal>(value);
}

} // namespace detail

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::cell(local_ordinal_type lid) const -> const CellInfo&
{
    check_cell(lid);
    return d_cells[static_cast<std::size_t>(lid)];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face(local_ordinal_type lid) const -> const FaceInfo&
{
    check_face(lid);
    return d_faces[static_cast<std::size_t>(lid)];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::cell_global_id(local_ordinal_type lid) const -> const EntityId&
{
    return cell(lid).global_id;
}

template<TpetraTypePack Pack>
inline bool Mesh<Pack>::is_owned_cell(local_ordinal_type lid) const
{
    return cell(lid).owned;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::faces(local_ordinal_type cell_lid) const -> const ArrLO&
{
    return cell(cell_lid).faces;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::node_coord(stk::mesh::Entity node) const -> Vec3
{
    if (!node.is_local_offset_valid() || d_coord_field == nullptr)
    {
        throw std::runtime_error("Cannot read node coordinates from the STK mesh.");
    }

    const double* coord = stk::mesh::field_data(*d_coord_field, node);
    if (coord == nullptr)
    {
        throw std::runtime_error("Coordinate field has no data for node "
                               + std::to_string(d_bulk.identifier(node)) + ".");
    }

    const auto dim = spatial_dimension();
    return Vec3{coord[0], dim > 1 ? coord[1] : 0.0, dim > 2 ? coord[2] : 0.0};
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::node_coord_by_id(EntityId node_id) const -> Vec3
{
    const auto node = d_bulk.get_entity(stk::topology::NODE_RANK, node_id);
    if (!node.is_local_offset_valid())
    {
        throw std::out_of_range("Node id not found: " + std::to_string(node_id));
    }

    return node_coord(node);
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::element_node_coords(stk::mesh::Entity elem) const -> Arr<Vec3>
{
    if (!elem.is_local_offset_valid())
    {
        throw std::out_of_range("Invalid STK element entity.");
    }

    const auto num_nodes = d_bulk.num_nodes(elem);
    const auto* nodes = d_bulk.begin_nodes(elem);

    Arr<Vec3> coords;
    coords.reserve(num_nodes);

    for (unsigned i = 0; i < num_nodes; ++i)
    {
        coords.push_back(node_coord(nodes[i]));
    }

    return coords;
}

template<TpetraTypePack Pack>
inline real_t Mesh<Pack>::cell_volume(local_ordinal_type lid) const
{
    return cell(lid).volume;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::cell_centroid(local_ordinal_type lid) const -> const Vec3&
{
    return cell(lid).center;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::owner_cell(local_ordinal_type fid) const -> local_ordinal_type
{
    return face(fid).owner;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::neighbor_cell(local_ordinal_type fid) const -> local_ordinal_type
{
    const auto& info = face(fid);
    return info.neighbor.value_or(detail::invalid_local_ordinal<local_ordinal_type>());
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::opposite_cell(local_ordinal_type fid, local_ordinal_type cell_lid) const -> local_ordinal_type
{
    const auto& info = face(fid);
    check_cell(cell_lid);

    if (info.owner == cell_lid)
    {
        return neighbor_cell(fid);
    }
    if (info.neighbor && *info.neighbor == cell_lid)
    {
        return info.owner;
    }

    throw std::invalid_argument("Cell is not adjacent to requested face.");
}

template<TpetraTypePack Pack>
inline real_t Mesh<Pack>::face_area(local_ordinal_type fid) const
{
    return face(fid).area;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face_normal(local_ordinal_type fid) const -> const Vec3&
{
    return face(fid).unit_normal_from_owner;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face_centroid(local_ordinal_type fid) const -> const Vec3&
{
    return face(fid).center;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face_normal_outward(local_ordinal_type fid,
                                            local_ordinal_type cell_lid) const -> const Vec3&
{
    const auto& info = face(fid);
    check_cell(cell_lid);

    if (info.owner == cell_lid)
    {
        return info.unit_normal_from_owner;
    }
    if (info.neighbor && *info.neighbor == cell_lid)
    {
        return info.unit_normal_from_neighbor;
    }

    throw std::invalid_argument("Cell is not adjacent to requested face.");
}

template<TpetraTypePack Pack>
inline bool Mesh<Pack>::is_exterior_face(local_ordinal_type fid) const
{
    return !face(fid).neighbor.has_value();
}

template<TpetraTypePack Pack>
inline bool Mesh<Pack>::is_interior_face(local_ordinal_type fid) const
{
    return face(fid).neighbor.has_value();
}

template<TpetraTypePack Pack>
inline bool Mesh<Pack>::is_boundary_face(local_ordinal_type fid) const
{
    return is_exterior_face(fid) && boundary_id(fid) != invalid_boundary_id;
}

template<TpetraTypePack Pack>
inline int Mesh<Pack>::boundary_id(local_ordinal_type fid) const
{
    return face(fid).boundary_id;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::boundary_name(local_ordinal_type fid) const -> const std::string&
{
    return face(fid).boundary_name;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face_patch(int patch_id) const -> const ArrLO&
{
    static const ArrLO empty_patch;
    const auto iter = d_boundary_id_to_faces.find(patch_id);
    return iter == d_boundary_id_to_faces.end() ? empty_patch : iter->second;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::find_local_cell(EntityId gid) const -> local_ordinal_type
{
    const auto iter = d_cell_gid_to_lid.find(gid);
    if (iter == d_cell_gid_to_lid.end())
    {
        return detail::invalid_local_ordinal<local_ordinal_type>();
    }

    return iter->second;
}

template<TpetraTypePack Pack>
template <class T>
auto Mesh<Pack>::make_vector_view(const std::string& name, const std::vector<T>& data)
    -> kokkos_1dview<const T>
{
    kokkos_1dview<T> view(name, data.size());
    auto host_view = Kokkos::create_mirror_view(view);

    for (std::size_t i = 0; i < data.size(); ++i)
    {
        host_view(i) = data[i];
    }

    Kokkos::deep_copy(view, host_view);
    return view;
}

template<TpetraTypePack Pack>
auto Mesh<Pack>::make_vectorV3D_view(const std::string& name,
                                     const std::vector<Vec3>& data)
    -> kokkos_vec3view<const real_t>
{
    kokkos_vec3view<real_t> view(name, data.size());
    auto host_view = Kokkos::create_mirror_view(view);

    for (std::size_t i = 0; i < data.size(); ++i)
    {
        host_view(i, 0) = data[i].x;
        host_view(i, 1) = data[i].y;
        host_view(i, 2) = data[i].z;
    }

    Kokkos::deep_copy(view, host_view);
    return view;
}

} // namespace SimpleFluid
