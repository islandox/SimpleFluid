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
inline auto Mesh<Pack>::cell_global_id(local_ordinal_type lid) const -> const global_ordinal_type&
{
    return cell(lid).global_id;
}

template<TpetraTypePack Pack>
inline bool Mesh<Pack>::is_owned_cell(local_ordinal_type lid) const
{
    return cell(lid).owned;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::faces(local_ordinal_type cell_lid) const -> const ViewLO&
{
    return cell(cell_lid).faces;
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
    return face(fid).neighbor;
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
    if (info.neighbor != invalid_id<local_ordinal_type>() && info.neighbor == cell_lid)
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
    if (info.neighbor != invalid_id<local_ordinal_type>() && info.neighbor == cell_lid)
    {
        return info.unit_normal_from_neighbor;
    }

    throw std::invalid_argument("Cell is not adjacent to requested face.");
}

template<TpetraTypePack Pack>
inline bool Mesh<Pack>::is_exterior_face(local_ordinal_type fid) const
{
    return face(fid).neighbor == invalid_id<local_ordinal_type>();
}

template<TpetraTypePack Pack>
inline bool Mesh<Pack>::is_interior_face(local_ordinal_type fid) const
{
    return face(fid).neighbor != invalid_id<local_ordinal_type>();
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
inline auto Mesh<Pack>::global_to_local_cell(global_ordinal_type gid) const 
    -> local_ordinal_type
{
    const auto iter = d_cell_gid_to_lid.find(gid);
    if (iter == d_cell_gid_to_lid.end())
    {
        return invalid_id<local_ordinal_type>();
    }

    return iter->second;
}


template<TpetraTypePack Pack>
std::string Mesh<Pack>::make_face_key(ViewGO node_ids)
{
    return make_face_key(ArrGO(node_ids.begin(), node_ids.end()));
}


template<TpetraTypePack Pack>
std::string Mesh<Pack>::make_face_key(ArrGO node_ids)
{
    std::sort(node_ids.begin(), node_ids.end());

    std::ostringstream key;
    for (std::size_t i = 0; i < node_ids.size(); ++i)
    {
        if (i != 0)
        {
            key << ':';
        }
        key << node_ids[i];
    }

    return key.str();
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
