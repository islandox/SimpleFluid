/**
 * @file Mesh_inline_functions.hh
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <Tpetra_Core.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
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

/**
 * @brief Checks that a size_t value can be safely converted to an Ordinal.
 * 
 * @tparam Ordinal 
 * @param value 
 * @param label 
 * @return Ordinal converted value
 */
template <class Ordinal>
inline Ordinal checked_size_to_ordinal(std::size_t value, std::string_view label)
{
    // This check is necessary to prevent overflow when converting from size_t to Ordinal.
    // Note that max() is reserved for invalid local ordinals, so the maximum valid value is one less than max().
    if(value >= static_cast<std::size_t>(std::numeric_limits<Ordinal>::max()))
    {
        throw std::overflow_error("Value for " + std::string(label) + " exceeds maximum representable by ordinal type.");
    }

    return static_cast<Ordinal>(value);
}

template <class Ordinal>
inline void replace_crs_row(Ordinal row,
                            std::vector<Ordinal>& offsets,
                            std::vector<Ordinal>& ids,
                            const std::vector<Ordinal>& values)
{
    const auto row_index = static_cast<std::size_t>(row);
    CHECK(row_index + 1 < offsets.size());

    const auto begin = static_cast<std::size_t>(offsets[row_index]);
    const auto end = static_cast<std::size_t>(offsets[row_index + 1]);
    CHECK(begin <= end);
    CHECK(end <= ids.size());

    const auto old_count = end - begin;
    const auto new_count = values.size();
    const auto new_total = ids.size() - old_count + new_count;
    checked_size_to_ordinal<Ordinal>(new_total, "CRS storage size");

    const auto erase_begin = ids.begin() + static_cast<typename std::vector<Ordinal>::difference_type>(begin);
    const auto erase_end = ids.begin() + static_cast<typename std::vector<Ordinal>::difference_type>(end);
    const auto insert_pos = ids.erase(erase_begin, erase_end);
    ids.insert(insert_pos, values.begin(), values.end());

    if (new_count >= old_count)
    {
        const auto delta = checked_size_to_ordinal<Ordinal>(new_count - old_count, "CRS row size delta");
        for (std::size_t i = row_index + 1; i < offsets.size(); ++i)
        {
            offsets[i] += delta;
        }
    }
    else
    {
        const auto delta = checked_size_to_ordinal<Ordinal>(old_count - new_count, "CRS row size delta");
        for (std::size_t i = row_index + 1; i < offsets.size(); ++i)
        {
            CHECK(offsets[i] >= delta);
            offsets[i] -= delta;
        }
    }
}

} // namespace detail

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::add_cell(global_ordinal_type gid, CellType type,
                                 real_t volume, const Vec3& centroid) -> local_ordinal_type
{
    if (d_cell_gid_to_lid.contains(gid))
    {
        throw std::invalid_argument("Duplicate cell global id: " + std::to_string(gid));
    }

    const auto lid = detail::checked_size_to_ordinal<local_ordinal_type>(d_cell_gid.size(), "cell count");

    if (d_cell_face_offset.empty())
    {
        d_cell_face_offset.push_back(0);
    }
    if (d_cell_node_offset.empty())
    {
        d_cell_node_offset.push_back(0);
    }

    d_cell_gid.push_back(gid);
    d_cell_gid_to_lid.emplace(gid, lid);

    d_cell_type.push_back(static_cast<int>(type));
    d_cell_volume.push_back(volume);
    d_cell_centroid.push_back(centroid);

    d_cell_face_offset.push_back(d_cell_face_offset.back());
    d_cell_node_offset.push_back(d_cell_node_offset.back());

    d_num_local_cells = detail::checked_size_to_ordinal<local_ordinal_type>(d_cell_gid.size(), "cell count");
    d_num_owned_cells = d_num_local_cells;
    d_num_global_cells = static_cast<global_ordinal_type>(d_num_owned_cells);

    return lid;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::add_face(local_ordinal_type owner, local_ordinal_type neighbor,
                                 FaceType type, int patch_id,
                                 real_t area, const Vec3& area_vector,
                                 const Vec3& centroid) -> local_ordinal_type
{
    check_cell(owner);

    const auto invalid_lid = detail::invalid_local_ordinal<local_ordinal_type>();
    if (neighbor != invalid_lid)
    {
        check_cell(neighbor);
    }

    const auto lid = detail::checked_size_to_ordinal<local_ordinal_type>(d_face_owner.size(), "face count");

    if (d_face_node_offset.empty())
    {
        d_face_node_offset.push_back(0);
    }

    d_face_owner.push_back(owner);
    d_face_neighbor.push_back(neighbor);

    d_face_type.push_back(static_cast<int>(type));
    d_face_patch.push_back(patch_id);

    d_face_area.push_back(area);
    d_face_area_vector.push_back(area_vector);
    d_face_centroid.push_back(centroid);

    d_face_node_offset.push_back(d_face_node_offset.back());

    d_num_faces = detail::checked_size_to_ordinal<local_ordinal_type>(d_face_owner.size(), "face count");

    return lid;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::add_node(const Vec3& coord) -> local_ordinal_type
{
    const auto lid = detail::checked_size_to_ordinal<local_ordinal_type>(d_node_coord.size(), "node count");

    d_node_coord.push_back(coord);
    d_num_nodes = detail::checked_size_to_ordinal<local_ordinal_type>(d_node_coord.size(), "node count");

    return lid;
}

template<TpetraTypePack Pack>
inline void Mesh<Pack>::set_cell_faces(local_ordinal_type cell_lid, const ArrLO& face_ids)
{
    check_cell(cell_lid);
    for (const auto face_id : face_ids)
    {
        check_face(face_id);
    }

    detail::replace_crs_row(cell_lid, d_cell_face_offset, d_cell_face_ids, face_ids);
}

template<TpetraTypePack Pack>
inline void Mesh<Pack>::set_cell_nodes(local_ordinal_type cell_lid, const ArrLO& node_ids)
{
    check_cell(cell_lid);
    for (const auto node_id : node_ids)
    {
        check_node(node_id);
    }

    detail::replace_crs_row(cell_lid, d_cell_node_offset, d_cell_node_ids, node_ids);
}

template<TpetraTypePack Pack>
inline void Mesh<Pack>::set_face_nodes(local_ordinal_type face_lid, const ArrLO& node_ids)
{
    check_face(face_lid);
    for (const auto node_id : node_ids)
    {
        check_node(node_id);
    }

    detail::replace_crs_row(face_lid, d_face_node_offset, d_face_node_ids, node_ids);
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::cell_global_id(local_ordinal_type lid) const -> global_ordinal_type
{
    check_cell(lid);
    return d_cell_gid[lid];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::cell_local_id(global_ordinal_type gid) const -> local_ordinal_type
{
    const auto iter = d_cell_gid_to_lid.find(gid);
    if (iter == d_cell_gid_to_lid.end())
    {
        throw std::out_of_range("Cell global id not found: " + std::to_string(gid));
    }

    return iter->second;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::cell_type(local_ordinal_type lid) const -> CellType
{
    check_cell(lid);
    return static_cast<CellType>(d_cell_type[lid]);
}

template<TpetraTypePack Pack>
inline real_t Mesh<Pack>::cell_volume(local_ordinal_type lid) const
{
    check_cell(lid);
    return d_cell_volume[lid];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::cell_centroid(local_ordinal_type lid) const -> const Vec3&
{
    check_cell(lid);
    return d_cell_centroid[lid];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face_owner(local_ordinal_type fid) const -> local_ordinal_type
{
    check_face(fid);
    return d_face_owner[fid];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face_neighbor(local_ordinal_type fid) const -> local_ordinal_type
{
    check_face(fid);
    return d_face_neighbor[fid];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face_type(local_ordinal_type fid) const -> FaceType
{
    check_face(fid);
    return static_cast<FaceType>(d_face_type[fid]);
}

template<TpetraTypePack Pack>
inline int Mesh<Pack>::face_patch(local_ordinal_type fid) const
{
    check_face(fid);
    return d_face_patch[fid];
}

template<TpetraTypePack Pack>
inline real_t Mesh<Pack>::face_area(local_ordinal_type fid) const
{
    check_face(fid);
    return d_face_area[fid];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face_area_vector(local_ordinal_type fid) const -> const Vec3&
{
    check_face(fid);
    return d_face_area_vector[fid];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face_centroid(local_ordinal_type fid) const -> const Vec3&
{
    check_face(fid);
    return d_face_centroid[fid];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::cell_face_begin(local_ordinal_type lid) const -> local_ordinal_type
{
    check_cell(lid);
    return d_cell_face_offset[lid];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::cell_face_end(local_ordinal_type lid) const -> local_ordinal_type
{
    check_cell(lid);
    return d_cell_face_offset[static_cast<std::size_t>(lid) + 1];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::cell_face(local_ordinal_type cell_lid, local_ordinal_type i) const -> local_ordinal_type
{
    const auto begin = cell_face_begin(cell_lid);
    const auto end = cell_face_end(cell_lid);
    CHECK(i < end - begin);
    return d_cell_face_ids[begin + i];
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
