/**
 * @file Mesh.ipp
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Inline method definitions and detail helpers for the Mesh class.
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

/**
 * @brief Safely convert a size_t to an ordinal type with overflow checking.
 *
 * @tparam Ordinal Target ordinal type.
 * @param value Size value to convert.
 * @param label Descriptive label used in error messages.
 * @return The value cast to the ordinal type.
 * @throws std::overflow_error if the value exceeds the ordinal type's maximum.
 */
template <class Ordinal>
inline Ordinal checked_size_to_ordinal(std::size_t value, std::string_view label)
{
    CHECK(value < static_cast<std::size_t>(std::numeric_limits<Ordinal>::max()),
          "Value for " + std::string(label)
          + " exceeds maximum representable by ordinal type.",
          std::overflow_error);

    return static_cast<Ordinal>(value);
}

} // namespace detail

template<TpetraTypePack Pack>
inline void Mesh<Pack>::check_cell(local_ordinal_type lid) const
{
#if !defined(NDEBUG) || defined(SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS)
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        if (lid < 0)
        {
            throw std::out_of_range("Cell local id cannot be negative: "
                                  + std::to_string(lid));
        }
    }

    if (static_cast<std::size_t>(lid) >= num_local_cells())
    {
        throw std::out_of_range("Cell local id is out of bounds: "
                              + std::to_string(lid));
    }
#else
    (void)lid;
#endif
}

template<TpetraTypePack Pack>
inline void Mesh<Pack>::check_face(local_ordinal_type lid) const
{
#if !defined(NDEBUG) || defined(SIMPLEFLUID_ENABLE_RUNTIME_BOUNDS_CHECKS)
    if constexpr (std::is_signed_v<local_ordinal_type>)
    {
        if (lid < 0)
        {
            throw std::out_of_range("Face local id cannot be negative: "
                                  + std::to_string(lid));
        }
    }

    if (static_cast<std::size_t>(lid) >= num_faces())
    {
        throw std::out_of_range("Face local id is out of bounds: "
                              + std::to_string(lid));
    }
#else
    (void)lid;
#endif
}

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
    check_cell(lid);
    const auto index = static_cast<std::size_t>(lid);
    if (index < d_owned_cell_global_ids.size())
    {
        return d_owned_cell_global_ids[index];
    }

    return d_ghost_cell_global_ids[index - d_owned_cell_global_ids.size()];
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::node_coord(global_ordinal_type node_gid) const -> const Vec3&
{
    const auto iter = d_node_gid_to_lid.find(node_gid);
    if (iter == d_node_gid_to_lid.end())
    {
        throw std::out_of_range("Node global id is not local to this mesh: "
                              + std::to_string(node_gid));
    }

    return d_node_coords[static_cast<std::size_t>(iter->second)];
}

template<TpetraTypePack Pack>
inline bool Mesh<Pack>::is_owned_cell(local_ordinal_type lid) const
{
    return cell(lid).owned;
}

template<TpetraTypePack Pack>
inline bool Mesh<Pack>::is_owned_face(local_ordinal_type fid) const
{
    return is_owned_cell(owner_cell(fid));
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::faces(local_ordinal_type cell_lid) const -> const ViewLO&
{
    return cell(cell_lid).faces;
}

template<TpetraTypePack Pack>
inline auto Mesh<Pack>::face_distances(local_ordinal_type cell_lid) const -> const ViewReal&
{
    return cell(cell_lid).face_distances;
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
inline real_t Mesh<Pack>::face_cell_center_distance(local_ordinal_type fid) const
{
    return face(fid).cell_center_distance;
}

/**
 * @brief Compute the distance from a given cell center to a face center.
 *
 * @tparam Pack Tpetra type pack.
 * @param fid Face local ID.
 * @param cell_lid Cell local ID.
 * @return Distance from the cell center to the face center.
 * @throws std::invalid_argument if the cell is not adjacent to the face.
 */
template<TpetraTypePack Pack>
inline real_t Mesh<Pack>::cell_to_face_distance(local_ordinal_type fid,
                                                local_ordinal_type cell_lid) const
{
    const auto& info = face(fid);
    check_cell(cell_lid);

    if (info.owner == cell_lid)
    {
        return info.owner_to_face_distance;
    }
    if (info.neighbor != invalid_id<local_ordinal_type>() && info.neighbor == cell_lid)
    {
        return info.neighbor_to_face_distance;
    }

    throw std::invalid_argument("Cell is not adjacent to requested face.");
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

/**
 * @brief Return the unit normal of a face pointing outward from a given cell.
 *
 * @tparam Pack Tpetra type pack.
 * @param fid Face local ID.
 * @param cell_lid Cell local ID.
 * @return Const reference to the outward-pointing unit normal vector.
 * @throws std::invalid_argument if the cell is not adjacent to the face.
 */
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

/**
 * @brief Retrieve the boundary name for a boundary face.
 *
 * @tparam Pack Tpetra type pack.
 * @param fid Face local ID.
 * @return Const reference to the boundary name string.
 * @throws std::out_of_range if the face is not a boundary face.
 */
template<TpetraTypePack Pack>
inline auto Mesh<Pack>::boundary_name(local_ordinal_type fid) const
    -> const std::string&
{
    if (!is_boundary_face(fid))
    {
        throw std::out_of_range("Requested face is not a boundary face.");
    }

    return boundary_patch_name(boundary_id(fid));
}

/**
 * @brief Retrieve the boundary name for a boundary patch.
 *
 * @tparam Pack Tpetra type pack.
 * @param patch_id Boundary patch ID.
 * @return Const reference to the boundary name string.
 * @throws std::out_of_range if the boundary patch is not found.
 */
template<TpetraTypePack Pack>
inline auto Mesh<Pack>::boundary_patch_name(int patch_id) const
    -> const std::string&
{
    const auto iter = d_boundary_id_to_name.find(patch_id);
    if (iter == d_boundary_id_to_name.end())
    {
        throw std::out_of_range("Requested boundary patch is not found.");
    }
    return d_boundary_id_to_name.at(patch_id);
}

/**
 * @brief Retrieve the list of face IDs for a boundary patch.
 *
 * @tparam Pack Tpetra type pack.
 * @param patch_id Boundary patch ID.
 * @return Const reference to the list of face IDs.
 * @throws std::out_of_range if the boundary patch is not found.
 */
template<TpetraTypePack Pack>
inline auto Mesh<Pack>::boundary_face_patch(int patch_id) const -> const BoundaryFacePatch&
{
    const auto iter = d_boundary_id_to_face_patch.find(patch_id);
    if (iter == d_boundary_id_to_face_patch.end())
    {
        throw std::out_of_range("Requested boundary patch is not found.");
    }
    return iter->second;
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


/**
 * @brief Create a unique string key from a set of face node global IDs.
 *
 * Sorts the node IDs and concatenates them with ':' separators.
 *
 * @tparam Pack Tpetra type pack.
 * @param node_ids View of node global IDs defining the face.
 * @return Sorted, colon-separated string key uniquely identifying the face.
 */
template<TpetraTypePack Pack>
std::string Mesh<Pack>::make_face_key(ViewGO node_ids)
{
    return make_face_key(ArrGO(node_ids.begin(), node_ids.end()));
}


/**
 * @brief Create a unique string key from a sorted vector of face node global IDs.
 *
 * Concatenates sorted node IDs with ':' separators to form a unique face key.
 *
 * @tparam Pack Tpetra type pack.
 * @param node_ids Vector of node global IDs (sorted in place).
 * @return Sorted, colon-separated string key uniquely identifying the face.
 */
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

/**
 * @brief Create a 1D Kokkos device view from a host vector.
 *
 * Copies the host data to a mirror view, then deep-copies to the device.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam T Data type stored in the vector.
 * @param name Label for the Kokkos view.
 * @param data Host vector to copy to the device.
 * @return 1D Kokkos view on the device containing a const copy of the data.
 */
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

/**
 * @brief Create a 2D Kokkos device view (Nx3) from a vector of Vec3.
 *
 * Copies the x, y, z components of each Vec3 to a 2D Kokkos view on the device.
 *
 * @tparam Pack Tpetra type pack.
 * @param name Label for the Kokkos view.
 * @param data Host vector of 3D vectors to copy to the device.
 * @return N×3 Kokkos view on the device containing a const copy of the data.
 */
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
