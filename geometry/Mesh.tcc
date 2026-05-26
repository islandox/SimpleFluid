/**
 * @file Mesh.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief
 * @version 0.1
 * @date 2026-05-25
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "Mesh.hh"

#include <Tpetra_Core.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace SimpleFluid
{

namespace detail
{

template <class Vec3>
inline Vec3 average(const std::vector<Vec3>& points)
{
    Vec3 result;
    if (points.empty())
    {
        return result;
    }

    for (const auto& point : points)
    {
        result = result + point;
    }

    return result / static_cast<typename Vec3::scalar_t>(points.size());
}

template <class Vec3>
inline real_t tetra_volume(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d)
{
    return std::abs((b - a).dot((c - a).cross(d - a))) / 6.0;
}

template <class Vec3>
inline real_t hex_volume(const std::vector<Vec3>& x)
{
    CHECK(x.size() >= 8);
    return tetra_volume(x[0], x[1], x[3], x[4])
         + tetra_volume(x[1], x[2], x[3], x[6])
         + tetra_volume(x[1], x[4], x[5], x[6])
         + tetra_volume(x[3], x[4], x[6], x[7])
         + tetra_volume(x[1], x[3], x[4], x[6]);
}

template <class Vec3>
inline real_t wedge_volume(const std::vector<Vec3>& x)
{
    CHECK(x.size() >= 6);
    return tetra_volume(x[0], x[1], x[2], x[3])
         + tetra_volume(x[1], x[2], x[4], x[3])
         + tetra_volume(x[2], x[4], x[5], x[3]);
}

template <class Vec3>
inline Vec3 face_area_vector(const std::vector<Vec3>& x)
{
    CHECK(x.size() == 3 || x.size() == 4);

    if (x.size() == 3)
    {
        return (x[1] - x[0]).cross(x[2] - x[0]) * 0.5;
    }

    return ((x[1] - x[0]).cross(x[2] - x[0])
          + (x[2] - x[0]).cross(x[3] - x[0])) * 0.5;
}

inline int vtk_cell_type(stk::topology topo)
{
    if (topo == stk::topology::HEX_8)
    {
        return 12;
    }
    if (topo == stk::topology::WEDGE_6)
    {
        return 13;
    }

    throw std::runtime_error("VTU export encountered an unsupported cell topology: "
                           + topo.name());
}

inline int mesh_cell_type(stk::topology topo)
{
    if (topo == stk::topology::HEX_8)
    {
        return 4;
    }
    if (topo == stk::topology::WEDGE_6)
    {
        return 3;
    }

    throw std::runtime_error("Unsupported cell topology: " + topo.name());
}

inline int mesh_face_type(std::size_t node_count)
{
    if (node_count == 3)
    {
        return 3;
    }
    if (node_count == 4)
    {
        return 4;
    }

    throw std::runtime_error("Unsupported face node count: "
                           + std::to_string(node_count));
}

} // namespace detail

template<TpetraTypePack Pack>
Mesh<Pack>::Mesh()
{
}

template<TpetraTypePack Pack>
void Mesh<Pack>::assemble()
{
    prefer_owned_face_owners();
    check_connectivity();
    create_maps();
    create_device_views();
}

template<TpetraTypePack Pack>
void Mesh<Pack>::export_vtu(const std::string&) const
{
    throw std::runtime_error("Mesh::export_vtu requires a concrete mesh backend.");
}


template<TpetraTypePack Pack>
void Mesh<Pack>::create_maps()
{
    ArrGO owned_gids;
    owned_gids.reserve(d_owned_cell_global_ids.size());
    for (const auto gid : d_owned_cell_global_ids)
    {
        owned_gids.push_back(static_cast<global_ordinal_type>(gid));
    }

    ArrGO overlap_gids;
    overlap_gids.reserve(d_cells.size());
    for (const auto& cell_info : d_cells)
    {
        overlap_gids.push_back(static_cast<global_ordinal_type>(cell_info.global_id));
    }

    const auto invalid_global_size = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const global_ordinal_type index_base = 0;
    const auto comm = Tpetra::getDefaultComm();

    d_owned_cell_map = Teuchos::rcp(new map_type(
        invalid_global_size,
        owned_gids.data(),
        detail::checked_size_to_ordinal<local_ordinal_type>(owned_gids.size(), "owned cell map"),
        index_base,
        comm));

    d_overlap_cell_map = Teuchos::rcp(new map_type(
        invalid_global_size,
        overlap_gids.data(),
        detail::checked_size_to_ordinal<local_ordinal_type>(overlap_gids.size(), "overlap cell map"),
        index_base,
        comm));
}

template<TpetraTypePack Pack>
void Mesh<Pack>::create_device_views()
{
    ArrGO cell_gid;
    ArrInt cell_type;
    ArrReal cell_volume_values;
    ArrVec3 cell_centroid_values;
    ArrLO cell_face_offset{0};
    ArrLO cell_face_ids;

    cell_gid.reserve(d_cells.size());
    cell_type.reserve(d_cells.size());
    cell_volume_values.reserve(d_cells.size());
    cell_centroid_values.reserve(d_cells.size());

    for (const auto& cell_info : d_cells)
    {
        cell_gid.push_back(static_cast<global_ordinal_type>(cell_info.global_id));
        cell_type.push_back(static_cast<int>(cell_info.type));
        cell_volume_values.push_back(cell_info.volume);
        cell_centroid_values.push_back(cell_info.center);

        cell_face_ids.insert(cell_face_ids.end(), cell_info.faces.begin(), cell_info.faces.end());
        cell_face_offset.push_back(detail::checked_size_to_ordinal<local_ordinal_type>(
            cell_face_ids.size(), "cell-face connectivity"));
    }

    ArrLO face_owner;
    ArrLO face_neighbor;
    ArrInt face_type;
    ArrInt face_patch_values;
    ArrReal face_area_values;
    ArrVec3 face_normal_values;
    ArrVec3 face_centroid_values;

    face_owner.reserve(d_faces.size());
    face_neighbor.reserve(d_faces.size());
    face_type.reserve(d_faces.size());
    face_patch_values.reserve(d_faces.size());
    face_area_values.reserve(d_faces.size());
    face_normal_values.reserve(d_faces.size());
    face_centroid_values.reserve(d_faces.size());

    for (const auto& face_info : d_faces)
    {
        face_owner.push_back(face_info.owner);
        face_neighbor.push_back(face_info.neighbor);
        face_type.push_back(static_cast<int>(face_info.type));
        face_patch_values.push_back(face_info.boundary_id);
        face_area_values.push_back(face_info.area);
        face_normal_values.push_back(face_info.unit_normal_from_owner);
        face_centroid_values.push_back(face_info.center);
    }

    ArrVec3 node_coord_values;
    ArrLO cell_node_offset{0};
    ArrLO cell_node_ids;
    ArrLO face_node_offset{0};
    ArrLO face_node_ids;

    for (const auto cell : d_cells)
    {
        for (auto node_id : cell.node_ids)
        {
            cell_node_ids.push_back(d_node_gid_to_lid.at(node_id));
        }

        cell_node_offset.push_back(detail::checked_size_to_ordinal<local_ordinal_type>(
            cell_node_ids.size(), "cell-node connectivity"));
    }

    for (const auto& face_info : d_faces)
    {
        for (const auto node_id : face_info.node_ids)
        {
            face_node_ids.push_back(d_node_gid_to_lid.at(node_id));
        }

        face_node_offset.push_back(detail::checked_size_to_ordinal<local_ordinal_type>(
            face_node_ids.size(), "face-node connectivity"));
    }

    d_device_views.cell_gid = make_vector_view("cell_gid", cell_gid);
    d_device_views.cell_type = make_vector_view("cell_type", cell_type);

    d_device_views.cell_volume = make_vector_view("cell_volume", cell_volume_values);
    d_device_views.cell_centroid = make_vectorV3D_view("cell_centroid", cell_centroid_values);

    d_device_views.cell_face_offset = make_vector_view("cell_face_offset", cell_face_offset);
    d_device_views.cell_face_ids = make_vector_view("cell_face_ids", cell_face_ids);

    d_device_views.face_owner = make_vector_view("face_owner", face_owner);
    d_device_views.face_neighbor = make_vector_view("face_neighbor", face_neighbor);
    d_device_views.face_type = make_vector_view("face_type", face_type);
    d_device_views.face_patch = make_vector_view("face_patch", face_patch_values);

    d_device_views.face_area = make_vector_view("face_area", face_area_values);
    d_device_views.face_area_vector = make_vectorV3D_view("face_area_vector", face_normal_values);
    d_device_views.face_centroid = make_vectorV3D_view("face_centroid", face_centroid_values);

    d_device_views.node_coord = make_vectorV3D_view("node_coord", node_coord_values);

    d_device_views.cell_node_offset = make_vector_view("cell_node_offset", cell_node_offset);
    d_device_views.cell_node_ids = make_vector_view("cell_node_ids", cell_node_ids);

    d_device_views.face_node_offset = make_vector_view("face_node_offset", face_node_offset);
    d_device_views.face_node_ids = make_vector_view("face_node_ids", face_node_ids);
}


template<TpetraTypePack Pack>
void Mesh<Pack>::prefer_owned_face_owners()
{
    for (auto& face_info : d_faces)
    {
        if (face_info.neighbor == invalid_id<local_ordinal_type>()) continue;

        const auto owner = face_info.owner;
        const auto neighbor = face_info.neighbor;
        if (!d_cells[static_cast<std::size_t>(owner)].owned
            && d_cells[static_cast<std::size_t>(neighbor)].owned)
        {
            face_info.owner = neighbor;
            face_info.neighbor = owner;
        }
    }
}


template<TpetraTypePack Pack>
void Mesh<Pack>::check_connectivity() const
{
    CHECK(d_owned_cell_ids.size() == d_owned_cell_global_ids.size());
    CHECK(d_owned_cell_ids.size() <= d_cells.size());

    for (size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto& cell = d_cells[lid];
        CHECK(cell.type != CellType::INVALID);

        const auto iter = d_cell_gid_to_lid.find(cell.global_id);
        CHECK(iter != d_cell_gid_to_lid.end());
        CHECK(iter->second == static_cast<local_ordinal_type>(lid));

        for (const auto fid : cell.faces)
        {
            CHECK(static_cast<std::size_t>(fid) < d_faces.size());
        }
    }

    for (const auto lid : d_owned_cell_ids)
    {
        CHECK(static_cast<std::size_t>(lid) < d_cells.size());
        CHECK(d_cells[static_cast<std::size_t>(lid)].owned);
    }

    for (std::size_t fid = 0; fid < d_faces.size(); ++fid)
    {
        const auto& face = d_faces[fid];
        CHECK(static_cast<std::size_t>(face.owner) < d_cells.size());
        CHECK(face.neighbor == invalid_id<local_ordinal_type>()
              || static_cast<std::size_t>(face.neighbor) < d_cells.size());
        CHECK((face.type == FaceType::TRIANGLE && face.node_ids.size() == 3)
         || (face.type == FaceType::QUAD && face.node_ids.size() == 4));
        CHECK(face.area >= 0.0);

        if (face.boundary_id != invalid_boundary_id)
        {
            const auto patch_iter = d_boundary_id_to_faces.find(face.boundary_id);
            CHECK(patch_iter != d_boundary_id_to_faces.end());
            CHECK(std::find(patch_iter->second.begin(), patch_iter->second.end(),
                            static_cast<local_ordinal_type>(fid)) != patch_iter->second.end());
        }
    }
}

} // namespace SimpleFluid
