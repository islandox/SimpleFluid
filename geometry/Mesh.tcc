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
#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_util/parallel/Parallel.hpp>

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

inline std::shared_ptr<stk::mesh::MetaData> make_stk_meta_data()
{
    stk::mesh::MeshBuilder builder(stk::parallel_machine_world());
    builder.set_spatial_dimension(3);
    return builder.create_meta_data();
}

inline std::unique_ptr<stk::mesh::BulkData>
make_stk_bulk_data(const std::shared_ptr<stk::mesh::MetaData>& meta)
{
    stk::mesh::MeshBuilder builder(stk::parallel_machine_world());
    return builder.create(meta);
}

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
    : d_meta_storage(detail::make_stk_meta_data())
    , d_bulk_storage(detail::make_stk_bulk_data(d_meta_storage))
    , d_io(stk::parallel_machine_world())
    , d_meta(*d_meta_storage)
    , d_bulk(*d_bulk_storage)
{
    d_io.set_bulk_data(d_bulk);
}

template<TpetraTypePack Pack>
void Mesh<Pack>::assemble()
{
    initialize_boundary_id_maps();
    build_cell_list();
    compute_cell_geometry();
    build_face_table();
    prefer_owned_face_owners();
    compute_face_geometry();
    assign_boundary_ids_from_stk_side_parts();
    check_connectivity();
    create_maps();
    create_device_views();
}

template<TpetraTypePack Pack>
void Mesh<Pack>::export_vtu(const std::string& filename) const
{
    check_connectivity();

    std::unordered_map<EntityId, local_ordinal_type> node_lid;
    ArrVec3 node_coords;
    ArrLO cell_node_offset{0};
    ArrLO cell_node_ids;

    auto append_node = [&](stk::mesh::Entity node) -> local_ordinal_type
    {
        const auto node_id = d_bulk.identifier(node);
        const auto iter = node_lid.find(node_id);
        if (iter != node_lid.end())
        {
            return iter->second;
        }

        const auto lid = detail::checked_size_to_ordinal<local_ordinal_type>(
            node_coords.size(), "VTU node count");
        node_lid.emplace(node_id, lid);
        node_coords.push_back(node_coord(node));
        return lid;
    };

    for (const auto elem : d_cell_entities)
    {
        const auto num_nodes = d_bulk.num_nodes(elem);
        const auto* nodes = d_bulk.begin_nodes(elem);

        for (unsigned i = 0; i < num_nodes; ++i)
        {
            cell_node_ids.push_back(append_node(nodes[i]));
        }

        cell_node_offset.push_back(detail::checked_size_to_ordinal<local_ordinal_type>(
            cell_node_ids.size(), "VTU cell-node connectivity"));
    }

    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error("Failed to open VTU output file: " + filename);
    }

    out << std::setprecision(std::numeric_limits<real_t>::max_digits10);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << node_coords.size()
        << "\" NumberOfCells=\"" << d_cells.size() << "\">\n";

    out << "      <PointData/>\n";
    out << "      <CellData>\n";
    out << "        <DataArray type=\"Int64\" Name=\"cell_gid\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        out << d_cells[lid].global_id << (lid + 1 == d_cells.size() ? "" : " ");
    }
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"cell_type\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        out << detail::mesh_cell_type(d_cells[lid].topology)
            << (lid + 1 == d_cells.size() ? "" : " ");
    }
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Float64\" Name=\"cell_volume\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        out << d_cells[lid].volume << (lid + 1 == d_cells.size() ? "" : " ");
    }
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Float64\" Name=\"cell_centroid\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& cell_info : d_cells)
    {
        out << "          " << cell_info.center.x << " "
            << cell_info.center.y << " " << cell_info.center.z << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </CellData>\n";

    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& coord : node_coords)
    {
        out << "          " << coord.x << " " << coord.y << " " << coord.z << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    out << "      <Cells>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto begin = static_cast<std::size_t>(cell_node_offset[lid]);
        const auto end = static_cast<std::size_t>(cell_node_offset[lid + 1]);

        out << "          ";
        for (std::size_t i = begin; i < end; ++i)
        {
            out << cell_node_ids[i] << (i + 1 == end ? "" : " ");
        }
        out << "\n";
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        out << cell_node_offset[lid + 1] << (lid + 1 == d_cells.size() ? "" : " ");
    }
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        out << detail::vtk_cell_type(d_cells[lid].topology)
            << (lid + 1 == d_cells.size() ? "" : " ");
    }
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    if (!out)
    {
        throw std::runtime_error("Failed while writing VTU output file: " + filename);
    }
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
        cell_type.push_back(detail::mesh_cell_type(cell_info.topology));
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
        face_neighbor.push_back(face_info.neighbor.value_or(
            detail::invalid_local_ordinal<local_ordinal_type>()));
        face_type.push_back(detail::mesh_face_type(face_info.node_ids.size()));
        face_patch_values.push_back(face_info.boundary_id);
        face_area_values.push_back(face_info.area);
        face_normal_values.push_back(face_info.unit_normal_from_owner);
        face_centroid_values.push_back(face_info.center);
    }

    std::unordered_map<EntityId, local_ordinal_type> node_id_to_lid;
    ArrVec3 node_coord_values;
    ArrLO cell_node_offset{0};
    ArrLO cell_node_ids;
    ArrLO face_node_offset{0};
    ArrLO face_node_ids;

    auto get_node_lid = [&](stk::mesh::Entity node) -> local_ordinal_type
    {
        const auto node_id = d_bulk.identifier(node);
        const auto iter = node_id_to_lid.find(node_id);
        if (iter != node_id_to_lid.end())
        {
            return iter->second;
        }

        const auto lid = detail::checked_size_to_ordinal<local_ordinal_type>(
            node_coord_values.size(), "node count");
        node_id_to_lid.emplace(node_id, lid);
        node_coord_values.push_back(node_coord(node));
        return lid;
    };

    auto get_node_lid_by_id = [&](EntityId node_id) -> local_ordinal_type
    {
        const auto iter = node_id_to_lid.find(node_id);
        if (iter != node_id_to_lid.end())
        {
            return iter->second;
        }

        const auto node = d_bulk.get_entity(stk::topology::NODE_RANK, node_id);
        if (!node.is_local_offset_valid())
        {
            throw std::runtime_error("Face references missing node id: "
                                   + std::to_string(node_id));
        }

        return get_node_lid(node);
    };

    for (const auto elem : d_cell_entities)
    {
        const auto num_nodes = d_bulk.num_nodes(elem);
        const auto* nodes = d_bulk.begin_nodes(elem);

        for (unsigned i = 0; i < num_nodes; ++i)
        {
            cell_node_ids.push_back(get_node_lid(nodes[i]));
        }

        cell_node_offset.push_back(detail::checked_size_to_ordinal<local_ordinal_type>(
            cell_node_ids.size(), "cell-node connectivity"));
    }

    for (const auto& face_info : d_faces)
    {
        for (const auto node_id : face_info.node_ids)
        {
            face_node_ids.push_back(get_node_lid_by_id(node_id));
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
void Mesh<Pack>::build_cell_list()
{
    d_cell_entities.clear();
    d_cells.clear();
    d_faces.clear();
    d_owned_cell_ids.clear();
    d_owned_cell_global_ids.clear();
    d_cell_gid_to_lid.clear();
    face_key_to_face_.clear();
    d_boundary_id_to_faces.clear();

    d_coord_field = nullptr;
    if (const auto* coord_base = d_meta.coordinate_field(); coord_base != nullptr)
    {
        d_coord_field = dynamic_cast<stk::mesh::Field<double>*>(
            const_cast<stk::mesh::FieldBase*>(coord_base));
    }

    if (d_coord_field == nullptr && !d_meta.coordinate_field_name().empty())
    {
        d_coord_field = d_meta.get_field<double>(stk::topology::NODE_RANK,
                                                 d_meta.coordinate_field_name());
    }

    if (d_coord_field == nullptr)
    {
        d_coord_field = d_meta.get_field<double>(stk::topology::NODE_RANK, "coordinates");
    }

    auto append_bucket_cells = [&](bool owned)
    {
        const auto& buckets = d_bulk.buckets(stk::topology::ELEMENT_RANK);
        for (const auto* bucket : buckets)
        {
            if (bucket == nullptr || bucket->owned() != owned)
            {
                continue;
            }

            const auto topo = bucket->topology();
            if (!is_supported_volume_topology(topo))
            {
                continue;
            }

            for (const auto elem : *bucket)
            {
                const auto lid = detail::checked_size_to_ordinal<local_ordinal_type>(
                    d_cells.size(), "cell count");
                const auto gid = d_bulk.identifier(elem);

                if (!d_cell_gid_to_lid.emplace(gid, lid).second)
                {
                    throw std::runtime_error("Duplicate cell global id: "
                                           + std::to_string(gid));
                }

                CellInfo cell_info;
                cell_info.global_id = gid;
                cell_info.topology = topo;
                cell_info.owned = owned;

                d_cell_entities.push_back(elem);
                d_cells.push_back(std::move(cell_info));

                if (owned)
                {
                    d_owned_cell_ids.push_back(lid);
                    d_owned_cell_global_ids.push_back(gid);
                }
            }
        }
    };

    append_bucket_cells(true);
    append_bucket_cells(false);
}

template<TpetraTypePack Pack>
void Mesh<Pack>::compute_cell_geometry()
{
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto coords = element_node_coords(d_cell_entities[lid]);
        auto& cell_info = d_cells[lid];

        cell_info.center = detail::average(coords);

        if (cell_info.topology == stk::topology::HEX_8)
        {
            cell_info.volume = detail::hex_volume(coords);
        }
        else if (cell_info.topology == stk::topology::WEDGE_6)
        {
            cell_info.volume = detail::wedge_volume(coords);
        }
        else
        {
            throw std::runtime_error("Unsupported cell topology: "
                                   + cell_info.topology.name());
        }
    }
}

template<TpetraTypePack Pack>
void Mesh<Pack>::build_face_table()
{
    d_faces.clear();
    face_key_to_face_.clear();
    for (auto& cell_info : d_cells)
    {
        cell_info.faces.clear();
    }

    for (std::size_t cell_index = 0; cell_index < d_cells.size(); ++cell_index)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell_index);
        const auto elem = d_cell_entities[cell_index];
        const auto topo = d_cells[cell_index].topology;
        const auto* elem_nodes = d_bulk.begin_nodes(elem);
        const auto num_elem_nodes = d_bulk.num_nodes(elem);

        for (unsigned side = 0; side < topo.num_sides(); ++side)
        {
            const auto ordinals = side_node_ordinals(topo, side);
            std::vector<EntityId> face_node_ids;
            face_node_ids.reserve(ordinals.size());

            for (const auto ordinal : ordinals)
            {
                CHECK(ordinal < num_elem_nodes);
                face_node_ids.push_back(d_bulk.identifier(elem_nodes[ordinal]));
            }

            const auto key = make_face_key(face_node_ids);
            const auto iter = face_key_to_face_.find(key);
            if (iter == face_key_to_face_.end())
            {
                const auto fid = detail::checked_size_to_ordinal<local_ordinal_type>(
                    d_faces.size(), "face count");

                FaceInfo face_info;
                face_info.node_ids = std::move(face_node_ids);
                face_info.owner = cell_lid;

                d_faces.push_back(std::move(face_info));
                face_key_to_face_.emplace(key, fid);
                d_cells[cell_index].faces.push_back(fid);
            }
            else
            {
                auto& face_info = d_faces[static_cast<std::size_t>(iter->second)];
                if (face_info.neighbor)
                {
                    throw std::runtime_error("Non-manifold face encountered.");
                }

                face_info.neighbor = cell_lid;
                d_cells[cell_index].faces.push_back(iter->second);
            }
        }
    }
}

template<TpetraTypePack Pack>
void Mesh<Pack>::prefer_owned_face_owners()
{
    for (auto& face_info : d_faces)
    {
        if (!face_info.neighbor)
        {
            continue;
        }

        const auto owner = face_info.owner;
        const auto neighbor = *face_info.neighbor;
        if (!d_cells[static_cast<std::size_t>(owner)].owned
            && d_cells[static_cast<std::size_t>(neighbor)].owned)
        {
            face_info.owner = neighbor;
            face_info.neighbor = owner;
        }
    }
}

template<TpetraTypePack Pack>
void Mesh<Pack>::compute_face_geometry()
{
    for (auto& face_info : d_faces)
    {
        std::vector<Vec3> coords;
        coords.reserve(face_info.node_ids.size());

        for (const auto node_id : face_info.node_ids)
        {
            coords.push_back(node_coord_by_id(node_id));
        }

        face_info.center = detail::average(coords);

        auto area_vector = detail::face_area_vector(coords);
        face_info.area = area_vector.norm();
        if (face_info.area <= 0.0)
        {
            throw std::runtime_error("Degenerate face encountered.");
        }

        auto normal = area_vector / face_info.area;
        const auto owner_to_face = face_info.center
                                 - d_cells[static_cast<std::size_t>(face_info.owner)].center;
        if (normal.dot(owner_to_face) < 0.0)
        {
            normal = normal * -1.0;
        }

        face_info.unit_normal_from_owner = normal;
        face_info.unit_normal_from_neighbor = normal * -1.0;
    }
}

template<TpetraTypePack Pack>
void Mesh<Pack>::assign_boundary_ids_from_stk_side_parts()
{
    d_boundary_id_to_faces.clear();

    const auto side_rank = d_meta.side_rank();
    std::unordered_map<std::string, const stk::mesh::Part*> face_key_to_part;

    if (side_rank != stk::topology::INVALID_RANK)
    {
        const auto& side_buckets = d_bulk.buckets(side_rank);
        for (const auto* bucket : side_buckets)
        {
            if (bucket == nullptr)
            {
                continue;
            }

            const auto* boundary_part = choose_boundary_part(*bucket, side_rank);
            if (boundary_part == nullptr)
            {
                continue;
            }

            for (const auto side_entity : *bucket)
            {
                const auto num_nodes = d_bulk.num_nodes(side_entity);
                const auto* nodes = d_bulk.begin_nodes(side_entity);

                std::vector<EntityId> node_ids;
                node_ids.reserve(num_nodes);
                for (unsigned i = 0; i < num_nodes; ++i)
                {
                    node_ids.push_back(d_bulk.identifier(nodes[i]));
                }

                face_key_to_part.emplace(make_face_key(std::move(node_ids)), boundary_part);
            }
        }
    }

    for (std::size_t fid = 0; fid < d_faces.size(); ++fid)
    {
        auto& face_info = d_faces[fid];
        face_info.boundary_id = invalid_boundary_id;
        face_info.boundary_name.clear();

        if (face_info.neighbor)
        {
            continue;
        }

        const auto iter = face_key_to_part.find(make_face_key(face_info.node_ids));
        if (iter == face_key_to_part.end())
        {
            continue;
        }

        face_info.boundary_name = iter->second->name();
        face_info.boundary_id = get_or_create_boundary_id(face_info.boundary_name);
        d_boundary_id_to_faces[face_info.boundary_id].push_back(
            static_cast<local_ordinal_type>(fid));
    }
}

template<TpetraTypePack Pack>
void Mesh<Pack>::initialize_boundary_id_maps()
{
    d_boundary_id_to_name.clear();
    d_boundary_name_to_id.clear();
    d_boundary_id_to_faces.clear();
    d_next_boundary_id = 1;

    for (const auto& [name, id] : d_options.boundary_name_to_id)
    {
        if (id == invalid_boundary_id)
        {
            throw std::invalid_argument("Boundary id cannot use invalid_boundary_id.");
        }

        const auto [name_iter, inserted_name] = d_boundary_name_to_id.emplace(name, id);
        if (!inserted_name && name_iter->second != id)
        {
            throw std::invalid_argument("Conflicting boundary id for boundary name: " + name);
        }

        const auto [id_iter, inserted_id] = d_boundary_id_to_name.emplace(id, name);
        if (!inserted_id && id_iter->second != name)
        {
            throw std::invalid_argument("Boundary id assigned to multiple names: "
                                      + std::to_string(id));
        }

        d_next_boundary_id = std::max(d_next_boundary_id, id + 1);
    }
}

template<TpetraTypePack Pack>
int Mesh<Pack>::get_or_create_boundary_id(const std::string& name)
{
    const auto iter = d_boundary_name_to_id.find(name);
    if (iter != d_boundary_name_to_id.end())
    {
        return iter->second;
    }

    if (!d_options.auto_assign_boundary_ids)
    {
        throw std::out_of_range("Boundary name has no configured id: " + name);
    }

    const auto id = d_next_boundary_id++;
    d_boundary_name_to_id.emplace(name, id);
    d_boundary_id_to_name.emplace(id, name);
    return id;
}

template<TpetraTypePack Pack>
bool Mesh<Pack>::is_supported_volume_topology(stk::topology topo)
{
    return topo == stk::topology::HEX_8 || topo == stk::topology::WEDGE_6;
}

template<TpetraTypePack Pack>
auto Mesh<Pack>::side_node_ordinals(stk::topology topo,
                                    unsigned side_ordinal) -> std::vector<unsigned>
{
    if (side_ordinal >= topo.num_sides())
    {
        throw std::out_of_range("Side ordinal is out of range for topology "
                              + topo.name() + ".");
    }

    std::vector<unsigned> ordinals(topo.side_topology(side_ordinal).num_nodes());
    topo.side_node_ordinals(side_ordinal, ordinals.begin());
    return ordinals;
}

template<TpetraTypePack Pack>
std::string Mesh<Pack>::make_face_key(std::vector<EntityId> node_ids)
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
bool Mesh<Pack>::is_candidate_boundary_part(const stk::mesh::Part& part,
                                            stk::mesh::EntityRank side_rank) const
{
    if (part.primary_entity_rank() != side_rank)
    {
        return false;
    }
    if (stk::mesh::is_auto_declared_part(part) || stk::mesh::is_topology_root_part(part))
    {
        return false;
    }
    if (d_boundary_name_to_id.contains(part.name()))
    {
        return true;
    }

    return d_options.auto_assign_boundary_ids;
}

template<TpetraTypePack Pack>
auto Mesh<Pack>::choose_boundary_part(const stk::mesh::Bucket& bucket,
                                      stk::mesh::EntityRank side_rank) const
    -> const stk::mesh::Part*
{
    const stk::mesh::Part* first_candidate = nullptr;
    const stk::mesh::Part* first_io_surface = nullptr;

    for (const auto* part : bucket.supersets())
    {
        if (part == nullptr || !is_candidate_boundary_part(*part, side_rank))
        {
            continue;
        }

        if (d_boundary_name_to_id.contains(part->name()))
        {
            return part;
        }
        if (first_io_surface == nullptr && stk::io::is_part_surface_io_part(*part))
        {
            first_io_surface = part;
        }
        if (first_candidate == nullptr)
        {
            first_candidate = part;
        }
    }

    return first_io_surface != nullptr ? first_io_surface : first_candidate;
}

template<TpetraTypePack Pack>
void Mesh<Pack>::check_connectivity() const
{
    CHECK(d_cell_entities.size() == d_cells.size());
    CHECK(d_owned_cell_ids.size() == d_owned_cell_global_ids.size());
    CHECK(d_owned_cell_ids.size() <= d_cells.size());

    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto& cell_info = d_cells[lid];
        CHECK(is_supported_volume_topology(cell_info.topology));

        const auto iter = d_cell_gid_to_lid.find(cell_info.global_id);
        CHECK(iter != d_cell_gid_to_lid.end());
        CHECK(iter->second == static_cast<local_ordinal_type>(lid));

        for (const auto fid : cell_info.faces)
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
        const auto& face_info = d_faces[fid];
        CHECK(static_cast<std::size_t>(face_info.owner) < d_cells.size());
        CHECK(!face_info.neighbor
              || static_cast<std::size_t>(*face_info.neighbor) < d_cells.size());
        CHECK(face_info.node_ids.size() == 3 || face_info.node_ids.size() == 4);
        CHECK(face_info.area >= 0.0);

        if (face_info.boundary_id != invalid_boundary_id)
        {
            const auto patch_iter = d_boundary_id_to_faces.find(face_info.boundary_id);
            CHECK(patch_iter != d_boundary_id_to_faces.end());
            CHECK(std::find(patch_iter->second.begin(), patch_iter->second.end(),
                            static_cast<local_ordinal_type>(fid)) != patch_iter->second.end());
        }
    }
}

} // namespace SimpleFluid
