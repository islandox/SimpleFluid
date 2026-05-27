/**
 * @file STKMesh.tcc
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief
 * @version 0.1
 * @date 2026-05-26
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "STKMesh.hh"

#include <stk_io/IossBridge.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_util/parallel/Parallel.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace SimpleFluid
{

namespace stkmesh_detail
{

/**
 * @brief Create and configure STK metadata for a 3D mesh.
 *
 * @return Shared pointer to STK metadata.
 */
inline std::shared_ptr<stk::mesh::MetaData> make_stk_meta_data()
{
    stk::mesh::MeshBuilder builder(stk::parallel_machine_world());
    builder.set_spatial_dimension(3);
    return builder.create_meta_data();
}

/**
 * @brief Create STK bulk data object for a mesh.
 *
 * @param meta STK metadata describing the mesh topology and fields.
 * @return Shared pointer to STK bulk data.
 */
inline std::shared_ptr<stk::mesh::BulkData>
make_stk_bulk_data(const std::shared_ptr<stk::mesh::MetaData>& meta)
{
    stk::mesh::MeshBuilder builder(stk::parallel_machine_world());
    return std::shared_ptr<stk::mesh::BulkData>(builder.create(meta).release());
}

/**
 * @brief Convert an STK topology to the mesh cell type enum.
 *
 * @tparam Pack Tpetra type pack used by STKMesh.
 * @param topo STK cell topology.
 * @return Corresponding STKMesh cell type.
 * @throws std::runtime_error for unsupported topologies.
 */
template <TpetraTypePack Pack>
inline auto topology_to_cell_type(stk::topology topo) -> typename STKMesh<Pack>::CellType
{
    if (topo == stk::topology::HEX_8)
    {
        return STKMesh<Pack>::CellType::HEXAHEDRON;
    }
    if (topo == stk::topology::WEDGE_6)
    {
        return STKMesh<Pack>::CellType::TRIPRISM;
    }

    throw std::runtime_error("Unsupported cell topology: " + topo.name());
}


/**
 * @brief Determine face type from its node count.
 *
 * @tparam Pack Tpetra type pack used by STKMesh.
 * @param node_count Number of nodes on the face.
 * @return Corresponding STKMesh face type.
 * @throws std::runtime_error for unsupported face node counts.
 */
template <TpetraTypePack Pack>
inline auto face_type_from_node_count(std::size_t node_count) -> typename STKMesh<Pack>::FaceType
{
    if (node_count == 3)
    {
        return STKMesh<Pack>::FaceType::TRIANGLE;
    }
    if (node_count == 4)
    {
        return STKMesh<Pack>::FaceType::QUAD;
    }

    throw std::runtime_error("Unsupported face node count: "
                           + std::to_string(node_count));
}

/**
 * @brief Convert mesh cell type to VTU cell type identifier.
 *
 * @tparam Pack Tpetra type pack used by STKMesh.
 * @param type Mesh cell type.
 * @return VTU cell type code.
 * @throws std::runtime_error if the cell type cannot be exported.
 */
template <TpetraTypePack Pack>
inline int vtk_cell_type(typename STKMesh<Pack>::CellType type)
{
    switch (type)
    {
        case STKMesh<Pack>::CellType::HEXAHEDRON:
            return 12;
        case STKMesh<Pack>::CellType::TRIPRISM:
            return 13;
        default:
            break;
    }

    throw std::runtime_error("VTU export encountered an unsupported cell type.");
}

} // namespace stkmesh_detail

/**
 * @brief Construct an empty STK-based mesh.
 *
 * Initializes STK metadata, bulk data, and coordinate I/O support.
 */
template<TpetraTypePack Pack>
STKMesh<Pack>::STKMesh()
    : Mesh<Pack>()
{
    d_stk.meta = stkmesh_detail::make_stk_meta_data();
    d_stk.bulk = stkmesh_detail::make_stk_bulk_data(d_stk.meta);
    d_stk.io.set_bulk_data(*d_stk.bulk);
    d_spatial_dim = static_cast<int>(d_stk.meta->spatial_dimension());
}

/**
 * @brief Construct an STK mesh by reading an external mesh file.
 *
 * @param mesh_filename Path to the Exodus mesh file.
 * @param options Optional STK mesh reading options.
 */
template<TpetraTypePack Pack>
STKMesh<Pack>::STKMesh(const std::string& mesh_filename, const Options& options)
    : Mesh<Pack>()
{
    if (mesh_filename.empty())
    {
        throw std::runtime_error("Invalid mesh filename: " + mesh_filename);
    }

    d_stk.options = options;
    d_stk.meta = stkmesh_detail::make_stk_meta_data();
    d_stk.bulk = stkmesh_detail::make_stk_bulk_data(d_stk.meta);
    d_stk.io.set_bulk_data(*d_stk.bulk);
    d_stk.io.add_mesh_database(mesh_filename, "exodus", stk::io::READ_MESH);
    d_stk.io.create_input_mesh();
    d_stk.io.populate_bulk_data();
    d_spatial_dim = static_cast<int>(d_stk.meta->spatial_dimension());
}

/**
 * @brief Assemble the mesh geometry and connectivity into host and device views.
 */
template<TpetraTypePack Pack>
void STKMesh<Pack>::assemble()
{
    initialize_boundary_id_maps();
    build_cell_list();
    compute_cell_geometry();
    build_face_table();
    prefer_owned_face_owners();
    compute_face_geometry();
    assign_boundary_ids_from_stk_side_parts();
    create_cell_face_distances();

    check_connectivity();
    create_maps();
    create_device_views();
}

/**
 * @brief Populate the internal cell list from STK mesh elements.
 *
 * This creates cell records, captures node IDs, and builds ownership maps.
 */
template<TpetraTypePack Pack>
void STKMesh<Pack>::build_cell_list()
{
    d_spatial_dim = static_cast<int>(d_stk.meta->spatial_dimension());

    d_stk.cell_entities.clear();
    d_cells.clear();
    d_faces.clear();
    d_owned_cell_ids.clear();
    d_owned_cell_global_ids.clear();
    d_ghost_cell_global_ids.clear();
    d_cell_owned_face_ids.clear();
    d_cell_face_distances.clear();
    d_cell_owned_node_global_ids.clear();
    d_face_owned_node_global_ids.clear();
    d_node_coords.clear();
    d_cell_gid_to_lid.clear();
    d_node_gid_to_lid.clear();
    d_face_key_to_face.clear();
    d_boundary_id_to_faces.clear();

    d_stk.coord_field = nullptr;
    if (const auto* coord_base = d_stk.meta->coordinate_field(); coord_base != nullptr)
    {
        d_stk.coord_field = dynamic_cast<stk::mesh::Field<double>*>(
            const_cast<stk::mesh::FieldBase*>(coord_base));
    }

    if (d_stk.coord_field == nullptr && !d_stk.meta->coordinate_field_name().empty())
    {
        d_stk.coord_field = d_stk.meta->get_field<double>(
            stk::topology::NODE_RANK, d_stk.meta->coordinate_field_name());
    }

    if (d_stk.coord_field == nullptr)
    {
        d_stk.coord_field = d_stk.meta->get_field<double>(stk::topology::NODE_RANK,
                                                          "coordinates");
    }
    if (d_stk.coord_field == nullptr)
    {
        throw std::runtime_error("STK mesh does not contain a double coordinates field.");
    }

    std::vector<ArrGO> cell_node_ids;

    auto append_bucket_cells = [&](bool owned)
    {
        const auto& buckets = d_stk.bulk->buckets(stk::topology::ELEMENT_RANK);
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
                const auto gid = static_cast<global_ordinal_type>(d_stk.bulk->identifier(elem));

                if (!d_cell_gid_to_lid.emplace(gid, lid).second)
                {
                    throw std::runtime_error("Duplicate cell global id: "
                                           + std::to_string(gid));
                }

                CellInfo cell_info;
                cell_info.type = stkmesh_detail::topology_to_cell_type<Pack>(topo);
                cell_info.owned = owned;

                ArrGO node_ids;
                const auto num_nodes = d_stk.bulk->num_nodes(elem);
                const auto* nodes = d_stk.bulk->begin_nodes(elem);
                node_ids.reserve(num_nodes);
                for (unsigned i = 0; i < num_nodes; ++i)
                {
                    const auto node_gid = static_cast<global_ordinal_type>(
                        d_stk.bulk->identifier(nodes[i]));
                    node_ids.push_back(node_gid);

                    if (d_node_gid_to_lid.find(node_gid) == d_node_gid_to_lid.end())
                    {
                        const auto node_lid = detail::checked_size_to_ordinal<local_ordinal_type>(
                            d_node_coords.size(), "node count");
                        d_node_gid_to_lid.emplace(node_gid, node_lid);
                        d_node_coords.push_back(node_coord(nodes[i]));
                    }
                }

                d_stk.cell_entities.push_back(elem);
                d_cells.push_back(std::move(cell_info));
                cell_node_ids.push_back(std::move(node_ids));

                if (owned)
                {
                    d_owned_cell_ids.push_back(lid);
                    d_owned_cell_global_ids.push_back(gid);
                }
                else
                {
                    d_ghost_cell_global_ids.push_back(gid);
                }
            }
        }
    };

    append_bucket_cells(true);
    append_bucket_cells(false);

    std::size_t total_cell_nodes = 0;
    for (const auto& node_ids : cell_node_ids)
    {
        total_cell_nodes += node_ids.size();
    }

    d_cell_owned_node_global_ids.reserve(total_cell_nodes);
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto offset = d_cell_owned_node_global_ids.size();
        d_cell_owned_node_global_ids.insert(d_cell_owned_node_global_ids.end(),
                                            cell_node_ids[lid].begin(),
                                            cell_node_ids[lid].end());
        d_cells[lid].node_gids = ViewGO(d_cell_owned_node_global_ids.data() + offset,
                                       cell_node_ids[lid].size());
    }
}

/**
 * @brief Compute centroids and volumes for all cells.
 */
template<TpetraTypePack Pack>
void STKMesh<Pack>::compute_cell_geometry()
{
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto coords = element_node_coords(d_stk.cell_entities[lid]);
        auto& cell_info = d_cells[lid];

        cell_info.center = MeshUtils::average(coords);

        if (cell_info.type == CellType::HEXAHEDRON)
        {
            cell_info.volume = MeshUtils::hex_volume(coords);
        }
        else if (cell_info.type == CellType::TRIPRISM)
        {
            cell_info.volume = MeshUtils::wedge_volume(coords);
        }
        else
        {
            throw std::runtime_error("Unsupported cell type.");
        }
    }
}

/**
 * @brief Build face connectivity and owner/neighbor relationships.
 */
template<TpetraTypePack Pack>
void STKMesh<Pack>::build_face_table()
{
    d_faces.clear();
    d_face_key_to_face.clear();
    d_face_owned_node_global_ids.clear();
    d_cell_owned_face_ids.clear();

    std::vector<ArrLO> cell_face_ids(d_cells.size());

    std::size_t max_face_nodes = 0;
    for (const auto elem : d_stk.cell_entities)
    {
        const auto topo = d_stk.bulk->bucket(elem).topology();
        for (unsigned side = 0; side < topo.num_sides(); ++side)
        {
            max_face_nodes += topo.side_topology(side).num_nodes();
        }
    }
    d_face_owned_node_global_ids.reserve(max_face_nodes);

    for (std::size_t cell_index = 0; cell_index < d_cells.size(); ++cell_index)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell_index);
        const auto elem = d_stk.cell_entities[cell_index];
        const auto topo = d_stk.bulk->bucket(elem).topology();
        const auto* elem_nodes = d_stk.bulk->begin_nodes(elem);
        const auto num_elem_nodes = d_stk.bulk->num_nodes(elem);

        for (unsigned side = 0; side < topo.num_sides(); ++side)
        {
            const auto ordinals = side_node_ordinals(topo, side);
            ArrGO face_node_ids;
            face_node_ids.reserve(ordinals.size());

            for (const auto ordinal : ordinals)
            {
                CHECK(ordinal < num_elem_nodes);
                face_node_ids.push_back(static_cast<global_ordinal_type>(
                    d_stk.bulk->identifier(elem_nodes[ordinal])));
            }

            const auto key = make_face_key(face_node_ids);
            const auto iter = d_face_key_to_face.find(key);
            if (iter == d_face_key_to_face.end())
            {
                const auto fid = detail::checked_size_to_ordinal<local_ordinal_type>(
                    d_faces.size(), "face count");

                const auto offset = d_face_owned_node_global_ids.size();
                d_face_owned_node_global_ids.insert(d_face_owned_node_global_ids.end(),
                                                    face_node_ids.begin(),
                                                    face_node_ids.end());

                FaceInfo face_info;
                face_info.type = stkmesh_detail::face_type_from_node_count<Pack>(
                    face_node_ids.size());
                face_info.owner = cell_lid;
                face_info.neighbor = invalid_id<local_ordinal_type>();
                face_info.node_gids = ViewGO(d_face_owned_node_global_ids.data() + offset,
                                            face_node_ids.size());

                d_faces.push_back(std::move(face_info));
                d_face_key_to_face.emplace(key, fid);
                cell_face_ids[cell_index].push_back(fid);
            }
            else
            {
                auto& face_info = d_faces[static_cast<std::size_t>(iter->second)];
                if (face_info.neighbor != invalid_id<local_ordinal_type>())
                {
                    throw std::runtime_error("Non-manifold face encountered.");
                }

                face_info.neighbor = cell_lid;
                cell_face_ids[cell_index].push_back(iter->second);
            }
        }
    }

    std::size_t total_cell_faces = 0;
    for (const auto& faces : cell_face_ids)
    {
        total_cell_faces += faces.size();
    }

    d_cell_owned_face_ids.reserve(total_cell_faces);
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto offset = d_cell_owned_face_ids.size();
        d_cell_owned_face_ids.insert(d_cell_owned_face_ids.end(),
                                     cell_face_ids[lid].begin(),
                                     cell_face_ids[lid].end());
        d_cells[lid].faces = ViewLO(d_cell_owned_face_ids.data() + offset,
                                    cell_face_ids[lid].size());
    }
}

/**
 * @brief Compute face centroids, areas, and unit normals.
 */
template<TpetraTypePack Pack>
void STKMesh<Pack>::compute_face_geometry()
{
    for (auto& face_info : d_faces)
    {
        std::vector<Vec3> coords;
        coords.reserve(face_info.node_gids.size());

        for (const auto node_id : face_info.node_gids)
        {
            coords.push_back(node_coord_by_id(static_cast<EntityId>(node_id)));
        }

        face_info.center = MeshUtils::average(coords);

        auto area_vector = MeshUtils::face_area_vector(coords);
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
        face_info.owner_to_face_distance = owner_to_face.norm();

        if (face_info.neighbor != invalid_id<local_ordinal_type>())
        {
            const auto neighbor_to_face =
                face_info.center
              - d_cells[static_cast<std::size_t>(face_info.neighbor)].center;
            face_info.neighbor_to_face_distance = neighbor_to_face.norm();
            face_info.cell_center_distance =
                (d_cells[static_cast<std::size_t>(face_info.neighbor)].center
               - d_cells[static_cast<std::size_t>(face_info.owner)].center).norm();
        }
        else
        {
            face_info.neighbor_to_face_distance = 0.0;
            face_info.cell_center_distance = 0.0;
        }
    }
}

/**
 * @brief Initialize boundary ID mappings from options.
 */
template<TpetraTypePack Pack>
void STKMesh<Pack>::initialize_boundary_id_maps()
{
    d_boundary_id_to_name.clear();
    d_boundary_name_to_id.clear();
    d_boundary_id_to_faces.clear();
    d_next_boundary_id = 1;

    for (const auto& [name, id] : d_stk.options.boundary_name_to_id)
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

/**
 * @brief Get or create a boundary ID for a named side part.
 *
 * @param name Boundary part name.
 * @return Numeric boundary id.
 * @throws std::out_of_range if boundary names are not auto-assigned and the name is unknown.
 */
template<TpetraTypePack Pack>
int STKMesh<Pack>::get_or_create_boundary_id(const std::string& name)
{
    const auto iter = d_boundary_name_to_id.find(name);
    if (iter != d_boundary_name_to_id.end())
    {
        return iter->second;
    }

    if (!d_stk.options.auto_assign_boundary_ids)
    {
        throw std::out_of_range("Boundary name has no configured id: " + name);
    }

    const auto id = d_next_boundary_id++;
    d_boundary_name_to_id.emplace(name, id);
    d_boundary_id_to_name.emplace(id, name);
    return id;
}

/**
 * @brief Assign boundary IDs to exterior faces based on STK side parts.
 */
template<TpetraTypePack Pack>
void STKMesh<Pack>::assign_boundary_ids_from_stk_side_parts()
{
    d_boundary_id_to_faces.clear();

    const auto side_rank = d_stk.meta->side_rank();
    std::unordered_map<std::string, const stk::mesh::Part*> face_key_to_part;

    if (side_rank != stk::topology::INVALID_RANK)
    {
        const auto& side_buckets = d_stk.bulk->buckets(side_rank);
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
                const auto num_nodes = d_stk.bulk->num_nodes(side_entity);
                const auto* nodes = d_stk.bulk->begin_nodes(side_entity);

                ArrGO node_ids;
                node_ids.reserve(num_nodes);
                for (unsigned i = 0; i < num_nodes; ++i)
                {
                    node_ids.push_back(static_cast<global_ordinal_type>(
                        d_stk.bulk->identifier(nodes[i])));
                }

                face_key_to_part.emplace(make_face_key(std::move(node_ids)), boundary_part);
            }
        }
    }

    for (std::size_t fid = 0; fid < d_faces.size(); ++fid)
    {
        auto& face_info = d_faces[fid];
        face_info.boundary_id = invalid_boundary_id;

        if (face_info.neighbor != invalid_id<local_ordinal_type>())
        {
            continue;
        }

        const auto iter = face_key_to_part.find(make_face_key(face_info.node_gids));
        if (iter == face_key_to_part.end())
        {
            continue;
        }

        auto boundary_name = iter->second->name();
        face_info.boundary_id = get_or_create_boundary_id(boundary_name);
        d_boundary_id_to_faces[face_info.boundary_id].push_back(
            static_cast<local_ordinal_type>(fid));
    }
}

/**
 * @brief Check whether a topology is a supported volume cell type.
 *
 * @param topo STK topology to query.
 * @return True if the topology is supported, false otherwise.
 */
template<TpetraTypePack Pack>
bool STKMesh<Pack>::is_supported_volume_topology(stk::topology topo)
{
    return topo == stk::topology::HEX_8 || topo == stk::topology::WEDGE_6;
}

/**
 * @brief Retrieve the node ordinals for a face on a volume topology.
 *
 * @param topo Volume cell topology.
 * @param side_ordinal Side ordinal to query.
 * @return List of node ordinals defining the side.
 */
template<TpetraTypePack Pack>
auto STKMesh<Pack>::side_node_ordinals(stk::topology topo,
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

/**
 * @brief Check whether a part should be considered a candidate boundary part.
 *
 * @param part STK part to examine.
 * @param side_rank Side rank in the mesh.
 * @return True if the part is a candidate boundary part.
 */
template<TpetraTypePack Pack>
bool STKMesh<Pack>::is_candidate_boundary_part(const stk::mesh::Part& part,
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

    return d_stk.options.auto_assign_boundary_ids;
}

/**
 * @brief Choose the best boundary part for a side bucket.
 *
 * @param bucket Side bucket containing candidate parts.
 * @param side_rank Side rank to filter by.
 * @return Pointer to the chosen STK part, or nullptr if none qualify.
 */
template<TpetraTypePack Pack>
auto STKMesh<Pack>::choose_boundary_part(const stk::mesh::Bucket& bucket,
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

/**
 * @brief Export the mesh to a VTU file.
 *
 * @param filename Output VTU filename.
 * @throws std::runtime_error if file writing fails.
 */
template<TpetraTypePack Pack>
void STKMesh<Pack>::export_vtu(const std::string& filename) const
{
    check_connectivity();

    std::unordered_map<EntityId, local_ordinal_type> node_lid;
    ArrVec3 node_coords;
    ArrLO cell_node_offset{0};
    ArrLO cell_node_ids;

    auto append_node = [&](stk::mesh::Entity node) -> local_ordinal_type
    {
        const auto node_id = d_stk.bulk->identifier(node);
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

    for (const auto elem : d_stk.cell_entities)
    {
        const auto num_nodes = d_stk.bulk->num_nodes(elem);
        const auto* nodes = d_stk.bulk->begin_nodes(elem);

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
    for (auto gid : d_owned_cell_global_ids)
    {
        out << gid << " ";
    }
    for (size_t i = 0; i < d_ghost_cell_global_ids.size(); ++i)
    {
        out << d_ghost_cell_global_ids[i] << (i + 1 == d_ghost_cell_global_ids.size() ? "" : " ");
    }

    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"cell_type\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        out << static_cast<int>(d_cells[lid].type)
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
        out << stkmesh_detail::vtk_cell_type<Pack>(d_cells[lid].type)
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

} // namespace SimpleFluid
