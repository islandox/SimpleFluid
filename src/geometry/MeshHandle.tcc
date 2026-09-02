/**
 * @file MeshHandle.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementations for MeshHandle construction and VTU output.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "MeshHandle.hh"

#include <array>

namespace SimpleFluid
{

/**
 * @brief Construct and distribute a Cartesian mesh handle.
 *
 * @param mesh Cartesian mesh to wrap.
 * @param options Partition override and ghost-layer configuration.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(
    CartesianPtr mesh,
    DistributionOptions options)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_orthogonal(std::get<CartesianPtr>(d_mesh), options);
}

/**
 * @brief Construct and distribute a cylindrical mesh handle.
 *
 * @param mesh Cylindrical mesh to wrap.
 * @param options Partition override and ghost-layer configuration.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(
    CylindricalPtr mesh,
    DistributionOptions options)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_orthogonal(std::get<CylindricalPtr>(d_mesh), options);
}

/**
 * @brief Construct a serial semi-structured mesh handle.
 *
 * @param mesh Semi-structured mesh to wrap.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(SemiStructuredPtr mesh)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_semi_structured(std::get<SemiStructuredPtr>(d_mesh));
}

/**
 * @brief Construct a serial unstructured mesh handle.
 *
 * @param mesh Unstructured mesh to wrap.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(UnstructuredPtr mesh)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_unstructured(std::get<UnstructuredPtr>(d_mesh));
}

/**
 * @brief Construct a handle from an already partitioned unstructured mesh.
 *
 * @param mesh Rank-local unstructured mesh geometry.
 * @param indexer Mapping from compact local IDs to original global IDs.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(
    UnstructuredPtr mesh,
    const unstructured_indexer_type& indexer)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_unstructured(
        std::get<UnstructuredPtr>(d_mesh), indexer,
        Tpetra::getDefaultComm());
}

/**
 * @brief Construct a handle from an existing STK mesh adapter.
 *
 * @param mesh STK adapter to wrap.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(STKAdapterPtr mesh)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_stk(std::get<STKAdapterPtr>(d_mesh));
}

/**
 * @brief Adapt a legacy distributed mesh and construct its handle.
 *
 * @param mesh Legacy mesh to adapt.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(SP<const SimpleFluid::Mesh<Pack>> mesh)
    : MeshHandle(std::make_shared<STKAdapter>(std::move(mesh)))
{
}

/**
 * @brief Initialize ownership, overlap, connectivity, and maps for an
 * orthogonal mesh.
 *
 * The mesh is partitioned along its largest coordinate when more than one
 * partition is requested.
 *
 * @tparam MeshType Cartesian or cylindrical orthogonal mesh type.
 * @param mesh Orthogonal mesh to initialize from.
 * @param options Partition override and ghost-layer configuration.
 * @throws std::invalid_argument if partition options are inconsistent or the
 * requested partitioning cannot be represented by the mesh.
 */
template<TpetraTypePack Pack>
template<class MeshType>
void MeshHandle<Pack>::initialize_orthogonal(
    SP<const MeshType> mesh,
    DistributionOptions options)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto rank = options.partition.value_or(
        static_cast<size_t>(comm->getRank()));
    const auto ranks = options.partitions.value_or(
        static_cast<size_t>(comm->getSize()));
    if (options.partition.has_value() != options.partitions.has_value())
    {
        throw std::invalid_argument(
            "Mesh distribution override requires both partition "
            "and partition count.");
    }
    if (rank >= ranks)
    {
        throw std::invalid_argument(
            "Mesh distribution partition index is out of range.");
    }

    if (ranks == 1)
    {
        initialize_serial(*mesh);
        return;
    }

    const auto& dimensions = mesh->indexer().num_cells_per_dim;
    size_t coordinate = 0;
    for (size_t candidate = 1; candidate < 3; ++candidate)
    {
        if (dimensions[candidate] > dimensions[coordinate])
        {
            coordinate = candidate;
        }
    }
    if (ranks > dimensions[coordinate])
    {
        throw std::invalid_argument(
            "Orthogonal mesh has fewer cells than MPI ranks along "
            "its largest coordinate.");
    }

    Meshes::OrthoMeshPartitioner partitioner(
        mesh->topology(),
        static_cast<typename Meshes::OrthoMeshPartitioner::Dimension>(
            coordinate),
        ranks,
        static_cast<Meshes::OrthoMeshPartitioner::Ordinal>(
            options.ghost_layers));

    std::vector<size_t> owned_cells;
    std::vector<size_t> ghost_cells;
    for (const auto id : partitioner.owned_cells(rank))
    {
        owned_cells.push_back(mesh->cell_local_id(id));
    }
    for (const auto id : partitioner.ghost_cells(rank))
    {
        ghost_cells.push_back(mesh->cell_local_id(id));
    }
    initialize_cells(std::move(owned_cells), std::move(ghost_cells));

    std::unordered_set<size_t> seen_faces;
    std::vector<size_t> owned_faces;
    std::vector<size_t> overlap_faces;
    for (const auto cell_geometry_lid : d_indexer.cell_global_ids())
    {
        const typename MeshType::cell_id_t typed_cell =
            mesh->indexer().cell_id(cell_geometry_lid);
        const auto typed_faces = mesh->cell_faces(typed_cell);
        for (const auto face : typed_faces)
        {
            const auto face_geometry_lid =
                mesh->indexer().face_ordinal(face);
            if (!seen_faces.insert(face_geometry_lid).second)
            {
                continue;
            }
            if (partitioner.is_owned_face(rank, face))
            {
                owned_faces.push_back(face_geometry_lid);
            }
            else
            {
                overlap_faces.push_back(face_geometry_lid);
            }
        }
    }
    initialize_faces(std::move(owned_faces), std::move(overlap_faces));

    constexpr std::array<std::array<unsigned, 3>, 8> corners{{
        {{0, 0, 0}},
        {{1, 0, 0}},
        {{1, 1, 0}},
        {{0, 1, 0}},
        {{0, 0, 1}},
        {{1, 0, 1}},
        {{1, 1, 1}},
        {{0, 1, 1}}
    }};
    std::unordered_set<size_t> seen_nodes;
    std::vector<size_t> local_nodes;
    for (const auto cell_geometry_lid : d_indexer.cell_global_ids())
    {
        const auto cell = mesh->cell_id(
            static_cast<size_t>(cell_geometry_lid));
        for (const auto& corner : corners)
        {
            const typename MeshType::NodeID node{
                static_cast<unsigned>(
                    (cell.i + corner[0])
                    % mesh->indexer().num_nodes_per_dim[0]),
                static_cast<unsigned>(
                    (cell.j + corner[1])
                    % mesh->indexer().num_nodes_per_dim[1]),
                static_cast<unsigned>(
                    (cell.k + corner[2])
                    % mesh->indexer().num_nodes_per_dim[2])};
            const auto global_node = mesh->node_local_id(node);
            if (seen_nodes.insert(global_node).second)
            {
                local_nodes.push_back(global_node);
            }
        }
    }
    d_indexer.set_nodes(checked_global_ids(std::move(local_nodes)));
    initialize_cell_faces();
    initialize_boundary_batches(*mesh);
    create_maps(comm);
}

/**
 * @brief Initialize a semi-structured mesh handle in serial.
 *
 * @param mesh Semi-structured mesh to initialize from.
 * @throws std::runtime_error if used with more than one MPI rank.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_semi_structured(
    SemiStructuredPtr mesh)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 1)
    {
        throw std::runtime_error(
            "SemiStructuredXY_Z does not yet support multi-rank "
            "distribution.");
    }
    initialize_serial(*mesh);
}

/**
 * @brief Initialize a serial unstructured mesh handle.
 *
 * @param mesh Unstructured mesh to initialize from.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_unstructured(
    UnstructuredPtr mesh)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 1)
    {
        throw std::invalid_argument(
            "Distributed UnstructuredMesh must be partitioned before "
            "constructing MeshHandle.");
    }
    initialize_serial(*mesh);
}

/**
 * @brief Initialize from rank-local geometry and its global indexer.
 *
 * @param mesh Previously partitioned unstructured mesh.
 * @param indexer Mapping from mesh-local IDs to original global IDs.
 * @param comm Communicator that owns the partition.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_unstructured(
    UnstructuredPtr mesh,
    const unstructured_indexer_type& indexer,
    Teuchos::RCP<const typename Pack::comm_type> comm)
{
    if (mesh->num_cells() != indexer.num_local_cells()
        || mesh->num_owned_cells() != indexer.num_owned_cells()
        || mesh->num_faces() != indexer.num_local_faces()
        || mesh->num_owned_faces() != indexer.num_owned_faces()
        || mesh->num_nodes() != indexer.num_local_nodes())
    {
        throw std::invalid_argument(
            "Partitioned UnstructuredMesh does not match its indexer.");
    }

    auto global_ids = [&](const auto& ids)
    {
        std::vector<size_t> flattened;
        flattened.reserve(ids.size());
        for (const auto id : ids)
        {
            using id_type = std::remove_cvref_t<decltype(id)>;
            if constexpr (
                std::numeric_limits<id_type>::digits
                > std::numeric_limits<size_t>::digits)
            {
                if (id > static_cast<id_type>(
                        std::numeric_limits<size_t>::max()))
                {
                    throw std::overflow_error(
                        "MeshHandle unstructured global ID overflow.");
                }
            }
            flattened.push_back(static_cast<size_t>(id));
        }
        return checked_global_ids(std::move(flattened));
    };

    initialize_indexer(indexer_type(
        global_ids(indexer.owned_cell_global_ids()),
        global_ids(indexer.ghost_cell_global_ids()),
        global_ids(indexer.owned_face_global_ids()),
        global_ids(indexer.overlap_face_global_ids()),
        global_ids(indexer.node_global_ids())));
    initialize_cell_faces();
    initialize_boundary_batches(*mesh);
    create_maps(comm ? std::move(comm) : Tpetra::getDefaultComm());
}

/**
 * @brief Import ownership, overlap, connectivity, and maps from an STK
 * adapter.
 *
 * @param adapter Adapter providing the legacy mesh interface.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_stk(STKAdapterPtr adapter)
{
    const auto& mesh = adapter->mesh();
    bool identity_order = true;
    bool saw_overlap_face = false;
    for (size_t lid = 0; lid < mesh.num_faces(); ++lid)
    {
        if (mesh.is_owned_face(checked_local(lid)))
        {
            if (saw_overlap_face)
            {
                identity_order = false;
                break;
            }
        }
        else
        {
            saw_overlap_face = true;
        }
    }
    if (!identity_order)
    {
        d_legacy_face_geometry_lids.reserve(mesh.num_faces());
        for (size_t lid = 0; lid < mesh.num_faces(); ++lid)
        {
            if (mesh.is_owned_face(checked_local(lid)))
            {
                d_legacy_face_geometry_lids.push_back(
                    checked_local(lid));
            }
        }
        for (size_t lid = 0; lid < mesh.num_faces(); ++lid)
        {
            if (!mesh.is_owned_face(checked_local(lid)))
            {
                d_legacy_face_geometry_lids.push_back(
                    checked_local(lid));
            }
        }
        d_legacy_face_local_lids.resize(
            d_legacy_face_geometry_lids.size());
        for (size_t local = 0;
             local < d_legacy_face_geometry_lids.size();
             ++local)
        {
            d_legacy_face_local_lids[
                static_cast<size_t>(
                    d_legacy_face_geometry_lids[local])] =
                checked_local(local);
        }
        // Interleaved ownership is uncommon. Preserve the established
        // owned-first handle ordering with a compact permutation and
        // materialize cell connectivity only for this fallback.
        initialize_cell_faces();
    }
    initialize_boundary_batches(*adapter);

    d_owned_cell_map = mesh.owned_cell_map();
    d_overlap_cell_map = mesh.overlap_cell_map();
    d_owned_face_map = mesh.owned_face_map();
    d_boundary_face_map = mesh.boundary_face_map();
    std::vector<global_ordinal_type> overlap_face_gids;
    overlap_face_gids.reserve(num_faces());
    for (size_t face_lid = 0; face_lid < num_faces(); ++face_lid)
    {
        const auto geometry_lid =
            geometry_face_lid(checked_local(face_lid));
        overlap_face_gids.push_back(
            mesh.face_global_id(
                checked_local(static_cast<size_t>(geometry_lid))));
    }
    d_overlap_face_map = make_map(
        Tpetra::getDefaultComm(), overlap_face_gids);
}

/**
 * @brief Initialize all cells and faces as locally owned on one rank.
 *
 * @tparam MeshType Concrete mesh type.
 * @param mesh Mesh whose complete geometry is locally available.
 */
template<TpetraTypePack Pack>
template<class MeshType>
void MeshHandle<Pack>::initialize_serial(const MeshType& mesh)
{
    std::vector<size_t> cells(mesh.num_cells());
    for (size_t lid = 0; lid < cells.size(); ++lid)
    {
        cells[lid] = lid;
    }
    initialize_cells(std::move(cells), {});

    std::vector<size_t> faces(mesh.num_faces());
    for (size_t lid = 0; lid < faces.size(); ++lid)
    {
        faces[lid] = lid;
    }
    initialize_faces(std::move(faces), {});

    std::vector<size_t> nodes(mesh.num_nodes());
    for (size_t lid = 0; lid < nodes.size(); ++lid)
    {
        nodes[lid] = lid;
    }
    d_indexer.set_nodes(checked_global_ids(std::move(nodes)));
    initialize_cell_faces();
    initialize_boundary_batches(mesh);
    create_maps(Tpetra::getDefaultComm());
}

/**
 * @brief Store owned and ghost cell global IDs in the local/global indexer.
 *
 * Owned cells are placed before ghost cells and reverse lookup is populated.
 *
 * @param owned Geometry-local IDs of owned cells.
 * @param ghost Geometry-local IDs of ghost cells.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_cells(
    std::vector<size_t> owned,
    std::vector<size_t> ghost)
{
    d_indexer.set_cells(
        checked_global_ids(std::move(owned)),
        checked_global_ids(std::move(ghost)));
}

/**
 * @brief Store owned and overlap face global IDs in the local/global indexer.
 *
 * Owned faces are placed before overlap faces and reverse lookup is populated.
 *
 * @param owned Geometry-local IDs of owned faces.
 * @param overlap Geometry-local IDs of non-owned locally visible faces.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_faces(
    std::vector<size_t> owned,
    std::vector<size_t> overlap)
{
    d_indexer.set_faces(
        checked_global_ids(std::move(owned)),
        checked_global_ids(std::move(overlap)));
}

/**
 * @brief Replace the handle's local/global indexer with a prepared layout.
 * @tparam Pack Tpetra type pack.
 * @param indexer Complete owned-first indexer to store.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_indexer(indexer_type indexer)
{
    d_indexer = std::move(indexer);
}

/**
 * @brief Lazily materialize legacy identity/permutation lookup tables.
 *
 * Production geometry and field paths use the wrapped mesh arrays directly;
 * this preserves the historical indexer() API only for callers that request
 * it explicitly.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::materialize_legacy_indexer() const
{
    if (d_legacy_indexer_materialized)
    {
        return;
    }
    const auto legacy = legacy_mesh();
    if (!legacy)
    {
        return;
    }

    std::vector<size_t> owned_cells;
    std::vector<size_t> ghost_cells;
    owned_cells.reserve(legacy->num_owned_cells());
    ghost_cells.reserve(
        legacy->num_local_cells() - legacy->num_owned_cells());
    for (size_t local = 0; local < legacy->num_local_cells(); ++local)
    {
        (local < legacy->num_owned_cells()
             ? owned_cells
             : ghost_cells)
            .push_back(static_cast<size_t>(geometry_cell_lid(
                checked_local(local))));
    }

    std::vector<size_t> owned_faces;
    std::vector<size_t> overlap_faces;
    const auto owned_face_count = num_owned_faces();
    owned_faces.reserve(owned_face_count);
    overlap_faces.reserve(num_faces() - owned_face_count);
    for (size_t local = 0; local < num_faces(); ++local)
    {
        const auto geometry = static_cast<size_t>(
            geometry_face_lid(checked_local(local)));
        (local < owned_face_count
             ? owned_faces
             : overlap_faces)
            .push_back(geometry);
    }

    d_indexer = indexer_type(
        checked_global_ids(std::move(owned_cells)),
        checked_global_ids(std::move(ghost_cells)),
        checked_global_ids(std::move(owned_faces)),
        checked_global_ids(std::move(overlap_faces)),
        std::vector<global_ordinal_type>{});
    d_legacy_indexer_materialized = true;
}

/**
 * @brief Materialize cell-to-face connectivity in MeshHandle local IDs.
 *
 * Faces not present in the local overlap are omitted.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_cell_faces()
{
    d_cell_face_offsets.clear();
    d_cell_face_lids.clear();
    d_cell_face_offsets.reserve(num_local_cells() + 1);
    d_cell_face_offsets.push_back(0);

    visit(
        [&](const auto& mesh)
        {
            for (size_t local_lid = 0;
                 local_lid < num_local_cells();
                 ++local_lid)
            {
                const auto geometry_lid = geometry_cell_lid(
                    checked_local(local_lid));
                const auto geometry_faces =
                    mesh.faces(mesh.cell_id(
                        static_cast<size_t>(geometry_lid)));
                d_cell_face_lids.reserve(
                    d_cell_face_lids.size() + geometry_faces.size());
                for (const auto geometry_face : geometry_faces)
                {
                    const auto face_geometry_lid =
                        static_cast<size_t>(
                            mesh.face_local_id(geometry_face));
                    const auto local_face =
                        geometry_to_local_face(face_geometry_lid);
                    if (local_face != invalid_local_id())
                    {
                        d_cell_face_lids.push_back(local_face);
                    }
                }
                d_cell_face_offsets.push_back(d_cell_face_lids.size());
            }
        });
}

/**
 * @brief Materialize locally visible boundary batches.
 *
 * Supports both view-based and legacy materialized boundary-batch APIs.
 *
 * @tparam MeshType Concrete mesh or adapter type.
 * @param mesh Mesh providing boundary batch metadata and face IDs.
 */
template<TpetraTypePack Pack>
template<class MeshType>
void MeshHandle<Pack>::initialize_boundary_batches(
    const MeshType& mesh)
{
    auto materialize_batch =
        [&](int batch_id, const auto& source_faces)
    {
        BoundaryFaceBatch batch;
        batch.id = batch_id;
        for (const auto face : source_faces)
        {
            const auto geometry_lid =
                static_cast<size_t>(mesh.face_local_id(face));
            const auto local_face = geometry_to_local_face(geometry_lid);
            if (local_face != invalid_local_id())
            {
                batch.face_lids.push_back(local_face);
            }
        }
        if (!batch.face_lids.empty())
        {
            d_boundary_names.emplace(
                batch_id, mesh.boundary_batch_name(batch_id));
            d_boundary_batches.emplace(batch_id, std::move(batch));
        }
    };

    if constexpr (std::ranges::range<
                      decltype(mesh.boundary_face_batch(0))>)
    {
        // New view-based API: boundary_face_batch() returns a view
        for (int batch_id : mesh.boundary_batch_ids())
        {
            materialize_batch(
                batch_id, mesh.boundary_face_batch(batch_id));
        }
    }
    else
    {
        // Legacy materialized-map API
        for (const auto& [batch_id, source_batch] :
             mesh.boundary_batches())
        {
            materialize_batch(batch_id, source_batch.face_lids);
        }
    }
}

/**
 * @brief Export owned mesh cells to VTU.
 *
 * STK meshes delegate to their native exporter; other mesh families are
 * converted to VTU connectivity by this handle.
 *
 * @param filename Requested output filename.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::export_vtu(const std::string& filename) const
{
    visit(
        [&](const auto& mesh)
        {
            using mesh_type = std::decay_t<decltype(mesh)>;
            if constexpr (std::is_same_v<mesh_type, STKAdapter>)
            {
                mesh.export_vtu(filename);
            }
            else if constexpr (std::is_same_v<mesh_type, SemiStructured>)
            {
                write_vtu(filename, semi_structured_vtu_topology(mesh));
            }
            else if constexpr (std::is_same_v<mesh_type, Unstructured>)
            {
                write_vtu(filename, unstructured_vtu_topology(mesh));
            }
            else
            {
                write_vtu(filename, orthogonal_vtu_topology(mesh));
            }
        });
}

/**
 * @brief Build reusable VTU topology for the owned cells.
 *
 * @return Immutable topology in the same owned-cell order used by fields.
 */
template<TpetraTypePack Pack>
VTUWriter::TopologyHandle MeshHandle<Pack>::vtu_topology() const
{
    return visit(
        [&](const auto& mesh) -> VTUWriter::TopologyHandle
        {
            using mesh_type = std::decay_t<decltype(mesh)>;
            if constexpr (std::is_same_v<mesh_type, STKAdapter>)
            {
                return legacy_vtu_topology(mesh);
            }
            else if constexpr (std::is_same_v<mesh_type, SemiStructured>)
            {
                return semi_structured_vtu_topology(mesh);
            }
            else if constexpr (std::is_same_v<mesh_type, Unstructured>)
            {
                return unstructured_vtu_topology(mesh);
            }
            else
            {
                return orthogonal_vtu_topology(mesh);
            }
        });
}

/**
 * @brief Add the MPI rank suffix required for per-rank VTU output.
 *
 * @param filename Requested output filename.
 * @return The unchanged serial filename or a rank-specific parallel filename.
 */
template<TpetraTypePack Pack>
std::string MeshHandle<Pack>::local_output_filename(
    const std::string& filename) const
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() <= 1)
    {
        return filename;
    }

    const auto suffix = "_rank" + std::to_string(comm->getRank());
    const auto dot = filename.rfind('.');
    return dot == std::string::npos
         ? filename + suffix
         : filename.substr(0, dot) + suffix + filename.substr(dot);
}

/**
 * @brief Add global IDs, volumes, and centroids for owned VTU cells.
 *
 * @param writer Writer receiving the geometry cell-data arrays.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::add_geometry_cell_data(VTUWriter& writer) const
{
    VTUWriter::Int64Data cell_ids;
    VTUWriter::ScalarData cell_volumes;
    VTUWriter::VectorData cell_centroids;
    cell_ids.reserve(num_owned_cells());
    cell_volumes.reserve(num_owned_cells());
    cell_centroids.reserve(num_owned_cells());
    for (size_t lid = 0; lid < num_owned_cells(); ++lid)
    {
        const auto local_id = checked_local(lid);
        cell_ids.push_back(static_cast<global_index_t>(
            cell_global_id(local_id)));
        cell_volumes.push_back(cell_volume(local_id));
        cell_centroids.push_back(cell_centroid(local_id));
    }

    writer.add_int64_cell_data("cell_gid", std::move(cell_ids));
    writer.add_scalar_cell_data("cell_volume", std::move(cell_volumes));
    writer.add_vector_cell_data("cell_centroid", std::move(cell_centroids));
}

/**
 * @brief Collect all mesh node coordinates as VTU points.
 *
 * @tparam MeshType Concrete mesh type.
 * @param mesh Mesh providing node coordinates.
 * @return Point coordinates in mesh-local node order.
 */
template<TpetraTypePack Pack>
template<class MeshType>
VTUWriter::VectorData MeshHandle<Pack>::collect_vtu_points(
    const MeshType& mesh) const
{
    VTUWriter::VectorData points;
    points.reserve(mesh.num_nodes());
    for (size_t lid = 0; lid < mesh.num_nodes(); ++lid)
    {
        points.push_back(mesh.node_coordinates(mesh.node_id(lid)));
    }
    return points;
}

/**
 * @brief Write assembled geometry and owned-cell data to a VTU file.
 *
 * @param filename Requested output filename.
 * @param points VTU point coordinates.
 * @param connectivity Flattened cell-to-point connectivity.
 * @param offsets End offsets for each cell in the connectivity array.
 * @param cell_types VTK cell type identifiers.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::write_vtu(
    const std::string& filename,
    VTUWriter::TopologyHandle topology) const
{
    VTUWriter writer(std::move(topology));
    add_geometry_cell_data(writer);
    writer.write(local_output_filename(filename));
}

/**
 * @brief Build topology for the exact legacy mesh wrapped by this handle.
 */
template<TpetraTypePack Pack>
VTUWriter::TopologyHandle MeshHandle<Pack>::legacy_vtu_topology(
    const STKAdapter& mesh) const
{
    std::unordered_map<global_ordinal_type, global_index_t> node_lids;
    VTUWriter::VectorData points;
    VTUWriter::Int64Data connectivity;
    VTUWriter::Int64Data offsets;
    VTUWriter::UInt8Data cell_types;
    offsets.reserve(num_owned_cells());
    cell_types.reserve(num_owned_cells());

    const auto& legacy = mesh.mesh();
    auto append_node =
        [&](global_ordinal_type node_gid) -> global_index_t
    {
        const auto [position, inserted] = node_lids.emplace(
            node_gid, static_cast<global_index_t>(points.size()));
        if (inserted)
        {
            points.push_back(legacy.node_coord(node_gid));
        }
        return position->second;
    };

    for (size_t lid = 0; lid < num_owned_cells(); ++lid)
    {
        const auto geometry_lid = geometry_cell_lid(checked_local(lid));
        const auto& cell = legacy.cell(checked_local(
            static_cast<size_t>(geometry_lid)));
        for (const auto node_gid : cell.node_gids)
        {
            connectivity.push_back(append_node(node_gid));
        }
        offsets.push_back(static_cast<global_index_t>(connectivity.size()));
        cell_types.push_back(static_cast<std::uint8_t>(
            MeshUtils::vtu_cell_type_code(cell.type)));
    }

    return VTUWriter::make_topology(
        std::move(points),
        std::move(connectivity),
        std::move(offsets),
        std::move(cell_types));
}

/**
 * @brief Export owned orthogonal cells as VTK hexahedra.
 *
 * @tparam MeshType Cartesian or cylindrical orthogonal mesh type.
 * @param mesh Mesh providing node and cell indexing.
 * @param filename Requested output filename.
 */
template<TpetraTypePack Pack>
template<class MeshType>
VTUWriter::TopologyHandle MeshHandle<Pack>::orthogonal_vtu_topology(
    const MeshType& mesh) const
{
    VTUWriter::Int64Data connectivity;
    VTUWriter::Int64Data offsets;
    VTUWriter::UInt8Data cell_types;
    connectivity.reserve(num_owned_cells() * 8);
    offsets.reserve(num_owned_cells());
    cell_types.reserve(num_owned_cells());

    constexpr std::array<std::array<unsigned, 3>, 8> corners{{
        {{0, 0, 0}},
        {{1, 0, 0}},
        {{1, 1, 0}},
        {{0, 1, 0}},
        {{0, 0, 1}},
        {{1, 0, 1}},
        {{1, 1, 1}},
        {{0, 1, 1}}
    }};
    const auto& indexer = mesh.indexer();
    for (size_t lid = 0; lid < num_owned_cells(); ++lid)
    {
        const auto cell = mesh.cell_id(static_cast<size_t>(
            geometry_cell_lid(checked_local(lid))));
        for (const auto& corner : corners)
        {
            const typename MeshType::NodeID node{
                static_cast<unsigned>(
                    (cell.i + corner[0]) % indexer.num_nodes_per_dim[0]),
                static_cast<unsigned>(
                    (cell.j + corner[1]) % indexer.num_nodes_per_dim[1]),
                static_cast<unsigned>(
                    (cell.k + corner[2]) % indexer.num_nodes_per_dim[2])};
            connectivity.push_back(static_cast<global_index_t>(
                mesh.node_local_id(node)));
        }
        offsets.push_back(static_cast<global_index_t>(connectivity.size()));
        cell_types.push_back(12); // VTK_HEXAHEDRON
    }

    return VTUWriter::make_topology(
        collect_vtu_points(mesh),
        std::move(connectivity),
        std::move(offsets),
        std::move(cell_types));
}

/**
 * @brief Export owned semi-structured cells using their extruded topology.
 *
 * Triangular base cells become wedges, quadrilateral base cells become
 * hexahedra, and other polygons use VTK convex point sets.
 *
 * @param mesh Semi-structured mesh providing layered node connectivity.
 * @param filename Requested output filename.
 */
template<TpetraTypePack Pack>
VTUWriter::TopologyHandle MeshHandle<Pack>::semi_structured_vtu_topology(
    const SemiStructured& mesh) const
{
    VTUWriter::Int64Data connectivity;
    VTUWriter::Int64Data offsets;
    VTUWriter::UInt8Data cell_types;
    offsets.reserve(num_owned_cells());
    cell_types.reserve(num_owned_cells());
    const auto& indexer = mesh.indexer();
    for (size_t lid = 0; lid < num_owned_cells(); ++lid)
    {
        const auto cell = mesh.cell_id(static_cast<size_t>(
            geometry_cell_lid(checked_local(lid))));
        const auto& xy_nodes = mesh.xy_cell_nodes()[cell.ij];
        const auto top_layer = (cell.k + 1) % indexer.num_node_layers;
        for (const auto xy_node : xy_nodes)
        {
            connectivity.push_back(static_cast<global_index_t>(
                mesh.node_local_id(
                    SemiStructured::NodeID{xy_node, cell.k})));
        }
        for (const auto xy_node : xy_nodes)
        {
            connectivity.push_back(static_cast<global_index_t>(
                mesh.node_local_id(
                    SemiStructured::NodeID{xy_node, top_layer})));
        }

        offsets.push_back(static_cast<global_index_t>(connectivity.size()));
        if (xy_nodes.size() == 3)
        {
            cell_types.push_back(13); // VTK_WEDGE
        }
        else if (xy_nodes.size() == 4)
        {
            cell_types.push_back(12); // VTK_HEXAHEDRON
        }
        else
        {
            cell_types.push_back(41); // VTK_CONVEX_POINT_SET
        }
    }

    return VTUWriter::make_topology(
        collect_vtu_points(mesh),
        std::move(connectivity),
        std::move(offsets),
        std::move(cell_types));
}

/**
 * @brief Export owned unstructured cells using their explicit node lists.
 *
 * @param mesh Unstructured mesh providing node connectivity.
 * @param filename Requested output filename.
 */
template<TpetraTypePack Pack>
VTUWriter::TopologyHandle MeshHandle<Pack>::unstructured_vtu_topology(
    const Unstructured& mesh) const
{
    VTUWriter::Int64Data connectivity;
    VTUWriter::Int64Data offsets;
    VTUWriter::UInt8Data cell_types;
    offsets.reserve(num_owned_cells());
    cell_types.reserve(num_owned_cells());

    for (size_t lid = 0; lid < num_owned_cells(); ++lid)
    {
        const auto cell = mesh.cell_id(static_cast<size_t>(
            geometry_cell_lid(checked_local(lid))));
        for (const auto node : mesh.cell_nodes(cell))
        {
            connectivity.push_back(static_cast<global_index_t>(
                mesh.node_local_id(node)));
        }
        offsets.push_back(static_cast<global_index_t>(
            connectivity.size()));
        cell_types.push_back(static_cast<std::uint8_t>(
            MeshUtils::vtu_cell_type_code(mesh.cell_type(cell))));
    }

    return VTUWriter::make_topology(
        collect_vtu_points(mesh),
        std::move(connectivity),
        std::move(offsets),
        std::move(cell_types));
}

} // namespace SimpleFluid
