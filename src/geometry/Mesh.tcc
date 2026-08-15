/**
 * @file Mesh.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template method implementations for the Mesh class.
 * @version 0.1
 * @date 2026-05-25
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "Mesh.hh"

#include <Tpetra_Core.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_DefaultMpiComm.hpp>

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
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Default constructor for Mesh.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
Mesh<Pack>::Mesh()
{
}

/**
 * @brief Assign contiguous Tpetra global IDs to owned and ghost cells.
 *
 * Computes a globally contiguous GID block for this process's owned cells
 * (offset = sum of owned cells on all lower-ranked processes) and resolves
 * the corresponding Tpetra GIDs for ghost cells through Tpetra's distributed
 * directory.  The resulting maps are stored in:
 * - d_ghost_cell_tpetra_gids
 * - d_mesh_gid_to_tpetra_gid
 * - d_tpetra_gid_to_mesh_gid
 *
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::assign_contiguous_tpetra_gids()
{
    if (d_contiguous_tpetra_gids_assigned)
    {
        return; // already assigned
    }

    const auto comm = Tpetra::getDefaultComm();
    const int myrank = comm->getRank();
    const int nranks = comm->getSize();
    MPI_Comm raw_comm = MPI_COMM_NULL;

    const int my_owned_count = static_cast<int>(d_owned_cell_global_ids.size());

    // --- Gather owned cell counts from all ranks ---
    std::vector<int> all_owned_counts(static_cast<size_t>(nranks), 0);

    if (nranks > 1)
    {
        auto* mpi_comm = dynamic_cast<const Teuchos::MpiComm<int>*>(comm.get());
        if (mpi_comm == nullptr)
        {
            throw std::runtime_error("assign_contiguous_tpetra_gids requires an MPI communicator.");
        }
        raw_comm = static_cast<MPI_Comm>(*(mpi_comm->getRawMpiComm()));

        MPI_Allgather(&my_owned_count, 1, MPI_INT,
                      all_owned_counts.data(), 1, MPI_INT, raw_comm);
    }
    else
    {
        all_owned_counts[0] = my_owned_count;
    }

    // --- Compute global offset for this process ---
    global_ordinal_type global_offset = 0;
    for (int r = 0; r < myrank; ++r)
    {
        global_offset += static_cast<global_ordinal_type>(all_owned_counts[static_cast<size_t>(r)]);
    }
    d_tpetra_gid_offset = global_offset;

    // --- Assign contiguous Tpetra GIDs to owned cells ---
    d_tpetra_gid_to_mesh_gid.resize(static_cast<size_t>(my_owned_count));
    for (int i = 0; i < my_owned_count; ++i)
    {
        const auto tpetra_gid = global_offset + static_cast<global_ordinal_type>(i);
        d_tpetra_gid_to_mesh_gid[static_cast<size_t>(i)] = d_owned_cell_global_ids[static_cast<size_t>(i)];
        d_mesh_gid_to_tpetra_gid[d_owned_cell_global_ids[static_cast<size_t>(i)]] = tpetra_gid;
    }

    // --- Resolve ghost cell Tpetra GIDs ---
    const int my_ghost_count = static_cast<int>(d_ghost_cell_global_ids.size());
    d_ghost_cell_tpetra_gids.resize(static_cast<size_t>(my_ghost_count));

    int any_rank_has_ghosts = my_ghost_count > 0 ? 1 : 0;
    if (nranks > 1)
    {
        int global_has_ghosts = 0;
        MPI_Allreduce(
            &any_rank_has_ghosts,
            &global_has_ghosts,
            1,
            MPI_INT,
            MPI_MAX,
            raw_comm);
        any_rank_has_ghosts = global_has_ghosts;
    }

    if (any_rank_has_ghosts != 0)
    {
        // The noncontiguous map's Directory distributes ownership metadata
        // instead of replicating every owned mesh GID on every rank.
        const auto invalid_global_size =
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
        global_ordinal_type local_minimum_mesh_gid =
            std::numeric_limits<global_ordinal_type>::max();
        if (!d_owned_cell_global_ids.empty())
        {
            local_minimum_mesh_gid = *std::min_element(
                d_owned_cell_global_ids.begin(),
                d_owned_cell_global_ids.end());
        }
        global_ordinal_type index_base = local_minimum_mesh_gid;
        Teuchos::reduceAll(
            *comm,
            Teuchos::REDUCE_MIN,
            1,
            &local_minimum_mesh_gid,
            &index_base);
        if (std::none_of(
                all_owned_counts.begin(),
                all_owned_counts.end(),
                [](int count) { return count != 0; }))
        {
            index_base = global_ordinal_type{};
        }

        const auto owned_mesh_gid_map = Teuchos::rcp(new map_type(
            invalid_global_size,
            d_owned_cell_global_ids.data(),
            static_cast<local_ordinal_type>(my_owned_count),
            index_base,
            comm));

        std::vector<int> owner_ranks(static_cast<size_t>(my_ghost_count));
        std::vector<local_ordinal_type> owner_local_ids(
            static_cast<size_t>(my_ghost_count));
        const Teuchos::ArrayView<const global_ordinal_type> ghost_gid_view(
            d_ghost_cell_global_ids.data(), my_ghost_count);
        const Teuchos::ArrayView<int> owner_rank_view(
            owner_ranks.data(), my_ghost_count);
        const Teuchos::ArrayView<local_ordinal_type> owner_local_id_view(
            owner_local_ids.data(), my_ghost_count);

        // This lookup is collective: ranks without ghosts participate with
        // empty views while ranks with ghosts submit only their local queries.
        const auto lookup_status = owned_mesh_gid_map->getRemoteIndexList(
            ghost_gid_view, owner_rank_view, owner_local_id_view);

        int local_missing = lookup_status == Tpetra::IDNotPresent ? 1 : 0;
        int any_rank_missing = local_missing;
        if (nranks > 1)
        {
            MPI_Allreduce(
                &local_missing,
                &any_rank_missing,
                1,
                MPI_INT,
                MPI_MAX,
                raw_comm);
        }
        if (any_rank_missing != 0)
        {
            if (local_missing != 0)
            {
                const auto invalid_local_id =
                    Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
                for (int i = 0; i < my_ghost_count; ++i)
                {
                    if (owner_ranks[static_cast<size_t>(i)] < 0
                        || owner_local_ids[static_cast<size_t>(i)]
                            == invalid_local_id)
                    {
                        throw std::runtime_error(
                            "Ghost cell mesh GID "
                            + std::to_string(d_ghost_cell_global_ids[
                                static_cast<size_t>(i)])
                            + " not found in any process's owned list.");
                    }
                }
            }
            throw std::runtime_error(
                "A ghost cell mesh GID requested by another process was "
                "not found in any process's owned list.");
        }

        std::vector<global_ordinal_type> rank_offsets(
            static_cast<size_t>(nranks), global_ordinal_type{});
        for (int rank = 1; rank < nranks; ++rank)
        {
            rank_offsets[static_cast<size_t>(rank)] =
                rank_offsets[static_cast<size_t>(rank - 1)]
                + static_cast<global_ordinal_type>(
                    all_owned_counts[static_cast<size_t>(rank - 1)]);
        }

        for (int i = 0; i < my_ghost_count; ++i)
        {
            const auto mesh_gid = d_ghost_cell_global_ids[static_cast<size_t>(i)];
            const auto owner_rank = owner_ranks[static_cast<size_t>(i)];
            const auto owner_local_id = owner_local_ids[static_cast<size_t>(i)];
            const auto tpetra_gid =
                rank_offsets[static_cast<size_t>(owner_rank)]
                + static_cast<global_ordinal_type>(owner_local_id);
            d_ghost_cell_tpetra_gids[static_cast<size_t>(i)] = tpetra_gid;
            d_mesh_gid_to_tpetra_gid[mesh_gid] = tpetra_gid;
        }
    }

    d_contiguous_tpetra_gids_assigned = true;
}

/**
 * @brief Clear contiguous Tpetra cell-ID state before rebuilding a mesh.
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::reset_contiguous_tpetra_gids() noexcept
{
    d_ghost_cell_tpetra_gids.clear();
    d_mesh_gid_to_tpetra_gid.clear();
    d_tpetra_gid_to_mesh_gid.clear();
    d_tpetra_gid_offset = 0;
    d_contiguous_tpetra_gids_assigned = false;
}

/**
 * @brief Rebuild compact boundary batches from the finalized local faces.
 *
 * Partitioning changes face local ordinals, so batches produced while
 * importing source geometry must not survive a local rebuild.
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::rebuild_boundary_face_batches()
{
    decltype(d_boundary_id_to_face_batch){}.swap(
        d_boundary_id_to_face_batch);
    for (size_t fid = 0; fid < d_faces.size(); ++fid)
    {
        const auto boundary_id = d_faces[fid].boundary_id;
        if (boundary_id == invalid_boundary_id)
        {
            continue;
        }

        auto& face_batch = d_boundary_id_to_face_batch[boundary_id];
        face_batch.id = boundary_id;
        face_batch.face_lids.push_back(
            detail::checked_size_to_ordinal<local_ordinal_type>(
                fid, "boundary face local id"));
    }
}

/**
 * @brief create Tpetra maps for owned and overlap cells based on the mesh connectivity information.
 * 
 * @tparam Pack 
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::create_maps()
{
    // Topology and geometry are final at this point. Persist compact host
    // operator views before map construction uses the ownership accessors.
    create_host_views();

    if (d_boundary_id_to_face_batch.empty())
    {
        rebuild_boundary_face_batches();
    }

    // Ensure contiguous Tpetra GIDs are assigned before building maps.
    assign_contiguous_tpetra_gids();

    // --- Build Tpetra overlap GID list (owned + ghost, all using Tpetra GIDs) ---
    ArrGO overlap_tpetra_gids;
    overlap_tpetra_gids.reserve(d_owned_cell_global_ids.size() + d_ghost_cell_tpetra_gids.size());
    for (local_ordinal_type lid = 0; lid < static_cast<local_ordinal_type>(d_owned_cell_global_ids.size()); ++lid)
    {
        const auto tpetra_gid = lid + d_tpetra_gid_offset;
        overlap_tpetra_gids.push_back(tpetra_gid);
    }
    overlap_tpetra_gids.append_range(d_ghost_cell_tpetra_gids);

    const auto invalid_global_size = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const global_ordinal_type index_base = 0;
    const auto comm = Tpetra::getDefaultComm();

    // Creates a contiguous Tpetra map.
    d_owned_cell_map = Teuchos::rcp(new map_type(
        invalid_global_size,
        num_owned_cells(),
        index_base,
        comm));

    d_overlap_cell_map = Teuchos::rcp(new map_type(
        invalid_global_size,
        overlap_tpetra_gids.data(),
        detail::checked_size_to_ordinal<local_ordinal_type>(overlap_tpetra_gids.size(), "overlap cell map"),
        index_base,
        comm));

    ArrGO owned_face_gids;
    owned_face_gids.reserve(d_faces.size());
    for (size_t fid = 0; fid < d_faces.size(); ++fid)
    {
        const auto face_lid =
            detail::checked_size_to_ordinal<local_ordinal_type>(
                fid, "face local id");
        if (!is_owned_face(face_lid))
        {
            continue;
        }

        owned_face_gids.push_back(
            fid < d_owned_face_global_ids.size()
          ? d_owned_face_global_ids[fid]
          : static_cast<global_ordinal_type>(fid));
    }

    // Owned face map: includes all faces owned by locally owned cells.
    d_owned_face_map = Teuchos::rcp(new map_type(
        invalid_global_size,
        owned_face_gids.data(),
        detail::checked_size_to_ordinal<local_ordinal_type>(owned_face_gids.size(), "owned face map"),
        index_base,
        comm));

    ArrGO boundary_face_gids;
    for (const auto& [batch_id, face_batch] : d_boundary_id_to_face_batch)
    {
        (void)batch_id;
        for (const auto& face_lid : face_batch.face_lids)
        {
            if (!is_owned_face(face_lid))
            {
                continue;
            }
            const auto face_index = static_cast<size_t>(face_lid);
            boundary_face_gids.push_back(
                face_index < d_owned_face_global_ids.size()
              ? d_owned_face_global_ids[face_index]
              : static_cast<global_ordinal_type>(face_lid));
        }
    }
    // Boundary face map: includes all faces on physical boundaries (subset of owned faces)
    d_boundary_face_map = Teuchos::rcp(new map_type(
        invalid_global_size,
        boundary_face_gids.data(),
        detail::checked_size_to_ordinal<local_ordinal_type>(boundary_face_gids.size(), "boundary face map"),
        index_base,
        comm));
}

/**
 * @brief Precompute static distances from each cell center to its face centers.
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::create_cell_face_distances()
{
    d_cell_face_distances.clear();

    size_t total_cell_faces = 0;
    for (const auto& cell_info : d_cells)
    {
        total_cell_faces += cell_info.faces.size();
    }
    d_cell_face_distances.reserve(total_cell_faces);

    for (size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto cell_lid =
            detail::checked_size_to_ordinal<local_ordinal_type>(lid, "cell local id");
        const auto offset = d_cell_face_distances.size();

        for (const auto face_lid : d_cells[lid].faces)
        {
            if (static_cast<size_t>(face_lid) >= d_faces.size())
            {
                throw std::out_of_range(
                    "Cell references a face outside the local mesh.");
            }
            const auto& face_info =
                d_faces[static_cast<size_t>(face_lid)];
            if (face_info.owner == cell_lid)
            {
                d_cell_face_distances.push_back(
                    face_info.owner_to_face_distance);
            }
            else if (face_info.neighbor == cell_lid)
            {
                d_cell_face_distances.push_back(
                    face_info.neighbor_to_face_distance);
            }
            else
            {
                throw std::invalid_argument(
                    "Cell is not adjacent to one of its faces.");
            }
        }

        d_cells[lid].face_distances =
            ViewReal(d_cell_face_distances.data() + offset,
                     d_cells[lid].faces.size());
    }
}

/**
 * @brief Materialize compact host-side cell and face operator views.
 *
 * Host finite-volume loops can read dense topology or geometry streams
 * without striding through the much larger metadata records. These arrays
 * also provide the source for the device mirrors.
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::create_host_views()
{
    // Any previous device mirrors refer to the old host topology/geometry.
    reset_device_views();

    auto& cell_geometry = d_host_views.cell_geometry;
    cell_geometry.volume.clear();
    cell_geometry.centroid.clear();
    cell_geometry.volume.reserve(d_cells.size());
    cell_geometry.centroid.reserve(d_cells.size());

    for (const auto& cell_info : d_cells)
    {
        cell_geometry.volume.push_back(cell_info.volume);
        cell_geometry.centroid.push_back(cell_info.center);
    }

    auto& face_topology = d_host_views.face_topology;
    face_topology.owner.clear();
    face_topology.neighbor.clear();
    face_topology.type.clear();
    face_topology.boundary_id.clear();
    face_topology.owner.reserve(d_faces.size());
    face_topology.neighbor.reserve(d_faces.size());
    face_topology.type.reserve(d_faces.size());
    face_topology.boundary_id.reserve(d_faces.size());

    auto& face_geometry = d_host_views.face_geometry;
    face_geometry.area.clear();
    face_geometry.unit_normal_from_owner.clear();
    face_geometry.area_vector.clear();
    face_geometry.centroid.clear();
    face_geometry.owner_to_neighbor.clear();
    face_geometry.owner_to_face_distance.clear();
    face_geometry.neighbor_to_face_distance.clear();
    face_geometry.cell_center_distance.clear();
    face_geometry.area.reserve(d_faces.size());
    face_geometry.unit_normal_from_owner.reserve(d_faces.size());
    face_geometry.area_vector.reserve(d_faces.size());
    face_geometry.centroid.reserve(d_faces.size());
    face_geometry.owner_to_neighbor.reserve(d_faces.size());
    face_geometry.owner_to_face_distance.reserve(d_faces.size());
    face_geometry.neighbor_to_face_distance.reserve(d_faces.size());
    face_geometry.cell_center_distance.reserve(d_faces.size());

    for (const auto& face_info : d_faces)
    {
        face_topology.owner.push_back(face_info.owner);
        face_topology.neighbor.push_back(face_info.neighbor);
        face_topology.type.push_back(static_cast<int>(face_info.type));
        face_topology.boundary_id.push_back(face_info.boundary_id);

        face_geometry.area.push_back(face_info.area);
        face_geometry.unit_normal_from_owner.push_back(
            face_info.unit_normal_from_owner);
        face_geometry.area_vector.push_back(
            face_info.unit_normal_from_owner * face_info.area);
        face_geometry.centroid.push_back(face_info.center);
        face_geometry.owner_to_face_distance.push_back(
            face_info.owner_to_face_distance);
        face_geometry.neighbor_to_face_distance.push_back(
            face_info.neighbor_to_face_distance);
        face_geometry.cell_center_distance.push_back(
            face_info.cell_center_distance);

        if (face_info.neighbor == invalid_id<local_ordinal_type>())
        {
            face_geometry.owner_to_neighbor.push_back(Vec3{});
        }
        else
        {
            face_geometry.owner_to_neighbor.push_back(
                d_cells[static_cast<size_t>(face_info.neighbor)].center
                - d_cells[static_cast<size_t>(face_info.owner)].center);
        }
    }
}

/**
 * @brief create Kokkos views for mesh data on the device.
 * 
 * @tparam Pack 
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::create_device_views() const
{
    if (d_device_views_created)
    {
        return;
    }

    if (d_host_views.cell_geometry.volume.size() != d_cells.size()
        || d_host_views.face_topology.owner.size() != d_faces.size())
    {
        throw std::logic_error(
            "Device mesh views require finalized host mesh views.");
    }

    ArrGO cell_gid;
    ArrInt cell_type;
    ArrLO cell_face_offset{0};
    ArrLO cell_face_ids;
    ArrReal cell_face_distance_values;

    cell_gid.reserve(d_cells.size());
    cell_type.reserve(d_cells.size());

    for (size_t i = 0; i < d_owned_cell_global_ids.size(); ++i)
    {
        const auto tpetra_gid = d_tpetra_gid_offset + static_cast<global_ordinal_type>(i);
        cell_gid.push_back(tpetra_gid);
    }
    cell_gid.append_range(d_ghost_cell_tpetra_gids);

    for (const auto& cell_info : d_cells)
    {
        cell_type.push_back(static_cast<int>(cell_info.type));

        cell_face_ids.insert(cell_face_ids.end(), cell_info.faces.begin(), cell_info.faces.end());
        cell_face_distance_values.insert(cell_face_distance_values.end(),
                                         cell_info.face_distances.begin(),
                                         cell_info.face_distances.end());
        cell_face_offset.push_back(detail::checked_size_to_ordinal<local_ordinal_type>(
            cell_face_ids.size(), "cell-face connectivity"));
    }

    ArrLO cell_node_offset{0};
    ArrLO cell_node_ids;
    ArrLO face_node_offset{0};
    ArrLO face_node_ids;

    for (const auto& cell : d_cells)
    {
        for (auto node_id : cell.node_gids)
        {
            cell_node_ids.push_back(d_node_gid_to_lid.at(node_id));
        }

        cell_node_offset.push_back(detail::checked_size_to_ordinal<local_ordinal_type>(
            cell_node_ids.size(), "cell-node connectivity"));
    }

    for (const auto& face_info : d_faces)
    {
        for (const auto node_id : face_info.node_gids)
        {
            face_node_ids.push_back(d_node_gid_to_lid.at(node_id));
        }

        face_node_offset.push_back(detail::checked_size_to_ordinal<local_ordinal_type>(
            face_node_ids.size(), "face-node connectivity"));
    }

    d_device_views.cell_gid = make_vector_view("cell_gid", cell_gid);
    d_device_views.cell_type = make_vector_view("cell_type", cell_type);

    d_device_views.cell_volume = make_vector_view(
        "cell_volume", d_host_views.cell_geometry.volume);
    d_device_views.cell_centroid = make_vectorV3D_view(
        "cell_centroid", d_host_views.cell_geometry.centroid);

    d_device_views.cell_face_offset = make_vector_view("cell_face_offset", cell_face_offset);
    d_device_views.cell_face_ids = make_vector_view("cell_face_ids", cell_face_ids);
    d_device_views.cell_face_distance = make_vector_view("cell_face_distance",
                                                         cell_face_distance_values);

    d_device_views.face_owner = make_vector_view(
        "face_owner", d_host_views.face_topology.owner);
    d_face_neighbor_device = make_mutable_vector_view(
        "face_neighbor", d_host_views.face_topology.neighbor);
    d_device_views.face_neighbor = d_face_neighbor_device;
    d_device_views.face_type = make_vector_view(
        "face_type", d_host_views.face_topology.type);
    d_device_views.face_batch = make_vector_view(
        "face_batch", d_host_views.face_topology.boundary_id);

    d_device_views.face_area = make_vector_view(
        "face_area", d_host_views.face_geometry.area);
    d_device_views.face_area_vector = make_vectorV3D_view(
        "face_area_vector", d_host_views.face_geometry.area_vector);
    d_device_views.face_centroid = make_vectorV3D_view(
        "face_centroid", d_host_views.face_geometry.centroid);

    d_device_views.node_coord =
        make_vectorV3D_view("node_coord", d_node_coords);

    d_device_views.cell_node_offset = make_vector_view("cell_node_offset", cell_node_offset);
    d_device_views.cell_node_ids = make_vector_view("cell_node_ids", cell_node_ids);

    d_device_views.face_node_offset = make_vector_view("face_node_offset", face_node_offset);
    d_device_views.face_node_ids = make_vector_view("face_node_ids", face_node_ids);
    d_device_views_created = true;
}

/**
 * @brief Release cached device mirrors after host topology changes.
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::reset_device_views() const noexcept
{
    d_device_views = DeviceViews{};
    d_face_neighbor_device = kokkos_1dview<local_ordinal_type>{};
    d_device_views_created = false;
}

/**
 * @brief Change face owner to a locally owned cell if the current owner is a ghost cell and the neighbor is owned.
 * 
 * @tparam Pack 
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::prefer_owned_face_owners()
{
    for (auto& face_info : d_faces)
    {
        if (face_info.neighbor == invalid_id<local_ordinal_type>()) continue;

        const auto owner = face_info.owner;
        const auto neighbor = face_info.neighbor;
        if (!d_cells[static_cast<size_t>(owner)].owned
            && d_cells[static_cast<size_t>(neighbor)].owned)
        {
            face_info.owner = neighbor;
            face_info.neighbor = owner;
            std::swap(
                face_info.unit_normal_from_owner,
                face_info.unit_normal_from_neighbor);
            std::swap(
                face_info.owner_to_face_distance,
                face_info.neighbor_to_face_distance);
        }
    }
}


/**
 * @brief Validate internal consistency of the mesh connectivity and data structures.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::check_connectivity() const
{
    CHECK(d_owned_cell_ids.size() == d_owned_cell_global_ids.size());
    CHECK(d_owned_cell_ids.size() <= d_cells.size());

    for (size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto& cell = d_cells[lid];
        CHECK(cell.type != CellType::INVALID);

        for (const auto fid : cell.faces)
        {
            CHECK(static_cast<size_t>(fid) < d_faces.size());
        }
    }

    for (const auto lid : d_owned_cell_ids)
    {
        const auto gid = cell_global_id(lid);
        CHECK(d_cell_gid_to_lid.find(gid) != d_cell_gid_to_lid.end());
        CHECK(lid == d_cell_gid_to_lid.at(gid));
        CHECK(static_cast<size_t>(lid) < d_cells.size());
        CHECK(d_cells[static_cast<size_t>(lid)].owned);
    }
    for (size_t owned = 0; owned < d_owned_cell_ids.size(); ++owned)
    {
        CHECK(
            d_owned_cell_ids[owned]
            == detail::checked_size_to_ordinal<local_ordinal_type>(
                owned, "owned cell local id"));
    }
    for (size_t ghost = d_owned_cell_ids.size();
         ghost < d_cells.size();
         ++ghost)
    {
        CHECK(!d_cells[ghost].owned);
    }
    for (auto gid : d_ghost_cell_global_ids)
    {
        CHECK(d_cell_gid_to_lid.find(gid) != d_cell_gid_to_lid.end());
        auto lid = d_cell_gid_to_lid.at(gid);
        CHECK(lid == d_cell_gid_to_lid.at(gid));
    }

    for (size_t fid = 0; fid < d_faces.size(); ++fid)
    {
        const auto& face = d_faces[fid];
        CHECK(static_cast<size_t>(face.owner) < d_cells.size());
        CHECK(face.neighbor == invalid_id<local_ordinal_type>()
              || static_cast<size_t>(face.neighbor) < d_cells.size());
        CHECK((face.type == FaceType::TRIANGLE && face.node_gids.size() == 3)
         || (face.type == FaceType::QUAD && face.node_gids.size() == 4));
        CHECK(face.area >= 0.0);

        if (face.boundary_id != invalid_boundary_id)
        {
            const auto batch_iter = d_boundary_id_to_face_batch.find(face.boundary_id);
            CHECK(batch_iter != d_boundary_id_to_face_batch.end());
            auto& face_lids = batch_iter->second.face_lids;
            CHECK(std::find(face_lids.begin(), face_lids.end(),
                            static_cast<local_ordinal_type>(fid)) != face_lids.end());
        }
    }
}

} // namespace SimpleFluid
