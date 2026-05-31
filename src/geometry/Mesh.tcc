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
#include <unordered_map>
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
 * the corresponding Tpetra GIDs for ghost cells by gathering the owned
 * mesh-GID lists from all ranks.  The resulting maps are stored in:
 * - d_owned_cell_tpetra_gids
 * - d_ghost_cell_tpetra_gids
 * - d_mesh_gid_to_tpetra_gid
 * - d_tpetra_gid_to_mesh_gid
 *
 * Idempotent: if d_owned_cell_tpetra_gids is already populated the call is a no-op.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::assign_contiguous_tpetra_gids()
{
    if (!d_owned_cell_tpetra_gids.empty())
    {
        return; // already assigned
    }

    const auto comm = Tpetra::getDefaultComm();
    const int myrank = comm->getRank();
    const int nranks = comm->getSize();

    const int my_owned_count = static_cast<int>(d_owned_cell_global_ids.size());

    // --- Gather owned cell counts from all ranks ---
    std::vector<int> all_owned_counts(static_cast<std::size_t>(nranks), 0);

    if (nranks > 1)
    {
        auto* mpi_comm = dynamic_cast<const Teuchos::MpiComm<int>*>(comm.get());
        if (mpi_comm == nullptr)
        {
            throw std::runtime_error("assign_contiguous_tpetra_gids requires an MPI communicator.");
        }
        const auto raw_comm = static_cast<MPI_Comm>(*(mpi_comm->getRawMpiComm()));

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
        global_offset += static_cast<global_ordinal_type>(all_owned_counts[static_cast<std::size_t>(r)]);
    }
    d_tpetra_gid_offset = global_offset;

    const global_ordinal_type total_global_owned =
        global_offset + static_cast<global_ordinal_type>(my_owned_count);
    for (int r = myrank + 1; r < nranks; ++r)
    {
        // total_global_owned not strictly needed, computed for completeness
        (void)r;
    }

    // --- Assign contiguous Tpetra GIDs to owned cells ---
    d_owned_cell_tpetra_gids.resize(static_cast<std::size_t>(my_owned_count));
    d_tpetra_gid_to_mesh_gid.resize(static_cast<std::size_t>(my_owned_count));
    for (int i = 0; i < my_owned_count; ++i)
    {
        const auto tpetra_gid = global_offset + static_cast<global_ordinal_type>(i);
        d_owned_cell_tpetra_gids[static_cast<std::size_t>(i)] = tpetra_gid;
        d_tpetra_gid_to_mesh_gid[static_cast<std::size_t>(i)] = d_owned_cell_global_ids[static_cast<std::size_t>(i)];
        d_mesh_gid_to_tpetra_gid[d_owned_cell_global_ids[static_cast<std::size_t>(i)]] = tpetra_gid;
    }

    // --- Resolve ghost cell Tpetra GIDs ---
    const int my_ghost_count = static_cast<int>(d_ghost_cell_global_ids.size());
    d_ghost_cell_tpetra_gids.resize(static_cast<std::size_t>(my_ghost_count));

    if (my_ghost_count > 0 && nranks > 1)
    {
        auto* mpi_comm_ptr = dynamic_cast<const Teuchos::MpiComm<int>*>(comm.get());
        const auto raw_comm = static_cast<MPI_Comm>(*(mpi_comm_ptr->getRawMpiComm()));

        // --- Gather all owned mesh GIDs from all processes ---
        std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
        for (int r = 1; r < nranks; ++r)
        {
            displs[static_cast<std::size_t>(r)] =
                displs[static_cast<std::size_t>(r - 1)]
                + all_owned_counts[static_cast<std::size_t>(r - 1)];
        }
        const int total_global = displs.back() + all_owned_counts.back();

        // Determine MPI datatype for global_ordinal_type
        MPI_Datatype mpi_go_type;
        if constexpr (std::is_same_v<global_ordinal_type, long long>)
        {
            mpi_go_type = MPI_LONG_LONG;
        }
        else if constexpr (std::is_same_v<global_ordinal_type, long>)
        {
            mpi_go_type = MPI_LONG;
        }
        else if constexpr (std::is_same_v<global_ordinal_type, int>)
        {
            mpi_go_type = MPI_INT;
        }
        else
        {
            // Fallback: use byte size to determine type
            if (sizeof(global_ordinal_type) == 8)
                mpi_go_type = MPI_LONG_LONG;
            else if (sizeof(global_ordinal_type) == 4)
                mpi_go_type = MPI_INT;
            else
                throw std::runtime_error("Unsupported global_ordinal_type size for MPI.");
        }

        std::vector<global_ordinal_type> all_owned_gids(
            static_cast<std::size_t>(std::max(total_global, 1)));

        MPI_Allgatherv(
            d_owned_cell_global_ids.data(), my_owned_count, mpi_go_type,
            all_owned_gids.data(), all_owned_counts.data(), displs.data(),
            mpi_go_type, raw_comm);

        // --- Build global mesh_gid -> tpetra_gid lookup ---
        std::unordered_map<global_ordinal_type, global_ordinal_type> global_mesh_to_tpetra;
        for (int r = 0; r < nranks; ++r)
        {
            global_ordinal_type rank_offset = 0;
            for (int rr = 0; rr < r; ++rr)
            {
                rank_offset += static_cast<global_ordinal_type>(
                    all_owned_counts[static_cast<std::size_t>(rr)]);
            }
            const int count = all_owned_counts[static_cast<std::size_t>(r)];
            const int begin = displs[static_cast<std::size_t>(r)];
            for (int i = 0; i < count; ++i)
            {
                const auto mesh_gid = all_owned_gids[static_cast<std::size_t>(begin + i)];
                const auto tpetra_gid = rank_offset + static_cast<global_ordinal_type>(i);
                global_mesh_to_tpetra[mesh_gid] = tpetra_gid;
            }
        }

        // --- Resolve Tpetra GIDs for ghost cells ---
        for (int i = 0; i < my_ghost_count; ++i)
        {
            const auto mesh_gid = d_ghost_cell_global_ids[static_cast<std::size_t>(i)];
            const auto iter = global_mesh_to_tpetra.find(mesh_gid);
            if (iter == global_mesh_to_tpetra.end())
            {
                throw std::runtime_error(
                    "Ghost cell mesh GID " + std::to_string(mesh_gid)
                    + " not found in any process's owned list.");
            }
            const auto tpetra_gid = iter->second;
            d_ghost_cell_tpetra_gids[static_cast<std::size_t>(i)] = tpetra_gid;
            d_mesh_gid_to_tpetra_gid[mesh_gid] = tpetra_gid;
        }
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
    if (d_boundary_id_to_face_patch.empty())
    {
        for (std::size_t fid = 0; fid < d_faces.size(); ++fid)
        {
            const auto boundary_id = d_faces[fid].boundary_id;
            if (boundary_id == invalid_boundary_id)
            {
                continue;
            }

            auto& face_patch = d_boundary_id_to_face_patch[boundary_id];
            face_patch.id = boundary_id;
            face_patch.face_lids.push_back(
                detail::checked_size_to_ordinal<local_ordinal_type>(
                    fid, "boundary face local id"));
        }
    }

    // Ensure contiguous Tpetra GIDs are assigned before building maps.
    assign_contiguous_tpetra_gids();

    // --- Build Tpetra overlap GID list (owned + ghost, all using Tpetra GIDs) ---
    ArrGO overlap_tpetra_gids = d_owned_cell_tpetra_gids;
    overlap_tpetra_gids.reserve(overlap_tpetra_gids.size() + d_ghost_cell_tpetra_gids.size());
    overlap_tpetra_gids.insert(overlap_tpetra_gids.end(),
                               d_ghost_cell_tpetra_gids.begin(),
                               d_ghost_cell_tpetra_gids.end());

    const auto invalid_global_size = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const global_ordinal_type index_base = 0;
    const auto comm = Tpetra::getDefaultComm();

    d_owned_cell_map = Teuchos::rcp(new map_type(
        invalid_global_size,
        d_owned_cell_tpetra_gids.data(),
        detail::checked_size_to_ordinal<local_ordinal_type>(d_owned_cell_tpetra_gids.size(), "owned cell map"),
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
    for (std::size_t fid = 0; fid < d_faces.size(); ++fid)
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
    for (const auto& [patch_id, face_patch] : d_boundary_id_to_face_patch)
    {
        (void)patch_id;
        for (const auto& face_lid : face_patch.face_lids)
        {
            if (!is_owned_face(face_lid))
            {
                continue;
            }
            const auto face_index = static_cast<std::size_t>(face_lid);
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

    std::size_t total_cell_faces = 0;
    for (const auto& cell_info : d_cells)
    {
        total_cell_faces += cell_info.faces.size();
    }
    d_cell_face_distances.reserve(total_cell_faces);

    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto cell_lid =
            detail::checked_size_to_ordinal<local_ordinal_type>(lid, "cell local id");
        const auto offset = d_cell_face_distances.size();

        for (const auto face_lid : d_cells[lid].faces)
        {
            d_cell_face_distances.push_back(cell_to_face_distance(face_lid, cell_lid));
        }

        d_cells[lid].face_distances =
            ViewReal(d_cell_face_distances.data() + offset,
                     d_cells[lid].faces.size());
    }
}

/**
 * @brief create Kokkos views for mesh data on the device.
 * 
 * @tparam Pack 
 */
template<TpetraTypePack Pack>
void Mesh<Pack>::create_device_views()
{
    ArrGO cell_gid;
    ArrInt cell_type;
    ArrReal cell_volume_values;
    ArrVec3 cell_centroid_values;
    ArrLO cell_face_offset{0};
    ArrLO cell_face_ids;
    ArrReal cell_face_distance_values;

    cell_gid.reserve(d_cells.size());
    cell_type.reserve(d_cells.size());
    cell_volume_values.reserve(d_cells.size());
    cell_centroid_values.reserve(d_cells.size());

    cell_gid.append_range(d_owned_cell_tpetra_gids);
    cell_gid.append_range(d_ghost_cell_tpetra_gids);

    for (const auto& cell_info : d_cells)
    {
        cell_type.push_back(static_cast<int>(cell_info.type));
        cell_volume_values.push_back(cell_info.volume);
        cell_centroid_values.push_back(cell_info.center);

        cell_face_ids.insert(cell_face_ids.end(), cell_info.faces.begin(), cell_info.faces.end());
        cell_face_distance_values.insert(cell_face_distance_values.end(),
                                         cell_info.face_distances.begin(),
                                         cell_info.face_distances.end());
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

    ArrVec3 node_coord_values = d_node_coords;
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

    d_device_views.cell_volume = make_vector_view("cell_volume", cell_volume_values);
    d_device_views.cell_centroid = make_vectorV3D_view("cell_centroid", cell_centroid_values);

    d_device_views.cell_face_offset = make_vector_view("cell_face_offset", cell_face_offset);
    d_device_views.cell_face_ids = make_vector_view("cell_face_ids", cell_face_ids);
    d_device_views.cell_face_distance = make_vector_view("cell_face_distance",
                                                         cell_face_distance_values);

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
        if (!d_cells[static_cast<std::size_t>(owner)].owned
            && d_cells[static_cast<std::size_t>(neighbor)].owned)
        {
            face_info.owner = neighbor;
            face_info.neighbor = owner;
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

    for (std::size_t lid = 0; lid < d_cells.size(); ++lid)
    {
        const auto& cell = d_cells[lid];
        CHECK(cell.type != CellType::INVALID);

        for (const auto fid : cell.faces)
        {
            CHECK(static_cast<std::size_t>(fid) < d_faces.size());
        }
    }

    for (const auto lid : d_owned_cell_ids)
    {
        const auto gid = cell_global_id(lid);
        CHECK(d_cell_gid_to_lid.find(gid) != d_cell_gid_to_lid.end());
        CHECK(lid == d_cell_gid_to_lid.at(gid));
        CHECK(static_cast<std::size_t>(lid) < d_cells.size());
        CHECK(d_cells[static_cast<std::size_t>(lid)].owned);
    }
    for (auto gid : d_ghost_cell_global_ids)
    {
        CHECK(d_cell_gid_to_lid.find(gid) != d_cell_gid_to_lid.end());
        auto lid = d_cell_gid_to_lid.at(gid);
        CHECK(lid == d_cell_gid_to_lid.at(gid));
    }

    for (std::size_t fid = 0; fid < d_faces.size(); ++fid)
    {
        const auto& face = d_faces[fid];
        CHECK(static_cast<std::size_t>(face.owner) < d_cells.size());
        CHECK(face.neighbor == invalid_id<local_ordinal_type>()
              || static_cast<std::size_t>(face.neighbor) < d_cells.size());
        CHECK((face.type == FaceType::TRIANGLE && face.node_gids.size() == 3)
         || (face.type == FaceType::QUAD && face.node_gids.size() == 4));
        CHECK(face.area >= 0.0);

        if (face.boundary_id != invalid_boundary_id)
        {
            const auto patch_iter = d_boundary_id_to_face_patch.find(face.boundary_id);
            CHECK(patch_iter != d_boundary_id_to_face_patch.end());
            auto& face_lids = patch_iter->second.face_lids;
            CHECK(std::find(face_lids.begin(), face_lids.end(),
                            static_cast<local_ordinal_type>(fid)) != face_lids.end());
        }
    }
}

} // namespace SimpleFluid
