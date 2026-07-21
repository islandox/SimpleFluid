/**
 * @file MeshPartitioner.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief template implementations for MeshPartitioner class.
 * @version 0.1
 * @date 2026-05-29
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "MeshPartitioner.hh"

#include <map>
#include <type_traits>
#include "modules/Zoltan2.hh"

using namespace SimpleFluid;


/**
 * @brief Main entry point — partition source mesh cells across MPI ranks.
 *
 * Five-phase algorithm:
 * 1. **Zoltan2/ParMETIS partition** — build a CRS graph from cell
 *    adjacency and call Zoltan2 to assign each cell a destination rank.
 * 2. **Cell redistribution** — serialise owned cells into packets,
 *    exchange them via MPI_Alltoallv.
 * 3. **Ghost detection** — for each owned cell, identify adjacent
 *    off-rank cells that are not already owned.
 * 4. **Ghost exchange** — request ghost packets from the ranks that
 *    own them and validate non-owning views over the received bytes.
 * 5. **Mesh rebuild** — reconstruct the per-rank d_cells, d_faces,
 *    node tables, and face geometry from owned + ghost packets.
 *
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 * Source cells may be replicated, distributed, or held only on rank zero.
 *
 * @param mesh The source mesh to partition (modified in place).
 * @param comm Teuchos MPI communicator.
 * @return true if partitioning occurred, false if single-rank or
 *         already partitioned.
 */
template<TpetraTypePack Pack>
bool MeshPartitioner<Pack>::partition(Mesh<Pack>& mesh, const Teuchos::RCP<const comm_type>& comm)
{
    const int nranks = comm->getSize();
    const int myrank = comm->getRank();
    if (nranks <= 1) return false;
    const int local_already_partitioned =
        mesh.d_ghost_cell_global_ids.empty() ? 0 : 1;
    int any_already_partitioned = 0;
    MPI_Allreduce(
        &local_already_partitioned,
        &any_already_partitioned,
        1,
        MPI_INT,
        MPI_MAX,
        get_mpi_comm(*comm));
    if (any_already_partitioned != 0) return false;

    auto source_selection =
        compute_source_selection(mesh, comm);
    auto gid_to_rank =
        compute_gid_to_rank_map(mesh, source_selection, comm);

    auto unique_packets_by_gid = [](std::vector<PacketView>& pkts)
    {
        std::unordered_set<GO> seen;
        pkts.erase(
            std::remove_if(pkts.begin(), pkts.end(),
                [&](const PacketView& packet)
                {
                    return !seen.insert(packet.gid()).second;
                }),
            pkts.end());
    };

    std::vector<char> owned_packet_storage;
    std::vector<PacketView> owned_pkts;
    {
        std::vector<std::uint32_t> packet_counts(
            static_cast<size_t>(nranks), 0);
        std::vector<size_t> body_bytes(
            static_cast<size_t>(nranks), 0);
        for (size_t index = 0;
             index < mesh.d_owned_cell_global_ids.size();
             ++index)
        {
            const auto gid = mesh.d_owned_cell_global_ids[index];
            if (!source_selection.provides(gid, myrank))
            {
                continue;
            }
            const auto destination = gid_to_rank.find(gid);
            if (destination == gid_to_rank.end())
            {
                throw std::runtime_error(
                    "MeshPartitioner source rank is missing the "
                    "partition assignment for cell GID "
                    + std::to_string(gid));
            }
            const auto rank_index =
                static_cast<size_t>(destination->second);
            if (rank_index >= static_cast<size_t>(nranks))
            {
                throw std::runtime_error(
                    "MeshPartitioner produced an invalid destination "
                    "rank.");
            }
            ++packet_counts[rank_index];
            body_bytes[rank_index] +=
                Packet::serialized_mesh_cell_size(
                    mesh, mesh.d_owned_cell_ids[index]);
        }

        std::vector<int> scnt(nranks, 0);
        for (int rank = 0; rank < nranks; ++rank)
        {
            const auto rank_index = static_cast<size_t>(rank);
            if (packet_counts[rank_index] == 0)
            {
                continue;
            }
            const auto bytes =
                sizeof(std::uint32_t) + body_bytes[rank_index];
            if (bytes > static_cast<size_t>(
                    std::numeric_limits<int>::max()))
            {
                throw std::overflow_error(
                    "MeshPartitioner send buffer exceeds MPI int "
                    "count capacity.");
            }
            scnt[rank_index] = static_cast<int>(bytes);
        }

        std::vector<int> rcnt(nranks, 0);
        MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, get_mpi_comm(*comm));
        std::vector<int> sd(nranks, 0), rd(nranks, 0);
        for (int r = 1; r < nranks; ++r) {
            sd[static_cast<size_t>(r)] = sd[static_cast<size_t>(r - 1)] + scnt[static_cast<size_t>(r - 1)];
            rd[static_cast<size_t>(r)] = rd[static_cast<size_t>(r - 1)] + rcnt[static_cast<size_t>(r - 1)];
        }

        const auto total_send = sd.back() + scnt.back();
        std::vector<char> flat_s(
            static_cast<size_t>(std::max(total_send, 1)));
        std::vector<size_t> write_offsets(
            static_cast<size_t>(nranks), 0);
        for (int rank = 0; rank < nranks; ++rank)
        {
            const auto rank_index = static_cast<size_t>(rank);
            if (packet_counts[rank_index] == 0)
            {
                continue;
            }
            const auto offset =
                static_cast<size_t>(sd[rank_index]);
            std::memcpy(
                flat_s.data() + offset,
                &packet_counts[rank_index],
                sizeof(std::uint32_t));
            write_offsets[rank_index] =
                offset + sizeof(std::uint32_t);
        }

        for (size_t index = 0;
             index < mesh.d_owned_cell_global_ids.size();
             ++index)
        {
            const auto gid = mesh.d_owned_cell_global_ids[index];
            if (!source_selection.provides(gid, myrank))
            {
                continue;
            }
            const auto destination = gid_to_rank.at(gid);
            const auto rank_index =
                static_cast<size_t>(destination);
            const auto lid = mesh.d_owned_cell_ids[index];
            const auto bytes =
                Packet::serialized_mesh_cell_size(mesh, lid);
            Packet::serialize_mesh_cell(
                flat_s.data() + write_offsets[rank_index],
                bytes,
                mesh,
                lid,
                gid,
                [&](GO neighbor_gid)
                {
                    const auto neighbor =
                        gid_to_rank.find(neighbor_gid);
                    if (neighbor == gid_to_rank.end())
                    {
                        throw std::runtime_error(
                            "MeshPartitioner source rank is missing "
                            "the partition assignment for neighboring "
                            "cell GID "
                            + std::to_string(neighbor_gid));
                    }
                    return neighbor->second;
                });
            write_offsets[rank_index] += bytes;
        }

        owned_packet_storage.resize(
            static_cast<size_t>(std::max(rd.back() + rcnt.back(), 1)));
        MPI_Alltoallv(flat_s.data(), scnt.data(), sd.data(), MPI_CHAR,
                     owned_packet_storage.data(), rcnt.data(), rd.data(), MPI_CHAR,
                     get_mpi_comm(*comm));
        std::vector<char>{}.swap(flat_s);

        for (int r = 0; r < nranks; ++r) {
            int sz = rcnt[static_cast<size_t>(r)];
            if (sz <= 0) continue;
            auto pkts = Packet::view_packets(
                owned_packet_storage.data()
                    + rd[static_cast<size_t>(r)],
                static_cast<size_t>(sz));
            owned_pkts.insert(
                owned_pkts.end(),
                pkts.begin(),
                pkts.end());
        }
    }
    std::vector<GO>{}.swap(source_selection.local_source_gids);
    decltype(gid_to_rank){}.swap(gid_to_rank);
    unique_packets_by_gid(owned_pkts);

    // All source cells are now represented by packets on their destination
    // ranks. Release the global legacy arrays before ghost responses create
    // additional packet copies and serialized buffers.
    release_rebuildable_storage(mesh);

    std::vector<char> ghost_packet_storage;
    std::vector<PacketView> ghost_pkts;
    {
        std::vector<std::vector<GO>> greq(nranks);
        {
            // Determine ghosts, build their requests, then release the hash
            // sets before packet response serialization.
            std::unordered_set<GO> owned_set;
            for (const auto& packet : owned_pkts)
            {
                owned_set.insert(packet.gid());
            }
            std::unordered_map<GO, int> ghost_ranks;
            for (const auto& packet : owned_pkts)
            {
                packet.for_each_face(
                    [&](const typename PacketView::Face& face)
                    {
                        const auto neighbor_gid =
                            face.neighbor_gid;
                        if (neighbor_gid != invalid_id<GO>()
                            && owned_set.find(neighbor_gid)
                                == owned_set.end())
                        {
                            const auto neighbor_rank =
                                face.neighbor_rank;
                            if (neighbor_rank < 0
                                || neighbor_rank >= nranks)
                            {
                                throw std::runtime_error(
                                    "MeshPartitioner cell packet has an "
                                    "invalid neighboring rank.");
                            }
                            const auto [entry, inserted] =
                                ghost_ranks.emplace(
                                    neighbor_gid, neighbor_rank);
                            if (!inserted
                                && entry->second != neighbor_rank)
                            {
                                throw std::runtime_error(
                                    "MeshPartitioner received inconsistent "
                                    "owner ranks for a ghost cell.");
                            }
                        }
                    });
            }
            for (const auto& [gid, rank] : ghost_ranks)
            {
                greq[static_cast<size_t>(rank)]
                    .push_back(gid);
            }
        }

        // Request ghosts.
        std::vector<int> req_c(nranks, 0), resp_c(nranks, 0);
        for (int r = 0; r < nranks; ++r) req_c[static_cast<size_t>(r)] = static_cast<int>(greq[static_cast<size_t>(r)].size());
        MPI_Alltoall(req_c.data(), 1, MPI_INT, resp_c.data(), 1, MPI_INT, get_mpi_comm(*comm));
        std::vector<int> req_d(nranks, 0), resp_d(nranks, 0);
        for (int r = 1; r < nranks; ++r) {
            req_d[static_cast<size_t>(r)] = req_d[static_cast<size_t>(r - 1)] + req_c[static_cast<size_t>(r - 1)];
            resp_d[static_cast<size_t>(r)] = resp_d[static_cast<size_t>(r - 1)] + resp_c[static_cast<size_t>(r - 1)];
        }
        const auto total_requests = req_d.back() + req_c.back();
        std::vector<GO> flat_rq(
            static_cast<size_t>(std::max(total_requests, 1)));
        for (int r = 0; r < nranks; ++r)
        {
            auto& requests = greq[static_cast<size_t>(r)];
            std::copy(
                requests.begin(),
                requests.end(),
                flat_rq.begin() + req_d[static_cast<size_t>(r)]);
            std::vector<GO>{}.swap(requests);
        }
        std::vector<std::vector<GO>>{}.swap(greq);
        std::vector<GO> flat_rs(static_cast<size_t>(std::max(resp_d.back() + resp_c.back(), 1)));
        MPI_Alltoallv(flat_rq.data(), req_c.data(), req_d.data(), mpi_go_type(),
                      flat_rs.data(), resp_c.data(), resp_d.data(), mpi_go_type(),
                      get_mpi_comm(*comm));
        std::vector<GO>{}.swap(flat_rq);

        std::unordered_map<GO, const PacketView*> owned_lookup;
        owned_lookup.reserve(owned_pkts.size());
        for (const auto& packet : owned_pkts)
        {
            owned_lookup[packet.gid()] = &packet;
        }

        std::vector<std::uint32_t> response_packet_counts(
            static_cast<size_t>(nranks), 0);
        std::vector<size_t> response_body_bytes(
            static_cast<size_t>(nranks), 0);
        for (int rank = 0; rank < nranks; ++rank)
        {
            const auto rank_index = static_cast<size_t>(rank);
            const auto count = resp_c[rank_index];
            const GO* requested =
                flat_rs.data() + resp_d[rank_index];
            for (int request = 0; request < count; ++request)
            {
                const auto packet =
                    owned_lookup.find(requested[request]);
                if (packet == owned_lookup.end())
                {
                    throw std::runtime_error(
                        "MeshPartitioner received a ghost request for "
                        "a cell not owned on this rank.");
                }
                ++response_packet_counts[rank_index];
                response_body_bytes[rank_index] +=
                    packet->second->serialized_size();
            }
        }

        std::vector<int> gr_scnt(nranks, 0);
        for (int rank = 0; rank < nranks; ++rank)
        {
            const auto rank_index = static_cast<size_t>(rank);
            if (response_packet_counts[rank_index] == 0)
            {
                continue;
            }
            const auto bytes =
                sizeof(std::uint32_t)
              + response_body_bytes[rank_index];
            if (bytes > static_cast<size_t>(
                    std::numeric_limits<int>::max()))
            {
                throw std::overflow_error(
                    "MeshPartitioner ghost send buffer exceeds MPI "
                    "int count capacity.");
            }
            gr_scnt[rank_index] = static_cast<int>(bytes);
        }

        std::vector<int> gr_rcnt(nranks, 0);
        MPI_Alltoall(gr_scnt.data(), 1, MPI_INT, gr_rcnt.data(), 1, MPI_INT, get_mpi_comm(*comm));
        std::vector<int> gr_sd(nranks, 0), gr_rd(nranks, 0);
        for (int r = 1; r < nranks; ++r) {
            gr_sd[static_cast<size_t>(r)] = gr_sd[static_cast<size_t>(r - 1)] + gr_scnt[static_cast<size_t>(r - 1)];
            gr_rd[static_cast<size_t>(r)] = gr_rd[static_cast<size_t>(r - 1)] + gr_rcnt[static_cast<size_t>(r - 1)];
        }

        const auto ghost_total_send = gr_sd.back() + gr_scnt.back();
        std::vector<char> gr_flat_s(
            static_cast<size_t>(std::max(ghost_total_send, 1)));
        std::vector<size_t> response_write_offsets(
            static_cast<size_t>(nranks), 0);
        for (int rank = 0; rank < nranks; ++rank)
        {
            const auto rank_index = static_cast<size_t>(rank);
            if (response_packet_counts[rank_index] == 0)
            {
                continue;
            }
            const auto offset =
                static_cast<size_t>(gr_sd[rank_index]);
            std::memcpy(
                gr_flat_s.data() + offset,
                &response_packet_counts[rank_index],
                sizeof(std::uint32_t));
            response_write_offsets[rank_index] =
                offset + sizeof(std::uint32_t);
        }
        for (int rank = 0; rank < nranks; ++rank)
        {
            const auto rank_index = static_cast<size_t>(rank);
            const auto count = resp_c[rank_index];
            const GO* requested =
                flat_rs.data() + resp_d[rank_index];
            for (int request = 0; request < count; ++request)
            {
                const auto& packet =
                    *owned_lookup.at(requested[request]);
                const auto bytes =
                    packet.serialized_size();
                std::memcpy(
                    gr_flat_s.data()
                        + response_write_offsets[rank_index],
                    packet.data(),
                    bytes);
                response_write_offsets[rank_index] += bytes;
            }
        }
        std::vector<GO>{}.swap(flat_rs);
        decltype(owned_lookup){}.swap(owned_lookup);

        ghost_packet_storage.resize(
            static_cast<size_t>(
                std::max(gr_rd.back() + gr_rcnt.back(), 1)));
        MPI_Alltoallv(
            gr_flat_s.data(), gr_scnt.data(), gr_sd.data(), MPI_CHAR,
            ghost_packet_storage.data(),
            gr_rcnt.data(),
            gr_rd.data(),
            MPI_CHAR,
            get_mpi_comm(*comm));
        std::vector<char>{}.swap(gr_flat_s);

        for (int r = 0; r < nranks; ++r) {
            int sz = gr_rcnt[static_cast<size_t>(r)];
            if (sz <= 0) continue;
            auto pkts = Packet::view_packets(
                ghost_packet_storage.data()
                    + gr_rd[static_cast<size_t>(r)],
                static_cast<size_t>(sz));
            ghost_pkts.insert(
                ghost_pkts.end(),
                pkts.begin(),
                pkts.end());
        }
    }

    // Remove ghost packets whose GID is already owned (can happen
    // when the partition assigns adjacent cells that share a face to
    // the same rank, but the ghost detection still flags them).
    {
        std::unordered_set<GO> owned_gids;
        for (const auto& packet : owned_pkts)
        {
            owned_gids.insert(packet.gid());
        }
        std::unordered_set<GO> ghost_gids;
        ghost_pkts.erase(
            std::remove_if(ghost_pkts.begin(), ghost_pkts.end(),
                [&](const PacketView& packet) {
                    const auto gid = packet.gid();
                    return owned_gids.count(gid) > 0
                        || !ghost_gids.insert(gid).second;
                }),
            ghost_pkts.end());
    }

    // Periodic boundary faces retain a valid neighbor but use distinct node
    // keys on the paired sides. Recreate the face-key mapping from packets so
    // ranks that never held the source geometry can restore those neighbors.
    std::unordered_map<std::string, GO> periodic_pairs;
    auto collect_periodic_pairs =
        [&](const std::vector<PacketView>& packets)
        {
            for (const auto& packet : packets)
            {
                packet.for_each_face(
                    [&](const typename PacketView::Face& face)
                    {
                        if (face.boundary_id
                                == Mesh<Pack>::invalid_boundary_id
                            || face.neighbor_gid == invalid_id<GO>())
                        {
                            return;
                        }
                        periodic_pairs[mesh.make_face_key(
                            typename Mesh<Pack>::ViewGO(
                                const_cast<GO*>(
                                    face.node_gids.data()),
                                face.node_count))] =
                            face.neighbor_gid;
                    });
            }
        };
    collect_periodic_pairs(owned_pkts);
    collect_periodic_pairs(ghost_pkts);

    rebuild(mesh, owned_pkts, ghost_pkts, periodic_pairs);
    return true;
}

/**
 * @brief Partition a replicated CRTP unstructured mesh.
 *
 * Cell geometry local IDs are used as graph global IDs while the input mesh is
 * replicated. The input object is replaced by compact rank-local geometry,
 * while the returned indexer retains those original IDs.
 *
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 * @param mesh Replicated unstructured mesh geometry.
 * @param comm Teuchos MPI communicator.
 * @return Global indexing and ownership metadata for the rebuilt mesh.
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::partition(
    Meshes::UnstructuredMesh& mesh,
    const Teuchos::RCP<const comm_type>& comm) -> UnstructuredPartition
{
    const auto nranks = comm->getSize();
    const auto myrank = comm->getRank();
    if (nranks <= 1)
    {
        std::vector<int> owner_ranks(mesh.num_cells(), 0);
        return rebuild(mesh, std::move(owner_ranks), myrank);
    }

    return rebuild(
        mesh,
        compute_unstructured_owner_ranks(mesh, comm),
        myrank);
}

/**
 * @brief Build the distributed cell-adjacency graph for a legacy mesh.
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 * @param mesh Mesh whose owned-cell adjacency is distributed into graph rows.
 * @param comm Communicator defining graph row ownership.
 * @return Local graph rows plus their local and remote column IDs.
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::build_partition_graph(
    const Mesh<Pack>& mesh,
    const SourceSelection& source_selection,
    const Teuchos::RCP<const comm_type>& comm) -> PartitionGraph
{
    const auto nranks = comm->getSize();
    const auto myrank = comm->getRank();
    const auto cell_count = mesh.d_owned_cell_global_ids.size();

    // ParMETIS expects graph rows to be distributed. The selected source
    // rank derives each row from its local geometry, then sends only that
    // compact adjacency row to the historical modulo row owner. This keeps
    // full geometry on rank zero while avoiding an all-rows-on-root graph.
    std::vector<std::vector<GO>> rows_by_rank(
        static_cast<size_t>(nranks));
    for (size_t i = 0; i < cell_count; ++i)
    {
        const auto gid = mesh.d_owned_cell_global_ids[i];
        if (!source_selection.provides(gid, myrank))
        {
            continue;
        }

        const auto lid = mesh.d_owned_cell_ids[i];
        const auto& cell = mesh.d_cells[static_cast<size_t>(lid)];
        std::unordered_set<GO> neighbors;
        for (const auto face_lid : cell.faces)
        {
            const auto& face = mesh.d_faces[static_cast<size_t>(face_lid)];
            if (face.owner == lid && face.neighbor != invalid_lid)
            {
                neighbors.insert(mesh.cell_global_id(face.neighbor));
            }
            else if (face.neighbor == lid)
            {
                neighbors.insert(mesh.cell_global_id(face.owner));
            }
        }

        std::vector<GO> adjacency(
            neighbors.begin(), neighbors.end());
        std::sort(adjacency.begin(), adjacency.end());

        const int row_rank = static_cast<int>(
            static_cast<size_t>(gid)
            % static_cast<size_t>(nranks));
        auto& row_buffer =
            rows_by_rank[static_cast<size_t>(row_rank)];
        row_buffer.push_back(gid);
        row_buffer.push_back(
            static_cast<GO>(adjacency.size()));
        row_buffer.insert(
            row_buffer.end(),
            adjacency.begin(),
            adjacency.end());
    }

    std::vector<int> send_counts(
        static_cast<size_t>(nranks), 0);
    std::vector<int> receive_counts(
        static_cast<size_t>(nranks), 0);
    for (int rank = 0; rank < nranks; ++rank)
    {
        send_counts[static_cast<size_t>(rank)] =
            static_cast<int>(
                rows_by_rank[static_cast<size_t>(rank)].size());
    }
    MPI_Alltoall(
        send_counts.data(),
        1,
        MPI_INT,
        receive_counts.data(),
        1,
        MPI_INT,
        get_mpi_comm(*comm));

    std::vector<int> send_displacements(
        static_cast<size_t>(nranks), 0);
    std::vector<int> receive_displacements(
        static_cast<size_t>(nranks), 0);
    for (int rank = 1; rank < nranks; ++rank)
    {
        send_displacements[static_cast<size_t>(rank)] =
            send_displacements[static_cast<size_t>(rank - 1)]
          + send_counts[static_cast<size_t>(rank - 1)];
        receive_displacements[static_cast<size_t>(rank)] =
            receive_displacements[static_cast<size_t>(rank - 1)]
          + receive_counts[static_cast<size_t>(rank - 1)];
    }

    const int total_send =
        send_displacements.back() + send_counts.back();
    const int total_receive =
        receive_displacements.back() + receive_counts.back();
    std::vector<GO> flat_send(
        static_cast<size_t>(std::max(total_send, 1)));
    size_t send_offset = 0;
    for (const auto& rank_rows : rows_by_rank)
    {
        std::copy(
            rank_rows.begin(),
            rank_rows.end(),
            flat_send.begin()
                + static_cast<std::ptrdiff_t>(send_offset));
        send_offset += rank_rows.size();
    }
    std::vector<GO> flat_receive(
        static_cast<size_t>(std::max(total_receive, 1)));
    MPI_Alltoallv(
        flat_send.data(),
        send_counts.data(),
        send_displacements.data(),
        mpi_go_type(),
        flat_receive.data(),
        receive_counts.data(),
        receive_displacements.data(),
        mpi_go_type(),
        get_mpi_comm(*comm));

    std::vector<std::pair<GO, std::vector<GO>>> local_rows;
    std::unordered_set<GO> column_gids;
    for (int source_rank = 0;
         source_rank < nranks;
         ++source_rank)
    {
        size_t offset = static_cast<size_t>(
            receive_displacements[
                static_cast<size_t>(source_rank)]);
        const size_t end = offset + static_cast<size_t>(
            receive_counts[static_cast<size_t>(source_rank)]);
        while (offset < end)
        {
            if (end - offset < 2)
            {
                throw std::runtime_error(
                    "MeshPartitioner received a truncated graph row.");
            }
            const auto gid = flat_receive[offset++];
            const auto raw_neighbor_count =
                flat_receive[offset++];
            if constexpr (std::is_signed_v<GO>)
            {
                if (raw_neighbor_count < 0)
                {
                    throw std::runtime_error(
                        "MeshPartitioner received an invalid graph row size.");
                }
            }
            const size_t neighbor_count =
                static_cast<size_t>(raw_neighbor_count);
            if (neighbor_count > end - offset)
            {
                throw std::runtime_error(
                    "MeshPartitioner received a truncated graph adjacency.");
            }

            std::vector<GO> adjacency(
                flat_receive.begin()
                    + static_cast<std::ptrdiff_t>(offset),
                flat_receive.begin()
                    + static_cast<std::ptrdiff_t>(
                        offset + neighbor_count));
            offset += neighbor_count;
            local_rows.emplace_back(gid, std::move(adjacency));
        }
    }

    std::sort(
        local_rows.begin(),
        local_rows.end(),
        [](const auto& left, const auto& right)
        {
            return left.first < right.first;
        });

    PartitionGraph graph;
    graph.row_gids.reserve(local_rows.size());
    graph.row_adjacency.reserve(local_rows.size());
    for (auto& [gid, adjacency] : local_rows)
    {
        graph.row_gids.push_back(gid);
        column_gids.insert(gid);
        for (const auto neighbor_gid : adjacency)
        {
            column_gids.insert(neighbor_gid);
        }
        graph.row_adjacency.push_back(std::move(adjacency));
    }
    graph.column_gids.assign(column_gids.begin(), column_gids.end());
    std::sort(graph.column_gids.begin(), graph.column_gids.end());
    return graph;
}

/**
 * @brief Build the distributed cell-adjacency graph for CRTP geometry.
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 * @param mesh Replicated unstructured geometry.
 * @param comm Communicator defining graph row ownership.
 * @return Local graph rows plus the replicated column ID set.
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::build_partition_graph(
    const Meshes::UnstructuredMesh& mesh,
    const Teuchos::RCP<const comm_type>& comm) -> PartitionGraph
{
    const auto nranks = comm->getSize();
    const auto myrank = comm->getRank();
    PartitionGraph graph;

    graph.column_gids.reserve(mesh.num_cells());
    for (size_t lid = 0; lid < mesh.num_cells(); ++lid)
    {
        graph.column_gids.push_back(static_cast<GO>(lid));
    }

    for (size_t cell_lid = 0; cell_lid < mesh.num_cells(); ++cell_lid)
    {
        if (static_cast<int>(cell_lid % static_cast<size_t>(nranks))
            != myrank)
        {
            continue;
        }

        graph.row_gids.push_back(static_cast<GO>(cell_lid));
        const auto cell = mesh.cell_id(cell_lid);
        std::unordered_set<GO> neighbors;
        for (const auto face : mesh.faces(cell))
        {
            const auto other = mesh.opposite_cell(face, cell);
            if (other == Meshes::UnstructuredMesh::invalid_cell_id())
            {
                continue;
            }
            neighbors.insert(static_cast<GO>(
                mesh.cell_local_id(other)));
        }

        auto& adjacency =
            graph.row_adjacency.emplace_back(
                neighbors.begin(),
                neighbors.end());
        std::sort(adjacency.begin(), adjacency.end());
    }
    return graph;
}

/**
 * @brief Partition a distributed adjacency graph with Zoltan2/ParMETIS.
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 * @param partition_graph Local graph rows and referenced columns.
 * @param comm Communicator used by Tpetra, Zoltan2, and the result gather.
 * @param gather_root Rank that receives the complete result, or -1 to
 *        replicate it on all ranks.
 * @return Global map from cell ID to destination rank on the requested
 *         recipients.
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::solve_partition_graph(
    const PartitionGraph& partition_graph,
    const Teuchos::RCP<const comm_type>& comm,
    int gather_root)
    -> std::unordered_map<GO, int>
{
    const auto nranks = comm->getSize();
    const auto myrank = comm->getRank();
    if (gather_root >= nranks)
    {
        throw std::invalid_argument(
            "MeshPartitioner partition-result root is invalid.");
    }
    const auto invalid_size =
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

    Teuchos::RCP<const map_type> row_map;
    if (!partition_graph.row_gids.empty())
    {
        row_map = Teuchos::rcp(new map_type(
            invalid_size,
            partition_graph.row_gids.data(),
            partition_graph.row_gids.size(),
            GO{},
            comm));
    }
    else
    {
        row_map = Teuchos::rcp(new map_type(
            invalid_size,
            size_t{},
            GO{},
            comm));
    }

    Teuchos::RCP<const map_type> column_map;
    if (!partition_graph.column_gids.empty())
    {
        column_map = Teuchos::rcp(new map_type(
            invalid_size,
            partition_graph.column_gids.data(),
            partition_graph.column_gids.size(),
            GO{},
            comm));
    }
    else
    {
        column_map = Teuchos::rcp(new map_type(
            invalid_size,
            size_t{},
            GO{},
            comm));
    }

    size_t max_adjacency = 0;
    for (const auto& adjacency : partition_graph.row_adjacency)
    {
        max_adjacency = std::max(max_adjacency, adjacency.size());
    }

    auto graph = Teuchos::rcp(new graph_type(
        row_map,
        column_map,
        max_adjacency));
    for (size_t row = 0; row < partition_graph.row_gids.size(); ++row)
    {
        const auto& adjacency = partition_graph.row_adjacency[row];
        if (!adjacency.empty())
        {
            graph->insertGlobalIndices(
                partition_graph.row_gids[row],
                adjacency.size(),
                adjacency.data());
        }
    }
    graph->fillComplete();

    std::unordered_map<GO, int> local_parts;
    {
        using adapter_type = Zoltan2::TpetraRowGraphAdapter<graph_type>;
        Teuchos::ParameterList parameters;
        parameters.set("algorithm", "parmetis");
        parameters.set("num_global_parts", nranks);
        adapter_type adapter(graph);
        Zoltan2::PartitioningProblem<adapter_type> problem(
            &adapter,
            &parameters,
            comm);
        problem.solve();

        const auto& solution = problem.getSolution();
        const auto* parts = solution.getPartListView();
        for (size_t row = 0; row < partition_graph.row_gids.size(); ++row)
        {
            local_parts[partition_graph.row_gids[row]] = parts[row];
        }
    }

    std::vector<GO> local_pairs;
    local_pairs.reserve(local_parts.size() * 2);
    for (const auto& [cell_gid, owner_rank] : local_parts)
    {
        local_pairs.push_back(cell_gid);
        local_pairs.push_back(static_cast<GO>(owner_rank));
    }

    const auto my_pair_count =
        static_cast<int>(local_pairs.size());
    std::vector<int> counts(static_cast<size_t>(nranks), 0);
    if (gather_root >= 0)
    {
        MPI_Gather(
            &my_pair_count,
            1,
            MPI_INT,
            counts.data(),
            1,
            MPI_INT,
            gather_root,
            get_mpi_comm(*comm));
    }
    else
    {
        MPI_Allgather(
            &my_pair_count,
            1,
            MPI_INT,
            counts.data(),
            1,
            MPI_INT,
            get_mpi_comm(*comm));
    }

    std::vector<int> displacements(static_cast<size_t>(nranks), 0);
    if (gather_root < 0 || myrank == gather_root)
    {
        for (int r = 1; r < nranks; ++r)
        {
            displacements[static_cast<size_t>(r)] =
                displacements[static_cast<size_t>(r - 1)]
              + counts[static_cast<size_t>(r - 1)];
        }
    }

    const auto total_pairs =
        displacements.back() + counts.back();
    std::vector<GO> gathered_pairs(
        static_cast<size_t>(
            std::max(
                gather_root < 0 || myrank == gather_root
                    ? total_pairs
                    : 0,
                1)));
    if (gather_root >= 0)
    {
        MPI_Gatherv(
            local_pairs.data(),
            my_pair_count,
            mpi_go_type(),
            gathered_pairs.data(),
            counts.data(),
            displacements.data(),
            mpi_go_type(),
            gather_root,
            get_mpi_comm(*comm));
        if (myrank != gather_root)
        {
            return {};
        }
    }
    else
    {
        MPI_Allgatherv(
            local_pairs.data(),
            my_pair_count,
            mpi_go_type(),
            gathered_pairs.data(),
            counts.data(),
            displacements.data(),
            mpi_go_type(),
            get_mpi_comm(*comm));
    }

    local_parts.clear();
    for (int pair = 0; pair + 1 < total_pairs; pair += 2)
    {
        local_parts[gathered_pairs[static_cast<size_t>(pair)]] =
            static_cast<int>(
                gathered_pairs[static_cast<size_t>(pair + 1)]);
    }
    return local_parts;
}

/**
 * @brief Compute owned-first local cell, face, and node ordering.
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 * @tparam CellFaces Callable returning faces of a cell local ID.
 * @tparam OppositeCell Callable returning the opposite cell across a face.
 * @tparam FaceNodes Callable returning nodes of a face local ID.
 * @tparam CellNodes Callable returning nodes of a cell local ID.
 * @param cell_count Number of source cells.
 * @param cell_owner_ranks Destination rank for every source cell.
 * @param rank Rank whose owned and overlap entities are selected.
 * @param cell_faces Cell-to-face connectivity callback.
 * @param opposite_cell Face-and-cell to opposite-cell callback.
 * @param face_nodes Face-to-node connectivity callback.
 * @param cell_nodes Cell-to-node connectivity callback.
 * @return Owned-first entity ordering for the rebuilt local mesh.
 * @throws std::invalid_argument If owner-rank metadata has the wrong size.
 */
template<TpetraTypePack Pack>
template<class CellFaces, class OppositeCell, class FaceNodes, class CellNodes>
auto MeshPartitioner<Pack>::reorder_local_entities(
    size_t cell_count,
    const std::vector<int>& cell_owner_ranks,
    int rank,
    CellFaces&& cell_faces,
    OppositeCell&& opposite_cell,
    FaceNodes&& face_nodes,
    CellNodes&& cell_nodes) -> LocalEntityOrder
{
    if (cell_owner_ranks.size() != cell_count)
    {
        throw std::invalid_argument(
            "Mesh partition has wrong cell count.");
    }

    LocalEntityOrder order;
    order.owned_cells.reserve(cell_count);
    std::vector<bool> ghost_cell_flags(cell_count, false);
    for (size_t cell_lid = 0; cell_lid < cell_count; ++cell_lid)
    {
        if (cell_owner_ranks[cell_lid] != rank)
        {
            continue;
        }

        order.owned_cells.push_back(cell_lid);
        for (const auto face : cell_faces(cell_lid))
        {
            const auto neighbor =
                static_cast<size_t>(opposite_cell(face, cell_lid));
            if (neighbor >= cell_count)
            {
                continue;
            }
            if (cell_owner_ranks[neighbor] != rank)
            {
                ghost_cell_flags[neighbor] = true;
            }
        }
    }

    for (size_t cell_lid = 0; cell_lid < ghost_cell_flags.size(); ++cell_lid)
    {
        if (ghost_cell_flags[cell_lid])
        {
            order.ghost_cells.push_back(cell_lid);
        }
    }

    std::unordered_set<size_t> seen_faces;
    auto append_cell_faces =
        [&](const std::vector<size_t>& cells, std::vector<size_t>& faces)
    {
        for (const auto cell_lid : cells)
        {
            for (const auto face : cell_faces(cell_lid))
            {
                const auto face_lid = static_cast<size_t>(face);
                if (seen_faces.insert(face_lid).second)
                {
                    faces.push_back(face_lid);
                }
            }
        }
    };
    append_cell_faces(order.owned_cells, order.owned_faces);
    append_cell_faces(order.ghost_cells, order.overlap_faces);

    std::unordered_map<size_t, int> node_owner_ranks;
    for (size_t cell_lid = 0; cell_lid < cell_count; ++cell_lid)
    {
        for (const auto node : cell_nodes(cell_lid))
        {
            const auto node_lid = static_cast<size_t>(node);
            const auto [owner, inserted] = node_owner_ranks.emplace(
                node_lid, cell_owner_ranks[cell_lid]);
            if (!inserted)
            {
                owner->second = std::min(
                    owner->second, cell_owner_ranks[cell_lid]);
            }
        }
    }

    std::unordered_set<size_t> seen_nodes;
    std::vector<size_t> local_nodes;
    auto append_nodes = [&](const auto& ids)
    {
        for (const auto node : ids)
        {
            const auto node_lid = static_cast<size_t>(node);
            if (seen_nodes.insert(node_lid).second)
            {
                local_nodes.push_back(node_lid);
            }
        }
    };
    for (const auto cell_lid : order.owned_cells)
    {
        append_nodes(cell_nodes(cell_lid));
    }
    for (const auto cell_lid : order.ghost_cells)
    {
        append_nodes(cell_nodes(cell_lid));
    }
    for (const auto face_lid : order.owned_faces)
    {
        append_nodes(face_nodes(face_lid));
    }
    for (const auto face_lid : order.overlap_faces)
    {
        append_nodes(face_nodes(face_lid));
    }

    for (const auto node_lid : local_nodes)
    {
        if (node_owner_ranks.at(node_lid) == rank)
        {
            order.owned_nodes.push_back(node_lid);
        }
        else
        {
            order.overlap_nodes.push_back(node_lid);
        }
    }

    return order;
}

/**
 * @brief Compute the destination rank of every replicated CRTP mesh cell.
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 * @param mesh Replicated unstructured geometry.
 * @param comm Communicator participating in partitioning.
 * @return Owner rank indexed by source cell local ID.
 * @throws std::runtime_error If the partition result is incomplete or invalid.
 */
template<TpetraTypePack Pack>
std::vector<int> MeshPartitioner<Pack>::compute_unstructured_owner_ranks(
    const Meshes::UnstructuredMesh& mesh,
    const Teuchos::RCP<const comm_type>& comm)
{
    const auto nranks = comm->getSize();
    if (nranks <= 1)
    {
        return std::vector<int>(mesh.num_cells(), 0);
    }

    const auto rank_by_gid =
        solve_partition_graph(build_partition_graph(mesh, comm), comm);
    std::vector<int> owner_ranks(mesh.num_cells(), -1);
    for (const auto& [cell_gid, owner_rank] : rank_by_gid)
    {
        const auto cell_lid = static_cast<size_t>(cell_gid);
        if (cell_lid >= owner_ranks.size())
        {
            throw std::runtime_error(
                "Unstructured mesh partitioner returned an invalid "
                "cell ID.");
        }
        owner_ranks[cell_lid] = owner_rank;
    }

    if (std::find(owner_ranks.begin(), owner_ranks.end(), -1)
        != owner_ranks.end())
    {
        throw std::runtime_error(
            "Unstructured mesh partitioner did not assign every cell.");
    }
    return owner_ranks;
}

/**
 * @brief Rebuild replicated CRTP geometry into one rank's owned and ghost mesh.
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 * @param mesh Replicated mesh replaced in place with rank-local geometry.
 * @param cell_owner_ranks Destination rank for every source cell.
 * @param rank Rank whose local mesh is built.
 * @return Local/global indexer and source-cell owner ranks.
 * @throws std::invalid_argument If owner-rank metadata has the wrong size.
 * @throws std::runtime_error If rebuilt face identity or ownership is inconsistent.
 * @throws std::overflow_error If a source entity ID exceeds global ordinals.
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::rebuild(
    Meshes::UnstructuredMesh& mesh,
    std::vector<int> cell_owner_ranks,
    int rank) -> UnstructuredPartition
{
    auto order = reorder_local_entities(
        mesh.num_cells(),
        cell_owner_ranks,
        rank,
        [&](size_t cell_lid)
        {
            return mesh.faces(mesh.cell_id(cell_lid));
        },
        [&](const auto face, size_t cell_lid) -> size_t
        {
            const auto neighbor =
                mesh.opposite_cell(face, mesh.cell_id(cell_lid));
            if (neighbor == Meshes::UnstructuredMesh::invalid_cell_id())
            {
                return mesh.num_cells();
            }
            return mesh.cell_local_id(neighbor);
        },
        [&](size_t face_lid)
        {
            return mesh.face_nodes(mesh.face_id(face_lid));
        },
        [&](size_t cell_lid)
        {
            return mesh.cell_nodes(mesh.cell_id(cell_lid));
        });

    auto to_cell_ids = [&](std::vector<size_t> local_ids)
    {
        std::vector<typename indexer_type::cell_id_t> global_ids;
        global_ids.reserve(local_ids.size());
        for (const auto local_id : local_ids)
        {
            global_ids.push_back(mesh.cell_id(local_id));
        }
        return global_ids;
    };
    auto to_face_ids = [&](const std::vector<size_t>& local_ids)
    {
        std::vector<typename indexer_type::face_id_t> global_ids;
        global_ids.reserve(local_ids.size());
        for (const auto local_id : local_ids)
        {
            global_ids.push_back(mesh.face_id(local_id));
        }
        return global_ids;
    };
    auto to_node_ids = [&](const std::vector<size_t>& local_ids)
    {
        std::vector<typename indexer_type::node_id_t> global_ids;
        global_ids.reserve(local_ids.size());
        for (const auto local_id : local_ids)
        {
            global_ids.push_back(mesh.node_id(local_id));
        }
        return global_ids;
    };

    const auto owned_cell_ids = to_cell_ids(order.owned_cells);
    const auto ghost_cell_ids = to_cell_ids(order.ghost_cells);
    const auto owned_node_ids = to_node_ids(order.owned_nodes);
    const auto overlap_node_ids = to_node_ids(order.overlap_nodes);
    std::vector<typename indexer_type::node_id_t> node_global_ids =
        owned_node_ids;
    node_global_ids.insert(
        node_global_ids.end(),
        overlap_node_ids.begin(),
        overlap_node_ids.end());

    std::vector<size_t> source_cell_lids = order.owned_cells;
    source_cell_lids.insert(
        source_cell_lids.end(),
        order.ghost_cells.begin(),
        order.ghost_cells.end());
    std::vector<size_t> source_face_lids = order.owned_faces;
    source_face_lids.insert(
        source_face_lids.end(),
        order.overlap_faces.begin(),
        order.overlap_faces.end());

    using UnstructuredMesh = Meshes::UnstructuredMesh;
    using NodeID = UnstructuredMesh::NodeID;
    using FaceID = UnstructuredMesh::FaceID;
    using FaceKey = std::vector<NodeID>;

    std::unordered_map<NodeID, NodeID> local_node_by_global;
    Arr<Vec3> local_nodes;
    local_nodes.reserve(node_global_ids.size());
    for (size_t local = 0; local < node_global_ids.size(); ++local)
    {
        const auto global_id = node_global_ids[local];
        local_node_by_global.emplace(
            global_id, static_cast<NodeID>(local));
        local_nodes.push_back(mesh.node_coordinates(global_id));
    }

    Arr<UnstructuredMesh::CellDefinition> local_cells;
    local_cells.reserve(source_cell_lids.size());
    for (const auto source_lid : source_cell_lids)
    {
        const auto source_id = mesh.cell_id(source_lid);
        UnstructuredMesh::CellDefinition cell;
        cell.type = mesh.cell_type(source_id);
        cell.node_ids.reserve(mesh.cell_nodes(source_id).size());
        for (const auto node_id : mesh.cell_nodes(source_id))
        {
            cell.node_ids.push_back(local_node_by_global.at(node_id));
        }
        local_cells.push_back(std::move(cell));
    }

    Arr<UnstructuredMesh::BoundaryFaceDefinition> local_boundaries;
    for (const auto source_lid : source_face_lids)
    {
        const auto source_id = mesh.face_id(source_lid);
        if (!mesh.is_boundary_face(source_id))
        {
            continue;
        }

        UnstructuredMesh::BoundaryFaceDefinition boundary;
        boundary.boundary_id = mesh.boundary_id(source_id);
        boundary.name = mesh.boundary_name(source_id);
        boundary.node_ids.reserve(mesh.face_nodes(source_id).size());
        for (const auto node_id : mesh.face_nodes(source_id))
        {
            boundary.node_ids.push_back(local_node_by_global.at(node_id));
        }
        local_boundaries.push_back(std::move(boundary));
    }

    UnstructuredMesh local_mesh(
        local_nodes,
        local_cells,
        local_boundaries,
        order.owned_cells.size(),
        order.owned_faces.size());

    auto source_face_key = [&](FaceID source_id)
    {
        FaceKey key = mesh.face_nodes(source_id);
        std::sort(key.begin(), key.end());
        return key;
    };
    std::map<FaceKey, FaceID> source_face_by_key;
    for (const auto source_lid : source_face_lids)
    {
        const auto source_id = mesh.face_id(source_lid);
        source_face_by_key.emplace(source_face_key(source_id), source_id);
    }

    std::vector<typename indexer_type::face_id_t> face_global_ids;
    face_global_ids.reserve(local_mesh.num_faces());
    for (size_t local_lid = 0;
         local_lid < local_mesh.num_faces();
         ++local_lid)
    {
        FaceKey key;
        for (const auto local_node_id :
             local_mesh.face_nodes(local_mesh.face_id(local_lid)))
        {
            key.push_back(node_global_ids.at(
                static_cast<size_t>(local_node_id)));
        }
        std::sort(key.begin(), key.end());
        const auto source_face = source_face_by_key.find(key);
        if (source_face == source_face_by_key.end())
        {
            throw std::runtime_error(
                "MeshPartitioner rebuilt an unknown local face.");
        }
        face_global_ids.push_back(source_face->second);
    }

    if (face_global_ids.size() != source_face_lids.size())
    {
        throw std::runtime_error(
            "MeshPartitioner rebuilt the wrong number of local faces.");
    }

    const auto owned_source_face_ids = to_face_ids(order.owned_faces);
    const std::unordered_set<FaceID> owned_face_ids(
        owned_source_face_ids.begin(),
        owned_source_face_ids.end());
    for (size_t local_lid = 0; local_lid < face_global_ids.size(); ++local_lid)
    {
        const bool expected_owned = local_lid < order.owned_faces.size();
        if (owned_face_ids.contains(face_global_ids[local_lid])
            != expected_owned)
        {
            throw std::runtime_error(
                "MeshPartitioner local face ownership is not contiguous.");
        }
    }

    const auto owned_face_end = face_global_ids.begin()
        + static_cast<std::ptrdiff_t>(order.owned_faces.size());
    std::vector<typename indexer_type::face_id_t> owned_face_global_ids(
        face_global_ids.begin(), owned_face_end);
    std::vector<typename indexer_type::face_id_t> overlap_face_global_ids(
        owned_face_end, face_global_ids.end());

    auto checked_global_ordinal = [](const auto global_id) -> GO
    {
        if (!std::in_range<GO>(global_id))
        {
            throw std::overflow_error(
                "MeshPartitioner global entity ID exceeds the global "
                "ordinal type.");
        }
        return static_cast<GO>(global_id);
    };

    std::vector<typename indexer_type::CellMapping> owned_cell_mappings;
    std::vector<typename indexer_type::CellMapping> ghost_cell_mappings;
    owned_cell_mappings.reserve(owned_cell_ids.size());
    ghost_cell_mappings.reserve(ghost_cell_ids.size());
    for (size_t local = 0; local < owned_cell_ids.size(); ++local)
    {
        const auto global_id = owned_cell_ids[local];
        owned_cell_mappings.push_back({
            local_mesh.cell_id(local),
            global_id,
            checked_global_ordinal(global_id)});
    }
    for (size_t ghost = 0; ghost < ghost_cell_ids.size(); ++ghost)
    {
        const auto local = owned_cell_ids.size() + ghost;
        const auto global_id = ghost_cell_ids[ghost];
        ghost_cell_mappings.push_back({
            local_mesh.cell_id(local),
            global_id,
            checked_global_ordinal(global_id)});
    }

    std::vector<typename indexer_type::FaceMapping> owned_face_mappings;
    std::vector<typename indexer_type::FaceMapping> overlap_face_mappings;
    owned_face_mappings.reserve(owned_face_global_ids.size());
    overlap_face_mappings.reserve(overlap_face_global_ids.size());
    for (size_t local = 0; local < owned_face_global_ids.size(); ++local)
    {
        const auto global_id = owned_face_global_ids[local];
        owned_face_mappings.push_back({
            local_mesh.face_id(local),
            global_id,
            checked_global_ordinal(global_id)});
    }
    for (size_t overlap = 0;
         overlap < overlap_face_global_ids.size();
         ++overlap)
    {
        const auto local = owned_face_global_ids.size() + overlap;
        const auto global_id = overlap_face_global_ids[overlap];
        overlap_face_mappings.push_back({
            local_mesh.face_id(local),
            global_id,
            checked_global_ordinal(global_id)});
    }

    std::vector<typename indexer_type::NodeMapping> owned_node_mappings;
    std::vector<typename indexer_type::NodeMapping> overlap_node_mappings;
    owned_node_mappings.reserve(owned_node_ids.size());
    overlap_node_mappings.reserve(overlap_node_ids.size());
    for (size_t local = 0; local < owned_node_ids.size(); ++local)
    {
        const auto global_id = owned_node_ids[local];
        owned_node_mappings.push_back({
            local_mesh.node_id(local),
            global_id,
            checked_global_ordinal(global_id)});
    }
    for (size_t overlap = 0; overlap < overlap_node_ids.size(); ++overlap)
    {
        const auto local = owned_node_ids.size() + overlap;
        const auto global_id = overlap_node_ids[overlap];
        overlap_node_mappings.push_back({
            local_mesh.node_id(local),
            global_id,
            checked_global_ordinal(global_id)});
    }

    UnstructuredPartition partition;
    partition.indexer = indexer_type(
        std::move(owned_cell_mappings),
        std::move(ghost_cell_mappings),
        std::move(owned_face_mappings),
        std::move(overlap_face_mappings),
        std::move(owned_node_mappings),
        std::move(overlap_node_mappings));
    partition.cell_owner_ranks = std::move(cell_owner_ranks);
    mesh = std::move(local_mesh);

    return partition;
}

/**
 * @brief Compute the mapping from cell global ID to destination MPI rank.
 *
 * Builds a Tpetra CRS graph from face-based cell adjacency, wraps it in
 * a Zoltan2 row-graph adapter, and solves the partitioning problem with
 * ParMETIS. The complete partition result is gathered only to rank zero,
 * which returns each source rank's own and adjacent assignments.
 *
 * @param mesh The source mesh whose owned cells are to be mapped.
 * @param source_selection Cells this rank is responsible for providing.
 * @param comm Teuchos MPI communicator.
 * @return Assignments required to serialize source cells on this rank.
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::compute_gid_to_rank_map(
    const Mesh<Pack>& mesh,
    const SourceSelection& source_selection,
    const Teuchos::RCP<const comm_type>& comm)
    -> std::unordered_map<GO, int>
{
    constexpr int coordinator = 0;
    const auto nranks = comm->getSize();
    const auto myrank = comm->getRank();
    auto full_partition = solve_partition_graph(
        build_partition_graph(mesh, source_selection, comm),
        comm,
        coordinator);

    // The production path has all source geometry on rank zero. It can retain
    // the root-only result directly; every other rank stores no global
    // partition metadata.
    if (source_selection.is_single_source()
        && source_selection.single_source_rank == coordinator)
    {
        return full_partition;
    }

    // A source rank needs destinations only for cells it serializes and for
    // their direct neighbors, whose ranks are embedded in outgoing packets.
    std::vector<GO> needed_gids;
    for (size_t index = 0;
         index < mesh.d_owned_cell_global_ids.size();
         ++index)
    {
        const auto gid = mesh.d_owned_cell_global_ids[index];
        if (!source_selection.provides(gid, myrank))
        {
            continue;
        }
        needed_gids.push_back(gid);
        const auto lid = mesh.d_owned_cell_ids[index];
        const auto& cell = mesh.d_cells[static_cast<size_t>(lid)];
        for (const auto face_lid : cell.faces)
        {
            const auto& face =
                mesh.d_faces[static_cast<size_t>(face_lid)];
            if (face.owner == lid && face.neighbor != invalid_lid)
            {
                needed_gids.push_back(
                    mesh.cell_global_id(face.neighbor));
            }
            else if (face.neighbor == lid)
            {
                needed_gids.push_back(
                    mesh.cell_global_id(face.owner));
            }
        }
    }
    std::sort(needed_gids.begin(), needed_gids.end());
    needed_gids.erase(
        std::unique(needed_gids.begin(), needed_gids.end()),
        needed_gids.end());
    if (needed_gids.size()
        > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            "MeshPartitioner local partition request exceeds MPI int "
            "count capacity.");
    }

    const int needed_count = static_cast<int>(needed_gids.size());
    std::vector<int> request_counts(
        static_cast<size_t>(nranks), 0);
    MPI_Gather(
        &needed_count,
        1,
        MPI_INT,
        request_counts.data(),
        1,
        MPI_INT,
        coordinator,
        get_mpi_comm(*comm));

    std::vector<int> request_displacements(
        static_cast<size_t>(nranks), 0);
    if (myrank == coordinator)
    {
        for (int rank = 1; rank < nranks; ++rank)
        {
            request_displacements[static_cast<size_t>(rank)] =
                request_displacements[static_cast<size_t>(rank - 1)]
              + request_counts[static_cast<size_t>(rank - 1)];
        }
    }
    const auto total_requests =
        request_displacements.back() + request_counts.back();
    std::vector<GO> gathered_requests(
        static_cast<size_t>(
            std::max(myrank == coordinator ? total_requests : 0, 1)));
    MPI_Gatherv(
        needed_gids.data(),
        needed_count,
        mpi_go_type(),
        gathered_requests.data(),
        request_counts.data(),
        request_displacements.data(),
        mpi_go_type(),
        coordinator,
        get_mpi_comm(*comm));
    std::vector<GO>{}.swap(needed_gids);

    std::vector<int> response_counts(
        static_cast<size_t>(nranks), 0);
    std::vector<int> response_displacements(
        static_cast<size_t>(nranks), 0);
    std::vector<GO> response_pairs;
    int missing_assignment = 0;
    if (myrank == coordinator)
    {
        response_pairs.reserve(
            static_cast<size_t>(total_requests) * 2);
        for (int rank = 0; rank < nranks; ++rank)
        {
            const auto rank_index = static_cast<size_t>(rank);
            response_displacements[rank_index] =
                static_cast<int>(response_pairs.size());
            const auto begin = request_displacements[rank_index];
            const auto end = begin + request_counts[rank_index];
            for (int request = begin; request < end; ++request)
            {
                const auto gid =
                    gathered_requests[static_cast<size_t>(request)];
                const auto assignment = full_partition.find(gid);
                if (assignment == full_partition.end())
                {
                    missing_assignment = 1;
                    continue;
                }
                response_pairs.push_back(gid);
                response_pairs.push_back(
                    static_cast<GO>(assignment->second));
            }
            response_counts[rank_index] =
                static_cast<int>(response_pairs.size())
              - response_displacements[rank_index];
        }
    }
    MPI_Bcast(
        &missing_assignment,
        1,
        MPI_INT,
        coordinator,
        get_mpi_comm(*comm));
    if (missing_assignment != 0)
    {
        throw std::runtime_error(
            "MeshPartitioner did not assign every source neighbor.");
    }
    decltype(full_partition){}.swap(full_partition);

    int local_pair_count = 0;
    MPI_Scatter(
        response_counts.data(),
        1,
        MPI_INT,
        &local_pair_count,
        1,
        MPI_INT,
        coordinator,
        get_mpi_comm(*comm));
    std::vector<GO> local_pairs(
        static_cast<size_t>(std::max(local_pair_count, 1)));
    MPI_Scatterv(
        response_pairs.data(),
        response_counts.data(),
        response_displacements.data(),
        mpi_go_type(),
        local_pairs.data(),
        local_pair_count,
        mpi_go_type(),
        coordinator,
        get_mpi_comm(*comm));

    std::unordered_map<GO, int> local_partition;
    local_partition.reserve(
        static_cast<size_t>(local_pair_count / 2));
    for (int pair = 0; pair + 1 < local_pair_count; pair += 2)
    {
        local_partition.emplace(
            local_pairs[static_cast<size_t>(pair)],
            static_cast<int>(
                local_pairs[static_cast<size_t>(pair + 1)]));
    }
    return local_partition;
}

/**
 * @brief Select one rank to provide each legacy cell to the partitioner.
 *
 * A single source is represented by one rank number. Fully replicated input
 * selects the modulo row owner, while general sparse input selects an actual
 * holder. Only rank zero temporarily sees the complete selection; selected
 * GIDs are scattered back to their source ranks.
 *
 * @param mesh Mesh containing zero or more source cells on this rank.
 * @param comm Communicator participating in partitioning.
 * @return Source cells represented locally or by one source rank.
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::compute_source_selection(
    const Mesh<Pack>& mesh,
    const Teuchos::RCP<const comm_type>& comm)
    -> SourceSelection
{
    constexpr int coordinator = 0;
    const int nranks = comm->getSize();
    const int myrank = comm->getRank();
    if (mesh.d_owned_cell_global_ids.size()
        > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            "MeshPartitioner local source exceeds MPI int count "
            "capacity.");
    }
    const int my_count =
        static_cast<int>(mesh.d_owned_cell_global_ids.size());
    std::vector<int> counts(static_cast<size_t>(nranks), 0);
    MPI_Allgather(
        &my_count,
        1,
        MPI_INT,
        counts.data(),
        1,
        MPI_INT,
        get_mpi_comm(*comm));

    int source_rank_count = 0;
    int single_source_rank = -1;
    for (int rank = 0; rank < nranks; ++rank)
    {
        if (counts[static_cast<size_t>(rank)] > 0)
        {
            ++source_rank_count;
            single_source_rank = rank;
        }
    }
    if (source_rank_count == 0)
    {
        throw std::runtime_error(
            "MeshPartitioner requires at least one source cell.");
    }
    if (source_rank_count == 1)
    {
        SourceSelection selection;
        selection.single_source_rank = single_source_rank;
        return selection;
    }

    std::vector<int> displacements(static_cast<size_t>(nranks), 0);
    for (int rank = 1; rank < nranks; ++rank)
    {
        displacements[static_cast<size_t>(rank)] =
            displacements[static_cast<size_t>(rank - 1)]
          + counts[static_cast<size_t>(rank - 1)];
    }

    const int total_count =
        displacements.back() + counts.back();
    std::vector<GO> gathered_gids(
        static_cast<size_t>(
            std::max(myrank == coordinator ? total_count : 0, 1)));
    MPI_Gatherv(
        mesh.d_owned_cell_global_ids.data(),
        my_count,
        mpi_go_type(),
        gathered_gids.data(),
        counts.data(),
        displacements.data(),
        mpi_go_type(),
        coordinator,
        get_mpi_comm(*comm));

    std::vector<int> selected_counts(
        static_cast<size_t>(nranks), 0);
    std::vector<int> selected_displacements(
        static_cast<size_t>(nranks), 0);
    std::vector<GO> selected_flat;
    if (myrank == coordinator)
    {
        std::unordered_map<GO, int> source_rank_for_gid;
        source_rank_for_gid.reserve(static_cast<size_t>(total_count));
        for (int rank = 0; rank < nranks; ++rank)
        {
            const int begin = displacements[static_cast<size_t>(rank)];
            const int end = begin + counts[static_cast<size_t>(rank)];
            for (int index = begin; index < end; ++index)
            {
                source_rank_for_gid.emplace(
                    gathered_gids[static_cast<size_t>(index)],
                    rank);
            }
        }

        // Prefer the modulo row owner when that rank holds the cell. Sparse
        // cells remain on their lowest actual holder.
        for (int rank = 0; rank < nranks; ++rank)
        {
            const int begin = displacements[static_cast<size_t>(rank)];
            const int end = begin + counts[static_cast<size_t>(rank)];
            for (int index = begin; index < end; ++index)
            {
                const auto gid =
                    gathered_gids[static_cast<size_t>(index)];
                const int preferred_rank = static_cast<int>(
                    static_cast<size_t>(gid)
                    % static_cast<size_t>(nranks));
                if (rank == preferred_rank)
                {
                    source_rank_for_gid[gid] = rank;
                }
            }
        }

        std::vector<std::vector<GO>> selected_by_rank(
            static_cast<size_t>(nranks));
        for (const auto& [gid, rank] : source_rank_for_gid)
        {
            selected_by_rank[static_cast<size_t>(rank)]
                .push_back(gid);
        }
        for (int rank = 0; rank < nranks; ++rank)
        {
            auto& gids =
                selected_by_rank[static_cast<size_t>(rank)];
            std::sort(gids.begin(), gids.end());
            selected_counts[static_cast<size_t>(rank)] =
                static_cast<int>(gids.size());
            if (rank > 0)
            {
                selected_displacements[static_cast<size_t>(rank)] =
                    selected_displacements[
                        static_cast<size_t>(rank - 1)]
                  + selected_counts[
                        static_cast<size_t>(rank - 1)];
            }
            selected_flat.insert(
                selected_flat.end(), gids.begin(), gids.end());
        }
    }

    int local_selected_count = 0;
    MPI_Scatter(
        selected_counts.data(),
        1,
        MPI_INT,
        &local_selected_count,
        1,
        MPI_INT,
        coordinator,
        get_mpi_comm(*comm));
    SourceSelection selection;
    selection.local_source_gids.resize(
        static_cast<size_t>(local_selected_count));
    MPI_Scatterv(
        selected_flat.data(),
        selected_counts.data(),
        selected_displacements.data(),
        mpi_go_type(),
        selection.local_source_gids.data(),
        local_selected_count,
        mpi_go_type(),
        coordinator,
        get_mpi_comm(*comm));
    return selection;
}


/**
 * @brief Release all legacy storage replaced by a local packet rebuild.
 */
template<TpetraTypePack Pack>
void MeshPartitioner<Pack>::release_rebuildable_storage(
    Mesh<Pack>& mesh)
{
    auto release_container = []<class Container>(Container& container)
    {
        Container{}.swap(container);
    };

    // Root may still own allocations sized for the complete source mesh.
    // Replace every rebuildable container rather than clear()ing it so local
    // reserve() calls cannot retain global capacities or hash buckets.
    release_container(mesh.d_cells);
    release_container(mesh.d_faces);
    release_container(mesh.d_owned_cell_ids);
    release_container(mesh.d_owned_cell_global_ids);
    release_container(mesh.d_ghost_cell_global_ids);
    release_container(mesh.d_owned_face_global_ids);
    release_container(mesh.d_cell_gid_to_lid);
    release_container(mesh.d_node_gid_to_lid);
    release_container(mesh.d_node_coords);
    release_container(mesh.d_cell_owned_face_ids);
    release_container(mesh.d_cell_face_distances);
    release_container(mesh.d_cell_owned_node_global_ids);
    release_container(mesh.d_face_owned_node_global_ids);
    release_container(mesh.d_face_key_to_face);
    release_container(mesh.d_boundary_id_to_face_batch);
    release_container(mesh.d_ghost_cell_tpetra_gids);
    release_container(mesh.d_mesh_gid_to_tpetra_gid);
    release_container(mesh.d_tpetra_gid_to_mesh_gid);
    mesh.d_host_views = typename Mesh<Pack>::HostViews{};
    mesh.reset_device_views();
    mesh.d_owned_cell_map = Teuchos::null;
    mesh.d_overlap_cell_map = Teuchos::null;
    mesh.d_owned_face_map = Teuchos::null;
    mesh.d_boundary_face_map = Teuchos::null;

    // Clear contiguous Tpetra GID assignments; they will be recomputed by create_maps().
    mesh.reset_contiguous_tpetra_gids();
}

/**
 * @brief Rebuild the mesh data structures from owned and ghost cell packets.
 *
 * Clears and repopulates d_cells, d_faces, node tables, face lookups,
 * cell-face adjacency, and face geometry (centroids, normals, areas,
 * distances).  Face deduplication uses sorted-node-GID keys.
 *
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 * @param mesh Mesh instance to rebuild (modified in place).
 * @param owned_pkts Cell packets destined for this rank.
 * @param ghost_pkts Cell packets describing off-rank neighbours.
 * @param periodic_pairs Face-key → paired-GID map for periodic boundary faces.
 */
template<TpetraTypePack Pack>
void MeshPartitioner<Pack>::rebuild(
    Mesh<Pack>& mesh,
    const std::vector<PacketView>& owned_pkts,
    const std::vector<PacketView>& ghost_pkts,
    const std::unordered_map<std::string, GO>& periodic_pairs)
{
    // Idempotent for the normal path, which releases the source immediately
    // after the owned-packet exchange.
    release_rebuildable_storage(mesh);

    const auto total_cells = owned_pkts.size() + ghost_pkts.size();
    size_t total_cell_nodes = 0;
    size_t maximum_face_nodes = 0;
    for (const auto* packets : {&owned_pkts, &ghost_pkts})
    {
        for (const auto& packet : *packets)
        {
            total_cell_nodes += packet.header().node_count;
            packet.for_each_face(
                [&](const typename PacketView::Face& face)
                {
                    maximum_face_nodes += face.node_count;
                });
        }
    }
    mesh.d_cells.reserve(total_cells);
    mesh.d_owned_cell_ids.reserve(owned_pkts.size());
    mesh.d_owned_cell_global_ids.reserve(owned_pkts.size());
    mesh.d_ghost_cell_global_ids.reserve(ghost_pkts.size());
    mesh.d_cell_owned_node_global_ids.reserve(total_cell_nodes);
    mesh.d_face_owned_node_global_ids.reserve(maximum_face_nodes);

    std::unordered_map<GO, Vec3> needed_nodes;
    for (const auto* packets : {&owned_pkts, &ghost_pkts})
    {
        for (const auto& packet : *packets)
        {
            packet.for_each_node(
                [&](const typename PacketView::Node& node)
                {
                    needed_nodes.emplace(node.gid, node.coordinate);
                });
            packet.for_each_face(
                [&](const typename PacketView::Face& face)
                {
                    for (std::uint32_t node = 0;
                         node < face.node_count;
                         ++node)
                    {
                        if (!needed_nodes.contains(
                                face.node_gids[node]))
                        {
                            throw std::runtime_error(
                                "MeshPartitioner cell packet is missing "
                                "a face-node coordinate.");
                        }
                    }
                });
        }
    }
    mesh.d_node_coords.reserve(needed_nodes.size());
    for (const auto& [node_gid, coordinate] : needed_nodes) {
        mesh.d_node_gid_to_lid[node_gid] =
            static_cast<LO>(mesh.d_node_coords.size());
        mesh.d_node_coords.push_back(coordinate);
    }

    auto add_cell =
        [&](const PacketView& packet, bool owned) -> LO
    {
        const auto header = packet.header();
        LO lid = static_cast<LO>(mesh.d_cells.size());
        mesh.d_cell_gid_to_lid[header.gid] = lid;
        CellInfo ci;
        ci.owned = owned;
        ci.type = header.cell_type;
        ci.center = header.center;
        ci.volume = header.volume;
        const size_t offset =
            mesh.d_cell_owned_node_global_ids.size();
        packet.for_each_node(
            [&](const typename PacketView::Node& node)
            {
                mesh.d_cell_owned_node_global_ids.push_back(node.gid);
            });
        ci.node_gids = typename Mesh<Pack>::ViewGO(
            mesh.d_cell_owned_node_global_ids.data() + offset,
            header.node_count);
        ci.faces = typename Mesh<Pack>::ViewLO(nullptr, 0);
        ci.face_distances = typename Mesh<Pack>::ViewReal(nullptr, 0);
        mesh.d_cells.push_back(std::move(ci));
        return lid;
    };
    for (const auto& packet : owned_pkts)
    {
        const auto lid = add_cell(packet, true);
        mesh.d_owned_cell_ids.push_back(lid);
        mesh.d_owned_cell_global_ids.push_back(packet.gid());
    }
    for (const auto& packet : ghost_pkts)
    {
        add_cell(packet, false);
        mesh.d_ghost_cell_global_ids.push_back(packet.gid());
    }

    // Build faces
    std::vector<std::vector<LO>> cfl(mesh.d_cells.size());
    auto add_packet_faces =
        [&](const std::vector<PacketView>& packets,
            size_t cell_offset)
        {
            for (size_t packet_index = 0;
                 packet_index < packets.size();
                 ++packet_index)
            {
                const auto& packet = packets[packet_index];
                const auto cl = static_cast<LO>(
                    cell_offset + packet_index);
                packet.for_each_face(
                    [&](const typename PacketView::Face& face)
                    {
                        const auto key = Mesh<Pack>::make_face_key(
                            typename Mesh<Pack>::ViewGO(
                                const_cast<GO*>(
                                    face.node_gids.data()),
                                face.node_count));
                        const auto existing =
                            mesh.d_face_key_to_face.find(key);
                        if (existing == mesh.d_face_key_to_face.end())
                        {
                            const auto fid =
                                static_cast<LO>(mesh.d_faces.size());
                            const auto node_offset =
                                mesh.d_face_owned_node_global_ids.size();
                            mesh.d_face_owned_node_global_ids.insert(
                                mesh.d_face_owned_node_global_ids.end(),
                                face.node_gids.begin(),
                                face.node_gids.begin()
                                    + face.node_count);
                            FaceInfo face_info;
                            face_info.type =
                                face.node_count == 3
                                    ? MeshUtils::FaceType::TRIANGLE
                                    : MeshUtils::FaceType::QUAD;
                            face_info.boundary_id = face.boundary_id;
                            face_info.owner = cl;
                            face_info.neighbor = invalid_lid;
                            face_info.node_gids =
                                typename Mesh<Pack>::ViewGO(
                                    mesh.d_face_owned_node_global_ids.data()
                                        + node_offset,
                                    face.node_count);
                            mesh.d_faces.push_back(std::move(face_info));
                            mesh.d_owned_face_global_ids.push_back(
                                face.global_id);
                            mesh.d_face_key_to_face.emplace(key, fid);
                            cfl[static_cast<size_t>(cl)].push_back(fid);
                            return;
                        }

                        const auto fid = existing->second;
                        if (mesh.d_owned_face_global_ids[
                                static_cast<size_t>(fid)]
                            != face.global_id)
                        {
                            throw std::runtime_error(
                                "MeshPartitioner received inconsistent global "
                                "IDs for the same face.");
                        }
                        auto& face_info =
                            mesh.d_faces[static_cast<size_t>(fid)];
                        if (face_info.boundary_id
                                == Mesh<Pack>::invalid_boundary_id
                            && face.boundary_id
                                != Mesh<Pack>::invalid_boundary_id)
                        {
                            face_info.boundary_id =
                                face.boundary_id;
                        }
                        if (face_info.neighbor == invalid_lid)
                        {
                            face_info.neighbor = cl;
                            cfl[static_cast<size_t>(cl)]
                                .push_back(fid);
                        }
                        else if (face_info.owner == cl
                                 || face_info.neighbor == cl)
                        {
                            cfl[static_cast<size_t>(cl)]
                                .push_back(fid);
                        }
                        // A third cell referencing the same face is
                        // non-manifold and is intentionally omitted,
                        // preserving prior behavior.
                    });
            }
        };
    add_packet_faces(owned_pkts, 0);
    add_packet_faces(ghost_pkts, owned_pkts.size());

    // Compute face geometry
    for (auto& fi : mesh.d_faces) {
        std::vector<Vec3> fcs; fcs.reserve(fi.node_gids.size());
        for (auto ngo : fi.node_gids) {
            auto nit = mesh.d_node_gid_to_lid.find(ngo);
            if (nit != mesh.d_node_gid_to_lid.end()) fcs.push_back(mesh.d_node_coords[static_cast<size_t>(nit->second)]);
        }
        if (fcs.size() < 3) continue;
        fi.center = MeshUtils::average(fcs);
        auto av = MeshUtils::face_area_vector(fcs);
        fi.area = av.norm();
        if (fi.area <= 0.0) continue;
        auto nml = av / fi.area;
        auto& oc = mesh.d_cells[static_cast<size_t>(fi.owner)];
        auto ov = fi.center - oc.center;
        if (nml.dot(ov) < 0.0) nml = nml * -1.0;
        fi.unit_normal_from_owner = nml; fi.unit_normal_from_neighbor = nml * -1.0;
        fi.owner_to_face_distance = ov.norm();
        if (fi.neighbor != invalid_lid) {
            auto& nc = mesh.d_cells[static_cast<size_t>(fi.neighbor)];
            auto nv = fi.center - nc.center;
            fi.neighbor_to_face_distance = nv.norm();
            fi.cell_center_distance = (nc.center - oc.center).norm();
        }
    }

    // Set cell face views
    size_t tcf = 0; for (auto& cf : cfl) tcf += cf.size();

    // Restore periodic boundary face pairs from the original mesh.
    for (auto& [key, pgid] : periodic_pairs)
    {
        auto it = mesh.d_face_key_to_face.find(key);
        if (it == mesh.d_face_key_to_face.end()) continue;
        auto& fi = mesh.d_faces[static_cast<size_t>(it->second)];

        // Recompute cell-centre distance for the periodic pair if the
        // paired cell is available locally.
        auto plid = mesh.d_cell_gid_to_lid.find(pgid);
        if (plid != mesh.d_cell_gid_to_lid.end())
        {
            fi.neighbor = plid->second;
            auto& oc = mesh.d_cells[static_cast<size_t>(fi.owner)];
            auto& pc = mesh.d_cells[static_cast<size_t>(plid->second)];
            fi.cell_center_distance = (pc.center - oc.center).norm();
        }
    }

    // DEBUG: verify GID consistency
    for (size_t i = 0; i < mesh.d_owned_cell_global_ids.size(); ++i) {
        GO g = mesh.d_owned_cell_global_ids[i];
        LO l = static_cast<LO>(i);
        if (mesh.d_cell_gid_to_lid.at(g) != l) {
            std::cerr << "Rank " << Tpetra::getDefaultComm()->getRank()
                        << " MISMATCH: d_owned_cell_global_ids[" << i << "]=" << g
                        << " maps to LID " << mesh.d_cell_gid_to_lid.at(g)
                        << " but expected " << l << std::endl;
        }
    }
    mesh.d_cell_owned_face_ids.reserve(tcf);
    for (size_t ci = 0; ci < mesh.d_cells.size(); ++ci) {
        size_t off = mesh.d_cell_owned_face_ids.size();
        auto& cf = cfl[ci];
        mesh.d_cell_owned_face_ids.insert(mesh.d_cell_owned_face_ids.end(), cf.begin(), cf.end());
        mesh.d_cells[ci].faces = typename Mesh<Pack>::ViewLO(mesh.d_cell_owned_face_ids.data() + off, cf.size());
    }

    // Face matching is complete; retaining string keys would keep one
    // allocation and one hash node per local face for the solver lifetime.
    decltype(mesh.d_face_key_to_face){}.swap(
        mesh.d_face_key_to_face);
}
