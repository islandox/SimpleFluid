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

#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_TpetraRowGraphAdapter.hpp>

using namespace SimpleFluid;


/**
 * @brief Main entry point — partition a replicated mesh across MPI ranks.
 *
 * Five-phase algorithm:
 * 1. **Zoltan2/ParMETIS partition** — build a CRS graph from cell
 *    adjacency and call Zoltan2 to assign each cell a destination rank.
 * 2. **Cell redistribution** — serialise owned cells into packets,
 *    exchange them via MPI_Alltoallv.
 * 3. **Ghost detection** — for each owned cell, identify adjacent
 *    off-rank cells that are not already owned.
 * 4. **Ghost exchange** — request ghost packets from the ranks that
 *    own them, receive and deserialise.
 * 5. **Mesh rebuild** — reconstruct the per-rank d_cells, d_faces,
 *    node tables, and face geometry from owned + ghost packets.
 *
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 * @param mesh The replicated mesh to partition (modified in place).
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
    if (!mesh.d_ghost_cell_global_ids.empty()) return false;

    auto orig_node_coords = mesh.d_node_coords;
    auto orig_node_gid_to_lid = mesh.d_node_gid_to_lid;
    auto gid_to_rank = compute_gid_to_rank_map(mesh, comm);

    auto unique_packets_by_gid = [](std::vector<Packet>& pkts)
    {
        std::unordered_set<GO> seen;
        pkts.erase(
            std::remove_if(pkts.begin(), pkts.end(),
                [&](const Packet& p) { return !seen.insert(p.gid).second; }),
            pkts.end());
    };

    std::unordered_map<GO, int> source_rank_for_gid;
    {
        const int my_count = static_cast<int>(mesh.d_owned_cell_global_ids.size());
        std::vector<int> counts(static_cast<size_t>(nranks), 0);
        MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      get_mpi_comm(*comm));

        std::vector<int> displs(static_cast<size_t>(nranks), 0);
        for (int r = 1; r < nranks; ++r) {
            displs[static_cast<size_t>(r)] =
                displs[static_cast<size_t>(r - 1)]
                + counts[static_cast<size_t>(r - 1)];
        }

        const int total_count = displs.back() + counts.back();
        std::vector<GO> gathered_gids(static_cast<size_t>(std::max(total_count, 1)));
        MPI_Allgatherv(mesh.d_owned_cell_global_ids.data(), my_count, mpi_go_type(),
                       gathered_gids.data(), counts.data(), displs.data(), mpi_go_type(),
                       get_mpi_comm(*comm));

        for (int r = 0; r < nranks; ++r) {
            const int begin = displs[static_cast<size_t>(r)];
            const int end = begin + counts[static_cast<size_t>(r)];
            for (int i = begin; i < end; ++i) {
                auto [it, inserted] = source_rank_for_gid.emplace(gathered_gids[static_cast<size_t>(i)], r);
                if (!inserted) it->second = std::min(it->second, r);
            }
        }

        for (const auto& [gid, dest] : gid_to_rank) {
            (void)dest;
            if (source_rank_for_gid.find(gid) == source_rank_for_gid.end()) {
                throw std::runtime_error("MeshPartitioner missing source rank for cell GID "
                                       + std::to_string(gid));
            }
        }
    }

    std::vector<std::vector<Packet>> send_p(nranks);
    std::unordered_set<GO> packed_gids;
    for (size_t i = 0; i < mesh.d_owned_cell_global_ids.size(); ++i) {
        GO gid = mesh.d_owned_cell_global_ids[i];
        if (source_rank_for_gid.at(gid) != myrank || !packed_gids.insert(gid).second) continue;

        LO lid = mesh.d_owned_cell_ids[i];
        auto& cell = mesh.d_cells[static_cast<size_t>(lid)];
        int dest = gid_to_rank.at(gid);
        Packet p;
        p.gid = gid; p.cell_type = cell.type; p.center = cell.center; p.volume = cell.volume;
        p.node_gids.assign(cell.node_gids.begin(), cell.node_gids.end());
        p.node_coords.reserve(p.node_gids.size());
        for (const auto node_gid : p.node_gids) {
            p.node_coords.push_back(mesh.node_coord(node_gid));
        }
        for (auto fid : cell.faces) {
            auto& face = mesh.d_faces[static_cast<size_t>(fid)];
            std::vector<GO> fn(face.node_gids.begin(), face.node_gids.end());
            p.face_node_keys.push_back(std::move(fn));
            p.face_global_ids.push_back(mesh.face_global_id(fid));
            p.face_boundary_ids.push_back(face.boundary_id);
        }
        send_p[static_cast<size_t>(dest)].push_back(std::move(p));
    }

    // Serialize and exchange
    std::vector<std::vector<char>> sbufs(nranks);
    std::vector<int> scnt(nranks, 0);
    for (int r = 0; r < nranks; ++r) {
        if (send_p[static_cast<size_t>(r)].empty()) continue;
        sbufs[static_cast<size_t>(r)] = Packet::serialize_packets(send_p[static_cast<size_t>(r)]);
        scnt[static_cast<size_t>(r)] = static_cast<int>(sbufs[static_cast<size_t>(r)].size());
    }

    std::vector<int> rcnt(nranks, 0);
    MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, get_mpi_comm(*comm));
    std::vector<int> sd(nranks, 0), rd(nranks, 0);
    for (int r = 1; r < nranks; ++r) {
        sd[static_cast<size_t>(r)] = sd[static_cast<size_t>(r - 1)] + scnt[static_cast<size_t>(r - 1)];
        rd[static_cast<size_t>(r)] = rd[static_cast<size_t>(r - 1)] + rcnt[static_cast<size_t>(r - 1)];
    }
    std::vector<char> flat_s;
    { size_t off = 0; for (int r = 0; r < nranks; ++r) {
        auto& b = sbufs[static_cast<size_t>(r)]; if (b.empty()) continue;
        flat_s.resize(off + b.size()); std::memcpy(flat_s.data() + off, b.data(), b.size()); off += b.size();
    } if (flat_s.empty()) flat_s.resize(1); }
    std::vector<char> rbuf(static_cast<size_t>(std::max(rd.back() + rcnt.back(), 1)));
    MPI_Alltoallv(flat_s.data(), scnt.data(), sd.data(), MPI_CHAR,
                    rbuf.data(), rcnt.data(), rd.data(), MPI_CHAR, get_mpi_comm(*comm));

    std::vector<Packet> owned_pkts;
    { size_t off = 0; for (int r = 0; r < nranks; ++r) {
        int sz = rcnt[static_cast<size_t>(r)]; if (sz <= 0) continue;
        auto pkts = Packet::deserialize_packets(rbuf.data() + off, static_cast<size_t>(sz));
        owned_pkts.insert(owned_pkts.end(), std::make_move_iterator(pkts.begin()), std::make_move_iterator(pkts.end()));
        off += static_cast<size_t>(sz);
    }}
    unique_packets_by_gid(owned_pkts);

    // Determine ghosts
    std::unordered_set<GO> owned_set;
    for (auto& p : owned_pkts) owned_set.insert(p.gid);
    std::unordered_map<std::string, std::pair<GO, GO>> fk2cells;
    for (size_t fid = 0; fid < mesh.d_faces.size(); ++fid) {
        auto& face = mesh.d_faces[fid];
        if (face.neighbor == invalid_lid) continue;
        GO og = mesh.cell_global_id(face.owner);
        GO ng = mesh.cell_global_id(face.neighbor);
        std::vector<GO> fns(face.node_gids.begin(), face.node_gids.end());
        std::sort(fns.begin(), fns.end());
        fk2cells[mesh.make_face_key(typename Mesh<Pack>::ViewGO(const_cast<GO*>(fns.data()), fns.size()))] = {og, ng};
    }
    std::unordered_set<GO> ghost_set;
    for (auto& p : owned_pkts) {
        for (auto& fn : p.face_node_keys) {
            std::string key = mesh.make_face_key(typename Mesh<Pack>::ViewGO(const_cast<GO*>(fn.data()), fn.size()));
            auto it = fk2cells.find(key);
            if (it == fk2cells.end()) continue;
            auto [g1, g2] = it->second;
            GO other = (g1 != p.gid) ? g1 : g2;
            if (other != p.gid && owned_set.find(other) == owned_set.end()) ghost_set.insert(other);
        }
    }

    // Periodic faces are represented as boundary-marked faces with a valid
    // neighbour.  They still need a face-key -> paired-cell-GID map because
    // the paired cell does not share the same topological face nodes.
    std::unordered_map<std::string, GO> original_periodic_pairs;
    for (size_t fid = 0; fid < mesh.d_faces.size(); ++fid)
    {
        auto& face = mesh.d_faces[fid];
        if (face.boundary_id == Mesh<Pack>::invalid_boundary_id) continue;
        const auto pid = face.neighbor;
        if (pid == invalid_lid) continue;

        auto pgid = mesh.cell_global_id(pid);
        std::vector<GO> fns(face.node_gids.begin(), face.node_gids.end());
        std::sort(fns.begin(), fns.end());
        auto key = mesh.make_face_key(
            typename Mesh<Pack>::ViewGO(const_cast<GO*>(fns.data()), fns.size()));
        original_periodic_pairs[key] = pgid;

        if (owned_set.find(pgid) == owned_set.end())
        {
            ghost_set.insert(pgid);
        }
    }

    // Request ghosts
    std::vector<std::vector<GO>> greq(nranks);
    for (GO g : ghost_set) greq[static_cast<size_t>(gid_to_rank.at(g))].push_back(g);
    std::vector<int> req_c(nranks, 0), resp_c(nranks, 0);
    for (int r = 0; r < nranks; ++r) req_c[static_cast<size_t>(r)] = static_cast<int>(greq[static_cast<size_t>(r)].size());
    MPI_Alltoall(req_c.data(), 1, MPI_INT, resp_c.data(), 1, MPI_INT, get_mpi_comm(*comm));
    std::vector<int> req_d(nranks, 0), resp_d(nranks, 0);
    for (int r = 1; r < nranks; ++r) {
        req_d[static_cast<size_t>(r)] = req_d[static_cast<size_t>(r - 1)] + req_c[static_cast<size_t>(r - 1)];
        resp_d[static_cast<size_t>(r)] = resp_d[static_cast<size_t>(r - 1)] + resp_c[static_cast<size_t>(r - 1)];
    }
    std::vector<GO> flat_rq;
    for (int r = 0; r < nranks; ++r) { auto& gr = greq[static_cast<size_t>(r)]; flat_rq.insert(flat_rq.end(), gr.begin(), gr.end()); }
    if (flat_rq.empty()) flat_rq.resize(1);
    std::vector<GO> flat_rs(static_cast<size_t>(std::max(resp_d.back() + resp_c.back(), 1)));
    MPI_Alltoallv(flat_rq.data(), req_c.data(), req_d.data(), mpi_go_type(),
                    flat_rs.data(), resp_c.data(), resp_d.data(), mpi_go_type(), get_mpi_comm(*comm));

    std::unordered_map<GO, const Packet*> olu;
    for (auto& p : owned_pkts) olu[p.gid] = &p;
    std::vector<std::vector<Packet>> gresp(nranks);
    for (int r = 0; r < nranks; ++r) {
        int cnt = resp_c[static_cast<size_t>(r)]; if (cnt <= 0) continue;
        GO* rs = flat_rs.data() + resp_d[static_cast<size_t>(r)];
        for (int i = 0; i < cnt; ++i) { auto it = olu.find(rs[i]); if (it != olu.end()) gresp[static_cast<size_t>(r)].push_back(*it->second); }
    }

    std::vector<std::vector<char>> gr_sbufs(nranks);
    std::vector<int> gr_scnt(nranks, 0);
    for (int r = 0; r < nranks; ++r) {
        if (gresp[static_cast<size_t>(r)].empty()) continue;
        gr_sbufs[static_cast<size_t>(r)] = Packet::serialize_packets(gresp[static_cast<size_t>(r)]);
        gr_scnt[static_cast<size_t>(r)] = static_cast<int>(gr_sbufs[static_cast<size_t>(r)].size());
    }
    std::vector<int> gr_rcnt(nranks, 0);
    MPI_Alltoall(gr_scnt.data(), 1, MPI_INT, gr_rcnt.data(), 1, MPI_INT, get_mpi_comm(*comm));
    std::vector<int> gr_sd(nranks, 0), gr_rd(nranks, 0);
    for (int r = 1; r < nranks; ++r) {
        gr_sd[static_cast<size_t>(r)] = gr_sd[static_cast<size_t>(r - 1)] + gr_scnt[static_cast<size_t>(r - 1)];
        gr_rd[static_cast<size_t>(r)] = gr_rd[static_cast<size_t>(r - 1)] + gr_rcnt[static_cast<size_t>(r - 1)];
    }
    std::vector<char> gr_flat_s;
    { size_t off = 0; for (int r = 0; r < nranks; ++r) {
        auto& b = gr_sbufs[static_cast<size_t>(r)]; if (b.empty()) continue;
        gr_flat_s.resize(off + b.size()); std::memcpy(gr_flat_s.data() + off, b.data(), b.size()); off += b.size();
    } if (gr_flat_s.empty()) gr_flat_s.resize(1); }
    std::vector<char> gr_rbuf(static_cast<size_t>(std::max(gr_rd.back() + gr_rcnt.back(), 1)));
    MPI_Alltoallv(gr_flat_s.data(), gr_scnt.data(), gr_sd.data(), MPI_CHAR,
                    gr_rbuf.data(), gr_rcnt.data(), gr_rd.data(), MPI_CHAR, get_mpi_comm(*comm));

    std::vector<Packet> ghost_pkts;
    { size_t off = 0; for (int r = 0; r < nranks; ++r) {
        int sz = gr_rcnt[static_cast<size_t>(r)]; if (sz <= 0) continue;
        auto pkts = Packet::deserialize_packets(gr_rbuf.data() + off, static_cast<size_t>(sz));
        ghost_pkts.insert(ghost_pkts.end(), std::make_move_iterator(pkts.begin()), std::make_move_iterator(pkts.end()));
        off += static_cast<size_t>(sz);
    }}

    // Remove ghost packets whose GID is already owned (can happen
    // when the partition assigns adjacent cells that share a face to
    // the same rank, but the ghost detection still flags them).
    {
        std::unordered_set<GO> owned_gids;
        for (auto& p : owned_pkts) owned_gids.insert(p.gid);
        std::unordered_set<GO> ghost_gids;
        ghost_pkts.erase(
            std::remove_if(ghost_pkts.begin(), ghost_pkts.end(),
                [&](const Packet& p) {
                    return owned_gids.count(p.gid) > 0 || !ghost_gids.insert(p.gid).second;
                }),
            ghost_pkts.end());
    }

    rebuild(mesh, owned_pkts, ghost_pkts, orig_node_coords,
            orig_node_gid_to_lid, original_periodic_pairs);
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
 * @return Local graph rows plus the replicated column ID set.
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::build_partition_graph(
    const Mesh<Pack>& mesh,
    const Teuchos::RCP<const comm_type>& comm) -> PartitionGraph
{
    const auto nranks = comm->getSize();
    const auto myrank = comm->getRank();
    PartitionGraph graph;
    const auto cell_count = mesh.d_owned_cell_global_ids.size();

    std::unordered_map<GO, std::vector<GO>> adjacency_by_gid;
    graph.column_gids.reserve(cell_count);
    for (size_t i = 0; i < cell_count; ++i)
    {
        const auto gid = mesh.d_owned_cell_global_ids[i];
        const auto lid = mesh.d_owned_cell_ids[i];
        graph.column_gids.push_back(gid);

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

        auto& adjacency = adjacency_by_gid[gid];
        adjacency.assign(neighbors.begin(), neighbors.end());
        std::sort(adjacency.begin(), adjacency.end());
    }

    for (const auto gid : graph.column_gids)
    {
        if (static_cast<int>(
                static_cast<size_t>(gid) % static_cast<size_t>(nranks))
            != myrank)
        {
            continue;
        }
        graph.row_gids.push_back(gid);
        graph.row_adjacency.push_back(adjacency_by_gid.at(gid));
    }
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
 * @return Global map from cell ID to destination rank.
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::solve_partition_graph(
    const PartitionGraph& partition_graph,
    const Teuchos::RCP<const comm_type>& comm)
    -> std::unordered_map<GO, int>
{
    const auto nranks = comm->getSize();
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
    MPI_Allgather(&my_pair_count, 1, MPI_INT,
                  counts.data(), 1, MPI_INT,
                  get_mpi_comm(*comm));

    std::vector<int> displacements(static_cast<size_t>(nranks), 0);
    for (int r = 1; r < nranks; ++r)
    {
        displacements[static_cast<size_t>(r)] =
            displacements[static_cast<size_t>(r - 1)]
          + counts[static_cast<size_t>(r - 1)];
    }

    const auto total_pairs =
        displacements.back() + counts.back();
    std::vector<GO> gathered_pairs(
        static_cast<size_t>(std::max(total_pairs, 1)));
    MPI_Allgatherv(local_pairs.data(), my_pair_count, mpi_go_type(),
                   gathered_pairs.data(), counts.data(),
                   displacements.data(), mpi_go_type(),
                   get_mpi_comm(*comm));

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
 * ParMETIS.  Results are gathered across all ranks via MPI_Allgatherv.
 *
 * @param mesh The replicated mesh whose owned cells are to be mapped.
 * @param comm Teuchos MPI communicator.
 * @return Unordered map from cell GID to destination rank.
 * @tparam Pack Tpetra scalar, ordinal, graph, and communicator types.
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::compute_gid_to_rank_map(
    const Mesh<Pack>& mesh, const Teuchos::RCP<const comm_type>& comm) -> std::unordered_map<GO, int>
{
    return solve_partition_graph(build_partition_graph(mesh, comm), comm);
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
 * @param orig_coords Original node coordinates before partitioning.
 * @param orig_ng2l Original node GID-to-local-index map.
 * @param periodic_pairs Face-key → paired-GID map for periodic boundary faces.
 */
template<TpetraTypePack Pack>
void MeshPartitioner<Pack>::rebuild(Mesh<Pack>& mesh, const std::vector<Packet>& owned_pkts, const std::vector<Packet>& ghost_pkts,
                    const std::vector<Vec3>& orig_coords, const std::unordered_map<GO, LO>& orig_ng2l,
                    const std::unordered_map<std::string, GO>& periodic_pairs) {
    mesh.d_cells.clear(); mesh.d_faces.clear(); mesh.d_owned_cell_ids.clear();
    mesh.d_owned_cell_global_ids.clear(); mesh.d_ghost_cell_global_ids.clear();
    mesh.d_cell_gid_to_lid.clear(); mesh.d_node_gid_to_lid.clear(); mesh.d_node_coords.clear();
    mesh.d_cell_owned_face_ids.clear(); mesh.d_cell_face_distances.clear();
    mesh.d_cell_owned_node_global_ids.clear(); mesh.d_face_owned_node_global_ids.clear();
    mesh.d_face_key_to_face.clear();
    mesh.d_owned_face_global_ids.clear();
    // Clear contiguous Tpetra GID assignments; they will be recomputed by create_maps().
    mesh.d_ghost_cell_tpetra_gids.clear();
    mesh.d_mesh_gid_to_tpetra_gid.clear();
    mesh.d_tpetra_gid_to_mesh_gid.clear();
    mesh.d_tpetra_gid_offset = 0;

    const auto total_cells = owned_pkts.size() + ghost_pkts.size();
    size_t total_cell_nodes = 0;
    size_t maximum_face_nodes = 0;
    for (const auto* packets : {&owned_pkts, &ghost_pkts})
    {
        for (const auto& packet : *packets)
        {
            total_cell_nodes += packet.node_gids.size();
            for (const auto& face_nodes : packet.face_node_keys)
            {
                maximum_face_nodes += face_nodes.size();
            }
        }
    }
    mesh.d_cells.reserve(total_cells);
    mesh.d_owned_cell_ids.reserve(owned_pkts.size());
    mesh.d_owned_cell_global_ids.reserve(owned_pkts.size());
    mesh.d_ghost_cell_global_ids.reserve(ghost_pkts.size());
    mesh.d_cell_owned_node_global_ids.reserve(total_cell_nodes);
    mesh.d_face_owned_node_global_ids.reserve(maximum_face_nodes);

    std::unordered_map<GO, Vec3> needed_nodes;
    for (const auto* packets : {&owned_pkts, &ghost_pkts}) {
        for (const auto& packet : *packets) {
            if (packet.node_gids.size() != packet.node_coords.size()) {
                throw std::runtime_error(
                    "MeshPartitioner cell packet has inconsistent node coordinates.");
            }
            if (packet.face_node_keys.size()
                != packet.face_boundary_ids.size()) {
                throw std::runtime_error(
                    "MeshPartitioner cell packet has inconsistent boundary IDs.");
            }
            if (packet.face_node_keys.size()
                != packet.face_global_ids.size()) {
                throw std::runtime_error(
                    "MeshPartitioner cell packet has inconsistent face global IDs.");
            }
            for (size_t node = 0; node < packet.node_gids.size(); ++node) {
                needed_nodes.emplace(
                    packet.node_gids[node], packet.node_coords[node]);
            }
            for (const auto& face_nodes : packet.face_node_keys) {
                for (const auto node_gid : face_nodes) {
                    if (needed_nodes.contains(node_gid)) continue;
                    const auto original = orig_ng2l.find(node_gid);
                    if (original != orig_ng2l.end()) {
                        needed_nodes.emplace(
                            node_gid,
                            orig_coords[static_cast<size_t>(original->second)]);
                    }
                }
            }
        }
    }
    mesh.d_node_coords.reserve(needed_nodes.size());
    for (const auto& [node_gid, coordinate] : needed_nodes) {
        mesh.d_node_gid_to_lid[node_gid] =
            static_cast<LO>(mesh.d_node_coords.size());
        mesh.d_node_coords.push_back(coordinate);
    }

    auto add_cell = [&](const Packet& p, bool owned) -> LO {
        LO lid = static_cast<LO>(mesh.d_cells.size());
        mesh.d_cell_gid_to_lid[p.gid] = lid;
        CellInfo ci;
        ci.owned = owned; ci.type = p.cell_type; ci.center = p.center; ci.volume = p.volume;
        size_t off = mesh.d_cell_owned_node_global_ids.size();
        mesh.d_cell_owned_node_global_ids.insert(mesh.d_cell_owned_node_global_ids.end(), p.node_gids.begin(), p.node_gids.end());
        ci.node_gids = typename Mesh<Pack>::ViewGO(mesh.d_cell_owned_node_global_ids.data() + off, p.node_gids.size());
        ci.faces = typename Mesh<Pack>::ViewLO(nullptr, 0);
        ci.face_distances = typename Mesh<Pack>::ViewReal(nullptr, 0);
        mesh.d_cells.push_back(std::move(ci));
        return lid;
    };
    for (auto& p : owned_pkts) { LO lid = add_cell(p, true); mesh.d_owned_cell_ids.push_back(lid); mesh.d_owned_cell_global_ids.push_back(p.gid); }
    for (auto& p : ghost_pkts) { add_cell(p, false); mesh.d_ghost_cell_global_ids.push_back(p.gid); }

    // Build faces
    std::vector<std::vector<LO>> cfl(mesh.d_cells.size());
    auto all = owned_pkts; all.insert(all.end(), ghost_pkts.begin(), ghost_pkts.end());
    for (size_t ci = 0; ci < all.size(); ++ci) {
        auto& p = all[ci]; LO cl = static_cast<LO>(ci);
        for (size_t packet_face = 0;
             packet_face < p.face_node_keys.size();
             ++packet_face) {
            auto& fn = p.face_node_keys[packet_face];
            const auto boundary_id =
                p.face_boundary_ids[packet_face];
            const auto face_global_id =
                p.face_global_ids[packet_face];
            std::string key = Mesh<Pack>::make_face_key(typename Mesh<Pack>::ViewGO(const_cast<GO*>(fn.data()), fn.size()));
            auto it = mesh.d_face_key_to_face.find(key);
            if (it == mesh.d_face_key_to_face.end()) {
                LO fid = static_cast<LO>(mesh.d_faces.size());
                size_t off = mesh.d_face_owned_node_global_ids.size();
                mesh.d_face_owned_node_global_ids.insert(mesh.d_face_owned_node_global_ids.end(), fn.begin(), fn.end());
                FaceInfo fi;
                fi.type = (fn.size() == 3) ? MeshUtils::FaceType::TRIANGLE : MeshUtils::FaceType::QUAD;
                fi.boundary_id = boundary_id;
                fi.owner = cl; fi.neighbor = invalid_lid;
                fi.node_gids = typename Mesh<Pack>::ViewGO(mesh.d_face_owned_node_global_ids.data() + off, fn.size());
                mesh.d_faces.push_back(std::move(fi));
                mesh.d_owned_face_global_ids.push_back(face_global_id);
                mesh.d_face_key_to_face[key] = fid;
                cfl[static_cast<size_t>(cl)].push_back(fid);
            } else {
                LO fid = it->second;
                if (mesh.d_owned_face_global_ids[static_cast<size_t>(fid)]
                    != face_global_id)
                {
                    throw std::runtime_error(
                        "MeshPartitioner received inconsistent global IDs for the same face.");
                }
                auto& fi = mesh.d_faces[static_cast<size_t>(fid)];
                if (fi.boundary_id == Mesh<Pack>::invalid_boundary_id
                    && boundary_id != Mesh<Pack>::invalid_boundary_id)
                {
                    fi.boundary_id = boundary_id;
                }
                if (fi.neighbor == invalid_lid) {
                    fi.neighbor = cl;
                    cfl[static_cast<size_t>(cl)].push_back(fid);
                } else if (fi.owner == cl || fi.neighbor == cl) {
                    // face already complete; only add if cell is owner/neighbor
                    cfl[static_cast<size_t>(cl)].push_back(fid);
                }
                // else: third cell referencing same face (non-manifold) — skip
            }
        }
    }

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
}
