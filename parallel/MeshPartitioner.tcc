/**
 * @file MeshPartitioner.tcc
 * @author your name (you@domain.com)
 * @brief template implementations for MeshPartitioner class.
 * @version 0.1
 * @date 2026-05-29
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "MeshPartitioner.hh"

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
        std::vector<int> counts(static_cast<std::size_t>(nranks), 0);
        MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      get_mpi_comm(*comm));

        std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
        for (int r = 1; r < nranks; ++r) {
            displs[static_cast<std::size_t>(r)] =
                displs[static_cast<std::size_t>(r - 1)]
                + counts[static_cast<std::size_t>(r - 1)];
        }

        const int total_count = displs.back() + counts.back();
        std::vector<GO> gathered_gids(static_cast<std::size_t>(std::max(total_count, 1)));
        MPI_Allgatherv(mesh.d_owned_cell_global_ids.data(), my_count, mpi_go_type(),
                       gathered_gids.data(), counts.data(), displs.data(), mpi_go_type(),
                       get_mpi_comm(*comm));

        for (int r = 0; r < nranks; ++r) {
            const int begin = displs[static_cast<std::size_t>(r)];
            const int end = begin + counts[static_cast<std::size_t>(r)];
            for (int i = begin; i < end; ++i) {
                auto [it, inserted] = source_rank_for_gid.emplace(gathered_gids[static_cast<std::size_t>(i)], r);
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
    for (std::size_t i = 0; i < mesh.d_owned_cell_global_ids.size(); ++i) {
        GO gid = mesh.d_owned_cell_global_ids[i];
        if (source_rank_for_gid.at(gid) != myrank || !packed_gids.insert(gid).second) continue;

        LO lid = mesh.d_owned_cell_ids[i];
        auto& cell = mesh.d_cells[static_cast<std::size_t>(lid)];
        int dest = gid_to_rank.at(gid);
        Packet p;
        p.gid = gid; p.cell_type = cell.type; p.center = cell.center; p.volume = cell.volume;
        p.node_gids.assign(cell.node_gids.begin(), cell.node_gids.end());
        for (auto fid : cell.faces) {
            auto& face = mesh.d_faces[static_cast<std::size_t>(fid)];
            std::vector<GO> fn(face.node_gids.begin(), face.node_gids.end());
            std::sort(fn.begin(), fn.end());
            p.face_node_keys.push_back(std::move(fn));
        }
        send_p[static_cast<std::size_t>(dest)].push_back(std::move(p));
    }

    // Serialize and exchange
    std::vector<std::vector<char>> sbufs(nranks);
    std::vector<int> scnt(nranks, 0);
    for (int r = 0; r < nranks; ++r) {
        if (send_p[static_cast<std::size_t>(r)].empty()) continue;
        sbufs[static_cast<std::size_t>(r)] = Packet::serialize_packets(send_p[static_cast<std::size_t>(r)]);
        scnt[static_cast<std::size_t>(r)] = static_cast<int>(sbufs[static_cast<std::size_t>(r)].size());
    }

    std::vector<int> rcnt(nranks, 0);
    MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, get_mpi_comm(*comm));
    std::vector<int> sd(nranks, 0), rd(nranks, 0);
    for (int r = 1; r < nranks; ++r) {
        sd[static_cast<std::size_t>(r)] = sd[static_cast<std::size_t>(r - 1)] + scnt[static_cast<std::size_t>(r - 1)];
        rd[static_cast<std::size_t>(r)] = rd[static_cast<std::size_t>(r - 1)] + rcnt[static_cast<std::size_t>(r - 1)];
    }
    std::vector<char> flat_s;
    { std::size_t off = 0; for (int r = 0; r < nranks; ++r) {
        auto& b = sbufs[static_cast<std::size_t>(r)]; if (b.empty()) continue;
        flat_s.resize(off + b.size()); std::memcpy(flat_s.data() + off, b.data(), b.size()); off += b.size();
    } if (flat_s.empty()) flat_s.resize(1); }
    std::vector<char> rbuf(static_cast<std::size_t>(std::max(rd.back() + rcnt.back(), 1)));
    MPI_Alltoallv(flat_s.data(), scnt.data(), sd.data(), MPI_CHAR,
                    rbuf.data(), rcnt.data(), rd.data(), MPI_CHAR, get_mpi_comm(*comm));

    std::vector<Packet> owned_pkts;
    { std::size_t off = 0; for (int r = 0; r < nranks; ++r) {
        int sz = rcnt[static_cast<std::size_t>(r)]; if (sz <= 0) continue;
        auto pkts = Packet::deserialize_packets(rbuf.data() + off, static_cast<std::size_t>(sz));
        owned_pkts.insert(owned_pkts.end(), std::make_move_iterator(pkts.begin()), std::make_move_iterator(pkts.end()));
        off += static_cast<std::size_t>(sz);
    }}
    unique_packets_by_gid(owned_pkts);

    // Determine ghosts
    std::unordered_set<GO> owned_set;
    for (auto& p : owned_pkts) owned_set.insert(p.gid);
    std::unordered_map<std::string, std::pair<GO, GO>> fk2cells;
    for (std::size_t fid = 0; fid < mesh.d_faces.size(); ++fid) {
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

    // Request ghosts
    std::vector<std::vector<GO>> greq(nranks);
    for (GO g : ghost_set) greq[static_cast<std::size_t>(gid_to_rank.at(g))].push_back(g);
    std::vector<int> req_c(nranks, 0), resp_c(nranks, 0);
    for (int r = 0; r < nranks; ++r) req_c[static_cast<std::size_t>(r)] = static_cast<int>(greq[static_cast<std::size_t>(r)].size());
    MPI_Alltoall(req_c.data(), 1, MPI_INT, resp_c.data(), 1, MPI_INT, get_mpi_comm(*comm));
    std::vector<int> req_d(nranks, 0), resp_d(nranks, 0);
    for (int r = 1; r < nranks; ++r) {
        req_d[static_cast<std::size_t>(r)] = req_d[static_cast<std::size_t>(r - 1)] + req_c[static_cast<std::size_t>(r - 1)];
        resp_d[static_cast<std::size_t>(r)] = resp_d[static_cast<std::size_t>(r - 1)] + resp_c[static_cast<std::size_t>(r - 1)];
    }
    std::vector<GO> flat_rq;
    for (int r = 0; r < nranks; ++r) { auto& gr = greq[static_cast<std::size_t>(r)]; flat_rq.insert(flat_rq.end(), gr.begin(), gr.end()); }
    if (flat_rq.empty()) flat_rq.resize(1);
    std::vector<GO> flat_rs(static_cast<std::size_t>(std::max(resp_d.back() + resp_c.back(), 1)));
    MPI_Alltoallv(flat_rq.data(), req_c.data(), req_d.data(), mpi_go_type(),
                    flat_rs.data(), resp_c.data(), resp_d.data(), mpi_go_type(), get_mpi_comm(*comm));

    std::unordered_map<GO, const Packet*> olu;
    for (auto& p : owned_pkts) olu[p.gid] = &p;
    std::vector<std::vector<Packet>> gresp(nranks);
    for (int r = 0; r < nranks; ++r) {
        int cnt = resp_c[static_cast<std::size_t>(r)]; if (cnt <= 0) continue;
        GO* rs = flat_rs.data() + resp_d[static_cast<std::size_t>(r)];
        for (int i = 0; i < cnt; ++i) { auto it = olu.find(rs[i]); if (it != olu.end()) gresp[static_cast<std::size_t>(r)].push_back(*it->second); }
    }

    std::vector<std::vector<char>> gr_sbufs(nranks);
    std::vector<int> gr_scnt(nranks, 0);
    for (int r = 0; r < nranks; ++r) {
        if (gresp[static_cast<std::size_t>(r)].empty()) continue;
        gr_sbufs[static_cast<std::size_t>(r)] = Packet::serialize_packets(gresp[static_cast<std::size_t>(r)]);
        gr_scnt[static_cast<std::size_t>(r)] = static_cast<int>(gr_sbufs[static_cast<std::size_t>(r)].size());
    }
    std::vector<int> gr_rcnt(nranks, 0);
    MPI_Alltoall(gr_scnt.data(), 1, MPI_INT, gr_rcnt.data(), 1, MPI_INT, get_mpi_comm(*comm));
    std::vector<int> gr_sd(nranks, 0), gr_rd(nranks, 0);
    for (int r = 1; r < nranks; ++r) {
        gr_sd[static_cast<std::size_t>(r)] = gr_sd[static_cast<std::size_t>(r - 1)] + gr_scnt[static_cast<std::size_t>(r - 1)];
        gr_rd[static_cast<std::size_t>(r)] = gr_rd[static_cast<std::size_t>(r - 1)] + gr_rcnt[static_cast<std::size_t>(r - 1)];
    }
    std::vector<char> gr_flat_s;
    { std::size_t off = 0; for (int r = 0; r < nranks; ++r) {
        auto& b = gr_sbufs[static_cast<std::size_t>(r)]; if (b.empty()) continue;
        gr_flat_s.resize(off + b.size()); std::memcpy(gr_flat_s.data() + off, b.data(), b.size()); off += b.size();
    } if (gr_flat_s.empty()) gr_flat_s.resize(1); }
    std::vector<char> gr_rbuf(static_cast<std::size_t>(std::max(gr_rd.back() + gr_rcnt.back(), 1)));
    MPI_Alltoallv(gr_flat_s.data(), gr_scnt.data(), gr_sd.data(), MPI_CHAR,
                    gr_rbuf.data(), gr_rcnt.data(), gr_rd.data(), MPI_CHAR, get_mpi_comm(*comm));

    std::vector<Packet> ghost_pkts;
    { std::size_t off = 0; for (int r = 0; r < nranks; ++r) {
        int sz = gr_rcnt[static_cast<std::size_t>(r)]; if (sz <= 0) continue;
        auto pkts = Packet::deserialize_packets(gr_rbuf.data() + off, static_cast<std::size_t>(sz));
        ghost_pkts.insert(ghost_pkts.end(), std::make_move_iterator(pkts.begin()), std::make_move_iterator(pkts.end()));
        off += static_cast<std::size_t>(sz);
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

    rebuild(mesh, owned_pkts, ghost_pkts, orig_node_coords, orig_node_gid_to_lid);
    return true;
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
 */
template<TpetraTypePack Pack>
auto MeshPartitioner<Pack>::compute_gid_to_rank_map(
    const Mesh<Pack>& mesh, const Teuchos::RCP<const comm_type>& comm) -> std::unordered_map<GO, int>
{
    int nranks = comm->getSize(), myrank = comm->getRank();
    std::unordered_map<GO, int> result;
    auto nc = mesh.d_owned_cell_global_ids.size();

    std::unordered_map<GO, std::vector<GO>> all_adj;
    for (std::size_t i = 0; i < nc; ++i) {
        GO gid = mesh.d_owned_cell_global_ids[i];
        LO lid = mesh.d_owned_cell_ids[i];
        auto& cell = mesh.d_cells[static_cast<std::size_t>(lid)];
        std::unordered_set<GO> ns;
        for (auto fid : cell.faces) {
            auto& face = mesh.d_faces[static_cast<std::size_t>(fid)];
            if (face.owner == lid && face.neighbor != invalid_lid) ns.insert(mesh.cell_global_id(face.neighbor));
            else if (face.neighbor == lid) ns.insert(mesh.cell_global_id(face.owner));
        }
        all_adj[gid] = std::vector<GO>(ns.begin(), ns.end());
    }

    std::vector<GO> my_gids;
    std::vector<std::vector<GO>> my_adj;
    for (std::size_t i = 0; i < nc; ++i) {
        GO gid = mesh.d_owned_cell_global_ids[i];
        if (static_cast<int>(static_cast<std::size_t>(gid) % static_cast<std::size_t>(nranks)) == myrank) {
            my_gids.push_back(gid);
            my_adj.push_back(all_adj[gid]);
        }
    }

    auto inv_g = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    Teuchos::RCP<const map_type> row_map;
    if (!my_gids.empty())
        row_map = Teuchos::rcp(new map_type(inv_g, my_gids.data(), static_cast<std::size_t>(my_gids.size()), static_cast<GO>(0), comm));
    else
        row_map = Teuchos::rcp(new map_type(inv_g, static_cast<std::size_t>(0), static_cast<GO>(0), comm));
    Teuchos::RCP<const map_type> col_map;
    if (!mesh.d_owned_cell_global_ids.empty())
        col_map = Teuchos::rcp(new map_type(inv_g, mesh.d_owned_cell_global_ids.data(), static_cast<std::size_t>(nc), static_cast<GO>(0), comm));
    else
        col_map = Teuchos::rcp(new map_type(inv_g, static_cast<std::size_t>(0), static_cast<GO>(0), comm));

    std::size_t max_adj = 0;
    for (auto& a : my_adj) if (a.size() > max_adj) max_adj = a.size();
    Teuchos::RCP<graph_type> graph = Teuchos::rcp(new graph_type(row_map, col_map, max_adj));
    for (std::size_t i = 0; i < my_gids.size(); ++i) {
        auto& adj = my_adj[i];
        if (!adj.empty()) graph->insertGlobalIndices(my_gids[i], adj.size(), adj.data());
    }
    graph->fillComplete();

    {
        using adapter_t = Zoltan2::TpetraRowGraphAdapter<graph_type>;
        Teuchos::ParameterList params;
        params.set("algorithm", "parmetis");
        params.set("num_global_parts", nranks);
        adapter_t adapter(graph);
        Zoltan2::PartitioningProblem<adapter_t> problem(&adapter, &params, comm);
        problem.solve();
        auto& sol = problem.getSolution();
        auto* pl = sol.getPartListView();
        for (std::size_t i = 0; i < my_gids.size(); ++i) result[my_gids[i]] = pl[i];
    }

    // Allgatherv full map
    {
        std::vector<GO> pairs; pairs.reserve(result.size() * 2);
        for (auto& [g, r] : result) { pairs.push_back(g); pairs.push_back(static_cast<GO>(r)); }
        int mc = static_cast<int>(pairs.size());
        std::vector<int> ac(static_cast<std::size_t>(nranks));
        MPI_Allgather(&mc, 1, MPI_INT, ac.data(), 1, MPI_INT, get_mpi_comm(*comm));
        std::vector<int> ad(static_cast<std::size_t>(nranks), 0);
        for (int r = 1; r < nranks; ++r) ad[static_cast<std::size_t>(r)] = ad[static_cast<std::size_t>(r - 1)] + ac[static_cast<std::size_t>(r - 1)];
        int tot = ad.back() + ac.back();
        std::vector<GO> ap(static_cast<std::size_t>(std::max(tot, 0)));
        MPI_Allgatherv(pairs.data(), mc, mpi_go_type(), ap.data(), ac.data(), ad.data(), mpi_go_type(), get_mpi_comm(*comm));
        result.clear();
        for (std::size_t i = 0; i + 1 < ap.size(); i += 2) result[ap[i]] = static_cast<int>(ap[i + 1]);
    }
    return result;
}


/**
 * @brief Rebuild the mesh data structures from owned and ghost cell packets.
 *
 * Clears and repopulates d_cells, d_faces, node tables, face lookups,
 * cell-face adjacency, and face geometry (centroids, normals, areas,
 * distances).  Face deduplication uses sorted-node-GID keys.
 *
 * @param mesh Mesh instance to rebuild (modified in place).
 * @param owned_pkts Cell packets destined for this rank.
 * @param ghost_pkts Cell packets describing off-rank neighbours.
 * @param orig_coords Original node coordinates before partitioning.
 * @param orig_ng2l Original node GID-to-local-index map.
 */
template<TpetraTypePack Pack>
void MeshPartitioner<Pack>::rebuild(Mesh<Pack>& mesh, const std::vector<Packet>& owned_pkts, const std::vector<Packet>& ghost_pkts,
                    const std::vector<Vec3>& orig_coords, const std::unordered_map<GO, LO>& orig_ng2l) {
    mesh.d_cells.clear(); mesh.d_faces.clear(); mesh.d_owned_cell_ids.clear();
    mesh.d_owned_cell_global_ids.clear(); mesh.d_ghost_cell_global_ids.clear();
    mesh.d_cell_gid_to_lid.clear(); mesh.d_node_gid_to_lid.clear(); mesh.d_node_coords.clear();
    mesh.d_cell_owned_face_ids.clear(); mesh.d_cell_face_distances.clear();
    mesh.d_cell_owned_node_global_ids.clear(); mesh.d_face_owned_node_global_ids.clear();
    mesh.d_face_key_to_face.clear();

    std::unordered_set<GO> needed_nodes;
    for (auto& p : owned_pkts) { for (auto n : p.node_gids) needed_nodes.insert(n); for (auto& fn : p.face_node_keys) for (auto n : fn) needed_nodes.insert(n); }
    for (auto& p : ghost_pkts) { for (auto n : p.node_gids) needed_nodes.insert(n); for (auto& fn : p.face_node_keys) for (auto n : fn) needed_nodes.insert(n); }
    for (GO nid : needed_nodes) {
        auto it = orig_ng2l.find(nid); if (it == orig_ng2l.end()) continue;
        mesh.d_node_gid_to_lid[nid] = static_cast<LO>(mesh.d_node_coords.size());
        mesh.d_node_coords.push_back(orig_coords[static_cast<std::size_t>(it->second)]);
    }

    auto add_cell = [&](const Packet& p, bool owned) -> LO {
        LO lid = static_cast<LO>(mesh.d_cells.size());
        mesh.d_cell_gid_to_lid[p.gid] = lid;
        CellInfo ci;
        ci.owned = owned; ci.type = p.cell_type; ci.center = p.center; ci.volume = p.volume;
        std::size_t off = mesh.d_cell_owned_node_global_ids.size();
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
    for (std::size_t ci = 0; ci < all.size(); ++ci) {
        auto& p = all[ci]; LO cl = static_cast<LO>(ci);
        for (auto& fn : p.face_node_keys) {
            std::string key = Mesh<Pack>::make_face_key(typename Mesh<Pack>::ViewGO(const_cast<GO*>(fn.data()), fn.size()));
            auto it = mesh.d_face_key_to_face.find(key);
            if (it == mesh.d_face_key_to_face.end()) {
                LO fid = static_cast<LO>(mesh.d_faces.size());
                std::size_t off = mesh.d_face_owned_node_global_ids.size();
                mesh.d_face_owned_node_global_ids.insert(mesh.d_face_owned_node_global_ids.end(), fn.begin(), fn.end());
                FaceInfo fi;
                fi.type = (fn.size() == 3) ? MeshUtils::FaceType::TRIANGLE : MeshUtils::FaceType::QUAD;
                fi.boundary_id = Mesh<Pack>::invalid_boundary_id;
                fi.owner = cl; fi.neighbor = invalid_lid;
                fi.node_gids = typename Mesh<Pack>::ViewGO(mesh.d_face_owned_node_global_ids.data() + off, fn.size());
                mesh.d_faces.push_back(std::move(fi));
                mesh.d_face_key_to_face[key] = fid;
                cfl[static_cast<std::size_t>(cl)].push_back(fid);
            } else {
                LO fid = it->second;
                auto& fi = mesh.d_faces[static_cast<std::size_t>(fid)];
                if (fi.neighbor == invalid_lid) {
                    fi.neighbor = cl;
                    cfl[static_cast<std::size_t>(cl)].push_back(fid);
                } else if (fi.owner == cl || fi.neighbor == cl) {
                    // face already complete; only add if cell is owner/neighbor
                    cfl[static_cast<std::size_t>(cl)].push_back(fid);
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
            if (nit != mesh.d_node_gid_to_lid.end()) fcs.push_back(mesh.d_node_coords[static_cast<std::size_t>(nit->second)]);
        }
        if (fcs.size() < 3) continue;
        fi.center = MeshUtils::average(fcs);
        auto av = MeshUtils::face_area_vector(fcs);
        fi.area = av.norm();
        if (fi.area <= 0.0) continue;
        auto nml = av / fi.area;
        auto& oc = mesh.d_cells[static_cast<std::size_t>(fi.owner)];
        auto ov = fi.center - oc.center;
        if (nml.dot(ov) < 0.0) nml = nml * -1.0;
        fi.unit_normal_from_owner = nml; fi.unit_normal_from_neighbor = nml * -1.0;
        fi.owner_to_face_distance = ov.norm();
        if (fi.neighbor != invalid_lid) {
            auto& nc = mesh.d_cells[static_cast<std::size_t>(fi.neighbor)];
            auto nv = fi.center - nc.center;
            fi.neighbor_to_face_distance = nv.norm();
            fi.cell_center_distance = (nc.center - oc.center).norm();
        }
    }

    // Set cell face views
    std::size_t tcf = 0; for (auto& cf : cfl) tcf += cf.size();
    // DEBUG: verify GID consistency
    for (std::size_t i = 0; i < mesh.d_owned_cell_global_ids.size(); ++i) {
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
    for (std::size_t ci = 0; ci < mesh.d_cells.size(); ++ci) {
        std::size_t off = mesh.d_cell_owned_face_ids.size();
        auto& cf = cfl[ci];
        mesh.d_cell_owned_face_ids.insert(mesh.d_cell_owned_face_ids.end(), cf.begin(), cf.end());
        mesh.d_cells[ci].faces = typename Mesh<Pack>::ViewLO(mesh.d_cell_owned_face_ids.data() + off, cf.size());
    }
}
