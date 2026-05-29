/**
 * @file MeshPartitioner.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Zoltan2-based mesh domain decomposition for parallel runs.
 * @version 0.1
 * @date 2026-05-29
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "geometry/Mesh.hh"
#include "parallel/MPI_interface.hh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

namespace SimpleFluid {

namespace partition_detail {

/**
 * @brief Serialisable packet carrying the geometry and connectivity of a single cell.
 *
 * Used to redistribute cells across MPI ranks after partitioning.
 *
 * @tparam Pack Tpetra type pack providing GO, LO, map, and graph types.
 */
template<TpetraTypePack Pack>
struct CellPacket {
    using GO = typename Pack::global_ordinal_type;
    using Vec3 = MeshUtils::Vec3;
    using CellType = MeshUtils::CellType;

    GO gid{};
    CellType cell_type = CellType::INVALID;
    Vec3 center{};
    double volume = 0.0;
    std::vector<GO> node_gids;
    std::vector<std::vector<GO>> face_node_keys;

    /**
     * @brief Serialise a vector of cell packets into a flat byte buffer for MPI transmission.
     *
     * @param pkts Vector of cell packets to serialise.
     * @return Flat byte buffer suitable for MPI_Alltoallv.
     */
    static std::vector<char> serialize_packets(const std::vector<CellPacket>& pkts) {
        std::vector<char> buf;
        std::uint32_t n = static_cast<std::uint32_t>(pkts.size());
        append(buf, n);
        for (auto& p : pkts) {
            append(buf, p.gid);
            append(buf, static_cast<std::int32_t>(p.cell_type));
            append(buf, p.center.x); append(buf, p.center.y); append(buf, p.center.z);
            append(buf, p.volume);
            std::uint32_t nn = static_cast<std::uint32_t>(p.node_gids.size());
            append(buf, nn);
            for (auto id : p.node_gids) append(buf, id);
            std::uint32_t nf = static_cast<std::uint32_t>(p.face_node_keys.size());
            append(buf, nf);
            for (auto& fn : p.face_node_keys) {
                std::uint32_t fnsz = static_cast<std::uint32_t>(fn.size());
                append(buf, fnsz);
                for (auto id : fn) append(buf, id);
            }
        }
        return buf;
    }

    /**
     * @brief Deserialise a flat byte buffer back into a vector of cell packets.
     *
     * @param data Pointer to the byte buffer received via MPI.
     * @param size Unused; size is encoded in the buffer header.
     * @return Reconstructed vector of cell packets.
     */
    static std::vector<CellPacket> deserialize_packets(const char* data, std::size_t /*size*/) {
        std::vector<CellPacket> pkts;
        const char* p = data;
        auto rd = [&](auto& v) { std::memcpy(&v, p, sizeof(v)); p += sizeof(v); };
        auto rd_gos = [&](std::vector<GO>& v, std::uint32_t n) { v.resize(n); for (auto& x : v) rd(x); };
        std::uint32_t count = 0; rd(count); pkts.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            CellPacket pk;
            rd(pk.gid); std::int32_t ct = 0; rd(ct); pk.cell_type = static_cast<CellType>(ct);
            rd(pk.center.x); rd(pk.center.y); rd(pk.center.z); rd(pk.volume);
            std::uint32_t nn = 0; rd(nn); rd_gos(pk.node_gids, nn);
            std::uint32_t nf = 0; rd(nf); pk.face_node_keys.resize(nf);
            for (std::uint32_t f = 0; f < nf; ++f) {
                std::uint32_t fnsz = 0; rd(fnsz); rd_gos(pk.face_node_keys[f], fnsz);
            }
            pkts.push_back(std::move(pk));
        }
        return pkts;
    }

private:
    template<typename T>
    static void append(std::vector<char>& b, const T& v) {
        auto* cp = reinterpret_cast<const char*>(&v);
        b.insert(b.end(), cp, cp + sizeof(T));
    }
};

} // namespace partition_detail

/**
 * @brief Parallel mesh partitioner using Zoltan2/ParMETIS.
 *
 * Decomposes a replicated mesh into per-rank subdomains via a 5-phase
 * algorithm: (1) Zoltan2 partition, (2) cell redistribution, (3) ghost
 * detection, (4) ghost exchange, (5) mesh rebuild.
 *
 * @tparam Pack Tpetra type pack providing GO, LO, map, graph, and comm types.
 */
template<TpetraTypePack Pack>
class MeshPartitioner {
public:
    using GO = typename Pack::global_ordinal_type;
    using LO = typename Pack::local_ordinal_type;
    using map_type = typename Pack::map_type;
    using graph_type = typename Pack::graph_type;
    using comm_type = typename Pack::comm_type;
    using Vec3 = MeshUtils::Vec3;
    using CellInfo = typename Mesh<Pack>::CellInfo;
    using FaceInfo = typename Mesh<Pack>::FaceInfo;
    using Packet = partition_detail::CellPacket<Pack>;
    static constexpr LO invalid_lid = invalid_id<LO>();

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
    static bool partition(Mesh<Pack>& mesh, const Teuchos::RCP<const comm_type>& comm);

private:
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
    static std::unordered_map<GO, int> compute_gid_to_rank_map(
        const Mesh<Pack>& mesh, const Teuchos::RCP<const comm_type>& comm);

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
    static void rebuild(Mesh<Pack>& mesh, 
                        const std::vector<Packet>& owned_pkts,
                        const std::vector<Packet>& ghost_pkts,
                        const std::vector<Vec3>& orig_coords,
                        const std::unordered_map<GO, LO>& orig_ng2l);


    /**
     * @brief Extract the raw MPI_Comm from a Teuchos communicator wrapper.
     *
     * @param tc Teuchos communicator (must wrap an MpiComm<int>).
     * @return Raw MPI_Comm, or MPI_COMM_NULL if the cast fails.
     */
    static MPI_Comm get_mpi_comm(const comm_type& tc) {
        auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&tc);
        if (mpi) return *(mpi->getRawMpiComm()); return MPI_COMM_NULL;
    }
    /**
     * @brief Return the MPI datatype corresponding to the global ordinal type.
     *
     * @return MPI_INT64_T if GO is 8 bytes, MPI_INT32_T otherwise.
     */
    static MPI_Datatype mpi_go_type() { return sizeof(GO) == 8 ? MPI_INT64_T : MPI_INT32_T; }
};

} // namespace SimpleFluid
