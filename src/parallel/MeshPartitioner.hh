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
#include "utils/export_helper.hh"
#include "geometry/MeshUtils.hh"
#include "geometry/mesh/LocalGlobalIndexer.hh"
#include "geometry/mesh/UnstructuredMesh.hh"
#include "geometry/mesh/UnstructuredMeshData.hh"
#include "parallel/MPI_interface.hh"
#include "parallel/CompositePartition.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace SimpleFluid {

/**
 * @brief Parallel mesh partitioner using Zoltan2/ParMETIS.
 *
 * Decomposes source mesh cells into per-rank subdomains via a 5-phase
 * algorithm: (1) Zoltan2 partition, (2) cell redistribution, (3) ghost
 * detection, (4) ghost exchange, (5) mesh rebuild.
 * Source cells may be replicated, distributed, or held only on rank zero.
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
    using indexer_type =
        Meshes::UnstructuredMesh::local_global_indexer_t<LO, GO>;
    static constexpr LO invalid_lid = invalid_id<LO>();

    /**
     * @brief Local entity lists for a partitioned CRTP unstructured mesh.
     *
     * The input mesh is rebuilt to contain only entities visible on this
     * rank. The indexer maps its compact local IDs to IDs in the original
     * replicated mesh. Owned entities precede overlap entities, and
     * `cell_owner_ranks` is indexed by rebuilt local cell ordinal.
     */
    struct UnstructuredPartition
    {
        indexer_type indexer;
        std::vector<int> cell_owner_ranks;
    };

    /**
     * @brief Partition a replicated CRTP unstructured mesh.
     *
     * Cell geometry local IDs are used as graph global IDs. The input mesh
     * is replaced by rank-local geometry and the returned partition
     * contains its rebuilt local/global cell, face, and node indexer.
     *
     * @pre Every rank in @p comm calls this function collectively.
     * @param mesh Replicated geometry, rebuilt in place for the local rank.
     * @param comm Teuchos MPI communicator.
     * @return Global indexing and ownership metadata for the rebuilt mesh.
     */
    static UnstructuredPartition partition(
        Meshes::UnstructuredMesh& mesh,
        const Teuchos::RCP<const comm_type>& comm);

    /** @brief Compact native geometry and its source/algebraic identity maps. */
    struct NativeUnstructuredPartition
    {
        std::shared_ptr<Meshes::UnstructuredMesh> geometry;
        indexer_type indexer;
        std::vector<int> cell_owner_ranks;
    };

    /**
     * @brief Collectively distribute root-only or uniquely owned native records.
     *
     * Only the owned prefix of source.cells supplies graph rows. Duplicate
     * source cells and inconsistent shared entities fail collectively. The
     * default ParMETIS policy is retained; geometry travels through sparse
     * directory/owner exchanges and is never gathered or replicated globally.
     * Source IDs and optional algebraic ordinals remain distinct in the indexer.
     * Faces incident on owned cells retain the existing overlapping ownership
     * convention. All ranks retain the complete boundary catalog.
     *
     * @param halo_depth Number of adjacency layers, at least one.
     * @param contiguous_algebraic_ids Explicit factory compatibility policy:
     *        assign rank-contiguous cell ordinals after redistribution. Face
     *        and node ordinals retain their record values.
     */
    static NativeUnstructuredPartition partition_native(
        Meshes::UnstructuredMeshData source,
        const Teuchos::RCP<const comm_type>& comm,
        unsigned halo_depth = 1,
        bool contiguous_algebraic_ids = false);

    /** @brief Collectively plan ownership without changing source geometry. */
    static CompositePartitionPlan<Pack> make_plan(
        const CompositeMeshSource& source,
        const CompositePartitionOptions& options,
        const Teuchos::RCP<const comm_type>& comm);

    /** @brief Apply a plan, retaining compact regions and local explicit payloads. */
    static CompositePartition<Pack> distribute(
        const CompositeMeshSource& source,
        const CompositePartitionPlan<Pack>& plan);

    static CompositePartition<Pack> partition(
        const CompositeMeshSource& source,
        const CompositePartitionOptions& options,
        const Teuchos::RCP<const comm_type>& comm);

private:
    /**
     * @brief Distributed graph rows and referenced column cell IDs.
     *
     * Rows are owned by this rank; adjacency is parallel to `row_gids`, while
     * `column_gids` is the unique set of local and remote referenced cells.
     */
    struct PartitionGraph
    {
        std::vector<GO> row_gids;
        std::vector<std::vector<GO>> row_adjacency;
        std::vector<GO> column_gids;
        std::vector<double> row_weights;
        std::vector<std::vector<double>> row_edge_weights;
        double imbalance_tolerance = 1.1;
    };

    /**
     * @brief Owned-first ordering of source entity ordinals.
     *
     * Cells are divided into owned and ghost sets; faces and nodes are divided
     * into owned and overlap sets before the rank-local mesh is rebuilt.
     */
    struct LocalEntityOrder
    {
        std::vector<size_t> owned_cells;
        std::vector<size_t> ghost_cells;
        std::vector<size_t> owned_faces;
        std::vector<size_t> overlap_faces;
        std::vector<size_t> owned_nodes;
        std::vector<size_t> overlap_nodes;
    };


    static PartitionGraph build_partition_graph(
        const Meshes::UnstructuredMesh& mesh,
        const Teuchos::RCP<const comm_type>& comm);

    /**
     * @brief Collectively partition a distributed adjacency graph with Zoltan2.
     * @param graph Global-cell adjacency graph distributed across the communicator.
     * @param comm Communicator containing all graph participants.
     * @param gather_root Rank that receives the complete result, or -1 to
     *        replicate it on all ranks, or -2 to retain local row assignments.
     * @return Destination rank indexed by global cell identifier on the
     *         requested recipients.
     */
    static std::unordered_map<GO, int> solve_partition_graph(
        const PartitionGraph& graph,
        const Teuchos::RCP<const comm_type>& comm,
        int gather_root = -1);

    template<class CellFaces, class OppositeCell, class FaceNodes, class CellNodes>
    static LocalEntityOrder reorder_local_entities(
        size_t cell_count,
        const std::vector<int>& cell_owner_ranks,
        int rank,
        CellFaces&& cell_faces,
        OppositeCell&& opposite_cell,
        FaceNodes&& face_nodes,
        CellNodes&& cell_nodes);

    /**
     * @brief Collectively assign every unstructured cell to an owner rank.
     * @return Owner rank indexed by source cell ordinal.
     */
    static std::vector<int> compute_unstructured_owner_ranks(
        const Meshes::UnstructuredMesh& mesh,
        const Teuchos::RCP<const comm_type>& comm);

    /**
     * @brief Replace replicated geometry with this rank's owned-first subset.
     * @return Rebuilt local/global indexer and local cell owners.
     */
    static UnstructuredPartition rebuild(
        Meshes::UnstructuredMesh& mesh,
        std::vector<int> cell_owner_ranks,
        int rank);

    /**
     * @brief Extract the raw MPI_Comm from a Teuchos communicator wrapper.
     *
     * @param tc Teuchos communicator (must wrap an MpiComm<int>).
     * @return Raw MPI_Comm, or MPI_COMM_NULL if the cast fails.
     */
    static my_mpi::Comm get_mpi_comm(const comm_type& tc) {
        auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&tc);
        if (mpi) return *(mpi->getRawMpiComm()); return MPI_COMM_NULL;
    }
    /**
     * @brief Return the MPI datatype corresponding to the global ordinal type.
     *
     * @return Type-deduced MPI datatype via my_mpi::type_trait.
     */
    static MPI_Datatype mpi_go_type() { return my_mpi::type_trait<GO>(); }
};

} // namespace SimpleFluid
