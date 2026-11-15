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
#include "utils/export_helper.hh"
#include "MeshPartitioner.hh"

#include <map>
#include <type_traits>
#include <exception>
#include <optional>

#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_TpetraRowGraphAdapter.hpp>

using namespace SimpleFluid;


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
auto MeshPartitioner<Pack>::solve_partition_graph(const PartitionGraph& partition_graph,
    const Teuchos::RCP<const comm_type>& comm, int gather_root) -> std::unordered_map<GO, int>
{
    const auto nranks = comm->getSize();
    const auto myrank = comm->getRank();
    // The catch path must not allocate: a failed allocation is itself one of
    // the local failures that must be agreed before the next collective.
    const auto agreed_phase = [&](const char* name, auto&& operation)
    {
        std::exception_ptr failure;
        try
        {
            operation();
        }
        catch (...)
        {
            failure = std::current_exception();
        }
        int local_failed = failure ? 1 : 0, any_failed = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_failed, &any_failed);
        if (any_failed)
        {
            if (failure)
                std::rethrow_exception(failure);
            throw std::runtime_error(std::string("MeshPartitioner graph phase failed: ") + name);
        }
    };
    int min_root = 0, max_root = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &gather_root, &min_root);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &gather_root, &max_root);
    size_t max_adjacency = 0;
    agreed_phase("input validation",
        [&]
        {
            if (min_root != max_root || gather_root < -2 || gather_root >= nranks)
                throw std::invalid_argument("MeshPartitioner partition-result root is invalid or inconsistent.");
            const auto rows = partition_graph.row_gids.size();
            if (!std::in_range<LO>(rows) || partition_graph.row_adjacency.size() != rows ||
                (!partition_graph.row_weights.empty() && partition_graph.row_weights.size() != rows) ||
                (!partition_graph.row_edge_weights.empty() && partition_graph.row_edge_weights.size() != rows))
                throw std::invalid_argument("Invalid local partition graph shape.");
            for (size_t row = 0; row < rows; ++row)
            {
                const auto count = partition_graph.row_adjacency[row].size();
                if (!partition_graph.row_edge_weights.empty() && partition_graph.row_edge_weights[row].size() != count)
                    throw std::invalid_argument("Invalid partition edge-weight count.");
                max_adjacency = std::max(max_adjacency, count);
            }
        });
    const auto invalid_size = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    // Stack-backed object storage avoids a caller-side new allocation before
    // constructors that perform their own collectives. Third-party internal
    // collective/error handling remains the responsibility of Tpetra/Zoltan2.
    std::optional<map_type> row_storage, column_storage;
    Teuchos::RCP<const map_type> row_map, column_map;
    agreed_phase("row map construction",
        [&]
        {
            row_storage.emplace(
                invalid_size, partition_graph.row_gids.data(), partition_graph.row_gids.size(), GO{}, comm);
        });
    agreed_phase("row map reference", [&] { row_map = Teuchos::rcpFromRef(*row_storage); });
    agreed_phase("column map construction",
        [&]
        {
            column_storage.emplace(
                invalid_size, partition_graph.column_gids.data(), partition_graph.column_gids.size(), GO{}, comm);
        });
    agreed_phase("column map reference", [&] { column_map = Teuchos::rcpFromRef(*column_storage); });
    std::optional<graph_type> graph_storage;
    Teuchos::RCP<graph_type> graph;
    agreed_phase("local graph insertion",
        [&]
        {
            graph_storage.emplace(row_map, column_map, max_adjacency);
            graph = Teuchos::rcpFromRef(*graph_storage);
            for (size_t row = 0; row < partition_graph.row_gids.size(); ++row)
            {
                const auto& adjacency = partition_graph.row_adjacency[row];
                if (!adjacency.empty())
                    graph->insertGlobalIndices(partition_graph.row_gids[row], adjacency.size(), adjacency.data());
            }
        });
    agreed_phase("graph fillComplete", [&] { graph->fillComplete(); });

    using adapter_type = Zoltan2::TpetraRowGraphAdapter<graph_type>;
    using weight_type = typename adapter_type::scalar_t;
    std::optional<Teuchos::ParameterList> parameters;
    agreed_phase("partition parameters",
        [&]
        {
            parameters.emplace();
            parameters->set("algorithm", "parmetis");
            parameters->set("num_global_parts", nranks);
        });
    int weighted = partition_graph.row_weights.empty() ? 0 : 1, any_weighted = 0;
    int edge_weighted = partition_graph.row_edge_weights.empty() ? 0 : 1, any_edge_weighted = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &weighted, &any_weighted);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &edge_weighted, &any_edge_weighted);
    std::vector<weight_type> weights, edge_weights;
    agreed_phase("partition weights",
        [&]
        {
            if ((any_weighted && partition_graph.row_weights.size() != partition_graph.row_gids.size()) ||
                (any_edge_weighted && partition_graph.row_edge_weights.size() != partition_graph.row_gids.size()))
                throw std::invalid_argument("Missing weights for local partition graph rows.");
            if (any_weighted)
                parameters->set("imbalance_tolerance", partition_graph.imbalance_tolerance);
            weights.assign(partition_graph.row_weights.begin(), partition_graph.row_weights.end());
            if (any_edge_weighted)
            {
                edge_weights.reserve(graph->getLocalNumEntries());
                for (size_t row = 0; row < partition_graph.row_gids.size(); ++row)
                {
                    typename graph_type::local_inds_host_view_type columns;
                    graph->getLocalRowView(static_cast<LO>(row), columns);
                    for (size_t edge = 0; edge < columns.extent(0); ++edge)
                    {
                        const auto gid = graph->getColMap()->getGlobalElement(columns(edge));
                        const auto& neighbors = partition_graph.row_adjacency[row];
                        const auto found = std::find(neighbors.begin(), neighbors.end(), gid);
                        if (found == neighbors.end())
                            throw std::logic_error("Partition graph edge weight is missing.");
                        edge_weights.push_back(static_cast<weight_type>(
                            partition_graph.row_edge_weights.at(row).at(found - neighbors.begin())));
                    }
                }
            }
        });
    // The installed Zoltan2 pointer setters instantiate invalid code. Its
    // ParMETIS GraphModel consumes these documented virtual host views.
    class WeightedAdapter final : public adapter_type
    {
    public:
        WeightedAdapter(const Teuchos::RCP<const graph_type>& input, int vertex_count, int edge_count,
            std::vector<weight_type> vertices, std::vector<weight_type> edges)
            : adapter_type(input, vertex_count, edge_count), d_vertices(std::move(vertices)), d_edges(std::move(edges))
        {
        }
        void getVertexWeightsView(const weight_type*& values, int& stride, int index) const override
        {
            if (index != 0)
                throw std::out_of_range("Invalid partition vertex-weight index.");
            values = d_vertices.data();
            stride = 1;
        }
        void getEdgeWeightsView(const weight_type*& values, int& stride, int index) const override
        {
            if (index != 0)
                throw std::out_of_range("Invalid partition edge-weight index.");
            values = d_edges.data();
            stride = 1;
        }

    private:
        std::vector<weight_type> d_vertices, d_edges;
    };
    std::optional<WeightedAdapter> adapter;
    agreed_phase("partition adapter",
        [&] { adapter.emplace(graph, any_weighted, any_edge_weighted, std::move(weights), std::move(edge_weights)); });
    std::optional<Zoltan2::PartitioningProblem<adapter_type>> problem;
    agreed_phase("partition problem construction", [&] { problem.emplace(&*adapter, &*parameters, comm); });
    agreed_phase("partition solve", [&] { problem->solve(); });
    std::unordered_map<GO, int> local_parts;
    agreed_phase("partition result extraction",
        [&]
        {
            const auto* parts = problem->getSolution().getPartListView();
            if (!partition_graph.row_gids.empty() && !parts)
                throw std::runtime_error("Missing partition result rows.");
            local_parts.reserve(partition_graph.row_gids.size());
            for (size_t row = 0; row < partition_graph.row_gids.size(); ++row)
            {
                if (parts[row] < 0 || parts[row] >= nranks)
                    throw std::runtime_error("Invalid partition destination.");
                local_parts.emplace(partition_graph.row_gids[row], parts[row]);
            }
        });
    if (gather_root == -2)
        return local_parts;

    std::vector<GO> local_pairs, gathered_pairs;
    std::vector<int> counts, displacements;
    int my_pair_count = 0, total_pairs = 0;
    agreed_phase("partition result gather preparation",
        [&]
        {
            if (local_parts.size() > static_cast<size_t>(std::numeric_limits<int>::max()) / 2)
                throw std::overflow_error("Partition result count exceeds MPI int.");
            local_pairs.reserve(local_parts.size() * 2);
            for (const auto& [cell_gid, owner_rank] : local_parts)
            {
                local_pairs.push_back(cell_gid);
                local_pairs.push_back(static_cast<GO>(owner_rank));
            }
            my_pair_count = static_cast<int>(local_pairs.size());
            counts.assign(nranks, 0);
            displacements.assign(nranks, 0);
        });
    if (gather_root >= 0)
        MPI_Gather(&my_pair_count, 1, MPI_INT, counts.data(), 1, MPI_INT, gather_root, get_mpi_comm(*comm));
    else
        MPI_Allgather(&my_pair_count, 1, MPI_INT, counts.data(), 1, MPI_INT, get_mpi_comm(*comm));
    agreed_phase("partition result receive allocation",
        [&]
        {
            if (gather_root < 0 || myrank == gather_root)
                for (int r = 0; r < nranks; ++r)
                {
                    if (counts[r] < 0 || counts[r] > std::numeric_limits<int>::max() - total_pairs)
                        throw std::overflow_error("Partition gathered result exceeds MPI int.");
                    displacements[r] = total_pairs;
                    total_pairs += counts[r];
                }
            gathered_pairs.resize(static_cast<size_t>(std::max(total_pairs, 1)));
        });
    if (gather_root >= 0)
        MPI_Gatherv(local_pairs.data(), my_pair_count, mpi_go_type(), gathered_pairs.data(), counts.data(),
            displacements.data(), mpi_go_type(), gather_root, get_mpi_comm(*comm));
    else
        MPI_Allgatherv(local_pairs.data(), my_pair_count, mpi_go_type(), gathered_pairs.data(), counts.data(),
            displacements.data(), mpi_go_type(), get_mpi_comm(*comm));
    agreed_phase("gathered partition result extraction",
        [&]
        {
            local_parts.clear();
            if (gather_root >= 0 && myrank != gather_root)
                return;
            for (int pair = 0; pair + 1 < total_pairs; pair += 2)
                local_parts.emplace(gathered_pairs[pair], static_cast<int>(gathered_pairs[pair + 1]));
        });
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
        if (cell.type == MeshUtils::CellType::POLYHEDRON)
        {
            for (const auto face : mesh.faces(source_id))
            {
                Arr<NodeID> nodes;
                for (const auto node_id : mesh.face_nodes(face))
                    nodes.push_back(local_node_by_global.at(node_id));
                if (mesh.owner_cell(face) != source_id) std::ranges::reverse(nodes);
                cell.face_node_ids.push_back(std::move(nodes));
            }
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

    // An empty destination owns no local topology. Keep ordinary mesh input
    // validation intact and use the dedicated rank-local empty value.
    auto local_mesh = source_cell_lids.empty()
        ? UnstructuredMesh::empty_partition()
        : UnstructuredMesh(local_nodes, local_cells, local_boundaries,
            order.owned_cells.size(), order.owned_faces.size());

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

#include "parallel/NativeMeshDistribution.tcc"
