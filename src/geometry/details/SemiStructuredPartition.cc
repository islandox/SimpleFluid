/** @file SemiStructuredPartition.cc
 * @brief Distributed Zoltan2/ParMETIS partition of compact polygonal XY adjacency.
 */
#include "geometry/details/SemiStructuredPartition.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_TpetraRowGraphAdapter.hpp>
#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace SimpleFluid::detail
{
namespace
{
/** Fence local setup failures before the next graph/backend collective. */
void require_collective_success(const Teuchos::Comm<int>& comm, const std::exception_ptr& error)
{
    const int failed = error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
    if (any_failed)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::invalid_argument("SemiStructuredXY_Z XY graph partition failed on another rank.");
    }
}
} // namespace

std::vector<size_t> partition_semi_structured_xy(const Meshes::SemiStructMeshTopo& topology, size_t parts,
    bool allow_empty, const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    using Pack = DefaultTpetraTypes;
    using GO = Pack::global_ordinal_type;
    using Map = Pack::map_type;
    using Graph = Pack::graph_type;
    if (comm.is_null())
        throw std::invalid_argument("XY graph partition requires a communicator.");
    const size_t cells = topology.indexer().num_cells_per_layer;
    const size_t active_parts = std::min(cells, parts);
    std::exception_ptr error;
    std::vector<int> local_parts;
    try
    {
        if (!comm || parts == 0 || parts > static_cast<size_t>(comm->getSize()) || cells == 0 ||
            cells > static_cast<size_t>(std::numeric_limits<GO>::max()) ||
            active_parts > static_cast<size_t>(std::numeric_limits<int>::max()) || (!allow_empty && parts > cells))
            throw std::invalid_argument("Invalid compact SemiStructuredXY_Z XY graph partition dimensions.");
        local_parts.assign(cells, -1);
    }
    catch (...)
    {
        error = std::current_exception();
    }
    require_collective_success(*comm, error);
    if (active_parts == 1)
        return std::vector<size_t>(cells, 0);
    if (active_parts == cells)
    {
        // One vertex per active part is the unique nonempty partition up to labels.
        std::vector<size_t> result(cells);
        std::iota(result.begin(), result.end(), size_t{});
        return result;
    }

    const bool active = static_cast<size_t>(comm->getRank()) < active_parts;
    const auto graph_comm = comm->split(active ? 0 : -1, comm->getRank());
    error = {};
    if (active)
    {
        try
        {
            const size_t graph_rank = static_cast<size_t>(graph_comm->getRank());
            const size_t begin = cells / active_parts * graph_rank + std::min(cells % active_parts, graph_rank);
            const size_t count = cells / active_parts + (graph_rank < cells % active_parts ? 1 : 0);
            std::vector<GO> rows, columns;
            std::vector<std::vector<GO>> adjacency;
            size_t maximum_degree = 0;
            std::exception_ptr setup_error;
            try
            {
                rows.resize(count);
                adjacency.resize(count);
                for (size_t row = 0; row < count; ++row)
                {
                    const size_t cell = begin + row;
                    rows[row] = static_cast<GO>(cell);
                    auto& neighbors = adjacency[row];
                    for (const auto side :
                        topology.cell_side_faces(static_cast<Meshes::SemiStructMeshTopo::Ordinal>(cell)))
                    {
                        const auto& edge = topology.side_face(side);
                        const auto other = edge.owner == cell ? edge.neighbor : edge.owner;
                        if (other != Meshes::SemiStructMeshTopo::invalid_ordinal)
                            neighbors.push_back(static_cast<GO>(other));
                    }
                    std::sort(neighbors.begin(), neighbors.end());
                    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
                    maximum_degree = std::max(maximum_degree, neighbors.size());
                    columns.push_back(rows[row]);
                    columns.insert(columns.end(), neighbors.begin(), neighbors.end());
                }
                std::sort(columns.begin(), columns.end());
                columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
            }
            catch (...)
            {
                setup_error = std::current_exception();
            }
            require_collective_success(*graph_comm, setup_error);
            const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
            const auto row_map = Teuchos::rcp(new Map(invalid, rows.data(), rows.size(), GO{}, graph_comm));
            const auto column_map = Teuchos::rcp(new Map(invalid, columns.data(), columns.size(), GO{}, graph_comm));
            Teuchos::RCP<Graph> graph;
            setup_error = {};
            try
            {
                graph = Teuchos::rcp(new Graph(row_map, column_map, maximum_degree));
                for (size_t row = 0; row < count; ++row)
                    if (!adjacency[row].empty())
                        graph->insertGlobalIndices(rows[row], adjacency[row].size(), adjacency[row].data());
            }
            catch (...)
            {
                setup_error = std::current_exception();
            }
            require_collective_success(*graph_comm, setup_error);
            graph->fillComplete();
            using Adapter = Zoltan2::TpetraRowGraphAdapter<Graph>;
            std::unique_ptr<Adapter> adapter;
            Teuchos::ParameterList parameters;
            setup_error = {};
            try
            {
                adapter = std::make_unique<Adapter>(graph);
                parameters.set("algorithm", "parmetis");
                parameters.set("num_global_parts", static_cast<int>(active_parts));
                parameters.set("imbalance_tolerance", 1.1);
            }
            catch (...)
            {
                setup_error = std::current_exception();
            }
            require_collective_success(*graph_comm, setup_error);
            Zoltan2::PartitioningProblem<Adapter> problem(adapter.get(), &parameters, graph_comm);
            problem.solve();
            const auto* assignments = problem.getSolution().getPartListView();
            for (size_t row = 0; row < count; ++row)
                local_parts[begin + row] = assignments[row];
        }
        catch (...)
        {
            error = std::current_exception();
        }
    }
    require_collective_success(*comm, error);
    std::vector<int> assignments;
    error = {};
    try
    {
        assignments.assign(cells, -1);
    }
    catch (...)
    {
        error = std::current_exception();
    }
    require_collective_success(*comm, error);
    // Only compact XY labels are replicated, never an expanded 3D graph.
    for (size_t begin = 0; begin < cells;)
    {
        const size_t count = std::min(cells - begin, static_cast<size_t>(std::numeric_limits<int>::max()));
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, static_cast<int>(count), local_parts.data() + begin,
            assignments.data() + begin);
        begin += count;
    }
    std::vector<size_t> minima, labels, canonical, result;
    error = {};
    try
    {
        minima.assign(active_parts, cells);
        labels.resize(active_parts);
        canonical.resize(active_parts);
        result.resize(cells);
        for (size_t cell = 0; cell < cells; ++cell)
        {
            const auto part = assignments[cell];
            if (part < 0 || static_cast<size_t>(part) >= active_parts)
                throw std::runtime_error("SemiStructuredXY_Z graph backend returned an invalid XY part.");
            minima[static_cast<size_t>(part)] = std::min(minima[static_cast<size_t>(part)], cell);
        }
        if (!allow_empty && std::find(minima.begin(), minima.end(), cells) != minima.end())
            throw std::runtime_error("SemiStructuredXY_Z graph backend returned an empty requested XY part.");
        std::iota(labels.begin(), labels.end(), size_t{});
        std::stable_sort(labels.begin(), labels.end(), [&](size_t a, size_t b) { return minima[a] < minima[b]; });
        for (size_t label = 0; label < labels.size(); ++label)
            canonical[labels[label]] = label;
        for (size_t cell = 0; cell < cells; ++cell)
            result[cell] = canonical[static_cast<size_t>(assignments[cell])];
    }
    catch (...)
    {
        error = std::current_exception();
    }
    require_collective_success(*comm, error);
    return result;
}
} // namespace SimpleFluid::detail
