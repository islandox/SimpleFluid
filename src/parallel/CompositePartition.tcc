/** @file CompositePartition.tcc
 *  @brief Composite graph planning and native payload distribution.
 */
#pragma once
#include "parallel/MeshPartitioner.hh"
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <Teuchos_CommHelpers.hpp>

namespace SimpleFluid::composite_partition_detail
{
using ID = uint64_t;

// Planning traffic is self-contained and bounded before entering MPI calls.
struct Writer
{
    std::vector<char> bytes;
    template<class T> void value(T value)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto position = bytes.size();
        bytes.resize(position + sizeof(T));
        std::memcpy(bytes.data() + position, &value, sizeof(T));
    }
    template<class T> void vector(const std::vector<T>& values)
    {
        value<uint64_t>(values.size());
        for (const auto entry : values) value<T>(entry);
    }
};
struct Reader
{
    std::span<const char> bytes;
    template<class T> T value()
    {
        if (bytes.size() < sizeof(T)) throw std::invalid_argument("Truncated composite partition plan.");
        T result; std::memcpy(&result, bytes.data(), sizeof(T));
        bytes = bytes.subspan(sizeof(T)); return result;
    }
    template<class T> std::vector<T> vector()
    {
        const auto count = value<uint64_t>();
        if (count > bytes.size() / sizeof(T)) throw std::invalid_argument("Invalid composite plan vector length.");
        std::vector<T> result(static_cast<size_t>(count));
        for (auto& entry : result) entry = value<T>();
        return result;
    }
};

template<class Comm, class Function>
void collective_check(const Comm& comm, Function&& function)
{
    std::exception_ptr error;
    try { function(); } catch (...) { error = std::current_exception(); }
    int failed = error ? 1 : 0, any_failed = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
    if (any_failed)
    {
        if (error) std::rethrow_exception(error);
        throw std::invalid_argument("Composite partition operation failed on another rank.");
    }
}

template<class Comm>
std::vector<char> scatter_bytes(const Comm& comm, int root,
                               const std::vector<std::vector<char>>& outgoing)
{
    if (comm.getSize() == 1) return outgoing.at(0);
    std::vector<int> counts(static_cast<size_t>(comm.getSize())), offsets(counts.size());
    std::vector<char> flattened;
    collective_check(comm, [&]
    {
        if (comm.getRank() != root) return;
        if (outgoing.size() != counts.size()) throw std::logic_error("Invalid composite destination count.");
        size_t total = 0;
        for (size_t rank = 0; rank < counts.size(); ++rank)
        {
            const auto bytes = outgoing[rank].size();
            if (bytes > static_cast<size_t>(std::numeric_limits<int>::max()) - total)
                throw std::overflow_error("Composite partition packets exceed MPI count capacity.");
            counts[rank] = static_cast<int>(bytes); offsets[rank] = static_cast<int>(total); total += bytes;
        }
        flattened.reserve(total);
        for (const auto& packet : outgoing) flattened.insert(flattened.end(), packet.begin(), packet.end());
    });
    const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&comm);
    if (!mpi) throw std::invalid_argument("Multi-rank composite partition requires an MPI communicator.");
    const MPI_Comm raw = *mpi->getRawMpiComm();
    int count = 0;
    MPI_Scatter(counts.data(), 1, MPI_INT, &count, 1, MPI_INT, root, raw);
    std::vector<char> incoming;
    collective_check(comm, [&] { incoming.resize(static_cast<size_t>(count)); });
    MPI_Scatterv(flattened.data(), counts.data(), offsets.data(), MPI_CHAR,
                 incoming.data(), count, MPI_CHAR, root, raw);
    return incoming;
}

inline void write_fragment(Writer& writer, const CompositeRegionFragment& fragment)
{
    writer.value<uint64_t>(fragment.region);
    writer.value<int>(static_cast<int>(fragment.family));
    writer.value<int>(static_cast<int>(fragment.selection.index()));
    std::visit([&](const auto& selection)
    {
        using T = std::remove_cvref_t<decltype(selection)>;
        if constexpr (std::same_as<T, CompositeRegionFragment::OrthogonalBox>)
        {
            for (auto v : selection.begin) writer.value<uint64_t>(v);
            for (auto v : selection.extent) writer.value<uint64_t>(v);
        }
        else if constexpr (std::same_as<T, CompositeRegionFragment::ExtrudedProduct>)
        {
            writer.vector(selection.base_cells);
            writer.value<uint64_t>(selection.layer_begin); writer.value<uint64_t>(selection.layer_count);
        }
        else writer.vector(selection.cells);
    }, fragment.selection);
}
inline CompositeRegionFragment read_fragment(Reader& reader)
{
    CompositeRegionFragment fragment;
    fragment.region = reader.value<uint64_t>();
    fragment.family = static_cast<Meshes::RegionLayout::Family>(reader.value<int>());
    switch (reader.value<int>())
    {
    case 0:
    {
        CompositeRegionFragment::OrthogonalBox selection;
        for (auto& v : selection.begin) v = reader.value<uint64_t>();
        for (auto& v : selection.extent) v = reader.value<uint64_t>();
        fragment.selection = selection; break;
    }
    case 1:
    {
        CompositeRegionFragment::ExtrudedProduct selection;
        selection.base_cells = reader.vector<ID>();
        selection.layer_begin = reader.value<uint64_t>(); selection.layer_count = reader.value<uint64_t>();
        fragment.selection = std::move(selection); break;
    }
    case 2: fragment.selection = CompositeRegionFragment::ExplicitCells{reader.vector<ID>()}; break;
    default: throw std::invalid_argument("Invalid composite fragment kind.");
    }
    return fragment;
}

struct Unit
{
    CompositeRegionFragment fragment;
    std::vector<ID> cells;
};

inline std::vector<Unit> make_units(const Meshes::MultiRegionMesh& mesh,
                                   const CompositePartitionOptions& options, int ranks)
{
    using Family = Meshes::RegionLayout::Family;
    std::vector<Unit> units;
    const auto target = std::max<size_t>(1, std::min(options.target_cells_per_fragment,
        mesh.num_cells() / (4 * static_cast<size_t>(ranks))));
    for (size_t region = 0; region < mesh.regions().size(); ++region)
    {
        const auto& layout = mesh.region_layout(region);
        auto add = [&](auto selection, std::vector<ID> native_cells)
        {
            Unit unit{{region, layout.family, std::move(selection)}, std::move(native_cells)};
            for (auto& cell : unit.cells) cell = mesh.composite_cell(region, cell);
            units.push_back(std::move(unit));
        };
        if (options.policy != CompositePartitionOptions::Policy::TopologyAware || layout.family == Family::Explicit)
        {
            for (ID cell = 0; cell < layout.cells; ++cell)
                add(CompositeRegionFragment::ExplicitCells{{cell}}, {cell});
        }
        else if (layout.family == Family::Rectilinear || layout.family == Family::Cylindrical)
        {
            const auto n = layout.extents;
            std::array<size_t, 3> block{1, 1, 1};
            size_t count = 1;
            for (;;)
            {
                size_t chosen = 3;
                for (size_t axis = 0; axis < 3; ++axis)
                    if (block[axis] < n[axis] && count / block[axis] * (block[axis] + 1) <= target
                        && (chosen == 3 || block[axis] < block[chosen])) chosen = axis;
                if (chosen == 3) break;
                count = count / block[chosen] * (block[chosen] + 1); ++block[chosen];
            }
            for (size_t k = 0; k < n[2]; k += block[2])
                for (size_t j = 0; j < n[1]; j += block[1])
                    for (size_t i = 0; i < n[0]; i += block[0])
                    {
                        CompositeRegionFragment::OrthogonalBox box{{i, j, k},
                            {std::min(block[0], n[0]-i), std::min(block[1], n[1]-j), std::min(block[2], n[2]-k)}};
                        std::vector<ID> cells;
                        for (size_t z = k; z < k + box.extent[2]; ++z)
                            for (size_t y = j; y < j + box.extent[1]; ++y)
                                for (size_t x = i; x < i + box.extent[0]; ++x)
                                    cells.push_back(x + n[0] * (y + n[1] * z));
                        add(box, std::move(cells));
                    }
        }
        else
        {
            const auto base_cells = layout.extents[0], layers = layout.extents[2];
            const auto layer_chunk = std::min(layers, target), base_chunk = std::max<size_t>(1, target / layer_chunk);
            std::vector<bool> selected(base_cells, false);
            for (ID seed = 0; seed < base_cells; ++seed)
            {
                if (selected[seed]) continue;
                std::vector<ID> patch{seed}; selected[seed] = true;
                for (size_t cursor = 0; cursor < patch.size() && patch.size() < base_chunk; ++cursor)
                    std::visit([&](const auto& provider)
                    {
                        for (const auto face : provider.topology().cell_faces(patch[cursor]))
                        {
                            const auto owner = provider.topology().owner_cell(face);
                            const auto neighbor = provider.topology().neighbor_cell(face);
                            const auto other = owner == patch[cursor] ? neighbor : owner;
                            if (other < base_cells && !selected[other] && patch.size() < base_chunk)
                            { selected[other] = true; patch.push_back(other); }
                        }
                    }, mesh.regions()[region]);
                std::sort(patch.begin(), patch.end());
                for (size_t z = 0; z < layers; z += layer_chunk)
                {
                    const auto depth = std::min(layer_chunk, layers-z);
                    std::vector<ID> cells;
                    for (size_t k = z; k < z + depth; ++k)
                        for (auto base : patch) cells.push_back(k * base_cells + base);
                    add(CompositeRegionFragment::ExtrudedProduct{patch, z, depth}, std::move(cells));
                }
            }
        }
    }
    return units;
}
} // namespace SimpleFluid::composite_partition_detail

namespace SimpleFluid
{
template<TpetraTypePack Pack>
CompositePartitionPlan<Pack> MeshPartitioner<Pack>::make_plan(
    const CompositeMeshSource& source, const CompositePartitionOptions& options,
    const Teuchos::RCP<const comm_type>& comm)
{
    namespace detail = composite_partition_detail;
    using ID = detail::ID;
    if (comm.is_null()) throw std::invalid_argument("Composite partition requires a communicator.");
    const int rank = comm->getRank(), ranks = comm->getSize();
    int root = source.source_rank();
    int root_min = 0, root_max = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &root, &root_min);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &root, &root_max);
    if (root_min != root_max || root < 0 || root >= ranks)
        throw std::invalid_argument("Composite source rank must be valid and identical on all ranks.");
    detail::collective_check(*comm, [&]
    {
        if (!options.ghost_layers || !options.target_cells_per_fragment
            || !std::isfinite(options.imbalance_tolerance) || options.imbalance_tolerance < 1.0
            || static_cast<int>(options.policy) < 0 || static_cast<int>(options.policy) > 2)
            throw std::invalid_argument("Invalid composite partition options.");
        if (rank == root)
        {
            if (!source.mesh()) throw std::invalid_argument("Composite source rank has no mesh.");
            source.mesh()->validate_static();
            if (source.mesh()->num_cells() > static_cast<size_t>(std::numeric_limits<GO>::max())
                || source.mesh()->num_faces() > static_cast<size_t>(std::numeric_limits<GO>::max())
                || source.mesh()->num_nodes() > static_cast<size_t>(std::numeric_limits<GO>::max()))
                throw std::overflow_error("Composite IDs exceed global ordinal capacity.");
        }
    });
    std::array<uint64_t, 4> control{static_cast<uint64_t>(options.policy), options.ghost_layers,
                                   options.target_cells_per_fragment, options.require_balance};
    auto reference = control;
    double tolerance = options.imbalance_tolerance;
    Teuchos::broadcast(*comm, root, 4, reference.data());
    Teuchos::broadcast(*comm, root, 1, &tolerance);
    detail::collective_check(*comm, [&]
    {
        if (reference != control || tolerance != options.imbalance_tolerance)
            throw std::invalid_argument("Composite partition options differ across ranks.");
    });
    CompositePartitionPlan<Pack> result;
    result.d_comm = comm; result.d_options = options; result.d_source_rank = root;
    std::vector<detail::Unit> units;
    std::vector<ID> cell_unit;
    PartitionGraph graph;
    graph.imbalance_tolerance = options.imbalance_tolerance;
    detail::collective_check(*comm, [&]
    {
        if (rank != root) return;
        const auto& mesh = *source.mesh();
        const auto execution = mesh.acquire_execution_view();
        result.d_source = source.mesh(); result.d_source_epoch = mesh.geometry_epoch();
        units = detail::make_units(mesh, options, ranks);
        cell_unit.resize(mesh.num_cells());
        for (ID unit = 0; unit < units.size(); ++unit)
            for (auto cell : units[unit].cells) cell_unit[cell] = unit;
        if (options.policy == CompositePartitionOptions::Policy::ContiguousCanonical) return;
        graph.row_gids.resize(units.size()); std::iota(graph.row_gids.begin(), graph.row_gids.end(), GO{});
        graph.column_gids = graph.row_gids;
        graph.row_adjacency.resize(units.size()); graph.row_edge_weights.resize(units.size());
        graph.row_weights.reserve(units.size());
        for (ID unit = 0; unit < units.size(); ++unit)
        {
            std::map<GO,double> neighbors;
            for (auto cell : units[unit].cells)
                for (auto face : mesh.cell_faces(cell))
                {
                    const auto other = mesh.opposite_cell(face, cell);
                    if (other != Meshes::MultiRegionMesh::invalid_cell_id() && cell_unit[other] != unit)
                        neighbors[static_cast<GO>(cell_unit[other])] += 1.0;
                }
            for (const auto& [neighbor,weight] : neighbors)
            {
                graph.row_adjacency[unit].push_back(neighbor);
                graph.row_edge_weights[unit].push_back(weight);
            }
            graph.row_weights.push_back(static_cast<double>(units[unit].cells.size()));
        }
    });
    uint64_t unit_count = units.size();
    Teuchos::broadcast(*comm, root, 1, &unit_count);
    std::unordered_map<GO, int> assignment;
    if (ranks > 1 && unit_count >= static_cast<uint64_t>(ranks)
        && options.policy != CompositePartitionOptions::Policy::ContiguousCanonical)
    {
        // Separate source placement from graph-row placement. ParMETIS gets
        // balanced nonempty row sets even when all geometry starts on one rank.
        std::vector<std::vector<char>> graph_packets;
        detail::collective_check(*comm, [&]
        {
            if (rank != root) return;
            graph_packets.resize(ranks);
            for (int destination = 0; destination < ranks; ++destination)
            {
                detail::Writer writer;
                const auto first = unit_count / ranks * destination + std::min<uint64_t>(unit_count % ranks,destination);
                const auto count = unit_count / ranks + (static_cast<uint64_t>(destination) < unit_count % ranks);
                writer.value<uint64_t>(count);
                for (uint64_t row = first; row < first+count; ++row)
                {
                    writer.value<GO>(graph.row_gids[row]); writer.value<double>(graph.row_weights[row]);
                    writer.vector(graph.row_adjacency[row]);
                    writer.vector(graph.row_edge_weights[row]);
                }
                graph_packets[destination] = std::move(writer.bytes);
            }
        });
        const auto graph_packet = detail::scatter_bytes(*comm,root,graph_packets);
        detail::collective_check(*comm, [&]
        {
            detail::Reader reader{graph_packet};
            const auto rows = reader.template value<uint64_t>();
            graph.row_gids.clear(); graph.row_weights.clear(); graph.row_adjacency.clear(); graph.column_gids.clear();
            graph.row_edge_weights.clear();
            std::set<GO> columns;
            for (uint64_t row = 0; row < rows; ++row)
            {
                graph.row_gids.push_back(reader.template value<GO>());
                graph.row_weights.push_back(reader.template value<double>());
                graph.row_adjacency.push_back(reader.template vector<GO>());
                graph.row_edge_weights.push_back(reader.template vector<double>());
                columns.insert(graph.row_gids.back());
                columns.insert(graph.row_adjacency.back().begin(),graph.row_adjacency.back().end());
            }
            graph.column_gids.assign(columns.begin(),columns.end());
            if (!reader.bytes.empty()) throw std::invalid_argument("Trailing composite graph packet data.");
        });
        assignment = solve_partition_graph(graph, comm, root);
    }
    std::vector<std::vector<char>> outgoing;
    detail::collective_check(*comm, [&]
    {
        if (rank != root) return;
        const auto& mesh = *source.mesh();
        const auto execution = mesh.acquire_execution_view();
        std::vector<int> owners(mesh.num_cells());
        std::vector<int> unit_owners(units.size());
        if (options.policy == CompositePartitionOptions::Policy::ContiguousCanonical)
        {
            size_t begin = 0;
            for (int destination = 0; destination < ranks; ++destination)
            {
                const auto count = mesh.num_cells() / ranks + (static_cast<size_t>(destination) < mesh.num_cells() % ranks);
                std::fill(owners.begin()+begin, owners.begin()+begin+count, destination); begin += count;
            }
            for (size_t unit = 0; unit < units.size(); ++unit) unit_owners[unit] = owners[units[unit].cells.front()];
        }
        else
        {
            for (size_t unit = 0; unit < units.size(); ++unit)
            {
                const int owner = assignment.empty() ? static_cast<int>(unit % ranks) : assignment.at(static_cast<GO>(unit));
                if (owner < 0 || owner >= ranks) throw std::logic_error("Composite graph returned invalid owner rank.");
                unit_owners[unit] = owner;
                for (auto cell : units[unit].cells) owners[cell] = owner;
            }
        }
        outgoing.resize(ranks);
        auto destination_cells = std::make_shared<std::vector<std::vector<ID>>>(ranks);
        std::vector<std::vector<ID>> owned_by_rank(ranks);
        for (ID cell = 0; cell < owners.size(); ++cell) owned_by_rank[owners[cell]].push_back(cell);
        auto& balance = result.d_balance;
        balance.total_cells = mesh.num_cells();
        for (const auto& owned : owned_by_rank) balance.max_owned_cells = std::max<uint64_t>(balance.max_owned_cells,owned.size());
        for (const auto& unit : units) balance.largest_unit = std::max<uint64_t>(balance.largest_unit,unit.cells.size());
        const double average = static_cast<double>(mesh.num_cells()) / ranks;
        balance.measured_imbalance = balance.max_owned_cells / average;
        balance.minimum_imbalance_bound = std::max<double>((mesh.num_cells()+ranks-1)/ranks,balance.largest_unit) / average;
        balance.tolerance_satisfied = balance.measured_imbalance <= options.imbalance_tolerance;
        if (options.require_balance && balance.minimum_imbalance_bound > options.imbalance_tolerance)
            throw std::invalid_argument("Composite balance tolerance is impossible for indivisible partition units.");
        if (options.require_balance && !balance.tolerance_satisfied)
            throw std::runtime_error("Composite graph partition exceeds the requested balance tolerance.");
        for (int destination = 0; destination < ranks; ++destination)
        {
            auto& cells = (*destination_cells)[destination];
            cells = owned_by_rank[destination]; const auto owned_count = cells.size();
            std::unordered_set<ID> seen(cells.begin(), cells.end());
            std::vector<ID> frontier = cells, ghosts;
            for (size_t layer = 0; layer < options.ghost_layers; ++layer)
            {
                std::vector<ID> next;
                for (auto cell : frontier)
                    for (auto face : mesh.cell_faces(cell))
                    {
                        const auto other = mesh.opposite_cell(face, cell);
                        if (other != Meshes::MultiRegionMesh::invalid_cell_id() && seen.insert(other).second)
                        { ghosts.push_back(other); next.push_back(other); }
                    }
                frontier = std::move(next);
            }
            std::sort(ghosts.begin(), ghosts.end()); cells.insert(cells.end(), ghosts.begin(), ghosts.end());
            if (cells.size() > static_cast<size_t>(std::numeric_limits<LO>::max()))
                throw std::overflow_error("Composite local cell IDs exceed ordinal capacity.");
            std::set<ID> owned_faces, overlap_faces, nodes;
            std::vector<int> local_owners; local_owners.reserve(cells.size());
            for (auto cell : cells)
            {
                local_owners.push_back(owners[cell]);
                for (auto node : mesh.cell_nodes(cell)) nodes.insert(node);
                for (auto face : mesh.cell_faces(cell))
                {
                    (owners[mesh.owner_cell(face)] == destination ? owned_faces : overlap_faces).insert(face);
                    for (auto node : mesh.face_nodes(face)) nodes.insert(node);
                }
            }
            if (owned_faces.size() + overlap_faces.size() > static_cast<size_t>(std::numeric_limits<LO>::max())
                || nodes.size() > static_cast<size_t>(std::numeric_limits<LO>::max()))
                throw std::overflow_error("Composite local entity IDs exceed ordinal capacity.");
            std::vector<ID> faces(owned_faces.begin(), owned_faces.end()); faces.insert(faces.end(), overlap_faces.begin(), overlap_faces.end());
            detail::Writer packet;
            packet.value<uint64_t>(mesh.num_cells()); packet.value<uint64_t>(mesh.num_faces()); packet.value<uint64_t>(mesh.num_nodes());
            packet.value<uint64_t>(balance.max_owned_cells); packet.value<uint64_t>(balance.largest_unit);
            packet.value<double>(balance.measured_imbalance); packet.value<double>(balance.minimum_imbalance_bound);
            packet.value<int>(balance.tolerance_satisfied);
            packet.value<uint64_t>(owned_count); packet.value<uint64_t>(owned_faces.size());
            packet.vector(cells); packet.vector(local_owners); packet.vector(faces);
            packet.vector(std::vector<ID>(nodes.begin(), nodes.end()));
            const auto fragment_count = std::count(unit_owners.begin(), unit_owners.end(), destination);
            packet.value<uint64_t>(fragment_count);
            for (size_t unit = 0; unit < units.size(); ++unit)
                if (unit_owners[unit] == destination) detail::write_fragment(packet, units[unit].fragment);
            outgoing[destination] = std::move(packet.bytes);
        }
        result.d_destination_cells = std::move(destination_cells);
    });
    const auto incoming = detail::scatter_bytes(*comm, root, outgoing);
    detail::collective_check(*comm, [&]
    {
        detail::Reader reader{incoming};
        result.d_global_cells = reader.template value<ID>(); result.d_global_faces = reader.template value<ID>();
        result.d_global_nodes = reader.template value<ID>();
        result.d_balance.total_cells = result.d_global_cells;
        result.d_balance.max_owned_cells = reader.template value<uint64_t>();
        result.d_balance.largest_unit = reader.template value<uint64_t>();
        result.d_balance.measured_imbalance = reader.template value<double>();
        result.d_balance.minimum_imbalance_bound = reader.template value<double>();
        result.d_balance.tolerance_satisfied = reader.template value<int>() != 0;
        result.d_owned_cells = reader.template value<uint64_t>(); result.d_owned_faces = reader.template value<uint64_t>();
        result.d_cells = reader.template vector<ID>(); result.d_cell_owners = reader.template vector<int>();
        result.d_faces = reader.template vector<ID>(); result.d_nodes = reader.template vector<ID>();
        const auto fragments = reader.template value<uint64_t>();
        for (uint64_t i = 0; i < fragments; ++i) result.d_fragments.push_back(detail::read_fragment(reader));
        if (!reader.bytes.empty() || result.d_owned_cells > result.d_cells.size()
            || result.d_owned_faces > result.d_faces.size() || result.d_cell_owners.size() != result.d_cells.size())
            throw std::invalid_argument("Invalid composite partition metadata.");
    });
    return result;
}

template<TpetraTypePack Pack>
CompositePartition<Pack> MeshPartitioner<Pack>::distribute(
    const CompositeMeshSource& source, const CompositePartitionPlan<Pack>& plan)
{
    namespace detail = composite_partition_detail;
    const auto& comm = plan.communicator();
    const auto root = plan.source_rank();
    std::vector<std::vector<char>> outgoing;
    detail::collective_check(*comm, [&]
    {
        if (source.source_rank() != root) throw std::invalid_argument("Composite plan belongs to another source rank.");
        if (comm->getRank() != root) return;
        const auto pinned = plan.d_source.lock();
        if (!source.mesh() || pinned.get() != source.mesh().get()
            || source.mesh()->geometry_epoch() != plan.d_source_epoch)
            throw std::invalid_argument("Composite partition source changed after planning.");
        const auto execution = source.mesh()->acquire_execution_view();
        if (!plan.d_destination_cells) throw std::invalid_argument("Composite plan has no source distribution payload.");
        outgoing.reserve(plan.d_destination_cells->size());
        for (const auto& cells : *plan.d_destination_cells)
            outgoing.push_back(source.mesh()->serialize_partition(cells));
    });
    const auto incoming = detail::scatter_bytes(*comm, root, outgoing);
    std::shared_ptr<Meshes::MultiRegionMesh> geometry;
    detail::collective_check(*comm, [&]
    {
        geometry = Meshes::MultiRegionMesh::deserialize_partition(incoming, comm);
    });
    auto local_plan = plan;
    local_plan.d_destination_cells.reset();
    local_plan.d_source.reset();
    return CompositePartition<Pack>(std::move(geometry), std::move(local_plan));
}

template<TpetraTypePack Pack>
CompositePartition<Pack> MeshPartitioner<Pack>::partition(
    const CompositeMeshSource& source, const CompositePartitionOptions& options,
    const Teuchos::RCP<const comm_type>& comm)
{
    return distribute(source, make_plan(source, options, comm));
}
} // namespace SimpleFluid
