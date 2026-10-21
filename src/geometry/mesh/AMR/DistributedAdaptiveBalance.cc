/** @file DistributedAdaptiveBalance.cc
 * @brief Sparse objective evaluation and measured-weight ownership proposals.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveBalance.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
void observation(double value)
{
    if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("Measured adaptive work must be finite and nonnegative.");
}
void add_bytes(std::uint64_t& total, std::uint64_t bytes)
{
    if (bytes > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("Measured migration byte count overflow.");
    total += bytes;
}
} // namespace
double AdaptiveMeasuredWork::total() const
{
    if (std::uint64_t(unit) > 1)
        throw std::invalid_argument("Unknown measured work unit.");
    observation(reconstruction);
    observation(interface);
    observation(population);
    auto result = reconstruction + interface + population;
    observation(result);
    return result;
}
void AdaptiveWorkRecorder::observe(
    const AdaptiveCellKey& key, std::uint64_t revision, double reconstruction, double interface, double population)
{
    observation(reconstruction);
    observation(interface);
    observation(population);
    auto found = records_.find(key);
    Record record;
    if (found != records_.end())
        record = found->second;
    else
    {
        record.revision = revision;
        record.work.unit = unit_;
    }
    if (record.revision != revision || record.samples == std::numeric_limits<std::uint64_t>::max())
        throw std::invalid_argument("Measured work revision differs or sample count exhausted.");
    record.work.reconstruction += reconstruction;
    record.work.interface += interface;
    record.work.population += population;
    (void) record.work.total();
    ++record.samples;
    records_.insert_or_assign(key, std::move(record));
}
void AdaptiveWorkRecorder::record_payload(const AdaptiveCellKey& key, std::uint64_t revision, std::uint64_t bytes)
{
    if (!bytes)
        throw std::invalid_argument("Migration payload must name actual nonzero encoded bytes.");
    auto found = records_.find(key);
    Record record;
    if (found != records_.end())
        record = found->second;
    else
    {
        record.revision = revision;
        record.work.unit = unit_;
    }
    if (record.revision != revision)
        throw std::invalid_argument("Measured payload revision differs.");
    record.work.migration_bytes = bytes;
    records_.insert_or_assign(key, std::move(record));
}
AdaptiveMeasuredWork AdaptiveWorkRecorder::snapshot(const AdaptiveCellKey& key, std::uint64_t revision) const
{
    auto found = records_.find(key);
    if (found == records_.end() || found->second.revision != revision || !found->second.samples ||
        !found->second.work.migration_bytes)
        throw std::invalid_argument("Measured work snapshot is missing, stale or lacks an encoded payload.");
    return found->second.work;
}
namespace
{
using namespace adaptive_detail;
std::uint64_t sum_bytes(std::uint64_t local, const Teuchos::Comm<int>& comm)
{
    collective(comm,
        [&]
        {
            if (local > std::numeric_limits<std::uint64_t>::max() / std::uint64_t(comm.getSize()))
                throw std::overflow_error("Global adaptive metric exceeds conservative sum bound.");
        });
    std::uint64_t result = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}
AdaptiveBalanceMetrics metrics(double local_compute, std::uint64_t cut, std::uint64_t moved, std::uint64_t moved_cells,
    const Teuchos::Comm<int>& comm, const AdaptiveBalanceOptions& options)
{
    AdaptiveBalanceMetrics result;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local_compute, &result.total_compute);
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local_compute, &result.maximum_compute);
    result.mean_compute = result.total_compute / comm.getSize();
    result.imbalance = result.mean_compute ? result.maximum_compute / result.mean_compute : 0;
    result.communication_cut_bytes = sum_bytes(cut, comm);
    result.moved_bytes = sum_bytes(moved, comm);
    result.moved_cells = sum_bytes(moved_cells, comm);
    collective(comm,
        [&]
        {
            observation(result.total_compute);
            result.objective = options.compute_weight * result.maximum_compute +
                               options.communication_weight * double(result.communication_cut_bytes) +
                               options.migration_weight * double(result.moved_bytes);
            observation(result.objective);
        });
    return result;
}
std::vector<int> heuristic(std::span<const AdaptiveBalanceCell> cells, double local_compute,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, const AdaptiveBalanceOptions& options)
{
    std::vector<double> loads, budgets;
    std::vector<int> result;
    std::vector<std::size_t> order;
    collective(*comm,
        [&]
        {
            Budget allocation{options.byte_budget};
            allocation.add(std::size_t(comm->getSize()) * 2 * sizeof(double));
            allocation.add(cells.size() * (sizeof(int) + sizeof(std::size_t)));
            loads.resize(comm->getSize());
            budgets.resize(comm->getSize());
            result.assign(cells.size(), comm->getRank());
            order.resize(cells.size());
        });
    Teuchos::gatherAll(*comm, 1, &local_compute, int(loads.size()), loads.data());
    collective(*comm,
        [&]
        {
            const double mean = std::accumulate(loads.begin(), loads.end(), 0.0) / loads.size();
            observation(mean);
            // Compact rank-only allocation of excess work. No cell assignments are
            // gathered. Indivisible cells are chosen locally against these budgets.
            for (int donor = 0; donor < comm->getSize(); ++donor)
            {
                double excess = std::max(0.0, loads[donor] - mean);
                for (int receiver = 0; receiver < comm->getSize() && excess > 0; ++receiver)
                {
                    if (receiver == donor)
                        continue;
                    double transfer = std::min(excess, std::max(0.0, mean - loads[receiver]));
                    loads[receiver] += transfer;
                    loads[donor] -= transfer;
                    excess -= transfer;
                    if (donor == comm->getRank())
                        budgets[receiver] += transfer;
                }
            }
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                [&](auto a, auto b) { return cells[a].work.total() > cells[b].work.total(); });
            for (auto index : order)
            {
                const double work = cells[index].work.total();
                auto target = std::max_element(budgets.begin(), budgets.end());
                if (target == budgets.end() || *target <= 0 || work <= 0 || work > 2 * (*target))
                    continue;
                result[index] = int(target - budgets.begin());
                *target = std::max(0.0, *target - work);
            }
        });
    return result;
}
} // namespace
AdaptiveBalancePlan balance_owned_cells(std::span<const AdaptiveBalanceCell> cells,
    std::span<const AdaptiveBalanceEdge> edges, AdaptiveBalanceSource source,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, AdaptiveBalanceOptions options,
    std::optional<std::span<const int>> external)
{
    using namespace adaptive_detail;
    if (comm.is_null())
        throw std::invalid_argument("Adaptive balance needs a communicator.");
    std::array<std::uint64_t, 9> semantic{options.byte_budget, std::uint64_t(options.work_unit),
        std::bit_cast<std::uint64_t>(options.compute_weight),
        std::bit_cast<std::uint64_t>(options.communication_weight),
        std::bit_cast<std::uint64_t>(options.migration_weight),
        std::bit_cast<std::uint64_t>(options.minimum_relative_improvement), source.topology, source.partition,
        std::uint64_t(external.has_value())},
        root = semantic;
    Teuchos::broadcast(*comm, 0, int(root.size()), root.data());
    std::vector<AdaptiveOwner> registrations, edge_registrations;
    std::vector<AdaptiveBytes> neighbors;
    double local_compute = 0;
    AdaptiveBalancePlan plan;
    plan.source = std::move(source);
    plan.objective_options = options;
    collective(*comm,
        [&]
        {
            if (root != semantic || !options.byte_budget || std::uint64_t(options.work_unit) > 1 ||
                !plan.source.identity || !std::isfinite(options.minimum_relative_improvement) ||
                options.minimum_relative_improvement < 0 || options.minimum_relative_improvement >= 1)
                throw std::invalid_argument("Invalid or rank-divergent adaptive balance semantics.");
            observation(options.compute_weight);
            observation(options.communication_weight);
            observation(options.migration_weight);
            if (options.compute_weight == 0 && options.communication_weight == 0 && options.migration_weight == 0)
                throw std::invalid_argument("Adaptive objective needs a positive coefficient.");
            if (external && external->size() != cells.size())
                throw std::invalid_argument("Balance proposal has wrong local extent.");
            Budget allocation{options.byte_budget};
            for (std::size_t i = 0; i < cells.size(); ++i)
            {
                const auto& cell = cells[i];
                if ((i && !(cells[i - 1].key < cell.key)) || cell.work.unit != options.work_unit ||
                    !cell.work.migration_bytes)
                    throw std::invalid_argument("Balance cells are unsorted, unmeasured or use different units.");
                local_compute += cell.work.total();
                auto key = encode_adaptive_key(cell.key);
                allocation.add(
                    3 * key.size() + sizeof(AdaptiveOwner) + sizeof(AdaptiveCellKey) + sizeof(std::uint64_t));
                registrations.push_back({std::move(key), comm->getRank(), cell.revision});
                plan.keys.push_back(cell.key);
                plan.revisions.push_back(cell.revision);
            }
            observation(local_compute);
            std::set<AdaptiveBytes> unique;
            std::set<AdaptiveCellKey> local_keys(plan.keys.begin(), plan.keys.end());
            for (const auto& edge : edges)
            {
                if (edge.identity.empty() || edge.first == edge.second || edge.first_owner != comm->getRank() ||
                    edge.second_owner < 0 || edge.second_owner >= comm->getSize() || !local_keys.contains(edge.first) ||
                    !unique.insert(edge.identity).second)
                    throw std::invalid_argument("Invalid canonical owned balance adjacency.");
                auto key = encode_adaptive_key(edge.second);
                allocation.add(key.size() + 2 * edge.identity.size() + sizeof(AdaptiveOwner));
                neighbors.push_back(std::move(key));
                edge_registrations.push_back({edge.identity, comm->getRank(), 0});
            }
        });
    DistributedAncestryDirectory directory(registrations, comm, options.byte_budget);
    DistributedAncestryDirectory edge_directory(edge_registrations, comm, options.byte_budget);
    auto remote = directory.lookup(neighbors);
    std::uint64_t old_cut = 0;
    collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < edges.size(); ++i)
            {
                if (remote[i].owner != edges[i].second_owner)
                    throw std::invalid_argument("Balance edge names a stale neighbor owner.");
                if (edges[i].first_owner != edges[i].second_owner)
                    add_bytes(old_cut, edges[i].communication_bytes);
            }
        });
    plan.before = metrics(local_compute, old_cut, 0, 0, *comm, options);
    plan.before.wire_bytes_exact = true;
    std::vector<int> proposal;
    if (external)
    {
        collective(*comm,
            [&]
            {
                Budget allocation{options.byte_budget};
                allocation.add(cells.size() * sizeof(int));
                proposal.assign(external->begin(), external->end());
            });
    }
    else
        proposal = heuristic(cells, local_compute, comm, options);
    std::vector<AdaptiveOwner> assignments;
    std::vector<OwnerPacket> load_packets;
    std::uint64_t moved = 0, moved_cells = 0;
    collective(*comm,
        [&]
        {
            Budget allocation{options.byte_budget};
            std::map<int, double> loads;
            for (std::size_t i = 0; i < cells.size(); ++i)
            {
                auto destination = proposal[i];
                if (destination < 0 || destination >= comm->getSize())
                    throw std::invalid_argument("Invalid proposed owner.");
                auto key = encode_adaptive_key(cells[i].key);
                allocation.add(key.size() + sizeof(AdaptiveOwner));
                assignments.push_back({std::move(key), comm->getRank(), std::uint64_t(destination)});
                loads[destination] += cells[i].work.total();
                if (destination != comm->getRank())
                {
                    add_bytes(moved, cells[i].work.migration_bytes);
                    add_bytes(moved_cells, 1);
                }
            }
            for (const auto& [rank, work] : loads)
            {
                observation(work);
                allocation.packet(8);
                Writer w;
                w.word(std::bit_cast<std::uint64_t>(work));
                load_packets.push_back({rank, std::move(w.data)});
            }
        });
    DistributedAncestryDirectory assignment_directory(assignments, comm, options.byte_budget);
    auto after_owners = assignment_directory.lookup(neighbors);
    auto transferred = exchange_owner_packets(load_packets, comm, options.byte_budget);
    double proposed_compute = 0;
    std::uint64_t new_cut = 0;
    collective(*comm,
        [&]
        {
            Budget allocation{options.byte_budget};
            allocation.add(cells.size() * (sizeof(AdaptiveCellKey) + sizeof(std::size_t) + 3 * sizeof(void*)));
            std::map<AdaptiveCellKey, std::size_t> indices;
            for (std::size_t i = 0; i < cells.size(); ++i)
                indices.emplace(cells[i].key, i);
            for (const auto& packet : transferred.packets)
            {
                Reader r{packet.bytes};
                auto work = std::bit_cast<double>(r.word());
                r.done();
                observation(work);
                proposed_compute += work;
            }
            observation(proposed_compute);
            for (std::size_t i = 0; i < edges.size(); ++i)
            {
                if (after_owners[i].owner != edges[i].second_owner ||
                    after_owners[i].revision >= std::uint64_t(comm->getSize()))
                    throw std::invalid_argument("Foreign proposed neighbor assignment.");
                if (proposal[indices.at(edges[i].first)] != int(after_owners[i].revision))
                    add_bytes(new_cut, edges[i].communication_bytes);
            }
        });
    plan.proposed = metrics(proposed_compute, new_cut, moved, moved_cells, *comm, options);
    collective(*comm,
        [&]
        {
            Budget allocation{options.byte_budget};
            allocation.add(cells.size() * sizeof(int));
            plan.accepted =
                plan.proposed.objective < plan.before.objective * (1 - options.minimum_relative_improvement);
            plan.localdestinations =
                plan.accepted ? std::move(proposal) : std::vector<int>(cells.size(), comm->getRank());
            plan.final = plan.accepted ? plan.proposed : plan.before;
        });
    return plan;
}
AdaptiveBalancePlan qualify_balance_wire_bytes(const AdaptiveBalancePlan& input, std::uint64_t local_bytes,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, AdaptiveBalanceOptions options)
{
    if (comm.is_null())
        throw std::invalid_argument("Packed balance qualification needs a communicator.");
    AdaptiveBalancePlan plan;
    adaptive_detail::collective(*comm, [&] { plan = input; });
    std::array<std::uint64_t, 5> semantic{std::uint64_t(plan.accepted),
        std::bit_cast<std::uint64_t>(options.compute_weight),
        std::bit_cast<std::uint64_t>(options.communication_weight),
        std::bit_cast<std::uint64_t>(options.migration_weight),
        std::bit_cast<std::uint64_t>(options.minimum_relative_improvement)},
        root = semantic;
    Teuchos::broadcast(*comm, 0, int(root.size()), root.data());
    adaptive_detail::collective(*comm,
        [&]
        {
            if (root != semantic || options != plan.objective_options || !plan.source.identity ||
                plan.localdestinations.size() != plan.keys.size())
                throw std::invalid_argument("Packed balance qualification has inconsistent source or objective.");
            observation(options.compute_weight);
            observation(options.communication_weight);
            observation(options.migration_weight);
            if (!std::isfinite(options.minimum_relative_improvement) || options.minimum_relative_improvement < 0 ||
                options.minimum_relative_improvement >= 1)
                throw std::invalid_argument("Invalid packed objective improvement bound.");
        });
    const auto actual = sum_bytes(local_bytes, *comm);
    adaptive_detail::collective(*comm,
        [&]
        {
            plan.proposed.moved_bytes = actual;
            plan.proposed.wire_bytes_exact = true;
            plan.proposed.objective = options.compute_weight * plan.proposed.maximum_compute +
                                      options.communication_weight * double(plan.proposed.communication_cut_bytes) +
                                      options.migration_weight * double(actual);
            observation(plan.proposed.objective);
            plan.accepted = plan.accepted && plan.proposed.objective <
                                                 plan.before.objective * (1 - options.minimum_relative_improvement);
            if (plan.accepted)
                plan.final = plan.proposed;
            else
            {
                std::fill(plan.localdestinations.begin(), plan.localdestinations.end(), comm->getRank());
                plan.final = plan.before;
            }
        });
    return plan;
}

} // namespace SimpleFluid::Meshes::AMR
