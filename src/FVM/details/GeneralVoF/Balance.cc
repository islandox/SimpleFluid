/** @file Balance.cc
 * @brief Sparse donor fan-out qualification and initial phase-only migration.
 */
#include "FVM/GeneralVoFBalance.hh"
#include "FVM/GeneralVoFWork.hh"
#include "FVM/details/GeneralVoF/BalanceForecast.hh"
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
namespace SimpleFluid::VoF
{
namespace
{
namespace AMR = Meshes::AMR;
namespace wire = AMR::adaptive_detail;
using Key = AMR::AdaptiveCellKey;
using Clock = std::chrono::steady_clock;
void seconds(double value)
{
    if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("Initial VoF balance requires finite nonnegative measured seconds.");
}
void add(std::uint64_t& total, std::uint64_t bytes)
{
    if (bytes > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("Initial VoF balance accounting overflow.");
    total += bytes;
}
void charge(wire::Budget& budget, std::size_t count, std::size_t width)
{
    if (width && count > std::numeric_limits<std::size_t>::max() / width)
        throw std::invalid_argument("Initial VoF balance allocation overflow.");
    budget.add(count * width);
}
double elapsed(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}
double maximum(double local, const Teuchos::Comm<int>& comm)
{
    double result = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local, &result);
    return result;
}
std::uint64_t sum(std::uint64_t local, const Teuchos::Comm<int>& comm)
{
    wire::collective(comm,
        [&]
        {
            if (local > std::numeric_limits<std::uint64_t>::max() / std::uint64_t(comm.getSize()))
                throw std::overflow_error("Initial VoF balance global byte sum overflow.");
        });
    std::uint64_t result = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}
void retain(GeneralVoFBalanceProposal& result, GeneralVoFBalanceRejection reason, int rank)
{
    result.rejection = reason;
    result.native_plan.accepted = false;
    std::fill(result.native_plan.localdestinations.begin(), result.native_plan.localdestinations.end(), rank);
    result.native_plan.final = result.native_plan.before;
}
std::size_t plan_storage(const AMR::AdaptiveBalancePlan& plan, std::size_t limit)
{
    wire::Budget storage{limit};
    charge(storage, plan.keys.capacity(), sizeof(Key));
    charge(storage, plan.revisions.capacity(), sizeof(std::uint64_t));
    charge(storage, plan.localdestinations.capacity(), sizeof(int));
    for (const auto& key : plan.keys)
        storage.add(AMR::encode_adaptive_key(key).capacity());
    return storage.used;
}
} // namespace
GeneralVoFBalanceProposal plan_partition_proposal(std::shared_ptr<const GeneralVoF> current,
    const GeneralVoFPreparedTransport& observed, GeneralVoFBalanceBinding binding, GeneralVoFBalanceOptions options)
{
    if (!current)
        throw std::invalid_argument("Initial VoF balance needs an immutable current.");
    const auto& hierarchy = *current->hierarchy();
    const auto comm = hierarchy.communicator();
    const int rank = comm->getRank();
    std::uint64_t binding_value = std::uint64_t(binding), root_binding = binding_value;
    Teuchos::broadcast(*comm, 0, 1, &root_binding);
    wire::collective(*comm,
        [&]
        {
            if (binding_value != root_binding ||
                binding_value > std::uint64_t(GeneralVoFBalanceBinding::AcceptedSuccessor))
                throw std::invalid_argument("Invalid or rank-divergent VoF balance observation binding.");
        });
    if (binding == GeneralVoFBalanceBinding::ObservedSource)
        current->validate_prepared_transport_source(observed);
    else
        current->validate_prepared_transport_candidate(observed);
    const auto limit =
        std::min({options.objective.byte_budget, current->options().byte_budget, hierarchy.options().byte_budget});
    wire::Budget retained{limit};
    GeneralVoFBalanceProposal result;
    result.binding = binding;
    result.current_phase_identity = current->accepted_identity();
    result.current_phase_control = current->accepted_control();
    result.observed_source_identity = observed.source_identity;
    result.observed_source_control = observed.source_control;
    result.observed_work = observed.balance_work;
    int classification_changed = 0, any_classification_changed = 0;
    std::vector<AMR::AdaptiveBalanceCell> cells;
    std::vector<AMR::AdaptiveBalanceEdge> edges;
    std::map<Key, std::size_t> indices;
    // Validate the public shared semantics before replacing the user cap with
    // an internal common lease. Rank-local current/native limits may differ.
    const std::array<std::uint64_t, 7> semantics{options.objective.byte_budget,
        std::uint64_t(options.objective.work_unit), std::bit_cast<std::uint64_t>(options.objective.compute_weight),
        std::bit_cast<std::uint64_t>(options.objective.communication_weight),
        std::bit_cast<std::uint64_t>(options.objective.migration_weight),
        std::bit_cast<std::uint64_t>(options.objective.minimum_relative_improvement),
        std::uint64_t(options.receiver_estimator)};
    auto root_semantics = semantics;
    Teuchos::broadcast(*comm, 0, int(root_semantics.size()), root_semantics.data());
    wire::collective(*comm,
        [&]
        {
            if (semantics != root_semantics || !options.objective.byte_budget ||
                options.objective.work_unit != AMR::AdaptiveWorkUnit::Seconds ||
                options.objective.communication_weight != 0 || options.objective.migration_weight != 0 ||
                !std::isfinite(options.objective.compute_weight) || options.objective.compute_weight <= 0 ||
                !std::isfinite(options.objective.minimum_relative_improvement) ||
                options.objective.minimum_relative_improvement < 0 ||
                options.objective.minimum_relative_improvement >= 1 ||
                std::uint64_t(options.receiver_estimator) > std::uint64_t(GeneralVoFReceiverEstimator::Maximum))
                throw std::invalid_argument("Initial VoF balance supports seconds and compute-only scoring.");
            if (!observed.balance_work || current->arithmetic_mode() != ArithmeticMode::Ordinary ||
                observed.backend != GeneralVoFTransportBackend::SweptPartition ||
                observed.family != GeneralVoFTransportFamily::InterfaceLocalSwept)
                throw std::invalid_argument("Initial VoF balance needs complete interface-local work capture.");
            const auto& sample = *observed.balance_work;
            if (!sample.fully_deduplicated || sample.source_identity != observed.source_identity ||
                sample.source_control != observed.source_control || sample.mesh_identity != observed.mesh_identity ||
                sample.topology_identity != observed.topology_identity ||
                sample.observation != hierarchy.observation() || sample.communicator != comm.getRawPtr() ||
                sample.time_stage != observed.time_stage || sample.timestep != observed.timestep ||
                sample.arithmetic_mode != observed.arithmetic_mode || sample.backend != observed.backend ||
                sample.cells.size() != current->owned_order().size())
                throw std::invalid_argument("Initial VoF work capture is stale, incomplete or foreign.");
            charge(retained, sample.cells.size(),
                2 * sizeof(AMR::AdaptiveBalanceCell) + sizeof(std::size_t) + sizeof(Key) + 6 * sizeof(void*));
            cells.reserve(sample.cells.size());
            auto owned = hierarchy.owned().begin();
            for (std::size_t i = 0; i < sample.cells.size(); ++i, ++owned)
            {
                const auto& row = sample.cells[i];
                const auto& key = owned->first;
                if (row.key != key || row.owner != rank || row.revision != hierarchy.owned().at(key).revision ||
                    row.calls > 1 || (row.calls == 0 && row.reconstruction.seconds != 0))
                    throw std::invalid_argument("Initial VoF work capture owned ordering or deduplication differs.");
                seconds(row.reconstruction.seconds);
                const auto encoded = AMR::encode_adaptive_batch(
                    std::span<const AMR::AdaptiveLeaf>(&hierarchy.owned().at(key), 1), {}, limit - retained.used);
                charge(retained, AMR::encode_adaptive_key(key).size(), 3);
                AMR::AdaptiveBalanceCell cell;
                cell.key = key;
                cell.revision = row.revision;
                cell.work.unit = AMR::AdaptiveWorkUnit::Seconds;
                cell.work.reconstruction = row.reconstruction.seconds;
                cell.work.migration_bytes = encoded.size();
                indices.emplace(key, i);
                cells.push_back(std::move(cell));
            }
            std::map<AMR::AdaptiveBytes, const AMR::AdaptiveTile*> canonical;
            for (const auto& tile : current->tiles())
                if (!tile.exterior && tile.second && tile.first_owner == rank)
                {
                    retained.add(tile.identity.size() + sizeof(void*) + 6 * sizeof(void*));
                    if (!canonical.emplace(tile.identity, &tile).second)
                        throw std::invalid_argument("Initial VoF canonical current contains duplicate faces.");
                }
            if (sample.faces.size() != canonical.size())
                throw std::invalid_argument("Initial VoF work capture omits canonical faces.");
            std::set<AMR::AdaptiveBytes> seen;
            charge(retained, sample.faces.size(), 2 * sizeof(AMR::AdaptiveBalanceEdge) + 8 * sizeof(void*));
            edges.reserve(sample.faces.size());
            for (const auto& face : sample.faces)
            {
                auto found = canonical.find(face.identity);
                if (found == canonical.end() || !seen.insert(face.identity).second)
                    throw std::invalid_argument("Initial VoF work capture duplicates or invents a canonical face.");
                const auto& tile = *found->second;
                if (face.first.key != tile.first || face.second.key != *tile.second || face.first.owner != rank ||
                    face.second.owner != tile.second_owner ||
                    face.first.revision != hierarchy.owned().at(face.first.key).revision)
                    throw std::invalid_argument("Initial VoF work face provenance differs from current.");
                seconds(face.setup.seconds);
                seconds(face.aggregate.seconds);
                cells.at(indices.at(face.first.key)).work.interface += face.setup.seconds + face.aggregate.seconds;
                charge(retained, face.identity.size(), 3);
                charge(retained,
                    AMR::encode_adaptive_key(face.first.key).size() + AMR::encode_adaptive_key(face.second.key).size(),
                    2);
                edges.push_back({face.identity, face.first.key, face.second.key, rank, face.second.owner, 0});
            }
            if (binding == GeneralVoFBalanceBinding::AcceptedSuccessor)
            {
                charge(retained, current->owned_order().size(), sizeof(GeneralVoFPhaseGeometry));
                const auto declarations = current->phase_geometry_declarations();
                const auto& phases = current->phase_volumes();
                if (declarations.size() != current->owned_order().size() || phases.size() != declarations.size())
                    throw std::invalid_argument("Current VoF classification layout differs from owned keys.");
                for (std::size_t i = 0; i < declarations.size(); ++i)
                {
                    const auto& row = sample.cells.at(indices.at(current->owned_order()[i]));
                    const bool needs_mixed =
                        declarations[i] == GeneralVoFPhaseGeometry::Infer && phases[i].gas > 0 && phases[i].carrier > 0;
                    if (needs_mixed && (row.calls != 1 || !row.mixed))
                        classification_changed = 1;
                }
            }
        });
    const auto& sample = *observed.balance_work;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &classification_changed, &any_classification_changed);
    if (any_classification_changed)
    {
        wire::collective(*comm,
            [&]
            {
                result.native_plan.source = {
                    hierarchy.identity(), hierarchy.topology_generation(), hierarchy.partition_generation()};
                result.native_plan.objective_options = options.objective;
                charge(retained, sample.cells.size(), sizeof(Key) + sizeof(std::uint64_t) + sizeof(int));
                result.native_plan.keys.reserve(sample.cells.size());
                result.native_plan.revisions.reserve(sample.cells.size());
                result.native_plan.localdestinations.reserve(sample.cells.size());
                for (const auto& row : sample.cells)
                {
                    retained.add(AMR::encode_adaptive_key(row.key).capacity());
                    result.native_plan.keys.push_back(row.key);
                    result.native_plan.revisions.push_back(row.revision);
                    result.native_plan.localdestinations.push_back(rank);
                }
            });
        retain(result, GeneralVoFBalanceRejection::ObservationClassificationChanged, rank);
        return result;
    }
    const std::uint64_t local_remaining = limit - retained.used;
    std::uint64_t common_remaining = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_remaining, &common_remaining);
    wire::collective(*comm,
        [&]
        {
            if (!common_remaining)
                throw std::invalid_argument("Initial VoF balance has no remaining generic planner lease.");
        });
    auto planner_options = options.objective;
    planner_options.byte_budget = std::size_t(common_remaining);
    result.native_plan = AMR::balance_owned_cells(cells, edges,
        {hierarchy.identity(), hierarchy.topology_generation(), hierarchy.partition_generation()}, comm,
        planner_options);
    std::size_t native_plan_bytes = 0;
    wire::collective(*comm,
        [&]
        {
            native_plan_bytes = plan_storage(result.native_plan, limit - retained.used);
            retained.add(native_plan_bytes);
        });
    auto forecast_options = options;
    forecast_options.objective.byte_budget = limit - retained.used;
    const auto forecast = general_vof_detail::forecast_partition_work(
        sample, result.native_plan.localdestinations, comm, forecast_options);
    result.measured_before = forecast.measured_before;
    result.forecast_proposed = forecast.forecast_proposed;
    if (!result.native_plan.accepted)
    {
        retain(result, GeneralVoFBalanceRejection::NoCoreImprovement, rank);
        return result;
    }
    if (forecast.incomplete_receiver_model)
    {
        retain(result, GeneralVoFBalanceRejection::IncompleteReceiverModel, rank);
        return result;
    }
    if (!(result.forecast_proposed.maximum_compute_seconds <
            result.measured_before.maximum_compute_seconds * (1 - options.objective.minimum_relative_improvement)))
    {
        retain(result, GeneralVoFBalanceRejection::NoFullComputeImprovement, rank);
        return result;
    }

    std::uint64_t native_bytes = 0;
    wire::collective(*comm,
        [&]
        {
            struct Group
            {
                std::vector<AMR::AdaptiveLeaf> leaves;
                std::vector<AMR::AdaptiveFamily> families;
            };
            std::map<int, Group> groups;
            std::map<AMR::AdaptiveBytes, int> routes;
            for (std::size_t i = 0; i < cells.size(); ++i)
            {
                const int peer = result.native_plan.localdestinations[i];
                auto key = AMR::encode_adaptive_key(cells[i].key);
                retained.add(key.size() + 2 * sizeof(AMR::AdaptiveLeaf) + 8 * sizeof(void*));
                // Leaf value copies own their key/surface arrays; immutable
                // recipes and geometry remain shared. Standalone wire size is
                // a conservative charge for those copied variable arrays.
                charge(retained, cells[i].work.migration_bytes, 2);
                routes.emplace(std::move(key), peer);
                groups[peer].leaves.push_back(hierarchy.owned().at(cells[i].key));
            }
            for (const auto& [key, family] : hierarchy.families())
            {
                retained.add(2 * sizeof(AMR::AdaptiveFamily));
                charge(retained, AMR::encode_adaptive_family(family).size(), 2);
                groups.at(routes.at(family.anchor)).families.push_back(family);
            }
            for (const auto& [peer, group] : groups)
                if (peer != rank)
                {
                    const auto encoded =
                        AMR::encode_adaptive_batch(group.leaves, group.families, limit - retained.used);
                    add(native_bytes, encoded.size());
                    add(native_bytes, 24); // one record and one peer stream per group
                }
        });
    wire::collective(*comm, [&] { retained.add(native_plan_bytes); });
    result.native_plan =
        AMR::qualify_balance_wire_bytes(result.native_plan, native_bytes, comm, result.native_plan.objective_options);
    retained.used -= native_plan_bytes; // the qualification copy replaced the original
    if (!result.native_plan.accepted)
    {
        retain(result, GeneralVoFBalanceRejection::NativeWireQualification, rank);
        return result;
    }
    result.local_native_application_bytes = native_bytes;
    result.global_native_application_bytes = sum(native_bytes, *comm);
    result.accepted = true;
    return result;
}
GeneralVoFBalanceResult plan_initial_partition(std::shared_ptr<const GeneralVoF> source,
    const GeneralVoFPreparedTransport& observed, GeneralVoFBalanceOptions options)
{
    auto proposal = plan_partition_proposal(source, observed, GeneralVoFBalanceBinding::ObservedSource, options);
    GeneralVoFBalanceResult result;
    static_cast<GeneralVoFBalanceProposal&>(result) = std::move(proposal);
    result.candidate = source;
    if (!result.accepted)
        return result;
    const auto& hierarchy = *source->hierarchy();
    const auto comm = hierarchy.communicator();
    const auto limit =
        std::min({options.objective.byte_budget, source->options().byte_budget, hierarchy.options().byte_budget});
    const auto migration_start = Clock::now();
    std::optional<GeneralVoF::Hierarchy::Plan> migration_holder;
    // The native planner finishes with a local shared-wrapper allocation.
    // Synchronize that final allocation as part of the enclosing phase too.
    wire::collective(*comm, [&] { migration_holder.emplace(hierarchy.plan_migration(result.native_plan)); });
    const auto& migration = *migration_holder;
    wire::collective(*comm,
        [&]
        {
            if (migration.actual_migration_application_bytes() != result.local_native_application_bytes)
                throw std::invalid_argument(
                    "Native initial migration bytes differ from exact prepacked application bytes.");
        });
    auto raw = hierarchy.apply(migration);
    AMR::AdaptiveBytes descriptor;
    wire::collective(*comm, [&] { descriptor = raw->configuration_descriptor(); });
    auto native = AMR::assemble_adaptive_native(*raw, descriptor);
    std::shared_ptr<const GeneralVoF::Hierarchy> native_bound;
    wire::collective(
        *comm, [&] { native_bound = raw->with_partition(native.partition, native.mesh->owned_cell_map()); });
    auto bound = hierarchy.apply(migration, std::move(native_bound));
    result.native_migration_seconds = maximum(elapsed(migration_start), *comm);
    const auto phase_start = Clock::now();
    GeneralVoFFiniteRemapOptions remap_options;
    remap_options.byte_budget = limit;
    auto remap = source->plan_remap(migration, bound, {}, remap_options);
    remap.validate_observations();
    wire::collective(*comm,
        [&]
        {
            if (!remap.ownership_only())
                throw std::invalid_argument("Initial VoF migration unexpectedly changed phase geometry.");
        });
    std::shared_ptr<const GeneralVoF> candidate;
    // All ranks enter candidate()'s own validation before its local allocation.
    // The surrounding phase synchronizes a possible allocation failure as well.
    wire::collective(*comm, [&] { candidate = remap.candidate(); });
    result.phase_migration_seconds = maximum(elapsed(phase_start), *comm);
    result.total_migration_seconds = maximum(elapsed(migration_start), *comm);
    result.native_application_bytes = result.global_native_application_bytes;
    result.candidate = std::move(candidate);
    result.accepted = true;
    return result;
}
} // namespace SimpleFluid::VoF
