/** @file BalanceForecast.cc
 * @brief Sparse explicit-destination donor fan-out and receiver work forecasts.
 */
#include "FVM/details/GeneralVoF/BalanceForecast.hh"
#include "FVM/GeneralVoFWork.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <bit>
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
GeneralVoFBalanceLoad load(double core, double receiver, std::uint64_t request, std::uint64_t reply,
    std::uint64_t consumers, std::uint64_t cut, const Teuchos::Comm<int>& comm)
{
    GeneralVoFBalanceLoad result;
    result.compute_available = true;
    wire::collective(comm,
        [&]
        {
            seconds(core);
            seconds(receiver);
            seconds(core + receiver);
        });
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &core, &result.total_core_seconds);
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &receiver, &result.total_receiver_seconds);
    result.maximum_compute_seconds = maximum(core + receiver, comm);
    result.mean_compute_seconds = (result.total_core_seconds + result.total_receiver_seconds) / comm.getSize();
    result.request_payload_bytes = sum(request, comm);
    result.reply_payload_bytes = sum(reply, comm);
    result.remote_donor_consumers = sum(consumers, comm);
    result.canonical_cut_faces = cut;
    wire::collective(comm,
        [&]
        {
            seconds(result.total_core_seconds);
            seconds(result.total_receiver_seconds);
            seconds(result.mean_compute_seconds);
        });
    return result;
}
struct DonorDemand
{
    std::set<int> before, after;
    struct Observation
    {
        double seconds = 0;
        bool mixed = false;
        std::uint64_t request = 0, reply = 0;
    };
    std::map<int, Observation> observed;
};
} // namespace
namespace general_vof_detail
{
GeneralVoFBalanceForecast forecast_partition_work(const GeneralVoFWorkSample& sample, std::span<const int> destinations,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, GeneralVoFBalanceOptions options)
{
    if (comm.is_null())
        throw std::invalid_argument("Initial VoF forecast needs a communicator.");
    const auto limit = options.objective.byte_budget;
    wire::Budget retained{limit};
    const int rank = comm->getRank(), ranks = comm->getSize();
    GeneralVoFBalanceForecast result;
    std::vector<double> cells;
    std::map<Key, std::size_t> indices;
    std::vector<AMR::AdaptiveOwner> registrations;
    std::vector<AMR::AdaptiveBytes> neighbors;
    std::uint64_t semantics = std::uint64_t(options.receiver_estimator), root_semantics = semantics;
    Teuchos::broadcast(*comm, 0, 1, &root_semantics);
    wire::collective(*comm,
        [&]
        {
            if (!limit || semantics != root_semantics ||
                semantics > std::uint64_t(GeneralVoFReceiverEstimator::Maximum) || !sample.fully_deduplicated ||
                destinations.size() != sample.cells.size())
                throw std::invalid_argument("Invalid or rank-divergent initial VoF forecast semantics.");
            charge(retained, sample.cells.size(),
                2 * sizeof(double) + sizeof(Key) + sizeof(std::size_t) + sizeof(AMR::AdaptiveOwner) +
                    8 * sizeof(void*));
            cells.reserve(sample.cells.size());
            registrations.reserve(sample.cells.size());
            for (std::size_t i = 0; i < sample.cells.size(); ++i)
            {
                const auto& cell = sample.cells[i];
                if (cell.owner != rank || (i && !(sample.cells[i - 1].key < cell.key)) || cell.calls > 1 ||
                    (!cell.calls && cell.reconstruction.seconds != 0) || destinations[i] < 0 ||
                    destinations[i] >= ranks)
                    throw std::invalid_argument("Invalid owned initial VoF forecast row or destination.");
                seconds(cell.reconstruction.seconds);
                auto key = AMR::encode_adaptive_key(cell.key);
                charge(retained, key.size(), 3);
                indices.emplace(cell.key, i);
                cells.push_back(cell.reconstruction.seconds);
                registrations.push_back({std::move(key), rank, std::uint64_t(destinations[i]), cell.revision});
            }
            std::set<AMR::AdaptiveBytes> unique;
            for (const auto& face : sample.faces)
            {
                const auto found = indices.find(face.first.key);
                if (face.identity.empty() || !unique.insert(face.identity).second || face.first.owner != rank ||
                    found == indices.end() || face.first.revision != sample.cells[found->second].revision ||
                    face.second.owner < 0 || face.second.owner >= ranks || face.first.key == face.second.key)
                    throw std::invalid_argument("Invalid canonical initial VoF forecast face.");
                seconds(face.setup.seconds);
                seconds(face.aggregate.seconds);
                cells[found->second] += face.setup.seconds + face.aggregate.seconds;
                auto key = AMR::encode_adaptive_key(face.second.key);
                retained.add(2 * face.identity.size() + key.size() + sizeof(AMR::AdaptiveBytes) + 6 * sizeof(void*));
                neighbors.push_back(std::move(key));
            }
        });
    AMR::DistributedAncestryDirectory assignments(registrations, comm, limit - retained.used);
    std::size_t lookup_cap = 0;
    wire::collective(*comm,
        [&]
        {
            retained.add(assignments.storage_bytes());
            lookup_cap = limit - retained.used;
            if (!lookup_cap)
                throw std::invalid_argument("Initial VoF forecast has no remaining directory lookup lease.");
        });
    auto neighbor_routes = assignments.lookup(neighbors, true, lookup_cap);
    std::uint64_t before_cut = 0, proposed_cut = 0;
    wire::collective(*comm,
        [&]
        {
            charge(retained, neighbor_routes.capacity(), sizeof(AMR::AdaptiveOwner));
            for (const auto& route : neighbor_routes)
                retained.add(route.key.capacity());
            for (std::size_t i = 0; i < sample.faces.size(); ++i)
            {
                const auto& face = sample.faces[i];
                const auto& route = neighbor_routes[i];
                if (route.owner != face.second.owner || route.address != face.second.revision ||
                    route.revision >= std::uint64_t(ranks))
                    throw std::invalid_argument("Initial VoF forecast face references a stale neighbor.");
                if (face.first.owner != face.second.owner)
                    add(before_cut, 1);
                if (destinations[indices.at(face.first.key)] != int(route.revision))
                    add(proposed_cut, 1);
            }
        });
    before_cut = sum(before_cut, *comm);
    proposed_cut = sum(proposed_cut, *comm);
    double before_core = 0, before_receiver = 0;
    wire::collective(*comm,
        [&]
        {
            for (const auto& cell : cells)
                before_core += cell;
            for (const auto& receiver : sample.receivers)
            {
                if (receiver.requester != rank || receiver.donor.owner == rank || receiver.donor.owner < 0 ||
                    receiver.donor.owner >= ranks || !receiver.request_payload_bytes || !receiver.reply_payload_bytes)
                    throw std::invalid_argument("Initial VoF receiver sample has invalid provenance or payload sizes.");
                seconds(receiver.remeasurement.seconds);
                before_receiver += receiver.remeasurement.seconds;
            }
        });
    std::uint64_t before_request = 0, before_reply = 0;
    wire::collective(*comm,
        [&]
        {
            for (const auto& receiver : sample.receivers)
            {
                add(before_request, receiver.request_payload_bytes);
                add(before_reply, receiver.reply_payload_bytes);
            }
        });
    result.measured_before =
        load(before_core, before_receiver, before_request, before_reply, sample.receivers.size(), before_cut, *comm);
    // No global key or face catalog: consumers send deduplicated demands to the
    // original donor owner, which holds that donor's proposed destination.
    std::vector<OwnerPacket> demand_packets;
    wire::collective(*comm,
        [&]
        {
            std::map<std::pair<Key, int>, std::pair<int, std::uint64_t>> unique;
            for (const auto& face : sample.faces)
            {
                if (!face.geometric && !face.donors.empty())
                    throw std::invalid_argument("Algebraic work capture has geometric donor dependencies.");
                const auto destination = destinations[indices.at(face.first.key)];
                for (const auto& donor : face.donors)
                {
                    if (donor.owner < 0 || donor.owner >= ranks)
                        throw std::invalid_argument("Initial VoF donor dependency has invalid owner.");
                    const auto key = AMR::encode_adaptive_key(donor.key);
                    const auto lookup = std::pair{donor.key, destination};
                    const auto previous = unique.find(lookup);
                    if (previous != unique.end())
                    {
                        if (previous->second != std::pair{donor.owner, donor.revision})
                            throw std::invalid_argument("Conflicting initial VoF donor dependency provenance.");
                        continue;
                    }
                    retained.add(
                        2 * key.size() + sizeof(Key) + 2 * sizeof(int) + sizeof(std::uint64_t) + 6 * sizeof(void*));
                    unique.emplace(lookup, std::pair{donor.owner, donor.revision});
                    retained.packet(key.size() + 32);
                    retained.add(sizeof(OwnerPacket));
                    wire::Writer w;
                    w.data.reserve(key.size() + 32);
                    w.word(0);
                    w.bytes(key);
                    w.word(donor.revision);
                    w.word(std::uint64_t(destination));
                    demand_packets.push_back({donor.owner, std::move(w.data)});
                }
            }
            for (const auto& receiver : sample.receivers)
            {
                const auto key = AMR::encode_adaptive_key(receiver.donor.key);
                retained.packet(key.size() + 64);
                retained.add(sizeof(OwnerPacket));
                wire::Writer w;
                w.data.reserve(key.size() + 64);
                w.word(1);
                w.bytes(key);
                w.word(receiver.donor.revision);
                w.word(receiver.mixed);
                w.word(std::bit_cast<std::uint64_t>(receiver.remeasurement.seconds));
                w.word(receiver.request_payload_bytes);
                w.word(receiver.reply_payload_bytes);
                demand_packets.push_back({receiver.donor.owner, std::move(w.data)});
            }
        });
    auto incoming = exchange_owner_packets(demand_packets, comm, limit - retained.used);
    std::map<Key, DonorDemand> demands;
    wire::collective(*comm,
        [&]
        {
            retained.add(incoming.stats.peak_scratch_bytes);
            for (const auto& packet : incoming.packets)
            {
                // Reader/decode scratch coexists with the retained exchange.
                charge(retained, packet.bytes.size(), 2);
                wire::Reader r{packet.bytes};
                const auto type = r.word();
                const auto key_bytes = r.bytes();
                const auto key = AMR::decode_adaptive_key(key_bytes);
                const auto revision = r.word();
                auto cell = indices.find(key);
                if (cell == indices.end() || sample.cells[cell->second].revision != revision ||
                    sample.cells[cell->second].calls != 1)
                    throw std::invalid_argument("Initial VoF donor demand is foreign, stale or unmeasured.");
                if (!demands.contains(key))
                    retained.add(key_bytes.size() + sizeof(Key) + sizeof(DonorDemand) + 6 * sizeof(void*));
                auto& demand = demands[key];
                if (type == 0)
                {
                    const auto destination = r.word();
                    if (destination >= std::uint64_t(ranks))
                        throw std::invalid_argument("Initial VoF donor consumer destination is invalid.");
                    retained.add(2 * (sizeof(int) + 6 * sizeof(void*)));
                    demand.before.insert(packet.peer);
                    demand.after.insert(int(destination));
                }
                else if (type == 1)
                {
                    DonorDemand::Observation observation;
                    auto mixed = r.word();
                    observation.mixed = bool(mixed);
                    observation.seconds = std::bit_cast<double>(r.word());
                    observation.request = r.word();
                    observation.reply = r.word();
                    seconds(observation.seconds);
                    retained.add(sizeof(DonorDemand::Observation) + sizeof(int) + 6 * sizeof(void*));
                    if (packet.peer == rank || mixed > 1 || !observation.request || !observation.reply ||
                        !demand.observed.emplace(packet.peer, observation).second)
                        throw std::invalid_argument("Initial VoF receiver observation is duplicated or invalid.");
                }
                else
                    throw std::invalid_argument("Unknown initial VoF donor packet.");
                r.done();
            }
        });
    std::vector<OwnerPacket> work_packets;
    std::uint64_t proposed_request = 0, proposed_reply = 0, proposed_consumers = 0;
    int incomplete = 0, any_incomplete = 0;
    wire::collective(*comm,
        [&]
        {
            struct Work
            {
                double core = 0, receiver = 0;
            };
            std::map<int, Work> loads;
            for (std::size_t i = 0; i < cells.size(); ++i)
                loads[destinations[i]].core += cells[i];
            for (const auto& [key, demand] : demands)
            {
                const auto index = indices.at(key);
                const auto& model = sample.cells[index];
                const int destination = destinations[index];
                std::size_t old_remote = 0;
                double estimate = 0;
                for (int requester : demand.before)
                    if (requester != rank)
                    {
                        ++old_remote;
                        const auto found = demand.observed.find(requester);
                        if (found == demand.observed.end() || found->second.mixed != model.mixed ||
                            found->second.request != model.request_payload_bytes ||
                            found->second.reply != model.reply_payload_bytes)
                            throw std::invalid_argument("Initial VoF receiver model is incomplete or inconsistent.");
                        if (model.mixed && found->second.seconds == 0)
                            incomplete = 1;
                        if (options.receiver_estimator == GeneralVoFReceiverEstimator::Maximum)
                            estimate = std::max(estimate, found->second.seconds);
                        else
                            estimate += found->second.seconds;
                    }
                if (old_remote != demand.observed.size())
                    throw std::invalid_argument("Initial VoF receiver sample has no matching donor consumer.");
                if (old_remote && options.receiver_estimator == GeneralVoFReceiverEstimator::Mean)
                    estimate /= old_remote;
                for (int requester : demand.after)
                    if (requester != destination)
                    {
                        if ((model.mixed && old_remote == 0) || !model.request_payload_bytes ||
                            !model.reply_payload_bytes)
                            incomplete = 1;
                        // Pure declarations have no receiver geometry remeasurement.
                        const auto existing = demand.observed.find(requester);
                        const auto receiver_seconds =
                            existing == demand.observed.end() ? estimate : existing->second.seconds;
                        loads[requester].receiver += model.mixed ? receiver_seconds : 0;
                        add(proposed_request, model.request_payload_bytes);
                        add(proposed_reply, model.reply_payload_bytes);
                        add(proposed_consumers, 1);
                    }
            }
            charge(retained, loads.size(), sizeof(int) + 2 * sizeof(double) + 6 * sizeof(void*));
            for (const auto& [peer, work] : loads)
            {
                seconds(work.core);
                seconds(work.receiver);
                retained.packet(16);
                retained.add(sizeof(OwnerPacket));
                wire::Writer w;
                w.data.reserve(16);
                w.word(std::bit_cast<std::uint64_t>(work.core));
                w.word(std::bit_cast<std::uint64_t>(work.receiver));
                work_packets.push_back({peer, std::move(w.data)});
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &incomplete, &any_incomplete);
    result.incomplete_receiver_model = bool(any_incomplete);
    auto work_incoming = exchange_owner_packets(work_packets, comm, limit - retained.used);
    double proposed_core = 0, proposed_receiver = 0;
    wire::collective(*comm,
        [&]
        {
            retained.add(work_incoming.stats.peak_scratch_bytes);
            for (const auto& packet : work_incoming.packets)
            {
                wire::Reader r{packet.bytes};
                const auto core = std::bit_cast<double>(r.word());
                const auto receiver = std::bit_cast<double>(r.word());
                r.done();
                seconds(core);
                seconds(receiver);
                proposed_core += core;
                proposed_receiver += receiver;
            }
        });
    result.forecast_proposed = load(
        proposed_core, proposed_receiver, proposed_request, proposed_reply, proposed_consumers, proposed_cut, *comm);
    result.forecast_proposed.compute_available = !any_incomplete;
    result.local_core_before = before_core;
    result.local_receiver_before = before_receiver;
    result.local_core_proposed = proposed_core;
    result.local_receiver_proposed = proposed_receiver;
    return result;
}
} // namespace general_vof_detail
} // namespace SimpleFluid::VoF
