/** @file GeneralVoF.cc
 * @brief Finite-inventory unsplit phase transport with transient exact native geometry proofs.
 */
#include "FVM/GeneralVoF.hh"
#include "FVM/details/ExactLiveStorage.hh"
#include "FVM/details/ExactPhaseAccumulator.hh"
#include "FVM/details/ExactPhaseLabelMap.hh"
#include "FVM/details/ExperimentalPhaseBackend.hh"
#include "FVM/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "utils/CompensatedSum.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
namespace SimpleFluid::VoF
{
namespace
{
namespace AMR = Meshes::AMR;
namespace wire = AMR::adaptive_detail;
using Key = AMR::AdaptiveCellKey;
using Measure = Meshes::ExactGeometryMeasure;
struct Cell
{
    Key key;
    std::uint64_t gid = 0, revision = 0;
    int owner = -1;
    Meshes::GeometryPoint centroid{};
    double native_volume = 0;
    PolyhedralPhaseSample phase;
    PhaseError occupancy;
    PhaseRoundoff roundoff;
    std::shared_ptr<const ExperimentalPhaseInventory> experimental_exact;
    GeneralVoFPhaseGeometry declaration = GeneralVoFPhaseGeometry::Infer;
    std::optional<PolyhedralInterface> retained_interface;
    bool inherited = false;
};
Meshes::ExactPhaseMeasures phase_measures(const Cell& cell)
{
    if (cell.experimental_exact)
        return cell.experimental_exact->phases;
    // Default policy: one-step lift only, no rational accepted history.
    return {Measure::volume(*cell.phase.geometry), Measure::from_value(cell.phase.gas_volume),
        Measure::from_value(cell.phase.carrier_volume)};
}
Measure finite_bound(double value)
{
    return Measure::from_value(value);
}
PolyhedralPhaseSample geometric_sample(const Cell&);
void occupancy(Cell& cell)
{
    if (!std::isfinite(cell.native_volume) || !(cell.native_volume > 0))
        throw std::invalid_argument("General VoF native metric is invalid.");
    const auto lifted = phase_measures(cell);
    const auto native = Measure::from_value(cell.native_volume);
    const auto defect = lifted.gas + lifted.carrier - lifted.bulk;
    const auto conversion = native - lifted.bulk;
    const auto absolute = [](const Measure& value) { return value.sign() < 0 ? value.scaled(-1) : value; };
    const auto input = native.scaled(8 * std::numeric_limits<double>::epsilon());
    if (absolute(conversion).compare(input) > 0 || absolute(defect - conversion).compare(input) > 0)
        throw std::invalid_argument("General VoF finite phase/native-volume input error exceeds its bound.");
    const auto diagnostic = detail::round_phase_residual(defect);
    const auto native_diagnostic = detail::round_phase_residual(conversion);
    const auto diagnostic_error =
        finite_bound(diagnostic.absolute_error) + finite_bound(native_diagnostic.absolute_error);
    cell.occupancy.difference = diagnostic.value;
    cell.occupancy.native_conversion = native_diagnostic.value;
    cell.occupancy.phase_input_bound = detail::phase_upper_bound(input + diagnostic_error);
    cell.occupancy.admitted_bound = detail::phase_upper_bound(absolute(conversion) + input + diagnostic_error);
}
void validate_cell(const Cell& cell, const GeneralVoFOptions& options)
{
    const bool experimental = options.inventory_policy == PhaseInventoryPolicy::ExperimentalExact;
    if (experimental != bool(cell.experimental_exact))
        throw std::invalid_argument("General VoF inventory policy/payload differs.");
    if (experimental)
    {
        detail::validate_experimental(
            *cell.experimental_exact, *cell.phase.geometry, cell.native_volume, options.byte_budget);
        const auto view = detail::experimental_view(*cell.experimental_exact, cell.native_volume);
        if (view.phases.gas != cell.phase.gas_volume || view.phases.carrier != cell.phase.carrier_volume)
            throw std::invalid_argument("Experimental exact diagnostic phase view differs.");
    }
    // Finite diagnostic views still have honest finite conversion provenance.
    const Meshes::ExactPhaseMeasures lifted{Measure::volume(*cell.phase.geometry),
        Measure::from_value(cell.phase.gas_volume), Measure::from_value(cell.phase.carrier_volume)};
    const auto defect = lifted.gas + lifted.carrier - lifted.bulk;
    const auto absolute = defect.sign() < 0 ? defect.scaled(-1) : defect;
    const auto& e = cell.occupancy;
    bool finite = std::isfinite(e.difference) && std::isfinite(e.native_conversion) && std::isfinite(e.fit_residual);
    for (double bound : {e.admitted_bound, e.phase_input_bound, e.fit_bound, e.transferred_bound,
             e.fit_allocation_bound, cell.roundoff.gas, cell.roundoff.carrier})
        finite = finite && std::isfinite(bound) && bound >= 0;
    if (!finite || !std::isfinite(cell.phase.gas_volume) || !std::isfinite(cell.phase.carrier_volume) ||
        cell.phase.gas_volume < 0 || cell.phase.carrier_volume < 0 ||
        detail::phase_signed_value(defect) != e.difference || absolute.compare(finite_bound(e.admitted_bound)) > 0 ||
        detail::phase_signed_value(Measure::from_value(cell.native_volume) - lifted.bulk) != e.native_conversion)
        throw std::invalid_argument("General VoF finite occupancy provenance is invalid.");
    // Physical occupancy was checked independently above. Geometry fitting
    // always uses complementary labels, not a fresh input tolerance on an
    // accepted inventory carrying explicit publication uncertainty.
    PolyhedralReconstruction::validate(geometric_sample(cell), options.reconstruction);
    if (std::uint64_t(cell.declaration) > 2)
        throw std::invalid_argument("General VoF invalid phase geometry declaration.");
    if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
    {
        const double minority =
            cell.declaration == GeneralVoFPhaseGeometry::PureGas ? cell.phase.carrier_volume : cell.phase.gas_volume;
        const auto metric = cell.phase.geometry->moments().volume;
        const double inherited_bound = std::max(e.admitted_bound, e.fit_bound);
        if (experimental)
        {
            const auto& x = *cell.experimental_exact;
            const auto actual = cell.declaration == GeneralVoFPhaseGeometry::PureGas ? x.phases.carrier : x.phases.gas;
            const auto bound =
                x.error.admitted_bound.compare(x.error.fit_bound) >= 0 ? x.error.admitted_bound : x.error.fit_bound;
            if (actual.compare(bound) > 0)
                throw std::invalid_argument("Experimental pure declaration contradicts exact inventory.");
        }
        if ((!experimental && cell.inherited && minority > inherited_bound) ||
            (!experimental && !cell.inherited &&
                minority >
                    metric.binary64_absolute_error() + 8 * std::numeric_limits<double>::epsilon() * metric.value))
            throw std::invalid_argument("General VoF pure declaration contradicts finite inventory.");
    }
}
void real(wire::Writer& w, double value)
{
    w.word(std::bit_cast<std::uint64_t>(value));
}
double real(wire::Reader& r)
{
    return std::bit_cast<double>(r.word());
}
void plane(wire::Writer& w, const PolyhedralInterface& p)
{
    for (auto v : p.gas.normal)
        real(w, v);
    for (auto v : p.gas.anchor)
        real(w, v);
    real(w, p.gas.offset);
    w.word(p.pure_gas);
    w.word(p.pure_carrier);
    real(w, p.stencil_volume_residual);
    real(w, p.stencil_condition);
    real(w, p.orientation_step);
    w.word(p.fit_iterations);
    w.word(p.fit_converged);
}
PolyhedralInterface plane(wire::Reader& r)
{
    PolyhedralInterface p;
    for (auto& v : p.gas.normal)
        v = real(r);
    for (auto& v : p.gas.anchor)
        v = real(r);
    p.gas.offset = real(r);
    p.pure_gas = r.word();
    p.pure_carrier = r.word();
    p.stencil_volume_residual = real(r);
    p.stencil_condition = real(r);
    p.orientation_step = real(r);
    p.fit_iterations = r.word();
    p.fit_converged = r.word();
    return p;
}
Meshes::ExactPhaseMeasures phases(const Meshes::ExactConvexFragment& fragment, const PolyhedralInterface& p)
{
    if (p.pure_gas)
        return {fragment.volume(), fragment.volume(), {}};
    if (p.pure_carrier)
        return {fragment.volume(), {}, fragment.volume()};
    return fragment.phases(p.gas);
}
GeneralVoFPhaseEnclosure diagnostic(const Meshes::ExactPhaseMeasures& measures)
{
    return {measures.bulk.enclosure(), measures.gas.enclosure(), measures.carrier.enclosure()};
}
double output(const Measure& measure)
{
    if (measure.sign() < 0)
        throw std::invalid_argument("General VoF phase measure is negative.");
    return measure.sign() ? measure.certified().as_double() : 0;
}
Meshes::ExactPhaseMeasures geometric_labels(const Cell& cell)
{
    const bool gas = phase_measures(cell).gas.compare(phase_measures(cell).carrier) <= 0;
    const auto minority = gas ? phase_measures(cell).gas : phase_measures(cell).carrier;
    const auto dominant = phase_measures(cell).bulk - minority;
    if (minority.sign() < 0 || dominant.sign() < 0)
        throw std::invalid_argument("General VoF physical minority has no complementary geometric representation.");
    // This is a geometric fitting label only. Authoritative independent phase
    // inventories and their inherited occupancy error are never overwritten.
    return {phase_measures(cell).bulk, gas ? minority : dominant, gas ? dominant : minority};
}
PolyhedralPhaseSample geometric_sample(const Cell& cell)
{
    const auto labels = geometric_labels(cell);
    return {cell.phase.geometry, detail::round_phase_value(labels.gas).value,
        detail::round_phase_value(labels.carrier).value};
}
} // namespace
struct GeneralVoF::Data
{
    struct Wall
    {
        AMR::AdaptiveBytes identity;
        Meshes::ExactPlanarFacePtr geometry;
    };
    struct Topology
    {
        AMR::AdaptiveNativePublication native;
        AMR::AdaptiveNativeDraft draft;
        std::vector<AMR::AdaptiveTile> tiles;
        std::vector<AMR::AdaptiveNode> nodes;
        std::map<std::uint64_t, Key> by_gid;
        std::map<Key, std::set<Key>> adjacency;
        std::map<std::uint64_t, std::set<Key>> node_fans;
        std::map<std::uint64_t, std::vector<Wall>> node_walls;
        std::vector<Key> order;
        std::map<Key, std::vector<std::size_t>> cell_tiles;
        std::map<std::uint64_t, std::vector<std::size_t>> node_tiles;
    };
    std::shared_ptr<const Topology> topology;
    std::shared_ptr<const Hierarchy> hierarchy;
    AMR::HierarchyObservation observation;
    std::shared_ptr<const Hierarchy> supplied_hierarchy;
    AMR::HierarchyObservation supplied_observation;
    GeneralVoFOptions options;
    std::uint64_t control = 0;
    std::weak_ptr<const void> adopted_source;
    std::uint64_t adopted_source_control = 0, adopted_transaction = 0, adopted_target_control = 0;
    std::map<Key, Cell> cells;
    std::vector<double> gas, carrier;
    std::vector<PhaseVolumes> phases;
    std::vector<PhaseRoundoff> roundoff;
    std::vector<PhaseError> occupancy;
    std::shared_ptr<const std::vector<Meshes::ExactPhaseMeasures>> experimental_catalog;
};
namespace
{
void refresh_experimental_catalog(GeneralVoF::Data& data)
{
    if (data.options.inventory_policy == PhaseInventoryPolicy::Finite)
    {
        data.experimental_catalog.reset();
        return;
    }
    wire::Budget budget{data.options.byte_budget};
    if (data.topology->order.size() > budget.limit / sizeof(Meshes::ExactPhaseMeasures))
        throw std::invalid_argument("Experimental owned inventory catalog exceeds budget.");
    budget.add(data.topology->order.size() * sizeof(Meshes::ExactPhaseMeasures));
    auto catalog = std::make_shared<std::vector<Meshes::ExactPhaseMeasures>>();
    catalog->reserve(data.topology->order.size());
    for (const auto& key : data.topology->order)
    {
        const auto& cell = data.cells.at(key);
        if (!cell.experimental_exact)
            throw std::invalid_argument("Missing experimental owned inventory.");
        catalog->push_back(cell.experimental_exact->phases);
    }
    data.experimental_catalog = std::move(catalog);
}
void current(const GeneralVoF::Data& data)
{
    if (!data.hierarchy->geometry_unchanged() || data.hierarchy->observation() != data.observation ||
        !data.supplied_hierarchy->geometry_unchanged() ||
        data.supplied_hierarchy->observation() != data.supplied_observation)
        throw std::invalid_argument("General VoF source geometry/control is stale.");
}
void retained_budget(const GeneralVoF::Data& data)
{
    wire::Budget budget{data.options.byte_budget};
    std::set<const void*> geometry;
    budget.add(sizeof(data) + sizeof(GeneralVoF::Data::Topology) + data.topology->order.capacity() * sizeof(Key) +
               (data.gas.capacity() + data.carrier.capacity()) * sizeof(double) +
               data.phases.capacity() * sizeof(PhaseVolumes) + data.roundoff.capacity() * sizeof(PhaseRoundoff) +
               data.occupancy.capacity() * sizeof(PhaseError));
    std::set<const void*> exact_payloads;
    const auto charge_exact = [&](const Meshes::ExactGeometryMeasure& value)
    {
        if (exact_payloads.insert(value.storage_identity()).second)
            budget.add(value.storage_bytes());
    };
    if (data.experimental_catalog)
        budget.add(sizeof(*data.experimental_catalog) +
                   data.experimental_catalog->capacity() * sizeof(Meshes::ExactPhaseMeasures));
    for (const auto& [key, cell] : data.cells)
    {
        budget.add(sizeof(Cell) + sizeof(Key) + 8 * sizeof(void*) + AMR::encode_adaptive_key(key).size());
        if (geometry.insert(cell.phase.geometry.get()).second)
            budget.add(cell.phase.geometry->storage_bytes());
        if (cell.experimental_exact)
        {
            budget.add(sizeof(ExperimentalPhaseInventory));
            const auto& x = *cell.experimental_exact;
            for (const auto* q :
                {&x.phases.bulk, &x.phases.gas, &x.phases.carrier, &x.error.difference, &x.error.admitted_bound,
                    &x.error.native_conversion, &x.error.phase_input_bound, &x.error.fit_residual, &x.error.fit_bound,
                    &x.error.transferred_bound, &x.error.fit_allocation_bound})
                charge_exact(*q);
        }
    }
    for (const auto& node : data.topology->nodes)
        budget.add(sizeof(node) + node.identity.capacity() + node.exact_identity.capacity());
    for (const auto& tile : data.topology->tiles)
    {
        budget.add(sizeof(tile) + tile.node_gids.capacity() * sizeof(std::uint64_t));
        if (geometry.insert(tile.geometry.get()).second)
            budget.add(tile.geometry->storage_bytes());
    }
    auto graph = [&](const auto& entries)
    {
        for (const auto& [key, values] : entries)
        {
            budget.add(sizeof(key) + sizeof(values) + 8 * sizeof(void*));
            for (const auto& value : values)
                budget.add(sizeof(value) + 4 * sizeof(void*) + AMR::encode_adaptive_key(value).size());
        }
    };
    graph(data.topology->adjacency);
    graph(data.topology->node_fans);
    for (const auto& [gid, key] : data.topology->by_gid)
        budget.add(sizeof(gid) + sizeof(Key) + 4 * sizeof(void*) + AMR::encode_adaptive_key(key).size());
    const auto incidence = [&](const auto& entries)
    {
        for (const auto& [key, indices] : entries)
        {
            budget.add(sizeof(key) + sizeof(indices) + 4 * sizeof(void*) +
                       indices.capacity() * sizeof(std::size_t));
            if constexpr (std::is_same_v<std::decay_t<decltype(key)>, Key>)
                budget.add(AMR::encode_adaptive_key(key).size());
        }
    };
    incidence(data.topology->cell_tiles);
    incidence(data.topology->node_tiles);
    for (const auto& [node, walls] : data.topology->node_walls)
    {
        budget.add(sizeof(node) + sizeof(walls) + walls.capacity() * sizeof(GeneralVoF::Data::Wall));
        for (const auto& wall : walls)
        {
            budget.add(wall.identity.capacity());
            if (geometry.insert(wall.geometry.get()).second)
                budget.add(wall.geometry->storage_bytes());
        }
    }
}
// Every incident owner registers its actual cell once. The node authority
// authenticates those registrations and returns the complete canonical fan;
// geometric coincidence and whichever ghost rows happen to be resident play no
// part in defining the interpolation stencil.
void complete_node_fans(GeneralVoF::Data& data, GeneralVoF::Data::Topology& topology)
{
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::vector<OwnerPacket> requests;
    std::map<std::uint64_t, const AMR::AdaptiveNode*> native_nodes;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::map<std::uint64_t, std::set<Key>> local;
            for (const auto& node : topology.draft.nodes)
            {
                budget.add(sizeof(decltype(native_nodes)::value_type) + 4 * sizeof(void*));
                native_nodes.emplace(node.gid, &node);
            }
            for (const auto& tile : topology.draft.tiles)
                for (auto node : tile.node_gids)
                    for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                        if (key && data.hierarchy->owned().contains(*key))
                        {
                            budget.add(sizeof(Key) + AMR::encode_adaptive_key(*key).size() + 8 * sizeof(void*));
                            local[node].insert(*key);
                        }
            for (const auto& [node, keys] : local)
            {
                wire::Writer w;
                const auto& identity = native_nodes.at(node)->identity;
                budget.packet(32 + identity.size());
                w.word(node);
                w.bytes(identity);
                w.word(keys.size());
                for (const auto& key : keys)
                {
                    const auto bytes = AMR::encode_adaptive_key(key);
                    budget.add(16 + bytes.size());
                    w.bytes(bytes);
                    w.word(data.hierarchy->owned().at(key).revision);
                }
                requests.push_back({int(node % std::uint64_t(comm->getSize())), std::move(w.data)});
                topology.node_fans.emplace(node, std::set<Key>{});
            }
        });
    auto received = exchange_owner_packets(requests, comm, limit);
    struct Registration
    {
        std::uint64_t node, revision;
        Key key;
        int peer;
    };
    std::vector<Registration> registrations;
    std::vector<Key> keys;
    std::map<std::uint64_t, std::set<int>> subscribers;
    std::map<std::uint64_t, AMR::AdaptiveBytes> identities;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                const auto node = r.word();
                auto identity = r.bytes();
                budget.add(identity.size() + sizeof(identity) + 8 * sizeof(void*));
                if (node % std::uint64_t(comm->getSize()) != std::uint64_t(comm->getRank()) ||
                    !subscribers[node].insert(packet.peer).second)
                    throw std::invalid_argument("General VoF duplicate or misrouted node fan registration.");
                const auto [it, fresh] = identities.emplace(node, identity);
                if (!fresh && it->second != identity)
                    throw std::invalid_argument("General VoF node fan identity differs.");
                const auto count = r.word();
                if (!count || count > data.options.maximum_stencil_cells)
                    throw std::invalid_argument("General VoF node fan budget exhausted.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    auto bytes = r.bytes();
                    budget.add(bytes.size() + sizeof(Registration) + sizeof(Key));
                    auto key = AMR::decode_adaptive_key(bytes);
                    const auto revision = r.word();
                    registrations.push_back({node, revision, key, packet.peer});
                    keys.push_back(std::move(key));
                }
                r.done();
            }
        });
    const auto owners = data.hierarchy->lookup_cells(keys);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::map<std::uint64_t, std::set<Key>> fans;
            for (std::size_t i = 0; i < registrations.size(); ++i)
            {
                const auto& entry = registrations[i];
                if (owners[i].owner != entry.peer || owners[i].revision != entry.revision ||
                    !fans[entry.node].insert(entry.key).second)
                    throw std::invalid_argument("General VoF unauthenticated node fan cell.");
                if (fans[entry.node].size() > data.options.maximum_stencil_cells)
                    throw std::invalid_argument("General VoF complete node fan budget exhausted.");
            }
            for (const auto& [node, fan] : fans)
                for (auto peer : subscribers.at(node))
                {
                    wire::Writer w;
                    budget.packet(24 + identities.at(node).size());
                    w.word(node);
                    w.bytes(identities.at(node));
                    w.word(fan.size());
                    for (const auto& key : fan)
                    {
                        const auto bytes = AMR::encode_adaptive_key(key);
                        budget.add(8 + bytes.size());
                        w.bytes(bytes);
                    }
                    replies.push_back({peer, std::move(w.data)});
                }
        });
    auto returned = exchange_owner_packets(replies, comm, limit);
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(native_nodes.size() * (sizeof(decltype(native_nodes)::value_type) + 4 * sizeof(void*)));
            std::set<std::uint64_t> seen;
            for (const auto& packet : returned.packets)
            {
                wire::Reader r{packet.bytes};
                const auto node = r.word();
                const auto identity = r.bytes();
                if (!topology.node_fans.contains(node) || !seen.insert(node).second ||
                    packet.peer != int(node % std::uint64_t(comm->getSize())))
                    throw std::invalid_argument("General VoF unexpected node fan reply.");
                const auto native = native_nodes.find(node);
                if (native == native_nodes.end() || native->second->identity != identity)
                    throw std::invalid_argument("General VoF node fan reply identity differs.");
                const auto count = r.word();
                if (!count || count > data.options.maximum_stencil_cells)
                    throw std::invalid_argument("General VoF node fan reply budget exhausted.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    const auto bytes = r.bytes();
                    budget.add(bytes.size() + sizeof(Key) + 8 * sizeof(void*));
                    if (!topology.node_fans.at(node).insert(AMR::decode_adaptive_key(bytes)).second)
                        throw std::invalid_argument("General VoF duplicate node fan member.");
                }
                r.done();
            }
            if (seen.size() != topology.node_fans.size())
                throw std::invalid_argument("General VoF incomplete canonical node fan.");
        });
}
std::uint64_t transaction(Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    std::uint64_t id = 0;
    wire::collective(*comm,
        [&]
        {
            if (comm->getRank() == 0)
            {
                static std::uint64_t next = 0;
                if (next == std::numeric_limits<std::uint64_t>::max())
                    throw std::overflow_error("General VoF transaction identity exhausted.");
                id = ++next;
            }
        });
    Teuchos::broadcast(*comm, 0, 1, &id);
    return id;
}
void agree_transaction(Teuchos::RCP<const Teuchos::Comm<int>> comm, std::uint64_t id)
{
    std::uint64_t low = 0, high = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &id, &low);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &id, &high);
    wire::collective(*comm,
        [&]
        {
            if (!id || low != high)
                throw std::invalid_argument("General VoF mixed collective trial controls.");
        });
}
void agree_options(const GeneralVoFOptions& options, Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{options.byte_budget};
            if (std::uint64_t(options.inventory_policy) > 1)
                throw std::invalid_argument("Unknown General VoF inventory policy.");
            budget.packet(17 * 8);
            wire::Writer w;
            for (auto value : {std::uint64_t(options.maximum_stencil_cells), std::uint64_t(options.maximum_halo_cells),
                     std::uint64_t(options.support_rings), std::uint64_t(options.inventory_policy),
                     std::uint64_t(options.reconstruction.maximum_offset_iterations),
                     std::uint64_t(options.reconstruction.maximum_fit_iterations),
                     std::uint64_t(options.reconstruction.maximum_stencil_cells),
                     std::uint64_t(options.reconstruction.maximum_clip_evaluations),
                     std::uint64_t(options.sweep.maximum_simplices), std::uint64_t(options.sweep.maximum_atoms),
                     std::uint64_t(options.sweep.maximum_pair_tests)})
                w.word(value);
            real(w, options.reconstruction.volume_tolerance);
            real(w, options.reconstruction.fit_tolerance);
            real(w, options.reconstruction.maximum_condition);
            real(w, options.reconstruction.orientation_tolerance);
            real(w, options.reconstruction.maximum_model_residual);
            packets.push_back({0, std::move(w.data)});
        });
    const auto received = exchange_owner_packets(packets, comm, options.byte_budget);
    wire::collective(*comm,
        [&]
        {
            if (comm->getRank() == 0)
            {
                if (received.packets.size() != std::size_t(comm->getSize()))
                    throw std::invalid_argument("General VoF missing algorithm controls.");
                for (const auto& packet : received.packets)
                    if (packet.bytes != received.packets.front().bytes)
                        throw std::invalid_argument("General VoF algorithm controls disagree across ranks.");
            }
        });
}
void import_phases(GeneralVoF::Data& data)
{
    auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, cell] : data.cells)
                if (cell.owner != comm->getRank())
                {
                    const auto encoded = AMR::encode_adaptive_key(key);
                    budget.packet(encoded.size() + 16);
                    wire::Writer w;
                    w.bytes(encoded);
                    w.word(cell.revision);
                    requests.push_back({cell.owner, std::move(w.data)});
                }
        });
    auto incoming = exchange_owner_packets(requests, comm, limit);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                auto bytes = r.bytes();
                auto key = AMR::decode_adaptive_key(bytes);
                auto revision = r.word();
                r.done();
                const auto& cell = data.cells.at(key);
                if (cell.owner != comm->getRank() || cell.revision != revision)
                    throw std::invalid_argument("General VoF stale phase request.");
                budget.packet(bytes.size() + 136);
                wire::Writer w;
                w.bytes(bytes);
                w.word(revision);
                w.word(std::uint64_t(data.options.inventory_policy));
                real(w, cell.phase.gas_volume);
                real(w, cell.phase.carrier_volume);
                w.word(std::uint64_t(cell.declaration));
                w.word(cell.inherited);
                real(w, cell.roundoff.gas);
                real(w, cell.roundoff.carrier);
                for (const auto* value :
                    {&cell.occupancy.difference, &cell.occupancy.admitted_bound, &cell.occupancy.native_conversion,
                        &cell.occupancy.phase_input_bound, &cell.occupancy.fit_residual, &cell.occupancy.fit_bound,
                        &cell.occupancy.transferred_bound, &cell.occupancy.fit_allocation_bound})
                    real(w, *value);
                if (data.options.inventory_policy == PhaseInventoryPolicy::ExperimentalExact)
                {
                    if (!cell.experimental_exact)
                        throw std::invalid_argument("Missing exact owner phase state.");
                    const auto encoded = detail::encode_experimental(*cell.experimental_exact, limit);
                    budget.add(8 + encoded.size());
                    w.bytes(encoded);
                }
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    auto received = exchange_owner_packets(replies, comm, limit);
    wire::collective(*comm,
        [&]
        {
            std::set<Key> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto key = AMR::decode_adaptive_key(r.bytes());
                auto revision = r.word();
                auto& cell = data.cells.at(key);
                if (cell.owner != packet.peer || cell.owner == comm->getRank() || cell.revision != revision ||
                    !seen.insert(key).second)
                    throw std::invalid_argument("General VoF unauthenticated phase reply.");
                if (r.word() != std::uint64_t(data.options.inventory_policy))
                    throw std::invalid_argument("General VoF ghost inventory policy differs.");
                cell.phase.gas_volume = real(r);
                cell.phase.carrier_volume = real(r);
                cell.declaration = GeneralVoFPhaseGeometry(r.word());
                cell.inherited = r.word();
                cell.roundoff.gas = real(r);
                cell.roundoff.carrier = real(r);
                for (auto* value :
                    {&cell.occupancy.difference, &cell.occupancy.admitted_bound, &cell.occupancy.native_conversion,
                        &cell.occupancy.phase_input_bound, &cell.occupancy.fit_residual, &cell.occupancy.fit_bound,
                        &cell.occupancy.transferred_bound, &cell.occupancy.fit_allocation_bound})
                    *value = real(r);
                if (data.options.inventory_policy == PhaseInventoryPolicy::ExperimentalExact)
                    cell.experimental_exact = detail::decode_experimental(r.bytes(), limit);
                else
                    cell.experimental_exact.reset();
                r.done();
                validate_cell(cell, data.options);
            }
            if (seen.size() != requests.size())
                throw std::invalid_argument("General VoF incomplete phase replies.");
        });
}
std::map<Key, PolyhedralInterface> reconstruct_local(const GeneralVoF::Data& data)
{
    std::map<Key, PolyhedralInterface> result;
    for (const auto& key : data.topology->order)
    {
        const auto& target = data.cells.at(key);
        validate_cell(target, data.options);
        if (target.retained_interface)
        {
            result.emplace(key, *target.retained_interface);
            continue;
        }
        if (target.declaration != GeneralVoFPhaseGeometry::Infer)
        {
            const bool gas = target.declaration == GeneralVoFPhaseGeometry::PureGas;
            const auto minority = gas ? target.phase.carrier_volume : target.phase.gas_volume;
            PolyhedralInterface p;
            p.pure_gas = gas;
            p.pure_carrier = !gas;
            const auto difference = (gas ? phase_measures(target).carrier : phase_measures(target).gas).scaled(-1);
            p.target_volume_difference = difference.enclosure();
            p.target_volume_residual = minority;
            result.emplace(key, p);
            continue;
        }
        if (target.phase.gas_volume == 0 || target.phase.carrier_volume == 0)
        {
            result.emplace(key, PolyhedralReconstruction::solve_offset(
                                    geometric_sample(target), {1, 0, 0}, data.options.reconstruction));
            continue;
        }
        std::set<Key> visited{key}, front{key};
        std::vector<PolyhedralPhaseSample> stencil;
        for (unsigned ring = 0; ring < data.options.support_rings; ++ring)
        {
            std::set<Key> next;
            for (const auto& center : front)
            {
                const auto found = data.topology->adjacency.find(center);
                if (found == data.topology->adjacency.end())
                    continue;
                for (const auto& neighbor : found->second)
                    if (data.cells.contains(neighbor) && visited.insert(neighbor).second)
                    {
                        if (stencil.size() == data.options.maximum_stencil_cells)
                            throw std::invalid_argument("General VoF stencil budget exhausted.");
                        stencil.push_back(geometric_sample(data.cells.at(neighbor)));
                        next.insert(neighbor);
                    }
            }
            front = std::move(next);
        }
        result.emplace(
            key, PolyhedralReconstruction::reconstruct(geometric_sample(target), stencil, data.options.reconstruction));
    }
    return result;
}
std::map<Key, PolyhedralInterface> import_planes(const GeneralVoF::Data& data)
{
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::map<Key, PolyhedralInterface> result;
    wire::collective(*comm,
        [&]
        {
            current(data);
            result = reconstruct_local(data);
        });
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, cell] : data.cells)
                if (cell.owner != comm->getRank())
                {
                    auto bytes = AMR::encode_adaptive_key(key);
                    budget.packet(bytes.size() + 16);
                    wire::Writer w;
                    w.bytes(bytes);
                    w.word(cell.revision);
                    requests.push_back({cell.owner, std::move(w.data)});
                }
        });
    auto incoming = exchange_owner_packets(requests, comm, limit);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                auto bytes = r.bytes();
                auto key = AMR::decode_adaptive_key(bytes);
                auto revision = r.word();
                r.done();
                const auto& cell = data.cells.at(key);
                if (cell.owner != comm->getRank() || cell.revision != revision)
                    throw std::invalid_argument("General VoF stale plane request.");
                budget.packet(bytes.size() + 128);
                wire::Writer w;
                w.bytes(bytes);
                w.word(revision);
                plane(w, result.at(key));
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    auto received = exchange_owner_packets(replies, comm, limit);
    wire::collective(*comm,
        [&]
        {
            std::set<Key> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto key = AMR::decode_adaptive_key(r.bytes());
                auto revision = r.word();
                const auto& cell = data.cells.at(key);
                if (cell.owner != packet.peer || cell.owner == comm->getRank() || cell.revision != revision ||
                    !seen.insert(key).second)
                    throw std::invalid_argument("General VoF unauthenticated plane reply.");
                auto fitted = plane(r);
                if (!fitted.pure_gas && !fitted.pure_carrier)
                {
                    const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                    const auto measures =
                        Meshes::ExactConvexFragment::from_geometry(*cell.phase.geometry)->phases(fitted.gas);
                    const auto residual = (gas ? measures.gas : measures.carrier) -
                                          Measure::from_value(gas ? cell.phase.gas_volume : cell.phase.carrier_volume);
                    fitted.target_volume_difference = residual.enclosure();
                    fitted.target_volume_residual = double(fitted.target_volume_difference.value);
                }
                result.emplace(key, std::move(fitted));
                r.done();
            }
            if (seen.size() != requests.size())
                throw std::invalid_argument("General VoF incomplete plane replies.");
        });
    return result;
}
} // namespace
GeneralVoF::GeneralVoF(std::shared_ptr<const Hierarchy> hierarchy, std::vector<Key> order, std::vector<double> gas,
    std::vector<double> carrier, GeneralVoFOptions options, std::vector<GeneralVoFPhaseGeometry> declarations)
    : GeneralVoF(std::move(hierarchy), std::move(order), std::move(gas), std::move(carrier), options,
          std::move(declarations), nullptr)
{
}
GeneralVoF::GeneralVoF(std::shared_ptr<const Hierarchy> hierarchy, std::vector<Key> order, std::vector<double> gas,
    std::vector<double> carrier, GeneralVoFOptions options, std::vector<GeneralVoFPhaseGeometry> declarations,
    const ExactPhaseTransferResult* transfer)
{
    const auto* capsules = transfer ? &transfer->owned() : nullptr;
    auto comm = hierarchy ? hierarchy->communicator() : Tpetra::getDefaultComm();
    std::shared_ptr<Data> data;
    std::shared_ptr<Data::Topology> topology;
    wire::collective(*comm,
        [&]
        {
            if (!hierarchy || !options.byte_budget || !options.maximum_stencil_cells || !options.maximum_halo_cells ||
                !options.support_rings || options.support_rings > 8 || order.size() != gas.size() ||
                order.size() != carrier.size() || order.size() != hierarchy->owned().size() ||
                (!declarations.empty() && declarations.size() != order.size()) ||
                (capsules && capsules->size() != order.size()))
                throw std::invalid_argument("Invalid General VoF source arrays/options.");
            wire::Budget budget{options.byte_budget};
            budget.add(order.size() * (sizeof(Cell) + sizeof(Key) + 2 * sizeof(double)));
            data = std::make_shared<Data>();
            topology = std::make_shared<Data::Topology>();
            data->topology = topology;
            data->hierarchy = hierarchy;
            data->observation = hierarchy->observation();
            data->supplied_hierarchy = hierarchy;
            data->supplied_observation = data->observation;
            data->options = options;
            current(*data);
            std::set<Key> seen;
            for (std::size_t i = 0; i < order.size(); ++i)
            {
                if (!hierarchy->owned().contains(order[i]) || !seen.insert(order[i]).second)
                    throw std::invalid_argument("General VoF input ownership differs.");
                const auto& leaf = hierarchy->owned().at(order[i]);
                Cell cell;
                cell.key = order[i];
                cell.owner = comm->getRank();
                cell.revision = leaf.revision;
                cell.phase = {hierarchy->geometry(leaf), gas[i], carrier[i]};
                cell.native_volume = hierarchy->metrics(leaf).volume;
                if (capsules)
                {
                    cell.inherited = true;
                    const auto& capsule = capsules->at(i);
                    if (capsule.key != cell.key || capsule.revision != cell.revision ||
                        capsule.phases.bulk != cell.native_volume)
                        throw std::invalid_argument("General VoF exact adoption geometry/ownership differs.");
                    cell.phase.gas_volume = capsule.phases.gas;
                    cell.phase.carrier_volume = capsule.phases.carrier;
                    cell.roundoff = capsule.roundoff;
                    const auto& e = capsule.error;
                    cell.occupancy = {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound,
                        e.fit_residual, e.fit_bound, e.transferred_bound, e.fit_allocation_bound};
                    if (capsule.geometry_ready)
                        cell.retained_interface = capsule.geometry;
                }
                else
                    occupancy(cell);
                if (options.inventory_policy == PhaseInventoryPolicy::ExperimentalExact)
                {
                    if (capsules)
                    {
                        const auto& capsule = capsules->at(i);
                        if (capsule.policy != options.inventory_policy || !capsule.experimental_exact)
                            throw std::invalid_argument("Experimental adoption policy/payload differs.");
                        cell.experimental_exact = capsule.experimental_exact;
                    }
                    else
                        cell.experimental_exact = detail::experimental_initial(*cell.phase.geometry, cell.native_volume,
                            cell.phase.gas_volume, cell.phase.carrier_volume, options.byte_budget);
                }
                else if (capsules &&
                         (capsules->at(i).policy != PhaseInventoryPolicy::Finite || capsules->at(i).experimental_exact))
                    throw std::invalid_argument("Finite adoption cannot acquire experimental history.");
                if (!declarations.empty())
                    cell.declaration = declarations[i];
                else if (!capsules)
                {
                    // Exact absence proves pure geometry at fresh bootstrap.
                    // Persist that evidence through the existing fixed-bound
                    // declaration policy; never classify a nonzero minority by
                    // magnitude or replace an explicit/adopted declaration.
                    if (cell.phase.carrier_volume == 0 && cell.phase.gas_volume > 0)
                        cell.declaration = GeneralVoFPhaseGeometry::PureGas;
                    else if (cell.phase.gas_volume == 0 && cell.phase.carrier_volume > 0)
                        cell.declaration = GeneralVoFPhaseGeometry::PureCarrier;
                }
                // Validates independent occupancy without changing represented phases.
                validate_cell(cell, options);
                budget.add(cell.phase.geometry->storage_bytes());
                data->cells.emplace(order[i], std::move(cell));
            }
        });
    agree_options(options, comm);
    data->control = transaction(comm);
    if (transfer)
    {
        data->adopted_source = transfer->source_identity();
        data->adopted_source_control = transfer->source_control();
        data->adopted_transaction = transfer->transaction();
        data->adopted_target_control = data->control;
    }
    topology->draft = AMR::fetch_adaptive_native_draft(*hierarchy);
    complete_node_fans(*data, *topology);
    for (unsigned ring = 1; ring <= options.support_rings; ++ring)
    {
        std::vector<Key> additional;
        wire::collective(*comm,
            [&]
            {
                wire::Budget budget{options.byte_budget};
                std::map<Key, std::set<Key>> graph;
                for (const auto& tile : topology->draft.tiles)
                    if (tile.second)
                    {
                        budget.add(4 * (sizeof(Key) + 8 * sizeof(void*) + AMR::encode_adaptive_key(tile.first).size() +
                                           AMR::encode_adaptive_key(*tile.second).size()));
                        graph[tile.first].insert(*tile.second);
                        graph[*tile.second].insert(tile.first);
                    }
                std::set<Key> seen, front;
                for (const auto& [key, leaf] : hierarchy->owned())
                {
                    seen.insert(key);
                    front.insert(key);
                }
                for (const auto& [node, fan] : topology->node_fans)
                    for (const auto& key : fan)
                    {
                        seen.insert(key);
                        front.insert(key);
                    }
                for (unsigned depth = 0; depth <= ring; ++depth)
                {
                    std::set<Key> next;
                    for (const auto& key : front)
                        for (const auto& neighbor : graph[key])
                            if (seen.insert(neighbor).second)
                            {
                                if (seen.size() - hierarchy->owned().size() > options.maximum_halo_cells)
                                    throw std::invalid_argument("General VoF halo cell budget exhausted.");
                                budget.add(sizeof(Key) + AMR::encode_adaptive_key(neighbor).size());
                                next.insert(neighbor);
                            }
                    front = std::move(next);
                }
                for (const auto& key : seen)
                    if (!hierarchy->owned().contains(key))
                        additional.push_back(key);
            });
        topology->draft = AMR::fetch_adaptive_native_draft(*hierarchy, additional);
    }
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{options.byte_budget};
            budget.add(topology->draft.cell_bindings.size() * (sizeof(Cell) + sizeof(Key)));
            std::map<Key, const AMR::AdaptiveLeaf*> leaves;
            for (const auto& leaf : topology->draft.owned)
                leaves.emplace(leaf.key, &leaf);
            for (const auto& leaf : topology->draft.ghosts)
                leaves.emplace(leaf.key, &leaf);
            for (const auto& binding : topology->draft.cell_bindings)
                if (binding.full)
                {
                    auto found = data->cells.find(binding.key);
                    const auto& retained_leaf = *leaves.at(binding.key);
                    const auto retained_geometry =
                        retained_leaf.geometry ? retained_leaf.geometry : hierarchy->geometry(retained_leaf);
                    if (found != data->cells.end() && (!found->second.phase.geometry->equivalent(*retained_geometry) ||
                                                          found->second.native_volume != binding.volume))
                        throw std::invalid_argument("General VoF retained native geometry/metric differs from source.");
                    if (found == data->cells.end())
                    {
                        const auto& leaf = retained_leaf;
                        Cell cell;
                        cell.key = binding.key;
                        cell.owner = binding.owner;
                        cell.revision = leaf.revision;
                        cell.phase.geometry = retained_geometry;
                        budget.add(cell.phase.geometry->storage_bytes());
                        found = data->cells.emplace(binding.key, std::move(cell)).first;
                    }
                    found->second.gid = binding.gid;
                    found->second.centroid = binding.centroid;
                    found->second.native_volume = binding.volume;
                    if (!topology->by_gid.emplace(binding.gid, binding.key).second)
                        throw std::invalid_argument("General VoF duplicate native cell GID.");
                }
            for (const auto& tile : topology->draft.tiles)
            {
                if (!tile.exact_second_image.empty())
                    throw std::invalid_argument("General VoF periodic image sweeps are not qualified.");
                if (tile.second)
                {
                    topology->adjacency[tile.first].insert(*tile.second);
                    topology->adjacency[*tile.second].insert(tile.first);
                }
                if (hierarchy->owned().contains(tile.first) ||
                    (tile.second && hierarchy->owned().contains(*tile.second)))
                    topology->tiles.push_back(tile);
            }
            // Preserve canonical tile order in every incidence row. Transport
            // sums and node-wall packets use this same order after publication.
            const auto append_index = [&](auto& entries, const auto& key, std::size_t index)
            {
                if (!entries.contains(key))
                {
                    budget.add(sizeof(key) + sizeof(std::vector<std::size_t>) + 4 * sizeof(void*));
                    if constexpr (std::is_same_v<std::decay_t<decltype(key)>, Key>)
                        budget.add(AMR::encode_adaptive_key(key).size());
                }
                auto& indices = entries[key];
                if (!indices.empty() && indices.back() == index)
                    return;
                if (indices.size() == indices.capacity())
                {
                    const auto capacity = indices.capacity() ? 2 * indices.capacity() : 1;
                    budget.add((capacity - indices.capacity()) * sizeof(std::size_t));
                    indices.reserve(capacity);
                }
                indices.push_back(index);
            };
            for (std::size_t index = 0; index < topology->tiles.size(); ++index)
            {
                const auto& tile = topology->tiles[index];
                append_index(topology->cell_tiles, tile.first, index);
                if (tile.second)
                    append_index(topology->cell_tiles, *tile.second, index);
                for (auto node : tile.node_gids)
                    append_index(topology->node_tiles, node, index);
            }
            std::set<std::uint64_t> control_nodes;
            for (const auto& tile : topology->tiles)
                control_nodes.insert(tile.node_gids.begin(), tile.node_gids.end());
            for (const auto& node : topology->draft.nodes)
                if (control_nodes.contains(node.gid))
                    topology->nodes.push_back(node);
            std::map<std::uint64_t, std::map<AMR::AdaptiveBytes, Meshes::ExactPlanarFacePtr>> walls;
            std::map<std::uint64_t, std::set<Key>> incident;
            for (const auto& tile : topology->draft.tiles)
                for (auto node : tile.node_gids)
                    if (control_nodes.contains(node))
                    {
                        const auto& fan = topology->node_fans.at(node);
                        for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                            if (key && fan.contains(*key) && incident[node].insert(*key).second)
                                budget.add(sizeof(Key) + AMR::encode_adaptive_key(*key).size() + 4 * sizeof(void*));
                        if (tile.exterior)
                        {
                            if (!fan.contains(tile.first))
                                throw std::invalid_argument("General VoF wall is outside authenticated node fan.");
                            auto& catalog = walls[node];
                            const auto found = catalog.find(tile.identity);
                            if (found == catalog.end())
                            {
                                budget.add(sizeof(Data::Wall) + tile.identity.size() + 4 * sizeof(void*));
                                catalog.emplace(tile.identity, tile.geometry);
                            }
                            else if (found->second->encode_incidence() != tile.geometry->encode_incidence())
                                throw std::invalid_argument("General VoF canonical wall certificates differ.");
                        }
                    }
            for (const auto& node : topology->nodes)
            {
                if (incident[node.gid] != topology->node_fans.at(node.gid))
                    throw std::invalid_argument(
                        "General VoF wall catalog lacks a complete authenticated incident fan.");
                for (const auto& key : topology->node_fans.at(node.gid))
                    if (!leaves.contains(key))
                        throw std::invalid_argument("General VoF wall catalog requires full incident native shells.");
                auto& retained = topology->node_walls[node.gid];
                for (const auto& [identity, geometry] : walls[node.gid])
                    retained.push_back({identity, geometry});
            }
            for (const auto& [key, leaf] : hierarchy->owned())
            {
                topology->order.push_back(key);
                data->gas.push_back(data->cells.at(key).phase.gas_volume);
                data->carrier.push_back(data->cells.at(key).phase.carrier_volume);
            }
        });
    import_phases(*data);
    AMR::AdaptiveBytes schema;
    wire::collective(*comm,
        [&]
        {
            schema = hierarchy->configuration_descriptor();
            if (schema.size() > options.byte_budget)
                throw std::invalid_argument("General VoF native schema budget exhausted.");
        });
    topology->native = AMR::assemble_adaptive_native(std::move(topology->draft), schema, comm, options.byte_budget);
    data->hierarchy = hierarchy->with_partition(topology->native.partition, topology->native.mesh->owned_cell_map());
    data->observation = data->hierarchy->observation();
    wire::collective(*comm,
        [&]
        {
            topology->order.clear();
            data->gas.clear();
            data->carrier.clear();
            data->phases.clear();
            data->roundoff.clear();
            data->occupancy.clear();
            for (std::size_t c = 0; c < topology->native.mesh->num_owned_cells(); ++c)
            {
                const auto key = topology->by_gid.at(topology->native.mesh->cell_global_id(int(c)));
                topology->order.push_back(key);
                data->gas.push_back(data->cells.at(key).phase.gas_volume);
                data->carrier.push_back(data->cells.at(key).phase.carrier_volume);
                data->phases.push_back({data->cells.at(key).native_volume, data->cells.at(key).phase.gas_volume,
                    data->cells.at(key).phase.carrier_volume});
                data->roundoff.push_back(data->cells.at(key).roundoff);
                data->occupancy.push_back(data->cells.at(key).occupancy);
            }
            refresh_experimental_catalog(*data);
        });
    wire::collective(*comm,
        [&]
        {
            retained_budget(*data);
            current(*data);
        });
    data_ = std::move(data);
}
std::shared_ptr<const void> GeneralVoF::accepted_identity() const noexcept
{
    return data_;
}
std::uint64_t GeneralVoF::accepted_control() const noexcept
{
    return data_->control;
}
bool GeneralVoF::accepted_unchanged() const noexcept
{
    return data_->hierarchy->geometry_unchanged() && data_->hierarchy->observation() == data_->observation &&
           data_->supplied_hierarchy->geometry_unchanged() &&
           data_->supplied_hierarchy->observation() == data_->supplied_observation;
}
void GeneralVoF::validate_transfer_target(const ExactPhaseTransferResult& result) const
{
    const auto comm = result.source_hierarchy()->communicator();
    wire::collective(*comm,
        [&]
        {
            if (data_->hierarchy->communicator().getRawPtr() != comm.getRawPtr() ||
                result.target_hierarchy()->communicator().getRawPtr() != comm.getRawPtr())
                throw std::invalid_argument("General VoF transfer target communicator differs.");
            current(*data_);
            if (result.inventory_policy() != data_->options.inventory_policy)
                throw std::invalid_argument("General VoF transfer target policy differs.");
        });
    result.validate_observations();
    agree_transaction(comm, data_->control);
    wire::collective(*comm,
        [&]
        {
            current(*data_);
            if (data_->adopted_source.lock() != result.source_identity() ||
                data_->adopted_source_control != result.source_control() ||
                data_->adopted_transaction != result.transaction() || data_->adopted_target_control != data_->control ||
                data_->hierarchy->identity() != result.target_hierarchy()->identity() ||
                data_->hierarchy->topology_generation() != result.target_hierarchy()->topology_generation() ||
                data_->hierarchy->partition_generation() != result.target_hierarchy()->partition_generation() ||
                result.owned().size() != data_->hierarchy->owned().size())
                throw std::invalid_argument("General VoF is not the actual adopted transfer publication.");
            // The immutable hierarchy token is preserved only by with_partition;
            // its checked binding authenticates geometry/provider/domain/native GIDs.
            // Compare exact owned state directly. No fitted planes, geometry shells
            // or full source/target capsule arrays are reconstructed here.
            for (const auto& capsule : result.owned())
            {
                const auto found = data_->cells.find(capsule.key);
                if (found == data_->cells.end() || !data_->hierarchy->owned().contains(capsule.key))
                    throw std::invalid_argument("General VoF adopted transfer ownership differs.");
                const auto& cell = found->second;
                if (capsule.policy != data_->options.inventory_policy ||
                    bool(capsule.experimental_exact) != bool(cell.experimental_exact))
                    throw std::invalid_argument("General VoF adopted policy/payload differs.");
                if (cell.experimental_exact)
                {
                    const auto& a = *cell.experimental_exact;
                    const auto& b = *capsule.experimental_exact;
                    for (const auto& pair : {std::pair{&a.phases.bulk, &b.phases.bulk},
                             std::pair{&a.phases.gas, &b.phases.gas}, std::pair{&a.phases.carrier, &b.phases.carrier},
                             std::pair{&a.error.difference, &b.error.difference},
                             std::pair{&a.error.admitted_bound, &b.error.admitted_bound},
                             std::pair{&a.error.native_conversion, &b.error.native_conversion},
                             std::pair{&a.error.phase_input_bound, &b.error.phase_input_bound},
                             std::pair{&a.error.fit_residual, &b.error.fit_residual},
                             std::pair{&a.error.fit_bound, &b.error.fit_bound},
                             std::pair{&a.error.transferred_bound, &b.error.transferred_bound},
                             std::pair{&a.error.fit_allocation_bound, &b.error.fit_allocation_bound}})
                        if (pair.first->compare(*pair.second))
                            throw std::invalid_argument("Experimental adopted authority/provenance differs.");
                }
                const auto declaration = capsule.geometry.pure_gas
                                             ? GeneralVoFPhaseGeometry::PureGas
                                             : (capsule.geometry.pure_carrier ? GeneralVoFPhaseGeometry::PureCarrier
                                                                              : GeneralVoFPhaseGeometry::Infer);
                if (cell.revision != capsule.revision || cell.declaration != declaration ||
                    bool(cell.retained_interface) != capsule.geometry_ready ||
                    cell.native_volume != capsule.phases.bulk || cell.phase.gas_volume != capsule.phases.gas ||
                    cell.phase.carrier_volume != capsule.phases.carrier || cell.roundoff.gas != capsule.roundoff.gas ||
                    cell.roundoff.carrier != capsule.roundoff.carrier)
                    throw std::invalid_argument("General VoF adopted transfer finite phases differ.");
                const auto& e = cell.occupancy;
                const auto& p = capsule.error;
                for (const auto& pair :
                    {std::pair{&e.difference, &p.difference}, std::pair{&e.admitted_bound, &p.admitted_bound},
                        std::pair{&e.native_conversion, &p.native_conversion},
                        std::pair{&e.phase_input_bound, &p.phase_input_bound},
                        std::pair{&e.fit_residual, &p.fit_residual}, std::pair{&e.fit_bound, &p.fit_bound},
                        std::pair{&e.transferred_bound, &p.transferred_bound},
                        std::pair{&e.fit_allocation_bound, &p.fit_allocation_bound}})
                    if (*pair.first != *pair.second)
                        throw std::invalid_argument("General VoF adopted transfer error provenance differs.");
            }
        });
}
ExactPhaseTransferSource GeneralVoF::export_transfer_source(bool prepare_geometry) const
{
    const auto comm = data_->hierarchy->communicator();
    std::vector<ExactPhaseTransferCapsule> capsules;
    wire::collective(*comm,
        [&]
        {
            current(*data_);
            wire::Budget budget{data_->options.byte_budget};
            const auto interfaces = prepare_geometry ? reconstruct_local(*data_) : std::map<Key, PolyhedralInterface>{};
            for (const auto& key : data_->topology->order)
            {
                budget.add(sizeof(ExactPhaseTransferCapsule) + AMR::encode_adaptive_key(key).size());
                const auto& cell = data_->cells.at(key);
                ExactPhaseTransferCapsule capsule;
                capsule.key = key;
                capsule.policy = data_->options.inventory_policy;
                capsule.experimental_exact = cell.experimental_exact;
                capsule.revision = cell.revision;
                capsule.phases = {cell.native_volume, cell.phase.gas_volume, cell.phase.carrier_volume};
                capsule.roundoff = cell.roundoff;
                const auto& e = cell.occupancy;
                capsule.error = {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound,
                    e.fit_residual, e.fit_bound, e.transferred_bound, e.fit_allocation_bound};
                if (prepare_geometry)
                {
                    capsule.geometry = interfaces.at(key);
                    capsule.geometry_ready = true;
                }
                else if (cell.retained_interface)
                {
                    capsule.geometry = *cell.retained_interface;
                    capsule.geometry_ready = true;
                }
                if (cell.declaration != GeneralVoFPhaseGeometry::Infer || !phase_measures(cell).gas.sign() ||
                    !phase_measures(cell).carrier.sign())
                {
                    capsule.geometry.pure_gas =
                        cell.declaration == GeneralVoFPhaseGeometry::PureGas || !phase_measures(cell).carrier.sign();
                    capsule.geometry.pure_carrier =
                        cell.declaration == GeneralVoFPhaseGeometry::PureCarrier || !phase_measures(cell).gas.sign();
                    capsule.geometry_ready = true;
                }
                if (capsule.geometry_ready)
                {
                    const bool gas = phase_measures(cell).gas.compare(phase_measures(cell).carrier) <= 0;
                    const auto target = gas ? phase_measures(cell).gas : phase_measures(cell).carrier;
                    const auto raw =
                        phases(*Meshes::ExactConvexFragment::from_geometry(*cell.phase.geometry), capsule.geometry);
                    const auto fit_diagnostic = detail::round_phase_residual(target - (gas ? raw.gas : raw.carrier));
                    capsule.error.fit_residual = fit_diagnostic.value;
                    const auto view = Measure::from_value(
                        gas ? geometric_sample(cell).gas_volume : geometric_sample(cell).carrier_volume);
                    const auto conversion = target - view;
                    auto fit_bound = view.scaled(data_->options.reconstruction.volume_tolerance) +
                                     (conversion.sign() < 0 ? conversion.scaled(-1) : conversion);
                    if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
                    {
                        if (cell.experimental_exact)
                        {
                            const auto& exact = cell.experimental_exact->error;
                            fit_bound = exact.admitted_bound.compare(exact.fit_bound) >= 0 ? exact.admitted_bound
                                                                                           : exact.fit_bound;
                        }
                        else
                            fit_bound = finite_bound(std::max(e.admitted_bound, e.fit_bound));
                    }
                    capsule.error.fit_bound =
                        detail::phase_upper_bound(fit_bound + finite_bound(fit_diagnostic.absolute_error));
                    // Reuse the same independent admission proof as transport;
                    // merely observing the fit residual never enlarges its bound.
                    (void) detail::ExactPhaseLabelMap::create(raw, gas, target, fit_bound, data_->options.byte_budget);
                    if (cell.experimental_exact)
                    {
                        auto prepared = std::make_shared<ExperimentalPhaseInventory>(*cell.experimental_exact);
                        prepared->error.fit_residual = target - (gas ? raw.gas : raw.carrier);
                        prepared->error.fit_bound = fit_bound;
                        capsule.experimental_exact = std::move(prepared);
                    }
                }
                capsules.push_back(std::move(capsule));
            }
        });
    std::optional<ExactPhaseTransferSource> result;
    wire::collective(*comm,
        [&]
        {
            result.emplace(ExactPhaseTransferSource(
                data_->hierarchy, data_, data_->control, std::move(capsules), data_->options.inventory_policy));
        });
    return std::move(*result);
}
std::shared_ptr<const GeneralVoF> GeneralVoF::adopt_transfer(const ExactPhaseTransferResult& result) const
{
    const auto comm = data_->hierarchy->communicator();
    result.validate_observations();
    std::vector<Key> keys;
    std::vector<double> gas, carrier;
    std::vector<GeneralVoFPhaseGeometry> declarations;
    wire::collective(*comm,
        [&]
        {
            current(*data_);
            if (result.inventory_policy() != data_->options.inventory_policy)
                throw std::invalid_argument("General VoF adoption inventory policy differs.");
            if (result.source_identity().get() != data_.get() || result.source_control() != data_->control ||
                result.source_hierarchy().get() != data_->hierarchy.get())
                throw std::invalid_argument("General VoF exact transfer belongs to another accepted source.");
            wire::Budget budget{data_->options.byte_budget};
            for (const auto& capsule : result.owned())
            {
                budget.add(sizeof(Key) + 2 * sizeof(double) + sizeof(GeneralVoFPhaseGeometry) +
                           AMR::encode_adaptive_key(capsule.key).size());
                keys.push_back(capsule.key);
                gas.push_back(capsule.phases.gas);
                carrier.push_back(capsule.phases.carrier);
                declarations.push_back(capsule.geometry.pure_gas
                                           ? GeneralVoFPhaseGeometry::PureGas
                                           : (capsule.geometry.pure_carrier ? GeneralVoFPhaseGeometry::PureCarrier
                                                                            : GeneralVoFPhaseGeometry::Infer));
            }
        });
    agree_transaction(comm, result.transaction());
    return std::shared_ptr<const GeneralVoF>(new GeneralVoF(result.target_hierarchy(), std::move(keys), std::move(gas),
        std::move(carrier), data_->options, std::move(declarations), &result));
}
const std::shared_ptr<const MeshHandle<>>& GeneralVoF::mesh() const noexcept
{
    return data_->topology->native.mesh;
}
const std::shared_ptr<const GeneralVoF::Hierarchy>& GeneralVoF::hierarchy() const noexcept
{
    return data_->hierarchy;
}
const std::vector<GeneralVoF::Key>& GeneralVoF::owned_order() const noexcept
{
    return data_->topology->order;
}
const std::vector<double>& GeneralVoF::gas_volumes() const noexcept
{
    return data_->gas;
}
const std::vector<double>& GeneralVoF::carrier_volumes() const noexcept
{
    return data_->carrier;
}
const std::vector<PhaseVolumes>& GeneralVoF::phase_volumes() const noexcept
{
    return data_->phases;
}
PhaseInventoryPolicy GeneralVoF::inventory_policy() const noexcept
{
    return data_->options.inventory_policy;
}
std::vector<Meshes::ExactPhaseMeasures> GeneralVoF::inventory_measures() const
{
    if (inventory_policy() == PhaseInventoryPolicy::ExperimentalExact)
        return experimental_phases();
    std::vector<Meshes::ExactPhaseMeasures> result;
    result.reserve(data_->phases.size());
    for (const auto& value : data_->phases)
        result.push_back(
            {Measure::from_value(value.bulk), Measure::from_value(value.gas), Measure::from_value(value.carrier)});
    return result;
}
const std::vector<Meshes::ExactPhaseMeasures>& GeneralVoF::experimental_phases() const
{
    if (inventory_policy() != PhaseInventoryPolicy::ExperimentalExact || !data_->experimental_catalog)
        throw std::invalid_argument("Exact inventory access requires explicit ExperimentalExact policy.");
    return *data_->experimental_catalog;
}
const std::vector<PhaseRoundoff>& GeneralVoF::phase_roundoff_bounds() const noexcept
{
    return data_->roundoff;
}
std::vector<Meshes::ExactPhaseMeasures> GeneralVoF::geometric_phases() const
{
    agree_transaction(data_->hierarchy->communicator(), data_->control);
    std::vector<Meshes::ExactPhaseMeasures> result;
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            wire::Budget budget{data_->options.byte_budget};
            for (const auto& key : data_->topology->order)
            {
                const auto& cell = data_->cells.at(key);
                validate_cell(cell, data_->options);
                auto labels = geometric_labels(cell);
                budget.add(sizeof(labels) + labels.bulk.storage_bytes() + labels.gas.storage_bytes() +
                           labels.carrier.storage_bytes());
                result.push_back(std::move(labels));
            }
        });
    return result;
}
std::vector<double> GeneralVoF::geometric_gas_fractions() const
{
    agree_transaction(data_->hierarchy->communicator(), data_->control);
    std::vector<double> result;
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            wire::Budget budget{data_->options.byte_budget};
            const auto one = Measure::from_value(1);
            for (const auto& key : data_->topology->order)
            {
                const auto& cell = data_->cells.at(key);
                validate_cell(cell, data_->options);
                const auto labels = geometric_labels(cell);
                if (labels.bulk.sign() <= 0)
                    throw std::invalid_argument("General VoF geometric fraction has nonpositive actual volume.");
                const auto fraction = labels.gas.divided_by(labels.bulk, data_->options.byte_budget);
                if (fraction.sign() < 0 || fraction.compare(one) > 0)
                    throw std::invalid_argument("General VoF geometric gas fraction is outside [0,1].");
                budget.add(sizeof(double) + fraction.storage_bytes());
                const double view = double(fraction.enclosure().value);
                if (!std::isfinite(view) || view < 0 || view > 1)
                    throw std::invalid_argument("General VoF geometric fraction has no finite bounded view.");
                result.push_back(view);
            }
        });
    return result;
}
const GeneralVoFOptions& GeneralVoF::options() const noexcept
{
    return data_->options;
}
const std::vector<PhaseError>& GeneralVoF::initial_occupancy_errors() const noexcept
{
    return data_->occupancy;
}
std::vector<PolyhedralInterface> GeneralVoF::reconstruct() const
{
    std::vector<PolyhedralInterface> result;
    auto comm = data_->hierarchy->communicator();
    wire::collective(*comm,
        [&]
        {
            current(*data_);
            auto planes = reconstruct_local(*data_);
            for (const auto& key : data_->topology->order)
                result.push_back(planes.at(key));
        });
    return result;
}
std::vector<GeneralVoFReconstructionPatch> GeneralVoF::reconstruction_patches() const
{
    const auto planes = import_planes(*data_);
    std::vector<GeneralVoFReconstructionPatch> result;
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            for (const auto& [key, cell] : data_->cells)
                result.push_back({cell.gid, cell.owner, cell.phase.geometry, planes.at(key)});
        });
    return result;
}
const std::vector<AMR::AdaptiveNode>& GeneralVoF::nodes() const noexcept
{
    return data_->topology->nodes;
}
const std::vector<AMR::AdaptiveTile>& GeneralVoF::tiles() const noexcept
{
    return data_->topology->tiles;
}
const std::vector<std::size_t>& GeneralVoF::node_tiles(std::uint64_t node) const
{
    return data_->topology->node_tiles.at(node);
}
std::vector<Meshes::ConvexGeometryPtr> GeneralVoF::owned_geometry() const
{
    std::vector<Meshes::ConvexGeometryPtr> result;
    for (const auto& key : data_->topology->order)
        result.push_back(data_->cells.at(key).phase.geometry);
    return result;
}
std::vector<std::uint64_t> GeneralVoF::node_cells(std::uint64_t node) const
{
    std::set<std::uint64_t> cells;
    const auto& fan = data_->topology->node_fans.at(node);
    std::set<Key> seen = fan, front = fan;
    for (const auto& key : fan)
        cells.insert(data_->cells.at(key).gid);
    for (unsigned ring = 0; ring < data_->options.support_rings; ++ring)
    {
        std::set<Key> next;
        for (const auto& key : front)
        {
            auto found = data_->topology->adjacency.find(key);
            if (found == data_->topology->adjacency.end())
                continue;
            for (const auto& neighbor : found->second)
            {
                if (!data_->cells.contains(neighbor))
                    throw std::invalid_argument("General VoF canonical node support is incomplete.");
                if (seen.insert(neighbor).second)
                {
                    if (seen.size() > data_->options.maximum_stencil_cells)
                        throw std::invalid_argument("General VoF node support budget exhausted.");
                    next.insert(neighbor);
                    cells.insert(data_->cells.at(neighbor).gid);
                }
            }
        }
        front = std::move(next);
    }
    return {cells.begin(), cells.end()};
}
std::vector<std::uint64_t> GeneralVoF::cell_support(std::uint64_t gid) const
{
    const auto key = data_->topology->by_gid.at(gid);
    if (!data_->hierarchy->owned().contains(key))
        throw std::invalid_argument("General VoF cell support requires an owned seed.");
    std::set<Key> seen{key}, front{key};
    for (unsigned ring = 0; ring < data_->options.support_rings; ++ring)
    {
        std::set<Key> next;
        for (const auto& center : front)
        {
            const auto found = data_->topology->adjacency.find(center);
            if (found == data_->topology->adjacency.end())
                continue;
            for (const auto& neighbor : found->second)
                if (data_->cells.contains(neighbor) && seen.insert(neighbor).second)
                {
                    if (seen.size() > data_->options.maximum_stencil_cells)
                        throw std::invalid_argument("General VoF cell support budget exhausted.");
                    next.insert(neighbor);
                }
        }
        front = std::move(next);
    }
    std::vector<std::uint64_t> result;
    for (const auto& member : seen)
        result.push_back(data_->cells.at(member).gid);
    return result;
}
std::vector<GeneralVoFSupportCell> GeneralVoF::support_cells() const
{
    std::vector<GeneralVoFSupportCell> result;
    for (const auto& [key, cell] : data_->cells)
        result.push_back({cell.gid, cell.revision, cell.owner, cell.phase.geometry, cell.centroid, key});
    return result;
}
const std::vector<GeneralVoFDonorBudget>& GeneralVoF::Trial::donor_budgets() const noexcept
{
    return budgets_;
}
const std::vector<PhaseRoundoff>& GeneralVoF::Trial::phase_roundoff_bounds() const noexcept
{
    return roundoff_;
}

const std::vector<GeneralVoFNodeTrajectory>& GeneralVoF::Trial::node_trajectories() const noexcept
{
    return trajectories_;
}
const std::vector<GeneralVoFFaceContribution>& GeneralVoF::Trial::face_contributions() const noexcept
{
    return faces_;
}
const std::vector<GeneralVoFTransfer>& GeneralVoF::Trial::transfers() const noexcept
{
    return transfers_;
}
std::shared_ptr<const GeneralVoF> GeneralVoF::Trial::candidate() const
{
    if (!source_ || !candidate_)
        throw std::invalid_argument("General VoF trial has no candidate.");
    agree_transaction(source_->hierarchy->communicator(), transaction_);
    std::shared_ptr<const GeneralVoF> result;
    wire::collective(*source_->hierarchy->communicator(),
        [&]
        {
            current(*source_);
            if (candidate_->options.inventory_policy != source_->options.inventory_policy)
                throw std::invalid_argument("General VoF trial candidate inventory policy differs.");
            result = std::shared_ptr<const GeneralVoF>(new GeneralVoF(candidate_));
        });
    return result;
}
namespace
{
// Sharded equality authorities compare full certificates/represented controls,
// never a geometry hash alone. All ranks, including empty owners, participate.
void controls(const GeneralVoF::Data& data, std::span<const GeneralVoFNodeVelocity> nodes,
    std::span<const GeneralVoFFaceVolume> faces, double dt, std::map<std::uint64_t, Meshes::GeometryPoint>& velocity,
    std::map<std::uint64_t, Measure>& flux, std::map<std::uint64_t, Meshes::ExactNodeTrajectory>& trajectories)
{
    auto comm = data.hierarchy->communicator();
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            current(data);
            if (!std::isfinite(dt) || dt < 0 || nodes.size() != data.topology->nodes.size() || faces.size() != data.topology->tiles.size())
                throw std::invalid_argument("General VoF controls do not cover retained nodes/faces.");
            wire::Budget budget{data.options.byte_budget};
            for (const auto& node : nodes)
            {
                for (auto x : node.velocity)
                    if (!std::isfinite(x))
                        throw std::invalid_argument("General VoF node velocity is nonfinite.");
                if (!velocity.emplace(node.node, node.velocity).second)
                    throw std::invalid_argument("General VoF repeats a node control.");
            }
            for (const auto& face : faces)
                if (!std::isfinite(face.integrated_volume) ||
                    !flux
                        .emplace(face.face, face.exact_integrated_volume ? *face.exact_integrated_volume
                                                                         : Measure::from_value(face.integrated_volume))
                        .second)
                    throw std::invalid_argument("General VoF invalid/repeated face control.");
            for (const auto& node : data.topology->nodes)
            {
                auto v = velocity.at(node.gid);
                std::vector<Meshes::ExactPlanarFacePtr> walls;
                const auto& catalog = data.topology->node_walls.at(node.gid);
                budget.add(catalog.size() * sizeof(Meshes::ExactPlanarFacePtr));
                for (const auto& wall : catalog)
                    walls.push_back(wall.geometry);
                Meshes::ExactTrajectoryLimits limits;
                limits.byte_budget = data.options.byte_budget;
                auto trajectory =
                    Meshes::ExactNodeTrajectory::backtrace_constrained(node.exact_identity, v, dt, walls, limits);
                budget.add(sizeof(trajectory) + trajectory.point.size() + trajectory.departure.size());
                budget.packet(56 + node.exact_identity.size());
                wire::Writer w;
                w.word(0);
                w.word(node.gid);
                w.bytes(node.exact_identity);
                for (auto x : v)
                    real(w, x);
                real(w, dt);
                budget.add(16 + trajectory.departure.size());
                w.bytes(trajectory.departure);
                w.word(catalog.size());
                for (const auto& wall : catalog)
                {
                    const auto certificate = wall.geometry->encode_incidence();
                    budget.add(16 + wall.identity.size() + certificate.size());
                    w.bytes(wall.identity);
                    w.bytes(certificate);
                }
                trajectories.emplace(node.gid, std::move(trajectory));
                packets.push_back({int(node.gid % comm->getSize()), std::move(w.data)});
            }
            for (const auto& tile : data.topology->tiles)
            {
                const auto phi = flux.at(tile.face_gid);
                if ((tile.exterior || dt == 0) && phi.sign() != 0)
                    throw std::invalid_argument("General VoF physical boundary has nonzero flux.");
                auto face = tile.geometry->encode_incidence();
                auto exact_phi = phi.encode(data.options.byte_budget);
                budget.packet(face.size() + exact_phi.size() + 32);
                wire::Writer w;
                w.word(1);
                w.word(tile.face_gid);
                w.bytes(face);
                w.bytes(exact_phi);
                packets.push_back({int(tile.face_gid % comm->getSize()), std::move(w.data)});
            }
            budget.packet(32);
            wire::Writer w;
            w.word(2);
            w.word(0);
            real(w, dt);
            w.word(data.control);
            packets.push_back({0, std::move(w.data)});
        });
    auto received = exchange_owner_packets(packets, comm, data.options.byte_budget);
    wire::collective(*comm,
        [&]
        {
            std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<char>> canonical;
            wire::Budget budget{data.options.byte_budget};
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto type = r.word(), id = r.word();
                if (type > 2 || int(id % comm->getSize()) != comm->getRank())
                    throw std::invalid_argument("General VoF control reached wrong authority.");
                budget.add(packet.bytes.size() +
                           sizeof(std::pair<const std::pair<std::uint64_t, std::uint64_t>, std::vector<char>>));
                auto [found, inserted] = canonical.emplace(std::pair{type, id}, packet.bytes);
                if (!inserted && found->second != packet.bytes)
                    throw std::invalid_argument("General VoF shared node/face controls disagree.");
            }
        });
}
std::vector<Meshes::ExactSweepDonor> donor_support(
    const GeneralVoF::Data& data, const Key& first, const std::optional<Key>& second = {})
{
    std::set<Key> seen{first}, front{first};
    if (second)
    {
        seen.insert(*second);
        front.insert(*second);
    }
    for (unsigned ring = 0; ring < data.options.support_rings; ++ring)
    {
        std::set<Key> next;
        for (const auto& key : front)
        {
            auto found = data.topology->adjacency.find(key);
            if (found == data.topology->adjacency.end())
                continue;
            for (const auto& neighbor : found->second)
                if (data.cells.contains(neighbor) && seen.insert(neighbor).second)
                {
                    if (seen.size() > data.options.maximum_stencil_cells)
                        throw std::invalid_argument("General VoF donor support budget exhausted.");
                    next.insert(neighbor);
                }
        }
        front = std::move(next);
    }
    std::vector<Meshes::ExactSweepDonor> result;
    for (const auto& key : seen)
    {
        const auto& cell = data.cells.at(key);
        result.push_back({cell.gid, cell.phase.geometry});
    }
    return result;
}
std::vector<Meshes::ExactSweepDonor> face_donor_support(
    const GeneralVoF::Data& data, const AMR::AdaptiveTile& face, wire::Budget& budget)
{
    std::map<Key, Meshes::ExactSweepDonor> selected;
    auto insert = [&](const Key& key)
    {
        const auto cell = data.cells.find(key);
        if (cell == data.cells.end() || !cell->second.phase.geometry)
            throw std::invalid_argument("General VoF face donor fan requires an authenticated full native shell.");
        if (!selected.contains(key))
        {
            if (selected.size() >= data.options.maximum_stencil_cells)
                throw std::invalid_argument("General VoF complete face donor fan budget exhausted.");
            budget.add(sizeof(Key) + sizeof(Meshes::ExactSweepDonor) + AMR::encode_adaptive_key(key).size() +
                       8 * sizeof(void*));
            selected.emplace(key, Meshes::ExactSweepDonor{cell->second.gid, cell->second.phase.geometry});
        }
    };
    for (const auto& donor : donor_support(data, face.first, face.second))
        insert(data.topology->by_gid.at(donor.id));
    // A high-valence native corner is not bounded by two face hops. These
    // complete fans were registered by ALL incident owners and their full
    // shells fetched before construction; geometric coincidence is irrelevant.
    for (auto node : face.node_gids)
        for (const auto& key : data.topology->node_fans.at(node))
            insert(key);
    std::vector<Meshes::ExactSweepDonor> result;
    budget.add(selected.size() * sizeof(Meshes::ExactSweepDonor));
    result.reserve(selected.size());
    for (const auto& [key, donor] : selected)
        result.push_back(donor);
    return result;
}
struct ExactAmounts
{
    Measure bulk, gas, carrier;
    void add(const Meshes::ExactPhaseMeasures& p, int sign = 1)
    {
        bulk = bulk + p.bulk.scaled(sign);
        gas = gas + p.gas.scaled(sign);
        carrier = carrier + p.carrier.scaled(sign);
    }
};
void transport_charge(wire::Budget& budget, std::size_t bytes, const char* phase, const char* kind = "")
{
    if (bytes > budget.limit - budget.used)
        throw std::invalid_argument(std::string("General VoF ") + phase +
                                    " budget exhausted: used=" + std::to_string(budget.used) +
                                    ", requested=" + std::to_string(bytes) + ", limit=" + std::to_string(budget.limit) +
                                    ", remaining=" + std::to_string(budget.limit - budget.used));
    budget.add(bytes);
}
struct ExactStorageBudget
{
    wire::Budget& bytes;
    std::set<const void*> fragments;
    const char* phase = "transport retained exact";
    void fragment(const Meshes::ExactConvexFragment& value)
    {
        if (fragments.insert(value.storage_identity()).second)
            transport_charge(bytes, value.storage_bytes() + 4 * sizeof(void*), phase, " fragment");
    }
    void measure(const Measure& value)
    {
        // Accumulator intermediates may already have died and their allocator
        // address may be reused. Charge each produced value conservatively;
        // pointer deduplication is valid only for the retained live catalog.
        transport_charge(bytes, value.storage_bytes(), phase, " measure");
    }
    void phases(const Meshes::ExactPhaseMeasures& value)
    {
        measure(value.bulk);
        measure(value.gas);
        measure(value.carrier);
    }
};
} // namespace
GeneralVoF::Trial GeneralVoF::plan_transport(std::span<const GeneralVoFNodeVelocity> node_velocity,
    std::span<const GeneralVoFFaceVolume> face_volume, double timestep) const
{
    const auto& data = *data_;
    auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::map<std::uint64_t, Meshes::GeometryPoint> velocity;
    std::map<std::uint64_t, Measure> flux;
    std::map<std::uint64_t, Meshes::ExactNodeTrajectory> canonical_trajectories;
    controls(data, node_velocity, face_volume, timestep, velocity, flux, canonical_trajectories);
    auto retain_trajectories = [&](Trial& trial)
    {
        wire::collective(*comm,
            [&]
            {
                wire::Budget budget{limit};
                for (const auto& [node, trajectory] : canonical_trajectories)
                {
                    budget.add(
                        sizeof(GeneralVoFNodeTrajectory) + trajectory.point.size() + trajectory.departure.size());
                    trial.trajectories_.push_back({node, trajectory});
                }
            });
    };
    wire::collective(*comm,
        [&]
        {
            for (const auto& key : data.topology->order)
            {
                Measure divergence;
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    const int sign = tile.first == key ? 1 : (tile.second && *tile.second == key ? -1 : 0);
                    if (sign)
                        divergence = divergence + flux.at(tile.face_gid).scaled(sign);
                }
                if (divergence.sign())
                    throw std::invalid_argument("General VoF projected flux violates source-free occupancy.");
            }
        });
    int local_motion = 0, motion = 0;
    for (const auto& [node, trajectory] : canonical_trajectories)
        local_motion |= trajectory.point != trajectory.departure;
    for (const auto& [face, value] : flux)
        local_motion |= value.sign() != 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_motion, &motion);
    if (!motion)
    {
        // A proven identity trajectory and zero canonical Phi need no inferred
        // interface normal. Net-zero Phi alone is NOT an identity test: crossed
        // nonzero trajectories continue through the full exact sweep proof.
        Trial trial;
        trial.source_ = data_;
        trial.candidate_ = data_;
        trial.transaction_ = transaction(comm);
        retain_trajectories(trial);
        wire::collective(*comm,
            [&]
            {
                wire::Budget budget{limit};
                budget.add(data.topology->order.size() * sizeof(PhaseRoundoff));
                trial.roundoff_.resize(data.topology->order.size());
                for (const auto& key : data.topology->order)
                {
                    budget.add(sizeof(GeneralVoFTransfer) + sizeof(GeneralVoFDonorBudget));
                    const auto& cell = data.cells.at(key);
                    trial.transfers_.push_back({cell.gid, cell.gid, cell.owner, cell.owner, phase_measures(cell),
                        diagnostic(phase_measures(cell))});
                    Meshes::ExactPhaseMeasures zero;
                    trial.budgets_.push_back({cell.gid, phase_measures(cell), zero, phase_measures(cell),
                        diagnostic(phase_measures(cell)), diagnostic(zero), diagnostic(phase_measures(cell))});
                }
            });
        return trial;
    }
    auto planes = import_planes(data);
    std::map<Key, detail::ExactPhaseLabelMap> labels;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, cell] : data.cells)
            {
                const auto raw =
                    phases(*Meshes::ExactConvexFragment::from_geometry(*cell.phase.geometry), planes.at(key));
                const bool gas = phase_measures(cell).gas.compare(phase_measures(cell).carrier) <= 0;
                const auto target = gas ? phase_measures(cell).gas : phase_measures(cell).carrier;
                const auto view = Measure::from_value(gas ? cell.phase.gas_volume : cell.phase.carrier_volume);
                const auto conversion = target - view;
                const auto absolute_conversion = conversion.sign() < 0 ? conversion.scaled(-1) : conversion;
                auto bound = view.scaled(data.options.reconstruction.volume_tolerance) + absolute_conversion;
                if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
                {
                    if (cell.experimental_exact)
                    {
                        const auto& e = cell.experimental_exact->error;
                        bound = e.admitted_bound.compare(e.fit_bound) >= 0 ? e.admitted_bound : e.fit_bound;
                    }
                    else
                        bound = finite_bound(std::max(cell.occupancy.admitted_bound, cell.occupancy.fit_bound));
                }
                auto label = detail::ExactPhaseLabelMap::create(raw, gas, target, bound, limit);
                budget.add(label.storage_bytes());
                labels.emplace(key, std::move(label));
            }
        });
    std::map<std::uint64_t, std::shared_ptr<const Meshes::ExactSweptFace>> sweeps;
    std::map<std::uint64_t, std::vector<Meshes::ExactSweepIntersection>> face_parts;
    std::map<std::uint64_t, std::vector<Meshes::ExactSweepDonor>> face_donors;
    std::vector<Meshes::ExactTransportAtom> local_atoms;
    std::map<Key, std::pair<Measure, Measure>> signed_repair;
    Trial trial;
    trial.source_ = data_;
    trial.transaction_ = transaction(comm);
    retain_trajectories(trial);
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            ExactStorageBudget exact_storage{budget};
            detail::ExactLiveStorage raw_live(budget, limit / 4);
            budget.add(data.topology->order.size() * sizeof(PhaseRoundoff));
            trial.roundoff_.resize(data.topology->order.size());
            for (const auto& tile : data.topology->tiles)
            {
                std::vector<Meshes::ExactNodeTrajectory> trajectories;
                budget.add(tile.node_gids.size() * sizeof(Meshes::ExactNodeTrajectory));
                for (auto gid : tile.node_gids)
                    trajectories.push_back(canonical_trajectories.at(gid));
                auto face = Meshes::ExactSweptFace::make(
                    *tile.geometry, trajectories, flux.at(tile.face_gid), data.options.sweep);
                auto candidates = face_donor_support(data, tile, budget);
                auto pieces = [&]
                {
                    try
                    {
                        return face->partition(candidates, data.options.sweep);
                    }
                    catch (const std::invalid_argument& error)
                    {
                        throw std::invalid_argument(
                            "General VoF raw face=" + std::to_string(tile.face_gid) +
                            " first=" + std::to_string(data.cells.at(tile.first).gid) +
                            " second=" + (tile.second ? std::to_string(data.cells.at(*tile.second).gid) : "wall") +
                            " source_control=" + std::to_string(data.control) + ": " + error.what());
                    }
                }();
                for (const auto& simplex : face->simplices())
                    exact_storage.fragment(*simplex.geometry);
                for (const auto& piece : pieces)
                    exact_storage.fragment(*piece.geometry);
                sweeps.emplace(tile.face_gid, std::move(face));
                face_parts.emplace(tile.face_gid, std::move(pieces));
                face_donors.emplace(tile.face_gid, std::move(candidates));
            }
            for (const auto& key : data.topology->order)
            {
                const auto& cell = data.cells.at(key);
                Meshes::ExactSweepRecipient recipient{cell.gid, cell.phase.geometry, {}};
                Measure divergence;
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    int sign = tile.first == key ? 1 : (tile.second && *tile.second == key ? -1 : 0);
                    if (!sign)
                        continue;
                    recipient.outward_faces.push_back({sweeps.at(tile.face_gid), sign});
                    divergence = divergence + flux.at(tile.face_gid).scaled(sign);
                }
                if (divergence.sign())
                    throw std::invalid_argument("General VoF projected flux violates source-free occupancy.");
                std::map<Key, Meshes::ExactSweepDonor> selected;
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    for (const auto& donor : face_donors.at(tile.face_gid))
                    {
                        const auto& donor_key = data.topology->by_gid.at(donor.id);
                        if (!selected.contains(donor_key))
                        {
                            if (selected.size() >= data.options.maximum_stencil_cells)
                                throw std::invalid_argument(
                                    "General VoF recipient face-donor union budget exhausted.");
                            budget.add(sizeof(Key) + sizeof(Meshes::ExactSweepDonor) +
                                       AMR::encode_adaptive_key(donor_key).size() + 8 * sizeof(void*));
                            selected.emplace(donor_key, donor);
                        }
                    }
                }
                std::vector<Meshes::ExactSweepDonor> donors;
                budget.add(selected.size() * sizeof(Meshes::ExactSweepDonor));
                for (const auto& [donor_key, donor] : selected)
                    donors.push_back(donor);
                auto partition = [&]
                {
                    try
                    {
                        return Meshes::ExactTransportPartition::local(
                            donors, std::span(&recipient, 1), data.options.sweep);
                    }
                    catch (const std::invalid_argument& error)
                    {
                        std::ostringstream context;
                        context << "General VoF local recipient=" << cell.gid << " source_control=" << data.control
                                << " donors=[";
                        for (const auto& donor : donors)
                            context << donor.id << ',';
                        context << "] incident_faces=[";
                        for (const auto index : data.topology->cell_tiles.at(key))
                        {
                            const auto& tile = data.topology->tiles[index];
                            std::uint64_t hash = 1469598103934665603ULL;
                            for (unsigned char byte : tile.geometry->encode())
                                hash = (hash ^ byte) * 1099511628211ULL;
                            context << tile.face_gid << ':' << hash << ',';
                        }
                        context << "]: " << error.what();
                        throw std::invalid_argument(context.str());
                    }
                }();
                budget.add(partition.atoms().size() * sizeof(Meshes::ExactTransportAtom));
                for (const auto& atom : partition.atoms())
                    exact_storage.fragment(*atom.geometry);
                local_atoms.insert(local_atoms.end(), partition.atoms().begin(), partition.atoms().end());
                // Independently verify phase row identities against the canonical
                // signed face ledger before any native-range conversion.
                std::map<std::uint64_t, detail::ExactPhaseAccumulator> row;
                const auto row_for = [&](std::uint64_t donor) -> detail::ExactPhaseAccumulator&
                {
                    if (!row.contains(donor))
                        budget.add(sizeof(decltype(row)::value_type) + 4 * sizeof(void*));
                    return row.try_emplace(donor, raw_live).first->second;
                };
                for (const auto& atom : partition.atoms())
                    row_for(atom.donor).add(phases(*atom.geometry, planes.at(data.topology->by_gid.at(atom.donor))));
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    const int sign = tile.first == key ? 1 : (tile.second && *tile.second == key ? -1 : 0);
                    if (!sign)
                        continue;
                    for (const auto& piece : face_parts.at(tile.face_gid))
                        row_for(piece.donor)
                            .add(phases(*piece.geometry, planes.at(data.topology->by_gid.at(piece.donor))), sign * piece.sign);
                }
                const auto original =
                    phases(*Meshes::ExactConvexFragment::from_geometry(*cell.phase.geometry), planes.at(key));
                row_for(cell.gid).add(original, -1);
                for (const auto& [donor, amount] : row)
                {
                    // Keep the original MAPPED-zero gate, including singular
                    // donor maps; raw-zero alone is not the same acceptance.
                    const auto paired = amount.apply(labels.at(data.topology->by_gid.at(donor)));
                    if (paired.bulk.sign() || paired.gas.sign() || paired.carrier.sign())
                        throw std::invalid_argument(
                            "General VoF signed donor face ledger differs from positive recipient atoms.");
                }
            }
            for (const auto& tile : data.topology->tiles)
                if (tile.first_owner == comm->getRank())
                {
                    using GroupKey = std::pair<std::uint64_t, int>;
                    std::map<GroupKey, detail::ExactPhaseAccumulator> grouped;
                    for (const auto& piece : face_parts.at(tile.face_gid))
                    {
                        const GroupKey key{piece.donor, piece.sign};
                        if (!grouped.contains(key))
                            budget.add(sizeof(decltype(grouped)::value_type) + 4 * sizeof(void*));
                        grouped.try_emplace(key, raw_live)
                            .first->second.add(phases(*piece.geometry, planes.at(data.topology->by_gid.at(piece.donor))));
                    }
                    for (const auto& [group, amount] : grouped)
                    {
                        const auto& key = data.topology->by_gid.at(group.first);
                        const auto& donor = data.cells.at(key);
                        auto paired = amount.apply(labels.at(key));
                        detail::ExactLiveStorage::Scope paired_scope(
                            raw_live, {&paired.bulk, &paired.gas, &paired.carrier});
                        auto difference = raw_live.difference(paired.gas, amount.raw().gas);
                        detail::ExactLiveStorage::Scope difference_scope(raw_live, {&difference});
                        auto absolute = difference.sign() < 0 ? raw_live.scaled(difference, -1) : difference;
                        detail::ExactLiveStorage::Scope absolute_scope(raw_live, {&absolute});
                        if (!signed_repair.contains(key))
                            budget.add(sizeof(decltype(signed_repair)::value_type) + 4 * sizeof(void*) +
                                       AMR::encode_adaptive_key(key).size());
                        auto& repair = signed_repair[key];
                        raw_live.track(repair.first);
                        raw_live.track(repair.second);
                        // create() changes only a or b: on positive pieces the
                        // gas correction has one fixed sign. Hence this grouped
                        // absolute value equals the original sum of absolutes.
                        raw_live.add(repair.first, absolute);
                        // Preserve the original per-PIECE allowance and keep
                        // opposite simplex signs in distinct groups.
                        if (amount.count() > 9007199254740992ULL)
                            throw std::invalid_argument(
                                "Exact phase group count exceeds exact binary64 integer range.");
                        auto bound = raw_live.scaled(labels.at(key).bound(), double(amount.count()));
                        detail::ExactLiveStorage::Scope bound_scope(raw_live, {&bound});
                        raw_live.add(repair.second, bound);
                        budget.add(3 * sizeof(GeneralVoFFaceContribution));
                        raw_live.retain(paired);
                        trial.faces_.push_back(
                            {tile.face_gid, donor.gid, donor.owner, group.second, paired, diagnostic(paired)});
                    }
                }
        });
    std::vector<OwnerPacket> repair_packets;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, repair] : signed_repair)
            {
                if (repair.first.compare(repair.second) > 0)
                    throw std::invalid_argument("General VoF cumulative signed phase correction exceeds bound.");
                const auto& donor = data.cells.at(key);
                const auto bytes = AMR::encode_adaptive_key(key);
                const auto absolute = repair.first.encode(limit), bound = repair.second.encode(limit);
                budget.packet(bytes.size() + absolute.size() + bound.size() + 40);
                wire::Writer w;
                w.bytes(bytes);
                w.word(donor.revision);
                w.bytes(absolute);
                w.bytes(bound);
                repair_packets.push_back({donor.owner, std::move(w.data)});
            }
        });
    auto received_repairs = exchange_owner_packets(repair_packets, comm, limit);
    std::map<Key, std::pair<Measure, Measure>> donor_signed_repair;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            ExactStorageBudget exact_storage{budget};
            std::set<std::pair<int, Key>> seen;
            for (const auto& packet : received_repairs.packets)
            {
                wire::Reader r{packet.bytes};
                const auto key = AMR::decode_adaptive_key(r.bytes());
                const auto revision = r.word();
                const auto& donor = data.cells.at(key);
                if (donor.owner != comm->getRank() || donor.revision != revision ||
                    !seen.emplace(packet.peer, key).second)
                    throw std::invalid_argument("General VoF unauthenticated signed correction record.");
                const auto absolute = Measure::decode(r.bytes(), limit), bound = Measure::decode(r.bytes(), limit);
                r.done();
                if (absolute.sign() < 0 || bound.sign() < 0 || absolute.compare(bound) > 0)
                    throw std::invalid_argument("General VoF invalid signed correction bound.");
                auto& total = donor_signed_repair[key];
                total.first = total.first + absolute;
                total.second = total.second + bound;
                exact_storage.measure(total.first);
                exact_storage.measure(total.second);
            }
        });
    std::vector<OwnerPacket>().swap(repair_packets);
    std::vector<OwnerPacket>().swap(received_repairs.packets);
    // Positive route atoms travel to the ORIGINAL actual donor owner. The
    // owner certifies a complete/disjoint source cover across all recipients.
    std::vector<OwnerPacket> atom_packets;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (std::size_t index = 0; index < local_atoms.size(); ++index)
            {
                const auto& atom = local_atoms[index];
                const auto& donor = data.cells.at(data.topology->by_gid.at(atom.donor));
                const auto& recipient = data.cells.at(data.topology->by_gid.at(atom.recipient));
                const bool local = donor.owner == comm->getRank();
                auto dk = AMR::encode_adaptive_key(donor.key), rk = AMR::encode_adaptive_key(recipient.key);
                std::vector<char> geometry;
                if (!local)
                    geometry = atom.geometry->encode(limit);
                budget.packet(dk.size() + rk.size() + geometry.size() + 64);
                wire::Writer w;
                w.bytes(dk);
                w.word(donor.revision);
                w.word(donor.gid);
                w.bytes(rk);
                w.word(recipient.gid);
                w.word(local ? 1 : 0);
                if (local)
                    w.word(index);
                else
                    w.bytes(geometry);
                atom_packets.push_back({donor.owner, std::move(w.data)});
            }
        });
    auto incoming = exchange_owner_packets(atom_packets, comm, limit);
    std::vector<OwnerPacket>().swap(atom_packets);
    struct Routed
    {
        Key donor, recipient;
        std::uint64_t recipient_gid = 0;
        int recipient_owner = -1;
        Meshes::ExactTransportAtom atom;
    };
    std::vector<Routed> routed;
    std::vector<Key> recipient_keys;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                auto dk = AMR::decode_adaptive_key(r.bytes());
                auto revision = r.word(), gid = r.word();
                auto rk = AMR::decode_adaptive_key(r.bytes());
                auto recipient = r.word();
                const auto tag = r.word();
                const auto& donor = data.cells.at(dk);
                if (donor.owner != comm->getRank() || donor.revision != revision || donor.gid != gid)
                    throw std::invalid_argument("General VoF atom donor authentication failed.");
                Meshes::ExactConvexFragmentPtr fragment;
                if (tag == 1)
                {
                    // Only a self packet can name this invocation's immutable,
                    // already certified local atom. Remote data never acquires
                    // certified provenance through an integer reference.
                    const auto index = r.word();
                    r.done();
                    if (packet.peer != comm->getRank() || index >= local_atoms.size())
                        throw std::invalid_argument("General VoF foreign/out-of-range local atom reference.");
                    const auto& original = local_atoms[std::size_t(index)];
                    const auto& original_donor = data.cells.at(data.topology->by_gid.at(original.donor));
                    const auto& original_recipient = data.cells.at(data.topology->by_gid.at(original.recipient));
                    if (!original.geometry || original.donor != gid || original.recipient != recipient ||
                        original_donor.key != dk || original_recipient.key != rk ||
                        original_recipient.owner != comm->getRank() || original_recipient.gid != recipient)
                        throw std::invalid_argument("General VoF local atom metadata mismatch.");
                    fragment = original.geometry;
                    budget.add(sizeof(Routed));
                }
                else if (tag == 0)
                {
                    if (packet.peer == comm->getRank())
                        throw std::invalid_argument("General VoF self atom omitted its local reference.");
                    auto bytes = r.bytes();
                    r.done();
                    budget.add(sizeof(Routed) + bytes.size());
                    fragment = Meshes::ExactConvexFragment::decode(bytes, limit);
                }
                else
                    throw std::invalid_argument("General VoF invalid atom route tag.");
                // Charge the referenced immutable payload in the receiver's
                // existing phase bound as well; aliasing does not grow its cap.
                budget.add(fragment->storage_bytes());
                routed.push_back({dk, rk, recipient, packet.peer, {gid, recipient, std::move(fragment)}});
                recipient_keys.push_back(rk);
            }
        });
    std::vector<OwnerPacket>().swap(incoming.packets);
    std::vector<Meshes::ExactTransportAtom>().swap(local_atoms);
    auto recipient_owners = data.hierarchy->lookup_cells(recipient_keys);
    struct LocalAmount
    {
        Key donor, recipient;
        std::uint64_t donor_gid = 0, donor_revision = 0, recipient_gid = 0;
        ExactAmounts amount;
    };
    std::vector<LocalAmount> local_amounts;
    std::vector<OwnerPacket> transfer_packets;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            detail::ExactLiveStorage live(budget, limit / 4);
            std::map<Key, std::vector<Meshes::ExactTransportAtom>> donor_atoms;
            std::map<Key, std::map<Key, ExactAmounts>> amounts;
            std::map<Key, Measure> route_repair;
            ExactStorageBudget exact_storage{budget};
            exact_storage.phase = "donor phase retained exact";
            std::map<Key, std::pair<int, std::uint64_t>> recipients;
            const auto metadata = [&](const Key& key, std::size_t record)
            {
                transport_charge(budget, record + 4 * sizeof(void*), "donor phase map metadata");
                std::visit(
                    [&](const auto& ancestry)
                    {
                        const auto path = [&](const auto& values)
                        {
                            if (values.capacity() > limit / sizeof(std::uint64_t))
                                throw std::invalid_argument("Donor phase key capacity exceeds budget.");
                            transport_charge(
                                budget, values.capacity() * sizeof(std::uint64_t), "donor phase key payload");
                        };
                        if constexpr (requires { ancestry.path; })
                            path(ancestry.path);
                        else if constexpr (requires { ancestry.base.path; })
                        {
                            path(ancestry.base.path);
                            path(ancestry.axial.path);
                        }
                    },
                    key.ancestry);
            };
            const auto amount_for = [&](const Key& donor, const Key& recipient) -> ExactAmounts&
            {
                if (!amounts.contains(donor))
                    metadata(donor, sizeof(decltype(amounts)::value_type));
                auto& row = amounts[donor];
                if (!row.contains(recipient))
                    metadata(recipient, sizeof(std::map<Key, ExactAmounts>::value_type));
                auto& result = row[recipient];
                live.track(result.bulk);
                live.track(result.gas);
                live.track(result.carrier);
                return result;
            };
            for (std::size_t i = 0; i < routed.size(); ++i)
            {
                const auto& route = routed[i];
                const auto& owner = recipient_owners[i];
                if (owner.owner != route.recipient_owner || owner.address != route.recipient_gid)
                    throw std::invalid_argument("General VoF atom recipient authentication failed.");
                if (!donor_atoms.contains(route.donor))
                    metadata(route.donor, sizeof(decltype(donor_atoms)::value_type));
                transport_charge(budget, 3 * sizeof(Meshes::ExactTransportAtom), "donor phase atom handles");
                donor_atoms[route.donor].push_back(route.atom);
                exact_storage.fragment(*route.atom.geometry);
                auto raw = phases(*route.atom.geometry, planes.at(route.donor));
                detail::ExactLiveStorage::Scope raw_scope(live, {&raw.bulk, &raw.gas, &raw.carrier});
                auto paired = labels.at(route.donor).apply(raw, live.operation_bytes());
                detail::ExactLiveStorage::Scope paired_scope(live, {&paired.bulk, &paired.gas, &paired.carrier});
                auto difference = live.difference(paired.gas, raw.gas);
                detail::ExactLiveStorage::Scope difference_scope(live, {&difference});
                if (!route_repair.contains(route.donor))
                    metadata(route.donor, sizeof(decltype(route_repair)::value_type));
                auto& correction = route_repair[route.donor];
                auto absolute = difference.sign() < 0 ? live.scaled(difference, -1) : difference;
                detail::ExactLiveStorage::Scope absolute_scope(live, {&absolute});
                live.add(correction, absolute);
                auto& amount = amount_for(route.donor, route.recipient);
                live.add(amount.bulk, paired.bulk);
                live.add(amount.gas, paired.gas);
                live.add(amount.carrier, paired.carrier);
                if (!recipients.contains(route.recipient))
                    metadata(route.recipient, sizeof(decltype(recipients)::value_type));
                recipients[route.recipient] = {route.recipient_owner, route.recipient_gid};
            }
            for (const auto& key : data.topology->order)
            {
                const auto& donor = data.cells.at(key);
                Meshes::ExactTransportPartition::certify_donor_cover(
                    {donor.gid, donor.phase.geometry}, donor_atoms[key], data.options.sweep);
                const auto discrepancy = labels.at(key).discrepancy();
                auto absolute_discrepancy = discrepancy.sign() < 0 ? live.scaled(discrepancy, -1) : discrepancy;
                detail::ExactLiveStorage::Scope discrepancy_scope(live, {&absolute_discrepancy});
                if (route_repair[key].compare(absolute_discrepancy) ||
                    route_repair[key].compare(labels.at(key).bound()) > 0)
                    throw std::invalid_argument(
                        "General VoF cumulative positive phase correction exceeds donor proof.");
                ExactAmounts exported;
                detail::ExactLiveStorage::Scope exported_scope(
                    live, {&exported.bulk, &exported.gas, &exported.carrier});
                for (const auto& [recipient, amount] : amounts[key])
                    if (recipient != key)
                    {
                        live.add(exported.bulk, amount.bulk);
                        live.add(exported.gas, amount.gas);
                        live.add(exported.carrier, amount.carrier);
                    }
                auto retained_gas = live.difference(phase_measures(donor).gas, exported.gas);
                detail::ExactLiveStorage::Scope retained_gas_scope(live, {&retained_gas});
                auto retained_carrier = live.difference(phase_measures(donor).carrier, exported.carrier);
                detail::ExactLiveStorage::Scope retained_carrier_scope(live, {&retained_carrier});
                if (retained_gas.sign() < 0 || retained_carrier.sign() < 0)
                    throw std::invalid_argument(
                        "General VoF donor phase capacity exceeded after bounded canonical correction.");
                Meshes::ExactPhaseMeasures exported_exact{exported.bulk, exported.gas, exported.carrier};
                Meshes::ExactPhaseMeasures retained_exact{
                    live.difference(phase_measures(donor).bulk, exported.bulk), retained_gas, retained_carrier};
                detail::ExactLiveStorage::Scope retained_exact_scope(
                    live, {&retained_exact.bulk, &retained_exact.gas, &retained_exact.carrier});
                if (!retained_exact.bulk.sign() && (retained_gas.sign() || retained_carrier.sign()))
                    throw std::invalid_argument(
                        "General VoF full donor export cannot retain a numerical occupancy defect.");
                transport_charge(budget, 3 * sizeof(GeneralVoFDonorBudget), "donor phase proof records");
                trial.budgets_.push_back({donor.gid, phase_measures(donor), exported_exact, retained_exact,
                    diagnostic(phase_measures(donor)), diagnostic(exported_exact), diagnostic(retained_exact)});
                auto& proof = trial.budgets_.back();
                proof.signed_repair_absolute = donor_signed_repair[key].first;
                proof.signed_repair_bound = donor_signed_repair[key].second;
                proof.route_repair_absolute = route_repair[key];
                proof.route_repair_bound = labels.at(key).bound();
                proof.full_cell_fit_discrepancy = discrepancy;
                live.retain(phase_measures(donor));
                live.retain(exported_exact);
                live.retain(retained_exact);
                live.retain(proof.signed_repair_absolute);
                live.retain(proof.signed_repair_bound);
                live.retain(proof.route_repair_absolute);
                live.retain(proof.route_repair_bound);
                live.retain(proof.full_cell_fit_discrepancy);
                auto& self = amount_for(key, key);
                live.replace(self.gas, retained_gas);
                live.replace(self.carrier, retained_carrier);
                if (!recipients.contains(key))
                    metadata(key, sizeof(decltype(recipients)::value_type));
                recipients[key] = {comm->getRank(), donor.gid};
                ExactAmounts column;
                detail::ExactLiveStorage::Scope column_scope(live, {&column.bulk, &column.gas, &column.carrier});
                for (const auto& [recipient, amount] : amounts[key])
                {
                    if (amount.gas.sign() < 0 || amount.carrier.sign() < 0)
                        throw std::invalid_argument("General VoF negative positive-route phase.");
                    live.add(column.gas, amount.gas);
                    live.add(column.carrier, amount.carrier);
                    if (recipient != key && live.sum(amount.gas, amount.carrier).compare(amount.bulk))
                        throw std::invalid_argument("General VoF routed phase complement differs from bulk.");
                }
                if (column.gas.compare(phase_measures(donor).gas) ||
                    column.carrier.compare(phase_measures(donor).carrier))
                    throw std::invalid_argument("General VoF independent donor phase column does not close.");
                for (const auto& [recipient, amount] : amounts[key])
                {
                    const auto [recipient_owner, recipient_gid] = recipients.at(recipient);
                    const bool self = recipient_owner == comm->getRank();
                    const auto key_bound = [&](const Key& value)
                    {
                        wire::Budget bound{live.operation_bytes() / 8};
                        bound.add(128); // all fixed typed-key fields and framing
                        std::visit(
                            [&](const auto& ancestry)
                            {
                                const auto path = [&](const auto& values)
                                {
                                    if (values.capacity() > bound.limit / sizeof(std::uint64_t))
                                        throw std::invalid_argument("Donor packet key exceeds operation arena.");
                                    bound.add(values.capacity() * sizeof(std::uint64_t));
                                };
                                if constexpr (requires { ancestry.path; })
                                    path(ancestry.path);
                                else if constexpr (requires { ancestry.base.path; })
                                {
                                    path(ancestry.base.path);
                                    path(ancestry.axial.path);
                                }
                            },
                            value.ancestry);
                        return bound.used;
                    };
                    const auto donor_key_bound = key_bound(key), recipient_key_bound = key_bound(recipient);
                    wire::Budget codec{live.operation_bytes() / 8};
                    codec.add(donor_key_bound);
                    codec.add(recipient_key_bound);
                    codec.add(72);
                    const auto measure_bound = [&](const Measure& value)
                    {
                        const auto bytes = value.storage_bytes();
                        codec.add(bytes);
                        codec.add(56); // exact measure certificate framing
                        return bytes + 56;
                    };
                    std::size_t bulk_bound = 0, gas_bound = 0, carrier_bound = 0;
                    if (!self)
                    {
                        bulk_bound = measure_bound(amount.bulk);
                        gas_bound = measure_bound(amount.gas);
                        carrier_bound = measure_bound(amount.carrier);
                    }
                    // Eight times this complete envelope was reserved before
                    // allocation. It covers all three returned encodings plus
                    // the next encode's integer copies/byte-vector growth and
                    // key vectors; no scalar operation overlaps this codec.
                    auto dk = AMR::encode_adaptive_key(key), rk = AMR::encode_adaptive_key(recipient);
                    if (dk.capacity() > 2 * donor_key_bound || rk.capacity() > 2 * recipient_key_bound)
                        throw std::invalid_argument("Donor packet key capacity exceeds its admission.");
                    std::vector<char> bulk, gas, carrier;
                    if (!self)
                    {
                        bulk = amount.bulk.encode(bulk_bound);
                        gas = amount.gas.encode(gas_bound);
                        carrier = amount.carrier.encode(carrier_bound);
                        if (bulk.capacity() > 2 * bulk_bound || gas.capacity() > 2 * gas_bound ||
                            carrier.capacity() > 2 * carrier_bound)
                            throw std::invalid_argument("Donor phase encoded capacity exceeds its admission.");
                    }
                    const auto payload_bound = dk.size() + rk.size() + bulk.size() + gas.size() + carrier.size() + 72;
                    if (payload_bound > limit / 2)
                        throw std::invalid_argument("Donor phase packet capacity exceeds its budget.");
                    transport_charge(budget, 3 * sizeof(OwnerPacket), "donor phase packet metadata");
                    transport_charge(budget, 2 * payload_bound, "donor phase packet capacity");
                    wire::Writer w;
                    w.data.reserve(payload_bound);
                    if (w.data.capacity() > 2 * payload_bound)
                        throw std::invalid_argument("Donor phase writer capacity exceeds admission.");
                    w.bytes(dk);
                    w.word(donor.gid);
                    w.word(donor.revision);
                    w.bytes(rk);
                    w.word(recipient_gid);
                    w.word(self ? 1 : 0);
                    if (self)
                    {
                        // Keep immutable exact handles until this invocation's
                        // self packets have been authenticated and consumed.
                        // Cover retained capacity and simultaneous old/new
                        // vector buffers during geometric capacity growth.
                        transport_charge(
                            budget, 3 * sizeof(LocalAmount) + dk.size() + rk.size(), "donor phase local amount record");
                        // Immutable handles alias the tracked amount slots, which
                        // remain live through this builder; do not charge twice.
                        live.retain({amount.bulk, amount.gas, amount.carrier});
                        w.word(local_amounts.size());
                        local_amounts.push_back({key, recipient, donor.gid, donor.revision, recipient_gid, amount});
                    }
                    else
                    {
                        w.bytes(bulk);
                        w.bytes(gas);
                        w.bytes(carrier);
                    }
                    transfer_packets.push_back({recipient_owner, std::move(w.data)});
                }
            }
        });
    std::vector<Routed>().swap(routed);
    auto received = exchange_owner_packets(transfer_packets, comm, limit);
    std::vector<OwnerPacket>().swap(transfer_packets);
    std::shared_ptr<Data> candidate;
    std::vector<Key> donor_keys;
    struct Transfer
    {
        Key donor, recipient;
        std::uint64_t gid = 0, revision = 0;
        int owner = -1;
        ExactAmounts amount;
    };
    std::vector<Transfer> transfers;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::set<std::pair<Key, Key>> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto dk = AMR::decode_adaptive_key(r.bytes());
                auto gid = r.word(), revision = r.word();
                auto rk = AMR::decode_adaptive_key(r.bytes());
                auto recipient_gid = r.word();
                const auto& recipient = data.cells.at(rk);
                if (recipient.owner != comm->getRank() || recipient.gid != recipient_gid ||
                    !seen.emplace(dk, rk).second)
                    throw std::invalid_argument("General VoF positive transfer recipient differs.");
                ExactAmounts amount;
                const auto tag = r.word();
                if (tag == 1)
                {
                    const auto index = r.word();
                    r.done();
                    if (packet.peer != comm->getRank() || index >= local_amounts.size())
                        throw std::invalid_argument("General VoF foreign/out-of-range local amount reference.");
                    const auto& original = local_amounts[std::size_t(index)];
                    const auto& donor = data.cells.at(dk);
                    if (donor.owner != comm->getRank() || donor.gid != gid || donor.revision != revision ||
                        original.donor != dk || original.recipient != rk || original.donor_gid != gid ||
                        original.donor_revision != revision || original.recipient_gid != recipient_gid)
                        throw std::invalid_argument("General VoF local amount metadata mismatch.");
                    amount = original.amount;
                }
                else if (tag == 0)
                {
                    if (packet.peer == comm->getRank())
                        throw std::invalid_argument("General VoF self amount omitted its local reference.");
                    amount.bulk = Measure::decode(r.bytes(), limit);
                    amount.gas = Measure::decode(r.bytes(), limit);
                    amount.carrier = Measure::decode(r.bytes(), limit);
                    r.done();
                }
                else
                    throw std::invalid_argument("General VoF invalid amount route tag.");
                budget.add(sizeof(Transfer));
                budget.add(amount.bulk.storage_bytes());
                budget.add(amount.gas.storage_bytes());
                budget.add(amount.carrier.storage_bytes());
                transfers.push_back({dk, rk, gid, revision, packet.peer, std::move(amount)});
                donor_keys.push_back(dk);
            }
        });
    std::vector<OwnerPacket>().swap(received.packets);
    std::vector<LocalAmount>().swap(local_amounts);
    auto donor_owners = data.hierarchy->lookup_cells(donor_keys);
    wire::collective(*comm,
        [&]
        {
            candidate = std::make_shared<Data>(data);
            candidate->control = trial.transaction_;
            std::map<Key, ExactAmounts> totals;
            for (std::size_t i = 0; i < transfers.size(); ++i)
            {
                const auto& transfer = transfers[i];
                const auto& owner = donor_owners[i];
                if (owner.owner != transfer.owner || owner.revision != transfer.revision ||
                    owner.address != transfer.gid)
                    throw std::invalid_argument("General VoF positive transfer donor differs.");
                const auto& amount = transfer.amount;
                auto& total = totals[transfer.recipient];
                total.bulk = total.bulk + amount.bulk;
                total.gas = total.gas + amount.gas;
                total.carrier = total.carrier + amount.carrier;
                Meshes::ExactPhaseMeasures exact{amount.bulk, amount.gas, amount.carrier};
                trial.transfers_.push_back({transfer.gid, data.cells.at(transfer.recipient).gid, transfer.owner,
                    comm->getRank(), exact, diagnostic(exact)});
            }
            candidate->gas.clear();
            candidate->carrier.clear();
            candidate->phases.clear();
            candidate->roundoff.clear();
            candidate->occupancy.clear();
            for (const auto& key : data.topology->order)
            {
                auto& cell = candidate->cells.at(key);
                const auto& total = totals.at(key);
                if (total.bulk.compare(Measure::volume(*cell.phase.geometry)))
                    throw std::invalid_argument("General VoF positive transfers do not fill actual target.");
                const auto before = phase_measures(data.cells.at(key));
                if (((total.gas + total.carrier) - total.bulk).compare((before.gas + before.carrier) - before.bulk))
                    throw std::invalid_argument(
                        "General VoF transport changed the independent source occupancy defect.");
                cell.retained_interface.reset();
                if (data.options.inventory_policy == PhaseInventoryPolicy::ExperimentalExact)
                {
                    cell.experimental_exact = detail::experimental_advance(*data.cells.at(key).experimental_exact,
                        {total.bulk, total.gas, total.carrier}, *cell.phase.geometry, cell.native_volume, limit);
                    const auto view = detail::experimental_view(*cell.experimental_exact, cell.native_volume);
                    cell.phase.gas_volume = view.phases.gas;
                    cell.phase.carrier_volume = view.phases.carrier;
                    cell.occupancy = view.error;
                    cell.roundoff = view.roundoff;
                }
                else
                {
                    // Round each independently conserved phase ONCE. The next step
                    // starts from these published values, never total's rational tail.
                    const auto gas = detail::round_phase_value(total.gas);
                    const auto carrier = detail::round_phase_value(total.carrier);
                    cell.phase.gas_volume = gas.value;
                    cell.phase.carrier_volume = carrier.value;
                    cell.roundoff = {gas.absolute_error, carrier.absolute_error};
                    const auto rounding = finite_bound(gas.absolute_error) + finite_bound(carrier.absolute_error);
                    cell.occupancy.admitted_bound =
                        detail::phase_upper_bound(finite_bound(cell.occupancy.admitted_bound) + rounding);
                    cell.occupancy.phase_input_bound =
                        detail::phase_upper_bound(finite_bound(cell.occupancy.phase_input_bound) + rounding);
                    const auto published = phase_measures(cell);
                    const auto diagnostic =
                        detail::round_phase_residual(published.gas + published.carrier - published.bulk);
                    cell.occupancy.difference = diagnostic.value;
                    cell.occupancy.admitted_bound = detail::phase_upper_bound(
                        finite_bound(cell.occupancy.admitted_bound) + finite_bound(diagnostic.absolute_error));
                    cell.occupancy.phase_input_bound = detail::phase_upper_bound(
                        finite_bound(cell.occupancy.phase_input_bound) + finite_bound(diagnostic.absolute_error));
                }
                candidate->phases.push_back({cell.native_volume, cell.phase.gas_volume, cell.phase.carrier_volume});
                candidate->roundoff.push_back(cell.roundoff);
                candidate->occupancy.push_back(cell.occupancy);
                if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
                {
                    const auto metric = cell.phase.geometry->moments().volume;
                    const double minority = cell.declaration == GeneralVoFPhaseGeometry::PureGas
                                                ? cell.phase.carrier_volume
                                                : cell.phase.gas_volume;
                    if (minority >
                        metric.binary64_absolute_error() + 8 * std::numeric_limits<double>::epsilon() * metric.value)
                        cell.declaration = GeneralVoFPhaseGeometry::Infer;
                }
                validate_cell(cell, data.options);
                candidate->gas.push_back(cell.phase.gas_volume);
                candidate->carrier.push_back(cell.phase.carrier_volume);
            }
            refresh_experimental_catalog(*candidate);
            std::copy(candidate->roundoff.begin(), candidate->roundoff.end(), trial.roundoff_.begin());
            current(data);
        });
    // Independent global phase checks use only fixed-size finite reductions.
    // Exact donor columns above certify the pre-publication route algebra;
    // these checks account explicitly for the once-rounded accepted values.
    std::array<double, 12> local{}, global{};
    wire::collective(*comm,
        [&]
        {
            std::array<::SimpleFluid::detail::CompensatedSum<>, 6> sums;
            for (const auto& key : data.topology->order)
            {
                const auto& old = data.cells.at(key);
                const auto& next = candidate->cells.at(key);
                sums[0] += static_cast<long double>(next.phase.gas_volume) - old.phase.gas_volume;
                sums[1] += static_cast<long double>(next.phase.carrier_volume) - old.phase.carrier_volume;
                sums[2] += next.roundoff.gas;
                sums[3] += next.roundoff.carrier;
                if (data.options.inventory_policy == PhaseInventoryPolicy::ExperimentalExact)
                {
                    sums[2] += old.roundoff.gas;
                    sums[3] += old.roundoff.carrier;
                }
                sums[4] += std::abs(static_cast<long double>(next.phase.gas_volume)) +
                           std::abs(static_cast<long double>(old.phase.gas_volume));
                sums[5] += std::abs(static_cast<long double>(next.phase.carrier_volume)) +
                           std::abs(static_cast<long double>(old.phase.carrier_volume));
            }
            for (std::size_t i = 0; i < sums.size(); ++i)
            {
                const auto exact_local = sums[i].value();
                local[i] = static_cast<double>(exact_local);
                if (!std::isfinite(local[i]))
                    throw std::invalid_argument("Finite phase reduction wire range is inconclusive.");
                const auto error = std::abs(exact_local - static_cast<long double>(local[i]));
                auto bound = static_cast<double>(error);
                if (static_cast<long double>(bound) < error)
                    bound = std::nextafter(bound, std::numeric_limits<double>::infinity());
                if (!std::isfinite(bound) || static_cast<long double>(bound) < error)
                    throw std::invalid_argument("Finite phase reduction conversion bound is inconclusive.");
                local[i + 6] = bound;
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    wire::collective(*comm,
        [&]
        {
            const long double count =
                48 * static_cast<long double>(data.hierarchy->global_cells()) + 64 * comm->getSize() + 128;
            const auto product = count * std::numeric_limits<double>::epsilon();
            if (!(product < .25L))
                throw std::invalid_argument("Finite phase reduction operation bound is exhausted.");
            // Per phase: three compensated streams cost at most14N
            // arithmetic operations, six value/radius reductions6(P-1), plus
            // conversion/finalization.48N+64P+128 conservatively covers those
            // paths AND the uncertainty of their observed L1 scale. The guard
            // implies an integer count<2^51, exactly representable in double.
            const auto u =
                Measure::from_value(static_cast<double>(count)).scaled(std::numeric_limits<double>::epsilon());
            const auto gamma_exact = u.divided_by(Measure::from_value(1) - u);
            const auto gamma = detail::phase_upper_bound(gamma_exact.divided_by(Measure::from_value(1) - gamma_exact));
            const auto up_add = [](long double x, long double y)
            {
                const auto sum = x + y;
                return sum == 0 && x == 0 && y == 0 ? 0.L
                                                    : std::nextafter(sum, std::numeric_limits<long double>::infinity());
            };
            const auto up_product = [](long double x, long double y)
            {
                if (x == 0 || y == 0)
                    return 0.L;
                return std::nextafter(x * y, std::numeric_limits<long double>::infinity());
            };
            for (unsigned phase = 0; phase < 2; ++phase)
            {
                // Convert/reduce only supported binary64 scalars. Include the
                // local wire radii and reduction/local arithmetic on phase L1;
                // no geometric/native bulk sets a minority tolerance floor.
                const long double radius = up_add(global[phase + 6], global[phase + 8]);
                const long double publication = global[phase + 2];
                const long double scale =
                    up_add(up_add(global[phase + 4], global[phase + 10]), up_add(publication, radius));
                const auto bound = up_add(up_add(publication, radius), up_product(gamma, scale));
                if (!std::isfinite(global[phase]) || !std::isfinite(bound) ||
                    std::abs(static_cast<long double>(global[phase])) > bound)
                    throw std::invalid_argument("Finite General VoF phase conservation exceeds its publication bound.");
            }
        });
    import_phases(*candidate);
    wire::collective(*comm, [&] { retained_budget(*candidate); });
    trial.candidate_ = std::move(candidate);
    return trial;
}
std::shared_ptr<const GeneralVoF> GeneralVoF::apply(const Trial& trial) const
{
    wire::collective(*data_->hierarchy->communicator(),
        [&]
        {
            current(*data_);
            if (trial.source_ != data_ || !trial.candidate_ ||
                trial.candidate_->options.inventory_policy != data_->options.inventory_policy)
                throw std::invalid_argument("General VoF trial belongs to another source.");
        });
    return trial.candidate();
}
} // namespace SimpleFluid::VoF
