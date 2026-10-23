/** @file GeneralVoF.cc
 * @brief Finite-inventory unsplit phase transport with transient exact native geometry proofs.
 */
#include "FVM/GeneralVoF.hh"
#include "FVM/details/ExactLiveStorage.hh"
#include "FVM/details/BoundedPhaseArithmetic.hh"
#include "FVM/details/DistributedAMROverlap.hh"
#include "FVM/details/BoundedVoFTransport.hh"
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
#include <tuple>
namespace SimpleFluid::VoF
{
namespace Exact
{
struct GeneralVoF
{
    using State = VoF::GeneralVoF;
    using Data = State::Data;
    using Trial = State::Trial;
    using Key = State::Key;
    static Trial plan_transport(const State&, std::span<const GeneralVoFNodeVelocity>,
        std::span<const GeneralVoFFaceVolume>, double);
    static void validate_transfer_target(const State&, const ExactPhaseTransferResult&);
    static ExactPhaseTransferSource export_transfer_source(const State&, bool);
    static std::shared_ptr<const State> adopt_transfer(const State&, const ExactPhaseTransferResult&);
};
} // namespace Exact
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
    Meshes::BoundedConvexGeometryPtr finite_geometry;
    PhaseError occupancy;
    PhaseRoundoff roundoff, finite_history;
    std::shared_ptr<const ExperimentalPhaseInventory> experimental_exact;
    GeneralVoFPhaseGeometry declaration = GeneralVoFPhaseGeometry::Infer;
    std::optional<PolyhedralInterface> retained_interface;
    bool inherited = false;
};
FinitePolyhedralPhaseSample finite_sample(const Cell&);
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
    if (cell.finite_geometry)
    {
        using namespace finite_detail;
        const auto actual = cell.finite_geometry->volume();
        const auto conversion = subtract(Value{cell.native_volume, 0}, actual);
        Sum defect_sum;
        defect_sum.add({cell.phase.gas_volume, 0});
        defect_sum.add({cell.phase.carrier_volume, 0});
        defect_sum.add(negate(actual));
        const auto defect = defect_sum.value();
        const auto input = static_cast<long double>(cell.native_volume) * 8 * std::numeric_limits<double>::epsilon();
        if (!(cell.native_volume > 0) || !std::isfinite(cell.native_volume) ||
            std::abs(conversion.value) > input + conversion.absolute_error ||
            std::abs(defect.value - conversion.value) > input + defect.absolute_error + conversion.absolute_error)
            throw std::invalid_argument("General VoF finite phase/native-volume input exceeds its arithmetic bound.");
        cell.occupancy.difference = static_cast<double>(defect.value);
        cell.occupancy.native_conversion = static_cast<double>(conversion.value);
        cell.occupancy.phase_input_bound = upper_double(upward(input + defect.absolute_error));
        cell.occupancy.admitted_bound = upper_double(upward(std::abs(conversion.value) + input +
            conversion.absolute_error + defect.absolute_error));
        return;
    }
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
    if (!experimental)
    {
        using namespace finite_detail;
        if (!cell.finite_geometry)
            throw std::invalid_argument("Finite General VoF is missing bounded geometry authority.");
        const auto actual = cell.finite_geometry->volume();
        Sum defect_sum;
        defect_sum.add({cell.phase.gas_volume, 0});
        defect_sum.add({cell.phase.carrier_volume, 0});
        defect_sum.add(negate(actual));
        const auto defect = defect_sum.value();
        const auto& e = cell.occupancy;
        bool valid = std::isfinite(cell.phase.gas_volume) && cell.phase.gas_volume >= 0 &&
            std::isfinite(cell.phase.carrier_volume) && cell.phase.carrier_volume >= 0 &&
            std::isfinite(e.difference) && std::isfinite(e.native_conversion) &&
            std::abs(defect.value) <= e.admitted_bound + defect.absolute_error;
        for (double bound : {e.admitted_bound, e.phase_input_bound, e.fit_bound, e.transferred_bound,
                 e.fit_allocation_bound, cell.roundoff.gas, cell.roundoff.carrier, cell.finite_history.gas, cell.finite_history.carrier})
            valid = valid && std::isfinite(bound) && bound >= 0;
        if (!valid || std::uint64_t(cell.declaration) > 2)
            throw std::invalid_argument("General VoF finite occupancy provenance is invalid.");
        PolyhedralReconstruction::validate(finite_sample(cell), options.reconstruction);
        if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
        {
            const auto minority = cell.declaration == GeneralVoFPhaseGeometry::PureGas ?
                cell.phase.carrier_volume : cell.phase.gas_volume;
            const auto bound = cell.inherited ? std::max(e.admitted_bound, e.fit_bound) :
                actual.absolute_error + 8 * std::numeric_limits<double>::epsilon() * actual.value;
            if (minority > bound)
                throw std::invalid_argument("General VoF pure declaration contradicts finite inventory.");
        }
        return;
    }
    // Explicit exact-policy diagnostic views retain their exact provenance.
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
    Exact::PolyhedralReconstruction::validate(geometric_sample(cell), options.reconstruction);
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
void finite_real(wire::Writer& w, long double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("Finite VoF wire value is nonfinite.");
    int exponent = 0;
    const auto mantissa = std::frexp(value, &exponent);
    const auto high = static_cast<double>(mantissa);
    const auto low = static_cast<double>(mantissa - high);
    if (static_cast<long double>(high) + low != mantissa)
        throw std::invalid_argument("Finite VoF wire significand exceeds its fixed representation.");
    real(w, high);
    real(w, low);
    w.word(std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
}
long double finite_real(wire::Reader& r)
{
    const auto high = real(r), low = real(r);
    const auto exponent = std::bit_cast<std::int64_t>(r.word());
    if (!std::isfinite(high) || !std::isfinite(low) || exponent < std::numeric_limits<int>::min() ||
        exponent > std::numeric_limits<int>::max())
        throw std::invalid_argument("Finite VoF wire scalar is malformed.");
    const auto result = std::scalbn(static_cast<long double>(high) + low, int(exponent));
    if (!std::isfinite(result) || ((high != 0 || low != 0) && result == 0))
        throw std::invalid_argument("Finite VoF wire scalar exceeds qualified range.");
    return result;
}
void finite_value(wire::Writer& w, Meshes::BoundedGeometryValue v)
{
    finite_real(w, v.value);
    finite_real(w, v.absolute_error);
}
Meshes::BoundedGeometryValue finite_value(wire::Reader& r)
{
    Meshes::BoundedGeometryValue v{finite_real(r), finite_real(r)};
    if (v.absolute_error < 0)
        throw std::invalid_argument("Finite VoF wire bound is negative.");
    return v;
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
    w.word(p.finite_geometry);
    if (p.finite_geometry)
        finite_value(w, p.finite_target_volume_difference);
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
    p.finite_geometry = r.word();
    if (p.finite_geometry)
        p.finite_target_volume_difference = finite_value(r);
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
        Meshes::BoundedPoint finite_normal;
        Meshes::BoundedFaceConstraint finite_constraint;
    };
    struct Topology
    {
        AMR::AdaptiveNativePublication native;
        AMR::AdaptiveNativeDraft draft;
        std::vector<AMR::AdaptiveTile> tiles;
        std::vector<AMR::AdaptiveNode> nodes;
        std::map<std::uint64_t, Meshes::BoundedPoint> finite_nodes;
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
    std::vector<PhaseRoundoff> roundoff, phase_history;
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
               data.phases.capacity() * sizeof(PhaseVolumes) + (data.roundoff.capacity() + data.phase_history.capacity()) * sizeof(PhaseRoundoff) +
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
        if (cell.finite_geometry && geometry.insert(cell.finite_geometry.get()).second)
            budget.add(cell.finite_geometry->storage_bytes());
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
    for (const auto& [id, point] : data.topology->finite_nodes)
        budget.add(sizeof(id) + 4 * sizeof(void*) + Meshes::bounded_point_storage_bytes(point));
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
            real(w, options.reconstruction.minimum_phase_support_fraction);
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
                budget.packet(bytes.size() + 152);
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
                real(w, cell.finite_history.gas);
                real(w, cell.finite_history.carrier);
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
                cell.finite_history.gas = real(r);
                cell.finite_history.carrier = real(r);
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
FinitePolyhedralPhaseSample finite_sample(const Cell& cell)
{
    const auto bulk = cell.finite_geometry->volume();
    const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
    const auto minority = gas ? cell.phase.gas_volume : cell.phase.carrier_volume;
    const long double dominant = bulk.value - minority;
    if (dominant < 0)
        throw std::invalid_argument("Finite VoF minority exceeds actual cell geometry.");
    return {cell.finite_geometry, gas ? minority : static_cast<double>(dominant),
        gas ? static_cast<double>(dominant) : minority};
}
std::vector<FinitePolyhedralPhaseSample> finite_stencil(const GeneralVoF::Data& data, const Key& key)
{
    std::set<Key> visited{key}, front{key};
    std::vector<FinitePolyhedralPhaseSample> result;
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
                    if (result.size() == data.options.maximum_stencil_cells)
                        throw std::invalid_argument("Finite VoF reconstruction stencil budget exhausted.");
                    result.push_back(finite_sample(data.cells.at(neighbor)));
                    next.insert(neighbor);
                }
        }
        front = std::move(next);
    }
    return result;
}
PolyhedralInterface reconstruct_finite_cell(const GeneralVoF::Data& data, const Key& key)
{
    const auto& cell = data.cells.at(key);
    validate_cell(cell, data.options);
    if (cell.retained_interface)
    {
        if (!cell.retained_interface->finite_geometry)
            throw std::invalid_argument("Finite VoF cannot adopt an exact-policy interface capsule.");
        return *cell.retained_interface;
    }
    if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
    {
        PolyhedralInterface result;
        result.finite_geometry = true;
        result.pure_gas = cell.declaration == GeneralVoFPhaseGeometry::PureGas;
        result.pure_carrier = !result.pure_gas;
        const auto minority = result.pure_gas ? cell.phase.carrier_volume : cell.phase.gas_volume;
        result.target_volume_residual = minority;
        result.finite_target_volume_difference = {-static_cast<long double>(minority), 0};
        return result;
    }
    if (cell.phase.gas_volume == 0 || cell.phase.carrier_volume == 0)
        return PolyhedralReconstruction::solve_offset(
            finite_sample(cell), {1, 0, 0}, data.options.reconstruction);
    return PolyhedralReconstruction::reconstruct(
        finite_sample(cell), finite_stencil(data, key), data.options.reconstruction);
}
std::map<Key, PolyhedralInterface> reconstruct_finite_local(const GeneralVoF::Data& data)
{
    std::map<Key, PolyhedralInterface> result;
    for (const auto& key : data.topology->order)
        result.emplace(key, reconstruct_finite_cell(data, key));
    return result;
}
std::map<Key, PolyhedralInterface> reconstruct_exact_local(const GeneralVoF::Data& data)
{
    std::map<Key, PolyhedralInterface> result;
    for (const auto& key : data.topology->order)
    {
        const auto& target = data.cells.at(key);
        validate_cell(target, data.options);
        if (target.retained_interface && !target.retained_interface->finite_geometry)
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
            result.emplace(key, Exact::PolyhedralReconstruction::solve_offset(
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
            key, Exact::PolyhedralReconstruction::reconstruct(geometric_sample(target), stencil, data.options.reconstruction));
    }
    return result;
}
std::map<Key, PolyhedralInterface> reconstruct_local(const GeneralVoF::Data& data)
{
    return data.options.inventory_policy == PhaseInventoryPolicy::Finite ?
        reconstruct_finite_local(data) : reconstruct_exact_local(data);
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
                if (data.options.inventory_policy == PhaseInventoryPolicy::Finite)
                {
                    if (!fitted.finite_geometry)
                        throw std::invalid_argument("Finite VoF imported an exact interface authority.");
                    if (!fitted.pure_gas && !fitted.pure_carrier)
                    {
                        const auto amounts = cell.finite_geometry->phases(fitted.gas);
                        const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                        fitted.finite_target_volume_difference = finite_detail::subtract(
                            gas ? amounts.gas : amounts.carrier,
                            {gas ? cell.phase.gas_volume : cell.phase.carrier_volume, 0});
                        fitted.target_volume_residual = static_cast<double>(fitted.finite_target_volume_difference.value);
                    }
                }
                else if (!fitted.pure_gas && !fitted.pure_carrier)
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
                if (options.inventory_policy == PhaseInventoryPolicy::Finite)
                    cell.finite_geometry = Meshes::BoundedConvexGeometry::from_geometry(*cell.phase.geometry);
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
                    cell.finite_history = capsule.accumulated_roundoff;
                    const auto& e = capsule.error;
                    cell.occupancy = {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound,
                        e.fit_residual, e.fit_bound, e.transferred_bound, e.fit_allocation_bound};
                    if (capsule.geometry_ready && options.inventory_policy == PhaseInventoryPolicy::ExperimentalExact)
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
                        if (options.inventory_policy == PhaseInventoryPolicy::Finite)
                            cell.finite_geometry = Meshes::BoundedConvexGeometry::from_geometry(*retained_geometry);
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
            std::map<std::uint64_t, std::set<Key>> node_source_cells;
            std::map<AMR::AdaptiveBytes, const AMR::AdaptiveTile*> wall_tiles;
            if (options.inventory_policy == PhaseInventoryPolicy::Finite)
                for (const auto& tile : topology->draft.tiles)
                {
                    for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                        if (key && data->cells.contains(*key))
                            for (auto node : tile.node_gids)
                                if (node_source_cells[node].insert(*key).second)
                                    budget.add(sizeof(Key) + AMR::encode_adaptive_key(*key).size() + 8 * sizeof(void*));
                    if (tile.exterior)
                        wall_tiles.try_emplace(tile.identity, &tile);
                }
            if (options.inventory_policy == PhaseInventoryPolicy::Finite)
                for (const auto& node : topology->draft.nodes)
                {
                    const auto fan = topology->node_fans.find(node.gid);
                    std::vector<Meshes::ConvexGeometryPtr> incident_sources;
                    if (fan != topology->node_fans.end() && !fan->second.empty())
                        for (const auto& key : fan->second)
                            incident_sources.push_back(data->cells.at(key).phase.geometry);
                    else if (const auto sources = node_source_cells.find(node.gid); sources != node_source_cells.end())
                        for (const auto& key : sources->second)
                            incident_sources.push_back(data->cells.at(key).phase.geometry);
                    if (incident_sources.empty())
                        throw std::invalid_argument("Finite VoF canonical node lacks an authenticated source fan.");
                    auto p = Meshes::bounded_source_point(incident_sources, node.exact_identity);
                    topology->finite_nodes.emplace(node.gid, std::move(p));
                }
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
                {
                    Meshes::BoundedPoint normal;
                    Meshes::BoundedFaceConstraint constraint;
                    if (options.inventory_policy == PhaseInventoryPolicy::Finite)
                    {
                        const auto found_face = wall_tiles.find(identity);
                        if (found_face == wall_tiles.end() || found_face->second->node_gids.size() < 3)
                            throw std::invalid_argument("Finite VoF wall lacks its authenticated native loop.");
                        const auto* face = found_face->second;
                        const auto& source_cell = data->cells.at(face->first);
                        constraint.geometry = source_cell.finite_geometry;
                        bool matched = false;
                        for (std::size_t index = 0; index < source_cell.phase.geometry->num_faces(); ++index)
                            if (source_cell.phase.geometry->face_geometry(index)->contains(*face->geometry))
                            {
                                constraint.face = index;
                                matched = true;
                                break;
                            }
                        if (!matched)
                            throw std::invalid_argument("Finite VoF source wall has no authenticated cell support.");
                        const auto& a = topology->finite_nodes.at(face->node_gids[0]);
                        for (std::size_t i = 1; i + 1 < face->node_gids.size(); ++i)
                            normal = finite_detail::add(normal, finite_detail::cross(
                                finite_detail::subtract(topology->finite_nodes.at(face->node_gids[i]), a),
                                finite_detail::subtract(topology->finite_nodes.at(face->node_gids[i + 1]), a)));
                    }
                    retained.push_back({identity, geometry, normal, constraint});
                }
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
            data->phase_history.clear();
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
                data->phase_history.push_back(data->cells.at(key).finite_history);
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
    Exact::GeneralVoF::validate_transfer_target(*this, result);
}
ExactPhaseTransferSource GeneralVoF::export_transfer_source(bool prepare_geometry) const
{
    return Exact::GeneralVoF::export_transfer_source(*this, prepare_geometry);
}
std::shared_ptr<const GeneralVoF> GeneralVoF::adopt_transfer(const ExactPhaseTransferResult& result) const
{
    return Exact::GeneralVoF::adopt_transfer(*this, result);
}
#include "FVM/details/GeneralVoFExactTransfer.inc"
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
const std::vector<PhaseRoundoff>& GeneralVoF::accumulated_phase_errors() const noexcept
{
    return data_->phase_history;
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
            if (inventory_policy() == PhaseInventoryPolicy::Finite)
            {
                for (const auto& key : data_->topology->order)
                {
                    const auto& cell = data_->cells.at(key);
                    validate_cell(cell, data_->options);
                    const auto bulk = cell.finite_geometry->volume();
                    const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                    const auto minority = gas ? cell.phase.gas_volume : cell.phase.carrier_volume;
                    const auto fraction = static_cast<long double>(minority) / bulk.value;
                    const auto value = gas ? fraction : 1 - fraction;
                    if (!(bulk.value > 0) || !std::isfinite(value) || value < 0 || value > 1)
                        throw std::invalid_argument("Finite VoF geometric fraction has no bounded view.");
                    result.push_back(static_cast<double>(value));
                }
                return;
            }
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
std::vector<GeneralVoFResolutionRequest> GeneralVoF::resolution_requests() const
{
    std::vector<GeneralVoFResolutionRequest> result;
    wire::collective(*data_->hierarchy->communicator(), [&]
    {
        current(*data_);
        if (inventory_policy() != PhaseInventoryPolicy::Finite)
            return;
        for (const auto& key : data_->topology->order)
        {
            const auto& cell = data_->cells.at(key);
            if (cell.declaration != GeneralVoFPhaseGeometry::Infer ||
                cell.phase.gas_volume == 0 || cell.phase.carrier_volume == 0)
                continue;
            auto assessment = PolyhedralReconstruction::assess_resolution(
                finite_sample(cell), finite_stencil(*data_, key), data_->options.reconstruction);
            if (!assessment.admitted())
                result.push_back({cell.gid, key, assessment});
        }
    });
    return result;
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
                result.push_back({cell.gid, cell.owner, cell.phase.geometry, planes.at(key), cell.finite_geometry});
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
std::vector<Meshes::BoundedPoint> GeneralVoF::finite_face_points(std::uint64_t face) const
{
    current(*data_);
    if (inventory_policy() != PhaseInventoryPolicy::Finite)
        throw std::invalid_argument("Finite canonical face points require Finite inventory policy.");
    const auto tile = std::find_if(data_->topology->tiles.begin(), data_->topology->tiles.end(),
        [&](const auto& tile) { return tile.face_gid == face; });
    if (tile == data_->topology->tiles.end())
        throw std::invalid_argument("Finite canonical face identity is not retained.");
    std::vector<Meshes::BoundedPoint> points;
    for (auto node : tile->node_gids)
        points.push_back(data_->topology->finite_nodes.at(node));
    return points;
}
std::vector<Meshes::BoundedConvexGeometryPtr> GeneralVoF::owned_finite_geometry() const
{
    if (inventory_policy() != PhaseInventoryPolicy::Finite)
        throw std::invalid_argument("Finite geometry access requires Finite inventory policy.");
    std::vector<Meshes::BoundedConvexGeometryPtr> result;
    for (const auto& key : data_->topology->order)
        result.push_back(data_->cells.at(key).finite_geometry);
    return result;
}
const std::vector<GeneralVoFFiniteFaceContribution>& GeneralVoF::Trial::finite_face_contributions() const noexcept
{
    return finite_faces_;
}
const std::vector<GeneralVoFFiniteTransfer>& GeneralVoF::Trial::finite_transfers() const noexcept
{
    return finite_transfers_;
}
const std::vector<GeneralVoFFiniteDonorBudget>& GeneralVoF::Trial::finite_donor_budgets() const noexcept
{
    return finite_budgets_;
}
const std::vector<GeneralVoFFiniteNodeTrajectory>& GeneralVoF::Trial::finite_node_trajectories() const noexcept
{
    return finite_trajectories_;
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
        result.push_back({cell.gid, cell.revision, cell.owner, cell.phase.geometry, cell.centroid, key, cell.finite_geometry});
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
namespace
{
namespace fp = finite_detail;
using FValue = Meshes::BoundedGeometryValue;
using FPoint = Meshes::BoundedPoint;
using FPhases = Meshes::BoundedPhaseMeasures;
void finite_point(wire::Writer& w, const FPoint& p)
{
    for (unsigned a = 0; a < 3; ++a)
        finite_value(w, {p.value[a], p.absolute_error[a]});
}
FPoint finite_point(wire::Reader& r)
{
    FPoint p;
    for (unsigned a = 0; a < 3; ++a)
    {
        const auto v = finite_value(r);
        p.value[a] = v.value;
        p.absolute_error[a] = v.absolute_error;
    }
    return p;
}
void finite_phases(wire::Writer& w, const FPhases& p)
{
    finite_value(w, p.bulk);
    finite_value(w, p.gas);
    finite_value(w, p.carrier);
}
FPhases finite_phases(wire::Reader& r)
{
    return {finite_value(r), finite_value(r), finite_value(r)};
}
struct FPhaseSum
{
    fp::Sum bulk, gas, carrier;
    void add(const FPhases& p, int sign = 1)
    {
        bulk.add(sign > 0 ? p.bulk : fp::negate(p.bulk));
        gas.add(sign > 0 ? p.gas : fp::negate(p.gas));
        carrier.add(sign > 0 ? p.carrier : fp::negate(p.carrier));
    }
    FPhases value() const { return {bulk.value(), gas.value(), carrier.value()}; }
};
FPhases finite_inventory(const Cell& cell)
{
    return {cell.finite_geometry->volume(), {cell.phase.gas_volume, 0}, {cell.phase.carrier_volume, 0}};
}
FPoint constrained_finite_departure(const GeneralVoF::Data& data, std::uint64_t node,
    const Meshes::GeometryPoint& velocity, double dt)
{
    std::vector<Meshes::BoundedFaceConstraint> walls;
    for (const auto& wall : data.topology->node_walls.at(node))
        walls.push_back(wall.finite_constraint);
    if (walls.size() > Meshes::ExactTrajectoryLimits{}.maximum_constraints)
        throw std::invalid_argument("Finite VoF complete wall catalog exceeds the unchanged constraint limit.");
    Meshes::BoundedGeometryLimits limits;
    limits.byte_budget = data.options.byte_budget;
    limits.maximum_operations = Meshes::ExactTrajectoryLimits{}.maximum_operations;
    return Meshes::bounded_backtrace_constrained(data.topology->finite_nodes.at(node), velocity, dt, walls, limits);
}
struct FLabel
{
    FValue gas_factor, carrier_factor, gas_history_factor, carrier_history_factor;
    bool gas_bulk = false, carrier_bulk = false;
    FPhases apply(const Meshes::BoundedConvexGeometry& geometry, const PolyhedralInterface& plane) const
    {
        const auto bulk = geometry.volume();
        const auto raw = (plane.pure_gas || plane.pure_carrier) ? FPhases{bulk, {}, {}} : geometry.phases(plane.gas);
        return {bulk, fp::multiply(gas_bulk ? bulk : raw.gas, gas_factor),
            fp::multiply(carrier_bulk ? bulk : raw.carrier, carrier_factor)};
    }
    FPhases history(const Meshes::BoundedConvexGeometry& geometry, const PolyhedralInterface& plane) const
    {
        const auto bulk = geometry.volume();
        const auto raw = (plane.pure_gas || plane.pure_carrier) ? FPhases{bulk, {}, {}} : geometry.phases(plane.gas);
        return {{}, fp::multiply(gas_bulk ? bulk : raw.gas, gas_history_factor),
            fp::multiply(carrier_bulk ? bulk : raw.carrier, carrier_history_factor)};
    }
};
FLabel finite_label(const Cell& cell, const PolyhedralInterface& plane, double tolerance)
{
    const auto inventory = finite_inventory(cell);
    FLabel result;
    if (plane.pure_gas || plane.pure_carrier)
    {
        result.gas_bulk = result.carrier_bulk = true;
        result.gas_factor = fp::divide(inventory.gas, inventory.bulk);
        result.carrier_factor = fp::divide(inventory.carrier, inventory.bulk);
        result.gas_history_factor = fp::divide({cell.finite_history.gas, 0}, inventory.bulk);
        result.carrier_history_factor = fp::divide({cell.finite_history.carrier, 0}, inventory.bulk);
        return result;
    }
    const auto raw = cell.finite_geometry->phases(plane.gas);
    const auto check = [&](FValue geometric, FValue physical)
    {
        if (physical.value == 0)
            return FValue{};
        const auto defect = fp::subtract(geometric, physical);
        const auto bound = std::abs(physical.value) * tolerance;
        if (std::abs(defect.value) + defect.absolute_error > bound + physical.absolute_error)
            throw std::invalid_argument("Finite VoF phase fit exceeds its independent phase budget.");
        return fp::divide(physical, geometric);
    };
    // The minority fit determines the plane; the dominant label additionally
    // includes the already admitted physical/native occupancy discrepancy.
    result.gas_factor = fp::divide(inventory.gas, raw.gas);
    result.carrier_factor = fp::divide(inventory.carrier, raw.carrier);
    result.gas_history_factor = fp::divide({cell.finite_history.gas, 0}, raw.gas);
    result.carrier_history_factor = fp::divide({cell.finite_history.carrier, 0}, raw.carrier);
    if (inventory.gas.value <= inventory.carrier.value)
        result.gas_factor = check(raw.gas, inventory.gas);
    else
        result.carrier_factor = check(raw.carrier, inventory.carrier);
    return result;
}
} // namespace
GeneralVoF::Trial GeneralVoF::plan_transport(std::span<const GeneralVoFNodeVelocity> node_velocity,
    std::span<const GeneralVoFFaceVolume> face_volume, double timestep) const
{
    if (inventory_policy() != PhaseInventoryPolicy::Finite)
        return Exact::GeneralVoF::plan_transport(*this, node_velocity, face_volume, timestep);
    const auto& data = *data_;
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    Meshes::BoundedGeometryPacketContext packet_context{{data.control,
        data.hierarchy->topology_generation(), data.hierarchy->partition_generation(), 0}};
    std::map<std::uint64_t, std::pair<FPoint, FPoint>> traces;
    std::map<std::uint64_t, FValue> flux;
    std::vector<OwnerPacket> packets;
    wire::collective(*comm, [&]
    {
        current(data);
        if (!std::isfinite(timestep) || timestep < 0 || node_velocity.size() != data.topology->nodes.size() ||
            face_volume.size() != data.topology->tiles.size())
            throw std::invalid_argument("Finite VoF controls do not cover retained nodes/faces.");
        wire::Budget budget{limit};
        std::map<std::uint64_t, Meshes::GeometryPoint> velocity;
        for (const auto& node : node_velocity)
        {
            if (!std::all_of(node.velocity.begin(), node.velocity.end(), [](auto x) { return std::isfinite(x); }) ||
                !velocity.emplace(node.node, node.velocity).second)
                throw std::invalid_argument("Finite VoF node control is nonfinite or duplicated.");
        }
        for (const auto& node : data.topology->nodes)
        {
            if (!velocity.contains(node.gid))
                throw std::invalid_argument("Finite VoF node controls contain a foreign identity.");
            const auto& source = data.topology->finite_nodes.at(node.gid);
            const auto departure = constrained_finite_departure(data, node.gid, velocity.at(node.gid), timestep);
            budget.add(sizeof(decltype(traces)::value_type) + 4 * sizeof(void*) +
                Meshes::bounded_point_storage_bytes(source) + Meshes::bounded_point_storage_bytes(departure));
            traces.emplace(node.gid, std::pair{source, departure});
            wire::Writer w;
            w.word(0);
            w.word(node.gid);
            w.bytes(node.identity);
            finite_point(w, source);
            finite_point(w, departure);
            for (auto x : velocity.at(node.gid))
                real(w, x);
            real(w, timestep);
            w.word(data.topology->node_walls.at(node.gid).size());
            for (const auto& wall : data.topology->node_walls.at(node.gid))
            {
                w.bytes(wall.identity);
                finite_point(w, wall.finite_normal);
            }
            budget.packet(w.data.size());
            packets.push_back({int(node.gid % comm->getSize()), std::move(w.data)});
        }
        for (const auto& face : face_volume)
        {
            if (face.exact_integrated_volume || !std::isfinite(face.integrated_volume))
                throw std::invalid_argument("Finite VoF cannot adopt an exact-policy face control.");
            const auto value = face.bounded_integrated_volume.value_or(FValue{face.integrated_volume, 0});
            if (!std::isfinite(value.value) || !std::isfinite(value.absolute_error) || value.absolute_error < 0 ||
                !flux.emplace(face.face, value).second)
                throw std::invalid_argument("Finite VoF face control is malformed or duplicated.");
        }
        for (const auto& tile : data.topology->tiles)
        {
            if (!flux.contains(tile.face_gid))
                throw std::invalid_argument("Finite VoF face controls contain a foreign identity.");
            const auto phi = flux.at(tile.face_gid);
            if ((tile.exterior || timestep == 0) && phi.value != 0)
                throw std::invalid_argument("Finite VoF impermeable boundary has nonzero flux.");
            wire::Writer w;
            w.word(1);
            w.word(tile.face_gid);
            w.bytes(tile.domain);
            w.bytes(tile.surface);
            w.bytes(tile.identity);
            finite_value(w, phi);
            budget.packet(w.data.size());
            packets.push_back({int(tile.face_gid % comm->getSize()), std::move(w.data)});
        }
        wire::Writer w;
        w.word(2);
        w.word(0);
        w.word(data.control);
        real(w, timestep);
        packets.push_back({0, std::move(w.data)});
    });
    const auto controls = exchange_owner_packets(packets, comm, limit);
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<char>> canonical;
        for (const auto& packet : controls.packets)
        {
            wire::Reader r{packet.bytes};
            const auto type = r.word(), id = r.word();
            if (type > 2 || int(id % comm->getSize()) != comm->getRank())
                throw std::invalid_argument("Finite VoF control reached a foreign authority.");
            budget.add(packet.bytes.size() + sizeof(decltype(canonical)::value_type));
            auto [it, inserted] = canonical.emplace(std::pair{type, id}, packet.bytes);
            if (!inserted && it->second != packet.bytes)
                throw std::invalid_argument("Finite VoF shared node/face controls disagree.");
        }
        for (const auto& key : data.topology->order)
        {
            fp::Sum sum;
            for (auto index : data.topology->cell_tiles.at(key))
            {
                const auto& tile = data.topology->tiles[index];
                const int sign = tile.first == key ? 1 : -1;
                sum.add(sign > 0 ? flux.at(tile.face_gid) : fp::negate(flux.at(tile.face_gid)));
            }
            const auto residual = sum.value();
            if (std::abs(residual.value) > residual.absolute_error || residual.absolute_error >
                data.cells.at(key).finite_geometry->volume().value * data.options.reconstruction.volume_tolerance)
                throw std::invalid_argument("Finite VoF canonical controls violate source-free occupancy.");
        }
    });
    Trial trial;
    trial.source_ = data_;
    trial.transaction_ = transaction(comm);
    packet_context.publication[3] = trial.transaction_;
    int local_motion = 0, motion = 0;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        budget.add(traces.size() * sizeof(GeneralVoFFiniteNodeTrajectory));
        trial.finite_trajectories_.reserve(traces.size());
        for (const auto& [id, trace] : traces)
        {
            local_motion |= !fp::same_point(trace.first, trace.second);
            trial.finite_trajectories_.push_back({id, trace.first, trace.second});
        }
        for (const auto& [face, value] : flux) local_motion |= value.value != 0;
    });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_motion, &motion);
    trial.roundoff_.resize(data.topology->order.size());
    if (!motion)
    {
        trial.candidate_ = data_;
        for (const auto& key : data.topology->order)
        {
            const auto& cell = data.cells.at(key);
            const auto old = finite_inventory(cell);
            trial.finite_transfers_.push_back({cell.gid, cell.gid, cell.owner, cell.owner, old, cell.finite_history});
            trial.finite_budgets_.push_back({cell.gid, old, {}, old});
        }
        return trial;
    }
    const auto planes = import_planes(data);
    std::map<Key, FLabel> labels;
    std::map<std::uint64_t, fp::Sweep> sweeps;
    std::vector<fp::Atom> local_atoms;
    std::map<std::tuple<std::uint64_t, std::uint64_t, int>, FPhaseSum> face_amounts;
    wire::collective(*comm, [&]
    {
        fp::Work work{data.options.sweep};
        work.limits.byte_budget = std::min(limit, work.limits.byte_budget);
        for (const auto& [key, cell] : data.cells)
            labels.emplace(key, finite_label(cell, planes.at(key), data.options.reconstruction.volume_tolerance));
        std::map<std::uint64_t, Meshes::BoundedConvexGeometryPtr> donors;
        for (const auto& [key, cell] : data.cells)
            donors.emplace(cell.gid, cell.finite_geometry);
        for (const auto& tile : data.topology->tiles)
        {
            work.pair_tests = 0; // Existing face-partition work-cap scope.
            auto sweep = fp::make_sweep(tile.face_gid, tile.node_gids, traces, flux.at(tile.face_gid), work);
            // Each signed simplex is covered by actual nonoverlapping native
            // donors before it can contribute a raw canonical face amount.
            for (const auto& piece : sweep.pieces)
            {
                std::vector<Meshes::BoundedConvexGeometryPtr> cover;
                for (const auto& [key, cell] : data.cells)
                {
                    work.test();
                    if (auto intersection = piece.geometry->intersect(*cell.finite_geometry))
                    {
                        work.atom(intersection);
                        cover.push_back(intersection);
                        if (tile.first_owner == comm->getRank())
                            face_amounts[{tile.face_gid, cell.gid, piece.sign}].add(
                                labels.at(key).apply(*intersection, planes.at(key)));
                    }
                }
                piece.geometry->certify_partition(cover, data.options.reconstruction.volume_tolerance);
            }
            sweeps.emplace(tile.face_gid, std::move(sweep));
        }
        for (const auto& key : data.topology->order)
        {
            const auto& cell = data.cells.at(key);
            std::vector<std::pair<const fp::Sweep*, int>> incident;
            for (auto index : data.topology->cell_tiles.at(key))
            {
                const auto& tile = data.topology->tiles[index];
                incident.push_back({&sweeps.at(tile.face_gid), tile.first == key ? 1 : -1});
            }
            work.pair_tests = 0; // Existing recipient-local partition work-cap scope.
            auto atoms = fp::recipient_partition(cell.gid, donors, incident, work,
                data.options.reconstruction.volume_tolerance);
            local_atoms.insert(local_atoms.end(), atoms.begin(), atoms.end());
        }
        for (const auto& [key, sum] : face_amounts)
        {
            auto value = sum.value();
            const int direction = std::get<2>(key);
            for (const auto& amount : {value.bulk, value.gas, value.carrier})
                if (amount.value < 0 || (amount.value != 0 && amount.value <= amount.absolute_error))
                    throw std::invalid_argument("Finite VoF raw donor stream positivity is unresolved.");
            const auto& donor = data.cells.at(data.topology->by_gid.at(std::get<1>(key)));
            trial.finite_faces_.push_back({std::get<0>(key), std::get<1>(key), donor.owner, direction, value});
        }
    });
    // Authenticate positive geometric fragments on the original donor owner.
    packets.clear();
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (std::size_t i = 0; i < local_atoms.size(); ++i)
        {
            const auto& atom = local_atoms[i];
            const auto& donor = data.cells.at(data.topology->by_gid.at(atom.donor));
            const auto& recipient = data.cells.at(data.topology->by_gid.at(atom.recipient));
            wire::Writer w;
            w.bytes(AMR::encode_adaptive_key(donor.key));
            w.word(donor.revision);
            w.word(donor.gid);
            w.bytes(AMR::encode_adaptive_key(recipient.key));
            w.word(recipient.gid);
            w.word(donor.owner == comm->getRank());
            if (donor.owner == comm->getRank())
                w.word(i);
            else
                w.bytes(atom.geometry->encode(packet_context, limit));
            budget.packet(w.data.size());
            packets.push_back({donor.owner, std::move(w.data)});
        }
    });
    const auto routed_packets = exchange_owner_packets(packets, comm, limit);
    struct Routed
    {
        Key donor, recipient;
        std::uint64_t recipient_gid;
        int recipient_owner;
        Meshes::BoundedConvexGeometryPtr geometry;
    };
    std::vector<Routed> routed;
    std::vector<Key> recipient_keys;
    std::vector<Meshes::BoundedConvexGeometryPtr> source_registry;
    for (const auto& [key, cell] : data.cells)
        source_registry.push_back(cell.finite_geometry);
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& packet : routed_packets.packets)
        {
            wire::Reader r{packet.bytes};
            auto donor_key = AMR::decode_adaptive_key(r.bytes());
            const auto revision = r.word(), gid = r.word();
            auto recipient_key = AMR::decode_adaptive_key(r.bytes());
            const auto recipient_gid = r.word(), local = r.word();
            const auto& donor = data.cells.at(donor_key);
            if (donor.owner != comm->getRank() || donor.gid != gid || donor.revision != revision)
                throw std::invalid_argument("Finite VoF atom donor authentication failed.");
            Meshes::BoundedConvexGeometryPtr geometry;
            if (local == 1)
            {
                const auto index = r.word();
                if (packet.peer != comm->getRank() || index >= local_atoms.size())
                    throw std::invalid_argument("Finite VoF foreign local atom reference.");
                const auto& original = local_atoms[index];
                if (original.donor != gid || original.recipient != recipient_gid ||
                    data.cells.at(data.topology->by_gid.at(recipient_gid)).key != recipient_key)
                    throw std::invalid_argument("Finite VoF local atom metadata differs.");
                geometry = original.geometry;
            }
            else if (local == 0 && packet.peer != comm->getRank())
            {
                Meshes::BoundedGeometryLimits decode_limits;
                decode_limits.byte_budget = limit;
                geometry = Meshes::BoundedConvexGeometry::decode(r.bytes(), packet_context, source_registry, decode_limits);
            }
            else
                throw std::invalid_argument("Finite VoF atom routing tag is malformed.");
            r.done();
            budget.add(sizeof(Routed) + geometry->storage_bytes());
            recipient_keys.push_back(recipient_key);
            routed.push_back({donor_key, recipient_key, recipient_gid, packet.peer, std::move(geometry)});
        }
    });
    const auto recipients = data.hierarchy->lookup_cells(recipient_keys);
    struct Amount
    {
        Key donor, recipient;
        std::uint64_t donor_gid = 0, donor_revision = 0, recipient_gid = 0;
        int recipient_owner = -1;
        FPhases value, history;
    };
    std::vector<Amount> amounts;
    wire::collective(*comm, [&]
    {
        fp::Work work{data.options.sweep};
        work.limits.byte_budget = std::min(limit, work.limits.byte_budget);
        std::map<Key, std::vector<Meshes::BoundedConvexGeometryPtr>> covers;
        std::map<std::pair<Key, Key>, FPhaseSum> groups, history_groups;
        std::map<Key, std::pair<int, std::uint64_t>> target_metadata;
        for (std::size_t i = 0; i < routed.size(); ++i)
        {
            const auto& atom = routed[i];
            if (recipients[i].owner != atom.recipient_owner || recipients[i].address != atom.recipient_gid)
                throw std::invalid_argument("Finite VoF atom recipient owner authentication failed.");
            target_metadata.emplace(atom.recipient, std::pair{atom.recipient_owner, atom.recipient_gid});
            covers[atom.donor].push_back(atom.geometry);
            groups[{atom.donor, atom.recipient}].add(labels.at(atom.donor).apply(*atom.geometry, planes.at(atom.donor)));
            history_groups[{atom.donor, atom.recipient}].add(labels.at(atom.donor).history(*atom.geometry, planes.at(atom.donor)));
            work.atom(atom.geometry);
        }
        for (const auto& key : data.topology->order)
        {
            const auto& donor = data.cells.at(key);
            donor.finite_geometry->certify_partition(covers[key], data.options.reconstruction.volume_tolerance);
            FPhaseSum exported, retained, total;
            for (const auto& [route, sum] : groups)
                if (route.first == key)
                {
                    const auto amount = sum.value();
                    total.add(amount);
                    (route.second == key ? retained : exported).add(amount);
                    const auto metadata = target_metadata.at(route.second);
                    amounts.push_back({key, route.second, donor.gid, donor.revision,
                        metadata.second, metadata.first, amount, history_groups.at(route).value()});
                }
            const auto old = finite_inventory(donor), delivered = total.value();
            for (auto pair : {std::pair{old.gas, delivered.gas}, std::pair{old.carrier, delivered.carrier}})
            {
                const auto difference = fp::subtract(pair.first, pair.second);
                if (std::abs(difference.value) > difference.absolute_error || difference.absolute_error >
                    std::abs(pair.first.value) * data.options.reconstruction.volume_tolerance + pair.first.absolute_error)
                    throw std::invalid_argument("Finite VoF donor phase closure exceeds its independent budget.");
            }
            trial.finite_budgets_.push_back({donor.gid, old, exported.value(), retained.value()});
        }
    });
    packets.clear();
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& amount : amounts)
        {
            wire::Writer w;
            w.bytes(AMR::encode_adaptive_key(amount.donor));
            w.word(amount.donor_gid);
            w.word(amount.donor_revision);
            w.bytes(AMR::encode_adaptive_key(amount.recipient));
            w.word(amount.recipient_gid);
            finite_phases(w, amount.value);
            finite_phases(w, amount.history);
            budget.packet(w.data.size());
            packets.push_back({amount.recipient_owner, std::move(w.data)});
        }
    });
    const auto incoming_amounts = exchange_owner_packets(packets, comm, limit);
    std::vector<Key> donor_keys;
    struct Transfer
    {
        Key donor, recipient;
        std::uint64_t gid, revision;
        int owner;
        FPhases amount, history;
    };
    std::vector<Transfer> transfers;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        std::set<std::pair<Key, Key>> seen;
        for (const auto& packet : incoming_amounts.packets)
        {
            wire::Reader r{packet.bytes};
            auto dk = AMR::decode_adaptive_key(r.bytes());
            const auto gid = r.word(), revision = r.word();
            auto rk = AMR::decode_adaptive_key(r.bytes());
            const auto recipient_gid = r.word();
            const auto& recipient = data.cells.at(rk);
            if (recipient.owner != comm->getRank() || recipient.gid != recipient_gid || !seen.emplace(dk, rk).second)
                throw std::invalid_argument("Finite VoF positive transfer recipient is duplicate or foreign.");
            auto amount = finite_phases(r);
            auto history = finite_phases(r);
            r.done();
            for (const auto& p : {amount.bulk, amount.gas, amount.carrier})
                if (p.value < 0 || (p.value != 0 && p.value <= p.absolute_error))
                    throw std::invalid_argument("Finite VoF positive transfer has unresolved nonnegativity.");
            budget.add(sizeof(Transfer));
            donor_keys.push_back(dk);
            transfers.push_back({dk, rk, gid, revision, packet.peer, amount, history});
        }
    });
    const auto donor_owners = data.hierarchy->lookup_cells(donor_keys);
    std::shared_ptr<Data> candidate;
    wire::collective(*comm, [&]
    {
        candidate = std::make_shared<Data>(data);
        candidate->control = trial.transaction_;
        std::map<Key, FPhaseSum> totals, histories;
        for (std::size_t i = 0; i < transfers.size(); ++i)
        {
            const auto& transfer = transfers[i];
            const auto& owner = donor_owners[i];
            if (owner.owner != transfer.owner || owner.revision != transfer.revision || owner.address != transfer.gid)
                throw std::invalid_argument("Finite VoF positive transfer donor is stale or foreign.");
            totals[transfer.recipient].add(transfer.amount);
            histories[transfer.recipient].add(transfer.history);
            trial.finite_transfers_.push_back({transfer.gid, data.cells.at(transfer.recipient).gid,
                transfer.owner, comm->getRank(), transfer.amount,
                {fp::upper_double(fp::bound_add(transfer.history.gas.value, transfer.history.gas.absolute_error)),
                 fp::upper_double(fp::bound_add(transfer.history.carrier.value, transfer.history.carrier.absolute_error))}});
        }
        candidate->gas.clear();
        candidate->carrier.clear();
        candidate->phases.clear();
        candidate->roundoff.clear();
        candidate->occupancy.clear();
        candidate->phase_history.clear();
        for (const auto& key : data.topology->order)
        {
            auto& cell = candidate->cells.at(key);
            const auto total = totals.at(key).value();
            const auto actual = cell.finite_geometry->volume();
            const auto closure = fp::subtract(total.bulk, actual);
            if (std::abs(closure.value) > closure.absolute_error || closure.absolute_error >
                actual.value * data.options.reconstruction.volume_tolerance)
                throw std::invalid_argument("Finite VoF transfers do not fill the admitted recipient geometry.");
            const auto gas = fp::publish(total.gas), carrier = fp::publish(total.carrier);
            cell.phase.gas_volume = gas.first;
            cell.phase.carrier_volume = carrier.first;
            cell.roundoff = {gas.second, carrier.second};
            const auto transported_history = histories.at(key).value();
            cell.finite_history.gas = fp::upper_double(fp::bound_add(gas.second,
                fp::bound_add(transported_history.gas.value, transported_history.gas.absolute_error)));
            cell.finite_history.carrier = fp::upper_double(fp::bound_add(carrier.second,
                fp::bound_add(transported_history.carrier.value, transported_history.carrier.absolute_error)));
            const auto previous = data.cells.at(key).occupancy;
            fp::Sum occupancy_sum;
            occupancy_sum.add({gas.first, gas.second});
            occupancy_sum.add({carrier.first, carrier.second});
            occupancy_sum.add(fp::negate(actual));
            const auto published = occupancy_sum.value();
            cell.occupancy.difference = static_cast<double>(published.value);
            cell.occupancy.admitted_bound = fp::upper_double(fp::upward(previous.admitted_bound +
                published.absolute_error + std::abs(closure.value)));
            cell.occupancy.phase_input_bound = fp::upper_double(fp::upward(previous.phase_input_bound +
                gas.second + carrier.second));
            cell.occupancy.transferred_bound = fp::upper_double(fp::upward(previous.transferred_bound +
                total.gas.absolute_error + total.carrier.absolute_error));
            cell.retained_interface.reset();
            if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
            {
                const auto minority = cell.declaration == GeneralVoFPhaseGeometry::PureGas ? carrier.first : gas.first;
                if (minority > actual.absolute_error + 8 * std::numeric_limits<double>::epsilon() * actual.value)
                    cell.declaration = GeneralVoFPhaseGeometry::Infer;
            }
            cell.inherited = true;
            validate_cell(cell, data.options);
            candidate->gas.push_back(gas.first);
            candidate->carrier.push_back(carrier.first);
            candidate->phases.push_back({cell.native_volume, gas.first, carrier.first});
            candidate->roundoff.push_back(cell.roundoff);
            candidate->occupancy.push_back(cell.occupancy);
            candidate->phase_history.push_back(cell.finite_history);
        }
        trial.roundoff_ = candidate->roundoff;
        current(data);
    });
    // Shared signed transfers give opposite incidences; an independent global
    // phase check accounts for wire reduction and one-time publication error.
    std::array<double, 6> local{}, global{};
    wire::collective(*comm, [&]
    {
        std::array<fp::Sum, 4> sums;
        for (std::size_t i = 0; i < data.topology->order.size(); ++i)
        {
            sums[0].add({static_cast<long double>(candidate->gas[i]) - data.gas[i], candidate->roundoff[i].gas});
            sums[1].add({static_cast<long double>(candidate->carrier[i]) - data.carrier[i], candidate->roundoff[i].carrier});
            sums[2].add({std::abs(static_cast<long double>(candidate->gas[i])) + std::abs(data.gas[i]), 0});
            sums[3].add({std::abs(static_cast<long double>(candidate->carrier[i])) + std::abs(data.carrier[i]), 0});
        }
        for (unsigned a = 0; a < 2; ++a)
        {
            const auto sum = sums[a].value();
            local[a] = static_cast<double>(sum.value);
            local[2 + a] = fp::upper_double(fp::upward(sum.absolute_error + std::abs(sum.value - local[a])));
            local[4 + a] = fp::upper_double(sums[2 + a].value().value);
        }
    });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    wire::collective(*comm, [&]
    {
        const auto operations = 8.L * comm->getSize() + 16;
        const auto u = operations * std::numeric_limits<double>::epsilon();
        if (!(u < .25L))
            throw std::invalid_argument("Finite VoF global phase reduction bound is exhausted.");
        for (unsigned phase = 0; phase < 2; ++phase)
        {
            const auto bound = global[2 + phase] + u / (1 - u) * (global[4 + phase] + global[2 + phase]);
            if (!std::isfinite(global[phase]) || !std::isfinite(bound) || std::abs(global[phase]) > bound)
                throw std::invalid_argument("Finite VoF global independent phase conservation failed.");
        }
        current(data);
    });
    trial.candidate_ = std::move(candidate);
    return trial;
}
namespace
{
FValue finite_remap_weighted(const FPhases& p, GeneralVoFFiniteRemapCarrier carrier,
    const std::array<double, 2>& density)
{
    switch (carrier)
    {
    case GeneralVoFFiniteRemapCarrier::BulkVolume: return p.bulk;
    case GeneralVoFFiniteRemapCarrier::GasVolume: return p.gas;
    case GeneralVoFFiniteRemapCarrier::CarrierVolume: return p.carrier;
    case GeneralVoFFiniteRemapCarrier::MixtureMass:
        return fp::add(fp::multiply(p.gas, {density[0], 0}), fp::multiply(p.carrier, {density[1], 0}));
    }
    throw std::invalid_argument("Unknown finite remap channel carrier.");
}
void finite_remap_error(wire::Writer& w, const PhaseError& e)
{
    for (double v : {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound,
             e.fit_residual, e.fit_bound, e.transferred_bound, e.fit_allocation_bound})
        real(w, v);
}
PhaseError finite_remap_error(wire::Reader& r)
{
    PhaseError e;
    e.difference = real(r); e.admitted_bound = real(r); e.native_conversion = real(r);
    e.phase_input_bound = real(r); e.fit_residual = real(r); e.fit_bound = real(r);
    e.transferred_bound = real(r); e.fit_allocation_bound = real(r);
    return e;
}
} // namespace

void GeneralVoF::FiniteRemapTrial::validate_observations() const
{
    if (!source_)
        throw std::invalid_argument("Finite remap has no authenticated source.");
    wire::collective(*source_->hierarchy->communicator(), [&]
    {
        current(*source_);
        if (!candidate_ || source_->options.inventory_policy != PhaseInventoryPolicy::Finite ||
            candidate_->options.inventory_policy != PhaseInventoryPolicy::Finite)
            throw std::invalid_argument("Finite remap has no compatible candidate.");
        current(*candidate_);
    });
}
std::shared_ptr<const GeneralVoF> GeneralVoF::FiniteRemapTrial::candidate() const
{
    validate_observations();
    return std::shared_ptr<const GeneralVoF>(new GeneralVoF(candidate_));
}
GeneralVoF::FiniteRemapTrial GeneralVoF::plan_remap(const Hierarchy::Plan& plan,
    std::shared_ptr<const Hierarchy> target, std::span<const GeneralVoFFiniteOwnedChannels> channels,
    GeneralVoFFiniteRemapOptions options) const
{
    const auto& data = *data_;
    const auto comm = data.hierarchy->communicator();
    const auto limit = std::min(data.options.byte_budget, options.byte_budget);
    std::map<Key, std::vector<FValue>> source_channels;
    std::vector<char> manifest;
    wire::collective(*comm, [&]
    {
        current(data);
        if (data.options.inventory_policy != PhaseInventoryPolicy::Finite || !target ||
            target->communicator().getRawPtr() != comm.getRawPtr() || !target->native_partition() || !limit ||
            !plan.collective_identity() || !std::isfinite(options.maximum_relative_target_defect) ||
            !(options.maximum_relative_target_defect > 0) || !std::isfinite(options.maximum_relative_target_fit) ||
            !(options.maximum_relative_target_fit > 0))
            throw std::invalid_argument("Finite remap requires compatible source, actual native target and options.");
        wire::Budget budget{limit};
        wire::Writer w;
        w.word(data.control);
        w.word(plan.collective_identity());
        w.word(options.channels.size());
        for (double density : options.phase_density)
        {
            if (!std::isfinite(density) || !(density > 0))
                throw std::invalid_argument("Finite remap phase density must be finite and positive.");
            real(w, density);
        }
        real(w, options.maximum_relative_target_defect);
        real(w, options.maximum_relative_target_fit);
        std::set<std::string> names;
        for (const auto& channel : options.channels)
        {
            if (channel.name.empty() || !names.insert(channel.name).second ||
                static_cast<unsigned>(channel.carrier) > static_cast<unsigned>(GeneralVoFFiniteRemapCarrier::MixtureMass))
                throw std::invalid_argument("Finite remap channel schema is malformed.");
            budget.add(channel.name.size() + sizeof(GeneralVoFFiniteRemapChannel));
            w.bytes(std::vector<char>(channel.name.begin(), channel.name.end()));
            w.word(static_cast<unsigned>(channel.carrier));
            w.word(channel.nonnegative);
        }
        budget.packet(w.data.size());
        manifest = std::move(w.data);
        if (options.channels.empty() && channels.empty())
            for (const auto& key : data.topology->order)
                source_channels.emplace(key, std::vector<FValue>{});
        else
            for (const auto& item : channels)
            {
                if (!data.hierarchy->owned().contains(item.key) || item.amounts.size() != options.channels.size() ||
                    !source_channels.emplace(item.key, item.amounts).second)
                    throw std::invalid_argument("Finite remap source channels have duplicate, foreign or missing keys.");
                budget.add(sizeof(GeneralVoFFiniteOwnedChannels) + item.amounts.size() * sizeof(FValue));
                for (std::size_t i = 0; i < item.amounts.size(); ++i)
                {
                    const auto v = item.amounts[i];
                    if (!std::isfinite(v.value) || !std::isfinite(v.absolute_error) || v.absolute_error < 0 ||
                        (options.channels[i].nonnegative && (v.lower() < 0 || v.value < 0)))
                        throw std::invalid_argument("Finite remap source channel is nonfinite or unresolved nonnegative.");
                }
            }
        if (source_channels.size() != data.topology->order.size())
            throw std::invalid_argument("Finite remap channels do not cover every source owner.");
    });
    std::uint64_t manifest_size = manifest.size();
    Teuchos::broadcast(*comm, 0, 1, &manifest_size);
    std::vector<char> agreed;
    wire::collective(*comm, [&]
    {
        if (manifest_size > limit || manifest_size > std::uint64_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Finite remap collective schema exceeds its payload cap.");
        agreed = comm->getRank() == 0 ? manifest : std::vector<char>(std::size_t(manifest_size));
    });
    Teuchos::broadcast(*comm, 0, int(manifest_size), agreed.data());
    wire::collective(*comm, [&]
    {
        if (agreed != manifest)
            throw std::invalid_argument("Finite remap schema, source publication or plan differs across ranks.");
    });
    target = data.hierarchy->apply(plan, std::move(target));
    const auto target_observation = target->observation();
    const bool ownership = target->topology_generation() == data.hierarchy->topology_generation();
    const auto catalog = detail::build_adaptive_overlap_catalog(*data.hierarchy, *target, limit);
    std::map<std::uint64_t, const AMR::AdaptiveCellBinding*> from, to;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& cell : catalog.source_cells)
        {
            budget.add(sizeof(decltype(from)::value_type) + 4 * sizeof(void*));
            if (!from.emplace(cell.gid, &cell).second)
                throw std::invalid_argument("Finite remap source catalog repeats a global ID.");
        }
        for (const auto& cell : catalog.target_cells)
        {
            budget.add(sizeof(decltype(to)::value_type) + 4 * sizeof(void*));
            if (!to.emplace(cell.gid, &cell).second)
                throw std::invalid_argument("Finite remap target catalog repeats a global ID.");
        }
    });
    struct Relation
    {
        Key source, target;
        std::uint64_t source_gid, source_revision, target_gid, target_revision;
        AMR::AdaptiveAncestryRelation::Kind kind;
        int target_owner;
        Meshes::BoundedConvexGeometryPtr overlap;
    };
    std::vector<Relation> requests;
    std::vector<Key> source_keys;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& relation : catalog.relations)
        {
            const auto& source_cell = *from.at(relation.source_gid);
            const auto& target_cell = *to.at(relation.target_gid);
            if (!target->owned().contains(target_cell.key))
                continue;
            budget.add(sizeof(Relation) + 2 * sizeof(Key));
            source_keys.push_back(source_cell.key);
            requests.push_back({source_cell.key, target_cell.key, source_cell.gid, 0, target_cell.gid,
                target->owned().at(target_cell.key).revision, relation.kind, comm->getRank(), {}});
        }
    });
    const auto locations = data.hierarchy->lookup_cells(source_keys);
    std::vector<OwnerPacket> packets;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (std::size_t i = 0; i < requests.size(); ++i)
        {
            auto& request = requests[i];
            request.source_revision = locations[i].revision;
            if (request.source_gid != locations[i].address)
                throw std::invalid_argument("Finite remap source catalog is stale.");
            wire::Writer w;
            w.bytes(AMR::encode_adaptive_key(request.source));
            w.word(request.source_gid); w.word(request.source_revision);
            w.bytes(AMR::encode_adaptive_key(request.target));
            w.word(request.target_gid); w.word(request.target_revision);
            w.word(static_cast<unsigned>(request.kind));
            budget.packet(w.data.size());
            packets.push_back({locations[i].owner, std::move(w.data)});
        }
    });
    const auto incoming = exchange_owner_packets(packets, comm, limit);
    std::vector<Relation> columns;
    std::vector<Key> target_keys;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        std::set<std::pair<Key, Key>> seen;
        std::map<std::pair<Key, Key>, AMR::AdaptiveAncestryRelation::Kind> admitted_relations;
        for (const auto& relation : catalog.relations)
            admitted_relations.emplace(std::pair{from.at(relation.source_gid)->key,
                to.at(relation.target_gid)->key}, relation.kind);
        for (const auto& packet : incoming.packets)
        {
            wire::Reader r{packet.bytes};
            Relation column;
            column.source = AMR::decode_adaptive_key(r.bytes());
            column.source_gid = r.word(); column.source_revision = r.word();
            column.target = AMR::decode_adaptive_key(r.bytes());
            column.target_gid = r.word(); column.target_revision = r.word();
            const auto kind = r.word();
            r.done();
            if (!data.hierarchy->owned().contains(column.source) ||
                !seen.emplace(column.source, column.target).second ||
                kind > static_cast<unsigned>(AMR::AdaptiveAncestryRelation::Kind::ProductIntersection))
                throw std::invalid_argument("Finite remap received duplicate, foreign or malformed overlap request.");
            column.kind = static_cast<AMR::AdaptiveAncestryRelation::Kind>(kind);
            const auto authenticated = admitted_relations.find({column.source, column.target});
            if (authenticated == admitted_relations.end() || authenticated->second != column.kind)
                throw std::invalid_argument("Finite remap request does not match authenticated native ancestry.");
            const auto& cell = data.cells.at(column.source);
            if (cell.gid != column.source_gid || cell.revision != column.source_revision)
                throw std::invalid_argument("Finite remap overlap source revision differs.");
            column.target_owner = packet.peer;
            budget.add(sizeof(Relation) + sizeof(Key));
            target_keys.push_back(column.target);
            columns.push_back(std::move(column));
        }
    });
    const auto target_locations = target->lookup_cells(target_keys);
    const auto target_leaves = target->fetch_leaves(target_keys, limit);
    std::map<Key, Meshes::BoundedConvexGeometryPtr> imported_targets;
    std::map<Key, std::vector<std::size_t>> by_source;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (std::size_t i = 0; i < columns.size(); ++i)
        {
            auto& column = columns[i];
            const auto& location = target_locations[i];
            const auto& leaf = target_leaves[i];
            const auto& source_leaf = data.hierarchy->owned().at(column.source);
            if (location.owner != column.target_owner || location.address != column.target_gid ||
                location.revision != column.target_revision || leaf.key != column.target ||
                leaf.revision != column.target_revision || leaf.material != source_leaf.material ||
                leaf.key.region != source_leaf.key.region)
                throw std::invalid_argument("Finite remap target owner/material/revision differs.");
            if (!imported_targets.contains(column.target))
            {
                Meshes::BoundedGeometryLimits limits;
                limits.byte_budget = limit;
                auto geometry = Meshes::BoundedConvexGeometry::from_geometry(*target->geometry(leaf), limits);
                budget.add(geometry->storage_bytes());
                imported_targets.emplace(column.target, std::move(geometry));
            }
            const auto& source_geometry = data.cells.at(column.source).finite_geometry;
            const auto& target_geometry = imported_targets.at(column.target);
            // Native plan ancestry authenticates exact containment for immediate
            // refinement/coarsening. Numerical partition checks below still
            // establish that the supplied whole pieces form a complete cover.
            if (column.kind == AMR::AdaptiveAncestryRelation::Kind::Unchanged ||
                column.kind == AMR::AdaptiveAncestryRelation::Kind::Coarsening)
                column.overlap = source_geometry;
            else if (column.kind == AMR::AdaptiveAncestryRelation::Kind::Refinement)
                column.overlap = target_geometry;
            else
                column.overlap = source_geometry->intersect(*target_geometry);
            if (!column.overlap)
                throw std::invalid_argument("Finite remap ancestry overlap is geometrically empty.");
            budget.add(sizeof(std::size_t) + column.overlap->storage_bytes());
            by_source[column.source].push_back(i);
        }
    });
    struct Output
    {
        Relation relation;
        FPhases phases, history;
        std::vector<FValue> amounts;
        FValue inherited_occupancy;
        PhaseError original_error;
        PhaseRoundoff original_roundoff;
        GeneralVoFPhaseGeometry declaration = GeneralVoFPhaseGeometry::Infer;
        std::optional<PolyhedralInterface> retained_interface;
    };
    std::vector<Output> outgoing;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& key : data.topology->order)
        {
            const auto& source = data.cells.at(key);
            const auto found = by_source.find(key);
            if (found == by_source.end() || found->second.empty())
                throw std::invalid_argument("Finite remap has an uncovered source owner.");
            const auto& indices = found->second;
            std::vector<Meshes::BoundedConvexGeometryPtr> cover;
            for (auto i : indices)
                cover.push_back(columns[i].overlap);
            source.finite_geometry->certify_partition(cover, options.maximum_relative_target_defect);
            const bool whole = indices.size() == 1 &&
                (columns[indices[0]].kind == AMR::AdaptiveAncestryRelation::Kind::Unchanged ||
                 columns[indices[0]].kind == AMR::AdaptiveAncestryRelation::Kind::Coarsening);
            PolyhedralInterface fitted;
            FLabel labels;
            if (!whole)
            {
                fitted = reconstruct_finite_cell(data, key);
                labels = finite_label(source, fitted, data.options.reconstruction.volume_tolerance);
            }
            const auto old = finite_inventory(source);
            // Carrier existence is a source contract even on ownership-only
            // copies; the whole-cell fast path cannot legitimize an unsupported
            // nonzero channel merely because it needs no geometric weighting.
            for (std::size_t c = 0; c < options.channels.size(); ++c)
            {
                const auto carrier = finite_remap_weighted(old, options.channels[c].carrier, options.phase_density);
                const auto amount = source_channels.at(key)[c];
                if (carrier.value == 0 && carrier.absolute_error == 0)
                {
                    if (amount.value != 0 || amount.absolute_error != 0)
                        throw std::invalid_argument("Finite remap nonzero channel has no physical carrier.");
                }
                else if (!(carrier.lower() > 0))
                    throw std::invalid_argument("Finite remap source channel carrier is unresolved.");
            }
            FPhaseSum total;
            std::vector<fp::Sum> channel_totals(options.channels.size());
            for (auto i : indices)
            {
                const auto& column = columns[i];
                Output output;
                output.relation = column;
                output.phases = whole ? old : labels.apply(*column.overlap, fitted);
                output.history = whole ? FPhases{{}, {source.finite_history.gas, 0},
                    {source.finite_history.carrier, 0}} : labels.history(*column.overlap, fitted);
                output.declaration = source.declaration;
                output.original_error = source.occupancy;
                output.original_roundoff = source.roundoff;
                if (ownership)
                    output.retained_interface = source.retained_interface;
                const auto bulk_share = whole ? FValue{1, 0} : fp::divide(output.phases.bulk, old.bulk);
                output.inherited_occupancy = fp::multiply({source.occupancy.admitted_bound, 0}, bulk_share);
                for (std::size_t c = 0; c < options.channels.size(); ++c)
                {
                    const auto& input = source_channels.at(key)[c];
                    FValue amount;
                    if (whole)
                        amount = input;
                    else
                    {
                        const auto denominator = finite_remap_weighted(old, options.channels[c].carrier, options.phase_density);
                        const auto numerator = finite_remap_weighted(output.phases, options.channels[c].carrier, options.phase_density);
                        if (denominator.value == 0 && denominator.absolute_error == 0)
                        {
                            if (input.value != 0 || input.absolute_error != 0 ||
                                numerator.value != 0 || numerator.absolute_error != 0)
                                throw std::invalid_argument("Finite remap nonzero channel has no physical carrier.");
                        }
                        else
                            amount = fp::multiply(input, fp::divide(numerator, denominator));
                    }
                    output.amounts.push_back(amount);
                    channel_totals[c].add(amount);
                }
                total.add(output.phases);
                budget.add(sizeof(Output) + output.amounts.size() * sizeof(FValue));
                outgoing.push_back(std::move(output));
            }
            const auto delivered = total.value();
            for (auto pair : {std::pair{old.gas, delivered.gas}, std::pair{old.carrier, delivered.carrier}})
            {
                const auto residual = fp::subtract(pair.second, pair.first);
                if (std::abs(residual.value) > residual.absolute_error ||
                    residual.absolute_error > std::abs(pair.first.value) * options.maximum_relative_target_fit)
                    throw std::invalid_argument("Finite remap source phase closure exceeds its unchanged target-fit bound.");
            }
            for (std::size_t c = 0; c < options.channels.size(); ++c)
            {
                const auto residual = fp::subtract(channel_totals[c].value(), source_channels.at(key)[c]);
                if (std::abs(residual.value) > residual.absolute_error)
                    throw std::invalid_argument("Finite remap weighted channel column is not conservative.");
            }
        }
    });
    packets.clear();
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& output : outgoing)
        {
            const auto& column = output.relation;
            wire::Writer w;
            w.bytes(AMR::encode_adaptive_key(column.source));
            w.word(column.source_gid); w.word(column.source_revision);
            w.bytes(AMR::encode_adaptive_key(column.target));
            w.word(column.target_gid); w.word(column.target_revision);
            finite_phases(w, output.phases);
            finite_phases(w, output.history);
            finite_value(w, output.inherited_occupancy);
            w.word(static_cast<unsigned>(output.declaration));
            w.word(output.amounts.size());
            for (auto amount : output.amounts)
                finite_value(w, amount);
            if (ownership)
            {
                finite_remap_error(w, output.original_error);
                real(w, output.original_roundoff.gas); real(w, output.original_roundoff.carrier);
                w.word(bool(output.retained_interface));
                if (output.retained_interface)
                    plane(w, *output.retained_interface);
            }
            budget.packet(w.data.size());
            packets.push_back({column.target_owner, std::move(w.data)});
        }
    });
    const auto responses = exchange_owner_packets(packets, comm, limit);
    struct Row
    {
        FPhaseSum phases, history;
        fp::Sum occupancy;
        std::vector<fp::Sum> channels;
        std::optional<GeneralVoFPhaseGeometry> declaration;
        PhaseError original_error;
        PhaseRoundoff original_roundoff;
        std::optional<PolyhedralInterface> retained_interface;
        std::size_t pieces = 0;
    };
    std::map<Key, Row> rows;
    FiniteRemapTrial result;
    result.source_ = data_;
    result.ownership_only_ = ownership;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& column : columns)
        {
            budget.add(sizeof(GeneralVoFFiniteCommonCell) + column.overlap->storage_bytes());
            result.common_cells_.push_back({column.source, column.target, column.source_gid,
                column.target_gid, column.target_owner, column.overlap});
        }
    });
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        std::map<std::pair<Key, Key>, std::size_t> expected;
        for (std::size_t i = 0; i < requests.size(); ++i)
            if (!expected.emplace(std::pair{requests[i].source, requests[i].target}, i).second)
                throw std::invalid_argument("Finite remap ancestry repeats a source/target pair.");
        std::set<std::pair<Key, Key>> seen;
        for (const auto& packet : responses.packets)
        {
            wire::Reader r{packet.bytes};
            const auto sk = AMR::decode_adaptive_key(r.bytes());
            const auto sgid = r.word(), revision = r.word();
            const auto tk = AMR::decode_adaptive_key(r.bytes());
            const auto tgid = r.word(), target_revision = r.word();
            const auto item = expected.find({sk, tk});
            if (item == expected.end() || !seen.emplace(sk, tk).second)
                throw std::invalid_argument("Finite remap output is duplicate or not an authenticated overlap.");
            const auto& request = requests[item->second];
            if (packet.peer != locations[item->second].owner || sgid != request.source_gid ||
                revision != request.source_revision || tgid != request.target_gid || target_revision != request.target_revision)
                throw std::invalid_argument("Finite remap result source/target ownership or revision differs.");
            const auto phases = finite_phases(r), history = finite_phases(r);
            const auto inherited_occupancy = finite_value(r);
            const auto declaration = r.word(), count = r.word();
            if (declaration > static_cast<unsigned>(GeneralVoFPhaseGeometry::PureCarrier) || count != options.channels.size())
                throw std::invalid_argument("Finite remap result schema is malformed.");
            for (auto value : {phases.bulk, phases.gas, phases.carrier, history.gas, history.carrier, inherited_occupancy})
                if (value.value < 0 || value.lower() < 0)
                    throw std::invalid_argument("Finite remap output has unresolved nonnegative amounts.");
            auto& row = rows[tk];
            if (row.pieces++ == 0)
            {
                row.channels.resize(options.channels.size());
                row.declaration = static_cast<GeneralVoFPhaseGeometry>(declaration);
            }
            else if (*row.declaration != static_cast<GeneralVoFPhaseGeometry>(declaration))
                row.declaration = GeneralVoFPhaseGeometry::Infer;
            row.phases.add(phases);
            row.history.add(history);
            row.occupancy.add(inherited_occupancy);
            for (auto& channel : row.channels)
                channel.add(finite_value(r));
            if (ownership)
            {
                if (row.pieces != 1 || sk != tk)
                    throw std::invalid_argument("Finite ownership migration changes cell ancestry.");
                row.original_error = finite_remap_error(r);
                row.original_roundoff = {real(r), real(r)};
                const auto retained = r.word();
                if (retained > 1)
                    throw std::invalid_argument("Finite migration interface tag is malformed.");
                if (retained)
                {
                    row.retained_interface = plane(r);
                    if (!row.retained_interface->finite_geometry)
                        throw std::invalid_argument("Finite migration cannot acquire an exact interface.");
                }
            }
            r.done();
            budget.add(sizeof(Row) + sizeof(GeneralVoFFiniteRemapRoute) + row.channels.size() * sizeof(fp::Sum));
            result.routes_.push_back({sk, tk, sgid, tgid, packet.peer, comm->getRank(), phases});
        }
        if (seen.size() != expected.size() || rows.size() != target->owned().size())
            throw std::invalid_argument("Finite remap outputs do not cover every target owner.");
    });
    // Independently validate recipient geometry before physical publication.
    // Sparse ancestry selects candidates; a net volume sum cannot certify cover.
    std::map<Key, Meshes::BoundedConvexGeometryPtr> local_source_geometry;
    std::map<Key, std::vector<Meshes::BoundedConvexGeometryPtr>> target_covers;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& request : requests)
        {
            if (!local_source_geometry.contains(request.source))
            {
                const auto existing = data.cells.find(request.source);
                auto geometry = existing != data.cells.end() ? existing->second.finite_geometry :
                    Meshes::BoundedConvexGeometry::from_geometry(*from.at(request.source_gid)->geometry);
                budget.add(geometry->storage_bytes());
                local_source_geometry.emplace(request.source, std::move(geometry));
            }
            if (!imported_targets.contains(request.target))
            {
                auto geometry = Meshes::BoundedConvexGeometry::from_geometry(*to.at(request.target_gid)->geometry);
                budget.add(geometry->storage_bytes());
                imported_targets.emplace(request.target, std::move(geometry));
            }
            const auto& source_geometry = local_source_geometry.at(request.source);
            const auto& target_geometry = imported_targets.at(request.target);
            auto overlap = source_geometry;
            if (request.kind == AMR::AdaptiveAncestryRelation::Kind::Refinement)
                overlap = target_geometry;
            else if (request.kind == AMR::AdaptiveAncestryRelation::Kind::ProductIntersection)
                overlap = source_geometry->intersect(*target_geometry);
            if (!overlap)
                throw std::invalid_argument("Finite remap recipient overlap is empty.");
            target_covers[request.target].push_back(std::move(overlap));
        }
        for (const auto& [key, leaf] : target->owned())
            imported_targets.at(key)->certify_partition(target_covers.at(key), options.maximum_relative_target_defect);
    });
    // Build only geometry/maps in a private pure placeholder. Actual remapped
    // inventories and their inherited error evidence are installed together.
    std::vector<Key> order;
    std::vector<double> zero, native_volume;
    wire::collective(*comm, [&]
    {
        for (const auto& [key, leaf] : target->owned())
        {
            order.push_back(key);
            zero.push_back(0);
            native_volume.push_back(target->metrics(leaf).volume);
        }
    });
    const auto placeholder = std::make_shared<const GeneralVoF>(target, order, zero, native_volume, data.options);
    auto candidate = std::make_shared<Data>(*placeholder->data_);
    wire::collective(*comm, [&]
    {
        candidate->gas.clear(); candidate->carrier.clear(); candidate->phases.clear();
        candidate->roundoff.clear(); candidate->phase_history.clear(); candidate->occupancy.clear();
        wire::Budget budget{limit};
        for (const auto& key : candidate->topology->order)
        {
            auto& cell = candidate->cells.at(key);
            auto& row = rows.at(key);
            const auto totals = row.phases.value(), history = row.history.value();
            const auto actual = cell.finite_geometry->volume();
            const auto closure = fp::subtract(totals.bulk, actual);
            if (std::abs(closure.value) > closure.absolute_error ||
                closure.absolute_error > actual.value * options.maximum_relative_target_defect)
                throw std::invalid_argument("Finite remap recipient coverage exceeds its geometric error budget.");
            const auto gas = fp::publish(totals.gas), carrier = fp::publish(totals.carrier);
            cell.phase.gas_volume = gas.first;
            cell.phase.carrier_volume = carrier.first;
            cell.roundoff = ownership ? row.original_roundoff : PhaseRoundoff{gas.second, carrier.second};
            cell.finite_history = {fp::upper_double(history.gas.upper()), fp::upper_double(history.carrier.upper())};
            if (!ownership)
            {
                cell.finite_history.gas = fp::upper_double(fp::upward(cell.finite_history.gas + gas.second));
                cell.finite_history.carrier = fp::upper_double(fp::upward(cell.finite_history.carrier + carrier.second));
            }
            cell.declaration = row.declaration.value_or(GeneralVoFPhaseGeometry::Infer);
            cell.retained_interface = ownership ? row.retained_interface : std::optional<PolyhedralInterface>{};
            cell.inherited = true;
            if (ownership)
                cell.occupancy = row.original_error;
            else
            {
                const auto discrepancy = fp::subtract(fp::add({gas.first, gas.second}, {carrier.first, carrier.second}), actual);
                const auto inherited = row.occupancy.value();
                const auto admitted = fp::upward(inherited.upper() + discrepancy.absolute_error + std::abs(closure.value));
                if (std::abs(discrepancy.value) > admitted ||
                    admitted > actual.value * options.maximum_relative_target_defect)
                    throw std::invalid_argument("Finite remap physical occupancy exceeds its original target-defect bound.");
                cell.occupancy.difference = static_cast<double>(discrepancy.value);
                cell.occupancy.admitted_bound = fp::upper_double(admitted);
                cell.occupancy.phase_input_bound = fp::upper_double(fp::upward(gas.second + carrier.second));
                cell.occupancy.transferred_bound = fp::upper_double(fp::upward(totals.gas.absolute_error + totals.carrier.absolute_error));
            }
            validate_cell(cell, data.options);
            candidate->gas.push_back(gas.first); candidate->carrier.push_back(carrier.first);
            candidate->phases.push_back({cell.native_volume, gas.first, carrier.first});
            candidate->roundoff.push_back(cell.roundoff); candidate->phase_history.push_back(cell.finite_history);
            candidate->occupancy.push_back(cell.occupancy);
            GeneralVoFFiniteOwnedChannels output;
            output.key = key;
            for (std::size_t i = 0; i < row.channels.size(); ++i)
            {
                const auto value = row.channels[i].value();
                if (options.channels[i].nonnegative && value.lower() < 0)
                    throw std::invalid_argument("Finite remap nonnegative channel output is unresolved.");
                output.amounts.push_back(value);
            }
            budget.add(sizeof(output) + output.amounts.size() * sizeof(FValue));
            result.channels_.push_back(std::move(output));
        }
        current(data);
        if (!target->geometry_unchanged() || target->observation() != target_observation)
            throw std::invalid_argument("Finite remap target observation changed during preparation.");
    });
    import_phases(*candidate);
    wire::collective(*comm, [&] { retained_budget(*candidate); });
    result.candidate_ = std::move(candidate);
    result.validate_observations();
    return result;
}
#include "FVM/details/GeneralVoFExactTransport.inc"
} // namespace SimpleFluid::VoF

namespace SimpleFluid::VoF
{
void GeneralVoF::enable_exact_geometry_statistics(bool enabled)
{
    Meshes::enable_exact_geometry_statistics(enabled);
}
void GeneralVoF::reset_geometry_statistics()
{
    Meshes::reset_exact_geometry_statistics();
    Meshes::reset_bounded_geometry_statistics();
}
Meshes::GeometryStatisticsSnapshot GeneralVoF::geometry_statistics()
{
    return {Meshes::exact_geometry_statistics_domain(), Meshes::bounded_geometry_statistics_domain(),
        Meshes::exact_geometry_statistics(), Meshes::bounded_geometry_statistics()};
}
}
