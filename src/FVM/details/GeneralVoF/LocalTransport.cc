/** @file LocalTransport.cc
 * @brief Conservative source-resolved signed canonical face transport.
 */
#include "FVM/details/GeneralVoF/LocalTransport.hh"
#include "FVM/details/GeneralVoF/FaceTransport.hh"
#include "FVM/details/GeneralVoF/PhaseAllocation.hh"
#include "FVM/details/GeneralVoF/Wire.hh"
#include <Teuchos_Array.hpp>
namespace SimpleFluid::VoF
{
namespace
{
constexpr auto local_mode = ArithmeticMode::Ordinary;
using LocalMath = fp::Arithmetic<local_mode>;
using Triple = std::array<long double, 3>;
using Sum = SimpleFluid::detail::CompensatedSum<long double>;
struct TripleSum
{
    std::array<Sum, 3> sums;
    void add(const Triple& v, long double factor = 1)
    {
        for (unsigned p = 0; p < 3; ++p)
            sums[p] += factor * v[p];
    }
    Triple value() const { return {sums[0].value(), sums[1].value(), sums[2].value()}; }
};
Triple values(const PLIC::DynamicPhaseMeasures& a)
{
    return {a.bulk.value, a.gas.value, a.carrier.value};
}
PLIC::DynamicPhaseMeasures amounts(const Triple& a)
{
    return {LocalMath::make_value(a[0]), LocalMath::make_value(a[1]), LocalMath::make_value(a[2])};
}
void triple(wire::Writer& w, const Triple& a)
{
    for (auto v : a)
        finite_real(w, v);
}
Triple triple(wire::Reader& r)
{
    return {finite_real(r), finite_real(r), finite_real(r)};
}
struct Source
{
    std::uint64_t gid = 0, revision = 0;
    int owner = -1;
    Triple old{};
    PhaseRoundoff history;
    PhaseError occupancy;
    GeneralVoFPhaseGeometry declaration = GeneralVoFPhaseGeometry::Infer;
};
Source source_record(const Cell& c)
{
    const auto volume = PLIC::ConvexCellGeometry<local_mode>::from_dynamic(c.finite_geometry)->volume().value;
    return {c.gid, c.revision, c.owner, {volume, c.phase.gas_volume, c.phase.carrier_volume}, c.finite_history,
        c.occupancy, c.declaration};
}
void source(wire::Writer& w, const Source& s)
{
    w.word(s.gid);
    w.word(s.revision);
    w.word(s.owner);
    triple(w, s.old);
    real(w, s.history.gas);
    real(w, s.history.carrier);
    for (auto value : {s.occupancy.difference, s.occupancy.admitted_bound, s.occupancy.phase_input_bound,
             s.occupancy.fit_bound, s.occupancy.transferred_bound, s.occupancy.fit_allocation_bound})
        real(w, value);
    w.word(std::uint64_t(s.declaration));
}
Source source(wire::Reader& r)
{
    Source s;
    s.gid = r.word();
    s.revision = r.word();
    s.owner = int(r.word());
    s.old = triple(r);
    s.history = {real(r), real(r)};
    s.occupancy.difference = real(r);
    s.occupancy.admitted_bound = real(r);
    s.occupancy.phase_input_bound = real(r);
    s.occupancy.fit_bound = real(r);
    s.occupancy.transferred_bound = real(r);
    s.occupancy.fit_allocation_bound = real(r);
    s.declaration = GeneralVoFPhaseGeometry(r.word());
    return s;
}
void stamp(wire::Writer& w, const GeneralVoF::Data& data, std::uint64_t transaction)
{
    w.word(data.control);
    w.word(transaction);
    w.word(data.hierarchy->topology_generation());
    w.word(data.hierarchy->partition_generation());
}
void stamp(wire::Reader& r, const GeneralVoF::Data& data, std::uint64_t transaction)
{
    if (r.word() != data.control || r.word() != transaction || r.word() != data.hierarchy->topology_generation() ||
        r.word() != data.hierarchy->partition_generation())
        throw std::invalid_argument("Interface ledger packet source or transaction differs.");
}
struct Group
{
    std::uint64_t donor = 0, revision = 0;
    int owner = -1, direction = 1;
    Triple low{}, high{};
    long double share = 0, allocation = 0, closure = 0;
    std::array<long double, 2> low_history{}, high_history{}, low_absent_fraction{}, high_absent_fraction{};
    std::array<bool, 2> low_geometry{}, high_geometry{};
};
struct Face
{
    const AMR::AdaptiveTile* tile = nullptr;
    bool geometric = false;
    long double theta = 1;
    std::map<std::pair<std::uint64_t, int>, Group> groups;
    GeneralVoFFaceChainDiagnostic chain;
};
Triple blend(const Group& g, long double theta)
{
    Triple result;
    for (unsigned p = 0; p < 3; ++p)
        result[p] = theta == 1 ? g.high[p] : theta == 0 ? g.low[p] : (1 - theta) * g.low[p] + theta * g.high[p];
    return result;
}
struct Demand
{
    Triple endpoint{}, donor{};
    long double closure_positive = 0, closure_negative = 0;
};
struct Request
{
    std::uint64_t face = 0, cell = 0;
    unsigned kind = 0;
    int peer = -1;
};
struct Coefficient
{
    int owner = -1;
    std::uint64_t revision = 0;
    TripleSum amount;
    Sum occupancy;
    std::array<Sum, 2> absent_history, absent_fraction, absent_magnitude;
    std::size_t operations = 1;
};
long double absent_weight(const Coefficient& coefficient, unsigned phase)
{
    const auto u = (64.L + 32.L * coefficient.operations) * std::numeric_limits<long double>::epsilon();
    if (!(u < .25L))
        throw std::invalid_argument("Interface absent-phase history arithmetic bound is exhausted.");
    return fp::bound_add(std::abs(coefficient.absent_fraction[phase].value()),
        fp::upward(u / (1 - u) * coefficient.absent_magnitude[phase].value()));
}
struct CellUpdate
{
    std::map<std::uint64_t, Coefficient> donors;
    TripleSum face_update, face_magnitude;
    Sum allocation, closure, geometry_difference, phase_cut_closure;
    long double absolute_geometry_difference = 0;
    bool gas_geometry = false, carrier_geometry = false;
};
long double ratio(long double available, long double demand)
{
    if (demand == 0)
        return 1;
    if (!(available > 0))
        return 0;
    const auto result = std::min(1.L, std::nextafter(available, 0.L) / demand);
    return result > 0 && result < 1 ? std::nextafter(result, 0.L) : result;
}
long double bound_product(long double a, long double b)
{
    return a == 0 || b == 0 ? 0 : fp::upward(fp::upward(std::abs(a)) * b);
}
} // namespace
GeneralVoF::Trial GeneralVoF::plan_interface_transport(Trial trial, const LocalTransportControls& controls) const
{
    const auto& data = *data_;
    const auto comm = data.hierarchy->communicator();
    const int rank = comm->getRank();
    const auto limit = data.options.byte_budget;
    const auto safe = [&](auto&& work) { wire::collective(*comm, std::forward<decltype(work)>(work)); };
    // Admit the donor-cell support without constructing a discarded candidate.
    auto low = admit_low_order_ledger(data, controls.face_volume, trial.timestep_);
    auto geometry =
        prepare_local_sweeps(data, controls.traces, controls.flux, controls.face_budgets, controls.packet_context);
    std::map<std::uint64_t, const AMR::AdaptiveTile*> tiles;
    std::map<std::uint64_t, Face> faces;
    std::map<std::uint64_t, Triple> low_export;
    std::map<std::uint64_t, Demand> demands;
    std::vector<Request> requests;
    std::vector<OwnerPacket> packets;
    std::map<std::tuple<std::uint64_t, unsigned, std::uint64_t>, int> expected_ratios;
    safe(
        [&]
        {
            trial.interface_local_ = true;
            trial.numerical_support_transfers_ = std::move(low.transfers);
            trial.numerical_support_budgets_ = std::move(low.budgets);
            trial.unbound_diagnostics_ = geometry.diagnostics;
            trial.unbound_diagnostics_.maximum_outgoing_courant = low.diagnostics.maximum_outgoing_courant;
            trial.unbound_diagnostics_.maximum_continuity_defect = low.diagnostics.maximum_continuity_defect;
            for (const auto& b : trial.numerical_support_budgets_)
                low_export.emplace(b.donor, values(b.exported));
            for (const auto& tile : data.topology->tiles)
                tiles.emplace(tile.face_gid, &tile);
            std::map<std::uint64_t, const GeneralVoFDirectedStream*> low_faces;
            for (const auto& s : low.streams)
                low_faces.emplace(s.face, &s);
            wire::Budget budget{limit};
            for (auto& entry : geometry.owned_faces)
            {
                const auto& tile = *tiles.at(entry.face);
                if (!tile.second || tile.first_owner != rank || tile.exterior || faces.contains(entry.face))
                    throw std::invalid_argument("Interface geometry canonical face ownership differs.");
                Face face;
                face.tile = &tile;
                face.geometric = entry.geometric;
                face.chain = entry.chain;
                if (!(entry.quality_theta > 0) || entry.quality_theta > 1 || !std::isfinite(entry.quality_theta))
                    throw std::invalid_argument("Interface geometric quality coefficient is invalid.");
                face.theta = entry.quality_theta;
                trial.unbound_diagnostics_.quality_limited_faces += face.theta < 1;
                trial.unbound_diagnostics_.minimum_quality_theta =
                    std::min(trial.unbound_diagnostics_.minimum_quality_theta, face.theta);
                if (low_faces.contains(entry.face))
                {
                    const auto& s = *low_faces.at(entry.face);
                    const auto& donor = data.cells.at(data.topology->by_gid.at(s.donor));
                    auto& g = face.groups[{s.donor, s.direction}];
                    g.donor = s.donor;
                    g.owner = s.donor_owner;
                    g.revision = donor.revision;
                    g.direction = s.direction;
                    g.low = {s.volume, s.gas, s.liquid};
                    const long double capacity =
                        static_cast<long double>(donor.phase.gas_volume) + donor.phase.carrier_volume;
                    g.low_absent_fraction = {s.volume / capacity, s.volume / capacity};
                    g.low_history = {s.volume / capacity * donor.finite_history.gas,
                        s.volume / capacity * donor.finite_history.carrier};
                    g.low_geometry = {donor.declaration != GeneralVoFPhaseGeometry::PureCarrier,
                        donor.declaration != GeneralVoFPhaseGeometry::PureGas};
                }
                if (entry.geometric)
                    for (const auto& high : entry.groups)
                    {
                        const auto& c = high.contribution;
                        const auto& donor = data.cells.at(data.topology->by_gid.at(c.donor));
                        if (c.face != entry.face || c.donor_owner != donor.owner ||
                            (c.direction != 1 && c.direction != -1))
                            throw std::invalid_argument("Interface geometric group source differs.");
                        auto& g = face.groups[{c.donor, c.direction}];
                        g.donor = c.donor;
                        g.owner = c.donor_owner;
                        g.revision = donor.revision;
                        g.direction = c.direction;
                        const auto high_values = values(c.amount);
                        for (unsigned p = 0; p < 3; ++p)
                        {
                            if (high_values[p] < 0)
                                throw std::invalid_argument("Interface geometric group has negative magnitude.");
                            g.high[p] += high_values[p];
                        }
                        g.share += high.bulk_share;
                        g.allocation += high.allocation;
                        g.closure += high.cut_closure;
                        g.high_history[0] += high.gas_history;
                        g.high_history[1] += high.carrier_history;
                        // An absent physical phase has pure geometric support;
                        // its normalized PhaseLabel history is exactly bulk/V.
                        g.high_absent_fraction[0] += high.bulk_share;
                        g.high_absent_fraction[1] += high.bulk_share;
                        g.high_geometry[0] = g.high_geometry[0] || high.gas_geometry;
                        g.high_geometry[1] = g.high_geometry[1] || high.carrier_geometry;
                    }
                else
                    for (auto& [key, g] : face.groups)
                    {
                        g.high = g.low;
                        g.high_history = g.low_history;
                        g.high_absent_fraction = g.low_absent_fraction;
                        g.high_geometry = g.low_geometry;
                    }
                budget.add(sizeof(Face) + face.groups.size() * (sizeof(Group) + 6 * sizeof(void*)));
                faces.emplace(entry.face, std::move(face));
            }
            for (const auto& [gid, tile] : tiles)
                if (tile->first_owner == rank && tile->second && !tile->exterior && !faces.contains(gid))
                    throw std::invalid_argument("Interface canonical geometry face coverage differs.");
            // The limiter sees source-resolved corrections before any cancellation
            // between opposite groups. A single coefficient scales every field.
            for (const auto& [gid, f] : faces)
            {
                TripleSum delta;
                long double closure_delta = 0;
                std::map<std::uint64_t, TripleSum> donor_delta;
                for (const auto& [key, g] : f.groups)
                {
                    Triple change;
                    for (unsigned p = 0; p < 3; ++p)
                        change[p] = g.direction * (g.high[p] - g.low[p]);
                    delta.add(change);
                    closure_delta += change[0] + g.direction * g.closure;
                    donor_delta[g.donor].add(change);
                }
                for (const auto& [cell, owner, incidence] :
                    {std::tuple{data.cells.at(f.tile->first).gid, f.tile->first_owner, 1},
                        std::tuple{data.cells.at(*f.tile->second).gid, f.tile->second_owner, -1}})
                {
                    wire::Writer w;
                    stamp(w, data, trial.transaction_);
                    w.word(gid);
                    w.word(0);
                    w.word(cell);
                    w.word(data.cells.at(data.topology->by_gid.at(cell)).revision);
                    const auto dv = delta.value();
                    Triple adverse;
                    for (unsigned p = 0; p < 3; ++p)
                        adverse[p] = fp::upward(std::max(incidence * dv[p], 0.L));
                    triple(w, adverse);
                    finite_real(w, std::max(-incidence * closure_delta, 0.L));
                    finite_real(w, std::max(incidence * closure_delta, 0.L));
                    budget.packet(w.data.size());
                    packets.push_back({owner, std::move(w.data)});
                    expected_ratios.emplace(std::tuple{gid, 0U, cell}, owner);
                }
                for (const auto& [donor, sum] : donor_delta)
                {
                    const auto& c = data.cells.at(data.topology->by_gid.at(donor));
                    const auto dv = sum.value();
                    Triple increase{};
                    for (const auto& [recipient, incidence] : {std::pair{data.cells.at(f.tile->first).gid, 1},
                             std::pair{data.cells.at(*f.tile->second).gid, -1}})
                        if (recipient != donor)
                            for (unsigned p = 0; p < 3; ++p)
                                increase[p] = fp::bound_add(increase[p], std::max(-incidence * dv[p], 0.L));
                    wire::Writer w;
                    stamp(w, data, trial.transaction_);
                    w.word(gid);
                    w.word(1);
                    w.word(donor);
                    w.word(c.revision);
                    triple(w, increase);
                    finite_real(w, 0);
                    finite_real(w, 0);
                    budget.packet(w.data.size());
                    packets.push_back({c.owner, std::move(w.data)});
                    expected_ratios.emplace(std::tuple{gid, 1U, donor}, c.owner);
                }
            }
        });
    auto received = exchange_owner_packets(packets, comm, limit);
    packets.clear();
    using GO = MeshHandle<>::global_ordinal_type;
    Teuchos::Array<GO> demanded_faces;
    safe(
        [&]
        {
            std::set<GO> unique;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                stamp(r, data, trial.transaction_);
                const auto face = r.word();
                if (face > std::uint64_t(std::numeric_limits<GO>::max()))
                    throw std::invalid_argument("Interface limiter face GID exceeds native range.");
                unique.insert(GO(face));
            }
            demanded_faces.assign(unique.begin(), unique.end());
        });
    Teuchos::Array<int> demanded_face_owners(demanded_faces.size(), -1);
    (void) data.topology->native.mesh->owned_face_map()->getRemoteIndexList(demanded_faces(), demanded_face_owners());
    std::map<std::uint64_t, int> face_owners;
    safe(
        [&]
        {
            for (std::size_t i = 0; i < std::size_t(demanded_faces.size()); ++i)
            {
                if (demanded_face_owners[i] < 0)
                    throw std::invalid_argument("Interface limiter references a foreign canonical face.");
                face_owners.emplace(std::uint64_t(demanded_faces[i]), demanded_face_owners[i]);
            }
            std::set<std::tuple<std::uint64_t, unsigned, std::uint64_t>> unique;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                stamp(r, data, trial.transaction_);
                Request request;
                request.face = r.word();
                request.kind = unsigned(r.word());
                request.cell = r.word();
                const auto revision = r.word();
                const auto& c = data.cells.at(data.topology->by_gid.at(request.cell));
                if (c.owner != rank || c.revision != revision || request.kind > 1 ||
                    face_owners.at(request.face) != packet.peer ||
                    !unique.emplace(request.face, request.kind, request.cell).second ||
                    (request.kind == 0 &&
                        (!tiles.contains(request.face) || tiles.at(request.face)->first_owner != packet.peer)))
                    throw std::invalid_argument("Interface limiter owner/revision/face differs.");
                const auto d = triple(r);
                auto& accumulated = demands[request.cell];
                for (unsigned p = 0; p < 3; ++p)
                {
                    if (d[p] < 0)
                        throw std::invalid_argument("Interface limiter has negative demand.");
                    auto& target = request.kind == 0 ? accumulated.endpoint[p] : accumulated.donor[p];
                    target = fp::bound_add(target, d[p]);
                }
                accumulated.closure_positive = fp::bound_add(accumulated.closure_positive, finite_real(r));
                accumulated.closure_negative = fp::bound_add(accumulated.closure_negative, finite_real(r));
                r.done();
                request.peer = packet.peer;
                requests.push_back(request);
            }
            std::map<std::uint64_t, long double> endpoint_ratio, donor_ratio;
            for (const auto& key : data.topology->order)
            {
                const auto& c = data.cells.at(key);
                const auto& demand = demands[c.gid];
                Sum canonical;
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    canonical += (tile.first == key ? -1 : 1) * controls.flux.at(tile.face_gid).value;
                }
                const auto old = source_record(c).old;
                // Use the unrounded positive ledger. Binary64 publication can
                // round an endpoint capacity upward before a limiting correction.
                const auto baseline = low.endpoint_capacity.at(c.gid);
                const Triple available{old[0] + canonical.value(), baseline[1], baseline[2]};
                long double er = 1, dr = 1;
                for (unsigned p = 0; p < 3; ++p)
                {
                    er = std::min(er, ratio(available[p], demand.endpoint[p]));
                    dr = std::min(dr, ratio(old[p] - low_export.at(c.gid)[p], demand.donor[p]));
                }
                const auto allowed = data.options.reconstruction.volume_tolerance * old[0];
                er = std::min({er, ratio(allowed - canonical.value(), demand.closure_positive),
                    ratio(allowed + canonical.value(), demand.closure_negative)});
                endpoint_ratio.emplace(c.gid, er);
                donor_ratio.emplace(c.gid, dr);
            }
            wire::Budget budget{limit};
            for (const auto& request : requests)
            {
                wire::Writer w;
                stamp(w, data, trial.transaction_);
                w.word(request.face);
                w.word(request.kind);
                w.word(request.cell);
                finite_real(w, request.kind == 0 ? endpoint_ratio.at(request.cell) : donor_ratio.at(request.cell));
                budget.packet(w.data.size());
                packets.push_back({request.peer, std::move(w.data)});
            }
        });
    received = exchange_owner_packets(packets, comm, limit);
    packets.clear();
    safe(
        [&]
        {
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                stamp(r, data, trial.transaction_);
                const auto face = r.word();
                const auto kind = unsigned(r.word());
                const auto cell = r.word();
                const auto key = std::tuple{face, kind, cell};
                const auto value = finite_real(r);
                r.done();
                if (!expected_ratios.contains(key) || expected_ratios.at(key) != packet.peer || value < 0 || value > 1)
                    throw std::invalid_argument("Interface limiter ratio authority differs.");
                faces.at(face).theta = std::min(faces.at(face).theta, value);
                expected_ratios.erase(key);
            }
            if (!expected_ratios.empty())
                throw std::invalid_argument("Interface limiter ratio coverage differs.");
            wire::Budget budget{limit};
            for (const auto& [gid, f] : faces)
            {
                auto& diagnostic = trial.unbound_diagnostics_;
                diagnostic.minimum_theta = std::min(diagnostic.minimum_theta, f.theta);
                diagnostic.limited_faces += f.theta < 1;
                if (f.geometric)
                {
                    const auto bin = f.theta == 0 ? 0u : f.theta == 1 ? 9u : unsigned(8 * f.theta) + 1;
                    ++diagnostic.geometric_theta_histogram[bin];
                }
                TripleSum signed_amount;
                for (const auto& [key, g] : f.groups)
                {
                    const auto current = blend(g, f.theta);
                    signed_amount.add(current, g.direction);
                    if (current[0] != 0 || current[1] != 0 || current[2] != 0)
                    {
                        trial.finite_faces_.push_back({gid, g.donor, g.owner, g.direction, amounts(current)});
                        ++diagnostic.streams;
                        diagnostic.limited_streams += f.theta < 1;
                        diagnostic.limited_volume += f.theta < 1 ? current[0] : 0;
                    }
                }
                if (std::abs(signed_amount.value()[0] - controls.flux.at(gid).value) > controls.face_budgets.at(gid))
                    throw std::invalid_argument("Interface final face chain exceeds original face allowance.");
                bool positive = false, negative = false;
                for (const auto& [key, g] : f.groups)
                    if (blend(g, f.theta)[0] > 0)
                    {
                        positive |= g.direction > 0;
                        negative |= g.direction < 0;
                    }
                diagnostic.gross_counterstreams += positive && negative;
                for (const auto& [cell, owner, incidence] :
                    {std::tuple{data.cells.at(f.tile->first).gid, f.tile->first_owner, 1},
                        std::tuple{data.cells.at(*f.tile->second).gid, f.tile->second_owner, -1}})
                {
                    wire::Writer w;
                    stamp(w, data, trial.transaction_);
                    w.word(gid);
                    w.word(cell);
                    w.word(incidence == 1);
                    finite_real(w, signed_amount.value()[0]);
                    w.word(f.groups.size());
                    for (const auto& [key, g] : f.groups)
                    {
                        const auto current = blend(g, f.theta);
                        w.word(g.donor);
                        w.word(g.owner);
                        w.word(g.revision);
                        w.word(g.direction == 1);
                        triple(w, current);
                        finite_real(w, f.geometric ? f.theta * g.share : 0);
                        finite_real(w, f.theta * g.allocation);
                        finite_real(w, f.theta * g.closure);
                        for (unsigned p = 0; p < 2; ++p)
                        {
                            finite_real(w, f.theta == 1 ? g.high_history[p]
                                           : f.theta == 0
                                               ? g.low_history[p]
                                               : (1 - f.theta) * g.low_history[p] + f.theta * g.high_history[p]);
                            finite_real(w, f.theta == 1   ? g.high_absent_fraction[p]
                                           : f.theta == 0 ? g.low_absent_fraction[p]
                                                          : (1 - f.theta) * g.low_absent_fraction[p] +
                                                                f.theta * g.high_absent_fraction[p]);
                            w.word(((1 - f.theta) * g.low[0] > 0 && g.low_geometry[p]) ||
                                   (f.theta * g.high[0] > 0 && g.high_geometry[p]));
                        }
                    }
                    budget.packet(w.data.size());
                    packets.push_back({owner, std::move(w.data)});
                }
            }
        });
    received = exchange_owner_packets(packets, comm, limit);
    packets.clear();
    std::map<std::uint64_t, CellUpdate> updates;
    std::map<std::uint64_t, std::pair<int, std::uint64_t>> needed_sources;
    safe(
        [&]
        {
            for (const auto& key : data.topology->order)
            {
                const auto& c = data.cells.at(key);
                const auto s = source_record(c);
                auto& coefficient = updates[c.gid].donors[c.gid];
                coefficient.owner = rank;
                coefficient.revision = c.revision;
                coefficient.amount.add(s.old);
                coefficient.occupancy += 1;
                coefficient.absent_history[0] += c.finite_history.gas;
                coefficient.absent_history[1] += c.finite_history.carrier;
                coefficient.absent_fraction[0] += 1;
                coefficient.absent_fraction[1] += 1;
                coefficient.absent_magnitude[0] += 1;
                coefficient.absent_magnitude[1] += 1;
                needed_sources.emplace(c.gid, std::pair{rank, c.revision});
            }
            std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
            std::set<std::uint64_t> chain_faces;
            for (const auto& tile : data.topology->tiles)
                if (tile.exterior || !tile.second)
                {
                    const auto zero = LocalMath::make_value(0);
                    trial.finite_face_chain_.push_back({tile.face_gid, controls.flux.at(tile.face_gid), zero, zero,
                        controls.face_budgets.at(tile.face_gid)});
                    chain_faces.insert(tile.face_gid);
                }
            wire::Budget budget{limit};
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                stamp(r, data, trial.transaction_);
                const auto face = r.word(), recipient = r.word();
                const int incidence = r.word() ? 1 : -1;
                if (!tiles.contains(face) || tiles.at(face)->first_owner != packet.peer ||
                    !updates.contains(recipient) || !seen.emplace(face, recipient).second)
                    throw std::invalid_argument("Interface final bundle canonical authority differs.");
                const auto& tile = *tiles.at(face);
                if ((incidence == 1 ? data.cells.at(tile.first).gid : data.cells.at(*tile.second).gid) != recipient)
                    throw std::invalid_argument("Interface final bundle incidence differs.");
                auto& update = updates.at(recipient);
                const auto constructed = finite_real(r);
                const auto difference = constructed - controls.flux.at(face).value;
                update.geometry_difference += incidence * difference;
                update.absolute_geometry_difference =
                    fp::bound_add(update.absolute_geometry_difference, std::abs(difference));
                if (chain_faces.insert(face).second)
                    trial.finite_face_chain_.push_back(
                        {face, controls.flux.at(face), LocalMath::make_value(constructed),
                            LocalMath::make_value(difference), controls.face_budgets.at(face)});
                const auto count = r.word();
                if (count > limit / sizeof(Coefficient))
                    throw PLIC::GeometryError(
                        PLIC::GeometryErrorCode::ResourceLimit, "Interface bundle count exceeds budget.");
                for (std::size_t i = 0; i < count; ++i)
                {
                    const auto donor = r.word();
                    const auto owner = int(r.word());
                    const auto revision = r.word();
                    const int direction = r.word() ? 1 : -1;
                    const auto current = triple(r);
                    const auto sign = -incidence * direction;
                    auto [found, inserted] = needed_sources.emplace(donor, std::pair{owner, revision});
                    if ((!inserted && found->second != std::pair{owner, revision}) || owner < 0 ||
                        owner >= comm->getSize())
                        throw std::invalid_argument("Interface donor binding differs across faces.");
                    auto& coefficient = update.donors[donor];
                    coefficient.owner = owner;
                    coefficient.revision = revision;
                    coefficient.amount.add(current, sign);
                    update.face_update.add(current, sign);
                    update.face_magnitude.add(current);
                    coefficient.occupancy += sign * finite_real(r);
                    update.allocation += sign * finite_real(r);
                    const auto phase_cut = finite_real(r);
                    update.closure += sign * (current[0] + phase_cut);
                    update.phase_cut_closure += -sign * phase_cut;
                    for (unsigned p = 0; p < 2; ++p)
                    {
                        coefficient.absent_history[p] += sign * finite_real(r);
                        const auto fraction = finite_real(r);
                        coefficient.absent_fraction[p] += sign * fraction;
                        coefficient.absent_magnitude[p] += std::abs(fraction);
                        ++coefficient.operations;
                        const auto presence = bool(r.word());
                        (p == 0 ? update.gas_geometry : update.carrier_geometry) |= presence;
                    }
                    budget.add(sizeof(Coefficient) + 6 * sizeof(void*));
                }
                r.done();
            }
            for (const auto& key : data.topology->order)
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    if (tile.second && !tile.exterior && !seen.contains({tile.face_gid, data.cells.at(key).gid}))
                        throw std::invalid_argument("Interface final bundle coverage differs.");
                }
            for (const auto& [gid, binding] : needed_sources)
            {
                wire::Writer w;
                stamp(w, data, trial.transaction_);
                w.word(gid);
                w.word(binding.second);
                budget.packet(w.data.size());
                packets.push_back({binding.first, std::move(w.data)});
            }
        });
    received = exchange_owner_packets(packets, comm, limit);
    packets.clear();
    safe(
        [&]
        {
            wire::Budget budget{limit};
            std::set<std::pair<int, std::uint64_t>> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                stamp(r, data, trial.transaction_);
                const auto gid = r.word(), revision = r.word();
                r.done();
                const auto& c = data.cells.at(data.topology->by_gid.at(gid));
                if (c.owner != rank || c.revision != revision || !seen.emplace(packet.peer, gid).second)
                    throw std::invalid_argument("Interface source descriptor owner/revision differs.");
                wire::Writer w;
                stamp(w, data, trial.transaction_);
                source(w, source_record(c));
                budget.packet(w.data.size());
                packets.push_back({packet.peer, std::move(w.data)});
            }
        });
    received = exchange_owner_packets(packets, comm, limit);
    packets.clear();
    std::map<std::uint64_t, Source> sources;
    safe(
        [&]
        {
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                stamp(r, data, trial.transaction_);
                const auto s = source(r);
                r.done();
                if (!needed_sources.contains(s.gid) || needed_sources.at(s.gid) != std::pair{s.owner, s.revision} ||
                    packet.peer != s.owner || !sources.emplace(s.gid, s).second)
                    throw std::invalid_argument("Interface source descriptor authentication differs.");
            }
            if (sources.size() != needed_sources.size())
                throw std::invalid_argument("Interface source descriptor coverage differs.");
            wire::Budget budget{limit};
            // Donor budgets consume combined W[c,d], rather than positive pieces
            // from distinct faces that may cancel in the same signed coefficient.
            for (const auto& [recipient, update] : updates)
                for (const auto& [donor, coefficient] : update.donors)
                {
                    const auto a = coefficient.amount.value();
                    wire::Writer w;
                    stamp(w, data, trial.transaction_);
                    w.word(donor);
                    w.word(coefficient.revision);
                    w.word(recipient);
                    triple(w, a);
                    budget.packet(w.data.size());
                    packets.push_back({coefficient.owner, std::move(w.data)});
                }
        });
    received = exchange_owner_packets(packets, comm, limit);
    packets.clear();
    Teuchos::Array<GO> budget_recipients;
    safe(
        [&]
        {
            std::set<GO> unique;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                stamp(r, data, trial.transaction_);
                (void) r.word();
                (void) r.word();
                const auto recipient = r.word();
                if (recipient > std::uint64_t(std::numeric_limits<GO>::max()))
                    throw std::invalid_argument("Interface receipt GID exceeds native range.");
                unique.insert(GO(recipient));
            }
            budget_recipients.assign(unique.begin(), unique.end());
        });
    Teuchos::Array<int> budget_recipient_owners(budget_recipients.size(), -1);
    (void) data.topology->native.mesh->owned_cell_map()->getRemoteIndexList(
        budget_recipients(), budget_recipient_owners());
    std::map<std::uint64_t, int> recipient_owners;
    safe(
        [&]
        {
            for (std::size_t i = 0; i < std::size_t(budget_recipients.size()); ++i)
            {
                if (budget_recipient_owners[i] < 0)
                    throw std::invalid_argument("Interface donor budget references a foreign recipient.");
                recipient_owners.emplace(std::uint64_t(budget_recipients[i]), budget_recipient_owners[i]);
            }
            struct Budget
            {
                TripleSum positive, negative, diagonal;
                bool self = false;
            };
            std::map<std::uint64_t, Budget> budgets;
            std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                stamp(r, data, trial.transaction_);
                const auto donor = r.word(), revision = r.word(), recipient = r.word();
                const auto a = triple(r);
                r.done();
                const auto& c = data.cells.at(data.topology->by_gid.at(donor));
                if (c.owner != rank || c.revision != revision || !seen.emplace(donor, recipient).second ||
                    recipient_owners.at(recipient) != packet.peer)
                    throw std::invalid_argument("Interface signed donor budget binding differs.");
                auto& budget = budgets[donor];
                if (recipient == donor)
                {
                    if (packet.peer != rank)
                        throw std::invalid_argument("Interface signed diagonal owner differs.");
                    budget.self = true;
                    budget.diagonal.add(a);
                }
                else
                {
                    Triple positive, negative;
                    for (unsigned p = 0; p < 3; ++p)
                    {
                        positive[p] = std::max(a[p], 0.L);
                        negative[p] = std::max(-a[p], 0.L);
                    }
                    budget.positive.add(positive);
                    budget.negative.add(negative);
                }
            }
            for (const auto& key : data.topology->order)
            {
                const auto& c = data.cells.at(key);
                const auto old = source_record(c).old;
                const auto& b = budgets.at(c.gid);
                const auto positive = b.positive.value(), negative = b.negative.value(), diagonal = b.diagonal.value();
                if (!b.self)
                    throw std::invalid_argument("Interface signed donor diagonal is absent.");
                for (unsigned p = 0; p < 3; ++p)
                {
                    Sum identity;
                    identity += diagonal[p];
                    identity += positive[p];
                    identity += -negative[p];
                    identity += -old[p];
                    const auto arithmetic = fp::upward(64 * std::numeric_limits<long double>::epsilon() *
                                                       (std::abs(diagonal[p]) + positive[p] + negative[p] + old[p]));
                    if (positive[p] > old[p] || std::abs(identity.value()) > arithmetic)
                        throw std::invalid_argument("Interface signed donor positive export or identity failed.");
                }
                trial.signed_budgets_.push_back(
                    {c.gid, amounts(old), amounts(positive), amounts(negative), amounts(diagonal)});
            }
        });
    std::shared_ptr<Data> candidate;
    safe(
        [&]
        {
            candidate = std::make_shared<Data>(data);
            candidate->control = trial.transaction_;
            candidate->gas.clear();
            candidate->carrier.clear();
            candidate->phases.clear();
            candidate->roundoff.clear();
            candidate->phase_history.clear();
            candidate->occupancy.clear();
            for (const auto& key : data.topology->order)
            {
                const auto& old = data.cells.at(key);
                auto& c = candidate->cells.at(key);
                const auto& update = updates.at(c.gid);
                TripleSum total;
                std::array<long double, 2> history{};
                long double inherited_occupancy = 0, inherited_fit_allocation = 0, inherited_input = 0;
                long double inherited_transferred = 0, pure_fit = 0;
                Sum occupancy_prediction;
                for (const auto& [donor, coefficient] : update.donors)
                {
                    const auto& s = sources.at(donor);
                    const auto a = coefficient.amount.value();
                    total.add(a);
                    PhaseRoundoff receipt_history;
                    for (unsigned p = 0; p < 2; ++p)
                    {
                        const auto old_phase = s.old[1 + p];
                        const auto old_history = p == 0 ? s.history.gas : s.history.carrier;
                        const auto amount = old_phase > 0 ? bound_product(a[1 + p] / old_phase, old_history)
                                                          : bound_product(absent_weight(coefficient, p), old_history);
                        history[p] = fp::bound_add(history[p], amount);
                        (p == 0 ? receipt_history.gas : receipt_history.carrier) = fp::upper_double(amount);
                    }
                    trial.signed_receipts_.push_back({donor, c.gid, s.owner, rank, amounts(a), receipt_history});
                    const auto r = coefficient.occupancy.value();
                    occupancy_prediction += r * s.occupancy.difference;
                    inherited_occupancy =
                        fp::bound_add(inherited_occupancy, bound_product(r, s.occupancy.admitted_bound));
                    inherited_fit_allocation =
                        fp::bound_add(inherited_fit_allocation, bound_product(r, s.occupancy.fit_allocation_bound));
                    inherited_input = fp::bound_add(inherited_input, bound_product(r, s.occupancy.phase_input_bound));
                    inherited_transferred =
                        fp::bound_add(inherited_transferred, bound_product(r, s.occupancy.transferred_bound));
                    if (c.declaration != GeneralVoFPhaseGeometry::Infer)
                    {
                        const auto p = c.declaration == GeneralVoFPhaseGeometry::PureGas ? 2U : 1U;
                        const auto factor =
                            s.old[p] > 0 ? std::abs(a[p] / s.old[p]) : absent_weight(coefficient, p - 1);
                        pure_fit = fp::bound_add(pure_fit,
                            bound_product(factor, std::max(s.occupancy.fit_bound, s.occupancy.admitted_bound)));
                    }
                }
                const auto final = total.value();
                const auto old_amount = source_record(old).old;
                const auto face_update = update.face_update.value(), gross = update.face_magnitude.value();
                const auto u = (64.L + 16.L * update.donors.size() + 16.L * data.topology->cell_tiles.at(key).size()) *
                               std::numeric_limits<long double>::epsilon();
                if (!(u < .25L))
                    throw std::invalid_argument("Interface ledger arithmetic operation bound is exhausted.");
                Triple arithmetic;
                for (unsigned p = 0; p < 3; ++p)
                {
                    arithmetic[p] = fp::upward(u / (1 - u) * (old_amount[p] + gross[p] + std::abs(final[p])));
                    if (final[p] < 0 || std::abs(final[p] - (old_amount[p] + face_update[p])) > arithmetic[p])
                        throw std::invalid_argument("Interface signed endpoint or face identity failed.");
                }
                const auto closure = update.closure.value();
                const auto allowed = data.options.reconstruction.volume_tolerance * old_amount[0];
                if (std::abs(closure) > allowed || std::abs(final[0] - old_amount[0]) > allowed)
                    throw std::invalid_argument("Interface final closure exceeds original cell allowance.");
                const auto gas = LocalMath::publish(LocalMath::make_value(final[1]));
                const auto carrier = LocalMath::publish(LocalMath::make_value(final[2]));
                c.phase.gas_volume = gas.first;
                c.phase.carrier_volume = carrier.first;
                c.roundoff = {fp::upper_double(fp::bound_add(gas.second, arithmetic[1])),
                    fp::upper_double(fp::bound_add(carrier.second, arithmetic[2]))};
                c.finite_history = {fp::upper_double(fp::bound_add(history[0], c.roundoff.gas)),
                    fp::upper_double(fp::bound_add(history[1], c.roundoff.carrier))};
                Sum published_defect_sum;
                published_defect_sum += gas.first;
                published_defect_sum += carrier.first;
                published_defect_sum += -old_amount[0];
                const auto published_defect = published_defect_sum.value();
                const auto publication = fp::bound_add(c.roundoff.gas, c.roundoff.carrier);
                const auto prediction = occupancy_prediction.value() + update.allocation.value() + closure;
                if (std::abs(published_defect - prediction) > fp::bound_add(publication, arithmetic[0]))
                    throw std::invalid_argument("Interface signed occupancy identity failed.");
                const auto provenance = allocation_provenance<local_mode>(LocalMath::make_value(published_defect),
                    LocalMath::make_value(update.allocation.value()), LocalMath::make_value(inherited_occupancy),
                    LocalMath::make_value(inherited_fit_allocation), LocalMath::make_value(closure),
                    fp::bound_add(publication, arithmetic[0]));
                c.occupancy.difference = double(published_defect);
                c.occupancy.admitted_bound = fp::upper_double(provenance.admitted_bound);
                c.occupancy.fit_allocation_bound = fp::upper_double(provenance.fit_allocation_bound);
                c.occupancy.phase_input_bound = fp::upper_double(fp::bound_add(inherited_input, publication));
                c.occupancy.transferred_bound = fp::upper_double(fp::bound_add(inherited_transferred, publication));
                c.retained_interface.reset();
                c.requires_interface = false;
                c.inherited = true;
                if ((c.declaration == GeneralVoFPhaseGeometry::PureGas && update.carrier_geometry) ||
                    (c.declaration == GeneralVoFPhaseGeometry::PureCarrier && update.gas_geometry))
                    c.declaration = GeneralVoFPhaseGeometry::Infer;
                if (gas.first == 0)
                    c.declaration = GeneralVoFPhaseGeometry::PureCarrier;
                if (carrier.first == 0)
                    c.declaration = GeneralVoFPhaseGeometry::PureGas;
                c.occupancy.fit_bound = c.declaration == GeneralVoFPhaseGeometry::Infer
                                            ? 0
                                            : fp::upper_double(fp::bound_add(pure_fit, publication));
                c.occupancy.fit_residual = c.declaration == GeneralVoFPhaseGeometry::Infer     ? 0
                                           : c.declaration == GeneralVoFPhaseGeometry::PureGas ? carrier.first
                                                                                               : gas.first;
                validate_cell<local_mode>(c, data.options);
                candidate->gas.push_back(gas.first);
                candidate->carrier.push_back(carrier.first);
                candidate->phases.push_back({c.native_volume, gas.first, carrier.first});
                candidate->roundoff.push_back(c.roundoff);
                candidate->phase_history.push_back(c.finite_history);
                candidate->occupancy.push_back(c.occupancy);
                Sum canonical;
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    canonical += (tile.first == key ? 1 : -1) * controls.flux.at(tile.face_gid).value;
                }
                Sum combined;
                combined += canonical.value();
                combined += update.geometry_difference.value();
                combined += update.phase_cut_closure.value();
                if (std::abs(combined.value()) > allowed)
                    throw std::invalid_argument(
                        "Interface combined diagnostic closure exceeds original cell allowance.");
                trial.finite_cell_closure_.push_back({key, c.gid, c.canonical_face_count, c.ordinary_quota_scale,
                    LocalMath::make_value(canonical.value()), LocalMath::make_value(update.geometry_difference.value()),
                    LocalMath::make_value(combined.value()), update.absolute_geometry_difference, allowed,
                    update.phase_cut_closure.value()});
                trial.unbound_diagnostics_.maximum_occupancy_defect =
                    std::max(trial.unbound_diagnostics_.maximum_occupancy_defect, std::abs(published_defect));
            }
            trial.roundoff_ = candidate->roundoff;
            current(data);
        });
    // Independent phase conservation excludes inherited history and geometric
    // occupancy allowances from its physical acceptance budget.
    std::array<double, 8> local{}, global{};
    safe(
        [&]
        {
            std::array<Sum, 8> sums;
            for (std::size_t i = 0; i < data.phases.size(); ++i)
                for (unsigned p = 0; p < 2; ++p)
                {
                    const auto old = p == 0 ? data.gas[i] : data.carrier[i];
                    const auto next = p == 0 ? candidate->gas[i] : candidate->carrier[i];
                    sums[p] += static_cast<long double>(next) - old;
                    sums[2 + p] += p == 0 ? candidate->roundoff[i].gas : candidate->roundoff[i].carrier;
                    sums[4 + p] += std::abs(static_cast<long double>(next)) + std::abs(static_cast<long double>(old));
                    sums[6 + p] += old;
                }
            for (unsigned p = 0; p < 2; ++p)
            {
                local[p] = double(sums[p].value());
                local[2 + p] = fp::upper_double(
                    fp::bound_add(sums[2 + p].value(), std::abs(sums[p].value() - static_cast<long double>(local[p]))));
                local[4 + p] = fp::upper_double(sums[4 + p].value());
                local[6 + p] = fp::upper_double(sums[6 + p].value());
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    safe(
        [&]
        {
            const auto u = (8.L * comm->getSize() + 16) * std::numeric_limits<double>::epsilon();
            if (!(u < .25L))
                throw std::invalid_argument("Interface ledger global arithmetic bound is exhausted.");
            for (unsigned p = 0; p < 2; ++p)
            {
                const auto bound = data.options.reconstruction.volume_tolerance * global[6 + p] + global[2 + p] +
                                   u / (1 - u) * (global[4 + p] + global[2 + p]);
                if (!std::isfinite(global[p]) || !std::isfinite(bound) || std::abs(global[p]) > bound)
                    throw std::invalid_argument("Interface ledger global independent phase conservation failed.");
            }
            current(data);
        });
    import_phases<local_mode>(*candidate);
    safe(
        [&]
        {
            current(data);
            retained_budget<local_mode>(*candidate);
        });
    trial.candidate_ = std::move(candidate);
    trial.validate_retained_algebraic_payload();
    return trial;
}
} // namespace SimpleFluid::VoF
