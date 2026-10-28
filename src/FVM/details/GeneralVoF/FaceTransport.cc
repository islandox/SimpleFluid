/** @file FaceTransport.cc
 * @brief Conservative fixed-mesh directed transport; geometry is measurement only.
 */
#include "FVM/details/GeneralVoF/FaceTransport.hh"
#include "FVM/details/GeneralVoF/Wire.hh"
#include "FVM/details/GeneralVoF/PhaseCleanup.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/PlaneMeasure.hh"
#include "utils/CompensatedSum.hh"
namespace SimpleFluid::VoF
{
namespace
{
using Stream = GeneralVoFDirectedStream;
using Backend = GeneralVoFTransportBackend;
constexpr auto mode = ArithmeticMode::Ordinary;
using Math = fp::Arithmetic<mode>;
PLIC::DynamicPhaseMeasures measures(long double q, long double g, long double l)
{
    return {Math::make_value(q), Math::make_value(g), Math::make_value(l)};
}
void write_stream(wire::Writer& w, const Stream& s)
{
    w.word(s.face);
    w.word(s.donor);
    w.word(s.recipient);
    w.word(s.donor_owner);
    w.word(s.recipient_owner);
    w.word(s.direction == 1);
    for (auto v : {s.volume, s.gas, s.liquid, s.mass, s.gas_low, s.gas_high, s.liquid_low, s.liquid_high, s.theta})
        finite_real(w, v);
    w.word(s.lower_order);
}
Stream read_stream(wire::Reader& r)
{
    Stream s;
    s.face = r.word();
    s.donor = r.word();
    s.recipient = r.word();
    s.donor_owner = int(r.word());
    s.recipient_owner = int(r.word());
    s.direction = r.word() ? 1 : -1;
    s.volume = finite_real(r);
    s.gas = finite_real(r);
    s.liquid = finite_real(r);
    s.mass = finite_real(r);
    s.gas_low = finite_real(r);
    s.gas_high = finite_real(r);
    s.liquid_low = finite_real(r);
    s.liquid_high = finite_real(r);
    s.theta = finite_real(r);
    s.lower_order = r.word();
    return s;
}
struct Totals
{
    SimpleFluid::detail::CompensatedSum<long double> bulk_out, bulk_in, gas_out, gas_in, liquid_out, liquid_in;
};
struct FaceLedger
{
    std::map<std::uint64_t, const AMR::AdaptiveTile*> tiles;
    std::map<std::uint64_t, std::set<int>> subscribers;
    std::map<std::uint64_t, Stream> authority, streams;
    std::map<std::uint64_t, Totals> totals;
    GeneralVoFTransportDiagnostics diagnostics;
};
struct FaceAdmission
{
    std::vector<GeneralVoFFiniteTransfer> transfers;
    std::vector<GeneralVoFFiniteDonorBudget> budgets;
};
FaceLedger build_face_ledger(const GeneralVoF::Data& data, std::span<const GeneralVoFFaceVolume> controls, double dt,
    std::array<double, 2> density, std::uint64_t time_stage)
{
    const auto comm = data.hierarchy->communicator();
    const auto rank = comm->getRank();
    const auto limit = data.options.byte_budget;
    std::map<std::uint64_t, const AMR::AdaptiveTile*> tiles;
    std::map<std::uint64_t, long double> phi, control_error;
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            current(data);
            if (data.options.arithmetic_mode != mode || !std::isfinite(dt) || dt < 0 ||
                controls.size() != data.topology->tiles.size() || !std::isfinite(density[0]) || density[0] <= 0 ||
                !std::isfinite(density[1]) || density[1] <= 0)
                throw std::invalid_argument(
                    "Face transport requires Ordinary, finite dt and positive constant densities.");
            wire::Budget budget{limit};
            budget.add(data.cells.size() * (sizeof(Cell) + sizeof(Totals)) + controls.size() * (sizeof(Stream) + 256));
            for (const auto& tile : data.topology->tiles)
            {
                tiles.emplace(tile.face_gid, &tile);
                if (!tile.second_image_identity.empty())
                    throw std::invalid_argument("Face transport periodic images are not qualified.");
            }
            for (const auto& c : controls)
            {
                if (!tiles.contains(c.face) || !std::isfinite(c.integrated_volume) || c.exact_integrated_volume ||
                    !phi.emplace(c.face, c.bounded_integrated_volume ? c.bounded_integrated_volume->value
                                                                     : static_cast<long double>(c.integrated_volume))
                        .second)
                    throw std::invalid_argument("Face transport control is foreign, duplicated or nonfinite.");
                if (c.bounded_integrated_volume)
                {
                    const auto& input = *c.bounded_integrated_volume;
                    fp::validate(input);
                    if (input.arithmetic_mode != mode || !std::isfinite(input.value) ||
                        !std::isfinite(input.absolute_error) || input.absolute_error < 0)
                        throw std::invalid_argument("Face transport arithmetic control differs.");
                }
                control_error.emplace(
                    c.face, c.bounded_integrated_volume ? c.bounded_integrated_volume->absolute_error : 0);
                const auto& tile = *tiles.at(c.face);
                if ((tile.exterior || !tile.second || dt == 0) && phi.at(c.face) != 0)
                    throw std::invalid_argument("Face transport requires zero flux at impermeable boundaries/zero dt.");
                wire::Writer w;
                w.word(c.face);
                w.word(data.control);
                w.word(time_stage);
                real(w, dt);
                real(w, density[0]);
                real(w, density[1]);
                finite_real(w, phi.at(c.face));
                finite_real(w, control_error.at(c.face));
                budget.packet(w.data.size());
                packets.push_back({tile.first_owner, std::move(w.data)});
            }
        });
    agree_transaction(comm, data.control);
    // Scalar stage controls also agree on empty ranks.
    std::array<double, 3> settings{dt, density[0], density[1]}, low{}, high{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 3, settings.data(), low.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 3, settings.data(), high.data());
    std::uint64_t first_stage = 0, last_stage = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &time_stage, &first_stage);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &time_stage, &last_stage);
    wire::collective(*comm,
        [&]
        {
            if (first_stage != last_stage)
                throw std::invalid_argument("Face transport time stages differ.");
        });
    wire::collective(*comm,
        [&]
        {
            if (low != high)
                throw std::invalid_argument("Face transport controls differ.");
        });
    GeneralVoFTransportDiagnostics diagnostics;
    std::map<std::uint64_t, std::set<int>> subscribers;
    std::map<std::uint64_t, Stream> authority;
    const auto requested = exchange_owner_packets(packets, comm, limit);
    packets.clear();
    wire::collective(*comm,
        [&]
        {
            std::map<std::uint64_t, AMR::AdaptiveBytes> agreed;
            for (const auto& p : requested.packets)
            {
                wire::Reader r{p.bytes};
                const auto face = r.word();
                if (!tiles.contains(face) || tiles.at(face)->first_owner != rank || r.word() != data.control ||
                    r.word() != time_stage || real(r) != dt || real(r) != density[0] || real(r) != density[1] ||
                    finite_real(r) != phi.at(face) || finite_real(r) != control_error.at(face))
                    throw std::invalid_argument("Face transport shared controls/ownership differ.");
                r.done();
                if (!subscribers[face].insert(p.peer).second)
                    throw std::invalid_argument("Face transport duplicate subscription.");
                const auto [it, inserted] = agreed.emplace(face, p.bytes);
                if (!inserted && it->second != p.bytes)
                    throw std::invalid_argument("Face transport shared controls differ.");
            }
            for (const auto& [face, peers] : subscribers)
            {
                const auto& tile = *tiles.at(face);
                const auto flux = phi.at(face);
                if (flux == 0)
                    continue;
                const auto& donor = data.cells.at(flux > 0 ? tile.first : *tile.second);
                const auto& recipient = data.cells.at(flux > 0 ? *tile.second : tile.first);
                const long double g = donor.phase.gas_volume, l = donor.phase.carrier_volume, q = std::abs(flux);
                if (!(g + l > 0))
                    throw std::invalid_argument("Face transport donor has no capacity.");
                Stream s{face, donor.gid, recipient.gid, donor.owner, recipient.owner, flux > 0 ? 1 : -1};
                s.volume = q;
                // Evaluate the smaller physical inventory directly, independently of geometry hints.
                if (g <= l)
                {
                    s.gas = q * (g / (g + l));
                    s.liquid = q - s.gas;
                }
                else
                {
                    s.liquid = q * (l / (g + l));
                    s.gas = q - s.liquid;
                }
                s.gas_low = s.gas_high = s.gas;
                s.liquid_low = s.liquid_high = s.liquid;
                s.mass = density[0] * s.gas + density[1] * s.liquid;
                authority.emplace(face, s);
            }
            wire::Budget budget{limit};
            for (const auto& [face, s] : authority)
                for (const auto peer : subscribers.at(face))
                {
                    wire::Writer w;
                    write_stream(w, s);
                    budget.packet(w.data.size());
                    packets.push_back({peer, std::move(w.data)});
                }
        });
    const auto distributed = exchange_owner_packets(packets, comm, limit);
    std::map<std::uint64_t, Stream> streams;
    std::map<std::uint64_t, Totals> totals;
    wire::collective(*comm,
        [&]
        {
            for (const auto& p : distributed.packets)
            {
                wire::Reader r{p.bytes};
                auto s = read_stream(r);
                r.done();
                if (!tiles.contains(s.face) || p.peer != tiles.at(s.face)->first_owner ||
                    !streams.emplace(s.face, s).second)
                    throw std::invalid_argument("Face transport stream ownership/identity differs.");
                auto& d = totals[s.donor];
                auto& c = totals[s.recipient];
                d.bulk_out += s.volume;
                d.gas_out += s.gas;
                d.liquid_out += s.liquid;
                c.bulk_in += s.volume;
                c.gas_in += s.gas;
                c.liquid_in += s.liquid;
            }
            for (const auto& [face, flux] : phi)
                if ((flux != 0) != streams.contains(face))
                    throw std::invalid_argument("Face stream coverage differs.");
            for (const auto& key : data.topology->order)
            {
                const auto& c = data.cells.at(key);
                const auto& t = totals[c.gid];
                const long double v = c.native_volume, g = c.phase.gas_volume, l = c.phase.carrier_volume;
                const auto out = t.bulk_out.value(), continuity = t.bulk_in.value() - out;
                if (out > std::min(v, g + l))
                    throw std::invalid_argument("Face transport outgoing-volume CFL exceeded.");
                if (std::abs(continuity) > data.options.reconstruction.volume_tolerance * v)
                    throw std::invalid_argument("Face transport bulk continuity defect.");
                if (t.gas_out.value() > g || t.liquid_out.value() > l)
                    throw std::invalid_argument("Face transport low-order phase export exceeds old inventory.");
                diagnostics.maximum_outgoing_courant = std::max(diagnostics.maximum_outgoing_courant, out / v);
                diagnostics.maximum_continuity_defect =
                    std::max(diagnostics.maximum_continuity_defect, std::abs(continuity));
            }
        });
    return {std::move(tiles), std::move(subscribers), std::move(authority), std::move(streams), std::move(totals),
        diagnostics};
}
template<class ConsumeCell, class ConsumeFace>
FaceAdmission admit_face_ledger(const GeneralVoF::Data& data, FaceLedger& ledger, bool homogeneous,
    ConsumeCell&& consume_cell, ConsumeFace&& consume_face)
{
    const auto comm = data.hierarchy->communicator();
    const auto rank = comm->getRank();
    const auto& tiles = ledger.tiles;
    const auto& streams = ledger.streams;
    auto& totals = ledger.totals;
    FaceAdmission admitted;
    std::array<double, 8> local{}, global{};
    wire::collective(*comm,
        [&]
        {
            std::array<fp::SumFor<mode>, 6> sums;
            std::array<fp::SumFor<mode>, 2> publication;
            for (auto& sum : sums)
                sum = fp::SumFor<mode>(data.options.arithmetic_mode);
            for (auto& sum : publication)
                sum = fp::SumFor<mode>(data.options.arithmetic_mode);
            std::map<std::pair<std::uint64_t, std::uint64_t>, GeneralVoFFiniteTransfer> routes;
            for (const auto& [face, s] : streams)
            {
                if (tiles.at(face)->first_owner == rank)
                {
                    ++ledger.diagnostics.streams;
                    if (s.theta < 1)
                    {
                        ++ledger.diagnostics.limited_streams;
                        ledger.diagnostics.limited_volume += s.volume;
                    }
                    if (s.lower_order)
                    {
                        ++ledger.diagnostics.fallback_streams;
                        ledger.diagnostics.fallback_volume += s.volume;
                    }
                    consume_face(GeneralVoFFiniteFaceContribution{
                        face, s.donor, s.donor_owner, s.direction, measures(s.volume, s.gas, s.liquid)});
                }
                if (s.recipient_owner == rank)
                {
                    auto& route = routes[{s.donor, s.recipient}];
                    route.donor = s.donor;
                    route.recipient = s.recipient;
                    route.donor_owner = s.donor_owner;
                    route.recipient_owner = rank;
                    route.amount.bulk.value += s.volume;
                    route.amount.gas.value += s.gas;
                    route.amount.carrier.value += s.liquid;
                }
            }
            for (const auto& key : data.topology->order)
            {
                const auto& old = data.cells.at(key);
                const auto& c = old;
                const auto& t = totals[c.gid];
                const long double g = old.phase.gas_volume, l = old.phase.carrier_volume;
                const long double rg = g - t.gas_out.value(), rl = l - t.liquid_out.value();
                auto retained = measures(c.native_volume - t.bulk_out.value(), rg, rl);
                admitted.budgets.push_back({c.gid, measures(c.native_volume, g, l),
                    measures(t.bulk_out.value(), t.gas_out.value(), t.liquid_out.value()), retained});
                routes[{c.gid, c.gid}] = {c.gid, c.gid, rank, rank, retained, {}};
            }
            std::map<std::uint64_t, fp::SumFor<mode>> gas, liquid;
            std::map<std::uint64_t, long double> history_g, history_l, pure_fit;
            std::set<std::uint64_t> mixed_receipts;
            for (auto& [pair, route] : routes)
            {
                route.amount = measures(route.amount.bulk.value, route.amount.gas.value, route.amount.carrier.value);
                gas[route.recipient].add(route.amount.gas);
                liquid[route.recipient].add(route.amount.carrier);
                const auto& donor = data.cells.at(data.topology->by_gid.at(route.donor));
                const auto phase_capacity =
                    static_cast<long double>(donor.phase.gas_volume) + donor.phase.carrier_volume;
                // The occupancy discrepancy stays local. Normalize absent-phase
                // uncertainty with physical capacity, including the self residual.
                const auto bulk_fraction = route.donor == route.recipient
                                               ? 1 - totals[donor.gid].bulk_out.value() / phase_capacity
                                               : route.amount.bulk.value / phase_capacity;
                const long double gf = donor.phase.gas_volume > 0 ? route.amount.gas.value / donor.phase.gas_volume
                                                                  : (homogeneous ? bulk_fraction : 0);
                const long double lf = donor.phase.carrier_volume > 0
                                           ? route.amount.carrier.value / donor.phase.carrier_volume
                                           : (homogeneous ? bulk_fraction : 0);
                if (homogeneous)
                {
                    const auto fraction = donor.declaration == GeneralVoFPhaseGeometry::PureGas ? lf : gf;
                    pure_fit[route.recipient] = fp::bound_add(pure_fit[route.recipient],
                        fp::upward(fp::upward(fraction) *
                                   std::max(donor.occupancy.fit_bound, donor.occupancy.admitted_bound)));
                }
                route.inherited_error = {fp::upper_double(fp::upward(fp::upward(gf) * donor.finite_history.gas)),
                    fp::upper_double(fp::upward(fp::upward(lf) * donor.finite_history.carrier))};
                history_g[route.recipient] = fp::bound_add(history_g[route.recipient], route.inherited_error.gas);
                history_l[route.recipient] = fp::bound_add(history_l[route.recipient], route.inherited_error.carrier);
                const auto& recipient = data.cells.at(data.topology->by_gid.at(route.recipient));
                if (route.donor != route.recipient && donor.declaration != recipient.declaration &&
                    ((recipient.declaration == GeneralVoFPhaseGeometry::PureGas && route.amount.carrier.value > 0) ||
                        (recipient.declaration == GeneralVoFPhaseGeometry::PureCarrier && route.amount.gas.value > 0)))
                    mixed_receipts.insert(route.recipient);
                admitted.transfers.push_back(route);
            }
            std::size_t owned_index = 0;
            for (const auto& key : data.topology->order)
            {
                const auto& old = data.cells.at(key);
                auto c = old;
                const auto gv = gas[c.gid].value(), lv = liquid[c.gid].value();
                const auto pg = Math::publish(gv), pl = Math::publish(lv);
                c.phase.gas_volume = pg.first;
                c.phase.carrier_volume = pl.first;
                const auto count = data.topology->cell_tiles.at(key).size();
                const auto u = (16.L + 8.L * count) * std::numeric_limits<long double>::epsilon();
                const auto ge = fp::upward(
                    u * (old.phase.gas_volume + totals[c.gid].gas_in.value() + totals[c.gid].gas_out.value()));
                const auto le = fp::upward(u * (old.phase.carrier_volume + totals[c.gid].liquid_in.value() +
                                                   totals[c.gid].liquid_out.value()));
                c.roundoff = {
                    fp::upper_double(fp::bound_add(pg.second, ge)), fp::upper_double(fp::bound_add(pl.second, le))};
                c.finite_history = {fp::upper_double(fp::bound_add(history_g[c.gid], c.roundoff.gas)),
                    fp::upper_double(fp::bound_add(history_l[c.gid], c.roundoff.carrier))};
                c.retained_interface.reset();
                c.inherited = true;
                // A pure geometric hint survives same-phase trace transport only.
                if (mixed_receipts.contains(c.gid))
                    c.declaration = GeneralVoFPhaseGeometry::Infer;
                if (c.phase.gas_volume == 0)
                    c.declaration = GeneralVoFPhaseGeometry::PureCarrier;
                if (c.phase.carrier_volume == 0)
                    c.declaration = GeneralVoFPhaseGeometry::PureGas;
                const auto geometric_volume =
                    PLIC::ConvexCellGeometry<mode>::from_dynamic(c.finite_geometry)->volume().value;
                SimpleFluid::detail::CompensatedSum<long double> defect_sum;
                defect_sum += pg.first;
                defect_sum += pl.first;
                defect_sum += -geometric_volume;
                const long double defect = defect_sum.value();
                const long double arithmetic = (16 + 4 * data.topology->cell_tiles.at(key).size()) *
                                               std::numeric_limits<long double>::epsilon() * c.native_volume;
                const auto closure = std::abs(totals[c.gid].bulk_in.value() - totals[c.gid].bulk_out.value());
                c.occupancy.difference = double(defect);
                c.occupancy.admitted_bound = fp::upper_double(
                    old.occupancy.admitted_bound + closure + c.roundoff.gas + c.roundoff.carrier + arithmetic);
                c.occupancy.transferred_bound = c.occupancy.admitted_bound;
                if (c.declaration == GeneralVoFPhaseGeometry::Infer)
                    c.occupancy.fit_bound = c.occupancy.fit_residual = 0;
                else if (homogeneous)
                {
                    c.occupancy.fit_residual = c.declaration == GeneralVoFPhaseGeometry::PureGas ? pl.first : pg.first;
                    const auto publication =
                        c.declaration == GeneralVoFPhaseGeometry::PureGas ? c.roundoff.carrier : c.roundoff.gas;
                    c.occupancy.fit_bound = fp::upper_double(fp::bound_add(pure_fit[c.gid], publication));
                }
                validate_cell<mode>(c, data.options);
                ledger.diagnostics.maximum_occupancy_defect =
                    std::max(ledger.diagnostics.maximum_occupancy_defect, std::abs(defect));
                consume_cell(c);
                const auto i = owned_index++;
                sums[0].add(Math::make_value(
                    data.options.arithmetic_mode, static_cast<long double>(pg.first) - data.gas[i], c.roundoff.gas));
                sums[1].add(Math::make_value(data.options.arithmetic_mode,
                    static_cast<long double>(pl.first) - data.carrier[i], c.roundoff.carrier));
                sums[2].add({std::abs(static_cast<long double>(pg.first)) + std::abs(data.gas[i]), 0});
                sums[3].add(Math::make_value(data.options.arithmetic_mode,
                    std::abs(static_cast<long double>(pl.first)) + std::abs(data.carrier[i])));
                sums[4].add(Math::make_value(data.options.arithmetic_mode, data.gas[i]));
                sums[5].add(Math::make_value(data.options.arithmetic_mode, data.carrier[i]));
                publication[0].add(Math::make_value(data.options.arithmetic_mode, c.roundoff.gas));
                publication[1].add(Math::make_value(data.options.arithmetic_mode, c.roundoff.carrier));
            }
            for (unsigned a = 0; a < 2; ++a)
            {
                const auto sum = sums[a].value();
                local[a] = static_cast<double>(sum.value);
                const auto narrowing = [&]
                {
                    if constexpr (mode == ArithmeticMode::Ordinary)
                        return publication[a].value().value;
                    else
                        return sum.absolute_error;
                }();
                local[2 + a] = fp::upper_double(fp::upward(narrowing + std::abs(sum.value - local[a])));
                local[4 + a] = fp::upper_double(sums[2 + a].value().value);
                local[6 + a] = fp::upper_double(sums[4 + a].value().value);
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    wire::collective(*comm,
        [&]
        {
            const auto operations = 8.L * comm->getSize() + 16;
            const auto u = operations * std::numeric_limits<double>::epsilon();
            if (!(u < .25L))
                throw std::invalid_argument("Finite VoF global phase reduction bound is exhausted.");
            for (unsigned phase = 0; phase < 2; ++phase)
            {
                const auto physical = [&]
                {
                    if constexpr (mode == ArithmeticMode::Ordinary)
                        return data.options.reconstruction.volume_tolerance * global[6 + phase];
                    else
                        return 0.L;
                }();
                const auto bound = physical + global[2 + phase] + u / (1 - u) * (global[4 + phase] + global[2 + phase]);
                if (!std::isfinite(global[phase]) || !std::isfinite(bound) || std::abs(global[phase]) > bound)
                    throw std::invalid_argument("Finite VoF global independent phase conservation failed.");
            }
            current(data);
        });
    return admitted;
}
} // namespace
general_vof_detail::AdmittedLowOrderLedger general_vof_detail::admit_low_order_ledger(
    const GeneralVoF::Data& data, std::span<const GeneralVoFFaceVolume> controls, double dt)
{
    const auto comm = data.hierarchy->communicator();
    auto ledger = build_face_ledger(data, controls, dt, {1, 1}, 0);
    AdmittedLowOrderLedger low;
    int local_streams = !ledger.streams.empty(), any_streams = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_streams, &any_streams);
    if (!any_streams)
    {
        wire::collective(*comm,
            [&]
            {
                for (const auto& key : data.topology->order)
                {
                    const auto& c = data.cells.at(key);
                    const auto old = measures(c.native_volume, c.phase.gas_volume, c.phase.carrier_volume);
                    low.transfers.push_back({c.gid, c.gid, c.owner, c.owner, old, c.finite_history});
                    low.budgets.push_back({c.gid, old, measures(0, 0, 0), old});
                    low.endpoint_capacity.emplace(
                        c.gid, std::array<long double, 3>{c.native_volume, c.phase.gas_volume, c.phase.carrier_volume});
                }
                current(data);
            });
    }
    else
    {
        auto admitted =
            admit_face_ledger(data, ledger, true, [](const Cell&) {}, [](const GeneralVoFFiniteFaceContribution&) {});
        low.transfers = std::move(admitted.transfers);
        low.budgets = std::move(admitted.budgets);
        wire::collective(*comm,
            [&]
            {
                // Preserve the original sorted receipt order and compensated
                // endpoint availability, independently of binary64 narrowing.
                std::map<std::uint64_t, std::array<SimpleFluid::detail::CompensatedSum<long double>, 3>> endpoint;
                wire::Budget budget{data.options.byte_budget};
                budget.add(data.topology->order.size() * (sizeof(decltype(endpoint)::value_type) + 4 * sizeof(void*)));
                for (const auto& route : low.transfers)
                {
                    endpoint[route.recipient][0] += route.amount.bulk.value;
                    endpoint[route.recipient][1] += route.amount.gas.value;
                    endpoint[route.recipient][2] += route.amount.carrier.value;
                }
                for (const auto& [gid, sum] : endpoint)
                    low.endpoint_capacity.emplace(gid, std::array{sum[0].value(), sum[1].value(), sum[2].value()});
            });
    }
    wire::collective(*comm,
        [&]
        {
            for (auto& [face, stream] : ledger.streams)
            {
                stream.mass = 0;
                low.streams.push_back(stream);
            }
            low.diagnostics = ledger.diagnostics;
            // No candidate is retained. Account the actual admitted support,
            // including endpoint nodes, and retain the accepted-state gate.
            wire::Budget budget{data.options.byte_budget};
            budget.add(sizeof(low) + low.streams.capacity() * sizeof(Stream) +
                       low.transfers.capacity() * sizeof(GeneralVoFFiniteTransfer) +
                       low.budgets.capacity() * sizeof(GeneralVoFFiniteDonorBudget) +
                       low.endpoint_capacity.size() *
                           (sizeof(decltype(low.endpoint_capacity)::value_type) + 4 * sizeof(void*)));
            retained_budget<mode>(data);
            current(data);
        });
    return low;
}
GeneralVoF::Trial GeneralVoF::plan_face_transport(std::span<const GeneralVoFFaceVolume> controls, double dt,
    std::array<double, 2> density, std::uint64_t time_stage) const
{
    return plan_face_transport_impl(controls, dt, density, time_stage, false);
}
GeneralVoF::Trial GeneralVoF::plan_face_transport_impl(std::span<const GeneralVoFFaceVolume> controls, double dt,
    std::array<double, 2> density, std::uint64_t time_stage, bool homogeneous, std::uint64_t transaction_id) const
{
    const auto& data = *data_;
    const auto comm = data.hierarchy->communicator();
    const auto rank = comm->getRank();
    const auto limit = data.options.byte_budget;
    wire::collective(*comm,
        [&]
        {
            if (!homogeneous && data.options.transport_backend == Backend::SweptPartition)
                throw std::invalid_argument(
                    "Face transport requires Ordinary, finite dt and positive constant densities.");
        });
    auto ledger = build_face_ledger(data, controls, dt, density, time_stage);
    auto& tiles = ledger.tiles;
    auto& subscribers = ledger.subscribers;
    auto& authority = ledger.authority;
    auto& streams = ledger.streams;
    auto& totals = ledger.totals;
    std::vector<OwnerPacket> packets;
    Trial trial;
    trial.source_ = data_;
    trial.transaction_ = homogeneous && transaction_id != 0 ? transaction_id : transaction(comm);
    trial.homogeneous_algebraic_ = homogeneous;
    trial.timestep_ = dt;
    std::shared_ptr<GeneralVoFPreparedTransport> prepared;
    wire::collective(*comm, [&] { prepared = std::make_shared<GeneralVoFPreparedTransport>(); });
    prepared->source_identity = data_;
    prepared->source_control = data.control;
    prepared->transaction = trial.transaction_;
    prepared->mesh_generation = data.hierarchy->partition_generation();
    prepared->time_stage = time_stage;
    prepared->timestep = dt;
    prepared->density = density;
    prepared->backend = data.options.transport_backend;
    if (homogeneous)
    {
        int local_streams = !streams.empty(), any_streams = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_streams, &any_streams);
        if (!any_streams)
        {
            // All controls, scalar agreement and canonical ownership have been
            // authenticated above. Exact zero flux is an identity publication,
            // retaining both phase histories without synthetic trajectories.
            wire::collective(*comm,
                [&]
                {
                    if (data.latest_phase_cleanup.empty())
                        trial.candidate_ = data_;
                    else
                    {
                        auto identity = std::make_shared<Data>(data);
                        identity->latest_phase_cleanup.clear();
                        trial.candidate_ = std::move(identity);
                    }
                    trial.roundoff_.resize(data.topology->order.size());
                    for (const auto& key : data.topology->order)
                    {
                        const auto& cell = data.cells.at(key);
                        const auto old = measures(cell.native_volume, cell.phase.gas_volume, cell.phase.carrier_volume);
                        trial.finite_transfers_.push_back(
                            {cell.gid, cell.gid, cell.owner, cell.owner, old, cell.finite_history});
                        trial.finite_budgets_.push_back({cell.gid, old, measures(0, 0, 0), old});
                    }
                    current(data);
                });
            trial.validate_retained_algebraic_payload();
            return trial;
        }
    }
    if (data.options.transport_backend == Backend::GeometricFace)
    {
        // Low-order admission above is mandatory, before any candidate geometry.
        // Each donor owner fits at most one plane and measures its own incident
        // faces. Only numerical arithmetic degeneracy can select lower order.
        packets.clear();
        wire::collective(*comm,
            [&]
            {
                std::map<std::uint64_t, std::optional<PLIC::PolyhedralInterface>> planes;
                wire::Budget budget{limit};
                for (const auto& [face, original] : streams)
                {
                    if (original.donor_owner != rank)
                        continue;
                    auto s = original;
                    const auto& donor_key = data.topology->by_gid.at(s.donor);
                    const auto& donor = data.cells.at(donor_key);
                    // Pure geometry with independent traces keeps its physical donor
                    // fraction. Neither a hint nor a rounded fraction erases a trace.
                    if (donor.declaration == GeneralVoFPhaseGeometry::Infer && donor.phase.gas_volume > 0 &&
                        donor.phase.carrier_volume > 0)
                    {
                        try
                        {
                            if (!planes.contains(s.donor))
                            {
                                try
                                {
                                    planes.emplace(s.donor, reconstruct_cell<mode>(data, donor_key));
                                }
                                catch (const PLIC::GeometryError& e)
                                {
                                    if (!e.recoverable_trial())
                                        throw;
                                    planes.emplace(s.donor, std::nullopt);
                                }
                            }
                            if (!planes.at(s.donor))
                                throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic,
                                    "Face transport donor plane is numerically unresolved.");
                            auto plane = planes.at(s.donor)->gas;
                            const auto area_vector = tiles.at(face)->geometry->area_vector();
                            const std::array<long double, 3> av{area_vector.x, area_vector.y, area_vector.z};
                            const auto area = std::hypot(av[0], av[1], av[2]);
                            if (!(area > 0) || !std::isfinite(area))
                                throw std::invalid_argument("Face transport has invalid native face metric.");
                            std::vector<std::array<long double, 3>> vertices;
                            for (const auto& point : finite_face_points<mode>(face))
                            {
                                auto local = point.value;
                                for (unsigned a = 0; a < 3; ++a)
                                    local[a] -= plane.anchor[a];
                                vertices.push_back(local);
                            }
                            long double orientation = 0;
                            const auto& tile = *tiles.at(face);
                            for (unsigned a = 0; a < 3; ++a)
                                orientation +=
                                    av[a] * (static_cast<long double>(data.cells.at(*tile.second).centroid[a]) -
                                                data.cells.at(tile.first).centroid[a]);
                            if (orientation == 0 || !std::isfinite(orientation))
                                throw std::invalid_argument("Face transport native orientation is unresolved.");
                            long double shift = 0;
                            for (unsigned a = 0; a < 3; ++a)
                                shift += plane.normal[a] * av[a] / area * (orientation > 0 ? 1 : -1) * s.direction *
                                         s.volume / area;
                            std::vector<long double> times{0, 1};
                            if (shift != 0)
                                for (const auto& p : vertices)
                                {
                                    long double distance = -plane.offset;
                                    for (unsigned a = 0; a < 3; ++a)
                                        distance += plane.normal[a] * p[a];
                                    const auto time = distance / shift;
                                    if (time > 0 && time < 1)
                                        times.push_back(time);
                                }
                            std::sort(times.begin(), times.end());
                            times.erase(std::unique(times.begin(), times.end()), times.end());
                            SimpleFluid::detail::CompensatedSum<long double> covered_gas, covered_liquid;
                            const auto sample = [&](long double time)
                            {
                                auto translated = plane;
                                translated.offset =
                                    static_cast<double>(static_cast<long double>(plane.offset) + shift * time);
                                auto opposite = translated;
                                for (auto& n : opposite.normal)
                                    n = -n;
                                opposite.offset = -opposite.offset;
                                const auto gas_area =
                                    PLIC::ordinary_plane_face_area(vertices, translated, plane.anchor);
                                const auto liquid_area =
                                    PLIC::ordinary_plane_face_area(vertices, opposite, plane.anchor);
                                ledger.diagnostics.plane_measurements += 2;
                                for (const auto& result : {gas_area, liquid_area})
                                    if (result.status != PLIC::PlaneMeasureStatus::Resolved || result.area > area ||
                                        result.area < 0)
                                        throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic,
                                            "Face phase area is numerically unresolved.");
                                return std::array<long double, 2>{gas_area.area, liquid_area.area};
                            };
                            // Polygon area is quadratic between vertex crossings.
                            // Interior Gauss points also handle the discontinuous
                            // face-parallel contact without counting an endpoint twice.
                            for (std::size_t i = 1; i < times.size(); ++i)
                            {
                                const auto half = (times[i] - times[i - 1]) / 2;
                                const auto midpoint = (times[i] + times[i - 1]) / 2;
                                const auto distance = half / std::sqrt(3.L);
                                const auto left = sample(midpoint - distance), right = sample(midpoint + distance);
                                covered_gas += half * (left[0] + right[0]);
                                covered_liquid += half * (left[1] + right[1]);
                            }
                            const auto gas_amount = s.volume * (covered_gas.value() / area);
                            const auto liquid_amount = s.volume * (covered_liquid.value() / area);
                            if (gas_amount < 0 || liquid_amount < 0 || gas_amount > s.volume ||
                                liquid_amount > s.volume || !std::isfinite(gas_amount) ||
                                !std::isfinite(liquid_amount) ||
                                std::abs(gas_amount + liquid_amount - s.volume) >
                                    data.options.reconstruction.volume_tolerance * s.volume)
                                throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic,
                                    "Face phase integral is numerically unresolved.");
                            // The smaller face integral need not be the donor-cell
                            // minority. Measure both sides directly before choosing
                            // the complement, retaining any representable small stream.
                            if (gas_amount <= liquid_amount)
                            {
                                s.gas_high = gas_amount;
                                s.liquid_high = s.volume - gas_amount;
                            }
                            else
                            {
                                s.liquid_high = liquid_amount;
                                s.gas_high = s.volume - liquid_amount;
                            }
                        }
                        catch (const PLIC::GeometryError& e)
                        {
                            if (!e.recoverable_trial())
                                throw;
                            s.lower_order = true;
                            s.gas_high = s.gas_low;
                            s.liquid_high = s.liquid_low;
                        }
                    }
                    wire::Writer w;
                    write_stream(w, s);
                    budget.packet(w.data.size());
                    packets.push_back({tiles.at(face)->first_owner, std::move(w.data)});
                }
            });
        const auto candidates = exchange_owner_packets(packets, comm, limit);
        wire::collective(*comm,
            [&]
            {
                std::set<std::uint64_t> seen;
                for (const auto& p : candidates.packets)
                {
                    wire::Reader r{p.bytes};
                    auto s = read_stream(r);
                    r.done();
                    if (!authority.contains(s.face) || p.peer != authority.at(s.face).donor_owner ||
                        !seen.insert(s.face).second)
                        throw std::invalid_argument("Geometric candidate donor ownership differs.");
                    authority.at(s.face) = s;
                }
                if (seen.size() != authority.size())
                    throw std::invalid_argument("Geometric candidates are incomplete.");
            });
        const auto distribute = [&]
        {
            packets.clear();
            wire::collective(*comm,
                [&]
                {
                    wire::Budget budget{limit};
                    for (const auto& [face, s] : authority)
                        for (const auto peer : subscribers.at(face))
                        {
                            wire::Writer w;
                            write_stream(w, s);
                            budget.packet(w.data.size());
                            packets.push_back({peer, std::move(w.data)});
                        }
                });
            const auto values = exchange_owner_packets(packets, comm, limit);
            wire::collective(*comm,
                [&]
                {
                    streams.clear();
                    for (const auto& p : values.packets)
                    {
                        wire::Reader r{p.bytes};
                        const auto s = read_stream(r);
                        r.done();
                        if (!tiles.contains(s.face) || p.peer != tiles.at(s.face)->first_owner ||
                            !streams.emplace(s.face, s).second)
                            throw std::invalid_argument("Limited stream ownership differs.");
                    }
                });
        };
        distribute();
        // Four independent positive budgets: endpoint gas/liquid positivity,
        // plus safe explicit export of old gas/liquid concentrations/history.
        using Four = std::array<long double, 4>;
        std::map<std::uint64_t, Four> demand, ratio;
        const auto correction = [](const Stream& s)
        { return s.gas_low <= s.liquid_low ? s.gas_high - s.gas_low : s.liquid_low - s.liquid_high; };
        wire::collective(*comm,
            [&]
            {
                for (const auto& [face, s] : streams)
                {
                    const auto a = correction(s);
                    if (a >= 0)
                    {
                        demand[s.donor][0] += a;
                        demand[s.donor][2] += a;
                        demand[s.recipient][1] += a;
                    }
                    else
                    {
                        demand[s.donor][1] -= a;
                        demand[s.donor][3] -= a;
                        demand[s.recipient][0] -= a;
                    }
                }
                for (const auto& key : data.topology->order)
                {
                    const auto& cell = data.cells.at(key);
                    const auto& t = totals[cell.gid];
                    const long double g = cell.phase.gas_volume, l = cell.phase.carrier_volume;
                    const Four available{g - t.gas_out.value() + t.gas_in.value(),
                        l - t.liquid_out.value() + t.liquid_in.value(), g - t.gas_out.value(),
                        l - t.liquid_out.value()};
                    Four r{1, 1, 1, 1};
                    for (unsigned a = 0; a < 4; ++a)
                        if (demand[cell.gid][a] > 0)
                        {
                            r[a] = std::min(1.L, available[a] / demand[cell.gid][a]);
                            if (r[a] > 0 && r[a] < 1)
                                r[a] = std::nextafter(r[a], 0.L);
                        }
                    ratio.emplace(cell.gid, r);
                }
                packets.clear();
                wire::Budget budget{limit};
                for (const auto& [face, s] : streams)
                    for (const auto endpoint : {s.donor, s.recipient})
                        if (ratio.contains(endpoint))
                        {
                            wire::Writer w;
                            w.word(face);
                            w.word(endpoint);
                            for (const auto r : ratio.at(endpoint))
                                finite_real(w, r);
                            budget.packet(w.data.size());
                            packets.push_back({tiles.at(face)->first_owner, std::move(w.data)});
                        }
            });
        const auto ratios = exchange_owner_packets(packets, comm, limit);
        wire::collective(*comm,
            [&]
            {
                std::map<std::pair<std::uint64_t, std::uint64_t>, Four> endpoint_ratios;
                for (const auto& p : ratios.packets)
                {
                    wire::Reader r{p.bytes};
                    const auto face = r.word(), endpoint = r.word();
                    if (!authority.contains(face))
                        throw std::invalid_argument("Limiter face is foreign.");
                    const auto& s = authority.at(face);
                    if ((endpoint != s.donor && endpoint != s.recipient) ||
                        p.peer != (endpoint == s.donor ? s.donor_owner : s.recipient_owner))
                        throw std::invalid_argument("Limiter endpoint ownership differs.");
                    Four values;
                    for (auto& value : values)
                        value = finite_real(r);
                    r.done();
                    if (!endpoint_ratios.emplace(std::pair{face, endpoint}, values).second)
                        throw std::invalid_argument("Limiter endpoint duplicated.");
                }
                for (auto& [face, s] : authority)
                {
                    const auto d = endpoint_ratios.at({face, s.donor}), c = endpoint_ratios.at({face, s.recipient});
                    const auto a = correction(s);
                    s.theta = a >= 0 ? std::min({d[0], d[2], c[1]}) : std::min({d[1], d[3], c[0]});
                    // Convex endpoint evaluation preserves a resolved microscopic
                    // high-order minority even when high-low rounds to -low.
                    const auto blend = [&](long double low_value, long double high_value)
                    {
                        if (s.theta == 0)
                            return low_value;
                        if (s.theta == 1)
                            return high_value;
                        return (1 - s.theta) * low_value + s.theta * high_value;
                    };
                    s.gas = blend(s.gas_low, s.gas_high);
                    s.liquid = blend(s.liquid_low, s.liquid_high);
                    if (s.gas <= s.liquid)
                        s.liquid = s.volume - s.gas;
                    else
                        s.gas = s.volume - s.liquid;
                    if (s.gas < 0 || s.liquid < 0 || s.gas > s.volume || s.liquid > s.volume)
                        throw std::invalid_argument("Limited phase stream is inadmissible.");
                    s.mass = density[0] * s.gas + density[1] * s.liquid;
                }
            });
        distribute();
        wire::collective(*comm,
            [&]
            {
                totals.clear();
                for (const auto& [face, s] : streams)
                {
                    auto& d = totals[s.donor];
                    auto& c = totals[s.recipient];
                    d.bulk_out += s.volume;
                    d.gas_out += s.gas;
                    d.liquid_out += s.liquid;
                    c.bulk_in += s.volume;
                    c.gas_in += s.gas;
                    c.liquid_in += s.liquid;
                }
                for (const auto& key : data.topology->order)
                {
                    const auto& c = data.cells.at(key);
                    const auto& t = totals[c.gid];
                    if (t.gas_out.value() > c.phase.gas_volume || t.liquid_out.value() > c.phase.carrier_volume)
                        throw std::invalid_argument("Limited phase export exceeds old donor inventory.");
                }
            });
    }
    // Public publication alone copies Data and imports the candidate ghosts.
    std::shared_ptr<Data> candidate;
    wire::collective(*comm,
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
        });
    auto admitted = admit_face_ledger(
        data, ledger, homogeneous,
        [&](const Cell& c)
        {
            const auto& old = data.cells.at(c.key);
            candidate->cells.at(c.key) = c;
            prepared->old_mass.push_back(density[0] * static_cast<long double>(old.phase.gas_volume) +
                                         density[1] * static_cast<long double>(old.phase.carrier_volume));
            prepared->new_mass.push_back(density[0] * static_cast<long double>(c.phase.gas_volume) +
                                         density[1] * static_cast<long double>(c.phase.carrier_volume));
            candidate->gas.push_back(c.phase.gas_volume);
            candidate->carrier.push_back(c.phase.carrier_volume);
            candidate->phases.push_back({c.native_volume, c.phase.gas_volume, c.phase.carrier_volume});
            candidate->roundoff.push_back(c.roundoff);
            candidate->phase_history.push_back(c.finite_history);
            candidate->occupancy.push_back(c.occupancy);
        },
        [&](const GeneralVoFFiniteFaceContribution& face) { trial.finite_faces_.push_back(face); });
    wire::collective(*comm,
        [&]
        {
            for (const auto& [face, stream] : streams)
                prepared->streams.push_back(stream);
            prepared->diagnostics = ledger.diagnostics;
            trial.finite_transfers_ = std::move(admitted.transfers);
            trial.finite_budgets_ = std::move(admitted.budgets);
            trial.roundoff_ = candidate->roundoff;
        });
    apply_phase_cleanup(*candidate);
    import_phases<mode>(*candidate);
    wire::collective(*comm,
        [&]
        {
            current(data);
            retained_budget<mode>(*candidate);
        });
    trial.candidate_ = std::move(candidate);
    if (homogeneous)
    {
        trial.unbound_streams_ = std::move(prepared->streams);
        for (auto& stream : trial.unbound_streams_)
            stream.mass = 0;
        trial.unbound_diagnostics_ = prepared->diagnostics;
        trial.validate_retained_algebraic_payload();
    }
    else
    {
        trial.prepared_ = std::move(prepared);
        trial.prepared_ = trial.prepare_transport(density, time_stage);
    }
    return trial;
}
} // namespace SimpleFluid::VoF
