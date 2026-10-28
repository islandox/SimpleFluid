/** @file PhaseCleanup.cc
 * @brief Owner-only transactional physical relabeling and persistent global budget.
 */
#include "FVM/details/GeneralVoF/PhaseCleanup.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
namespace
{
using State = GeneralVoFPhaseCleanupState;
long double add_upper(long double a, long double b)
{
    if (b == 0)
        return a;
    auto value = a + b;
    const auto z = value - a;
    const auto residual = (a - (value - z)) + (b - z);
    if (residual > 0)
        value = std::nextafter(value, std::numeric_limits<long double>::infinity());
    if (!std::isfinite(value))
        throw std::invalid_argument("General VoF phase cleanup ledger overflow.");
    return value;
}
void validate(const State& state, const GeneralVoFPhaseCleanupOptions* options = nullptr)
{
    if ((!state.policy_bound && state.cleaned_cells != 0) ||
        (state.policy_bound && (!std::isfinite(state.policy.relative_cutoff) || state.policy.relative_cutoff < 0 ||
                                   state.policy.relative_cutoff >= .5)) ||
        (options && state.policy_bound &&
            (state.policy.relative_cutoff != options->relative_cutoff ||
                state.policy.maximum_absolute_volume != options->maximum_absolute_volume)))
        throw std::invalid_argument("General VoF phase cleanup ledger policy differs from its original controls.");
    for (double value : state.policy.maximum_absolute_volume)
        if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument("General VoF phase cleanup checkpoint policy is malformed.");
    for (unsigned p = 0; p < 2; ++p)
        if (!std::isfinite(state.signed_volume[p]) || !std::isfinite(state.absolute_volume[p]) ||
            !std::isfinite(state.largest_absolute_volume[p]) || state.absolute_volume[p] < 0 ||
            state.largest_absolute_volume[p] < 0 || std::abs(state.signed_volume[p]) > state.absolute_volume[p] ||
            state.largest_absolute_volume[p] > state.absolute_volume[p] ||
            (options && state.absolute_volume[p] > options->maximum_absolute_volume[p]) ||
            (state.policy_bound && state.absolute_volume[p] > state.policy.maximum_absolute_volume[p]) ||
            (!state.cleaned_cells && (state.signed_volume[p] != 0 || state.absolute_volume[p] != 0 ||
                                         state.largest_absolute_volume[p] != 0)))
            throw std::invalid_argument("General VoF phase cleanup ledger is malformed or exceeds its budget.");
}
void write(wire::Writer& w, const State& state)
{
    w.word(1);
    w.word(state.cleaned_cells);
    w.word(state.policy_bound);
    real(w, state.policy.relative_cutoff);
    for (double value : state.policy.maximum_absolute_volume)
        real(w, value);
    for (unsigned p = 0; p < 2; ++p)
        for (auto value : {state.signed_volume[p], state.absolute_volume[p], state.largest_absolute_volume[p]})
            finite_real(w, value);
}
State read(wire::Reader& r)
{
    if (r.word() != 1)
        throw std::invalid_argument("General VoF phase cleanup checkpoint version differs.");
    State state;
    state.cleaned_cells = r.word();
    const auto bound = r.word();
    if (bound > 1)
        throw std::invalid_argument("General VoF phase cleanup checkpoint policy flag is malformed.");
    state.policy_bound = bound;
    state.policy.relative_cutoff = real(r);
    for (auto& value : state.policy.maximum_absolute_volume)
        value = real(r);
    for (unsigned p = 0; p < 2; ++p)
    {
        state.signed_volume[p] = finite_real(r);
        state.absolute_volume[p] = finite_real(r);
        state.largest_absolute_volume[p] = finite_real(r);
    }
    validate(state);
    return state;
}
} // namespace
void validate_phase_cleanup_options(const GeneralVoFOptions& options)
{
    const auto& cleanup = options.phase_cleanup;
    if (!std::isfinite(cleanup.relative_cutoff) || cleanup.relative_cutoff < 0 || cleanup.relative_cutoff >= .5)
        throw std::invalid_argument("General VoF phase cleanup cutoff must be finite in [0,0.5).");
    for (double value : cleanup.maximum_absolute_volume)
        if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument("General VoF phase cleanup absolute budgets must be finite and nonnegative.");
    if (cleanup.relative_cutoff > 0 && (options.arithmetic_mode != ArithmeticMode::Ordinary ||
                                           options.transport_backend != GeneralVoFTransportBackend::SweptPartition))
        throw std::invalid_argument("General VoF phase cleanup supports Ordinary SweptPartition only.");
}
void apply_phase_cleanup(GeneralVoF::Data& data)
{
    const auto comm = data.hierarchy->communicator();
    const auto& options = data.options.phase_cleanup;
    std::vector<GeneralVoFPhaseCleanupDelta> proposed;
    std::map<Key, Cell> replacements;
    State batch;
    batch.policy = options;
    batch.policy_bound = true;
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            validate_phase_cleanup_options(data.options);
            validate(data.phase_cleanup, &options);
            wire::Budget budget{data.options.byte_budget};
            if (options.relative_cutoff > 0)
                for (const auto& [key, source] : data.cells)
                    if (source.owner == comm->getRank())
                    {
                        const auto gas = source.phase.gas_volume, carrier = source.phase.carrier_volume;
                        const auto actual =
                            PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_dynamic(source.finite_geometry)
                                ->volume()
                                .value;
                        const unsigned minority = gas <= carrier ? 0 : 1;
                        const double small = minority == 0 ? gas : carrier;
                        if (!(small > 0) || static_cast<long double>(small) > options.relative_cutoff * actual)
                            continue;
                        Cell cell = source;
                        const double dominant = minority == 0 ? carrier : gas;
                        const double merged = dominant + small;
                        if (!std::isfinite(merged))
                            throw std::invalid_argument("General VoF phase cleanup endpoint overflow.");
                        cell.phase.gas_volume = minority == 0 ? 0 : merged;
                        cell.phase.carrier_volume = minority == 0 ? merged : 0;
                        GeneralVoFPhaseCleanupDelta delta{key, source.gid, {gas, carrier},
                            {cell.phase.gas_volume, cell.phase.carrier_volume},
                            {static_cast<long double>(cell.phase.gas_volume) - gas,
                                static_cast<long double>(cell.phase.carrier_volume) - carrier}};
                        // Keep cleanup separate from all arithmetic/history allowances.
                        ::SimpleFluid::detail::CompensatedSum<long double> difference;
                        difference += cell.phase.gas_volume;
                        difference += cell.phase.carrier_volume;
                        difference += -actual;
                        cell.occupancy.difference = static_cast<double>(difference.value());
                        cell.declaration =
                            minority == 0 ? GeneralVoFPhaseGeometry::PureCarrier : GeneralVoFPhaseGeometry::PureGas;
                        cell.retained_interface.reset();
                        cell.requires_interface = false;
                        cell.occupancy.fit_residual = 0;
                        validate_cell<ArithmeticMode::Ordinary>(cell, data.options);
                        budget.add(sizeof(cell) + sizeof(Key) + sizeof(delta) + 8 * sizeof(void*));
                        proposed.push_back(delta);
                        const auto replacement = replacements.emplace(key, std::move(cell)).first;
                        phase_cleanup_key_storage(proposed.back().key, budget);
                        phase_cleanup_key_storage(replacement->first, budget);
                        phase_cleanup_key_storage(replacement->second.key, budget);
                        ++batch.cleaned_cells;
                        for (unsigned p = 0; p < 2; ++p)
                        {
                            batch.signed_volume[p] += delta.volume[p];
                            batch.absolute_volume[p] = add_upper(batch.absolute_volume[p], std::abs(delta.volume[p]));
                            batch.largest_absolute_volume[p] =
                                std::max(batch.largest_absolute_volume[p], std::abs(delta.volume[p]));
                        }
                    }
            wire::Writer writer;
            write(writer, data.phase_cleanup);
            write(writer, batch);
            budget.packet(writer.data.size());
            packets.push_back({0, std::move(writer.data)});
        });
    auto incoming = exchange_owner_packets(packets, comm, data.options.byte_budget);
    State result = data.phase_cleanup;
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            if (comm->getRank() != 0)
                return;
            if (incoming.packets.size() != std::size_t(comm->getSize()))
                throw std::invalid_argument("General VoF phase cleanup batch has missing owners.");
            wire::Writer reference;
            write(reference, data.phase_cleanup);
            std::set<int> owners;
            for (const auto& packet : incoming.packets)
            {
                if (!owners.insert(packet.peer).second)
                    throw std::invalid_argument("General VoF phase cleanup batch repeats an owner.");
                wire::Reader reader{packet.bytes};
                const auto previous = read(reader);
                wire::Writer previous_bytes;
                write(previous_bytes, previous);
                if (previous_bytes.data != reference.data)
                    throw std::invalid_argument("General VoF phase cleanup replicated ledger differs across owners.");
                const auto increment = read(reader);
                reader.done();
                if (increment.cleaned_cells > std::numeric_limits<std::uint64_t>::max() - result.cleaned_cells)
                    throw std::invalid_argument("General VoF phase cleanup cell count overflow.");
                result.cleaned_cells += increment.cleaned_cells;
                for (unsigned p = 0; p < 2; ++p)
                {
                    result.signed_volume[p] += increment.signed_volume[p];
                    result.absolute_volume[p] = add_upper(result.absolute_volume[p], increment.absolute_volume[p]);
                    result.largest_absolute_volume[p] =
                        std::max(result.largest_absolute_volume[p], increment.largest_absolute_volume[p]);
                }
            }
            result.policy = options;
            result.policy_bound = true;
            validate(result, &options);
            wire::Writer writer;
            write(writer, result);
            wire::Budget budget{data.options.byte_budget};
            for (int rank = 0; rank < comm->getSize(); ++rank)
            {
                budget.packet(writer.data.size());
                replies.push_back({rank, writer.data});
            }
        });
    const auto returned = exchange_owner_packets(replies, comm, data.options.byte_budget);
    wire::collective(*comm,
        [&]
        {
            if (returned.packets.size() != 1 || returned.packets.front().peer != 0)
                throw std::invalid_argument("General VoF phase cleanup global admission is missing.");
            wire::Reader reader{returned.packets.front().bytes};
            result = read(reader);
            reader.done();
            validate(result, &options);
            for (auto& [key, cell] : replacements)
                data.cells.at(key) = std::move(cell);
            data.phase_cleanup = result;
            data.latest_phase_cleanup = std::move(proposed);
            // Construction has no completed phase arrays yet. Transport/remap do.
            for (std::size_t i = 0; i < data.topology->order.size(); ++i)
            {
                const auto& cell = data.cells.at(data.topology->order[i]);
                if (i < data.gas.size())
                    data.gas[i] = cell.phase.gas_volume;
                if (i < data.carrier.size())
                    data.carrier[i] = cell.phase.carrier_volume;
                if (i < data.phases.size())
                    data.phases[i] = {cell.native_volume, cell.phase.gas_volume, cell.phase.carrier_volume};
                if (i < data.occupancy.size())
                    data.occupancy[i] = cell.occupancy;
            }
        });
}
} // namespace SimpleFluid::VoF::general_vof_detail
namespace SimpleFluid::VoF
{
const GeneralVoFPhaseCleanupState& GeneralVoF::phase_cleanup_state() const noexcept
{
    return data_->phase_cleanup;
}
std::span<const GeneralVoFPhaseCleanupDelta> GeneralVoF::latest_phase_cleanup() const noexcept
{
    return data_->latest_phase_cleanup;
}
std::vector<char> GeneralVoF::phase_cleanup_checkpoint() const
{
    general_vof_detail::wire::Writer writer;
    general_vof_detail::write(writer, data_->phase_cleanup);
    return std::move(writer.data);
}
GeneralVoFPhaseCleanupState GeneralVoF::decode_phase_cleanup_checkpoint(std::span<const char> bytes)
{
    general_vof_detail::wire::Reader reader{bytes};
    auto state = general_vof_detail::read(reader);
    reader.done();
    return state;
}
} // namespace SimpleFluid::VoF
