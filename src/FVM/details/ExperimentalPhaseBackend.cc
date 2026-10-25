/** @file ExperimentalPhaseBackend.cc
 * @brief Bounded immutable experimental exact inventory/provenance publication.
 */
#include "FVM/details/ExperimentalPhaseBackend.hh"
#include "FVM/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace SimpleFluid::VoF::detail
{
namespace
{
using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
M absolute(const M& value)
{
    return value.sign() < 0 ? value.scaled(-1) : value;
}
void capacity(const ExperimentalPhaseInventory& value, std::size_t limit)
{
    std::size_t used = sizeof(value);
    if (used > limit)
        throw std::invalid_argument("Experimental exact inventory exceeds its byte budget.");
    for (const auto* item : {&value.phases.bulk, &value.phases.gas, &value.phases.carrier, &value.error.difference,
             &value.error.admitted_bound, &value.error.native_conversion, &value.error.phase_input_bound,
             &value.error.fit_residual, &value.error.fit_bound, &value.error.transferred_bound,
             &value.error.fit_allocation_bound})
    {
        const auto bytes = item->storage_bytes();
        if (bytes > limit - used)
            throw std::invalid_argument("Experimental exact inventory exceeds its byte budget.");
        used += bytes;
    }
}
} // namespace
void validate_experimental(const ExperimentalPhaseInventory& value,
    const SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>& geometry, double native_volume,
    std::size_t byte_budget)
{
    capacity(value, byte_budget);
    const auto& p = value.phases;
    const auto& e = value.error;
    const auto defect = p.gas + p.carrier - p.bulk;
    if (!std::isfinite(native_volume) || native_volume <= 0 || p.bulk.sign() <= 0 ||
        p.bulk.compare(M::volume(geometry)) || p.gas.sign() < 0 || p.carrier.sign() < 0 ||
        defect.compare(e.difference) || absolute(defect).compare(e.admitted_bound) > 0 ||
        (M::from_value(native_volume) - p.bulk).compare(e.native_conversion))
        throw std::invalid_argument("Experimental exact phase/provenance is invalid.");
    for (const auto* bound :
        {&e.admitted_bound, &e.phase_input_bound, &e.fit_bound, &e.transferred_bound, &e.fit_allocation_bound})
        if (bound->sign() < 0)
            throw std::invalid_argument("Experimental exact phase bound is negative.");
}
std::shared_ptr<const ExperimentalPhaseInventory> experimental_initial(
    const SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>& geometry, double native_volume,
    double gas, double carrier, std::size_t byte_budget)
{
    ExperimentalPhaseInventory value;
    value.phases = {M::volume(geometry), M::from_value(gas), M::from_value(carrier)};
    const auto native = M::from_value(native_volume);
    auto& e = value.error;
    e.difference = value.phases.gas + value.phases.carrier - value.phases.bulk;
    e.native_conversion = native - value.phases.bulk;
    e.phase_input_bound = native.scaled(8 * std::numeric_limits<double>::epsilon());
    if (absolute(e.native_conversion).compare(e.phase_input_bound) > 0 ||
        absolute(e.difference - e.native_conversion).compare(e.phase_input_bound) > 0)
        throw std::invalid_argument("Experimental exact bootstrap input error exceeds its bound.");
    e.admitted_bound = absolute(e.native_conversion) + e.phase_input_bound;
    validate_experimental(value, geometry, native_volume, byte_budget);
    return std::make_shared<const ExperimentalPhaseInventory>(std::move(value));
}
std::shared_ptr<const ExperimentalPhaseInventory> experimental_advance(const ExperimentalPhaseInventory& source,
    SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> phases,
    const SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>& geometry, double native_volume,
    std::size_t byte_budget)
{
    // Keep original exact source provenance; a physical step does not refresh
    // its allowance or silently project an independent inventory onto V-G.
    ExperimentalPhaseInventory value{std::move(phases), source.error};
    validate_experimental(value, geometry, native_volume, byte_budget);
    return std::make_shared<const ExperimentalPhaseInventory>(std::move(value));
}
ExperimentalPhaseView experimental_view(const ExperimentalPhaseInventory& state, double native_volume)
{
    const auto gas = round_phase_value(state.phases.gas), carrier = round_phase_value(state.phases.carrier);
    if ((state.phases.gas.sign() > 0 && gas.value == 0) || (state.phases.carrier.sign() > 0 && carrier.value == 0))
        throw std::runtime_error("Experimental exact positive final phase has no representable finite view.");
    ExperimentalPhaseView out;
    out.phases = {native_volume, gas.value, carrier.value};
    out.roundoff = {gas.absolute_error, carrier.absolute_error};
    const auto conversion = M::from_value(gas.absolute_error) + M::from_value(carrier.absolute_error);
    const auto defect = M::from_value(gas.value) + M::from_value(carrier.value) - state.phases.bulk;
    const auto diagnostic = round_phase_residual(defect);
    const auto native = round_phase_residual(M::from_value(native_volume) - state.phases.bulk);
    const auto fit = round_phase_residual(state.error.fit_residual);
    out.error.difference = diagnostic.value;
    out.error.native_conversion = native.value;
    out.error.fit_residual = fit.value;
    out.error.admitted_bound =
        phase_upper_bound(state.error.admitted_bound + conversion + M::from_value(diagnostic.absolute_error));
    out.error.phase_input_bound =
        phase_upper_bound(state.error.phase_input_bound + conversion + M::from_value(native.absolute_error) +
                          M::from_value(diagnostic.absolute_error));
    out.error.fit_bound = phase_upper_bound(state.error.fit_bound + conversion + M::from_value(fit.absolute_error));
    out.error.transferred_bound = phase_upper_bound(state.error.transferred_bound);
    out.error.fit_allocation_bound =
        phase_upper_bound(state.error.fit_allocation_bound + conversion + M::from_value(fit.absolute_error));
    return out;
}
std::vector<char> encode_experimental(const ExperimentalPhaseInventory& value, std::size_t byte_budget)
{
    capacity(value, byte_budget);
    Meshes::AMR::adaptive_detail::Writer w;
    Meshes::AMR::adaptive_detail::Budget bytes{byte_budget};
    bytes.add(16);
    w.word(0x5346455850493731ULL);
    w.word(1);
    for (const auto* item : {&value.phases.bulk, &value.phases.gas, &value.phases.carrier, &value.error.difference,
             &value.error.admitted_bound, &value.error.native_conversion, &value.error.phase_input_bound,
             &value.error.fit_residual, &value.error.fit_bound, &value.error.transferred_bound,
             &value.error.fit_allocation_bound})
    {
        auto encoded = item->encode(byte_budget);
        bytes.add(8);
        bytes.add(encoded.size());
        w.bytes(encoded);
    }
    return std::move(w.data);
}
std::shared_ptr<const ExperimentalPhaseInventory> decode_experimental(
    std::span<const char> packet, std::size_t byte_budget)
{
    if (packet.size() > byte_budget)
        throw std::invalid_argument("Experimental exact packet exceeds budget.");
    Meshes::AMR::adaptive_detail::Reader r{packet};
    if (r.word() != 0x5346455850493731ULL || r.word() != 1)
        throw std::invalid_argument("Unknown experimental exact packet format.");
    ExperimentalPhaseInventory value;
    for (auto* item : {&value.phases.bulk, &value.phases.gas, &value.phases.carrier, &value.error.difference,
             &value.error.admitted_bound, &value.error.native_conversion, &value.error.phase_input_bound,
             &value.error.fit_residual, &value.error.fit_bound, &value.error.transferred_bound,
             &value.error.fit_allocation_bound})
        *item = M::decode(r.bytes(), byte_budget);
    r.done();
    capacity(value, byte_budget);
    return std::make_shared<const ExperimentalPhaseInventory>(std::move(value));
}

} // namespace SimpleFluid::VoF::detail
