/** @file ExactPhaseTransfer.cc
 * @brief Exact overlap certificates with finite phase publication and transient weighted products.
 */
#include "FVM/ExactPhaseTransfer.hh"
#include "FVM/details/DistributedAMROverlap.hh"
#include "FVM/details/ExactPhaseLabelMap.hh"
#include "FVM/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
namespace SimpleFluid::VoF
{
namespace
{
namespace AMR = Meshes::AMR;
namespace wire = AMR::adaptive_detail;
using H = AMR::DistributedAdaptiveHierarchy;
using Key = AMR::AdaptiveCellKey;
using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
M abs(const M& value)
{
    return value.sign() < 0 ? value.scaled(-1) : value;
}
M product(const M& a, const M& b, std::size_t limit)
{
    return detail::exact_phase_product(a, b, limit);
}

// Transaction-private exact phase/error work only. No instance is retained by
// Source or Result. Their phase/error fields are finite; optional weighted
// products have only the separately documented remap-transaction lifetime.
struct WorkingMass
{
    M gas, carrier;
};
struct WorkingError
{
    M difference, admitted_bound, native_conversion, phase_input_bound;
    M fit_residual, fit_bound, transferred_bound, fit_allocation_bound;
};
struct WorkingCapsule
{
    Key key;
    std::uint64_t revision = 0;
    PhaseInventoryPolicy policy = PhaseInventoryPolicy::Finite;
    SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> phases;
    PLIC::PolyhedralInterface geometry;
    bool geometry_ready = false;
    WorkingError error;
    WorkingMass accumulated_roundoff;
    std::optional<WorkingMass> phase_mass;
    std::vector<M> amounts;
};
WorkingCapsule lift(const ExactPhaseTransferCapsule& c, const M& bulk)
{
    WorkingCapsule w;
    w.key = c.key;
    w.revision = c.revision;
    w.policy = c.policy;
    w.geometry = c.geometry;
    w.geometry_ready = c.geometry_ready;
    w.accumulated_roundoff = {M::from_value(c.accumulated_roundoff.gas), M::from_value(c.accumulated_roundoff.carrier)};
    w.phases = {bulk, M::from_value(c.phases.gas), M::from_value(c.phases.carrier)};
    w.error = {w.phases.gas + w.phases.carrier - bulk, M::from_value(c.error.admitted_bound),
        M::from_value(c.phases.bulk) - bulk, M::from_value(c.error.phase_input_bound),
        M::from_value(c.error.fit_residual), M::from_value(c.error.fit_bound), M::from_value(c.error.transferred_bound),
        M::from_value(c.error.fit_allocation_bound)};
    if (c.policy == PhaseInventoryPolicy::ExperimentalExact)
    {
        if (!c.experimental_exact)
            throw std::invalid_argument("Missing experimental transfer authority.");
        const auto& exact = *c.experimental_exact;
        w.phases = exact.phases;
        const auto& e = exact.error;
        w.error = {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound, e.fit_residual,
            e.fit_bound, e.transferred_bound, e.fit_allocation_bound};
    }
    if (c.phase_mass)
        w.phase_mass = WorkingMass{c.phase_mass->gas, c.phase_mass->carrier};
    w.amounts = c.amounts;
    return w;
}
struct Capacity
{
    wire::Budget budget;
    std::map<const void*, M> measures;
    explicit Capacity(std::size_t n) : budget{n} {}
    void measure(const M& value)
    {
        if (!measures.contains(value.storage_identity()))
        {
            // Hold the handle while deduplicating: newly lifted finite sources
            // otherwise die between columns and allocator addresses can repeat.
            budget.add(value.storage_bytes() + sizeof(std::pair<const void* const, M>) + 4 * sizeof(void*));
            measures.emplace(value.storage_identity(), value);
        }
    }
    void capsule(const WorkingCapsule& c)
    {
        budget.add(sizeof(c) + AMR::encode_adaptive_key(c.key).size() + c.amounts.capacity() * sizeof(M));
        measure(c.phases.bulk);
        measure(c.phases.gas);
        measure(c.phases.carrier);
        measure(c.accumulated_roundoff.gas);
        measure(c.accumulated_roundoff.carrier);
        for (const auto* q :
            {&c.error.difference, &c.error.admitted_bound, &c.error.native_conversion, &c.error.phase_input_bound,
                &c.error.fit_residual, &c.error.fit_bound, &c.error.transferred_bound, &c.error.fit_allocation_bound})
            measure(*q);
        if (c.phase_mass)
        {
            measure(c.phase_mass->gas);
            measure(c.phase_mass->carrier);
        }
        for (const auto& q : c.amounts)
            measure(q);
    }
    void capsule(const ExactPhaseTransferCapsule& c)
    {
        if (c.experimental_exact)
        {
            budget.add(sizeof(ExperimentalPhaseInventory));
            const auto& x = *c.experimental_exact;
            measure(x.phases.bulk);
            measure(x.phases.gas);
            measure(x.phases.carrier);
            for (const auto* q : {&x.error.difference, &x.error.admitted_bound, &x.error.native_conversion,
                     &x.error.phase_input_bound, &x.error.fit_residual, &x.error.fit_bound, &x.error.transferred_bound,
                     &x.error.fit_allocation_bound})
                measure(*q);
        }
        budget.add(sizeof(c) + AMR::encode_adaptive_key(c.key).size() + c.amounts.capacity() * sizeof(M));
        if (c.phase_mass)
        {
            measure(c.phase_mass->gas);
            measure(c.phase_mass->carrier);
        }
        for (const auto& amount : c.amounts)
            measure(amount);
    }
};
void real(wire::Writer& w, double value)
{
    w.word(std::bit_cast<std::uint64_t>(value));
}
double real(wire::Reader& r)
{
    const auto x = std::bit_cast<double>(r.word());
    if (!std::isfinite(x))
        throw std::invalid_argument("Nonfinite exact transfer plane metadata.");
    return x;
}
void wide(wire::Writer& w, long double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite exact transfer diagnostic.");
    w.bytes(std::span(reinterpret_cast<const char*>(&value), sizeof(value)));
}
long double wide(wire::Reader& r)
{
    auto bytes = r.bytes();
    if (bytes.size() != sizeof(long double))
        throw std::invalid_argument("Exact transfer diagnostic format mismatch.");
    long double value;
    std::memcpy(&value, bytes.data(), sizeof(value));
    if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite exact transfer diagnostic.");
    return value;
}
void plane(wire::Writer& w, const PLIC::PolyhedralInterface& p)
{
    for (auto x : p.gas.normal)
        real(w, x);
    for (auto x : p.gas.anchor)
        real(w, x);
    real(w, p.gas.offset);
    for (auto x : {p.target_volume_residual, p.stencil_volume_residual, p.stencil_condition, p.orientation_step})
        real(w, x);
    wide(w, p.target_volume_difference.value);
    wide(w, p.target_volume_difference.absolute_error);
    w.word(p.target_volume_difference.exact_nonzero);
    w.word(p.fit_iterations);
    w.word(p.fit_converged);
    w.word(p.pure_gas);
    w.word(p.pure_carrier);
}
bool boolean(wire::Reader& r)
{
    auto v = r.word();
    if (v > 1)
        throw std::invalid_argument("Invalid exact transfer boolean.");
    return v;
}
PLIC::PolyhedralInterface plane(wire::Reader& r)
{
    PLIC::PolyhedralInterface p;
    for (auto& x : p.gas.normal)
        x = real(r);
    for (auto& x : p.gas.anchor)
        x = real(r);
    p.gas.offset = real(r);
    p.target_volume_residual = real(r);
    p.stencil_volume_residual = real(r);
    p.stencil_condition = real(r);
    p.orientation_step = real(r);
    p.target_volume_difference.value = wide(r);
    p.target_volume_difference.absolute_error = wide(r);
    if (p.target_volume_difference.absolute_error < 0)
        throw std::invalid_argument("Negative diagnostic enclosure.");
    p.target_volume_difference.exact_nonzero = boolean(r);
    auto count = r.word();
    if (count > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Exact transfer iteration count overflow.");
    p.fit_iterations = unsigned(count);
    p.fit_converged = boolean(r);
    p.pure_gas = boolean(r);
    p.pure_carrier = boolean(r);
    return p;
}
AMR::AdaptiveBytes encode_work(const WorkingCapsule& c, std::size_t limit)
{
    wire::Budget budget{limit};
    const auto key_bytes = AMR::encode_adaptive_key(c.key);
    budget.add(272 + key_bytes.size());
    wire::Writer w;
    w.word(0x5346455054523731ULL);
    w.word(3);
    w.word(std::uint64_t(c.policy));
    w.word(sizeof(long double));
    w.word(std::numeric_limits<long double>::digits);
    w.word(std::numeric_limits<long double>::max_exponent);
    w.bytes(AMR::encode_adaptive_key(c.key));
    w.word(c.revision);
    w.word(c.geometry_ready);
    plane(w, c.geometry);
    auto put = [&](const M& q)
    {
        auto bytes = q.encode(limit);
        budget.add(bytes.size() + 8);
        w.bytes(bytes);
    };
    put(c.phases.bulk);
    put(c.phases.gas);
    put(c.phases.carrier);
    put(c.accumulated_roundoff.gas);
    put(c.accumulated_roundoff.carrier);
    for (const auto* q :
        {&c.error.difference, &c.error.admitted_bound, &c.error.native_conversion, &c.error.phase_input_bound,
            &c.error.fit_residual, &c.error.fit_bound, &c.error.transferred_bound, &c.error.fit_allocation_bound})
        put(*q);
    w.word(bool(c.phase_mass));
    if (c.phase_mass)
    {
        put(c.phase_mass->gas);
        put(c.phase_mass->carrier);
    }
    w.word(c.amounts.size());
    for (const auto& q : c.amounts)
        put(q);
    if (w.data.size() > limit)
        throw std::invalid_argument("Exact phase capsule exceeds byte budget.");
    return std::move(w.data);
}
WorkingCapsule decode_work(std::span<const char> bytes, std::size_t limit)
{
    if (bytes.size() > limit)
        throw std::invalid_argument("Exact phase capsule exceeds byte budget.");
    wire::Reader r{bytes};
    if (r.word() != 0x5346455054523731ULL || r.word() != 3)
        throw std::invalid_argument("Unknown policy-pinned working capsule format.");
    const auto policy = r.word();
    if (policy > 1)
        throw std::invalid_argument("Invalid working policy.");
    if (r.word() != sizeof(long double) || r.word() != std::numeric_limits<long double>::digits ||
        r.word() != std::numeric_limits<long double>::max_exponent)
        throw std::invalid_argument("Unknown exact phase capsule format.");
    WorkingCapsule c;
    c.policy = PhaseInventoryPolicy(policy);
    c.key = AMR::decode_adaptive_key(r.bytes());
    c.revision = r.word();
    c.geometry_ready = boolean(r);
    c.geometry = plane(r);
    auto get = [&] { return M::decode(r.bytes(), limit); };
    c.phases = {get(), get(), get()};
    c.accumulated_roundoff = {get(), get()};
    for (auto* q :
        {&c.error.difference, &c.error.admitted_bound, &c.error.native_conversion, &c.error.phase_input_bound,
            &c.error.fit_residual, &c.error.fit_bound, &c.error.transferred_bound, &c.error.fit_allocation_bound})
        *q = get();
    if (boolean(r))
        c.phase_mass = WorkingMass{get(), get()};
    const auto n = r.word();
    if (n > (bytes.size() - r.offset) / 8)
        throw std::invalid_argument("Invalid exact inventory count.");
    c.amounts.reserve(n);
    for (std::uint64_t i = 0; i < n; ++i)
        c.amounts.push_back(get());
    r.done();
    return c;
}
void validate_work_channels(const WorkingCapsule& c, std::span<const ExactTransferChannel> channels)
{
    if (c.amounts.size() != channels.size())
        throw std::invalid_argument("Exact transfer inventory schema differs.");
    if (c.phase_mass && (c.phase_mass->gas.sign() < 0 || c.phase_mass->carrier.sign() < 0 ||
                            (!c.phases.gas.sign() && c.phase_mass->gas.sign()) ||
                            (!c.phases.carrier.sign() && c.phase_mass->carrier.sign())))
        throw std::invalid_argument("Exact phase mass contradicts source carriers.");
    for (std::size_t i = 0; i < channels.size(); ++i)
    {
        const auto& channel = channels[i];
        if (channel.name.empty() || std::uint64_t(channel.carrier) > 3 ||
            (channel.nonnegative && c.amounts[i].sign() < 0))
            throw std::invalid_argument("Invalid exact transfer inventory.");
        if ((channel.carrier == ExactTransferCarrier::GasVolume && !c.phases.gas.sign() && c.amounts[i].sign()) ||
            (channel.carrier == ExactTransferCarrier::CarrierVolume && !c.phases.carrier.sign() && c.amounts[i].sign()))
            throw std::invalid_argument("Inventory has no positive phase carrier.");
        if (channel.carrier == ExactTransferCarrier::MixtureMass &&
            (!c.phase_mass || (c.phase_mass->gas + c.phase_mass->carrier).sign() <= 0))
            throw std::invalid_argument("Mass-weighted inventory requires explicit positive phase masses.");
    }
}
void validate_channels(const ExactPhaseTransferCapsule& c, std::span<const ExactTransferChannel> channels)
{
    if (c.amounts.size() != channels.size())
        throw std::invalid_argument("Finite transfer channel dimensions differ.");
    const bool has_gas = c.experimental_exact ? c.experimental_exact->phases.gas.sign() != 0 : c.phases.gas != 0;
    const bool has_carrier =
        c.experimental_exact ? c.experimental_exact->phases.carrier.sign() != 0 : c.phases.carrier != 0;
    if (c.phase_mass && (c.phase_mass->gas.sign() < 0 || c.phase_mass->carrier.sign() < 0 ||
                            (!has_gas && c.phase_mass->gas.sign()) || (!has_carrier && c.phase_mass->carrier.sign())))
        throw std::invalid_argument("Transfer mass product contradicts its finite carrier.");
    for (std::size_t i = 0; i < channels.size(); ++i)
    {
        const auto& q = channels[i];
        if (q.name.empty() || std::uint64_t(q.carrier) > 3 || (q.nonnegative && c.amounts[i].sign() < 0) ||
            (q.carrier == ExactTransferCarrier::GasVolume && !has_gas && c.amounts[i].sign()) ||
            (q.carrier == ExactTransferCarrier::CarrierVolume && !has_carrier && c.amounts[i].sign()) ||
            (q.carrier == ExactTransferCarrier::MixtureMass &&
                (!c.phase_mass || (c.phase_mass->gas + c.phase_mass->carrier).sign() <= 0)))
            throw std::invalid_argument("Invalid transfer product/channel carrier.");
    }
}
void validate_capsule(const H& h, const ExactPhaseTransferCapsule& c, std::span<const ExactTransferChannel> channels)
{
    if (std::uint64_t(c.policy) > 1 ||
        (c.policy == PhaseInventoryPolicy::ExperimentalExact) != bool(c.experimental_exact))
        throw std::invalid_argument("Transfer capsule policy/payload mismatch.");
    const auto found = h.owned().find(c.key);
    if (found == h.owned().end() || found->second.revision != c.revision ||
        c.phases.bulk != h.metrics(found->second).volume || !std::isfinite(c.phases.bulk) || c.phases.bulk <= 0 ||
        !std::isfinite(c.phases.gas) || !std::isfinite(c.phases.carrier) || c.phases.gas < 0 || c.phases.carrier < 0)
        throw std::invalid_argument("Finite source capsule ownership/geometry/phase differs.");
    for (double bound : {c.roundoff.gas, c.roundoff.carrier, c.accumulated_roundoff.gas, c.accumulated_roundoff.carrier, c.error.admitted_bound, c.error.phase_input_bound,
             c.error.fit_bound, c.error.transferred_bound, c.error.fit_allocation_bound})
        if (!std::isfinite(bound) || bound < 0)
            throw std::invalid_argument("Invalid finite source bound.");
    for (double value : {c.error.difference, c.error.native_conversion, c.error.fit_residual})
        if (!std::isfinite(value))
            throw std::invalid_argument("Nonfinite source diagnostic.");
    const auto bulk = M::volume(*h.geometry(found->second));
    const auto difference = M::from_value(c.phases.gas) + M::from_value(c.phases.carrier) - bulk;
    if (detail::phase_signed_value(difference) != c.error.difference ||
        detail::phase_signed_value(M::from_value(c.phases.bulk) - bulk) != c.error.native_conversion ||
        abs(difference).compare(M::from_value(c.error.admitted_bound)) > 0 ||
        (c.geometry.pure_gas && c.geometry.pure_carrier))
        throw std::invalid_argument("Finite source occupancy exceeds its proven bound.");
    if (c.experimental_exact)
    {
        const auto& x = *c.experimental_exact;
        const auto actual = x.phases.gas + x.phases.carrier - x.phases.bulk;
        if (x.phases.bulk.compare(bulk) || x.phases.gas.sign() < 0 || x.phases.carrier.sign() < 0 ||
            (x.phases.gas.sign() > 0 && c.phases.gas == 0) || (x.phases.carrier.sign() > 0 && c.phases.carrier == 0) ||
            detail::phase_signed_value(x.phases.gas) != c.phases.gas ||
            detail::phase_signed_value(x.phases.carrier) != c.phases.carrier || actual.compare(x.error.difference) ||
            abs(actual).compare(x.error.admitted_bound) > 0 ||
            (M::from_value(c.phases.bulk) - bulk).compare(x.error.native_conversion))
            throw std::invalid_argument("Experimental transfer authority/provenance differs.");
        for (const auto* bound : {&x.error.admitted_bound, &x.error.phase_input_bound, &x.error.fit_bound,
                 &x.error.transferred_bound, &x.error.fit_allocation_bound})
            if (bound->sign() < 0)
                throw std::invalid_argument("Negative experimental transfer bound.");
    }
    validate_channels(c, channels);
}
AMR::AdaptiveBytes encode(const ExactPhaseTransferCapsule& c, std::size_t limit)
{
    wire::Budget capacity{limit};
    const auto key_bytes = AMR::encode_adaptive_key(c.key);
    capacity.add(528 + key_bytes.size());
    wire::Writer w;
    w.word(0x5346465054523731ULL);
    w.word(3);
    w.word(std::uint64_t(c.policy));
    w.word(sizeof(long double));
    w.word(std::numeric_limits<long double>::digits);
    w.word(std::numeric_limits<long double>::max_exponent);
    w.bytes(AMR::encode_adaptive_key(c.key));
    w.word(c.revision);
    w.word(c.geometry_ready);
    plane(w, c.geometry);
    for (double value : {c.phases.bulk, c.phases.gas, c.phases.carrier, c.roundoff.gas, c.roundoff.carrier,
             c.accumulated_roundoff.gas, c.accumulated_roundoff.carrier, c.error.difference, c.error.admitted_bound, c.error.native_conversion, c.error.phase_input_bound,
             c.error.fit_residual, c.error.fit_bound, c.error.transferred_bound, c.error.fit_allocation_bound})
        real(w, value);
    const auto put = [&](const M& value)
    {
        auto bytes = value.encode(limit);
        capacity.add(bytes.size() + 8);
        w.bytes(bytes);
    };
    w.word(bool(c.experimental_exact));
    if (c.experimental_exact)
    {
        const auto& x = *c.experimental_exact;
        put(x.phases.bulk);
        put(x.phases.gas);
        put(x.phases.carrier);
        for (const auto* q :
            {&x.error.difference, &x.error.admitted_bound, &x.error.native_conversion, &x.error.phase_input_bound,
                &x.error.fit_residual, &x.error.fit_bound, &x.error.transferred_bound, &x.error.fit_allocation_bound})
            put(*q);
    }
    w.word(bool(c.phase_mass));
    if (c.phase_mass)
    {
        put(c.phase_mass->gas);
        put(c.phase_mass->carrier);
    }
    w.word(c.amounts.size());
    for (const auto& value : c.amounts)
        put(value);
    if (w.data.size() > limit)
        throw std::invalid_argument("Finite source capsule exceeds byte budget.");
    return std::move(w.data);
}
ExactPhaseTransferCapsule decode(std::span<const char> bytes, std::size_t limit)
{
    if (bytes.size() > limit)
        throw std::invalid_argument("Finite source capsule exceeds byte budget.");
    wire::Reader r{bytes};
    if (r.word() != 0x5346465054523731ULL || r.word() != 3)
        throw std::invalid_argument("Unknown policy-pinned source capsule format.");
    const auto policy = r.word();
    if (policy > 1)
        throw std::invalid_argument("Invalid source policy.");
    if (r.word() != sizeof(long double) || r.word() != std::numeric_limits<long double>::digits ||
        r.word() != std::numeric_limits<long double>::max_exponent)
        throw std::invalid_argument("Unknown finite source capsule format.");
    ExactPhaseTransferCapsule c;
    c.policy = PhaseInventoryPolicy(policy);
    c.key = AMR::decode_adaptive_key(r.bytes());
    c.revision = r.word();
    c.geometry_ready = boolean(r);
    c.geometry = plane(r);
    c.phases = {real(r), real(r), real(r)};
    c.roundoff = {real(r), real(r)};
    c.accumulated_roundoff = {real(r), real(r)};
    for (auto* q :
        {&c.error.difference, &c.error.admitted_bound, &c.error.native_conversion, &c.error.phase_input_bound,
            &c.error.fit_residual, &c.error.fit_bound, &c.error.transferred_bound, &c.error.fit_allocation_bound})
        *q = real(r);
    const auto get = [&] { return M::decode(r.bytes(), limit); };
    const bool has_exact = boolean(r);
    if (has_exact != (c.policy == PhaseInventoryPolicy::ExperimentalExact))
        throw std::invalid_argument("Source capsule policy/exact payload mismatch.");
    if (has_exact)
    {
        auto x = std::make_shared<ExperimentalPhaseInventory>();
        x->phases = {get(), get(), get()};
        for (auto* q : {&x->error.difference, &x->error.admitted_bound, &x->error.native_conversion,
                 &x->error.phase_input_bound, &x->error.fit_residual, &x->error.fit_bound, &x->error.transferred_bound,
                 &x->error.fit_allocation_bound})
            *q = get();
        c.experimental_exact = std::move(x);
    }
    if (boolean(r))
        c.phase_mass = ExactTransferPhaseMass{get(), get()};
    const auto n = r.word();
    if (n > (r.data.size() - r.offset) / 8)
        throw std::invalid_argument("Transfer product count exceeds packet.");
    if (n > limit / sizeof(M))
        throw std::invalid_argument("Transfer product allocation exceeds packet cap.");
    c.amounts.reserve(n);
    for (std::uint64_t i = 0; i < n; ++i)
        c.amounts.push_back(get());
    r.done();
    return c;
}
ExactPhaseTransferCapsule publish(const WorkingCapsule& c, double native_bulk, ExactPhaseTransferOptions options)
{
    ExactPhaseTransferCapsule out;
    out.key = c.key;
    out.revision = c.revision;
    out.policy = c.policy;
    out.geometry = c.geometry;
    out.geometry_ready = c.geometry_ready;
    const auto gas = detail::round_phase_value(c.phases.gas), carrier = detail::round_phase_value(c.phases.carrier);
    if (c.policy == PhaseInventoryPolicy::ExperimentalExact &&
        ((c.phases.gas.sign() > 0 && gas.value == 0) || (c.phases.carrier.sign() > 0 && carrier.value == 0)))
        throw std::invalid_argument("Experimental exact remap requires representable positive owned views.");
    out.phases = {native_bulk, gas.value, carrier.value};
    out.roundoff = {gas.absolute_error, carrier.absolute_error};
    if (c.policy == PhaseInventoryPolicy::Finite)
        out.accumulated_roundoff = {
            detail::phase_upper_bound(c.accumulated_roundoff.gas + M::from_value(gas.absolute_error)),
            detail::phase_upper_bound(c.accumulated_roundoff.carrier + M::from_value(carrier.absolute_error))};
    const auto round = M::from_value(gas.absolute_error) + M::from_value(carrier.absolute_error);
    const auto defect = M::from_value(gas.value) + M::from_value(carrier.value) - c.phases.bulk;
    const auto diagnostic = detail::round_phase_residual(defect);
    const auto native = M::from_value(native_bulk) - c.phases.bulk;
    const auto native_diagnostic = detail::round_phase_residual(native);
    const auto fit_diagnostic = detail::round_phase_residual(c.error.fit_residual);
    const auto admitted = c.error.admitted_bound + round + M::from_value(diagnostic.absolute_error);
    const auto fit = c.error.fit_allocation_bound + round + M::from_value(fit_diagnostic.absolute_error);
    if (abs(defect).compare(admitted) > 0 ||
        (c.policy == PhaseInventoryPolicy::Finite &&
            (admitted.compare(c.phases.bulk.scaled(options.maximum_relative_target_defect)) > 0 ||
                fit.compare(c.phases.bulk.scaled(options.maximum_relative_target_fit)) > 0)))
        throw std::invalid_argument("Finite target occupancy/fit publication exceeds derived bounds.");
    out.error.difference = diagnostic.value;
    out.error.admitted_bound = detail::phase_upper_bound(admitted);
    out.error.native_conversion = native_diagnostic.value;
    out.error.phase_input_bound = detail::phase_upper_bound(
        c.error.phase_input_bound + round + abs(native) + M::from_value(native_diagnostic.absolute_error));
    out.error.fit_residual = fit_diagnostic.value;
    out.error.fit_bound = out.error.fit_allocation_bound = detail::phase_upper_bound(fit);
    out.error.transferred_bound = detail::phase_upper_bound(c.error.transferred_bound);
    if (c.policy == PhaseInventoryPolicy::ExperimentalExact)
    {
        auto exact = std::make_shared<ExperimentalPhaseInventory>();
        exact->phases = c.phases;
        const auto& e = c.error;
        exact->error = {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound, e.fit_residual,
            e.fit_bound, e.transferred_bound, e.fit_allocation_bound};
        out.experimental_exact = std::move(exact);
    }
    // These bounded products belong only to the current remap transaction.
    // Consumers aggregate before narrowing u/p and never adopt them as history.
    if (c.phase_mass)
        out.phase_mass = ExactTransferPhaseMass{c.phase_mass->gas, c.phase_mass->carrier};
    out.amounts = c.amounts;
    return out;
}
void current(const H& h, const AMR::HierarchyObservation& observation)
{
    if (!h.geometry_unchanged() || h.observation() != observation)
        throw std::invalid_argument("Exact phase transfer source/target observation is stale.");
}
void agree(std::uint64_t control, PhaseInventoryPolicy policy, std::span<const ExactTransferChannel> channels,
    const ExactPhaseTransferOptions& options, Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            if (std::uint64_t(policy) > 1 || !control || !options.byte_budget ||
                !std::isfinite(options.maximum_relative_target_defect) || options.maximum_relative_target_defect < 0 ||
                !std::isfinite(options.maximum_relative_target_fit) || options.maximum_relative_target_fit < 0)
                throw std::invalid_argument("Invalid exact phase transfer controls.");
            wire::Writer w;
            w.word(control);
            w.word(std::uint64_t(policy));
            real(w, options.maximum_relative_target_defect);
            real(w, options.maximum_relative_target_fit);
            w.word(channels.size());
            std::set<std::string> names;
            for (const auto& c : channels)
            {
                if (c.name.empty() || !names.insert(c.name).second || std::uint64_t(c.carrier) > 3)
                    throw std::invalid_argument("Invalid exact transfer channel schema.");
                w.bytes(std::span(c.name));
                w.word(std::uint64_t(c.carrier));
                w.word(c.nonnegative);
            }
            wire::Budget budget{options.byte_budget};
            budget.packet(w.data.size());
            packets.push_back({0, std::move(w.data)});
        });
    const auto received = exchange_owner_packets(packets, comm, options.byte_budget);
    wire::collective(*comm,
        [&]
        {
            if (comm->getRank() == 0)
            {
                if (received.packets.size() != std::size_t(comm->getSize()))
                    throw std::invalid_argument("Missing exact transfer control declaration.");
                for (const auto& p : received.packets)
                    if (p.bytes != received.packets.front().bytes)
                        throw std::invalid_argument("Exact transfer controls/schema differ across ranks.");
            }
        });
}
} // namespace
struct ExactPhaseTransferSource::Data
{
    std::shared_ptr<const Hierarchy> hierarchy;
    std::shared_ptr<const void> identity;
    std::uint64_t control = 0;
    PhaseInventoryPolicy policy = PhaseInventoryPolicy::Finite;
    AMR::HierarchyObservation observation;
    std::vector<ExactPhaseTransferCapsule> owned;
    std::vector<ExactTransferChannel> channels;
};
struct ExactPhaseTransferResult::Data
{
    std::shared_ptr<const ExactPhaseTransferSource::Data> source;
    std::shared_ptr<const H> target;
    AMR::HierarchyObservation observation;
    std::uint64_t transaction = 0;
    bool ownership = false;
    std::vector<ExactPhaseTransferCapsule> owned;
    std::vector<std::shared_ptr<const H>> checkpoints;
    std::vector<AMR::HierarchyObservation> checkpoint_observations;
    std::vector<std::uint64_t> chain_ids;
};
ExactPhaseTransferSource::ExactPhaseTransferSource(std::shared_ptr<const Hierarchy> h, std::shared_ptr<const void> id,
    std::uint64_t control, std::vector<ExactPhaseTransferCapsule> owned, PhaseInventoryPolicy policy)
{
    if (!h || !id || !control || std::uint64_t(policy) > 1 || owned.size() != h->owned().size())
        throw std::invalid_argument("Invalid accepted exact phase source.");
    auto d = std::make_shared<Data>();
    d->hierarchy = std::move(h);
    d->identity = std::move(id);
    d->control = control;
    d->policy = policy;
    d->observation = d->hierarchy->observation();
    current(*d->hierarchy, d->observation);
    std::map<Key, ExactPhaseTransferCapsule> unique;
    for (auto& c : owned)
    {
        if (c.policy != policy)
            throw std::invalid_argument("Source policy differs from owned capsule.");
        validate_capsule(*d->hierarchy, c, {});
        if (!unique.emplace(c.key, std::move(c)).second)
            throw std::invalid_argument("Duplicate exact source key.");
    }
    for (auto& [k, c] : unique)
        d->owned.push_back(std::move(c));
    data_ = std::move(d);
}
ExactPhaseTransferSource::ExactPhaseTransferSource(std::shared_ptr<const Data> d) : data_(std::move(d)) {}
const std::shared_ptr<const ExactPhaseTransferSource::Hierarchy>& ExactPhaseTransferSource::hierarchy() const noexcept
{
    return data_->hierarchy;
}
const std::shared_ptr<const void>& ExactPhaseTransferSource::identity() const noexcept
{
    return data_->identity;
}
std::uint64_t ExactPhaseTransferSource::control() const noexcept
{
    return data_->control;
}
PhaseInventoryPolicy ExactPhaseTransferSource::inventory_policy() const noexcept
{
    return data_->policy;
}
const std::vector<ExactPhaseTransferCapsule>& ExactPhaseTransferSource::owned() const noexcept
{
    return data_->owned;
}
const std::vector<ExactTransferChannel>& ExactPhaseTransferSource::channels() const noexcept
{
    return data_->channels;
}
ExactPhaseTransferSource ExactPhaseTransferSource::with_channels(std::vector<ExactTransferChannel> channels,
    std::vector<ExactTransferOwnedChannels> values, ExactPhaseTransferOptions options) const
{
    const auto comm = data_->hierarchy->communicator();
    agree(data_->control, data_->policy, channels, options, comm);
    std::shared_ptr<Data> next;
    wire::collective(*comm,
        [&]
        {
            current(*data_->hierarchy, data_->observation);
            if (values.size() != data_->owned.size())
                throw std::invalid_argument("Exact inventory inputs do not cover local owners.");
            Capacity budget(options.byte_budget);
            next = std::make_shared<Data>(*data_);
            next->channels = std::move(channels);
            std::map<Key, ExactTransferOwnedChannels> keyed;
            for (auto& value : values)
                if (!keyed.emplace(value.key, std::move(value)).second)
                    throw std::invalid_argument("Duplicate exact inventory source.");
            for (auto& c : next->owned)
            {
                auto& input = keyed.at(c.key);
                c.amounts = std::move(input.amounts);
                c.phase_mass = std::move(input.phase_mass);
                validate_channels(c, next->channels);
                budget.capsule(c);
            }
        });
    return ExactPhaseTransferSource(std::move(next));
}
ExactPhaseTransferResult::ExactPhaseTransferResult(std::shared_ptr<const Data> d) : data_(std::move(d)) {}
const std::shared_ptr<const H>& ExactPhaseTransferResult::source_hierarchy() const noexcept
{
    return data_->source->hierarchy;
}
const std::shared_ptr<const H>& ExactPhaseTransferResult::target_hierarchy() const noexcept
{
    return data_->target;
}
const std::shared_ptr<const void>& ExactPhaseTransferResult::source_identity() const noexcept
{
    return data_->source->identity;
}
std::uint64_t ExactPhaseTransferResult::source_control() const noexcept
{
    return data_->source->control;
}
std::uint64_t ExactPhaseTransferResult::transaction() const noexcept
{
    return data_->transaction;
}
bool ExactPhaseTransferResult::ownership_only() const noexcept
{
    return data_->ownership;
}
PhaseInventoryPolicy ExactPhaseTransferResult::inventory_policy() const noexcept
{
    return data_->source->policy;
}
const std::vector<ExactPhaseTransferCapsule>& ExactPhaseTransferResult::owned() const noexcept
{
    return data_->owned;
}
const std::vector<ExactTransferChannel>& ExactPhaseTransferResult::channels() const noexcept
{
    return data_->source->channels;
}
void ExactPhaseTransferResult::validate_observations() const
{
    wire::collective(*data_->source->hierarchy->communicator(),
        [&]
        {
            current(*data_->source->hierarchy, data_->source->observation);
            current(*data_->target, data_->observation);
            for (std::size_t i = 0; i < data_->checkpoints.size(); ++i)
                current(*data_->checkpoints[i], data_->checkpoint_observations.at(i));
        });
}
namespace
{
std::vector<ExactPhaseTransferCapsule> migrate_capsules(
    const ExactPhaseTransferSource::Data& source, const H& target, std::size_t limit)
{
    const auto comm = source.hierarchy->communicator();
    std::vector<Key> keys;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, leaf] : target.owned())
            {
                budget.add(sizeof(Key) + AMR::encode_adaptive_key(key).size());
                keys.push_back(key);
            }
        });
    const auto locations = source.hierarchy->lookup_cells(keys);
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                auto bytes = AMR::encode_adaptive_key(keys[i]);
                budget.packet(bytes.size() + 24);
                wire::Writer w;
                w.word(i);
                w.bytes(bytes);
                w.word(locations[i].revision);
                requests.push_back({locations[i].owner, std::move(w.data)});
            }
        });
    auto incoming = exchange_owner_packets(requests, comm, limit);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            std::map<Key, const ExactPhaseTransferCapsule*> index;
            for (const auto& c : source.owned)
                index.emplace(c.key, &c);
            wire::Budget budget{limit};
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                auto ordinal = r.word();
                auto key = AMR::decode_adaptive_key(r.bytes());
                auto revision = r.word();
                r.done();
                const auto& c = *index.at(key);
                if (c.revision != revision)
                    throw std::invalid_argument("Stale exact ownership capsule request.");
                auto bytes = encode(c, limit);
                budget.packet(bytes.size() + 16);
                wire::Writer w;
                w.word(ordinal);
                w.bytes(bytes);
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    auto received = exchange_owner_packets(replies, comm, limit);
    std::vector<ExactPhaseTransferCapsule> result;
    wire::collective(*comm,
        [&]
        {
            Capacity budget(limit);
            budget.budget.add(
                keys.size() * (sizeof(std::optional<ExactPhaseTransferCapsule>) + sizeof(ExactPhaseTransferCapsule)));
            std::vector<std::optional<ExactPhaseTransferCapsule>> slots(keys.size());
            result.reserve(keys.size());
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto i = r.word();
                auto bytes = r.bytes();
                r.done();
                if (i >= keys.size() || slots[i] || packet.peer != locations[i].owner)
                    throw std::invalid_argument("Foreign or duplicate exact ownership capsule reply.");
                auto c = decode(bytes, limit);
                if (c.policy != source.policy || c.key != keys[i] || c.revision != locations[i].revision ||
                    target.owned().at(c.key).revision != c.revision)
                    throw std::invalid_argument("Exact ownership key/revision differs.");
                validate_capsule(target, c, source.channels);
                if (target.metrics(target.owned().at(c.key)).volume != c.phases.bulk)
                    throw std::invalid_argument("Ownership-only native metric conversion changed.");
                budget.capsule(c);
                slots[i] = std::move(c);
            }
            for (auto& c : slots)
            {
                if (!c)
                    throw std::invalid_argument("Missing exact ownership capsule reply.");
                result.push_back(std::move(*c));
            }
        });
    return result;
}
struct ColumnRecord
{
    Key source, target;
    std::uint64_t source_gid = 0, target_gid = 0, target_revision = 0;
    int owner = -1;
    SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary> geometry;
};
SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> raw_phases(
    const WorkingCapsule& c, const SimpleFluid::geometry::swept_volume::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>& fragment)
{
    if (!c.phases.gas.sign() || c.geometry.pure_carrier)
        return {fragment.volume(), {}, fragment.volume()};
    if (!c.phases.carrier.sign() || c.geometry.pure_gas)
        return {fragment.volume(), fragment.volume(), {}};
    if (!c.geometry_ready)
        throw std::invalid_argument("Topology phase splitting needs an accepted source reconstruction.");
    return fragment.phases(c.geometry.gas);
}
M weight(const M& part, const M& whole, std::size_t limit)
{
    if (!whole.sign())
    {
        if (part.sign())
            throw std::invalid_argument("Positive exact transfer has no carrier.");
        return {};
    }
    return part.divided_by(whole, limit);
}
std::vector<ExactPhaseTransferCapsule> remap_capsules(
    const ExactPhaseTransferSource::Data& source, const H& target, ExactPhaseTransferOptions options)
{
    const auto comm = source.hierarchy->communicator();
    const auto limit = options.byte_budget;
    auto overlaps = detail::build_adaptive_overlap(*source.hierarchy, target, limit);
    std::vector<Key> sources;
    std::vector<const AMR::AdaptiveOverlapRecord*> selected;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& record : overlaps->records())
                if (target.owned().contains(record.target_key))
                {
                    budget.add(sizeof(Key) + sizeof(void*) + AMR::encode_adaptive_key(record.source_key).size());
                    sources.push_back(record.source_key);
                    selected.push_back(&record);
                }
        });
    const auto source_owners = source.hierarchy->lookup_cells(sources);
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (std::size_t i = 0; i < selected.size(); ++i)
            {
                const auto& record = *selected[i];
                const auto& leaf = target.owned().at(record.target_key);
                auto sk = AMR::encode_adaptive_key(record.source_key), tk = AMR::encode_adaptive_key(record.target_key);
                budget.packet(sk.size() + tk.size() + 56);
                wire::Writer w;
                w.bytes(sk);
                w.word(source_owners[i].revision);
                w.word(record.source_gid);
                w.bytes(tk);
                w.word(leaf.revision);
                w.word(record.target_gid);
                requests.push_back({source_owners[i].owner, std::move(w.data)});
            }
        });
    auto incoming = exchange_owner_packets(requests, comm, limit);
    std::vector<ColumnRecord> columns;
    std::vector<Key> target_keys;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::set<std::pair<Key, Key>> seen;
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                ColumnRecord record;
                record.source = AMR::decode_adaptive_key(r.bytes());
                auto revision = r.word();
                record.source_gid = r.word();
                record.target = AMR::decode_adaptive_key(r.bytes());
                record.target_revision = r.word();
                record.target_gid = r.word();
                r.done();
                const auto& leaf = source.hierarchy->owned().at(record.source);
                if (leaf.revision != revision || !seen.emplace(record.source, record.target).second)
                    throw std::invalid_argument("Foreign or duplicate exact phase overlap request.");
                budget.add(sizeof(ColumnRecord) + AMR::encode_adaptive_key(record.source).size() +
                           AMR::encode_adaptive_key(record.target).size());
                record.owner = packet.peer;
                columns.push_back(std::move(record));
                target_keys.push_back(columns.back().target);
            }
        });
    const auto target_owners = target.lookup_cells(target_keys);
    const auto target_leaves = target.fetch_leaves(target_keys);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            Capacity budget(limit);
            std::map<Key, std::vector<ColumnRecord*>> grouped;
            for (std::size_t i = 0; i < columns.size(); ++i)
            {
                auto& record = columns[i];
                const auto& owner = target_owners[i];
                const auto& leaf = target_leaves[i];
                const auto& donor = source.hierarchy->owned().at(record.source);
                if (owner.owner != record.owner || owner.revision != record.target_revision ||
                    owner.address != record.target_gid || leaf.key != record.target ||
                    leaf.revision != record.target_revision || leaf.material != donor.material ||
                    leaf.key.region != donor.key.region)
                    throw std::invalid_argument("Exact phase overlap target owner/material differs.");
                record.geometry = source.hierarchy->geometry(donor)->intersect(*target.geometry(leaf));
                if (!record.geometry)
                    throw std::invalid_argument("Exact phase overlap is geometrically empty.");
                budget.budget.add(record.geometry->storage_bytes());
                grouped[record.source].push_back(&record);
            }
            for (const auto& finite_original : source.owned)
            {
                const auto original = lift(finite_original,
                    M::volume(*source.hierarchy->geometry(source.hierarchy->owned().at(finite_original.key))));
                const auto& records = grouped.at(original.key);
                const auto donor_geometry = source.hierarchy->geometry(source.hierarchy->owned().at(original.key));
                std::vector<SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>> cover;
                for (auto r : records)
                    cover.push_back(r->geometry);
                donor_geometry->certify_partition(cover);
                const bool whole = records.size() == 1;
                const bool gas = original.phases.gas.compare(original.phases.carrier) <= 0;
                const M physical_minor = gas ? original.phases.gas : original.phases.carrier;
                const M base_major = original.phases.bulk - physical_minor;
                if (base_major.sign() <= 0)
                    throw std::invalid_argument("Exact phase source has no positive dominant geometric capacity.");
                M raw_minor, raw_major;
                std::optional<detail::ExactPhaseLabelMap> label_map;
                std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> raw;
                if (!whole)
                {
                    for (auto r : records)
                    {
                        auto phase = raw_phases(original,
                            *SimpleFluid::geometry::swept_volume::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>::from_geometry(
                                *r->geometry));
                        raw_minor = raw_minor + (gas ? phase.gas : phase.carrier);
                        raw_major = raw_major + (gas ? phase.carrier : phase.gas);
                        raw.push_back(std::move(phase));
                    }
                    if ((raw_minor + raw_major).compare(original.phases.bulk))
                        throw std::invalid_argument("Raw exact overlap phase column does not fill actual source.");
                    SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> full{
                        original.phases.bulk, gas ? raw_minor : raw_major, gas ? raw_major : raw_minor};
                    label_map =
                        detail::ExactPhaseLabelMap::create(full, gas, physical_minor, original.error.fit_bound, limit);
                }
                M column_gas, column_carrier, column_bulk, column_history_gas, column_history_carrier;
                std::vector<M> column_amount(original.amounts.size());
                M column_mass_gas, column_mass_carrier;
                for (std::size_t i = 0; i < records.size(); ++i)
                {
                    const auto& record = *records[i];
                    WorkingCapsule out;
                    out.policy = source.policy;
                    out.key = record.target;
                    out.revision = record.target_revision;
                    out.phases.bulk = M::volume(*record.geometry);
                    out.geometry = original.geometry;
                    out.geometry_ready = false;
                    M minor, major, share, fit;
                    if (whole)
                    {
                        out.phases = original.phases;
                        share = M::from_value(1);
                        fit = {};
                    }
                    else
                    {
                        const auto p = gas ? raw[i].gas : raw[i].carrier;
                        const auto mapped = label_map->apply(raw[i]);
                        minor = gas ? mapped.gas : mapped.carrier;
                        major = gas ? mapped.carrier : mapped.gas;
                        if (minor.sign() < 0 || major.sign() < 0 || (minor + major).compare(out.phases.bulk))
                            throw std::invalid_argument("Paired phase transfer violates exact overlap capacity.");
                        share = weight(major, base_major, limit);
                        const auto defect = product(original.error.difference, share, limit);
                        const auto physical_major = major + defect;
                        if (physical_major.sign() < 0)
                            throw std::invalid_argument(
                                "Signed numerical defect makes a dominant physical phase negative.");
                        out.phases.gas = gas ? minor : physical_major;
                        out.phases.carrier = gas ? physical_major : minor;
                        fit = abs(minor - p);
                    }
                    const auto wg = weight(out.phases.gas, original.phases.gas, limit),
                               wc = weight(out.phases.carrier, original.phases.carrier, limit),
                               wb = weight(out.phases.bulk, original.phases.bulk, limit);
                    out.accumulated_roundoff = {
                        product(original.accumulated_roundoff.gas, original.phases.gas.sign() ? wg : wb, limit),
                        product(original.accumulated_roundoff.carrier, original.phases.carrier.sign() ? wc : wb, limit)};
                    out.error.difference = out.phases.gas + out.phases.carrier - out.phases.bulk;
                    out.error.transferred_bound = product(abs(original.error.difference), share, limit);
                    out.error.admitted_bound = product(original.error.admitted_bound, share, limit);
                    out.error.fit_allocation_bound = product(original.error.fit_allocation_bound, wb, limit) + fit;
                    out.error.fit_bound = out.error.fit_allocation_bound;
                    out.error.fit_residual = {};
                    if (original.phase_mass)
                        out.phase_mass = WorkingMass{product(original.phase_mass->gas, wg, limit),
                            product(original.phase_mass->carrier, wc, limit)};
                    for (std::size_t channel = 0; channel < source.channels.size(); ++channel)
                    {
                        M w;
                        switch (source.channels[channel].carrier)
                        {
                            case ExactTransferCarrier::BulkVolume:
                                w = wb;
                                break;
                            case ExactTransferCarrier::GasVolume:
                                w = wg;
                                break;
                            case ExactTransferCarrier::CarrierVolume:
                                w = wc;
                                break;
                            case ExactTransferCarrier::MixtureMass:
                                w = weight(out.phase_mass->gas + out.phase_mass->carrier,
                                    original.phase_mass->gas + original.phase_mass->carrier, limit);
                                break;
                        }
                        out.amounts.push_back(product(original.amounts[channel], w, limit));
                        column_amount[channel] = column_amount[channel] + out.amounts.back();
                    }
                    column_bulk = column_bulk + out.phases.bulk;
                    column_gas = column_gas + out.phases.gas;
                    column_carrier = column_carrier + out.phases.carrier;
                    column_history_gas = column_history_gas + out.accumulated_roundoff.gas;
                    column_history_carrier = column_history_carrier + out.accumulated_roundoff.carrier;
                    if (out.phase_mass)
                    {
                        column_mass_gas = column_mass_gas + out.phase_mass->gas;
                        column_mass_carrier = column_mass_carrier + out.phase_mass->carrier;
                    }
                    budget.capsule(out);
                    auto sk = AMR::encode_adaptive_key(original.key), bytes = encode_work(out, limit);
                    budget.budget.packet(sk.size() + bytes.size() + 32);
                    wire::Writer w;
                    w.bytes(sk);
                    w.word(original.revision);
                    w.word(record.source_gid);
                    w.bytes(bytes);
                    replies.push_back({record.owner, std::move(w.data)});
                }
                if (column_bulk.compare(original.phases.bulk) || column_gas.compare(original.phases.gas) ||
                    column_carrier.compare(original.phases.carrier))
                    throw std::invalid_argument("Exact source phase columns do not close.");
                if (column_history_gas.compare(original.accumulated_roundoff.gas) ||
                    column_history_carrier.compare(original.accumulated_roundoff.carrier))
                    throw std::invalid_argument("Explicit reference transfer lost finite arithmetic history.");
                for (std::size_t i = 0; i < column_amount.size(); ++i)
                    if (column_amount[i].compare(original.amounts[i]))
                        throw std::invalid_argument("Exact extensive inventory column does not close.");
                if (original.phase_mass && (column_mass_gas.compare(original.phase_mass->gas) ||
                                               column_mass_carrier.compare(original.phase_mass->carrier)))
                    throw std::invalid_argument("Exact source phase mass columns do not close.");
            }
        });
    auto received = exchange_owner_packets(replies, comm, limit);
    struct RowPart
    {
        Key source;
        std::uint64_t revision = 0, gid = 0;
        int owner = -1;
        WorkingCapsule capsule;
    };
    std::vector<RowPart> parts;
    std::vector<Key> source_keys;
    wire::collective(*comm,
        [&]
        {
            Capacity budget(limit);
            std::set<std::pair<Key, Key>> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                RowPart p;
                p.source = AMR::decode_adaptive_key(r.bytes());
                p.revision = r.word();
                p.gid = r.word();
                p.capsule = decode_work(r.bytes(), limit);
                r.done();
                p.owner = packet.peer;
                if (p.capsule.policy != source.policy || !target.owned().contains(p.capsule.key) ||
                    target.owned().at(p.capsule.key).revision != p.capsule.revision ||
                    !seen.emplace(p.source, p.capsule.key).second)
                    throw std::invalid_argument("Foreign or duplicate exact phase row reply.");
                budget.capsule(p.capsule);
                source_keys.push_back(p.source);
                parts.push_back(std::move(p));
            }
        });
    const auto owners = source.hierarchy->lookup_cells(source_keys);
    std::vector<ExactPhaseTransferCapsule> result;
    wire::collective(*comm,
        [&]
        {
            Capacity budget(limit);
            std::map<Key, WorkingCapsule> rows;
            std::map<Key, bool> all_gas, all_carrier;
            for (std::size_t i = 0; i < parts.size(); ++i)
            {
                auto& part = parts[i];
                const auto& owner = owners[i];
                if (owner.owner != part.owner || owner.revision != part.revision || owner.address != part.gid)
                    throw std::invalid_argument("Exact phase row source owner/revision/GID differs.");
                auto& c = part.capsule;
                auto [it, fresh] = rows.try_emplace(c.key);
                auto& out = it->second;
                if (fresh)
                {
                    out.key = c.key;
                    out.revision = c.revision;
                    out.policy = c.policy;
                    out.amounts.resize(source.channels.size());
                    if (c.phase_mass)
                        out.phase_mass = WorkingMass{};
                    all_gas[c.key] = true;
                    all_carrier[c.key] = true;
                }
                if (out.policy != c.policy || bool(out.phase_mass) != bool(c.phase_mass) ||
                    out.amounts.size() != c.amounts.size())
                    throw std::invalid_argument("Exact phase row inventory schema differs.");
                out.phases.bulk = out.phases.bulk + c.phases.bulk;
                out.phases.gas = out.phases.gas + c.phases.gas;
                out.phases.carrier = out.phases.carrier + c.phases.carrier;
                out.accumulated_roundoff.gas = out.accumulated_roundoff.gas + c.accumulated_roundoff.gas;
                out.accumulated_roundoff.carrier = out.accumulated_roundoff.carrier + c.accumulated_roundoff.carrier;
                out.error.difference = out.error.difference + c.error.difference;
                out.error.admitted_bound = out.error.admitted_bound + c.error.admitted_bound;
                out.error.transferred_bound = out.error.transferred_bound + c.error.transferred_bound;
                out.error.fit_allocation_bound = out.error.fit_allocation_bound + c.error.fit_allocation_bound;
                if (out.phase_mass)
                {
                    out.phase_mass->gas = out.phase_mass->gas + c.phase_mass->gas;
                    out.phase_mass->carrier = out.phase_mass->carrier + c.phase_mass->carrier;
                }
                for (std::size_t q = 0; q < out.amounts.size(); ++q)
                    out.amounts[q] = out.amounts[q] + c.amounts[q];
                all_gas[c.key] = all_gas[c.key] && c.geometry.pure_gas;
                all_carrier[c.key] = all_carrier[c.key] && c.geometry.pure_carrier;
            }
            for (const auto& [key, leaf] : target.owned())
            {
                auto c = std::move(rows.at(key));
                const auto bulk = M::volume(*target.geometry(leaf));
                if (c.phases.bulk.compare(bulk) || c.phases.gas.sign() < 0 || c.phases.carrier.sign() < 0 ||
                    (c.phases.gas + c.phases.carrier - bulk).compare(c.error.difference) ||
                    abs(c.error.difference).compare(c.error.admitted_bound) > 0 ||
                    c.error.admitted_bound.compare(bulk.scaled(options.maximum_relative_target_defect)) > 0)
                    throw std::invalid_argument("Derived exact target bulk/defect capacity failed.");
                if (c.error.fit_allocation_bound.compare(bulk.scaled(options.maximum_relative_target_fit)) > 0)
                    throw std::invalid_argument("Derived exact target phase-fit budget exhausted.");
                c.error.native_conversion = M::from_value(target.metrics(leaf).volume) - bulk;
                c.error.phase_input_bound = c.error.admitted_bound + abs(c.error.native_conversion);
                c.error.fit_bound = c.error.fit_allocation_bound;
                c.geometry.pure_gas = all_gas.at(key);
                c.geometry.pure_carrier = all_carrier.at(key);
                c.geometry_ready = false;
                validate_work_channels(c, source.channels);
                budget.capsule(c);
                result.push_back(publish(c, target.metrics(leaf).volume, options));
            }
        });
    return result;
}
} // namespace
ExactPhaseTransferResult ExactPhaseTransfer::prepare(const ExactPhaseTransferSource& input, const H::Plan& plan,
    std::shared_ptr<const H> target, ExactPhaseTransferOptions options)
{
    const std::array<ExactTransferPlanStep, 1> step{{{plan, std::move(target)}}};
    return prepare(input, step, options);
}
ExactPhaseTransferResult ExactPhaseTransfer::prepare(const ExactPhaseTransferSource& input,
    std::span<const ExactTransferPlanStep> chain, ExactPhaseTransferOptions options)
{
    const auto& source = *input.data_;
    const auto comm = source.hierarchy->communicator();
    agree(source.control, source.policy, source.channels, options, comm);
    std::array<std::uint64_t, 9> manifest{};
    wire::collective(*comm,
        [&]
        {
            if (chain.empty() || chain.size() > 8)
                throw std::invalid_argument("Exact transfer plan chain must contain one to eight steps.");
            wire::Budget budget{options.byte_budget};
            budget.add(chain.size() *
                       (sizeof(std::shared_ptr<const H>) + sizeof(AMR::HierarchyObservation) + sizeof(std::uint64_t)));
            manifest[0] = chain.size();
            std::set<std::uint64_t> ids;
            for (std::size_t i = 0; i < chain.size(); ++i)
            {
                const auto id = chain[i].plan.collective_identity();
                if (!chain[i].candidate || !chain[i].candidate->native_partition() ||
                    chain[i].candidate->communicator().getRawPtr() != comm.getRawPtr())
                    throw std::invalid_argument("Exact transfer chain needs actual compatible native candidates.");
                if (!id || !ids.insert(id).second)
                    throw std::invalid_argument("Invalid or repeated exact transfer chain transaction.");
                manifest[i + 1] = id;
            }
        });
    auto agreed_manifest = manifest;
    Teuchos::broadcast(*comm, 0, int(agreed_manifest.size()), agreed_manifest.data());
    wire::collective(*comm,
        [&]
        {
            if (manifest != agreed_manifest)
                throw std::invalid_argument("Exact transfer plan chains differ across ranks.");
        });
    wire::collective(*comm,
        [&]
        {
            current(*source.hierarchy, source.observation);
            if (!source.hierarchy->native_partition())
                throw std::invalid_argument("Exact phase transfer needs actual native-bound endpoints.");
        });
    std::vector<std::shared_ptr<const H>> checkpoints;
    std::vector<AMR::HierarchyObservation> checkpoint_observations;
    auto step_source = source.hierarchy;
    wire::collective(*comm,
        [&]
        {
            checkpoints.reserve(chain.size());
            checkpoint_observations.reserve(chain.size());
        });
    for (std::size_t i = 0; i < chain.size(); ++i)
    {
        auto candidate = step_source->apply(chain[i].plan, chain[i].candidate);
        wire::collective(*comm,
            [&]
            {
                checkpoints.push_back(candidate);
                checkpoint_observations.push_back(candidate->observation());
            });
        step_source = std::move(candidate);
    }
    auto target = step_source;
    std::shared_ptr<ExactPhaseTransferResult::Data> result;
    wire::collective(*comm,
        [&]
        {
            result = std::make_shared<ExactPhaseTransferResult::Data>();
            result->source = input.data_;
            result->target = target;
            result->observation = target->observation();
            result->transaction = chain.back().plan.collective_identity();
            result->checkpoints = std::move(checkpoints);
            result->checkpoint_observations = std::move(checkpoint_observations);
            result->chain_ids.assign(manifest.begin() + 1, manifest.begin() + 1 + chain.size());
            result->ownership = target->topology_generation() == source.hierarchy->topology_generation();
        });
    if (result->ownership)
        result->owned = migrate_capsules(source, *target, options.byte_budget);
    else
        result->owned = remap_capsules(source, *target, options);
    wire::collective(*comm,
        [&]
        {
            current(*source.hierarchy, source.observation);
            current(*target, result->observation);
            for (std::size_t i = 0; i < result->checkpoints.size(); ++i)
                current(*result->checkpoints[i], result->checkpoint_observations.at(i));
        });
    return ExactPhaseTransferResult(std::move(result));
}
} // namespace SimpleFluid::VoF
