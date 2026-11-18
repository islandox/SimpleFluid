/** @file NativeAllocationGraph.hh
 * @brief Mode-neutral first/last projection of actual native fixed allocations.
 */
#pragma once
#include "geometry/details/NativeAllocationEnvelope.hh"
namespace SimpleFluid::geometry_detail
{
struct NativeAllocationGraphSink
{
    void* context;
    bool (*edge)(void*, const std::shared_ptr<const void>&, std::size_t bytes, bool prepaid);
    const std::size_t* operation_used = nullptr;
};
class NativeAllocationGraphAccess
{
public:
    [[nodiscard]] static std::size_t validate_state(const std::shared_ptr<const NativeAllocationState>& state)
    {
        if (!state || state->self.lock().get() != state.get() || state->self.owner_before(state) ||
            state.owner_before(state->self) || !state->counter_owner || state->counter != state->counter_owner.get() ||
            !state->allocation.address || state->allocation.bytes != state->allocation_admission_bytes ||
            state->counter_allocations.object.address != state->counter || !state->counter_allocations.object.bytes ||
            !state->counter_allocations.control.address || !state->counter_allocations.control.bytes ||
            !native_envelope_detail::contains(state->allocation, state.get(), sizeof(NativeAllocationState)) ||
            !native_envelope_detail::contains(
                state->counter_allocations.object, state->used_counter, sizeof(std::size_t)))
            throw std::invalid_argument("Native graph lacks actual detached counter/state allocation receipts.");
        const auto first = state->counter_allocations.object.bytes, second = state->counter_allocations.control.bytes;
        if (second > std::numeric_limits<std::size_t>::max() - first)
            throw std::invalid_argument("Native graph scalar counter size overflows.");
        if (state->mirror.owner && !state->mirror.owner_proof.matches(state->mirror.owner))
            throw std::invalid_argument("Native mirror lacks its exact closed scalar-owner proof.");
        return first + second;
    }
    static void state(
        const std::shared_ptr<const NativeAllocationState>& state, const NativeAllocationGraphSink& sink, bool prepaid)
    {
        const auto bytes = validate_state(state);
        if (!sink.edge)
            throw std::invalid_argument("Native graph lacks its synchronous edge visitor.");
        const bool paid = prepaid || sink.operation_used == state->used_counter;
        if (sink.edge(sink.context, state, state->allocation_admission_bytes, paid))
        {
            if (state->mirror.owner)
                for (unsigned edge = 0; edge != 2; ++edge)
                    sink.edge(sink.context, state->mirror.owner, state->mirror.owner_proof.bytes(), paid);
            // The Counter has two actual incoming fields. Retention reaches
            // first on the first call; retirement reaches last on the second.
            // Either transition owns the one set of Counter child fields.
            const bool first = sink.edge(sink.context, state->counter_owner, bytes, paid);
            const bool second = sink.edge(sink.context, state->counter_owner, bytes, paid);
            if ((first || second) && state->mirror.owner)
                for (unsigned edge = 0; edge != 2; ++edge)
                    sink.edge(sink.context, state->mirror.owner, state->mirror.owner_proof.bytes(), paid);
        }
    }
    static bool fixed(const std::shared_ptr<const NativeAllocationEnvelope>& owner,
        const NativeAllocationGraphSink& sink, bool prepaid)
    {
        if (!owner)
            return false;
        if (!owner->authentic_owner(owner))
            throw std::invalid_argument("Native fixed envelope has a foreign shared-control owner.");
        const auto& domain = owner->state();
        (void) validate_state(domain);
        const auto bytes = owner->fixed_allocation_bytes();
        if (!sink.edge)
            throw std::invalid_argument("Native graph lacks its synchronous edge visitor.");
        const bool paid = prepaid || sink.operation_used == domain->used_counter;
        const bool transition = sink.edge(sink.context, owner, bytes, paid);
        if (transition)
            for (unsigned edge = 0; edge != 5; ++edge)
                state(domain, sink, paid);
        return transition;
    }
};
} // namespace SimpleFluid::geometry_detail
