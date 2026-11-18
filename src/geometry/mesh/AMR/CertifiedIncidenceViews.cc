#include "CertifiedIncidenceViews.hh"
#include "NativeContactFunding.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using Allocation = cartesian_detail::Allocation;
using Access = PLIC::details::GeometryIdentityAccess;
using Credits = std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>;
class MetadataEnvironment
{
    std::fenv_t caller_{};

public:
    MetadataEnvironment()
    {
        if (std::feholdexcept(&caller_))
            throw std::invalid_argument("Cannot isolate certified incidence identity comparison.");
    }
    ~MetadataEnvironment() { (void) std::fesetenv(&caller_); }
};
bool full_equal(const ContactSourcePoint& a, const ContactSourcePoint& b)
{
    const auto bytes = PLIC::canonical_key_size(a);
    if (bytes != PLIC::canonical_key_size(b))
        return false;
    for (std::size_t i = 0; i < bytes; i += 8)
        if (Access::canonical_word_at(a, i) != Access::canonical_word_at(b, i))
            return false;
    return true;
}
bool same_namespace(const ContactSourcePointSnapshot& a, const ContactSourcePointSnapshot& b)
{
    return a.slot_namespace() && a.slot_namespace() == b.slot_namespace();
}
bool same_slot(const ContactSourcePointSnapshot& saved, std::size_t slot, const ContactSourcePointSnapshot& current,
    std::size_t expected)
{
    return slot == expected && slot < saved.size() && expected < current.size() &&
           full_equal(saved.value({slot}), current.value({expected}));
}
/** The typed token survives body destruction in its custom shared deleter,
 * including the final weak-control tail. No callbacks or arena ownership. */
using contact_funding_detail::ControlCredit;
using contact_funding_detail::make_controlled;
constexpr auto control_bytes = ControlCredit::allowance();
template<class T>
void retain_owner(cartesian_detail::CreditConstruction& credits, const std::shared_ptr<T>& owner, const Budget* origin,
    std::size_t bytes, bool keeper, Credits& leases, Allocation& memory)
{
    const std::shared_ptr<const std::remove_const_t<T>> fixed = owner;
    credits.retain(fixed, origin, bytes, keeper, leases, memory);
}
template<class T>
bool descend_owner(cartesian_detail::CreditConstruction& credits, const std::shared_ptr<T>& owner,
    const Budget* origin, std::size_t bytes, bool keeper, Credits& leases, Allocation& memory)
{
    const std::shared_ptr<const std::remove_const_t<T>> fixed = owner;
    return credits.descend(fixed, origin, bytes, keeper, leases, memory);
}

void retain_snapshot(cartesian_detail::CreditConstruction& credits,
    const std::shared_ptr<const ContactSourcePointSnapshot>& snapshot, Credits& leases, Allocation& memory)
{
    if (!credits.descend(snapshot, snapshot->allocation_budget(), snapshot->metadata_bytes(),
            snapshot->has_arena_keeper(), leases, memory))
        return;
    const auto visit = [&](const auto& owner, auto origin, auto bytes, auto keeper)
    { return descend_owner(credits, owner, origin, bytes, keeper, leases, memory); };
    if (credits.observer)
        snapshot->visit_dependencies<CartesianDependencyVisit::Observed>(visit);
    else
        snapshot->visit_dependencies<CartesianDependencyVisit::Physical>(visit);
}
/** Five actual global component IDs occupy 60 bits; the two axes occupy four.
 * This is a saved-value selector, with no coordinate calculation. */
std::optional<std::size_t> quad_slot(const ContactSourcePointSnapshot& source, std::uint64_t code, std::size_t corner)
{
    if (corner >= 4)
        return {};
    constexpr std::uint64_t mask = 4095;
    std::array<std::size_t, 3> components{};
    for (unsigned axis = 0; axis < 3; ++axis)
        components[axis] = (code >> (12 * axis)) & mask;
    const auto a = unsigned((code >> 60) & 3), b = unsigned((code >> 62) & 3);
    if (a >= 3 || b >= 3 || a == b)
        return {};
    if (corner == 1 || corner == 2)
        components[a] = (code >> 36) & mask;
    if (corner == 2 || corner == 3)
        components[b] = (code >> 48) & mask;
    return source.find_components(components);
}
/** Enumerate frozen actual tuples in their scalar metadata order. This never
 * asks whether a point lies on a segment, constructs a lattice value, or calls
 * a geometry predicate. Candidate equality below is the only admission gate. */
std::optional<std::size_t> actual_axis_slot(const ContactSourcePointSnapshot& source,
    std::array<std::size_t, 3> components, unsigned axis, std::size_t first, std::size_t last,
    std::span<const std::uint64_t> mask, std::size_t ordinal)
{
    const auto start = source.canonical_component(axis, first).value();
    const auto finish = source.canonical_component(axis, last).value();
    const bool ascending = start < finish;
    std::optional<std::size_t> previous;
    for (std::size_t step = 0; step <= ordinal; ++step)
    {
        std::optional<std::size_t> best;
        const auto count = source.component_counts()[axis];
        for (std::size_t c = 0; c < count; ++c)
        {
            if (!mask.empty() && !(mask[c / 64] & (std::uint64_t(1) << (c % 64))))
                continue;
            const auto scalar = source.canonical_component(axis, c).value();
            if (ascending ? (scalar < start || scalar > finish) : (scalar > start || scalar < finish))
                continue;
            if (previous)
            {
                const auto before = source.canonical_component(axis, *previous).value();
                if (ascending ? scalar <= before : scalar >= before)
                    continue;
            }
            components[axis] = c;
            if (!source.find_components(components))
                continue;
            if (!best || (ascending ? scalar < source.canonical_component(axis, *best).value()
                                    : scalar > source.canonical_component(axis, *best).value()))
                best = c;
        }
        if (!best)
            return {};
        previous = best;
    }
    components[axis] = *previous;
    return source.find_components(components);
}
bool same_member(const PooledLineSubscriber& a, const PooledLineSubscriber& b)
{
    return a.resident == b.resident && a.support_authority == b.support_authority && a.support == b.support &&
           a.emission == b.emission && a.image == b.image && a.edge == b.edge && a.first == b.first &&
           a.last == b.last && a.source_generation == b.source_generation && a.native_source == b.native_source;
}
} // namespace

struct CertifiedIncidenceResultView::RemoteValue
{
    std::shared_ptr<const ContactSourcePointSnapshot> snapshot;
    std::shared_ptr<const PooledLineCompletion::Proof> proof;
    std::array<std::size_t, 3> components{};
    std::size_t first = 0, last = 0, count = 0;
    unsigned axis = 0;
    bool endpoints = false;
    std::vector<std::uint64_t> mask;
    Credits leases;
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    std::size_t charge = 0;
    const ControlCredit* control = nullptr;
    ~RemoteValue()
    {
        snapshot.reset();
        proof.reset();
        decltype(mask){}.swap(mask);
        decltype(leases){}.swap(leases);
        if (budget)
            budget->used -= charge;
    }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge + control_bytes; }
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const RemoteValue>& owner, std::size_t bytes) const
    {
        return NativeContactFundingAccess::split_named(owner, bytes, [](const RemoteValue& value) { return value.charge; },
            [](const RemoteValue& value) { return value.control; });
    }
    [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept
    {
        return {control, sizeof(ControlCredit)};
    }
};
CertifiedIncidenceResultView::CertifiedIncidenceResultView(std::shared_ptr<const RemoteValue> remote)
    : remote_(std::move(remote))
{
}
std::optional<CertifiedIncidenceResultView> CertifiedIncidenceResultView::certify_completed(
    const PooledLineCompletion& completion, std::size_t line, const PooledLineSubscriber& original,
    const PooledLineCompletionSnapshotAdmission& admission, std::span<const std::size_t> expected,
    const IncidencePools& pools, ContactHaloContext context, unsigned layer, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena)
{
    const auto& proof = completion.proof();
    const auto& source = admission.snapshot_;
    if (admission.point_pool_ != &pools.points || source != pools.points.source())
        return {};
    if (!proof || !source || expected.empty() || !source->slot_namespace() || proof->context() != context ||
        proof->layer() != layer || !original.native_source || !same_member(proof->original_request(), original) ||
        !std::ranges::equal(proof->line_key(), pools.lines.get(line)) ||
        !std::ranges::equal(proof->support_key(), pools.supports.get(original.support)) ||
        completion.result().size() != expected.size())
        return {};
    MetadataEnvironment environment;
    // The one existing reply owner provides the actual output identities via
    // the stable typed pool. Slot equality alone is never a certificate.
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (completion.result()[i] != expected[i] || expected[i] >= source->size() ||
            !full_equal(pools.points.get(completion.result()[i]), source->value({expected[i]})))
            return {};
    if (original.first >= source->size() || original.last >= source->size() ||
        !full_equal(proof->endpoint(0), source->value({original.first})) ||
        !full_equal(proof->endpoint(1), source->value({original.last})))
        return {};
    const bool endpoints = expected.size() == 2 && expected[0] == original.first && expected[1] == original.last;
    std::array<std::size_t, 3> components{};
    std::size_t first = 0, last = 0;
    unsigned axis = 3;
    bool interval = true;
    if (!endpoints)
    {
        const auto front = source->component_indices({expected.front()});
        const auto back = source->component_indices({expected.back()});
        if (!front || !back || expected.size() < 2)
            return {};
        components = *front;
        for (unsigned a = 0; a < 3; ++a)
            if ((*front)[a] != (*back)[a])
            {
                if (axis != 3)
                    return {};
                axis = a;
            }
        if (axis == 3)
            return {};
        first = (*front)[axis];
        last = (*back)[axis];
        const bool ascending =
            source->canonical_component(axis, first).value() < source->canonical_component(axis, last).value();
        std::size_t previous = first;
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            const auto selected = source->component_indices({expected[i]});
            if (!selected)
                return {};
            for (unsigned a = 0; a < 3; ++a)
                if (a != axis && (*selected)[a] != components[a])
                    return {};
            const auto scalar = source->canonical_component(axis, (*selected)[axis]).value();
            const auto before = source->canonical_component(axis, previous).value();
            if (i && (ascending ? scalar <= before : scalar >= before))
                return {};
            previous = (*selected)[axis];
            const auto candidate = actual_axis_slot(*source, components, axis, first, last, {}, i);
            interval = interval && candidate && same_slot(*source, *candidate, *source, expected[i]);
        }
        if (actual_axis_slot(*source, components, axis, first, last, {}, expected.size()))
            interval = false;
    }
    cartesian_detail::check_keeper(budget, keeper);
    if (!arena || arena->budget != &budget)
        throw std::invalid_argument("Completed incidence values require their explicit destination arena.");
    cartesian_detail::check_keeper(budget, arena->keeper);
    if (!keeper)
        keeper = arena->keeper;
    cartesian_detail::CreditConstruction credits(arena);
    Allocation imported(budget), borrowed_output(budget);
    Credits leases;
    // Foreign proof, source values and temporary reply DATA are admitted before
    // allocating any candidate descriptor, control or component selection mask.
    credits.retain(
        proof, proof->allocation_budget(), proof->metadata_bytes(), proof->has_arena_keeper(), leases, imported);
    retain_snapshot(credits, source, leases, imported);
    if (completion.allocation_budget() != &budget)
    {
        if (!completion.allocation_budget() || !completion.has_arena_keeper())
            throw std::invalid_argument("Foreign completed output has no lifetime keeper.");
        // A retained Layer/fork may already admit this exact immutable output
        // through the shared arena. Only genuinely borrowed construction data
        // needs an additional temporary scope, never a second body credit.
        if (!arena->admits_owner(&completion, completion.metadata_bytes()))
            borrowed_output.retain(completion.metadata_bytes());
    }
    Allocation body(budget);
    body.retain(sizeof(RemoteValue));
    auto [result, control] = make_controlled(budget, keeper, [] { return new RemoteValue; });
    result->control = control.get();
    result->snapshot = source;
    result->proof = proof;
    result->components = components;
    result->axis = axis;
    result->first = first;
    result->last = last;
    result->count = expected.size();
    result->endpoints = endpoints;
    if (!endpoints && !interval)
    {
        const auto component_count = source->component_counts()[axis];
        const auto words = component_count / 64 + (component_count % 64 != 0);
        body.reserve(result->mask, words);
        result->mask.assign(words, 0);
        for (const auto slot : expected)
        {
            const auto c = (*source->component_indices({slot}))[axis];
            result->mask[c / 64] |= std::uint64_t(1) << (c % 64);
        }
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            const auto candidate = actual_axis_slot(*source, components, axis, first, last, result->mask, i);
            if (!candidate || !same_slot(*source, *candidate, *source, expected[i]))
                return {};
        }
        if (actual_axis_slot(*source, components, axis, first, last, result->mask, expected.size()))
            return {};
    }
    result->leases.swap(leases);
    result->keeper = std::move(keeper);
    result->charge = cartesian_detail::add(body.bytes, imported.bytes);
    body.bytes = imported.bytes = 0;
    result->budget = &budget;
    credits.committed = true;
    return CertifiedIncidenceResultView(std::move(result));
}
const PooledLineSubscriber* CertifiedIncidenceResultView::original_request() const noexcept
{
    return remote_ ? &remote_->proof->original_request() : nullptr;
}
std::shared_ptr<const PooledLineCompletion::Proof> CertifiedIncidenceResultView::completion_proof() const noexcept
{
    return remote_ ? remote_->proof : nullptr;
}
bool CertifiedIncidenceResultView::imports_owner(const void* id) const noexcept
{
    if (recipe_)
        return id == recipe_.get() || id == recipe_->catalog().get() || id == snapshot().get() ||
               recipe_->imports_owner(id);
    if (id == remote_.get() || id == remote_->proof.get() || id == remote_->snapshot.get())
        return true;
    bool found = false;
    remote_->snapshot->visit_dependencies(
        [&](const auto& owner, const Budget*, std::size_t, bool) { found = found || id == owner.get(); });
    for (const auto& lease : remote_->leases)
        found = found || id == lease.get();
    return found;
}
CertifiedIncidenceResultView::CertifiedIncidenceResultView(
    std::shared_ptr<const CartesianSourceLineRangeRecipe> recipe, std::size_t query)
    : recipe_(std::move(recipe)), query_(query)
{
}
std::optional<CertifiedIncidenceResultView> CertifiedIncidenceResultView::certify(
    std::shared_ptr<const CartesianSourceLineRangeRecipe> recipe, std::size_t query,
    std::span<const std::size_t> expected, const ContactSourcePointSnapshot& current)
{
    if (!recipe || query >= recipe->size() || expected.empty())
        return {};
    const auto& source = recipe->captured_point_snapshot();
    if (!source || !same_namespace(*source, current) || recipe->captured_result_size(query) != expected.size())
        return {};
    MetadataEnvironment environment;
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (!same_slot(*source, recipe->captured_result_at(query, i), current, expected[i]))
            return {};
    return CertifiedIncidenceResultView(std::move(recipe), query);
}
std::size_t CertifiedIncidenceResultView::size() const
{
    return remote_ ? remote_->count : recipe_->captured_result_size(query_);
}
std::size_t CertifiedIncidenceResultView::operator[](std::size_t i) const
{
    if (i >= size())
        throw std::out_of_range("Invalid certified incidence result index.");
    if (!remote_)
        return recipe_->captured_result_at(query_, i);
    if (remote_->endpoints)
        return i ? remote_->proof->original_request().last : remote_->proof->original_request().first;
    MetadataEnvironment environment;
    const auto slot = actual_axis_slot(
        *remote_->snapshot, remote_->components, remote_->axis, remote_->first, remote_->last, remote_->mask, i);
    if (!slot)
        throw std::logic_error("Immutable completed incidence value selector changed.");
    return *slot;
}
ContactSourcePoint CertifiedIncidenceResultView::value(std::size_t i) const
{
    return snapshot()->value({(*this)[i]});
}
const std::shared_ptr<const CartesianSourceLineRangeRecipe>& CertifiedIncidenceResultView::recipe() const noexcept
{
    return recipe_;
}
const std::shared_ptr<const ContactSourcePointSnapshot>& CertifiedIncidenceResultView::snapshot() const noexcept
{
    return remote_ ? remote_->snapshot : recipe_->captured_point_snapshot();
}
std::size_t CertifiedIncidenceResultView::query_ordinal() const noexcept
{
    return query_;
}
void CertifiedIncidenceResultView::retain_dependencies(
    cartesian_detail::CreditConstruction& credits, Credits& leases, Allocation& memory) const
{
    if (recipe_)
    {
        if (credits.descend(recipe_, recipe_->allocation_budget(), recipe_->metadata_bytes(),
                recipe_->has_arena_keeper(), leases, memory))
            recipe_->retain_dependencies(credits, leases, memory);
        return;
    }
    if (!credits.descend(remote_, remote_->budget, remote_->metadata_bytes(), bool(remote_->keeper), leases, memory))
        return;
    credits.retain(remote_->proof, remote_->proof->allocation_budget(), remote_->proof->metadata_bytes(),
        remote_->proof->has_arena_keeper(), leases, memory);
    retain_snapshot(credits, remote_->snapshot, leases, memory);
    for (const auto& lease : remote_->leases)
        retain_owner(credits, lease, lease->budget, lease->metadata, bool(lease->keeper), leases, memory);
}
void CertifiedIncidenceResultView::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    if (recipe_)
    {
        recipe_->visit_borrowed_payload(visitor);
        return;
    }
    contact_payload_object(
        visitor, remote_.get(), ContactPayloadRole::CertifiedIncidenceResultValueObject, sizeof(RemoteValue), true);
    contact_payload_object(
        visitor, remote_->control, ContactPayloadRole::CartesianControlCreditObject, sizeof(ControlCredit), true);
    contact_payload_buffer(
        visitor, remote_.get(), ContactPayloadRole::CertifiedIncidenceResultValueMask, remote_->mask);
    contact_payload_buffer(visitor, remote_.get(), ContactPayloadRole::CertifiedIncidenceLoopLeases, remote_->leases);
    remote_->proof->visit_borrowed_payload(visitor);
    remote_->snapshot->visit_borrowed_payload(visitor);
    for (const auto& lease : remote_->leases)
        lease->visit_borrowed_payload(visitor);
}

struct CertifiedIncidenceResultParts::Storage
{
    std::vector<CertifiedIncidenceResultView> parts;
    Credits leases;
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    std::size_t charge = 0;
    const ControlCredit* control = nullptr;
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const Storage>& owner, std::size_t bytes) const
    {
        return NativeContactFundingAccess::split_named(owner, bytes, [](const Storage& value) { return value.charge; },
            [](const Storage& value) { return value.control; });
    }
    [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept
    {
        return {control, sizeof(ControlCredit)};
    }
    ~Storage()
    {
        decltype(parts){}.swap(parts);
        decltype(leases){}.swap(leases);
        if (budget)
            budget->used -= charge;
    }
};
CertifiedIncidenceResultParts::CertifiedIncidenceResultParts(std::shared_ptr<const Storage> value)
    : storage_(std::move(value))
{
}
CertifiedIncidenceResultParts CertifiedIncidenceResultParts::create(std::span<const CertifiedIncidenceResultView> parts,
    Budget& budget, std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena)
{
    cartesian_detail::check_keeper(budget, keeper);
    if (!arena || arena->budget != &budget)
        throw std::invalid_argument("Certified edge parts require their explicit destination arena.");
    cartesian_detail::check_keeper(budget, arena->keeper);
    if (!keeper)
        keeper = arena->keeper;
    cartesian_detail::CreditConstruction credits(arena);
    Allocation imported(budget);
    Credits leases;
    for (const auto& part : parts)
        part.retain_dependencies(credits, leases, imported);
    Allocation body(budget);
    body.retain(sizeof(Storage));
    auto [result, control] = make_controlled(budget, keeper, [] { return new Storage; });
    result->control = control.get();
    body.reserve(result->parts, parts.size());
    result->parts.assign(parts.begin(), parts.end());
    result->leases.swap(leases);
    result->keeper = std::move(keeper);
    result->charge = cartesian_detail::add(body.bytes, imported.bytes);
    body.bytes = imported.bytes = 0;
    result->budget = &budget;
    credits.committed = true;
    return CertifiedIncidenceResultParts(std::move(result));
}
std::span<const CertifiedIncidenceResultView> CertifiedIncidenceResultParts::parts() const noexcept
{
    return storage_->parts;
}
std::size_t CertifiedIncidenceResultParts::capacity() const noexcept
{
    return storage_->parts.capacity();
}
std::size_t CertifiedIncidenceResultParts::lease_capacity() const noexcept
{
    return storage_->leases.capacity();
}
std::size_t CertifiedIncidenceResultParts::metadata_bytes() const noexcept
{
    return storage_->charge + control_bytes;
}
std::size_t CertifiedIncidenceResultParts::known_payload_bytes() const noexcept
{
    return sizeof(Storage) + sizeof(ControlCredit) + capacity() * sizeof(CertifiedIncidenceResultView) +
           lease_capacity() * sizeof(Credits::value_type);
}
const void* CertifiedIncidenceResultParts::storage_identity() const noexcept
{
    return storage_.get();
}
bool CertifiedIncidenceResultParts::imports_owner(const void* id) const noexcept
{
    if (id == storage_.get())
        return true;
    for (const auto& part : storage_->parts)
        if (part.imports_owner(id))
            return true;
    for (const auto& lease : storage_->leases)
        if (id == lease.get())
            return true;
    return false;
}
void CertifiedIncidenceResultParts::retain_dependencies(
    cartesian_detail::CreditConstruction& credits, Credits& leases, Allocation& memory) const
{
    if (!credits.descend(storage_, storage_->budget, metadata_bytes(), bool(storage_->keeper), leases, memory))
        return;
    for (const auto& part : storage_->parts)
        part.retain_dependencies(credits, leases, memory);
    for (const auto& lease : storage_->leases)
        retain_owner(credits, lease, lease->budget, lease->metadata, bool(lease->keeper), leases, memory);
}
void CertifiedIncidenceResultParts::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    contact_payload_object(
        visitor, storage_.get(), ContactPayloadRole::CertifiedIncidenceResultPartsObject, sizeof(Storage), true);
    contact_payload_object(
        visitor, storage_->control, ContactPayloadRole::CartesianControlCreditObject, sizeof(ControlCredit), true);
    contact_payload_buffer(visitor, storage_.get(), ContactPayloadRole::CertifiedIncidenceLoopParts, storage_->parts);
    contact_payload_buffer(visitor, storage_.get(), ContactPayloadRole::CertifiedIncidenceLoopLeases, storage_->leases);
    for (const auto& part : storage_->parts)
        part.visit_borrowed_payload(visitor);
    for (const auto& lease : storage_->leases)
        lease->visit_borrowed_payload(visitor);
}

struct CertifiedIncidenceLoopView::Storage
{
    std::shared_ptr<const ContactSourcePointSnapshot> snapshot;
    std::uint64_t quad = 0;
    std::optional<CertifiedIncidenceResultParts> parts;
    Credits leases;
    std::size_t count = 0, charge = 0;
    std::size_t first_part = 0, part_count = 0;
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    const ControlCredit* control = nullptr; // borrowed from this body's deleter
    ~Storage()
    {
        snapshot.reset();
        parts.reset();
        decltype(leases){}.swap(leases);
        if (budget)
            budget->used -= charge;
    }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return charge + control_bytes; }
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const Storage>& owner, std::size_t bytes) const
    {
        return NativeContactFundingAccess::split_named(owner, bytes, [](const Storage& value) { return value.charge; },
            [](const Storage& value) { return value.control; });
    }
    [[nodiscard]] std::pair<const void*, std::size_t> borrowed_control_payload() const noexcept
    {
        return {control, sizeof(ControlCredit)};
    }
};
CertifiedIncidenceLoopView::CertifiedIncidenceLoopView(std::shared_ptr<const Storage> storage)
    : storage_(std::move(storage))
{
}
CertifiedIncidenceLoopView CertifiedIncidenceLoopView::publish(
    std::shared_ptr<const ContactSourcePointSnapshot> snapshot, std::uint64_t quad,
    std::optional<CertifiedIncidenceResultParts> parts, std::size_t count, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena, std::size_t first_part,
    std::size_t part_count)
{
    if (parts && (first_part > parts->parts().size() || part_count > parts->parts().size() - first_part))
        throw std::invalid_argument("Invalid certified incidence parts slice.");
    if (parts && !part_count)
        part_count = parts->parts().size() - first_part;
    cartesian_detail::check_keeper(budget, keeper);
    if (!arena || arena->budget != &budget)
        throw std::invalid_argument("Certified incidence loops require their explicit destination arena.");
    cartesian_detail::check_keeper(budget, arena->keeper);
    if (!keeper)
        keeper = arena->keeper;
    cartesian_detail::CreditConstruction credits(arena);
    Allocation imported(budget);
    Credits leases;
    // All foreign-owner admission precedes candidate descriptor/control/vector
    // allocation. No source state is modified; failure rolls back new credits.
    if (snapshot)
        retain_snapshot(credits, snapshot, leases, imported);
    if (parts)
        parts->retain_dependencies(credits, leases, imported);
    Allocation body(budget);
    body.retain(sizeof(Storage));
    auto [result, control] = make_controlled(budget, keeper, [] { return new Storage; });
    result->control = control.get();
    result->snapshot = std::move(snapshot);
    result->quad = quad;
    result->count = count;
    result->parts = std::move(parts);
    result->first_part = first_part;
    result->part_count = part_count;
    result->leases.swap(leases);
    result->keeper = std::move(keeper);
    result->charge = cartesian_detail::add(body.bytes, imported.bytes);
    body.bytes = imported.bytes = 0;
    result->budget = &budget;
    credits.committed = true;
    return CertifiedIncidenceLoopView(std::move(result));
}
std::optional<CertifiedIncidenceLoopView> CertifiedIncidenceLoopView::certify_quad(
    std::shared_ptr<const ContactSourcePointSnapshot> source, std::span<const std::size_t> expected, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena)
{
    if (!source || expected.size() != 4 || !source->slot_namespace())
        return {};
    MetadataEnvironment environment;
    std::array<std::array<std::size_t, 3>, 4> values{};
    for (unsigned p = 0; p < 4; ++p)
    {
        if (expected[p] >= source->size())
            return {};
        const auto components = source->component_indices({expected[p]});
        if (!components)
            return {};
        values[p] = *components;
    }
    unsigned a = 3, b = 3;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        if (values[0][axis] != values[1][axis])
        {
            if (a != 3)
                return {};
            a = axis;
        }
        if (values[0][axis] != values[3][axis])
        {
            if (b != 3)
                return {};
            b = axis;
        }
    }
    if (a == 3 || b == 3 || a == b)
        return {};
    auto opposite = values[0];
    opposite[a] = values[1][a];
    opposite[b] = values[3][b];
    if (opposite != values[2])
        return {};
    const std::array<std::size_t, 5> selected{values[0][0], values[0][1], values[0][2], values[1][a], values[3][b]};
    std::uint64_t code = 0;
    for (unsigned field = 0; field < 5; ++field)
    {
        if (selected[field] > 4095)
            return {};
        code |= std::uint64_t(selected[field]) << (12 * field);
    }
    code |= std::uint64_t(a) << 60;
    code |= std::uint64_t(b) << 62;
    for (unsigned p = 0; p < 4; ++p)
    {
        const auto slot = quad_slot(*source, code, p);
        if (!slot || !same_slot(*source, *slot, *source, expected[p]))
            return {};
    }
    return publish(std::move(source), code, {}, 4, budget, std::move(keeper), std::move(arena));
}
std::optional<CertifiedIncidenceLoopView> CertifiedIncidenceLoopView::certify_concatenation(
    std::span<const CertifiedIncidenceResultView> edges, std::span<const std::size_t> expected,
    const ContactSourcePointSnapshot& current, Budget& budget, std::shared_ptr<Budget> keeper,
    std::shared_ptr<CartesianAuthorityArena> arena)
{
    if (edges.empty() || expected.empty() || !current.slot_namespace())
        return {};
    MetadataEnvironment environment;
    std::size_t count = 0;
    for (const auto& edge : edges)
    {
        if (!same_namespace(*edge.snapshot(), current) || !edge.size())
            return {};
        const auto contribution = edge.size() - 1;
        if (contribution > expected.size() - count)
            return {};
        for (std::size_t i = 0; i < contribution; ++i)
            if (!same_slot(*edge.snapshot(), edge[i], current, expected[count + i]))
                return {};
        count += contribution;
    }
    if (count != expected.size())
        return {};
    auto parts = CertifiedIncidenceResultParts::create(edges, budget, keeper, arena);
    return publish({}, 0, std::move(parts), count, budget, std::move(keeper), std::move(arena));
}
std::optional<CertifiedIncidenceLoopView> CertifiedIncidenceLoopView::certify_concatenation(
    const CertifiedIncidenceResultParts& saved, std::span<const std::size_t> expected,
    const ContactSourcePointSnapshot& current, Budget& budget, std::shared_ptr<Budget> keeper,
    std::shared_ptr<CartesianAuthorityArena> arena)
{
    return certify_concatenation(
        saved, 0, saved.parts().size(), expected, current, budget, std::move(keeper), std::move(arena));
}
std::optional<CertifiedIncidenceLoopView> CertifiedIncidenceLoopView::certify_concatenation(
    const CertifiedIncidenceResultParts& saved, std::size_t first_part, std::size_t part_count,
    std::span<const std::size_t> expected, const ContactSourcePointSnapshot& current, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena)
{
    if (first_part > saved.parts().size() || part_count > saved.parts().size() - first_part || !part_count ||
        expected.empty() || !current.slot_namespace())
        return {};
    MetadataEnvironment environment;
    std::size_t count = 0;
    for (const auto& edge : saved.parts().subspan(first_part, part_count))
    {
        if (!same_namespace(*edge.snapshot(), current) || !edge.size())
            return {};
        const auto contribution = edge.size() - 1;
        if (contribution > expected.size() - count)
            return {};
        for (std::size_t i = 0; i < contribution; ++i)
            if (!same_slot(*edge.snapshot(), edge[i], current, expected[count + i]))
                return {};
        count += contribution;
    }
    if (count != expected.size())
        return {};
    return publish({}, 0, saved, count, budget, std::move(keeper), std::move(arena), first_part, part_count);
}
std::span<const CertifiedIncidenceResultView> CertifiedIncidenceLoopView::parts() const noexcept
{
    return storage_->parts ? storage_->parts->parts().subspan(storage_->first_part, storage_->part_count)
                           : std::span<const CertifiedIncidenceResultView>{};
}
bool CertifiedIncidenceLoopView::exact_replay(
    std::span<const std::size_t> expected, const ContactSourcePointSnapshot& current) const
{
    if (expected.size() != size())
        return false;
    if (storage_->snapshot && !same_namespace(*storage_->snapshot, current))
        return false;
    for (const auto& edge : parts())
        if (!same_namespace(*edge.snapshot(), current))
            return false;
    MetadataEnvironment environment;
    for (std::size_t i = 0; i < expected.size(); ++i)
        if ((*this)[i] != expected[i] || expected[i] >= current.size() ||
            !full_equal(value(i), current.value({expected[i]})))
            return false;
    return true;
}
const CertifiedIncidenceResultParts* CertifiedIncidenceLoopView::result_parts() const noexcept
{
    return storage_->parts ? &*storage_->parts : nullptr;
}
CertifiedIncidenceLoopView::Kind CertifiedIncidenceLoopView::kind() const noexcept
{
    return storage_->snapshot ? Kind::Quad : Kind::Concatenation;
}
std::size_t CertifiedIncidenceLoopView::size() const noexcept
{
    return storage_->count;
}
std::size_t CertifiedIncidenceLoopView::operator[](std::size_t i) const
{
    if (i >= size())
        throw std::out_of_range("Invalid certified incidence loop index.");
    if (storage_->snapshot)
    {
        const auto slot = quad_slot(*storage_->snapshot, storage_->quad, i);
        if (!slot)
            throw std::logic_error("Immutable certified incidence quad selector changed.");
        return *slot;
    }
    for (const auto& part : parts())
    {
        const auto count = part.size() - 1;
        if (i < count)
            return part[i];
        i -= count;
    }
    throw std::logic_error("Immutable certified incidence loop count changed.");
}
ContactSourcePoint CertifiedIncidenceLoopView::value(std::size_t i) const
{
    if (i >= size())
        throw std::out_of_range("Invalid certified incidence loop value index.");
    if (storage_->snapshot)
        return storage_->snapshot->value({(*this)[i]});
    for (const auto& part : parts())
    {
        const auto count = part.size() - 1;
        if (i < count)
            return part.value(i);
        i -= count;
    }
    throw std::logic_error("Immutable certified incidence loop count changed.");
}
std::size_t CertifiedIncidenceLoopView::part_count() const noexcept
{
    return parts().size();
}
std::size_t CertifiedIncidenceLoopView::part_capacity() const noexcept
{
    return storage_->parts ? storage_->parts->capacity() : 0;
}
std::size_t CertifiedIncidenceLoopView::lease_capacity() const noexcept
{
    return storage_->leases.capacity();
}
std::size_t CertifiedIncidenceLoopView::metadata_bytes() const noexcept
{
    return storage_->metadata_bytes();
}
std::size_t CertifiedIncidenceLoopView::known_payload_bytes() const noexcept
{
    return sizeof(Storage) + sizeof(ControlCredit) + storage_->leases.capacity() * sizeof(Credits::value_type);
}
const Budget* CertifiedIncidenceLoopView::allocation_budget() const noexcept
{
    return storage_->budget;
}
bool CertifiedIncidenceLoopView::has_arena_keeper() const noexcept
{
    return bool(storage_->keeper);
}
const void* CertifiedIncidenceLoopView::storage_identity() const noexcept
{
    return storage_.get();
}
std::pair<const void*, std::size_t> CertifiedIncidenceLoopView::borrowed_control_payload() const noexcept
{
    return storage_->borrowed_control_payload();
}
bool CertifiedIncidenceLoopView::imports_owner(const void* identity) const noexcept
{
    if (identity == storage_.get())
        return true;
    bool found = storage_->parts && storage_->parts->imports_owner(identity);
    if (storage_->snapshot)
    {
        found = found || identity == storage_->snapshot.get();
        storage_->snapshot->visit_dependencies(
            [&](const auto& owner, const Budget*, std::size_t, bool) { found = found || identity == owner.get(); });
    }
    for (const auto& part : parts())
        found = found || part.imports_owner(identity);
    for (const auto& lease : storage_->leases)
        found = found || identity == lease.get();
    return found;
}
void CertifiedIncidenceLoopView::retain_dependencies(
    cartesian_detail::CreditConstruction& credits, Credits& leases, Allocation& memory) const
{
    if (!credits.descend(storage_, storage_->budget, metadata_bytes(), bool(storage_->keeper), leases, memory))
        return;
    if (storage_->snapshot)
        retain_snapshot(credits, storage_->snapshot, leases, memory);
    if (storage_->parts)
        storage_->parts->retain_dependencies(credits, leases, memory);
    for (const auto& lease : storage_->leases)
        retain_owner(credits, lease, lease->budget, lease->metadata, bool(lease->keeper), leases, memory);
}
void CertifiedIncidenceLoopView::visit_borrowed_payload(
    const std::function<void(const ContactPayloadAllocation&)>& visitor) const
{
    contact_payload_object(
        visitor, storage_.get(), ContactPayloadRole::CertifiedIncidenceLoopObject, sizeof(Storage), true);
    contact_payload_object(
        visitor, storage_->control, ContactPayloadRole::CartesianControlCreditObject, sizeof(ControlCredit), true);
    if (storage_->parts)
        storage_->parts->visit_borrowed_payload(visitor);
    contact_payload_buffer(visitor, storage_.get(), ContactPayloadRole::CertifiedIncidenceLoopLeases, storage_->leases);
    if (storage_->snapshot)
        storage_->snapshot->visit_borrowed_payload(visitor);
    for (const auto& part : parts())
        part.visit_borrowed_payload(visitor);
    for (const auto& lease : storage_->leases)
        lease->visit_borrowed_payload(visitor);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
