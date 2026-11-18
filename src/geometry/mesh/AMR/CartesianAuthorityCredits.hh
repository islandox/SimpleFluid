#pragma once
#include "ContactAllocationScope.hh"
#include "geometry/mesh/AMR/CartesianIncidenceBudget.hh"
#include "geometry/mesh/AMR/ContactPayloadCensus.hh"
#include "geometry/mesh/AMR/NativeContactFunding.hh"
#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail::cartesian_detail
{
/** Immutable lease. Geometry ownership remains weak here; the sealed object
 * holding this lease independently owns its exact immutable dependencies.
 * Imported leases admit only metadata, never another arena's credit amount. */
struct AuthorityOwnerCredit
{
    static constexpr std::size_t control_bytes = 128;
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    std::weak_ptr<const void> owner;
    std::size_t payload = 0, metadata = 0;
    std::size_t charged_payload = payload;
    const Budget* origin = nullptr;
    bool credit_metadata = false;
    std::uint64_t arena_scope = 0;
    const void* owner_identity = nullptr; // observation only; ownership key remains the full weak control
    // The weak owner keeps its custom deleter alive. A borrowed typed-control
    // observation therefore remains valid even after that owner's body dies.
    const void* observed_control = nullptr;
    std::size_t observed_control_bytes = 0;
    PLIC::details::NativeWeakOwnerObservation target_control;
    ~AuthorityOwnerCredit()
    {
        if (budget) budget->used -= charged_payload + metadata - control_bytes;
    }
    [[nodiscard]] NativeContactOwnerFunding native_funding(
        const std::shared_ptr<const AuthorityOwnerCredit>& credit, std::size_t extent) const
    {
        auto funding = NativeContactFundingAccess::split_named<control_bytes>(credit, extent,
            [](const AuthorityOwnerCredit& value) { return value.metadata - control_bytes; },
            [](const AuthorityOwnerCredit& value) -> const auto& { return value.keeper; });
        funding.related_control = target_control;
        return funding;
    }
    static bool project(const std::shared_ptr<const AuthorityOwnerCredit>& credit,
        PLIC::details::GeometryOwnerGraphEditor& graph, bool retain)
    {
        if (!credit) return false;
        const auto bytes = credit->metadata;
        const auto funding = credit->native_funding(credit, bytes);
        return NativeContactFundingAccess::project(graph, credit, bytes, funding, retain);
    }
    /** Transfer an already admitted body to the shared owner allowance. */
    [[nodiscard]] std::size_t take_payload_charge() noexcept { return std::exchange(charged_payload, 0); }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        // owner stays weak: this census never promotes or visits its target.
        contact_payload_object(visitor, this, ContactPayloadRole::CartesianOwnerCreditObject, sizeof(*this), true);
        if (observed_control)
            contact_payload_object(visitor, observed_control, ContactPayloadRole::CartesianControlCreditObject,
                observed_control_bytes, true);
    }
};

/** A mutable directory belongs to an explicit budget domain, never to a seal.
 * Thus importing a seal cannot pin a directory that subsequently grows.
 * Entries observe physical expiry and do not keep geometry alive. */
struct ArenaCredits
{
    using Owner = std::weak_ptr<const void>;
    using PrepaidQuery = bool (*)(
        const void*, const std::shared_ptr<const void>&, const Budget*, std::size_t, bool credit_metadata) noexcept;
    using Entries = std::map<Owner, std::shared_ptr<AuthorityOwnerCredit>, std::owner_less<>>;
    static constexpr std::size_t node_bytes = sizeof(Entries::value_type) + 4 * sizeof(void*);
    static constexpr std::size_t control_bytes = AuthorityOwnerCredit::control_bytes;
    static constexpr std::size_t token_bytes = sizeof(AuthorityOwnerCredit) + control_bytes;
    Budget* budget;
    std::shared_ptr<Budget> keeper;
    std::shared_ptr<const geometry_detail::NativeAllocationState> native_index;
    Entries entries;
    using RetiredHints = std::map<const AuthorityOwnerCredit*, Entries::iterator>;
    static constexpr std::size_t hint_node_bytes = sizeof(RetiredHints::value_type) + 4 * sizeof(void*);
    RetiredHints retired_hints;
    std::size_t charge = 0;
    const std::uint64_t scope_id = next_contact_allocation_scope();
    std::weak_ptr<const void> prepaid_policy;
    PrepaidQuery prepaid_query = nullptr;

    ArenaCredits(Budget& b, std::shared_ptr<Budget> k) : budget(&b), keeper(std::move(k)) {}
    ~ArenaCredits()
    {
        const auto hints = retired_hints.size() * hint_node_bytes;
        retired_hints.clear();
        entries.clear();
        budget->used -= charge + hints;
    }
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor) const
    {
        contact_payload_object(visitor, this, ContactPayloadRole::CartesianArenaObject, sizeof(*this), true);
        for (const auto& entry : entries)
        {
            contact_payload_node_value(visitor, this, ContactPayloadRole::CartesianArenaEntries, entry);
            if (entry.second)
                entry.second->visit_borrowed_payload(visitor);
        }
        for (const auto& hint : retired_hints)
            contact_payload_node_value(visitor, this, ContactPayloadRole::CartesianArenaRetirementHints, hint);
    }
    void cancel_retirement(Entries::iterator at) noexcept
    {
        if (retired_hints.erase(at->second.get()))
            budget->used -= hint_node_bytes;
    }
    void cancel_retirement(const std::shared_ptr<const void>& owner) noexcept
    {
        if (const auto at = entries.find(Owner(owner)); at != entries.end())
            cancel_retirement(at);
    }
    bool hint_retirement(Entries::iterator at)
    {
        if (!prepaid_query || retired_hints.contains(at->second.get()))
            return false;
        budget->add(hint_node_bytes);
        try
        {
            retired_hints.emplace(at->second.get(), at);
        }
        catch (...)
        {
            budget->used -= hint_node_bytes;
            throw;
        }
        return true;
    }
    bool hint_retirement(const std::shared_ptr<const void>& owner)
    {
        const auto at = entries.find(Owner(owner));
        return at != entries.end() && hint_retirement(at);
    }
    void erase_entry(Entries::iterator at) noexcept
    {
        cancel_retirement(at);
        entries.erase(at);
        charge -= node_bytes;
        budget->used -= node_bytes;
    }
    void reap() noexcept
    {
        if (prepaid_query && !prepaid_policy.expired())
        {
            // Only edited or abandoned proofs enter this list. A stale
            // immutable control can remain alive through an escaped proof.
            for (auto hint = retired_hints.begin(); hint != retired_hints.end();)
            {
                const auto at = (hint++)->second;
                if (at->first.expired() && at->second.use_count() == 1)
                    erase_entry(at);
            }
            return;
        }
        for (auto at = entries.begin(); at != entries.end();)
        {
            const auto current = at++;
            if (current->first.expired() && current->second.use_count() == 1)
                erase_entry(current);
        }
    }
    [[nodiscard]] std::size_t private_metadata_bytes() const noexcept
    {
        return charge + entries.size() * token_bytes + retired_hints.size() * hint_node_bytes;
    }
    [[nodiscard]] bool owns_credit(const void* identity) const noexcept
    {
        for (const auto& [owner, credit] : entries)
            if (credit.get() == identity)
                return true;
        return false;
    }
    void set_prepaid_policy(std::weak_ptr<const void> policy, PrepaidQuery query) noexcept
    {
        prepaid_policy = std::move(policy);
        prepaid_query = query;
    }

    [[nodiscard]] bool prepaid(const std::shared_ptr<const void>& owner, const Budget* origin, std::size_t bytes,
        bool credit_metadata) const noexcept
    {
        const auto policy = prepaid_policy.lock();
        return policy && prepaid_query && prepaid_query(policy.get(), owner, origin, bytes, credit_metadata);
    }

    [[nodiscard]] std::shared_ptr<AuthorityOwnerCredit> find_paid_token(
        const std::shared_ptr<const void>& owner, const Budget* origin, std::size_t bytes) const noexcept
    {
        const auto at = entries.find(Owner(owner));
        if (at != entries.end() && at->second->owner_identity == owner.get() && at->second->origin == origin &&
            at->second->payload == bytes && at->second->charged_payload == bytes)
            return at->second;
        return {};
    }

    [[nodiscard]] bool admits_owner(const std::shared_ptr<const void>& owner, std::size_t bytes) const noexcept
    {
        const auto at = entries.find(Owner(owner));
        if (at == entries.end() || at->second->owner_identity != owner.get() || at->second->payload != bytes)
            return false;
        const auto& credit = *at->second;
        return credit.charged_payload == bytes || prepaid(owner, credit.origin, bytes, credit.credit_metadata);
    }
    [[nodiscard]] bool admits_owner(const void* identity, std::size_t bytes) const noexcept
    {
        for (const auto& [owner, credit] : entries)
            if (const auto actual = owner.lock();
                actual && actual.get() == identity && credit->payload == bytes &&
                (credit->charged_payload == bytes || prepaid(actual, credit->origin, bytes, credit->credit_metadata)))
                return true;
        return false;
    }
};

/** Construction transaction. New directory entries are removed on failure;
 * existing leases are immutable. Cumulative admission is never decremented. */
struct CreditConstruction
{
    using Entries = ArenaCredits::Entries;
    using Observer = bool (*)(void*, const std::shared_ptr<const void>&, const Budget*, std::size_t, bool has_keeper,
        bool credit_metadata, std::uint64_t owner_scope);
    std::shared_ptr<ArenaCredits> arena;
    Allocation scratch;
    std::vector<Entries::iterator> inserted;
    std::vector<AuthorityOwnerCredit*> recharged;
    void* observer_context = nullptr;
    Observer observer = nullptr;
    using FundingObserver = void (*)(void*, const NativeContactOwnerFunding&);
    FundingObserver funding_observer = nullptr;
    using Seen = std::map<ArenaCredits::Owner, bool, std::owner_less<>>;
    Seen seen;
    bool committed = false;
    void next_snapshot()
    {
        const auto bytes = product(seen.size(), sizeof(Seen::value_type) + 4 * sizeof(void*));
        seen.clear();
        scratch.bytes -= bytes;
        scratch.budget.used -= bytes;
    }

    explicit CreditConstruction(std::shared_ptr<ArenaCredits> a) : arena(std::move(a)), scratch(*arena->budget)
    {
        if (!arena->prepaid_query || arena->prepaid_policy.expired())
            arena->reap();
    }
    /** Observe exact dependencies without creating a typed directory or leases.
     * The callback controls whether a newly observed owner needs descent. */
    CreditConstruction(Budget& budget, void* context, Observer visit, FundingObserver funding = nullptr)
        : scratch(budget), observer_context(context), observer(visit), funding_observer(funding)
    {
        if (!observer)
            throw std::invalid_argument("Cartesian credit observation needs a visitor.");
    }
    ~CreditConstruction()
    {
        if (committed)
            return;
        for (auto* credit : recharged)
            arena->budget->used -= credit->take_payload_charge();
        for (auto at : inserted)
        {
            arena->erase_entry(at);
        }
    }
    template<class T>
    bool retain(const std::shared_ptr<const T>& owner, const Budget* origin, std::size_t bytes, bool has_keeper,
        std::vector<std::shared_ptr<AuthorityOwnerCredit>>& leases, Allocation& allocation,
        bool credit_metadata = false)
    {
        if (!owner)
            return false;
        credit_metadata |= std::is_same_v<std::remove_cv_t<T>, AuthorityOwnerCredit>;
        if (observer)
        {
            if (funding_observer)
            {
                const auto funding = NativeContactFundingAccess::capture(owner, bytes, origin);
                if (funding.original_bytes) funding_observer(observer_context, funding);
            }
            std::uint64_t scope = 0;
            if constexpr (std::is_same_v<std::remove_cv_t<T>, AuthorityOwnerCredit>)
                scope = owner->arena_scope;
            return observer(observer_context, owner, origin, bytes, has_keeper, credit_metadata, scope);
        }
        const ArenaCredits::Owner key(owner);
        if (seen.find(key) != seen.end())
            return false;
        scratch.retain(sizeof(Seen::value_type) + 4 * sizeof(void*));
        seen.emplace(key, true);
        if (origin == arena->budget)
            return true;
        if (!origin || !has_keeper)
            throw std::invalid_argument("Borrowed Cartesian owner has no source arena lifetime keeper.");
        auto at = arena->entries.find(key);
        if (at == arena->entries.end())
        {
            scratch.append(inserted);
            Allocation token(*arena->budget), node(*arena->budget);
            constexpr auto metadata = ArenaCredits::token_bytes;
            const bool prepaid = arena->prepaid(owner, origin, bytes, credit_metadata);
            const auto charged = prepaid ? 0 : bytes;
            token.retain(add(charged, metadata - ArenaCredits::control_bytes));
            node.retain(ArenaCredits::node_bytes);
            auto value = contact_funding_detail::make_controlled<ArenaCredits::control_bytes>(
                *arena->budget, arena->keeper, [] { return new AuthorityOwnerCredit; }).first;
            value->keeper = arena->keeper;
            value->owner = key;
            value->payload = bytes;
            value->metadata = metadata;
            value->charged_payload = charged;
            value->origin = origin;
            value->arena_scope = arena->scope_id;
            value->credit_metadata = credit_metadata;
            value->owner_identity = owner.get();
            if constexpr (requires { owner->borrowed_control_payload(); })
            {
                const auto control = owner->borrowed_control_payload();
                value->observed_control = control.first;
                value->observed_control_bytes = control.second;
            }
            const auto funding = NativeContactFundingAccess::capture(owner, bytes, origin);
            if (funding.original_bytes)
                value->target_control = funding.body.physical_observation();
            value->budget = arena->budget;
            token.bytes = 0;
            at = arena->entries.emplace(key, std::move(value)).first;
            arena->charge = add(arena->charge, node.bytes);
            node.bytes = 0;
            inserted.push_back(at);
            if (!prepaid)
                arena->hint_retirement(at);
        }
        else
        {
            auto& credit = *at->second;
            if (credit.payload != bytes || credit.origin != origin || credit.owner_identity != owner.get() ||
                credit.credit_metadata != credit_metadata)
                throw std::invalid_argument("Cartesian immutable owner binding changed after publication.");
            if (bytes && !credit.charged_payload && !arena->prepaid(owner, origin, bytes, credit_metadata))
            {
                scratch.append(recharged);
                arena->budget->add(bytes);
                credit.charged_payload = bytes;
                recharged.push_back(&credit);
            }
        }
        if (std::ranges::find(leases, at->second) == leases.end())
        {
            allocation.append(leases);
            leases.push_back(at->second);
        }
        return true;
    }

    template<class T>
    bool descend(const std::shared_ptr<const T>& owner, const Budget* origin, std::size_t bytes, bool has_keeper,
        std::vector<std::shared_ptr<AuthorityOwnerCredit>>& leases, Allocation& allocation,
        bool credit_metadata = false)
    {
        const bool first_use = retain(owner, origin, bytes, has_keeper, leases, allocation, credit_metadata);
        return observer ? first_use : static_cast<bool>(owner);
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail::cartesian_detail
