#pragma once
#include "geometry/mesh/AMR/CartesianIncidenceBudget.hh"
#include "geometry/mesh/AMR/ContactPayloadCensus.hh"
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
    Budget* budget;
    std::shared_ptr<Budget> keeper;
    std::weak_ptr<const void> owner;
    std::size_t payload, metadata;
    std::size_t charged_payload = payload;
    const Budget* origin = nullptr;
    bool credit_metadata = false;
    const void* owner_identity = nullptr; // observation only; ownership key remains the full weak control
    // The weak owner keeps its custom deleter alive. A borrowed typed-control
    // observation therefore remains valid even after that owner's body dies.
    const void* observed_control = nullptr;
    std::size_t observed_control_bytes = 0;
    ~AuthorityOwnerCredit() { budget->used -= charged_payload + metadata; }
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
    static constexpr std::size_t control_bytes = 128;
    Budget* budget;
    std::shared_ptr<Budget> keeper;
    Entries entries;
    std::size_t charge = 0;
    std::weak_ptr<const void> prepaid_policy;
    PrepaidQuery prepaid_query = nullptr;

    ArenaCredits(Budget& b, std::shared_ptr<Budget> k) : budget(&b), keeper(std::move(k)) {}
    ~ArenaCredits()
    {
        entries.clear();
        budget->used -= charge;
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
    }
    void reap() noexcept
    {
        for (auto at = entries.begin(); at != entries.end();)
            // A lease may outlive its observed payload. Keep its actual map
            // node until the final external lease releases the token, so the
            // accepted-arena overlap census includes that immutable metadata.
            if (at->first.expired() && at->second.use_count() == 1)
            {
                at = entries.erase(at);
                charge -= node_bytes;
                budget->used -= node_bytes;
            }
            else
                ++at;
    }
    /** Accepted-arena allocations that coexist with a candidate. Imported
     * immutable payload bodies are admitted by candidate owners separately.
     * Shared lease vectors cannot supply this census without duplicate credit. */
    [[nodiscard]] std::size_t private_metadata_bytes() const noexcept
    {
        auto result = charge;
        for (const auto& [owner, credit] : entries)
            result += credit->metadata;
        return result;
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
    using Observer = bool (*)(
        void*, const std::shared_ptr<const void>&, const Budget*, std::size_t, bool has_keeper, bool credit_metadata);
    std::shared_ptr<ArenaCredits> arena;
    Allocation scratch;
    std::vector<Entries::iterator> inserted;
    std::vector<AuthorityOwnerCredit*> recharged;
    void* observer_context = nullptr;
    Observer observer = nullptr;
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
        arena->reap();
    }
    /** Observe exact dependencies without creating a typed directory or leases.
     * The callback controls whether a newly observed owner needs descent. */
    CreditConstruction(Budget& budget, void* context, Observer visit)
        : scratch(budget), observer_context(context), observer(visit)
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
            arena->entries.erase(at);
            arena->charge -= ArenaCredits::node_bytes;
            arena->budget->used -= ArenaCredits::node_bytes;
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
            return observer(observer_context, owner, origin, bytes, has_keeper, credit_metadata);
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
            constexpr auto metadata = sizeof(AuthorityOwnerCredit) + ArenaCredits::control_bytes;
            const auto charged = arena->prepaid(owner, origin, bytes, credit_metadata) ? 0 : bytes;
            token.retain(add(charged, metadata));
            node.retain(ArenaCredits::node_bytes);
            auto value = std::make_shared<AuthorityOwnerCredit>(arena->budget, arena->keeper, key, bytes, metadata);
            value->charged_payload = charged;
            value->origin = origin;
            value->credit_metadata = credit_metadata;
            value->owner_identity = owner.get();
            if constexpr (requires { owner->borrowed_control_payload(); })
            {
                const auto control = owner->borrowed_control_payload();
                value->observed_control = control.first;
                value->observed_control_bytes = control.second;
            }
            token.bytes = 0;
            at = arena->entries.emplace(key, std::move(value)).first;
            arena->charge = add(arena->charge, node.bytes);
            node.bytes = 0;
            inserted.push_back(at);
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
