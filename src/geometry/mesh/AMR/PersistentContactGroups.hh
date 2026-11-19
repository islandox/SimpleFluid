/** @file PersistentContactGroups.hh
 * @brief Exact copy-on-write normalized contact group identity and ownership.
 */
#pragma once
#include "ContactPointGroups.hh"
#include "PersistentContactFacts.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Mutable group metadata is private to one halo. Immutable fan blocks live
 * in query pages. Group slots do not retain a previous controller. */
class PersistentContactGroups
{
    static constexpr auto absent = std::numeric_limits<std::size_t>::max();
    struct Group
    {
        std::uint64_t generation = 1;
        std::size_t domain = absent, cover = absent, next_free = absent;
        ContactPointRef normalized{};
        bool live = false;
    };
    struct Domain
    {
        ContactGroupAllocation admission;
        AdaptiveBytes value;
        Domain(ContactGroupAllocation bytes, const AdaptiveBytes& source)
            : admission(std::move(bytes)), value(source)
        {
            if (value.capacity() != source.size())
                throw std::invalid_argument("Contact domain capacity differs from admission.");
        }
    };

    struct DomainSlot
    {
        std::shared_ptr<const Domain> owner;
        std::size_t references = 0, next_free = absent;
        std::uint64_t digest = 0;
    };

public:
    PersistentContactGroups(std::shared_ptr<Budget> arena, ContactPhysicalResidency& residency)
        : facts(arena, residency), residency_(residency), arena_(std::move(arena)), groups_(arena_), domains_(arena_),
          group_ids_(arena_), domain_ids_(arena_) {}
    PersistentContactGroups(const PersistentContactGroups& source, std::shared_ptr<Budget> arena,
        ContactPhysicalResidency& residency)
        : facts(source.facts, arena, residency), residency_(residency), arena_(std::move(arena)), groups_(source.groups_, arena_),
          domains_(source.domains_, arena_), group_ids_(source.group_ids_, arena_), domain_ids_(source.domain_ids_, arena_),
          free_(source.free_), free_domain_(source.free_domain_), live_(source.live_) {}

    [[nodiscard]] std::optional<ContactGroupId> find(ContactPointKeyView key) const { return find_probe(key); }
    [[nodiscard]] std::optional<ContactGroupId> find_encoded(std::span<const char> key) const { return find_probe(key); }
    [[nodiscard]] ContactPointKeyView key(ContactGroupId id) const { return key(checked(id)); }
    [[nodiscard]] const AdaptiveBytes& domain(ContactGroupId id) const { return domains_[checked(id).domain].owner->value; }
    [[nodiscard]] ContactPointIdentity normalized(ContactGroupId id) const { return facts.get(checked(id).normalized); }
    [[nodiscard]] ContactPointRef normalized_reference(ContactGroupId id) const { return checked(id).normalized; }
    [[nodiscard]] std::size_t cover(ContactGroupId id) const { return checked(id).cover; }
    void set_cover(ContactGroupId id, std::size_t cover)
    {
        (void) checked(id);
        groups_.edit(id.slot).cover = cover;
    }
    [[nodiscard]] ContactGroupId intern(const AdaptiveBytes& domain, const ContactPointIdentity& point,
        std::optional<ContactPointArray::Locator> locator, const PersistentContactFacts::PointPool* pool)
    {
        if (const auto old = find({domain, &point}))
            return *old;
        const auto domain_slot = intern_domain(domain);
        const auto normalized = locator ? facts.retain(*locator, pool) : facts.retain(point);
        if (facts.get(normalized) != point)
            throw std::invalid_argument("Normalized contact point locator does not match key.");
        const auto slot = free_ == absent ? groups_.size() : free_;
        auto& target = groups_.edit(slot);
        free_ = target.next_free;
        target = {target.generation, domain_slot, absent, absent, normalized, true};
        group_ids_.insert(slot, [&](std::size_t id) { return ContactPointWireHasher{}(key(groups_[id])); });
        ++live_;
        return {slot, target.generation};
    }
    void retire(ContactGroupId id)
    {
        const auto& before = checked(id);
        if (before.cover != absent)
            throw std::invalid_argument("Retired contact group still has a cover.");
        group_ids_.erase(ContactPointWireHasher{}(key(before)), id.slot);
        facts.release(before.normalized);
        retire_domain(before.domain);
        auto& target = groups_.edit(id.slot);
        target.live = false;
        if (target.generation != std::numeric_limits<std::uint64_t>::max())
        {
            ++target.generation;
            target.next_free = free_;
            free_ = id.slot;
        }
        --live_;
    }
    [[nodiscard]] std::size_t size() const noexcept { return live_; }
    [[nodiscard]] std::size_t slot_count() const noexcept { return groups_.size(); }
    [[nodiscard]] std::size_t storage_bytes() const noexcept
    {
        return groups_.storage_bytes() + domains_.storage_bytes() + group_ids_.storage_bytes() +
               domain_ids_.storage_bytes() + facts.storage_bytes();
    }
    /** Group ledger categories exclude query/work fans and bindings. */
    [[nodiscard]] ContactStorageBreakdown storage_breakdown() const
    {
        ContactStorageBreakdown result;
        result.group_metadata = groups_.storage_bytes();
        result.owner_index_metadata = group_ids_.storage_bytes();
        result.domain_metadata = domains_.storage_bytes() + domain_ids_.storage_bytes();
        ContactSet<const void*> owners;
        visit_domain_owners(
            [&](const auto& owner, const auto&, std::size_t bytes)
            {
                if (owners.insert(owner.get()).second)
                    result.domain_shared_body += bytes;
            });
        facts.contribute_breakdown(result);
        return result;
    }
    [[nodiscard]] std::size_t copied_pages() const noexcept
    {
        return groups_.copied_pages() + domains_.copied_pages() + group_ids_.copied_pages() +
               domain_ids_.copied_pages() + facts.copied_pages();
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept
    {
        return groups_.overlap_bytes() + domains_.overlap_bytes() + group_ids_.overlap_bytes() +
               domain_ids_.overlap_bytes() + facts.overlap_bytes();
    }
    void finish_overlap() noexcept
    {
        groups_.finish_overlap();
        domains_.finish_overlap();
        group_ids_.finish_overlap();
        domain_ids_.finish_overlap();
        facts.finish_overlap();
    }
    template<class Visit> void visit_pages(Visit&& visit) const
    {
        groups_.visit_pages(visit);
        domains_.visit_pages(visit);
        group_ids_.visit_pages(visit);
        domain_ids_.visit_pages(visit);
        facts.visit_pages(visit);
    }
    template<class Visit> void visit_ids(Visit&& visit) const
    {
        for (std::size_t slot = 0; slot < groups_.size(); ++slot)
            if (const auto& group = groups_[slot]; group.live)
                visit(ContactGroupId{slot, group.generation});
    }
    template<class Visit> void visit_groups(Visit&& visit) const
    {
        visit_ids(
            [&](ContactGroupId id)
            {
                const auto& group = groups_[id.slot];
                visit(&group, id, domains_[group.domain].owner->value, group.normalized, group.cover);
            });
    }
    template<class Visit> void visit_domain_owners(Visit&& visit) const
    {
        for (std::size_t slot = 0; slot < domains_.size(); ++slot)
        {
            if (const auto& domain = domains_[slot].owner)
                visit(domain, domain->value, domain->admission.bytes());
        }
    }

    PersistentContactFacts facts;

private:
    [[nodiscard]] const Group& checked(ContactGroupId id) const
    {
        const auto& group = groups_[id.slot];
        if (group.generation != id.generation || !group.live)
            throw std::out_of_range("Stale contact group ID.");
        return group;
    }
    [[nodiscard]] ContactPointKeyView key(const Group& group) const
    {
        return {domains_[group.domain].owner->value, facts.get(group.normalized)};
    }
    template<class Probe> [[nodiscard]] std::optional<ContactGroupId> find_probe(const Probe& probe) const
    {
        const auto slot = group_ids_.find(ContactPointWireHasher{}(probe),
            [&](std::size_t id)
            {
                const auto value = key(groups_[id]);
                const ContactPointWireLess less;
                return !less(value, probe) && !less(probe, value);
            });
        if (slot)
            return ContactGroupId{*slot, groups_[*slot].generation};
        return {};
    }
    [[nodiscard]] std::size_t intern_domain(const AdaptiveBytes& domain)
    {
        const auto hash = [](const AdaptiveBytes& bytes) { return ContactPointWireHasher{}(std::span(bytes)); };
        const auto digest = hash(domain);
        if (const auto existing = domain_ids_.find(digest,
                [&](std::size_t slot) { return domains_[slot].owner->value == domain; }))
        {
            auto& entry = domains_.edit(*existing);
            if (entry.references == absent)
                throw std::invalid_argument("Contact domain reference count overflows.");
            ++entry.references;
            return *existing;
        }
        ContactGroupAllocation charge(arena_);
        charge.add(sizeof(Domain) + 128 + domain.size());
        const auto slot = free_domain_ == absent ? domains_.size() : free_domain_;
        auto owner = geometry_detail::domain_shared<geometry_detail::AllocationDomain::ContactHalo, Domain>(std::move(charge), domain);
        residency_.retain(ContactPhysicalResidency::Role::GroupDomain, owner, owner->admission.bytes(), arena_.get());
        auto& entry = domains_.edit(slot);
        free_domain_ = entry.next_free;
        entry = {std::move(owner), 1, absent, digest};
        domain_ids_.insert(slot, [&](std::size_t id) { return domains_[id].digest; });
        return slot;
    }
    void retire_domain(std::size_t slot)
    {
        auto& entry = domains_.edit(slot);
        if (!entry.references)
            throw std::invalid_argument("Contact domain reference count underflows.");
        if (--entry.references)
            return;
        domain_ids_.erase(entry.digest, slot);
        residency_.release(ContactPhysicalResidency::Role::GroupDomain, entry.owner);
        entry = {{}, 0, free_domain_, 0};
        free_domain_ = slot;
    }

    ContactPhysicalResidency& residency_;
    std::shared_ptr<Budget> arena_;
    ContactSlotPages<Group> groups_;
    ContactSlotPages<DomainSlot> domains_;
    ContactSlotIndex group_ids_, domain_ids_;
    std::size_t free_ = absent, free_domain_ = absent, live_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
