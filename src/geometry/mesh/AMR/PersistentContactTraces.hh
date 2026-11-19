/** @file PersistentContactTraces.hh
 * @brief Immutable ordered trace inventories shared by production contact bodies.
 */
#pragma once
#include "ContactPhysicalResidency.hh"
#include "ContactSlotIndex.hh"
#include "DistributedAdaptiveClosureRecords.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class PersistentContactTraces
{
    static constexpr auto absent = std::numeric_limits<std::size_t>::max();
    using Inventory = std::vector<AdaptiveTraceFace>;
    struct Owner
    {
        ContactGroupAllocation admission;
        Inventory values;
        explicit Owner(ContactGroupAllocation memory) : admission(std::move(memory)) {}
    };
    struct Slot
    {
        std::shared_ptr<const Owner> owner;
        std::size_t references = 0, next_free = absent;
        std::uint64_t digest = 0;
    };

public:
    struct Adoption
    {
        std::size_t transferred = 0, slot = absent;
    };
    PersistentContactTraces(std::shared_ptr<Budget> arena, ContactPhysicalResidency& residency)
        : arena_(std::move(arena)), residency_(residency), slots_(arena_), index_(arena_) {}
    PersistentContactTraces(const PersistentContactTraces& source, std::shared_ptr<Budget> arena,
        ContactPhysicalResidency& residency)
        : arena_(std::move(arena)), residency_(residency), slots_(source.slots_, arena_),
          index_(source.index_, arena_), free_(source.free_) {}
    Adoption adopt(ContactBody& record)
    {
        if (record.shared_traces)
            throw std::invalid_argument("Contact trace inventory was already adopted.");
        if (record.traces.empty() ||
            std::ranges::any_of(record.traces, [](const auto& face) { return bool(face.face); }))
            return {};
        const auto digest = hash(record.traces);
        const auto bytes = trace_bytes(record.traces);
        const auto found = index_.find(digest, [&](std::size_t id)
            {
                return std::ranges::equal(slots_[id].owner->values, record.traces,
                    [](const auto& a, const auto& b)
                    { return a.stem == b.stem && a.depth == b.depth && a.split == b.split && a.terminal == b.terminal; });
            });
        if (found)
        {
            auto& entry = slots_.edit(*found);
            if (entry.references == absent)
                throw std::invalid_argument("Contact trace reference count overflows.");
            ++entry.references;
            Inventory().swap(record.traces);
            arena_->used -= bytes;
            record.shared_traces = std::shared_ptr<const Inventory>(entry.owner, &entry.owner->values);
            return {bytes, *found};
        }
        const auto id = free_ == absent ? slots_.size() : free_;
        auto& entry = slots_.edit(id);
        ContactGroupAllocation charge(arena_);
        charge.add(sizeof(Owner) + 128);
        auto owner = geometry_detail::domain_shared<geometry_detail::AllocationDomain::ContactHalo, Owner>(
            std::move(charge));
        owner->admission.adopt_precharged(bytes);
        owner->values.swap(record.traces);
        residency_.retain(ContactPhysicalResidency::Role::TraceInventory, owner, owner->admission.bytes(), arena_.get());
        free_ = entry.next_free;
        entry = {owner, 1, absent, digest};
        index_.insert(id, [&](std::size_t i) { return slots_[i].digest; });
        record.shared_traces = std::shared_ptr<const Inventory>(owner, &owner->values);
        return {bytes, id};
    }
    void release(std::size_t id)
    {
        if (id == absent)
            return;
        auto& entry = slots_.edit(id);
        if (!entry.references)
            throw std::invalid_argument("Contact trace inventory is not active.");
        if (--entry.references)
            return;
        index_.erase(entry.digest, id);
        residency_.release(ContactPhysicalResidency::Role::TraceInventory, entry.owner);
        entry = {{}, 0, free_, 0};
        free_ = id;
    }
    [[nodiscard]] std::size_t copied_pages() const noexcept { return slots_.copied_pages() + index_.copied_pages(); }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept { return slots_.overlap_bytes() + index_.overlap_bytes(); }
    void finish_overlap() noexcept
    {
        slots_.finish_overlap();
        index_.finish_overlap();
    }
    template<class Visit> void visit_pages(Visit&& visit) const
    {
        slots_.visit_pages(visit);
        index_.visit_pages(visit);
    }
    template<class Visit> void visit_inventories(Visit&& visit) const
    {
        for (std::size_t id = 0; id < slots_.size(); ++id)
            if (const auto& owner = slots_[id].owner)
                visit(owner.get(), sizeof(Owner), owner->values);
    }

private:
    static std::size_t trace_bytes(const Inventory& values)
    {
        auto bytes = ContactGroupAllocation::product(values.capacity(), sizeof(AdaptiveTraceFace));
        for (const auto& value : values)
        {
            if (value.stem.capacity() > absent - bytes)
                throw std::invalid_argument("Contact trace inventory size overflows.");
            bytes += value.stem.capacity();
        }
        return bytes;
    }
    static std::uint64_t hash(std::span<const AdaptiveTraceFace> values)
    {
        std::uint64_t result = 14695981039346656037ULL;
        const auto byte = [&](std::uint8_t value) { result = (result ^ value) * 1099511628211ULL; };
        const auto word = [&](std::uint64_t value)
        {
            for (unsigned shift = 0; shift < 64; shift += 8)
                byte(static_cast<std::uint8_t>(value >> shift));
        };
        word(values.size());
        for (const auto& face : values)
        {
            word(face.stem.size());
            for (const auto value : face.stem)
                byte(static_cast<std::uint8_t>(value));
            word(face.depth);
            word(std::uint64_t(face.split));
            word(face.terminal);
        }
        return result;
    }

    std::shared_ptr<Budget> arena_;
    ContactPhysicalResidency& residency_;
    ContactSlotPages<Slot> slots_;
    ContactSlotIndex index_;
    std::size_t free_ = absent;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
