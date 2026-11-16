#pragma once
#include "CompactSupportDirectory.hh"
#include "ContactGroupAllocation.hh"
#include "ContactPayloadCensus.hh"
#include "DeferredOwnerCharges.hh"
#include "IdentityPool.hh"
#include "PersistentOrderedPages.hh"
#include "SmallMembership.hh"
#include "SupportRegistry.hh"
#include <functional>
#include <map>
#include <memory>
#include <set>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Compact membership references immutable full identities through pools. */
struct PooledSupportMember
{
    std::size_t cell = 0;
    std::uint64_t revision = 0, surface = 0, encoded_size = 0;
    int owner = -1;
};
/** Keeper-backed forks share immutable group pages. Each overflow membership
 * body belongs to one support slot in a root, so its bytes are additive across
 * the ordered tree. Persistent forks require a fresh candidate arena.
 * Borrowed arenas keep flat metadata. */
class PooledSupportRegistry
{
public:
    using BytesPool = CompactSupportDirectory;
    using CellPool = IdentityPool<AdaptiveCellKey>;
    using Slot = std::size_t;
    using Members = SmallMembership<PooledSupportMember>;
    PooledSupportRegistry(Budget&, BytesPool&, CellPool&, std::shared_ptr<Budget> keeper = {});
    PooledSupportRegistry(
        const PooledSupportRegistry&, Budget&, BytesPool&, CellPool&, std::shared_ptr<Budget> keeper = {});
    ~PooledSupportRegistry();
    PooledSupportRegistry(const PooledSupportRegistry&) = delete;
    PooledSupportRegistry& operator=(const PooledSupportRegistry&) = delete;
    void apply(std::span<const RegistryChange>);
    /** The returned group is borrowed until the next edit. Its spill, if any,
     * has independent shared ownership. */
    [[nodiscard]] const Members* find(Slot support) const;
    [[nodiscard]] std::vector<Slot> dirty() const;
    [[nodiscard]] IncidenceSupportKey support_key(Slot slot) const { return supports_->get(slot); }
    [[nodiscard]] std::uint64_t generation(Slot support) const;
    [[nodiscard]] RegistryMember materialize(const PooledSupportMember&) const;
    [[nodiscard]] const AdaptiveCellKey& member_cell(const PooledSupportMember& member) const
    {
        return cells_->get(member.cell);
    }
    [[nodiscard]] std::size_t dirty_count() const noexcept { return dirty_.size(); }
    void for_each_dirty(const std::function<void(Slot)>& visit) const
    {
        for (const auto slot : dirty_)
            visit(slot);
    }
    [[nodiscard]] const auto& dirty_slots() const noexcept { return dirty_; }
    void clear_dirty();
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t source_only_metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t retained_bytes() const noexcept;
    [[nodiscard]] std::size_t overlap_bytes() const noexcept;
    /** Release imported allowances for source-only paths after the operation. */
    void finish_overlap() noexcept;
    void reap_deferred() noexcept { deferred_.reap(); }

    /** Borrow resident values only; pool targets are separate census roots. */
    template<class Visitor> void visit_borrowed_payload(Visitor&& visitor, bool include_object = false) const
    {
        if (include_object)
            contact_payload_object(visitor, this, ParallelSupportPayloadRole::SupportRegistryObject, sizeof(*this));
        const auto visit_group = [&](const Group& group)
        {
            const auto& spill = group.members.spill();
            if (spill)
            {
                const void* owner = group.spill_owner ? static_cast<const void*>(group.spill_owner) : spill.get();
                const auto object_bytes = group.spill_owner ? sizeof(OwnedSpill) : sizeof(Members::Spill);
                contact_payload_object(
                    visitor, owner, ParallelSupportPayloadRole::SupportMembershipSpill, object_bytes, true);
                contact_payload_buffer(visitor, owner, ParallelSupportPayloadRole::SupportMembershipValues, *spill);
            }
        };
        if (persistent_)
        {
            contact_payload_object(
                visitor, persistent_.get(), ParallelSupportPayloadRole::SupportRegistryObject, sizeof(Persistent));
            Index::visit_pages(persistent_->root,
                [&](const Root& node)
                {
                    contact_payload_object(visitor, node.get(), ParallelSupportPayloadRole::SupportRegistryGroupNodes,
                        sizeof(typename Index::Node), true);
                    contact_payload_buffer(
                        visitor, node.get(), ParallelSupportPayloadRole::SupportRegistryGroupNodes, node->rows);
                    contact_payload_buffer(
                        visitor, node.get(), ParallelSupportPayloadRole::SupportRegistryGroupNodes, node->children);
                    for (const auto& row : node->rows)
                        visit_group(row.group);
                });
        }
        else
            for (const auto& value : groups_)
            {
                contact_payload_node_value(visitor, this, ParallelSupportPayloadRole::SupportRegistryGroupNodes, value);
                visit_group(value.second);
            }
        for (const auto& value : dirty_)
            contact_payload_node_value(visitor, this, ParallelSupportPayloadRole::SupportRegistryDirtyNodes, value);
        deferred_.visit_borrowed_payload(visitor);
    }

private:
    PooledSupportRegistry(Budget&, BytesPool&, CellPool&, std::shared_ptr<Budget>, bool use_persistent);
    struct OwnedSpill
    {
        ContactGroupAllocation charge;
        Members::Spill values;
    };
    struct Group
    {
        std::uint64_t generation = 0;
        Members members;
        // The aliased vector in members pins this enclosing object.
        const OwnedSpill* spill_owner = nullptr;
    };
    struct Row
    {
        Slot slot;
        Group group;
    };
    struct Less
    {
        bool operator()(const Row& a, const Row& b) const noexcept { return a.slot < b.slot; }
    };
    struct SpillWeights
    {
        static constexpr bool enabled = true;
        struct Summary
        {
            std::size_t total = 0, local = 0;
        };
        static void add_row(Summary&, const Row&, const Budget*);
        static void add_child(Summary&, const Summary&, bool child_same_arena);
    };
    using Index = persistent_ordered_detail::OrderedPages<Row, Less, false, persistent_ordered_detail::OrderedPageWork,
        SpillWeights>;
    using Root = Index::Ptr;
    struct Persistent
    {
        explicit Persistent(std::shared_ptr<Budget> keeper) : imported(std::move(keeper)) {}
        ContactGroupAllocation imported;
        Root root;
        bool overlapping = false;
    };
    static constexpr auto persistent_bytes = sizeof(Persistent) + 128;
    [[nodiscard]] const Group* find_group(Slot) const;
    [[nodiscard]] std::size_t inherited_bytes(const Root&) const noexcept;
    void replace_group(Slot, Group, bool expected);
    using Buckets = std::map<Slot, Group>;
    static constexpr std::size_t group_node_bytes = sizeof(Buckets::value_type) + 4 * sizeof(void*);
    static constexpr std::size_t dirty_node_bytes = sizeof(std::set<Slot>::value_type) + 4 * sizeof(void*);
    void admit(std::size_t);
    void release(std::size_t);
    // The deferred credits and registry shell still use the arena after the
    // last immutable page and its credit have been destroyed.
    std::shared_ptr<Budget> keeper_;
    Budget* budget_;
    DeferredOwnerCharges deferred_;
    BytesPool* supports_;
    CellPool* cells_;
    std::size_t charged_ = 0;
    Buckets groups_;
    struct SupportLess
    {
        const BytesPool* pool;
        bool operator()(Slot a, Slot b) const { return pool->get(a) < pool->get(b); }
    };
    std::set<Slot, SupportLess> dirty_;
    std::unique_ptr<Persistent> persistent_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
