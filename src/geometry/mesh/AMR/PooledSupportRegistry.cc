#include "PooledSupportRegistry.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::size_t bucket_base = 128;
std::size_t bytes_for(std::size_t count)
{
    if (count > (std::numeric_limits<std::size_t>::max() - bucket_base) / sizeof(PooledSupportMember))
        throw std::invalid_argument("Support bucket size overflow.");
    return count > 2 ? bucket_base + count * sizeof(PooledSupportMember) : 0;
}
} // namespace
PooledSupportRegistry::PooledSupportRegistry(
    Budget& budget, BytesPool& supports, CellPool& cells, std::shared_ptr<Budget> keeper)
    : PooledSupportRegistry(budget, supports, cells, std::move(keeper), true)
{
}
PooledSupportRegistry::PooledSupportRegistry(
    Budget& budget, BytesPool& supports, CellPool& cells, std::shared_ptr<Budget> keeper, bool use_persistent)
    : keeper_(std::move(keeper)), budget_(&budget), deferred_(budget), supports_(&supports), cells_(&cells),
      dirty_(SupportLess{&supports})
{
    if (keeper_ && keeper_.get() != &budget)
        throw std::invalid_argument("Support registry keeper does not own its budget.");
    if (!keeper_ || !use_persistent)
        return;
    admit(persistent_bytes);
    try
    {
        persistent_ = std::make_unique<Persistent>(keeper_);
    }
    catch (...)
    {
        release(persistent_bytes);
        throw;
    }
}
PooledSupportRegistry::PooledSupportRegistry(const PooledSupportRegistry& source, Budget& budget, BytesPool& supports,
    CellPool& cells, std::shared_ptr<Budget> keeper)
    : PooledSupportRegistry(budget, supports, cells, std::move(keeper), source.persistent_ && source.budget_ != &budget)
{
    try
    {
        if (persistent_)
        {
            const auto& root = source.persistent_->root;
            if (root)
                persistent_->imported.add(
                    persistent_ordered_detail::checked_sum(root->subtree_bytes, root->weights.total));
            persistent_->root = root;
            persistent_->overlapping = true;
        }
        else if (source.persistent_)
            (void) Index::visit_from(
                source.persistent_->root, [](const Row&) { return false; },
                [&](const Row& row)
                {
                    const auto* owner = row.group.spill_owner;
                    admit(persistent_ordered_detail::checked_sum(group_node_bytes, owner ? owner->charge.bytes() : 0));
                    groups_.emplace(row.slot, row.group);
                    return true;
                });
        else
        {
            for (const auto& [slot, bucket] : source.groups_)
                admit(group_node_bytes + bytes_for(bucket.members.size()));
            groups_ = source.groups_;
        }
        for (const auto slot : source.dirty_)
            admit(dirty_node_bytes);
        dirty_.insert(source.dirty_.begin(), source.dirty_.end());
    }
    catch (...)
    {
        persistent_.reset();
        Buckets().swap(groups_);
        dirty_.clear();
        budget_->used -= charged_;
        charged_ = 0;
        throw;
    }
}
PooledSupportRegistry::~PooledSupportRegistry()
{
    persistent_.reset();
    Buckets().swap(groups_);
    dirty_.clear();
    budget_->used -= charged_;
}
void PooledSupportRegistry::admit(std::size_t n)
{
    if (n > std::numeric_limits<std::size_t>::max() - charged_)
        throw std::invalid_argument("Support registry charge overflow.");
    budget_->add(n);
    charged_ += n;
}
void PooledSupportRegistry::release(std::size_t n)
{
    charged_ -= n;
    budget_->used -= n;
}
void PooledSupportRegistry::SpillWeights::add_row(Summary& sum, const Row& row, const Budget* arena)
{
    if (const auto* owner = row.group.spill_owner)
    {
        const auto bytes = owner->charge.bytes();
        sum.total = persistent_ordered_detail::checked_sum(sum.total, bytes);
        if (&owner->charge.arena() == arena)
            sum.local = persistent_ordered_detail::checked_sum(sum.local, bytes);
    }
}
void PooledSupportRegistry::SpillWeights::add_child(Summary& sum, const Summary& child, bool same_arena)
{
    sum.total = persistent_ordered_detail::checked_sum(sum.total, child.total);
    if (same_arena)
        sum.local = persistent_ordered_detail::checked_sum(sum.local, child.local);
}
std::size_t PooledSupportRegistry::inherited_bytes(const Root& root) const noexcept
{
    if (!root)
        return 0;
    const auto bytes = root->subtree_bytes + root->weights.total;
    return root->charge.arena.get() == budget_ ? bytes - root->same_arena_bytes - root->weights.local : bytes;
}
std::size_t PooledSupportRegistry::source_only_metadata_bytes() const noexcept
{
    return (persistent_ ? persistent_bytes : groups_.size() * group_node_bytes) + dirty_.size() * dirty_node_bytes +
           deferred_.metadata_bytes();
}
std::size_t PooledSupportRegistry::metadata_bytes() const noexcept
{
    const auto* root = persistent_ ? persistent_->root.get() : nullptr;
    return source_only_metadata_bytes() + (root ? root->subtree_bytes : 0);
}
std::size_t PooledSupportRegistry::retained_bytes() const noexcept
{
    if (!persistent_)
        return charged_ + deferred_.retained_bytes();
    const auto* root = persistent_->root.get();
    return metadata_bytes() + (root ? root->weights.total : 0) + overlap_bytes();
}
std::size_t PooledSupportRegistry::overlap_bytes() const noexcept
{
    return persistent_ ? persistent_->imported.bytes() - inherited_bytes(persistent_->root) : 0;
}
void PooledSupportRegistry::finish_overlap() noexcept
{
    if (persistent_)
    {
        persistent_->imported.release(overlap_bytes());
        persistent_->overlapping = false;
    }
}
const PooledSupportRegistry::Group* PooledSupportRegistry::find_group(Slot slot) const
{
    if (persistent_)
    {
        const auto* row = Index::lower_bound(persistent_->root, [&](const Row& value) { return value.slot < slot; });
        return row && row->slot == slot ? &row->group : nullptr;
    }
    const auto found = groups_.find(slot);
    return found == groups_.end() ? nullptr : &found->second;
}
void PooledSupportRegistry::replace_group(Slot slot, Group group, bool expected)
{
    const Row next{slot, std::move(group)};
    auto root = Index(persistent_->imported.keeper()).edit(persistent_->root, next, &next, expected);
    const auto inherited = inherited_bytes(root);
    if (inherited > persistent_->imported.bytes())
        throw std::logic_error("Support edit introduced an unadmitted foreign owner.");
    persistent_->root = std::move(root);
    if (!persistent_->overlapping)
        persistent_->imported.release(persistent_->imported.bytes() - inherited);
}
void PooledSupportRegistry::apply(std::span<const RegistryChange> changes)
{
    for (const auto& change : changes)
    {
        if (change.support.empty() || change.member.owner < 0)
            throw std::invalid_argument("Invalid support registry membership.");
        const auto old_support = supports_->find(change.support);
        const auto old_cell = cells_->find(change.member.cell);
        if (change.remove && (!old_support || !old_cell))
            throw std::invalid_argument("Unknown retired support member.");
        const auto support = old_support ? *old_support : supports_->intern(change.support);
        const auto cell = old_cell ? *old_cell : cells_->intern(change.member.cell);
        const auto* group = find_group(support);
        const Group empty;
        const auto& old = group ? *group : empty;
        if (old.generation == std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("Support generation overflow.");
        const auto& previous = old.members;
        const auto old_count = previous.size();
        const PooledSupportMember member{
            cell, change.member.revision, change.member.surface, change.member.encoded_size, change.member.owner};
        auto less = [&](const PooledSupportMember& a, const PooledSupportMember& b)
        {
            if (a.owner != b.owner)
                return a.owner < b.owner;
            const auto& ak = cells_->get(a.cell);
            const auto& bk = cells_->get(b.cell);
            if (ak < bk)
                return true;
            if (bk < ak)
                return false;
            return a.surface < b.surface;
        };
        const auto at = std::lower_bound(previous.begin(), previous.end(), member, less);
        const auto index = static_cast<std::size_t>(at - previous.begin());
        const bool equal = at != previous.end() && !less(member, *at) && !less(*at, member);
        if (change.remove)
        {
            if (!equal || at->revision != member.revision || at->encoded_size != member.encoded_size)
                throw std::invalid_argument("Stale retired support member.");
        }
        else if (equal)
            throw std::invalid_argument("Duplicate support registry member.");
        if (!change.remove && old_count == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Support membership capacity overflow.");
        const auto count = change.remove ? old_count - 1 : old_count + 1;
        const auto next_charge = bytes_for(count);
        const auto value = [&](std::size_t i)
        {
            return change.remove ? previous[i < index ? i : i + 1]
                   : i == index  ? member
                                 : previous[i < index ? i : i - 1];
        };
        if (persistent_)
        {
            static_assert(sizeof(OwnedSpill) + persistent_ordered_detail::control_bytes <= bucket_base);
            const OwnedSpill* spill_owner = nullptr;
            auto next = Members::make(count, value,
                [&]
                {
                    ContactGroupAllocation memory(persistent_->imported.keeper());
                    memory.add(next_charge);
                    auto spill = std::make_shared<OwnedSpill>(OwnedSpill{std::move(memory), {}});
                    spill_owner = spill.get();
                    auto* values = &spill->values;
                    return std::shared_ptr<Members::Spill>(std::move(spill), values);
                });
            const auto mark = !dirty_.contains(support);
            if (mark)
                admit(dirty_node_bytes);
            try
            {
                if (mark)
                    dirty_.insert(support);
                replace_group(support, {old.generation + 1, std::move(next), spill_owner}, group != nullptr);
            }
            catch (...)
            {
                if (mark)
                {
                    dirty_.erase(support);
                    release(dirty_node_bytes);
                }
                throw;
            }
        }
        else
        {
            if (!group)
                admit(group_node_bytes);
            if (next_charge)
                admit(next_charge);
            auto next = Members::make(count, value);
            auto old_spill = previous.spill();
            const auto old_charge = old_spill ? bytes_for(old_count) : 0;
            const bool old_pinned = old_spill && (old_spill.use_count() > 2 || deferred_.contains(old_spill));
            if (old_pinned && deferred_.adopt(old_spill, old_charge))
                charged_ -= old_charge;
            groups_.insert_or_assign(support, Group{old.generation + 1, std::move(next)});
            old_spill.reset();
            if (old_charge && !old_pinned)
                release(old_charge);
            if (!dirty_.contains(support))
            {
                admit(dirty_node_bytes);
                dirty_.insert(support);
            }
        }
    }
}
std::uint64_t PooledSupportRegistry::generation(Slot support) const
{
    const auto* found = find_group(support);
    if (!found)
        throw std::invalid_argument("Unknown support generation.");
    return found->generation;
}
const PooledSupportRegistry::Members* PooledSupportRegistry::find(Slot support) const
{
    const auto* found = find_group(support);
    return !found || found->members.empty() ? nullptr : &found->members;
}
RegistryMember PooledSupportRegistry::materialize(const PooledSupportMember& member) const
{
    return {cells_->get(member.cell), member.revision, member.surface, member.encoded_size, member.owner};
}
std::vector<PooledSupportRegistry::Slot> PooledSupportRegistry::dirty() const
{
    std::vector<Slot> result(dirty_.begin(), dirty_.end());
    std::sort(result.begin(), result.end(), [&](Slot a, Slot b) { return supports_->get(a) < supports_->get(b); });
    return result;
}
void PooledSupportRegistry::clear_dirty()
{
    const auto bytes = dirty_.size() * dirty_node_bytes;
    dirty_.clear();
    release(bytes);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
