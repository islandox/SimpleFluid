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
PooledSupportRegistry::PooledSupportRegistry(Budget& budget, BytesPool& supports, CellPool& cells)
    : budget_(&budget), deferred_(budget), supports_(&supports), cells_(&cells), dirty_(SupportLess{&supports})
{
}
PooledSupportRegistry::PooledSupportRegistry(
    const PooledSupportRegistry& source, Budget& budget, BytesPool& supports, CellPool& cells)
    : budget_(&budget), deferred_(budget), supports_(&supports), cells_(&cells), dirty_(SupportLess{&supports})
{
    try
    {
        for (const auto& [slot, bucket] : source.groups_)
        {
            admit(group_node_bytes);
            if (bucket.members.spill())
                admit(bytes_for(bucket.members.size()));
        }
        for (const auto slot : source.dirty_)
            admit(dirty_node_bytes);
        groups_ = source.groups_;
        dirty_.insert(source.dirty_.begin(), source.dirty_.end());
    }
    catch (...)
    {
        Buckets().swap(groups_);
        dirty_.clear();
        budget_->used -= charged_;
        charged_ = 0;
        throw;
    }
}
PooledSupportRegistry::~PooledSupportRegistry()
{
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
        auto found = groups_.find(support);
        if (found == groups_.end())
        {
            admit(group_node_bytes);
            found = groups_.emplace(support, Group{}).first;
        }
        if (found->second.generation == std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("Support generation overflow.");
        auto& previous = found->second.members;
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
        if (next_charge)
            admit(next_charge);
        auto next = Members::make(count, [&](std::size_t i)
            { return change.remove ? previous[i < index ? i : i + 1] :
                       i == index ? member : previous[i < index ? i : i - 1]; });
        auto old_spill = previous.spill();
        const auto old_charge = old_spill ? bytes_for(old_count) : 0;
        const bool old_pinned = old_spill && (old_spill.use_count() > 2 || deferred_.contains(old_spill));
        if (old_pinned && deferred_.adopt(old_spill, old_charge))
            charged_ -= old_charge;
        previous = std::move(next);
        old_spill.reset();
        if (old_charge && !old_pinned)
            release(old_charge);
        ++found->second.generation;
        if (!dirty_.contains(support))
        {
            admit(dirty_node_bytes);
            dirty_.insert(support);
        }
    }
}
std::uint64_t PooledSupportRegistry::generation(Slot support) const
{
    const auto found = groups_.find(support);
    if (found == groups_.end())
        throw std::invalid_argument("Unknown support generation.");
    return found->second.generation;
}
const PooledSupportRegistry::Members* PooledSupportRegistry::find(Slot support) const
{
    const auto found = groups_.find(support);
    return found == groups_.end() || found->second.members.empty() ? nullptr : &found->second.members;
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
