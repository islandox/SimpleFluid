#include "OwnedSupportMembership.hh"
#include "SupportSurfaceCodec.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::size_t owner = 128;
}
OwnedSupportMembership::OwnedSupportMembership(Budget& budget, IncidencePools& pools)
    : budget_(&budget), deferred_(budget), pools_(&pools)
{
}
OwnedSupportMembership::OwnedSupportMembership(
    const OwnedSupportMembership& source, Budget& budget, IncidencePools& pools)
    : budget_(&budget), deferred_(budget), pools_(&pools)
{
    try
    {
        for (const auto& [slot, cell] : source.cells_)
        {
            admit(cell_node_bytes);
            admit(cell->charge);
        }
        for (const auto& [slot, reverse] : source.reverse_)
        {
            admit(reverse_node_bytes);
            if (reverse.spill())
                admit(owner + reverse.size() * sizeof(Slot));
        }
        cells_ = source.cells_;
        reverse_ = source.reverse_;
    }
    catch (...)
    {
        Index().swap(cells_);
        ReverseIndex().swap(reverse_);
        budget_->used -= charged_;
        charged_ = 0;
        throw;
    }
}
OwnedSupportMembership::~OwnedSupportMembership()
{
    Index().swap(cells_);
    ReverseIndex().swap(reverse_);
    budget_->used -= charged_;
}
void OwnedSupportMembership::admit(std::size_t n)
{
    if (n > std::numeric_limits<std::size_t>::max() - charged_)
        throw std::invalid_argument("Owned support charge overflow.");
    budget_->add(n);
    charged_ += n;
}
void OwnedSupportMembership::release(std::size_t n)
{
    budget_->used -= n;
    charged_ -= n;
}
OwnedSupportMembership::Slot OwnedSupportMembership::add(const AdaptiveLeaf& leaf)
{
    const auto key = pools_->cells.intern(leaf.key);
    if (cells_.contains(key))
        throw std::invalid_argument("Duplicate owner-local cell membership.");
    if (leaf.surfaces.size() > (std::numeric_limits<std::size_t>::max() - owner) / sizeof(Member))
        throw std::invalid_argument("Owner-local surface count overflow.");
    const auto bytes = owner + leaf.surfaces.size() * sizeof(Member);
    admit(bytes);
    admit(cell_node_bytes);
    auto cell = std::make_shared<Cell>();
    cell->revision = leaf.revision;
    cell->charge = bytes;
    cell->members.reserve(leaf.surfaces.size());
    if (cell->members.capacity() != leaf.surfaces.size())
        throw std::invalid_argument("Owner surface capacity differs from admission.");
    for (std::size_t i = 0; i < leaf.surfaces.size(); ++i)
        cell->members.push_back(
            {pools_->supports.intern(leaf.surfaces[i].key), support_surface_encoded_size(leaf, i)});
    for (std::size_t i = 0; i < cell->members.size(); ++i)
    {
        const auto support = cell->members[i].support;
        if (std::none_of(cell->members.begin(), cell->members.begin() + i,
                [&](const Member& previous) { return previous.support == support; }))
            update_reverse(support, key, false);
    }
    cells_.emplace(key, std::move(cell));
    return key;
}
const OwnedSupportMembership::Cell& OwnedSupportMembership::at(Slot slot) const
{
    const auto found = cells_.find(slot);
    if (found == cells_.end())
        throw std::invalid_argument("Unknown owner-local cell membership.");
    return *found->second;
}
OwnedSupportMembership::Slot OwnedSupportMembership::require(const AdaptiveCellKey& key, std::uint64_t revision) const
{
    const auto slot = pools_->cells.find(key);
    if (!slot || at(*slot).revision != revision)
        throw std::invalid_argument("Stale owner-local cell retirement.");
    return *slot;
}
RegistryChange OwnedSupportMembership::change(Slot slot, std::size_t index, int rank, bool remove) const
{
    const auto& cell = at(slot);
    if (index >= cell.members.size())
        throw std::invalid_argument("Invalid owner-local surface ordinal.");
    const auto& member = cell.members[index];
    return {pools_->supports.get(member.support),
        {pools_->cells.get(slot), cell.revision, index, member.encoded_size, rank}, remove};
}
const OwnedSupportMembership::SupportCells* OwnedSupportMembership::support_cells(Slot support) const
{
    const auto found = reverse_.find(support);
    return found == reverse_.end() ? nullptr : &found->second;
}
void OwnedSupportMembership::update_reverse(Slot support, Slot cell, bool remove)
{
    auto found = reverse_.find(support);
    const SupportCells empty;
    const auto& previous = found == reverse_.end() ? empty : found->second;
    const auto count = previous.size();
    auto less = [&](Slot a, Slot b) { return pools_->cells.get(a) < pools_->cells.get(b); };
    const auto position = std::lower_bound(previous.begin(), previous.end(), cell, less);
    const auto index = static_cast<std::size_t>(position - previous.begin());
    const bool same = position != previous.end() && !less(*position, cell) && !less(cell, *position);
    if (remove ? !same : same)
        throw std::invalid_argument(remove ? "Unknown reverse support membership." : "Duplicate reverse support membership.");
    if (!remove && count == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Reverse support count overflow.");
    const auto next_count = remove ? count - 1 : count + 1;
    if (next_count > (std::numeric_limits<std::size_t>::max() - owner) / sizeof(Slot))
        throw std::invalid_argument("Reverse support capacity overflow.");
    const auto bytes = next_count > 2 ? owner + next_count * sizeof(Slot) : 0;
    if (bytes)
        admit(bytes);
    if (found == reverse_.end())
        admit(reverse_node_bytes);
    auto next = SupportCells::make(next_count, [&](std::size_t i)
        { return remove ? previous[i < index ? i : i + 1] : i == index ? cell : previous[i < index ? i : i - 1]; });
    auto old_spill = previous.spill();
    const auto old_charge = old_spill ? owner + count * sizeof(Slot) : 0;
    // The local alias above contributes one reference, in addition to this map.
    const bool old_pinned = old_spill && (old_spill.use_count() > 2 || deferred_.contains(old_spill));
    if (old_pinned && deferred_.adopt(old_spill, old_charge))
        charged_ -= old_charge;
    if (!next_count)
    {
        reverse_.erase(found);
        release(reverse_node_bytes);
    }
    else if (found == reverse_.end())
        reverse_.emplace(support, std::move(next));
    else
        found->second = std::move(next);
    old_spill.reset();
    if (old_charge && !old_pinned)
        release(old_charge);
}
void OwnedSupportMembership::erase(Slot slot)
{
    const auto found = cells_.find(slot);
    if (found == cells_.end())
        throw std::invalid_argument("Unknown owner-local cell retirement.");
    const auto bytes = found->second->charge;
    const bool pinned = found->second.use_count() > 1 || deferred_.contains(found->second);
    if (pinned && deferred_.adopt(found->second, bytes))
        charged_ -= bytes;
    const auto& members = found->second->members;
    for (std::size_t i = 0; i < members.size(); ++i)
        if (std::none_of(members.begin(), members.begin() + i,
                [&](const Member& previous) { return previous.support == members[i].support; }))
            update_reverse(members[i].support, slot, true);
    cells_.erase(found);
    if (!pinned)
        release(bytes);
    release(cell_node_bytes);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
