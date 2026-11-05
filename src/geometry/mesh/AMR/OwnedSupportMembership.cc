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
class CachedEnvironment
{
    std::fenv_t caller_{};

public:
    explicit CachedEnvironment(const std::fenv_t& captured)
    {
        if (std::fegetenv(&caller_))
            throw std::invalid_argument("Cannot capture cached membership caller environment.");
        if (std::fesetenv(&captured))
        {
            (void) std::fesetenv(&caller_);
            throw std::invalid_argument("Cannot restore cached membership evaluation environment.");
        }
    }
    ~CachedEnvironment() { (void) std::fesetenv(&caller_); }
    template<class F> void in_caller(const F& invoke)
    {
        std::fenv_t cached{};
        if (std::fegetenv(&cached) || std::fesetenv(&caller_))
            throw std::invalid_argument("Cannot restore membership callback environment.");
        try
        {
            invoke();
        }
        catch (...)
        {
            (void) std::fegetenv(&caller_);
            (void) std::fesetenv(&cached);
            throw;
        }
        const auto captured = std::fegetenv(&caller_);
        const auto restored = std::fesetenv(&cached);
        if (captured || restored)
            throw std::invalid_argument("Cannot capture membership callback environment.");
    }
};
} // namespace
OwnedSupportMembership::OwnedSupportMembership(Budget& budget, IncidencePools& pools)
    : budget_(&budget), deferred_(budget), pools_(&pools)
{
}
OwnedSupportMembership::OwnedSupportMembership(const OwnedSupportMembership& source, Budget& budget,
    IncidencePools& pools, std::shared_ptr<CartesianAuthorityArena> source_arena)
    : budget_(&budget), deferred_(budget), pools_(&pools)
{
    if (source_arena && source_arena->budget != &budget)
        throw std::invalid_argument("Foreign owner-membership source arena.");
    cartesian_detail::Allocation source_memory(budget);
    std::optional<cartesian_detail::CreditConstruction> source_credit;
    if (source_arena)
        source_credit.emplace(source_arena);
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
        if (source.source_)
        {
            if (source.source_->allocation_budget() != &budget && !source.source_->has_arena_keeper())
                throw std::invalid_argument("Borrowed Cartesian membership source lacks its original arena keeper.");
            if (source.source_->allocation_budget() != &budget)
            {
                if (!source_credit)
                    throw std::invalid_argument("Owner-membership fork requires its explicit source arena.");
                source_credit->retain(source.source_, source.source_->allocation_budget(),
                    source.source_->metadata_bytes(), source.source_->has_arena_keeper(), source_leases_,
                    source_memory);
            }
        }
        if (source.inventory_)
        {
            if (!source_credit)
                throw std::invalid_argument("Source inventory fork requires its explicit shared arena.");
            source_credit->retain(source.inventory_, source.inventory_->allocation_budget(),
                source.inventory_->metadata_bytes(), source.inventory_->has_arena_keeper(), source_leases_,
                source_memory);
            source.inventory_->visit_dependencies(
                [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keep)
                {
                    using E = typename std::remove_cvref_t<decltype(owner)>::element_type;
                    const std::shared_ptr<const std::remove_const_t<E>> fixed = owner;
                    source_credit->retain(fixed, origin, bytes, keep, source_leases_, source_memory);
                });
        }
        admit(source.retired_source_cells_.size() * retired_node_bytes);
        cells_ = source.cells_;
        reverse_ = source.reverse_;
        source_ = source.source_;
        inventory_ = source.inventory_;
        retired_source_cells_ = source.retired_source_cells_;
        source_environment_ = source.source_environment_;
        charged_ = cartesian_detail::add(charged_, std::exchange(source_memory.bytes, 0));
        if (source_credit)
            source_credit->committed = true;
    }
    catch (...)
    {
        Index().swap(cells_);
        ReverseIndex().swap(reverse_);
        source_.reset();
        decltype(source_leases_){}.swap(source_leases_);
        std::set<Slot>().swap(retired_source_cells_);
        budget_->used -= charged_;
        charged_ = 0;
        throw;
    }
}
OwnedSupportMembership::~OwnedSupportMembership()
{
    Index().swap(cells_);
    ReverseIndex().swap(reverse_);
    source_.reset();
    inventory_.reset();
    decltype(source_leases_){}.swap(source_leases_);
    construction_source_.reset();
    std::set<Slot>().swap(retired_source_cells_);
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
    return add_leaf(leaf, nullptr);
}
OwnedSupportMembership::Slot OwnedSupportMembership::add(const CartesianNativeIncidenceCell& cell)
{
    const auto& source = cell.source();
    // An environment change preserves the original explicit result. It does
    // not grant the source's older evaluation facts authority over this cell.
    const bool authentic = source && source->allocation_budget() == budget_ &&
                           source->evaluation_environment() == cell.evaluation_environment() &&
                           (!construction_source_ || construction_source_ == source);
    const auto slot = add_leaf(cell.leaf(), authentic ? source.get() : nullptr);
    if (authentic && !construction_source_)
    {
        construction_source_ = source;
        construction_environment_ = cell.evaluation_state();
    }
    return slot;
}
OwnedSupportMembership::Slot OwnedSupportMembership::add_leaf(
    const AdaptiveLeaf& leaf, const CartesianIncidenceSource* proof)
{
    const auto key = pools_->cells.intern(leaf.key);
    if (cells_.contains(key) || source_cell(key))
        throw std::invalid_argument("Duplicate owner-local cell membership.");
    if (leaf.surfaces.size() > (std::numeric_limits<std::size_t>::max() - owner) / sizeof(Member))
        throw std::invalid_argument("Owner-local surface count overflow.");
    const auto bytes = owner + leaf.surfaces.size() * sizeof(Member);
    admit(bytes);
    admit(cell_node_bytes);
    auto cell = std::make_shared<Cell>();
    cell->revision = leaf.revision;
    cell->construction_source = proof;
    cell->charge = bytes;
    cell->members.reserve(leaf.surfaces.size());
    if (cell->members.capacity() != leaf.surfaces.size())
        throw std::invalid_argument("Owner surface capacity differs from admission.");
    for (std::size_t i = 0; i < leaf.surfaces.size(); ++i)
        cell->members.push_back({pools_->supports.intern(leaf.surfaces[i].key), support_surface_encoded_size(leaf, i)});
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
bool OwnedSupportMembership::has_complete_construction_proof() const
{
    bool complete = construction_source_ && !source_ && construction_source_->cell_count() == cells_.size();
    if (complete)
        for (const auto& [slot, cell] : cells_)
        {
            const auto& key = pools_->cells.get(slot);
            const auto* ancestry = std::get_if<CartesianCellAncestry>(&key.ancestry);
            complete = complete && cell->construction_source == construction_source_.get() && ancestry &&
                       key.region == construction_source_->region() && construction_source_->contains(*ancestry) &&
                       cell->revision == construction_source_->revision(*ancestry) && cell->members.size() == 6;
            if (!complete)
                break;
        }
    return complete;
}
bool OwnedSupportMembership::publish_constructed_source()
{
    if (!has_complete_construction_proof())
    {
        construction_source_.reset();
        return false;
    }
    while (!cells_.empty())
        erase(cells_.begin()->first);
    source_ = std::move(construction_source_);
    source_environment_ = construction_environment_;
    return true;
}
void OwnedSupportMembership::publish_inventory(
    std::shared_ptr<const CartesianSourceInventory> inventory, std::shared_ptr<CartesianAuthorityArena> arena)
{
    if (!inventory || !arena || arena->budget != budget_)
        throw std::invalid_argument("Owner source inventory requires its exact shared arena.");
    cartesian_detail::Allocation memory(*budget_);
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases;
    cartesian_detail::CreditConstruction credits(arena);
    credits.retain(inventory, inventory->allocation_budget(), inventory->metadata_bytes(),
        inventory->has_arena_keeper(), leases, memory);
    inventory->visit_dependencies(
        [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keep)
        {
            using E = typename std::remove_cvref_t<decltype(owner)>::element_type;
            const std::shared_ptr<const std::remove_const_t<E>> fixed = owner;
            credits.retain(fixed, origin, bytes, keep, leases, memory);
        });
    for (const auto& [slot, cell] : cells_)
        if (const auto* batch = inventory->lookup(pools_->cells.get(slot)))
            if (cell->construction_source != batch->source().get() || cell->members.size() != 6 ||
                cell->revision != inventory->revision(pools_->cells.get(slot)))
                throw std::invalid_argument("Uncertified explicit owner member cannot enter source inventory.");
    for (auto it = cells_.begin(); it != cells_.end();)
    {
        const auto slot = it->first;
        ++it;
        if (inventory->contains(pools_->cells.get(slot)))
            erase(slot);
    }
    const auto previous = source_leases_.capacity() * sizeof(source_leases_[0]);
    source_leases_.swap(leases);
    decltype(leases){}.swap(leases);
    release(previous);
    charged_ = cartesian_detail::add(charged_, std::exchange(memory.bytes, 0));
    release(retired_source_cells_.size() * retired_node_bytes);
    decltype(retired_source_cells_){}.swap(retired_source_cells_);
    inventory_ = std::move(inventory);
    source_ = inventory_->batches().empty() ? nullptr : inventory_->batches().front().source();
    if (source_)
        source_environment_ = inventory_->batches().front().evaluation_state();
    construction_source_.reset();
    credits.committed = true;
}
const CartesianSourceInventory::Batch* OwnedSupportMembership::source_batch(Slot slot) const
{
    return inventory_ && !retired_source_cells_.contains(slot) ? inventory_->lookup(pools_->cells.get(slot)) : nullptr;
}
const CartesianCellAncestry* OwnedSupportMembership::source_cell(Slot slot) const
{
    if (inventory_)
        return source_batch(slot) ? std::get_if<CartesianCellAncestry>(&pools_->cells.get(slot).ancestry) : nullptr;
    if (!source_ || retired_source_cells_.contains(slot))
        return nullptr;
    const auto* cell = std::get_if<CartesianCellAncestry>(&pools_->cells.get(slot).ancestry);
    return cell && pools_->cells.get(slot).region == source_->region() && source_->contains(*cell) ? cell : nullptr;
}
std::array<OwnedSupportMembership::Member, 6> OwnedSupportMembership::generated_members(Slot slot) const
{
    const auto* cell = source_cell(slot);
    if (!cell)
        throw std::invalid_argument("Foreign procedural owner membership.");
    std::array<Member, 6> result{};
    std::size_t count = 0;
    const auto* batch = source_batch(slot);
    const auto& source = batch ? batch->source() : source_;
    CachedEnvironment environment(batch ? batch->evaluation_state() : source_environment_);
    source->visit_cell(*cell, *budget_,
        [&](const auto& occurrence)
        {
            if (occurrence.cell != pools_->cells.get(slot) || occurrence.ordinal != count || count == result.size())
                throw std::invalid_argument("Procedural owner membership order differs from certification.");
            const auto support = pools_->supports.find(occurrence.surface.key);
            if (!support)
                throw std::invalid_argument("Procedural owner membership identity differs from certification.");
            result[count++] = {*support, occurrence.packet.size()};
        });
    if (count != result.size())
        throw std::invalid_argument("Incomplete procedural owner membership.");
    return result;
}
std::size_t OwnedSupportMembership::member_count(Slot slot) const
{
    const auto explicit_cell = cells_.find(slot);
    if (explicit_cell != cells_.end())
        return explicit_cell->second->members.size();
    if (source_cell(slot))
        return 6;
    throw std::invalid_argument("Unknown owner-local cell membership.");
}
std::uint64_t OwnedSupportMembership::leaf_revision(Slot slot) const
{
    const auto explicit_cell = cells_.find(slot);
    if (explicit_cell != cells_.end())
        return explicit_cell->second->revision;
    if (const auto* cell = source_cell(slot))
        return source_batch(slot) ? source_batch(slot)->source()->revision(*cell) : source_->revision(*cell);
    throw std::invalid_argument("Unknown owner-local cell membership.");
}
void OwnedSupportMembership::visit_members(
    Slot slot, const CartesianBorrowedCallback<std::size_t, const Member&>& emit) const
{
    const auto explicit_cell = cells_.find(slot);
    if (explicit_cell != cells_.end())
    {
        for (std::size_t i = 0; i < explicit_cell->second->members.size(); ++i)
            emit(i, explicit_cell->second->members[i]);
        return;
    }
    const auto members = generated_members(slot);
    for (std::size_t i = 0; i < members.size(); ++i)
        emit(i, members[i]);
}
void OwnedSupportMembership::visit_support_members(
    Slot support, const CartesianBorrowedCallback<Slot, std::uint64_t, std::size_t, const Member&>& emit) const
{
    const auto visit = [&](const std::shared_ptr<const CartesianIncidenceSource>& source, const std::fenv_t& captured,
                           const CartesianSourceInventory::Batch* batch)
    {
        // Cached values are regenerated in their captured environment. Invoke
        // user callbacks after restoring their environment, one bounded cell
        // at a time; no full support/member cache is retained.
        // The source visitor already filters the complete key. Its callback
        // must not escape a borrowed occurrence or normalize its geometry.
        CachedEnvironment environment(captured);
        source->visit_surfaces(
            *budget_,
            [&](const auto& occurrence)
            {
                const auto cell = pools_->cells.find(occurrence.cell);
                const auto* ancestry = std::get_if<CartesianCellAncestry>(&occurrence.cell.ancestry);
                if (!cell || cells_.contains(*cell) || retired_source_cells_.contains(*cell) ||
                    (batch && (!ancestry || !batch->contains(*ancestry))))
                    return;
                const Member member{support, occurrence.packet.size()};
                environment.in_caller([&] { emit(*cell, occurrence.revision, occurrence.ordinal, member); });
            },
            pools_->supports.get(support));
    };
    if (inventory_)
        for (const auto& batch : inventory_->batches())
            visit(batch.source(), batch.evaluation_state(), &batch);
    else if (source_)
        visit(source_, source_environment_, nullptr);
    const auto* cells = support_cells(support);
    if (cells)
        for (const auto cell : *cells)
            for (std::size_t i = 0; i < at(cell).members.size(); ++i)
                if (at(cell).members[i].support == support)
                    emit(cell, at(cell).revision, i, at(cell).members[i]);
}
bool OwnedSupportMembership::has_support(Slot support) const
{
    bool present = false;
    visit_support_members(support, [&](auto, auto, auto, const auto&) { present = true; });
    return present;
}
std::size_t OwnedSupportMembership::maximum_change_capacity() const
{
    if (source_ && source_->domain().size() > std::numeric_limits<std::size_t>::max() - 64)
        throw std::invalid_argument("Cartesian registry cursor capacity overflows.");
    std::size_t result = source_ ? 64 + source_->domain().size() : 1;
    if (inventory_)
        for (const auto& batch : inventory_->batches())
            result = std::max(result, cartesian_detail::add(64, batch.source()->domain().size()));
    for (const auto& [slot, cell] : cells_)
        for (const auto& member : cell->members)
        {
            const auto ancestry = key_capacity(pools_->cells.get(slot));
            const auto key = pools_->supports.get(member.support).size();
            if (key > std::numeric_limits<std::size_t>::max() - ancestry)
                throw std::invalid_argument("Owner registry cursor capacity overflows.");
            result = std::max(result, key + ancestry);
        }
    return result;
}
void OwnedSupportMembership::certify_source(
    std::shared_ptr<const CartesianIncidenceSource> source, std::shared_ptr<CartesianAuthorityArena> source_arena)
{
    if (!source || source_ || source->cell_count() != cells_.size() ||
        source->evaluation_environment() != cartesian_evaluation_environment())
        throw std::invalid_argument("Cartesian owner membership source is incomplete or has stale evaluation facts.");
    if (source->allocation_budget() != budget_ && !source->has_arena_keeper())
        throw std::invalid_argument("Foreign Cartesian membership arena lacks a lifetime keeper.");
    if ((source_arena && source_arena->budget != budget_) || (source->allocation_budget() != budget_ && !source_arena))
        throw std::invalid_argument("Foreign Cartesian replay certification requires its explicit shared arena.");
    cartesian_detail::Allocation source_memory(*budget_);
    std::optional<cartesian_detail::CreditConstruction> source_credit;
    if (source_arena)
        source_credit.emplace(source_arena);
    std::fenv_t captured{};
    if (std::fegetenv(&captured))
        throw std::invalid_argument("Cannot capture certified membership environment.");
    std::size_t replayed = 0;
    source->visit_surfaces(*budget_,
        [&](const auto& occurrence)
        {
            const auto cell = pools_->cells.find(occurrence.cell);
            if (!cell || !cells_.contains(*cell))
                throw std::invalid_argument("Source replay contains foreign owner membership.");
            const auto& expected = at(*cell);
            if (expected.revision != occurrence.revision || expected.members.size() != 6 || occurrence.ordinal >= 6)
                throw std::invalid_argument("Source replay differs from owner member revision/order.");
            const auto& member = expected.members[occurrence.ordinal];
            if (pools_->supports.get(member.support) != occurrence.surface.key ||
                member.encoded_size != occurrence.packet.size())
                throw std::invalid_argument("Source replay differs from full owner support identity or packet size.");
            if (replayed == std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument("Source replay member count overflows.");
            ++replayed;
        });
    if (source->cell_count() > std::numeric_limits<std::size_t>::max() / 6 || replayed != source->cell_count() * 6)
        throw std::invalid_argument("Incomplete Cartesian owner membership replay.");
    if (source->allocation_budget() != budget_)
        source_credit->retain(source, source->allocation_budget(), source->metadata_bytes(), source->has_arena_keeper(),
            source_leases_, source_memory);
    while (!cells_.empty())
        erase(cells_.begin()->first);
    source_ = std::move(source);
    source_environment_ = captured;
    charged_ = cartesian_detail::add(charged_, std::exchange(source_memory.bytes, 0));
    if (source_credit)
        source_credit->committed = true;
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
    if (!slot || leaf_revision(*slot) != revision)
        throw std::invalid_argument("Stale owner-local cell retirement.");
    return *slot;
}
RegistryChange OwnedSupportMembership::change(Slot slot, std::size_t index, int rank, bool remove) const
{
    if (!cells_.contains(slot))
    {
        const auto members = generated_members(slot);
        if (index >= members.size())
            throw std::invalid_argument("Invalid procedural owner-local surface ordinal.");
        const auto& member = members[index];
        return {pools_->supports.get(member.support),
            {pools_->cells.get(slot), leaf_revision(slot), index, member.encoded_size, rank}, remove};
    }
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
        throw std::invalid_argument(
            remove ? "Unknown reverse support membership." : "Duplicate reverse support membership.");
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
        { return remove       ? previous[i < index ? i : i + 1]
                 : i == index ? cell
                              : previous[i < index ? i : i - 1]; });
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
    {
        if (!source_cell(slot))
            throw std::invalid_argument("Unknown owner-local cell retirement.");
        admit(retired_node_bytes);
        retired_source_cells_.insert(slot);
        return;
    }
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
