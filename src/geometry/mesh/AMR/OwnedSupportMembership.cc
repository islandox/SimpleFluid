#include "OwnedSupportMembership.hh"
#include "IncidencePhysicalOwners.hh"
#include "PersistentOrderedPages.hh"
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
template<class T> std::shared_ptr<T> make_membership_body(const std::shared_ptr<Budget>& arena, std::size_t bytes)
{
    using Charge = std::optional<ContactGroupAllocation>;
    static_assert(sizeof(T) + sizeof(Charge) + 4 * sizeof(void*) <= owner);
    ContactGroupAllocation charge(arena);
    charge.add(bytes);
    return std::shared_ptr<T>(new T,
        [charge = Charge(std::move(charge))](T* body) mutable
        {
            delete body;
            charge.reset();
        });
}
template<class T>
bool same_owner(const std::shared_ptr<const T>& first, const std::shared_ptr<const T>& second) noexcept
{
    return first.get() == second.get() && !first.owner_before(second) && !second.owner_before(first);
}
} // namespace

struct OwnedSupportMembership::Persistent
{
    using PageWork = persistent_ordered_detail::OrderedPageWork;
    struct CellRow
    {
        Slot slot = 0;
        std::shared_ptr<const Cell> cell;
        const Budget* origin = nullptr;
        [[nodiscard]] std::size_t bytes() const noexcept { return cell ? cell->charge : 0; }
        [[nodiscard]] std::size_t maximum() const noexcept { return cell ? cell->maximum_change : 0; }
    };
    struct ReverseRow
    {
        Slot slot = 0;
        SupportCells cells;
        const Budget* origin = nullptr;
        std::size_t spill_bytes = 0;
        [[nodiscard]] std::size_t bytes() const noexcept { return spill_bytes; }
        [[nodiscard]] std::size_t maximum() const noexcept { return 0; }
    };
    struct RetiredRow
    {
        Slot slot = 0;
        const Budget* origin = nullptr;
        [[nodiscard]] std::size_t bytes() const noexcept { return 0; }
        [[nodiscard]] std::size_t maximum() const noexcept { return 0; }
    };
    template<class Row> struct Less
    {
        bool operator()(const Row& a, const Row& b) const noexcept { return a.slot < b.slot; }
    };
    template<class Row> struct Weights
    {
        static constexpr bool enabled = true;
        struct Summary
        {
            std::size_t bytes = 0, local = 0, maximum = 0;
        };
        static void add_row(Summary& sum, const Row& row, const Budget* current)
        {
            sum.bytes = cartesian_detail::add(sum.bytes, row.bytes());
            if (row.origin == current)
                sum.local = cartesian_detail::add(sum.local, row.bytes());
            sum.maximum = std::max(sum.maximum, row.maximum());
        }
        static void add_child(Summary& sum, const Summary& child, bool local)
        {
            sum.bytes = cartesian_detail::add(sum.bytes, child.bytes);
            if (local)
                sum.local = cartesian_detail::add(sum.local, child.local);
            sum.maximum = std::max(sum.maximum, child.maximum);
        }
    };
    template<class Row>
    using Tree = persistent_ordered_detail::OrderedPages<Row, Less<Row>, false, PageWork, Weights<Row>>;
    template<class Row> using Root = typename Tree<Row>::Ptr;
    ContactGroupAllocation header, imported;
    Root<CellRow> cells;
    Root<ReverseRow> reverse;
    Root<RetiredRow> retired;
    bool keep_overlap = false;
    explicit Persistent(ContactGroupAllocation memory) : header(std::move(memory)), imported(header.keeper()) {}

    template<class RootType> static std::size_t total(const RootType& root) noexcept
    {
        return root ? root->subtree_bytes + root->weights.bytes : 0;
    }
    template<class RootType> std::size_t inherited(const RootType& root) const noexcept
    {
        if (!root)
            return 0;
        const auto bytes = total(root);
        return root->charge.arena.get() == &header.arena() ? bytes - root->same_arena_bytes - root->weights.local
                                                           : bytes;
    }
    template<class Row> static const Row* lower_bound(const Root<Row>& root, Slot slot) noexcept
    {
        return Tree<Row>::lower_bound(root, [slot](const Row& row) { return row.slot < slot; });
    }
    template<class Row> static const Row* find(const Root<Row>& root, Slot slot) noexcept
    {
        const auto* row = lower_bound<Row>(root, slot);
        return row && row->slot == slot ? row : nullptr;
    }
    template<class Row, class Visit> static void visit(const Root<Row>& root, const Visit& emit)
    {
        (void) Tree<Row>::visit_from(
            root, [](const Row&) { return false; },
            [&](const Row& row)
            {
                emit(row);
                return true;
            });
    }
    template<class RootType> static std::size_t size(const RootType& root) noexcept { return root ? root->size : 0; }
    template<class RootType> static std::size_t index_bytes(const RootType& root) noexcept
    {
        return root ? root->subtree_bytes : 0;
    }
    void fork_from(const Persistent& source)
    {
        if (&source.header.arena() == &header.arena())
            throw std::invalid_argument("Membership fork requires a distinct arena.");
        imported.add(cartesian_detail::add(
            cartesian_detail::add(total(source.cells), total(source.reverse)), total(source.retired)));
        keep_overlap = true;
        cells = source.cells;
        reverse = source.reverse;
        retired = source.retired;
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept
    {
        return imported.bytes() - inherited(cells) - inherited(reverse) - inherited(retired);
    }
    void finish_overlap() noexcept
    {
        imported.release(overlap_bytes());
        keep_overlap = false;
    }
    template<class RootType> void publish(RootType& root, RootType next)
    {
        const auto old = inherited(root), replacement = inherited(next);
        if (replacement > old)
            throw std::logic_error("Membership edit introduced unadmitted inherited storage.");
        root = std::move(next);
        if (!keep_overlap)
            imported.release(old - replacement);
    }
    template<class Row>
    void edit(Root<Row>& root, const Row& key, const Row* replacement, bool expected, PersistenceWork& work)
    {
        PageWork changed;
        auto next = Tree<Row>(header.keeper(), &changed).edit(root, key, replacement, expected);
        publish(root, std::move(next));
        work.copied_pages += changed.copied_pages;
        work.copied_bytes += changed.copied_bytes;
    }
};

OwnedSupportMembership::OwnedSupportMembership(
    Budget& budget, IncidencePools& pools, std::shared_ptr<Budget> keeper, ContactPhysicalResidency* residency)
    : keeper_(std::move(keeper)), budget_(&budget), deferred_(budget), pools_(&pools), residency_(residency)
{
    if ((keeper_ && keeper_.get() != &budget) || (!keeper_ && residency))
        throw std::invalid_argument("Foreign owner-membership arena keeper.");
    if (keeper_)
    {
        ContactGroupAllocation charge(keeper_);
        charge.add(sizeof(Persistent) + owner);
        persistent_ = std::make_unique<Persistent>(std::move(charge));
    }
}
OwnedSupportMembership::OwnedSupportMembership(const OwnedSupportMembership& source, Budget& budget,
    IncidencePools& pools, std::shared_ptr<CartesianAuthorityArena> source_arena, std::shared_ptr<Budget> keeper,
    ContactPhysicalResidency* residency)
    : OwnedSupportMembership(budget, pools, std::move(keeper), residency)
{
    if (source_arena && source_arena->budget != &budget)
        throw std::invalid_argument("Foreign owner-membership source arena.");
    if (bool(persistent_) != bool(source.persistent_) || bool(residency_) != bool(source.residency_))
        throw std::invalid_argument("Owner-membership fork changed its allocation mode.");
    if (!source.changed_explicit_cells_.empty())
        throw std::invalid_argument("Owner-membership fork requires completed inventory publication.");
    cartesian_detail::Allocation source_memory(budget);
    std::optional<cartesian_detail::CreditConstruction> source_credit;
    if (source_arena && !residency_)
        source_credit.emplace(source_arena);
    try
    {
        if (persistent_)
            persistent_->fork_from(*source.persistent_);
        else
        {
            for (const auto& [slot, cell] : source.cells_)
            {
                admit(cell_node_bytes);
                admit(cell->charge);
                ++work_.legacy_fork_visits;
            }
            for (const auto& [slot, reverse] : source.reverse_)
            {
                admit(reverse_node_bytes);
                if (reverse.spill())
                    admit(owner + reverse.size() * sizeof(Slot));
                ++work_.legacy_fork_visits;
            }
            admit(source.retired_source_cells_.size() * retired_node_bytes);
            cells_ = source.cells_;
            reverse_ = source.reverse_;
            retired_source_cells_ = source.retired_source_cells_;
        }
        if (!residency_ && source.source_ && source.source_->allocation_budget() != &budget)
        {
            if (!source.source_->has_arena_keeper() || !source_credit)
                throw std::invalid_argument(
                    "Borrowed owner-membership source requires its original and target arenas.");
            source_credit->retain(source.source_, source.source_->allocation_budget(), source.source_->metadata_bytes(),
                source.source_->has_arena_keeper(), source_leases_, source_memory);
        }
        if (!residency_ && source.inventory_)
        {
            if (!source_credit)
                throw std::invalid_argument("Source inventory fork requires its explicit shared arena.");
            source_credit->retain(source.inventory_, source.inventory_->allocation_budget(),
                source.inventory_->metadata_bytes(), source.inventory_->has_arena_keeper(), source_leases_,
                source_memory);
            source.inventory_->visit_physical_dependencies(
                [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keep)
                {
                    using E = typename std::remove_cvref_t<decltype(owner)>::element_type;
                    const std::shared_ptr<const std::remove_const_t<E>> fixed = owner;
                    source_credit->retain(fixed, origin, bytes, keep, source_leases_, source_memory);
                });
        }
        source_ = source.source_;
        inventory_ = source.inventory_;
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
    std::set<Slot>().swap(changed_explicit_cells_);
    budget_->used -= charged_;
}
std::size_t OwnedSupportMembership::explicit_cell_count() const noexcept
{
    return persistent_ ? Persistent::size(persistent_->cells) : cells_.size();
}
const OwnedSupportMembership::Cell* OwnedSupportMembership::find_explicit(Slot slot) const noexcept
{
    if (persistent_)
    {
        const auto* row = Persistent::find<Persistent::CellRow>(persistent_->cells, slot);
        return row ? row->cell.get() : nullptr;
    }
    const auto found = cells_.find(slot);
    return found == cells_.end() ? nullptr : found->second.get();
}
std::optional<OwnedSupportMembership::Slot> OwnedSupportMembership::first_explicit(Slot first) const noexcept
{
    if (persistent_)
    {
        const auto* row = Persistent::lower_bound<Persistent::CellRow>(persistent_->cells, first);
        return row ? std::optional<Slot>(row->slot) : std::nullopt;
    }
    const auto found = cells_.lower_bound(first);
    return found == cells_.end() ? std::nullopt : std::optional<Slot>(found->first);
}
void OwnedSupportMembership::visit_explicit_cells(const CartesianBorrowedCallback<Slot, const Cell&>& emit) const
{
    if (persistent_)
        Persistent::visit<Persistent::CellRow>(persistent_->cells, [&](const auto& row) { emit(row.slot, *row.cell); });
    else
        for (const auto& [slot, cell] : cells_)
            emit(slot, *cell);
}
bool OwnedSupportMembership::retired_source_contains(Slot slot) const noexcept
{
    return persistent_ ? Persistent::find<Persistent::RetiredRow>(persistent_->retired, slot) != nullptr
                       : retired_source_cells_.contains(slot);
}
std::size_t OwnedSupportMembership::retired_source_count() const noexcept
{
    return persistent_ ? Persistent::size(persistent_->retired) : retired_source_cells_.size();
}
void OwnedSupportMembership::clear_retired_source()
{
    if (persistent_)
        persistent_->publish(persistent_->retired, Persistent::Root<Persistent::RetiredRow>{});
    else
    {
        const auto bytes = retired_source_cells_.size() * retired_node_bytes;
        decltype(retired_source_cells_){}.swap(retired_source_cells_);
        release(bytes);
    }
}
void OwnedSupportMembership::clear_explicit()
{
    if (persistent_)
    {
        persistent_->publish(persistent_->cells, Persistent::Root<Persistent::CellRow>{});
        persistent_->publish(persistent_->reverse, Persistent::Root<Persistent::ReverseRow>{});
    }
    else
        while (const auto slot = first_explicit())
            erase(*slot);
}
std::size_t OwnedSupportMembership::metadata_bytes() const noexcept
{
    return source_overlap_metadata_bytes() +
           (persistent_ ? Persistent::index_bytes(persistent_->cells) + Persistent::index_bytes(persistent_->reverse) +
                              Persistent::index_bytes(persistent_->retired)
                        : 0);
}
std::size_t OwnedSupportMembership::source_overlap_metadata_bytes() const noexcept
{
    return (persistent_ ? persistent_->header.bytes()
                        : cells_.size() * cell_node_bytes + reverse_.size() * reverse_node_bytes +
                              retired_source_cells_.size() * retired_node_bytes) +
           changed_explicit_cells_.size() * retired_node_bytes + deferred_.metadata_bytes() +
           source_leases_.capacity() * sizeof(source_leases_[0]);
}
std::size_t OwnedSupportMembership::inherited_overlap_metadata_bytes() const noexcept
{
    return persistent_ ? persistent_->overlap_bytes() : 0;
}
void OwnedSupportMembership::finish_inherited_overlap() noexcept
{
    if (persistent_)
        persistent_->finish_overlap();
}
std::size_t OwnedSupportMembership::retained_bytes() const noexcept
{
    return charged_ + deferred_.retained_bytes() +
           (persistent_ ? persistent_->header.bytes() + Persistent::total(persistent_->cells) +
                              Persistent::total(persistent_->reverse) + Persistent::total(persistent_->retired) +
                              persistent_->overlap_bytes()
                        : 0) +
           (source_ && source_->allocation_budget() == budget_ ? source_->metadata_bytes() : 0) +
           (!source_ && construction_source_ && construction_source_->allocation_budget() == budget_
                   ? construction_source_->metadata_bytes()
                   : 0);
}
OwnedSupportMembership::PersistenceWork OwnedSupportMembership::persistence_work() const noexcept
{
    return work_;
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
    return add_leaf(leaf, {});
}
OwnedSupportMembership::Slot OwnedSupportMembership::add(const CartesianNativeIncidenceCell& cell)
{
    const auto& source = cell.source();
    // An environment change preserves the original explicit result. It does
    // not grant the source's older evaluation facts authority over this cell.
    const bool authentic = source && source->allocation_budget() == budget_ &&
                           source->evaluation_environment() == cell.evaluation_environment() &&
                           (!construction_source_ || same_owner(construction_source_, source));
    const auto slot = add_leaf(cell.leaf(), authentic ? source : nullptr);
    if (authentic && !construction_source_)
    {
        construction_source_ = source;
        construction_environment_ = cell.evaluation_state();
    }
    return slot;
}
OwnedSupportMembership::Slot OwnedSupportMembership::add_leaf(
    const AdaptiveLeaf& leaf, const std::shared_ptr<const CartesianIncidenceSource>& proof)
{
    const auto key = pools_->cells.intern(leaf.key);
    if (find_explicit(key) || source_cell(key))
        throw std::invalid_argument("Duplicate owner-local cell membership.");
    if (leaf.surfaces.size() > (std::numeric_limits<std::size_t>::max() - owner) / sizeof(Member))
        throw std::invalid_argument("Owner-local surface count overflow.");
    const auto bytes = owner + leaf.surfaces.size() * sizeof(Member);
    std::shared_ptr<Cell> cell;
    if (persistent_)
        cell = make_membership_body<Cell>(persistent_->header.keeper(), bytes);
    else
    {
        admit(bytes);
        admit(cell_node_bytes);
        cell = std::make_shared<Cell>();
    }
    cell->revision = leaf.revision;
    cell->construction_source = proof;
    cell->charge = bytes;
    cell->members.reserve(leaf.surfaces.size());
    if (cell->members.capacity() != leaf.surfaces.size())
        throw std::invalid_argument("Owner surface capacity differs from admission.");
    const auto ancestry = key_capacity(pools_->cells.get(key));
    for (std::size_t i = 0; i < leaf.surfaces.size(); ++i)
    {
        const auto support = pools_->supports.intern(leaf.surfaces[i].key);
        cell->members.push_back({support, support_surface_encoded_size(leaf, i)});
        cell->maximum_change =
            std::max(cell->maximum_change, cartesian_detail::add(ancestry, pools_->supports.get(support).size()));
    }
    for (std::size_t i = 0; i < cell->members.size(); ++i)
    {
        const auto support = cell->members[i].support;
        if (std::none_of(cell->members.begin(), cell->members.begin() + i,
                [&](const Member& previous) { return previous.support == support; }))
            update_reverse(support, key, false);
    }
    if (persistent_)
    {
        if (inventory_)
        {
            admit(retired_node_bytes);
            if (!changed_explicit_cells_.insert(key).second)
                throw std::logic_error("Repeated unpublished owner cell edit.");
        }
        const Persistent::CellRow row{key, std::move(cell), budget_};
        persistent_->edit(persistent_->cells, row, &row, false, work_);
    }
    else
        cells_.emplace(key, std::move(cell));
    return key;
}
bool OwnedSupportMembership::has_complete_construction_proof() const
{
    if (!construction_source_ || source_ || construction_source_->cell_count() != explicit_cell_count())
        return false;
    bool complete = true;
    visit_explicit_cells(
        [&](Slot slot, const Cell& cell)
        {
            const auto& key = pools_->cells.get(slot);
            const auto* ancestry = std::get_if<CartesianCellAncestry>(&key.ancestry);
            complete = complete && same_owner(cell.construction_source.lock(), construction_source_) && ancestry &&
                       key.region == construction_source_->region() && construction_source_->contains(*ancestry) &&
                       cell.revision == construction_source_->revision(*ancestry) && cell.members.size() == 6;
        });
    return complete;
}
bool OwnedSupportMembership::publish_constructed_source()
{
    if (!has_complete_construction_proof())
    {
        construction_source_.reset();
        return false;
    }
    if (residency_)
        observe_source(true, construction_source_, nullptr);
    clear_explicit();
    source_ = std::move(construction_source_);
    source_environment_ = construction_environment_;
    return true;
}
void OwnedSupportMembership::observe_source(
    bool add, const std::shared_ptr<const CartesianIncidenceSource>& source, CartesianAuthorityArena* arena)
{
    if (source)
    {
        if (!source->has_arena_keeper())
            throw std::invalid_argument("Owner Cartesian source lacks its immutable arena keeper.");
        IncidencePhysicalOwners(*residency_, arena, *budget_, add)
            .observe(IncidencePhysicalOwners::Role::SourceDependency, source, source->metadata_bytes(),
                source->allocation_budget());
    }
}
void OwnedSupportMembership::observe_inventory(
    bool add, const std::shared_ptr<const CartesianSourceInventory>& inventory, CartesianAuthorityArena* arena)
{
    IncidencePhysicalOwners owners(*residency_, arena, *budget_, add);
    using Role = IncidencePhysicalOwners::Role;
    if (!owners.observe(Role::SourceDependency, inventory, inventory->metadata_bytes(), inventory->allocation_budget()))
        return;
    inventory->visit_observed_dependencies(
        [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keep)
        {
            if (!origin || !keep)
                throw std::invalid_argument("Owner source inventory dependency has no immutable arena lifetime.");
            using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
            constexpr auto role = std::is_same_v<std::remove_const_t<Element>, cartesian_detail::AuthorityOwnerCredit>
                                      ? Role::CreditMetadata
                                      : Role::SourceDependency;
            return owners.observe(role, owner, bytes, origin);
        });
}
void OwnedSupportMembership::publish_inventory(
    std::shared_ptr<const CartesianSourceInventory> inventory, std::shared_ptr<CartesianAuthorityArena> arena)
{
    if (!inventory || !arena || arena->budget != budget_)
        throw std::invalid_argument("Owner source inventory requires its exact shared arena.");
    if (residency_ && !inventory->has_arena_keeper())
        throw std::invalid_argument("Owner source inventory lacks its original arena keeper.");
    const bool sparse = persistent_ && inventory->derived_from(inventory_);
    const auto new_batch = [&](const CartesianSourceInventory::Batch* batch)
    { return batch && inventory->is_added(*batch); };
    const auto validate = [&](Slot slot, const Cell& cell)
    {
        ++work_.inventory_validation_visits;
        const auto& key = pools_->cells.get(slot);
        if (const auto* batch = inventory->lookup(key))
        {
            if (!same_owner(cell.construction_source.lock(), batch->source()) || cell.members.size() != 6 ||
                cell.revision != inventory->revision(key) || (sparse && !new_batch(batch)))
                throw std::invalid_argument("Uncertified explicit owner member cannot enter source inventory.");
            return true;
        }
        return false;
    };
    std::size_t verified = 0;
    if (sparse)
    {
        std::size_t additions = 0;
        // Stable insertion slots make the appended range addressable without
        // walking clean batches or comparing addresses from different pages.
        for (auto slot = inventory->first_added_slot_; slot < inventory->next_slot_; ++slot)
        {
            const auto* batch = inventory->batch_at(slot);
            if (!batch)
                throw std::logic_error("Missing newly admitted source batch.");
            additions = cartesian_detail::add(additions, batch->cell_count());
        }
        for (const auto slot : changed_explicit_cells_)
            verified += validate(slot, at(slot));
        if (verified != additions || retired_source_count() > inventory_->cell_count() ||
            inventory->cell_count() !=
                cartesian_detail::add(inventory_->cell_count() - retired_source_count(), additions))
            throw std::invalid_argument("Derived owner inventory differs from its staged cell delta.");
        Persistent::visit<Persistent::RetiredRow>(persistent_->retired,
            [&](const auto& row)
            {
                ++work_.inventory_validation_visits;
                if (const auto* batch = inventory->lookup(pools_->cells.get(row.slot)); batch && !new_batch(batch))
                    throw std::invalid_argument("Derived owner inventory retained a retired source selector.");
            });
    }
    else
        visit_explicit_cells([&](Slot slot, const Cell& cell) { verified += validate(slot, cell); });

    cartesian_detail::Allocation memory(*budget_);
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases;
    std::optional<cartesian_detail::CreditConstruction> credits;
    if (residency_)
    {
        observe_inventory(true, inventory, arena.get());
        if (inventory_)
            observe_inventory(false, inventory_, arena.get());
        else
            observe_source(false, source_, arena.get());
    }
    else
    {
        credits.emplace(arena);
        credits->retain(inventory, inventory->allocation_budget(), inventory->metadata_bytes(),
            inventory->has_arena_keeper(), leases, memory);
        inventory->visit_physical_dependencies(
            [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keep)
            {
                using E = typename std::remove_cvref_t<decltype(owner)>::element_type;
                const std::shared_ptr<const std::remove_const_t<E>> fixed = owner;
                credits->retain(fixed, origin, bytes, keep, leases, memory);
            });
    }
    if (verified == explicit_cell_count())
        clear_explicit();
    else if (sparse)
    {
        for (auto position = changed_explicit_cells_.begin(); position != changed_explicit_cells_.end();)
        {
            const auto slot = *position++;
            if (inventory->contains(pools_->cells.get(slot)))
                erase(slot);
        }
    }
    else
    {
        for (auto position = first_explicit(); position;)
        {
            const auto slot = *position;
            if (inventory->contains(pools_->cells.get(slot)))
                erase(slot);
            position = slot == std::numeric_limits<Slot>::max() ? std::nullopt : first_explicit(slot + 1);
        }
    }
    const auto previous = source_leases_.capacity() * sizeof(source_leases_[0]);
    source_leases_.swap(leases);
    decltype(leases){}.swap(leases);
    release(previous);
    charged_ = cartesian_detail::add(charged_, std::exchange(memory.bytes, 0));
    clear_retired_source();
    const auto changed_bytes = changed_explicit_cells_.size() * retired_node_bytes;
    decltype(changed_explicit_cells_){}.swap(changed_explicit_cells_);
    release(changed_bytes);
    inventory_ = std::move(inventory);
    source_ = inventory_->batches().empty() ? nullptr : inventory_->batches().front().source();
    if (source_)
        source_environment_ = inventory_->batches().front().evaluation_state();
    construction_source_.reset();
    if (credits)
        credits->committed = true;
}
const CartesianSourceInventory::Batch* OwnedSupportMembership::source_batch(Slot slot) const
{
    return inventory_ && !retired_source_contains(slot) ? inventory_->lookup(pools_->cells.get(slot)) : nullptr;
}
const CartesianCellAncestry* OwnedSupportMembership::source_cell(Slot slot) const
{
    if (inventory_)
        return source_batch(slot) ? std::get_if<CartesianCellAncestry>(&pools_->cells.get(slot).ancestry) : nullptr;
    if (!source_ || retired_source_contains(slot))
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
    if (const auto* cell = find_explicit(slot))
        return cell->members.size();
    if (source_cell(slot))
        return 6;
    throw std::invalid_argument("Unknown owner-local cell membership.");
}
std::uint64_t OwnedSupportMembership::leaf_revision(Slot slot) const
{
    if (const auto* cell = find_explicit(slot))
        return cell->revision;
    if (const auto* cell = source_cell(slot))
        return source_batch(slot) ? source_batch(slot)->source()->revision(*cell) : source_->revision(*cell);
    throw std::invalid_argument("Unknown owner-local cell membership.");
}
void OwnedSupportMembership::visit_members(
    Slot slot, const CartesianBorrowedCallback<std::size_t, const Member&>& emit) const
{
    if (const auto* cell = find_explicit(slot))
    {
        for (std::size_t i = 0; i < cell->members.size(); ++i)
            emit(i, cell->members[i]);
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
                if (!cell || find_explicit(*cell) || retired_source_contains(*cell) ||
                    (batch && inventory_->lookup(occurrence.cell) != batch))
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
    const auto native = inventory_ && !inventory_->batches().empty()
                            ? cartesian_detail::add(64, inventory_->maximum_domain_bytes())
                            : (source_ ? cartesian_detail::add(64, source_->domain().size()) : 1);
    auto result = native;
    if (persistent_)
    {
        if (persistent_->cells)
            result = std::max(result, persistent_->cells->weights.maximum);
    }
    else
        for (const auto& [slot, cell] : cells_)
            result = std::max(result, cell->maximum_change);
    return result;
}
void OwnedSupportMembership::certify_source(
    std::shared_ptr<const CartesianIncidenceSource> source, std::shared_ptr<CartesianAuthorityArena> source_arena)
{
    if (!source || source_ || source->cell_count() != explicit_cell_count() ||
        source->evaluation_environment() != cartesian_evaluation_environment())
        throw std::invalid_argument("Cartesian owner membership source is incomplete or has stale evaluation facts.");
    if (source->allocation_budget() != budget_ && !source->has_arena_keeper())
        throw std::invalid_argument("Foreign Cartesian membership arena lacks a lifetime keeper.");
    if ((source_arena && source_arena->budget != budget_) || (source->allocation_budget() != budget_ && !source_arena))
        throw std::invalid_argument("Foreign Cartesian replay certification requires its explicit shared arena.");
    cartesian_detail::Allocation source_memory(*budget_);
    std::optional<cartesian_detail::CreditConstruction> source_credit;
    if (source_arena && !residency_)
        source_credit.emplace(source_arena);
    std::fenv_t captured{};
    if (std::fegetenv(&captured))
        throw std::invalid_argument("Cannot capture certified membership environment.");
    std::size_t replayed = 0;
    source->visit_surfaces(*budget_,
        [&](const auto& occurrence)
        {
            const auto cell = pools_->cells.find(occurrence.cell);
            if (!cell || !find_explicit(*cell))
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
    if (residency_)
        observe_source(true, source, source_arena.get());
    else if (source->allocation_budget() != budget_)
        source_credit->retain(source, source->allocation_budget(), source->metadata_bytes(), source->has_arena_keeper(),
            source_leases_, source_memory);
    clear_explicit();
    source_ = std::move(source);
    source_environment_ = captured;
    charged_ = cartesian_detail::add(charged_, std::exchange(source_memory.bytes, 0));
    if (source_credit)
        source_credit->committed = true;
}
const OwnedSupportMembership::Cell& OwnedSupportMembership::at(Slot slot) const
{
    if (const auto* cell = find_explicit(slot))
        return *cell;
    throw std::invalid_argument("Unknown owner-local cell membership.");
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
    if (!find_explicit(slot))
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
    if (persistent_)
    {
        const auto* row = Persistent::find<Persistent::ReverseRow>(persistent_->reverse, support);
        return row ? &row->cells : nullptr;
    }
    const auto found = reverse_.find(support);
    return found == reverse_.end() ? nullptr : &found->second;
}
void OwnedSupportMembership::update_reverse(Slot support, Slot cell, bool remove)
{
    const auto* current = support_cells(support);
    const SupportCells empty;
    const auto& previous = current ? *current : empty;
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
    const auto next_member = [&](std::size_t i)
    {
        return remove ? previous[i < index ? i : i + 1] : i == index ? cell : previous[i < index ? i : i - 1];
    };
    if (persistent_)
    {
        auto next = SupportCells::make(next_count, next_member,
            [&] { return make_membership_body<SupportCells::Spill>(persistent_->header.keeper(), bytes); });
        const Persistent::ReverseRow row{support, std::move(next), bytes ? budget_ : nullptr, bytes};
        persistent_->edit(persistent_->reverse, row, next_count ? &row : nullptr, bool(current), work_);
        return;
    }
    auto found = reverse_.find(support);
    if (bytes)
        admit(bytes);
    if (found == reverse_.end())
        admit(reverse_node_bytes);
    auto next = SupportCells::make(next_count, next_member);
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
    const auto* cell = find_explicit(slot);
    if (!cell)
    {
        if (!source_cell(slot))
            throw std::invalid_argument("Unknown owner-local cell retirement.");
        if (persistent_)
        {
            const Persistent::RetiredRow row{slot};
            persistent_->edit(persistent_->retired, row, &row, false, work_);
        }
        else
        {
            admit(retired_node_bytes);
            retired_source_cells_.insert(slot);
        }
        return;
    }
    const auto& members = cell->members;
    if (persistent_)
    {
        for (std::size_t i = 0; i < members.size(); ++i)
            if (std::none_of(members.begin(), members.begin() + i,
                    [&](const Member& previous) { return previous.support == members[i].support; }))
                update_reverse(members[i].support, slot, true);
        const Persistent::CellRow key{slot};
        persistent_->edit<Persistent::CellRow>(persistent_->cells, key, nullptr, true, work_);
        if (changed_explicit_cells_.erase(slot))
            release(retired_node_bytes);
        return;
    }
    const auto found = cells_.find(slot);
    const auto bytes = found->second->charge;
    const bool pinned = found->second.use_count() > 1 || deferred_.contains(found->second);
    if (pinned && deferred_.adopt(found->second, bytes))
        charged_ -= bytes;
    for (std::size_t i = 0; i < members.size(); ++i)
        if (std::none_of(members.begin(), members.begin() + i,
                [&](const Member& previous) { return previous.support == members[i].support; }))
            update_reverse(members[i].support, slot, true);
    cells_.erase(found);
    if (!pinned)
        release(bytes);
    release(cell_node_bytes);
}
void OwnedSupportMembership::visit_payload(
    const CartesianBorrowedCallback<const ContactPayloadAllocation&>& visitor) const
{
    const auto visit_cell = [&](const Cell& cell)
    {
        contact_payload_object(visitor, &cell, ContactPayloadRole::OwnedMembershipCell, sizeof(Cell), true);
        contact_payload_buffer(visitor, &cell, ContactPayloadRole::OwnedMembershipMembers, cell.members);
    };
    const auto visit_reverse = [&](const SupportCells& cells)
    {
        if (const auto& spill = cells.spill())
        {
            contact_payload_object(
                visitor, spill.get(), ContactPayloadRole::OwnedMembershipSpill, sizeof(SupportCells::Spill), true);
            contact_payload_buffer(visitor, spill.get(), ContactPayloadRole::OwnedMembershipValues, *spill);
        }
    };
    if (persistent_)
    {
        contact_payload_object(
            visitor, persistent_.get(), ContactPayloadRole::OwnedMembershipController, sizeof(Persistent));
        const auto page = [&](const auto& node, ContactPayloadRole role)
        {
            contact_payload_object(visitor, node.get(), role, sizeof(*node), true);
            contact_payload_buffer(visitor, node.get(), role, node->rows);
            contact_payload_buffer(visitor, node.get(), role, node->children);
        };
        Persistent::Tree<Persistent::CellRow>::visit_pages(
            persistent_->cells, [&](const auto& node) { page(node, ContactPayloadRole::OwnedMembershipNodes); });
        Persistent::Tree<Persistent::ReverseRow>::visit_pages(persistent_->reverse,
            [&](const auto& node) { page(node, ContactPayloadRole::OwnedMembershipReverseNodes); });
        Persistent::Tree<Persistent::RetiredRow>::visit_pages(persistent_->retired,
            [&](const auto& node) { page(node, ContactPayloadRole::OwnedMembershipRetiredNodes); });
        visit_explicit_cells([&](Slot, const Cell& cell) { visit_cell(cell); });
        Persistent::visit<Persistent::ReverseRow>(
            persistent_->reverse, [&](const auto& row) { visit_reverse(row.cells); });
    }
    else
    {
        for (const auto& entry : cells_)
        {
            contact_payload_node_value(visitor, this, ContactPayloadRole::OwnedMembershipNodes, entry);
            visit_cell(*entry.second);
        }
        for (const auto& entry : reverse_)
        {
            contact_payload_node_value(visitor, this, ContactPayloadRole::OwnedMembershipReverseNodes, entry);
            visit_reverse(entry.second);
        }
        for (const auto& entry : retired_source_cells_)
            contact_payload_node_value(visitor, this, ContactPayloadRole::OwnedMembershipRetiredNodes, entry);
    }
    for (const auto& entry : changed_explicit_cells_)
        contact_payload_node_value(visitor, this, ContactPayloadRole::OwnedMembershipRetiredNodes, entry);
    contact_payload_buffer(visitor, this, ContactPayloadRole::OwnedMembershipLeases, source_leases_);
    for (const auto& lease : source_leases_)
        lease->visit_borrowed_payload(visitor);
    if (inventory_)
        inventory_->visit_borrowed_payload(visitor);
    else if (source_)
        source_->visit_borrowed_payload(visitor);
    if (construction_source_)
        construction_source_->visit_borrowed_payload(visitor);
    deferred_.visit_borrowed_payload(visitor);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
