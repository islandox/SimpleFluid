/** @file CartesianSourceInventory.cc
 * @brief Budgeted immutable Cartesian ancestry transitions without leaf catalogs.
 */
#include "CartesianSourceInventory.hh"
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <variant>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using cartesian_detail::add;
using cartesian_detail::Allocation;
using cartesian_detail::product;
constexpr std::uint64_t inventory_magic = 0x534643494e563031ULL;
constexpr std::uint64_t source_magic = 0x5346434953524331ULL;

bool range_contains(const CartesianIncidenceRange& range, const CartesianCellAncestry& cell) noexcept
{
    if (range.level != cell.level)
        return false;
    for (unsigned axis = 0; axis < 3; ++axis)
        if (cell.index[axis] < range.lower[axis] || cell.index[axis] >= range.upper[axis])
            return false;
    return true;
}
std::size_t volume(const CartesianIncidenceRange& range)
{
    std::size_t result = 1;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        if (range.upper[axis] <= range.lower[axis])
            throw std::invalid_argument("Empty Cartesian inventory selector.");
        const auto width = range.upper[axis] - range.lower[axis];
        if (width > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Cartesian inventory selector is unrepresentable.");
        result = product(result, std::size_t(width));
    }
    return result;
}
bool overlap(const CartesianIncidenceRange& first, const CartesianIncidenceRange& second) noexcept
{
    const auto& coarse = first.level <= second.level ? first : second;
    const auto& fine = first.level <= second.level ? second : first;
    // Project fine ancestry intervals to the coarser integer lattice. This
    // detects any ancestor/descendant pair without shifting large indices up
    // or evaluating rounded physical coordinates. Source levels are <63.
    const auto shift = fine.level - coarse.level;
    for (unsigned axis = 0; axis < 3; ++axis)
        if (coarse.upper[axis] <= (fine.lower[axis] >> shift) ||
            ((fine.upper[axis] - 1) >> shift) + 1 <= coarse.lower[axis])
            return false;
    return true;
}
bool range_less(const CartesianIncidenceRange& first, const CartesianIncidenceRange& second) noexcept
{
    if (first.level != second.level)
        return first.level < second.level;
    if (first.lower != second.lower)
        return first.lower < second.lower;
    return first.upper < second.upper;
}
/** Merge ancestry rectangles only. This says nothing about physical faces. */
void coalesce(std::vector<CartesianIncidenceRange>& ranges)
{
    bool changed = true;
    while (changed)
    {
        changed = false;
        std::sort(ranges.begin(), ranges.end(), range_less);
        for (std::size_t first = 0; first < ranges.size() && !changed; ++first)
            for (std::size_t second = first + 1; second < ranges.size() && !changed; ++second)
            {
                auto& a = ranges[first];
                const auto& b = ranges[second];
                if (a.level != b.level)
                    continue;
                for (unsigned direction = 0; direction < 3; ++direction)
                {
                    bool same = true;
                    for (unsigned axis = 0; axis < 3; ++axis)
                        if (axis != direction && (a.lower[axis] != b.lower[axis] || a.upper[axis] != b.upper[axis]))
                            same = false;
                    if (same && (a.upper[direction] == b.lower[direction] || b.upper[direction] == a.lower[direction]))
                    {
                        a.lower[direction] = std::min(a.lower[direction], b.lower[direction]);
                        a.upper[direction] = std::max(a.upper[direction], b.upper[direction]);
                        ranges.erase(ranges.begin() + second);
                        changed = true;
                        break;
                    }
                }
            }
    }
    std::sort(ranges.begin(), ranges.end(), range_less);
}
/** Splitting one same-level half-open box needs at most six disjoint slabs. */
std::pair<std::array<CartesianIncidenceRange, 6>, std::size_t> split_at(
    CartesianIncidenceRange remainder, const CartesianCellAncestry& cell)
{
    if (!range_contains(remainder, cell))
        throw std::invalid_argument("Missing Cartesian inventory retirement selector.");
    std::array<CartesianIncidenceRange, 6> pieces{};
    std::size_t count = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        if (remainder.lower[axis] < cell.index[axis])
        {
            auto slab = remainder;
            slab.upper[axis] = cell.index[axis];
            pieces[count++] = slab;
            remainder.lower[axis] = cell.index[axis];
        }
        const auto after = cell.index[axis] + 1;
        if (after < remainder.upper[axis])
        {
            auto slab = remainder;
            slab.lower[axis] = after;
            pieces[count++] = slab;
            remainder.upper[axis] = after;
        }
    }
    return {pieces, count};
}
std::optional<CartesianIncidenceRange> merge(
    const CartesianIncidenceRange& first, const CartesianIncidenceRange& second) noexcept
{
    if (first.level != second.level)
        return std::nullopt;
    for (unsigned direction = 0; direction < 3; ++direction)
    {
        bool same = true;
        for (unsigned axis = 0; axis < 3; ++axis)
            if (axis != direction &&
                (first.lower[axis] != second.lower[axis] || first.upper[axis] != second.upper[axis]))
                same = false;
        if (same &&
            (first.upper[direction] == second.lower[direction] || second.upper[direction] == first.lower[direction]))
        {
            auto combined = first;
            combined.lower[direction] = std::min(first.lower[direction], second.lower[direction]);
            combined.upper[direction] = std::max(first.upper[direction], second.upper[direction]);
            return combined;
        }
    }
    return std::nullopt;
}
/** The trusted source encoding contains its complete kernel descriptor. No
 * parsed descriptor is promoted to authority by this comparison. */
std::span<const char> grid_descriptor(const CartesianIncidenceSource& source)
{
    Reader reader{source.identity()};
    if (reader.word() != source_magic)
        throw std::invalid_argument("Unknown trusted Cartesian source encoding.");
    for (unsigned control = 0; control < 4; ++control)
        (void) reader.word();
    const auto size = reader.word();
    if (size > reader.data.size() - reader.offset)
        throw std::invalid_argument("Truncated trusted Cartesian kernel descriptor.");
    return reader.data.subspan(reader.offset, std::size_t(size));
}
/** Activate the original environment only for signature authentication. Caller
 * controls and flags are restored even if signature extraction throws. */
void authenticate_environment(const CartesianSourceInventory::CapturedSource& captured)
{
    struct Restore
    {
        std::fenv_t caller{};
        bool active = false;
        ~Restore()
        {
            if (active)
                (void) std::fesetenv(&caller);
        }
    } restore;
    if (std::fegetenv(&restore.caller))
        throw std::invalid_argument("Cannot capture Cartesian inventory caller environment.");
    restore.active = true;
    if (std::fesetenv(&captured.environment))
        throw std::invalid_argument("Cannot restore Cartesian inventory source environment.");
    const auto signature = cartesian_evaluation_environment();
    if (std::fesetenv(&restore.caller))
        throw std::invalid_argument("Cannot restore Cartesian inventory caller environment.");
    restore.active = false;
    if (signature != captured.source->evaluation_environment())
        throw std::invalid_argument("Captured Cartesian inventory controls differ from the trusted source.");
}
} // namespace

bool CartesianSourceInventory::RangeLess::operator()(
    const CartesianIncidenceRange& first, const CartesianIncidenceRange& second) const noexcept
{
    return range_less(first, second);
}
bool CartesianSourceInventory::Endpoint::operator<(const Endpoint& other) const noexcept
{
    if (level == other.level)
        return index < other.index;
    if (level > other.level)
        return (index >> (level - other.level)) < other.index;
    const auto shift = other.level - level;
    const auto floor = other.index >> shift;
    return index < floor || (index == floor && other.index != (floor << shift));
}
bool CartesianSourceInventory::SelectorLess::operator()(const Selector& first, const Selector& second) const noexcept
{
    if (first.region != second.region)
        return first.region < second.region;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        const Endpoint a{first.range.lower[axis], first.range.level};
        const Endpoint b{second.range.lower[axis], second.range.level};
        if (a < b)
            return true;
        if (b < a)
            return false;
    }
    if (range_less(first.range, second.range))
        return true;
    if (range_less(second.range, first.range))
        return false;
    return first.batch < second.batch;
}
void CartesianSourceInventory::BatchWeights::add_row(Summary& total, const Batch& row, const Budget*)
{
    total.cells = add(total.cells, row.cell_count());
    total.maximum_domain = std::max(total.maximum_domain, row.source()->domain().size());
    total.identity_bytes =
        add(total.identity_bytes, add(add(24, row.source()->identity().size()), product(56, row.ranges().size())));
}
void CartesianSourceInventory::BatchWeights::add_child(Summary& total, const Summary& child, bool)
{
    total.cells = add(total.cells, child.cells);
    total.maximum_domain = std::max(total.maximum_domain, child.maximum_domain);
    total.identity_bytes = add(total.identity_bytes, child.identity_bytes);
}
void CartesianSourceInventory::SelectorWeights::add_row(Summary& total, const Selector& row, const Budget*)
{
    Summary value;
    value.present = true;
    value.first_region = value.last_region = row.region;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        value.lower[axis] = {row.range.lower[axis], row.range.level};
        value.upper[axis] = {row.range.upper[axis], row.range.level};
    }
    add_child(total, value, true);
}
void CartesianSourceInventory::SelectorWeights::add_child(Summary& total, const Summary& child, bool)
{
    if (!total.present)
    {
        total = child;
        return;
    }
    if (!child.present)
        return;
    total.first_region = std::min(total.first_region, child.first_region);
    total.last_region = std::max(total.last_region, child.last_region);
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        total.lower[axis] = std::min(total.lower[axis], child.lower[axis]);
        total.upper[axis] = std::max(total.upper[axis], child.upper[axis]);
    }
}

struct CartesianSourceInventory::Builder
{
    using Work = persistent_ordered_detail::OrderedPageWork;
    using Dirty = std::set<CartesianIncidenceRange, RangeLess>;
    static constexpr auto dirty_bytes = sizeof(Dirty::value_type) + 4 * sizeof(void*);
    CartesianSourceInventory& result;
    Work work;
    RangeTree ranges;
    BatchTree batches;
    SelectorTree selectors;
    Builder(CartesianSourceInventory& inventory, const std::shared_ptr<Budget>& arena,
        const std::shared_ptr<const geometry_detail::NativeAllocationState>& native)
        : result(inventory), ranges(arena, &work, {}, native), batches(arena, &work, {}, native),
          selectors(arena, &work, {}, native)
    {
    }

    template<class Visit>
    static bool visit_selectors(const SelectorTree::Node* node, std::uint64_t region,
        const std::array<Endpoint, 3>& lower, const std::array<Endpoint, 3>& upper, Visit& visit, DependencyWork* work)
    {
        if (!node)
            return true;
        if (work)
            ++work->selector_nodes;
        const auto& bounds = node->weights;
        if (!bounds.present || region < bounds.first_region || region > bounds.last_region)
            return true;
        for (unsigned axis = 0; axis < 3; ++axis)
            if (bounds.upper[axis] < lower[axis] || upper[axis] < bounds.lower[axis])
                return true;
        if (node->leaf())
        {
            for (const auto& row : node->rows)
            {
                if (work)
                    ++work->selector_rows;
                if (row.region != region)
                    continue;
                bool intersects = true;
                for (unsigned axis = 0; axis < 3; ++axis)
                    if (Endpoint{row.range.upper[axis], row.range.level} < lower[axis] ||
                        upper[axis] < Endpoint{row.range.lower[axis], row.range.level})
                        intersects = false;
                if (intersects && !visit(row))
                    return false;
            }
            return true;
        }
        for (const auto& child : node->children)
            if (!visit_selectors(child.get(), region, lower, upper, visit, work))
                return false;
        return true;
    }
    template<class Visit> void neighbors(std::uint64_t region, const CartesianIncidenceRange& range, Visit&& visit)
    {
        std::array<Endpoint, 3> lower{}, upper{};
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            lower[axis] = {range.lower[axis], range.level};
            upper[axis] = {range.upper[axis], range.level};
        }
        auto observe = [&](const Selector& row)
        {
            ++result.work_.selector_tests;
            return visit(row);
        };
        (void) visit_selectors(result.selectors_.get(), region, lower, upper, observe, &result.work_);
    }
    void add_selector(const Batch& batch, const CartesianIncidenceRange& range, bool check)
    {
        const auto region = batch.source_->region();
        if (check)
            neighbors(region, range,
                [&](const Selector& other)
                {
                    if (other.batch != batch.slot_ && overlap(range, other.range))
                        throw std::invalid_argument("Duplicate current Cartesian inventory ancestry.");
                    return true;
                });
        const Selector row{region, batch.slot_, range};
        result.selectors_ = selectors.edit(result.selectors_, row, &row, false);
    }
    void erase_selector(const Batch& batch, const CartesianIncidenceRange& range)
    {
        const Selector row{batch.source_->region(), batch.slot_, range};
        result.selectors_ = selectors.edit(result.selectors_, row, nullptr, true);
    }
    void insert(Batch& batch, const CartesianIncidenceRange& range)
    {
        add_selector(batch, range, false);
        batch.ranges_ = ranges.edit(batch.ranges_, range, &range, false);
    }
    void erase(Batch& batch, const CartesianIncidenceRange& range)
    {
        erase_selector(batch, range);
        batch.ranges_ = ranges.edit(batch.ranges_, range, nullptr, true);
    }
    static void track(Dirty& dirty, const CartesianIncidenceRange& range, Allocation& scratch)
    {
        scratch.retain(dirty_bytes);
        if (!dirty.insert(range).second)
            throw std::logic_error("Duplicate changed Cartesian selector.");
    }
    static void untrack(Dirty& dirty, const CartesianIncidenceRange& range, Allocation& scratch)
    {
        if (dirty.erase(range))
        {
            scratch.bytes -= dirty_bytes;
            scratch.budget.used -= dirty_bytes;
        }
    }
    void coalesce_changed(Batch& batch, Dirty& dirty, Allocation& scratch)
    {
        // The accepted ranges were already at the legacy merge fixed point.
        // Removing a range cannot make two unchanged rectangles adjacent.
        // Every new merge therefore touches a slab or a previous merge. Take
        // the same lexicographically first eligible pair as the full replay.
        for (;;)
        {
            using Pair = std::pair<CartesianIncidenceRange, CartesianIncidenceRange>;
            std::optional<Pair> first;
            const auto less = [](const Pair& a, const Pair& b)
            {
                if (range_less(a.first, b.first))
                    return true;
                if (range_less(b.first, a.first))
                    return false;
                return range_less(a.second, b.second);
            };
            for (const auto& range : dirty)
                neighbors(batch.source_->region(), range,
                    [&](const Selector& other)
                    {
                        if (other.batch == batch.slot_ && merge(range, other.range))
                        {
                            const auto ordered = std::minmax(range, other.range, range_less);
                            const Pair candidate{ordered.first, ordered.second};
                            if (!first || less(candidate, *first))
                                first = candidate;
                        }
                        return true;
                    });
            if (!first)
                return;
            const auto combined = *merge(first->first, first->second);
            erase(batch, first->first);
            erase(batch, first->second);
            untrack(dirty, first->first, scratch);
            untrack(dirty, first->second, scratch);
            insert(batch, combined);
            track(dirty, combined, scratch);
        }
    }
    void retire_batch(std::size_t slot, std::span<const Retirement* const> retired)
    {
        const auto* prior = result.batch_at(slot);
        if (!prior)
            throw std::logic_error("Missing Cartesian inventory retirement batch.");
        auto batch = *prior;
        ++result.work_.rebuilt_batches;
        if (retired.size() == batch.cells_)
        {
            for (const auto& range : batch.ranges())
                erase_selector(batch, range);
            result.batches_ = batches.edit(result.batches_, batch, nullptr, true);
            return;
        }
        Allocation scratch(*result.budget_);
        Dirty dirty;
        for (const auto* retirement : retired)
        {
            const auto* found = result.select(retirement->key, &result.work_);
            if (!found || found->batch != slot)
                throw std::logic_error("Authenticated Cartesian retirement lost its selector.");
            const auto previous = found->range;
            const auto& cell = std::get<CartesianCellAncestry>(retirement->key.ancestry);
            const auto [pieces, count] = split_at(previous, cell);
            erase(batch, previous);
            untrack(dirty, previous, scratch);
            for (const auto& range : std::span(pieces).first(count))
            {
                insert(batch, range);
                track(dirty, range, scratch);
            }
        }
        batch.cells_ -= retired.size();
        coalesce_changed(batch, dirty, scratch);
        result.batches_ = batches.edit(result.batches_, batch, &batch, true);
    }
    void add_batch(Batch batch, std::span<const CartesianIncidenceRange> input, bool check)
    {
        batch.ranges_ = ranges.build(input);
        for (const auto& range : input)
            add_selector(batch, range, check);
        result.batches_ = batches.edit(result.batches_, batch, &batch, false);
        ++result.work_.rebuilt_batches;
    }
};

bool CartesianSourceInventory::Batch::contains(const CartesianCellAncestry& cell) const noexcept
{
    for (const auto& range : ranges())
        if (range_contains(range, cell))
            return true;
    return false;
}
const CartesianSourceInventory::Batch* CartesianSourceInventory::batch_at(std::size_t slot) const noexcept
{
    const auto* batch = BatchTree::lower_bound(batches_, [slot](const Batch& row) { return row.slot_ < slot; });
    return batch && batch->slot_ == slot ? batch : nullptr;
}
const CartesianSourceInventory::Selector* CartesianSourceInventory::select(
    const AdaptiveCellKey& key, DependencyWork* work) const noexcept
{
    const auto* cell = std::get_if<CartesianCellAncestry>(&key.ancestry);
    if (!cell || cell->level >= 63)
        return nullptr;
    const Selector* found = nullptr;
    std::array<Endpoint, 3> point{};
    for (unsigned axis = 0; axis < 3; ++axis)
        point[axis] = {cell->index[axis], cell->level};
    auto test = [&](const Selector& row)
    {
        if (work)
            ++work->selector_tests;
        if (range_contains(row.range, *cell))
        {
            found = &row;
            return false;
        }
        return true;
    };
    (void) Builder::visit_selectors(selectors_.get(), key.region, point, point, test, work);
    return found;
}
const CartesianSourceInventory::Batch* CartesianSourceInventory::lookup(const AdaptiveCellKey& key) const noexcept
{
    const auto* match = select(key);
    return match ? batch_at(match->batch) : nullptr;
}
std::uint64_t CartesianSourceInventory::revision(const AdaptiveCellKey& key) const
{
    const auto* batch = lookup(key);
    if (!batch)
        throw std::invalid_argument("Missing Cartesian inventory member revision.");
    return batch->source_->revision(std::get<CartesianCellAncestry>(key.ancestry));
}
std::shared_ptr<const CartesianSourceInventory> CartesianSourceInventory::make(ContactHaloContext context, int owner,
    std::uint64_t version, std::span<const CapturedSource> sources, Budget& budget, std::shared_ptr<Budget> keeper,
    std::shared_ptr<CartesianAuthorityArena> arena)
{
    return construct(nullptr, context, owner, version, {}, sources, budget, std::move(keeper), std::move(arena));
}
std::shared_ptr<const CartesianSourceInventory> CartesianSourceInventory::derive(ContactHaloContext next,
    std::uint64_t version, std::span<const Retirement> retired, std::span<const CapturedSource> sources, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena) const
{
    return construct(this, next, owner_, version, retired, sources, budget, std::move(keeper), std::move(arena));
}
std::shared_ptr<const CartesianSourceInventory> CartesianSourceInventory::construct(const CartesianSourceInventory* old,
    ContactHaloContext context, int owner, std::uint64_t version, std::span<const Retirement> retired,
    std::span<const CapturedSource> sources, Budget& budget, std::shared_ptr<Budget> keeper,
    std::shared_ptr<CartesianAuthorityArena> arena)
{
    cartesian_detail::check_keeper(budget, keeper);
    if (!arena || arena->budget != &budget)
        throw std::invalid_argument("Cartesian inventory requires its explicit budget arena.");
    cartesian_detail::check_keeper(budget, arena->keeper);
    if (owner < 0 || !version || (old && version <= old->version_))
        throw std::invalid_argument("Cartesian inventory requires a valid owner and fresh version.");
    if (!keeper)
        keeper = arena->keeper;

    // Group exact predecessor retirements by borrowed typed key and stable
    // batch slot. The scratch is proportional to the supplied delta.
    struct KeyLess
    {
        bool operator()(const AdaptiveCellKey* first, const AdaptiveCellKey* second) const { return *first < *second; }
    };
    using RetiredKeys = std::set<const AdaptiveCellKey*, KeyLess>;
    using RetiredBatches = std::map<std::size_t, std::vector<const Retirement*>>;
    Allocation retirement_memory(budget);
    RetiredKeys keys;
    RetiredBatches retired_batches;
    DependencyWork retirement_work;
    for (const auto& item : retired)
    {
        const auto* selector = old ? old->select(item.key, &retirement_work) : nullptr;
        const auto* batch = selector ? old->batch_at(selector->batch) : nullptr;
        if (!batch || batch->source_->revision(std::get<CartesianCellAncestry>(item.key.ancestry)) != item.revision)
            throw std::invalid_argument("Missing or stale Cartesian inventory retirement.");
        retirement_memory.retain(sizeof(RetiredKeys::value_type) + 4 * sizeof(void*));
        if (!keys.insert(&item.key).second)
            throw std::invalid_argument("Repeated Cartesian inventory retirement.");
        auto position = retired_batches.find(selector->batch);
        if (position == retired_batches.end())
        {
            retirement_memory.retain(sizeof(RetiredBatches::value_type) + 4 * sizeof(void*));
            position = retired_batches.try_emplace(selector->batch).first;
        }
        retirement_memory.append(position->second);
        position->second.push_back(&item);
    }

    std::span<const char> descriptor;
    if (old && !old->batches().empty())
        descriptor = grid_descriptor(*old->batches().front().source_);
    for (const auto& captured : sources)
    {
        if (!captured.source || captured.source->owner() != owner || captured.source->context() != context)
            throw std::invalid_argument("New Cartesian source differs from current inventory owner/context.");
        authenticate_environment(captured);
        const auto current = grid_descriptor(*captured.source);
        if (!descriptor.empty() && !std::ranges::equal(descriptor, current))
            throw std::invalid_argument("Cartesian inventory kernel descriptors disagree.");
        descriptor = current;
    }

    cartesian_detail::CreditConstruction credits(arena);
    Allocation retained(budget);
    retained.retain(sizeof(CartesianSourceInventory));
    auto result = contact_funding_detail::make_controlled<source_control_bytes>(
        budget, keeper, [] { return new CartesianSourceInventory; }).first;
    result->budget_ = &budget;
    result->keeper_ = std::move(keeper);
    result->context_ = context;
    result->owner_ = owner;
    result->version_ = version;
    result->work_ = retirement_work;
    result->observed_dependencies_ = result->keeper_ && arena->prepaid_query && !arena->prepaid_policy.expired();
    if (old)
    {
        result->predecessor_ = old->weak_from_this();
        if (old->has_arena_keeper())
            if (const auto previous = result->predecessor_.lock())
                result->predecessor_control_ =
                    previous->native_funding(previous, previous->metadata_bytes()).body.physical_observation();
        result->next_slot_ = old->next_slot_;
    }
    result->first_added_slot_ = result->next_slot_;
    result->shared_dependencies_ =
        result->observed_dependencies_ && old && old->keeper_ &&
        arena->prepaid(old->shared_from_this(), old->allocation_budget(), old->metadata_bytes(), false);
    // Borrowed-arena compatibility roots never cross a publication fork.
    const auto page_arena =
        result->keeper_ ? result->keeper_ : std::shared_ptr<Budget>(std::shared_ptr<void>{}, &budget);
    Builder builder(*result, page_arena, arena->native_index);
    if (result->shared_dependencies_)
    {
        result->batches_ = old->batches_;
        result->selectors_ = old->selectors_;
    }
    else if (old)
        for (const auto& batch : old->batches())
        {
            ++result->work_.inherited_admission_visits;
            const auto retirement = retired_batches.find(batch.slot_);
            if (retirement != retired_batches.end() && retirement->second.size() == batch.cells_)
                continue;
            credits.retain(batch.source_, batch.source_->allocation_budget(), batch.source_->metadata_bytes(),
                batch.source_->has_arena_keeper(), result->leases_, retained);
            Allocation scratch(budget);
            std::vector<CartesianIncidenceRange> copied;
            scratch.reserve(copied, batch.ranges().size());
            for (const auto& range : batch.ranges())
                copied.push_back(range);
            builder.add_batch(batch, copied, false);
        }
    for (const auto& [slot, edits] : retired_batches)
    {
        if (result->batch_at(slot))
            builder.retire_batch(slot, edits);
        else if (edits.size() != old->batch_at(slot)->cells_)
            throw std::logic_error("Authenticated Cartesian retirement lost its predecessor batch.");
    }

    for (const auto& captured : sources)
    {
        if (!captured.source->cell_count())
            continue;
        if (result->next_slot_ == std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("Cartesian inventory batch slots exhausted.");
        credits.retain(captured.source, captured.source->allocation_budget(), captured.source->metadata_bytes(),
            captured.source->has_arena_keeper(), result->leases_, retained);
        Batch batch;
        batch.source_ = captured.source;
        batch.environment_ = captured.environment;
        batch.slot_ = result->next_slot_++;
        Allocation scratch(budget);
        std::vector<CartesianIncidenceRange> selected;
        scratch.reserve(selected, captured.source->ranges().size());
        selected.assign(captured.source->ranges().begin(), captured.source->ranges().end());
        coalesce(selected);
        for (const auto& range : selected)
            batch.cells_ = add(batch.cells_, volume(range));
        if (batch.cells_ != captured.source->cell_count())
            throw std::invalid_argument("Trusted Cartesian source selectors disagree with its cell count.");
        builder.add_batch(std::move(batch), selected, true);
    }
    if (result->batches_)
        (void) add(96, result->batches_->weights.identity_bytes);
    result->work_.allocated_pages = builder.work.copied_pages;
    result->work_.allocated_bytes = builder.work.copied_bytes;
    result->charge_ = retained.bytes + source_control_bytes;
    retained.bytes = 0;
    credits.committed = true;
    return result;
}

void CartesianSourceInventory::emit_identity(Writer& writer) const
{
    const auto size = identity_size();
    const auto before = writer.data.size();
    if (size > writer.data.capacity() - before)
        throw std::invalid_argument("Cartesian inventory wire identity lacks pre-admitted writer capacity.");
    writer.word(inventory_magic);
    writer.word(1);
    for (auto value : {context_.epoch, context_.topology, context_.partition, context_.round, context_.transaction})
        writer.word(value);
    writer.word(std::uint64_t(owner_));
    writer.word(version_);
    writer.word(cell_count());
    writer.word(batches().size());
    writer.word(0);
    for (const auto& batch : batches())
    {
        writer.bytes(batch.source_->identity());
        writer.word(batch.cells_);
        writer.word(batch.ranges().size());
        for (const auto& range : batch.ranges())
        {
            writer.word(range.level);
            for (auto value : range.lower)
                writer.word(value);
            for (auto value : range.upper)
                writer.word(value);
        }
    }
    if (writer.data.size() - before != size)
        throw std::logic_error("Cartesian inventory identity differs from its preflight.");
}
CartesianSourceInventory::EncodedIdentity::EncodedIdentity(Budget& budget, const CartesianSourceInventory& inventory)
    : memory_(budget)
{
    memory_.reserve(writer_.data, inventory.identity_size());
    inventory.emit_identity(writer_);
}
CartesianSourceInventory::EncodedIdentity CartesianSourceInventory::encode_identity(Budget& budget) const
{
    return EncodedIdentity(budget, *this);
}
bool CartesianSourceInventory::dependencies_admitted_in(const CartesianAuthorityArena& arena) const noexcept
{
    if (!observed_dependencies_ || budget_ != arena.budget)
        return false;
    if (const auto self = weak_from_this().lock(); self && arena.prepaid(self, budget_, metadata_bytes(), false))
        return true;
    if (shared_dependencies_)
    {
        const auto predecessor = predecessor_.lock();
        if (!predecessor ||
            !arena.prepaid(predecessor, predecessor->allocation_budget(), predecessor->metadata_bytes(), false))
            return false;
    }
    return std::ranges::all_of(leases_,
        [&](const auto& lease)
        {
            if (lease->budget != arena.budget)
                return false;
            if (lease->charged_payload == lease->payload)
                return true;
            const auto owner = lease->owner.lock();
            return owner && owner.get() == lease->owner_identity &&
                   arena.prepaid(owner, lease->origin, lease->payload, lease->credit_metadata);
        });
}

void CartesianSourceInventory::retain_dependencies(cartesian_detail::CreditConstruction& credits,
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>>& leases, Allocation& memory) const
{
    if (dependencies_admitted_in(*credits.arena))
        return;
    visit_physical_dependencies(
        [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keep)
        {
            using E = typename std::remove_cvref_t<decltype(owner)>::element_type;
            const std::shared_ptr<const std::remove_const_t<E>> fixed = owner;
            credits.retain(fixed, origin, bytes, keep, leases, memory);
        });
}

bool CartesianSourceInventory::derived_from(
    const std::shared_ptr<const CartesianSourceInventory>& source) const noexcept
{
    return source && !predecessor_.owner_before(source) && !source.owner_before(predecessor_) &&
           predecessor_.lock().get() == source.get();
}
NativeContactOwnerFunding CartesianSourceInventory::native_funding(
    const std::shared_ptr<const CartesianSourceInventory>& owner, std::size_t bytes) const
{
    auto funding = NativeContactFundingAccess::split_named<source_control_bytes>(owner, bytes,
        [](const CartesianSourceInventory& value) { return value.charge_ - source_control_bytes; },
        [](const CartesianSourceInventory& value) -> const auto& { return value.keeper_; });
    funding.related_control = predecessor_control_;
    return funding;
}
CartesianSourceInventory::~CartesianSourceInventory()
{
    // Drop actual immutable owners before returning the exclusive header.
    selectors_.reset();
    batches_.reset();
    decltype(leases_){}.swap(leases_);
    if (charge_)
        budget_->used -= charge_ - source_control_bytes;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
