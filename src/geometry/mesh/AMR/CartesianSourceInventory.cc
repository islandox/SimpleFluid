/** @file CartesianSourceInventory.cc
 * @brief Budgeted immutable Cartesian ancestry transitions without leaf catalogs.
 */
#include "CartesianSourceInventory.hh"
#include <algorithm>
#include <array>
#include <limits>
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
void retire(std::vector<CartesianIncidenceRange>& ranges, const CartesianCellAncestry& cell, Allocation& allocation)
{
    const auto at =
        std::find_if(ranges.begin(), ranges.end(), [&](const auto& range) { return range_contains(range, cell); });
    if (at == ranges.end())
        throw std::invalid_argument("Missing Cartesian inventory retirement selector.");
    const auto position = std::size_t(at - ranges.begin());
    auto remainder = *at;
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
        // Contains proves index + 1 <= upper, so the increment cannot wrap.
        const auto after = cell.index[axis] + 1;
        if (after < remainder.upper[axis])
        {
            auto slab = remainder;
            slab.lower[axis] = after;
            pieces[count++] = slab;
            remainder.upper[axis] = after;
        }
    }
    if (!count)
        ranges.erase(ranges.begin() + position);
    else
    {
        allocation.reserve(ranges, add(ranges.size(), count - 1));
        ranges[position] = pieces[0];
        for (std::size_t i = 1; i < count; ++i)
            ranges.push_back(pieces[i]);
    }
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

bool CartesianSourceInventory::Batch::contains(const CartesianCellAncestry& cell) const noexcept
{
    for (const auto& range : ranges_)
        if (range_contains(range, cell))
            return true;
    return false;
}
const CartesianSourceInventory::Batch* CartesianSourceInventory::lookup(const AdaptiveCellKey& key) const noexcept
{
    const auto* cell = std::get_if<CartesianCellAncestry>(&key.ancestry);
    if (!cell)
        return nullptr;
    for (const auto& batch : batches_)
        if (batch.source_->region() == key.region && batch.contains(*cell))
            return &batch;
    return nullptr;
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

    // Validate exact predecessor retirements before changing candidate selectors.
    // Delta-sized quadratic comparisons avoid a permanent per-cell catalog.
    for (std::size_t i = 0; i < retired.size(); ++i)
    {
        if (!old || !old->contains(retired[i].key) || old->revision(retired[i].key) != retired[i].revision)
            throw std::invalid_argument("Missing or stale Cartesian inventory retirement.");
        for (std::size_t j = 0; j < i; ++j)
            if (retired[i].key == retired[j].key)
                throw std::invalid_argument("Repeated Cartesian inventory retirement.");
    }
    std::span<const char> descriptor;
    if (old && !old->batches_.empty())
        descriptor = grid_descriptor(*old->batches_.front().source_);
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
    retained.retain(sizeof(CartesianSourceInventory) + 128);
    auto result = std::shared_ptr<CartesianSourceInventory>(new CartesianSourceInventory);
    result->budget_ = &budget;
    result->keeper_ = std::move(keeper);
    result->context_ = context;
    result->owner_ = owner;
    result->version_ = version;
    if (old)
        result->predecessor_ = old->weak_from_this();
    const auto completely_retired = [&](const Batch& batch)
    {
        std::size_t count = 0;
        for (const auto& item : retired)
            if (old->lookup(item.key) == &batch)
                ++count;
        return count == batch.cells_;
    };
    // Foreign immutable bodies coexist with candidate storage. Import their
    // exact owners before copying a single range, while omitting sources whose
    // entire active batch is retired by this authenticated delta.
    if (old)
        for (const auto& previous : old->batches_)
            if (!completely_retired(previous))
                credits.retain(previous.source_, previous.source_->allocation_budget(),
                    previous.source_->metadata_bytes(), previous.source_->has_arena_keeper(), result->leases_,
                    retained);
    for (const auto& captured : sources)
        if (captured.source->cell_count())
            credits.retain(captured.source, captured.source->allocation_budget(), captured.source->metadata_bytes(),
                captured.source->has_arena_keeper(), result->leases_, retained);
    retained.reserve(result->batches_, add(old ? old->batches_.size() : 0, sources.size()));
    if (old)
        for (const auto& previous : old->batches_)
        {
            if (completely_retired(previous))
                continue;
            result->batches_.emplace_back();
            auto& batch = result->batches_.back();
            batch.source_ = previous.source_;
            batch.environment_ = previous.environment_;
            retained.reserve(batch.ranges_, previous.ranges_.size());
            batch.ranges_.assign(previous.ranges_.begin(), previous.ranges_.end());
            batch.cells_ = previous.cells_;
        }
    for (const auto& item : retired)
    {
        const auto& cell = std::get<CartesianCellAncestry>(item.key.ancestry);
        bool found = false;
        for (auto& batch : result->batches_)
            if (batch.source_->region() == item.key.region && batch.contains(cell))
            {
                retire(batch.ranges_, cell, retained);
                --batch.cells_;
                found = true;
                break;
            }
        if (!found && !completely_retired(*old->lookup(item.key)))
            throw std::logic_error("Authenticated Cartesian retirement lost its predecessor selector.");
    }
    for (auto& batch : result->batches_)
        if (!batch.cells_)
        {
            const auto bytes = product(batch.ranges_.capacity(), sizeof(CartesianIncidenceRange));
            {
                std::vector<CartesianIncidenceRange> discarded;
                discarded.swap(batch.ranges_);
            }
            retained.bytes -= bytes;
            budget.used -= bytes;
        }
        else
            coalesce(batch.ranges_);
    std::erase_if(result->batches_, [](const auto& batch) { return batch.cells_ == 0; });
    result->added_begin_ = result->batches_.size();

    for (const auto& captured : sources)
    {
        if (!captured.source->cell_count())
            continue;
        result->batches_.emplace_back();
        auto& batch = result->batches_.back();
        batch.source_ = captured.source;
        batch.environment_ = captured.environment;
        retained.reserve(batch.ranges_, captured.source->ranges().size());
        batch.ranges_.assign(captured.source->ranges().begin(), captured.source->ranges().end());
        coalesce(batch.ranges_);
        for (const auto& range : batch.ranges_)
            batch.cells_ = add(batch.cells_, volume(range));
        if (batch.cells_ != captured.source->cell_count())
            throw std::invalid_argument("Trusted Cartesian source selectors disagree with its cell count.");
    }
    for (std::size_t i = 0; i < result->batches_.size(); ++i)
    {
        const auto& batch = result->batches_[i];
        for (std::size_t j = 0; j < i; ++j)
        {
            const auto& other = result->batches_[j];
            if (batch.source_->region() != other.source_->region())
                continue;
            for (const auto& first : batch.ranges_)
                for (const auto& second : other.ranges_)
                    if (overlap(first, second))
                        throw std::invalid_argument("Duplicate current Cartesian inventory ancestry.");
        }
        result->cells_ = add(result->cells_, batch.cells_);
        result->maximum_domain_bytes_ = std::max(result->maximum_domain_bytes_, batch.source_->domain().size());
    }

    // One exact reservation precedes Writer insertion. Full identities include
    // every actual axis value, domain, source context, revision and signature.
    auto size = std::size_t(96);
    for (const auto& batch : result->batches_)
        size = add(size, add(add(24, batch.source_->identity().size()), product(56, batch.ranges_.size())));
    Writer writer;
    retained.reserve(writer.data, size);
    writer.word(inventory_magic);
    writer.word(1);
    for (auto value : {context.epoch, context.topology, context.partition, context.round, context.transaction})
        writer.word(value);
    writer.word(std::uint64_t(owner));
    writer.word(version);
    writer.word(result->cells_);
    writer.word(result->batches_.size());
    writer.word(0); // Reserved fixed-format field; not a predecessor/history hash.
    for (const auto& batch : result->batches_)
    {
        writer.bytes(batch.source_->identity());
        writer.word(batch.cells_);
        writer.word(batch.ranges_.size());
        for (const auto& range : batch.ranges_)
        {
            writer.word(range.level);
            for (auto value : range.lower)
                writer.word(value);
            for (auto value : range.upper)
                writer.word(value);
        }
    }
    if (writer.data.size() != size)
        throw std::logic_error("Cartesian inventory identity differs from its preflight.");
    result->identity_ = std::move(writer.data);
    result->charge_ = retained.bytes;
    retained.bytes = 0;
    credits.committed = true;
    return result;
}
bool CartesianSourceInventory::derived_from(
    const std::shared_ptr<const CartesianSourceInventory>& source) const noexcept
{
    return source && !predecessor_.owner_before(source) && !source.owner_before(predecessor_) &&
           predecessor_.lock().get() == source.get();
}
CartesianSourceInventory::~CartesianSourceInventory()
{
    // Release actual allocations before their admission and keep the budget
    // keeper alive through the final decrement. No arena directory is pinned.
    {
        std::vector<Batch> discarded;
        discarded.swap(batches_);
    }
    {
        AdaptiveBytes discarded;
        discarded.swap(identity_);
    }
    {
        std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> discarded;
        discarded.swap(leases_);
    }
    if (budget_)
        budget_->used -= charge_;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
