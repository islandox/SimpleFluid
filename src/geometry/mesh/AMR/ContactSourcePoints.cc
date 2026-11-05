#include "ContactSourcePoints.hh"
#include "CartesianIncidenceBudget.hh"
#include "IdentityPoolHash.hh"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using Access = PLIC::details::GeometryIdentityAccess;
using cartesian_detail::add;
using cartesian_detail::product;
using Allocation = cartesian_detail::Allocation;
class MetadataEnvironment
{
    std::fenv_t caller_{};

public:
    MetadataEnvironment()
    {
        if (std::feholdexcept(&caller_))
            throw std::invalid_argument("Cannot isolate certified point metadata evaluation.");
    }
    ~MetadataEnvironment() { (void) std::fesetenv(&caller_); }
};
/** Kept in the deleter until the final weak control is destroyed. */
struct ControlCredit
{
    Budget* budget = nullptr;
    std::shared_ptr<Budget> keeper;
    std::size_t charge = 0;
    ~ControlCredit()
    {
        if (budget)
            budget->used -= charge;
    }
};
constexpr std::size_t control_bytes = sizeof(ControlCredit) + 2 * 128;
std::shared_ptr<ControlCredit> make_control(Budget& budget, const std::shared_ptr<Budget>& keeper)
{
    Allocation memory(budget);
    memory.retain(control_bytes);
    auto result = std::make_shared<ControlCredit>();
    result->budget = &budget;
    result->keeper = keeper;
    result->charge = std::exchange(memory.bytes, 0);
    return result;
}
bool full_equal(const ContactSourcePoint& first, const ContactSourcePoint& second)
{
    const auto bytes = PLIC::canonical_key_size(first);
    if (bytes != PLIC::canonical_key_size(second))
        return false;
    for (std::size_t offset = 0; offset < bytes; offset += 8)
        if (Access::canonical_word_at(first, offset) != Access::canonical_word_at(second, offset))
            return false;
    return true;
}
bool same_grid(const AMRGrid& a, const AMRGrid& b)
{
    if (a.root_blocks != b.root_blocks || a.block_cells != b.block_cells || a.periodic != b.periodic ||
        a.max_level != b.max_level || a.region_id != b.region_id || a.material_id != b.material_id)
        return false;
    for (unsigned axis = 0; axis < 3; ++axis)
        if (std::bit_cast<std::uint64_t>(a.origin[axis]) != std::bit_cast<std::uint64_t>(b.origin[axis]) ||
            std::bit_cast<std::uint64_t>(a.extent[axis]) != std::bit_cast<std::uint64_t>(b.extent[axis]))
            return false;
    return true;
}
} // namespace
ContactSourcePointSnapshot::ValueBlock::~ValueBlock()
{
    for (auto& values : components)
        std::remove_reference_t<decltype(values)>{}.swap(values);
    std::visit([](auto& values) { std::remove_reference_t<decltype(values)>{}.swap(values); }, selectors);
    std::visit([](auto& values) { std::remove_reference_t<decltype(values)>{}.swap(values); }, lookup);
    decltype(exceptions){}.swap(exceptions);
    if (budget)
        budget->used -= charge;
}
ContactSourcePointSnapshot::~ContactSourcePointSnapshot()
{
    decltype(blocks_){}.swap(blocks_);
    source_.reset();
    manifests_.reset();
    decltype(leases_){}.swap(leases_);
    if (budget_)
        budget_->used -= charge_;
}
std::array<std::size_t, 3> ContactSourcePointSnapshot::component_counts() const noexcept
{
    return component_counts_;
}
std::size_t ContactSourcePointSnapshot::exception_count() const noexcept
{
    std::size_t count = 0;
    for (const auto& block : blocks_)
        count += block->exceptions.size();
    return count;
}
std::size_t ContactSourcePointSnapshot::value_storage_bytes() const noexcept
{
    std::size_t bytes = 0;
    for (const auto& block : blocks_)
        bytes += block->metadata_bytes();
    return bytes;
}
std::size_t ContactSourcePointSnapshot::selector_width() const noexcept
{
    std::size_t width = 0;
    for (const auto& block : blocks_)
        width = std::max(width, std::visit(
                                    [](const auto& values)
                                    {
                                        using Element = typename std::remove_cvref_t<decltype(values)>::value_type;
                                        return sizeof(typename Element::value_type);
                                    },
                                    block->selectors));
    return width;
}
const ContactSourcePointSnapshot::ValueBlock& ContactSourcePointSnapshot::block_at(std::size_t index) const
{
    const auto at = std::upper_bound(blocks_.begin(), blocks_.end(), index,
        [](auto slot, const auto& block) { return slot < block->first + block->count; });
    if (at == blocks_.end() || index < (*at)->first)
        throw std::out_of_range("Invalid certified source point reference.");
    return **at;
}
const ContactSourcePointSnapshot::Component& ContactSourcePointSnapshot::component(
    unsigned axis, std::size_t index) const
{
    const auto at = std::upper_bound(blocks_.begin(), blocks_.end(), index, [axis](auto slot, const auto& block)
        { return slot < block->component_first[axis] + block->components[axis].size(); });
    if (at == blocks_.end() || index < (*at)->component_first[axis])
        throw std::out_of_range("Invalid certified source point component.");
    return (*at)->components[axis][index - (*at)->component_first[axis]];
}
ContactSourcePoint ContactSourcePointSnapshot::value(ContactSourcePointReference reference) const
{
    if (reference.value >= count_)
        throw std::out_of_range("Invalid certified source point reference.");
    const auto& block = block_at(std::size_t(reference.value));
    const auto at = std::lower_bound(block.exceptions.begin(), block.exceptions.end(), std::size_t(reference.value),
        [](const auto& entry, auto index) { return entry.first < index; });
    if (at != block.exceptions.end() && at->first == reference.value)
        return at->second;
    const auto indices = std::visit(
        [&](const auto& values)
        {
            const auto& v = values[reference.value - block.first];
            return std::array<std::size_t, 3>{v[0], v[1], v[2]};
        },
        block.selectors);
    return Access::point_components({component(0, indices[0]), component(1, indices[1]), component(2, indices[2])});
}
std::optional<std::array<std::size_t, 3>> ContactSourcePointSnapshot::component_indices(
    ContactSourcePointReference ref) const
{
    if (ref.value >= count_)
        return {};
    const auto& block = block_at(std::size_t(ref.value));
    const auto at = std::lower_bound(block.exceptions.begin(), block.exceptions.end(), std::size_t(ref.value),
        [](const auto& entry, auto index) { return entry.first < index; });
    if (at != block.exceptions.end() && at->first == ref.value)
        return {};
    return std::visit(
        [&](const auto& values)
        {
            const auto& v = values[ref.value - block.first];
            return std::array<std::size_t, 3>{v[0], v[1], v[2]};
        },
        block.selectors);
}
std::optional<std::size_t> ContactSourcePointSnapshot::find_components(const std::array<std::size_t, 3>& indices) const
{
    const MetadataEnvironment environment;
    if (hashing_ && !canonical_hash_)
        return {};
    for (unsigned axis = 0; axis < 3; ++axis)
        if (indices[axis] >= component_counts_[axis])
            return {};
    const auto point =
        Access::point_components({component(0, indices[0]), component(1, indices[1]), component(2, indices[2])});
    return find(point, hashing_ ? IdentityPoolHasher<ContactSourcePoint>{}(point) : 0);
}
std::optional<std::size_t> ContactSourcePointSnapshot::find(
    const ContactSourcePoint& point, std::uint64_t route_hash) const
{
    const MetadataEnvironment environment;
    for (const auto& block : blocks_)
    {
        const auto found = std::visit(
            [&](const auto& values) -> std::optional<std::size_t>
            {
                if (hashing_)
                {
                    using UInt = typename std::remove_cvref_t<decltype(values)>::value_type;
                    const auto empty = std::numeric_limits<UInt>::max();
                    const auto mask = values.size() - 1;
                    auto slot = std::size_t(route_hash) & mask;
                    for (std::size_t probes = 0; probes < values.size(); ++probes)
                    {
                        if (values[slot] == empty)
                            return {};
                        const auto candidate = std::size_t(values[slot]);
                        if (full_equal(value({candidate}), point))
                            return candidate;
                        slot = (slot + 1) & mask;
                    }
                    throw std::invalid_argument("Certified source lookup lacks an empty collision slot.");
                }
                std::size_t lower = 0, upper = values.size();
                while (lower < upper)
                {
                    const auto middle = lower + (upper - lower) / 2;
                    if (value({std::size_t(values[middle])}) < point)
                        lower = middle + 1;
                    else
                        upper = middle;
                }
                if (lower < values.size() && full_equal(value({std::size_t(values[lower])}), point))
                    return std::size_t(values[lower]);
                return {};
            },
            block->lookup);
        if (found)
            return found;
    }
    return {};
}
std::shared_ptr<const ContactSourcePointSnapshot> ContactSourcePointSnapshot::capture(
    const ContactSourcePointAdmission& admission, bool hashing, const ContactSourcePointRead& read, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena, bool canonical_hash)
{
    cartesian_detail::check_keeper(budget, keeper);
    if (!arena || arena->budget != &budget || !read || admission.owner_ < 0 || admission.layer_ >= 64)
        throw std::invalid_argument("Unbound or incomplete certified point source admission.");
    const auto& previous = admission.previousSnapshot_;
    if (previous && (previous->owner_ != admission.owner_ || !same_grid(previous->grid_, admission.grid_) ||
                        previous->hashing_ != hashing || previous->canonical_hash_ != canonical_hash ||
                        previous->count_ > admission.count_ || previous->generation_ >= admission.generation_))
        throw std::invalid_argument("Incompatible previous certified source point inventory.");
    if (!admission.manifests_ || admission.manifests_->context() != admission.context_ ||
        admission.manifests_->generation() != admission.generation_ ||
        !same_grid(admission.manifests_->grid(), admission.grid_))
        throw std::invalid_argument("Point source admission lacks its current closed source inventory.");
    Allocation memory(budget), scratch(budget), values_memory(budget);
    memory.retain(sizeof(ContactSourcePointSnapshot));
    const auto snapshot_control = make_control(budget, keeper);
    cartesian_detail::CreditConstruction credits(arena);
    auto result = std::shared_ptr<ContactSourcePointSnapshot>(
        new ContactSourcePointSnapshot, [snapshot_control](auto* body) { delete body; });
    result->control_charge_ = control_bytes;
    result->control_identity_ = snapshot_control.get();
    result->control_payload_bytes_ = sizeof(ControlCredit);
    result->grid_ = admission.grid_;
    result->context_ = admission.context_;
    result->owner_ = admission.owner_;
    result->generation_ = admission.generation_;
    result->layer_ = admission.layer_;
    result->count_ = admission.count_;
    result->evaluation_ = cartesian_evaluation_environment();
    result->hashing_ = hashing;
    result->canonical_hash_ = canonical_hash;
    result->source_ = admission.source_;
    result->manifests_ = admission.manifests_;
    auto retain = [&](const auto& owner, const Budget* origin, std::size_t bytes, bool has_keeper)
    {
        using Element = typename std::remove_cvref_t<decltype(owner)>::element_type;
        const std::shared_ptr<const std::remove_const_t<Element>> fixed = owner;
        credits.retain(fixed, origin, bytes, has_keeper, result->leases_, memory);
    };
    retain(result->manifests_, result->manifests_->allocation_budget(), result->manifests_->metadata_bytes(),
        result->manifests_->has_arena_keeper());
    result->manifests_->visit_dependencies(retain);
    if (result->source_)
        retain(result->source_, result->source_->allocation_budget(), result->source_->metadata_bytes(),
            result->source_->has_arena_keeper());
    const auto first = previous ? previous->count_ : 0;
    result->new_records_ = admission.count_ - first;
    const auto block_count = add(previous ? previous->blocks_.size() : 0, result->new_records_ ? 1 : 0);
    memory.reserve(result->blocks_, block_count);
    if (previous)
    {
        result->component_counts_ = previous->component_counts_;
        for (const auto& block : previous->blocks_)
        {
            retain(block, block->budget, block->metadata_bytes(), bool(block->keeper));
            result->blocks_.push_back(block);
        }
    }
    using AxisMap = std::map<long double, std::size_t>;
    std::array<AxisMap, 3> axes;
    const MetadataEnvironment environment;
    // Import components, never a full point copy or a historical proof object.
    if (previous)
        for (const auto& block : previous->blocks_)
            for (unsigned axis = 0; axis < 3; ++axis)
                for (std::size_t i = 0; i < block->components[axis].size(); ++i)
                {
                    scratch.retain(sizeof(AxisMap::value_type) + 4 * sizeof(void*));
                    axes[axis].emplace(block->components[axis][i].value(), block->component_first[axis] + i);
                }
    std::shared_ptr<ValueBlock> appended;
    if (result->new_records_)
    {
        values_memory.retain(sizeof(ValueBlock));
        const auto block_control = make_control(budget, keeper);
        appended = std::shared_ptr<ValueBlock>(new ValueBlock, [block_control](auto* body) { delete body; });
        appended->control_charge = control_bytes;
        appended->control_identity = block_control.get();
        appended->control_payload_bytes = sizeof(ControlCredit);
        appended->first = first;
        appended->count = result->new_records_;
        appended->component_first = result->component_counts_;
        result->blocks_.push_back(appended);
    }
    std::vector<std::array<std::size_t, 3>> selectors;
    std::vector<std::uint64_t> hashes;
    scratch.reserve(selectors, result->new_records_);
    selectors.resize(result->new_records_);
    if (hashing)
    {
        scratch.reserve(hashes, result->new_records_);
        hashes.resize(result->new_records_);
    }
    for (std::size_t i = 0; i < admission.count_; ++i)
    {
        bool supplied = false;
        read(i,
            [&](const ContactSourcePoint& point, std::uint64_t hash)
            {
                if (std::exchange(supplied, true))
                    throw std::invalid_argument("Duplicate certified point input.");
                if (i < first)
                {
                    if (!full_equal(point, previous->value({i})))
                        throw std::invalid_argument("Changed prior identity in certified source point inventory.");
                    // Recheck actual routing too; no old hash array is copied.
                    if (previous->find(point, hash) != std::optional<std::size_t>(i))
                        throw std::invalid_argument("Changed prior certified source point routing policy.");
                    return;
                }
                for (unsigned axis = 0; axis < 3; ++axis)
                {
                    const auto scalar = Access::point_component(point, axis);
                    auto at = axes[axis].lower_bound(scalar.value());
                    if (at == axes[axis].end() || at->first != scalar.value())
                    {
                        scratch.retain(sizeof(AxisMap::value_type) + 4 * sizeof(void*));
                        values_memory.append(appended->components[axis]);
                        const auto slot = result->component_counts_[axis];
                        const auto next = add(slot, 1);
                        appended->components[axis].push_back(scalar);
                        result->component_counts_[axis] = next;
                        at = axes[axis].emplace_hint(at, scalar.value(), slot);
                    }
                    selectors[i - first][axis] = at->second;
                }
                const auto& selected = selectors[i - first];
                const auto assembled = Access::point_components({result->component(0, selected[0]),
                    result->component(1, selected[1]), result->component(2, selected[2])});
                if (!full_equal(point, assembled))
                {
                    values_memory.append(appended->exceptions);
                    appended->exceptions.emplace_back(i, point);
                }
                if (hashing)
                    hashes[i - first] = hash;
            });
        if (!supplied)
            throw std::invalid_argument("Missing certified point input.");
    }
    if (appended)
    {
        std::size_t largest = 0;
        for (const auto count : result->component_counts_)
            if (count)
                largest = std::max(largest, count - 1);
        auto pack = [&]<class UInt>()
        {
            auto& values = appended->selectors.template emplace<Selectors<UInt>>();
            values_memory.reserve(values, selectors.size());
            for (const auto& value : selectors)
                values.push_back({UInt(value[0]), UInt(value[1]), UInt(value[2])});
        };
        if (largest <= std::numeric_limits<std::uint8_t>::max())
            pack.template operator()<std::uint8_t>();
        else if (largest <= std::numeric_limits<std::uint16_t>::max())
            pack.template operator()<std::uint16_t>();
        else if (largest <= std::numeric_limits<std::uint32_t>::max())
            pack.template operator()<std::uint32_t>();
        else
            pack.template operator()<std::uint64_t>();
        // Old blocks retain their own original widths and local lookup tables.
        auto index = [&]<class UInt>()
        {
            auto& values = appended->lookup.template emplace<std::vector<UInt>>();
            std::size_t size = appended->count;
            if (hashing)
            {
                size = 2;
                const auto required = product(appended->count, 2);
                while (size < required)
                    size = product(size, 2);
            }
            values_memory.reserve(values, size);
            values.resize(size, std::numeric_limits<UInt>::max());
            for (std::size_t i = first; i < admission.count_; ++i)
            {
                const auto point = result->value({i});
                const auto route = hashing ? hashes[i - first] : 0;
                if (previous && previous->find(point, route))
                    throw std::invalid_argument("Duplicate previous identity in certified source point suffix.");
                if (hashing)
                {
                    auto slot = std::size_t(route) & (size - 1);
                    while (values[slot] != std::numeric_limits<UInt>::max())
                    {
                        if (full_equal(result->value({std::size_t(values[slot])}), point))
                            throw std::invalid_argument(
                                "Duplicate actual identity in certified source point inventory.");
                        slot = (slot + 1) & (size - 1);
                    }
                    values[slot] = UInt(i);
                }
                else
                    values[i - first] = UInt(i);
            }
            if (!hashing)
            {
                std::sort(values.begin(), values.end(),
                    [&](auto a, auto b) { return result->value({std::size_t(a)}) < result->value({std::size_t(b)}); });
                for (std::size_t i = 1; i < values.size(); ++i)
                    if (full_equal(
                            result->value({std::size_t(values[i - 1])}), result->value({std::size_t(values[i])})))
                        throw std::invalid_argument("Duplicate actual identity in ordered source point inventory.");
            }
        };
        if (result->count_ <= std::numeric_limits<std::uint32_t>::max())
            index.template operator()<std::uint32_t>();
        else
            index.template operator()<std::uint64_t>();
        appended->budget = &budget;
        appended->keeper = keeper;
        appended->charge = std::exchange(values_memory.bytes, 0);
    }
    result->budget_ = &budget;
    result->keeper_ = std::move(keeper);
    result->charge_ = std::exchange(memory.bytes, 0);
    credits.committed = true;
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
