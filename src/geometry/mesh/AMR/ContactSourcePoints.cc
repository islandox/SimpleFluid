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
    decltype(exceptions){}.swap(exceptions);
    if (budget)
        budget->used -= charge;
}
ContactSourcePointSnapshot::~ContactSourcePointSnapshot()
{
    indexes_ = ContactPointValueIndex{};
    blocks_.reset();
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
    visit_blocks(
        [&](const Block& block)
        {
            count += block->exceptions.size();
            return true;
        });
    return count;
}
std::size_t ContactSourcePointSnapshot::value_storage_bytes() const noexcept
{
    std::size_t bytes = indexes_.metadata_bytes();
    visit_blocks(
        [&](const Block& block)
        {
            bytes += block->metadata_bytes();
            return true;
        });
    return bytes;
}
std::size_t ContactSourcePointSnapshot::selector_width() const noexcept
{
    std::size_t width = 0;
    visit_blocks(
        [&](const Block& block)
        {
            const auto block_width = std::visit(
                [](const auto& values)
                {
                    using Element = typename std::remove_cvref_t<decltype(values)>::value_type;
                    return sizeof(typename Element::value_type);
                },
                block->selectors);
            width = std::max(width, block_width);
            return true;
        });
    return width;
}
const ContactSourcePointSnapshot::ValueBlock& ContactSourcePointSnapshot::block_at(
    std::size_t index, LookupWork* work) const
{
    const auto* at = BlockDirectory::lower_bound(blocks_,
        [index, work](const Block& block)
        {
            if (work)
                ++work->directory_comparisons;
            return block->first + block->count <= index;
        });
    if (!at || index < (*at)->first)
        throw std::out_of_range("Invalid certified source point reference.");
    return **at;
}
const ContactSourcePointSnapshot::Component& ContactSourcePointSnapshot::component(
    unsigned axis, std::size_t index, LookupWork* work) const
{
    const auto* at = BlockDirectory::lower_bound(blocks_,
        [axis, index, work](const Block& block)
        {
            if (work)
                ++work->directory_comparisons;
            return block->component_first[axis] + block->components[axis].size() <= index;
        });
    const auto* value = at ? (*at)->component_at(axis, index) : nullptr;
    if (!value)
        throw std::out_of_range("Invalid certified source point component.");
    return *value;
}
ContactPointValueIndex::Reader ContactSourcePointSnapshot::index_reader() const noexcept
{
    return {this, [](const void* context, std::size_t slot, LookupWork* work)
        { return static_cast<const ContactSourcePointSnapshot*>(context)->value({slot}, work); },
        [](const void* context, unsigned axis, std::size_t slot, LookupWork* work) -> const Component&
        { return static_cast<const ContactSourcePointSnapshot*>(context)->component(axis, slot, work); }};
}
std::optional<std::size_t> ContactSourcePointSnapshot::find_component(
    unsigned axis, long double scalar, LookupWork* work) const
{
    return indexes_.find_component(index_reader(), axis, scalar, work);
}
ContactSourcePoint ContactSourcePointSnapshot::value(ContactSourcePointReference reference, LookupWork* work) const
{
    if (reference.value >= count_)
        throw std::out_of_range("Invalid certified source point reference.");
    const auto& block = block_at(std::size_t(reference.value), work);
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
    const auto select = [&](unsigned axis) -> const Component&
    {
        if (const auto* local = block.component_at(axis, indices[axis]))
            return *local;
        return component(axis, indices[axis], work);
    };
    return Access::point_components({select(0), select(1), select(2)});
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
    const ContactSourcePoint& point, std::uint64_t route_hash, LookupWork* work) const
{
    const MetadataEnvironment environment;
    return indexes_.find(index_reader(), point, route_hash, work);
}
std::shared_ptr<const ContactSourcePointSnapshot> ContactSourcePointSnapshot::capture(
    const ContactSourcePointAdmission& admission, bool hashing, const ContactSourcePointRead& read, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena, bool canonical_hash)
{
    return capture_impl(admission, hashing, read, budget, std::move(keeper), std::move(arena), canonical_hash, nullptr);
}
std::shared_ptr<const ContactSourcePointSnapshot> ContactSourcePointSnapshot::capture_impl(
    const ContactSourcePointAdmission& admission, bool hashing, const ContactSourcePointRead& read, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<CartesianAuthorityArena> arena, bool canonical_hash,
    const std::shared_ptr<const ContactSourcePointSnapshot>* accepted_prefix)
{
    cartesian_detail::check_keeper(budget, keeper);
    if (!arena || arena->budget != &budget || !read || admission.owner_ < 0 || admission.layer_ >= 64)
        throw std::invalid_argument("Unbound or incomplete certified point source admission.");
    const auto& previous = admission.previousSnapshot_;
    if (accepted_prefix && !same_snapshot(previous, *accepted_prefix))
        throw std::invalid_argument("Certified point admission changed its accepted prefix owner.");
    if (previous && (!exact_control(previous) || previous->owner_ != admission.owner_ ||
                        !same_grid(previous->grid_, admission.grid_) || previous->hashing_ != hashing ||
                        previous->canonical_hash_ != canonical_hash || previous->count_ > admission.count_ ||
                        previous->generation_ >= admission.generation_))
        throw std::invalid_argument("Incompatible previous certified source point inventory.");
    if (!admission.manifests_ || admission.manifests_->context() != admission.context_ ||
        admission.manifests_->generation() != admission.generation_ ||
        !same_grid(admission.manifests_->grid(), admission.grid_))
        throw std::invalid_argument("Point source admission lacks its current closed source inventory.");
    Allocation memory(budget), scratch(budget), routes_memory(budget), values_memory(budget);
    memory.retain(sizeof(ContactSourcePointSnapshot));
    const auto snapshot_control = make_control(budget, keeper);
    cartesian_detail::CreditConstruction credits(arena);
    auto result = std::shared_ptr<ContactSourcePointSnapshot>(
        new ContactSourcePointSnapshot, [snapshot_control](auto* body) { delete body; });
    result->self_ = result;
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
    cartesian_detail::retain_point_manifest(result->manifests_, credits, result->leases_, memory);
    if (result->source_)
        retain(result->source_, result->source_->allocation_budget(), result->source_->metadata_bytes(),
            result->source_->has_arena_keeper());
    const auto value_build_start = budget.cumulative;
    const auto first = previous ? previous->count_ : 0;
    result->new_records_ = admission.count_ - first;
    result->indexes_ = previous ? previous->indexes_
                                : ContactPointValueIndex{!hashing         ? ContactPointValueIndex::Routing::Ordered
                                                         : canonical_hash ? ContactPointValueIndex::Routing::Canonical
                                                                          : ContactPointValueIndex::Routing::Custom};
    if (previous)
    {
        result->component_counts_ = previous->component_counts_;
        result->blocks_ = previous->blocks_;
        result->slot_namespace_ = previous->slot_namespace_;
    }
    using AxisMap = std::map<long double, std::size_t>;
    std::array<AxisMap, 3> axes;
    const MetadataEnvironment environment;
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
        const Block block = appended;
        const auto directory_arena = keeper ? keeper : std::shared_ptr<Budget>(std::shared_ptr<Budget>{}, &budget);
        const BlockDirectory directory(directory_arena, &result->directory_work_);
        result->blocks_ = directory.edit(result->blocks_, block, &block, false);
        if (!result->slot_namespace_)
            result->slot_namespace_ = block.get();
    }
    std::vector<std::array<std::size_t, 3>> selectors;
    std::vector<std::uint64_t> hashes;
    scratch.reserve(selectors, result->new_records_);
    selectors.resize(result->new_records_);
    if (hashing)
    {
        routes_memory.reserve(hashes, result->new_records_);
        hashes.resize(result->new_records_);
    }
    for (std::size_t i = accepted_prefix ? first : 0; i < admission.count_; ++i)
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
                    ++result->verified_prefix_records_;
                    return;
                }
                for (unsigned axis = 0; axis < 3; ++axis)
                {
                    const auto scalar = Access::point_component(point, axis);
                    auto at = axes[axis].lower_bound(scalar.value());
                    if (at == axes[axis].end() || at->first != scalar.value())
                    {
                        if (previous)
                            if (const auto old =
                                    previous->find_component(axis, scalar.value(), &result->index_work_.probes))
                            {
                                selectors[i - first][axis] = *old;
                                continue;
                            }
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
        // Sort only the private suffix dictionaries. Accepted component slots
        // and the exact selectors that reference them remain immutable.
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            auto& values = appended->components[axis];
            result->indexed_axis_values_ = add(result->indexed_axis_values_, values.size());
            if (values.size() < 2)
                continue;
            Allocation remap_memory(budget);
            std::vector<std::size_t> remap;
            remap_memory.reserve(remap, values.size());
            remap.resize(values.size());
            const auto first_component = appended->component_first[axis];
            std::size_t rank = 0;
            for (const auto& item : axes[axis])
                remap[item.second - first_component] = rank++;
            std::sort(values.begin(), values.end(),
                [](const Component& a, const Component& b) { return a.value() < b.value(); });
            for (auto& point : selectors)
                if (point[axis] >= first_component)
                    point[axis] = first_component + remap[point[axis] - first_component];
        }
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
        appended->budget = &budget;
        appended->keeper = keeper;
        appended->charge = std::exchange(values_memory.bytes, 0);
    }
    // Final selectors and component tokens now live in immutable value blocks.
    // Destroy their construction scratch before admitting independent index pages.
    decltype(axes){}.swap(axes);
    decltype(selectors){}.swap(selectors);
    budget.used -= std::exchange(scratch.bytes, 0);
    result->value_build_bytes_ = budget.cumulative - value_build_start;
    const auto index_build_start = budget.cumulative;
    if (appended)
    {
        const auto page_arena = keeper ? keeper : std::shared_ptr<Budget>(std::shared_ptr<Budget>{}, &budget);
        result->indexes_.append(result->index_reader(), appended->component_first, result->component_counts_, first,
            result->count_, hashes, page_arena, result->index_work_);
    }
    result->index_build_bytes_ = budget.cumulative - index_build_start;
    // Shared residency already covers the dependencies beneath an exact
    // admitted page. Standalone arenas acquire their own foreign allowances.
    visit_value_dependencies<true>(result->blocks_,
        [&](const auto& owner, const Budget* origin, std::size_t bytes, bool has_keeper)
        {
            if (arena->prepaid(owner, origin, bytes, false))
                return false;
            if (origin != &budget)
                retain(owner, origin, bytes, has_keeper);
            return true;
        });
    result->indexes_.visit_dependencies<true>(
        [&](const auto& owner, const Budget* origin, std::size_t bytes, bool has_keeper)
        {
            if (arena->prepaid(owner, origin, bytes, false))
                return false;
            if (origin != &budget)
                retain(owner, origin, bytes, has_keeper);
            return true;
        });
    result->budget_ = &budget;
    result->keeper_ = std::move(keeper);
    result->charge_ = std::exchange(memory.bytes, 0);
    credits.committed = true;
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
