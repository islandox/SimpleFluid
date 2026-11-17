/** @file GeometryDemandOrder.cc
 * @brief Bounded compressed selector-page rewrites for sparse requested prefixes.
 */
#include "geometry/PLIC/details/GeometryDemandOrder.hh"
#include "geometry/PLIC/details/GeometryDemandPages.tcc"
#include <limits>
#include <variant>

namespace SimpleFluid::PLIC::details::geometry_demand
{
namespace
{
/** Bounded editing scratch is caller-stack storage; large requests retain an
 * exactly admitted heap fallback. No page/State retains this helper. */
template<class T> class EditRows
{
public:
    EditRows(std::size_t count, Admission admission) : count_(count)
    {
        if (count > leaf_capacity)
        {
            admission.charge(count, sizeof(T));
            heap_.resize(count);
            if (heap_.capacity() != count)
                throw GeometryError(GeometryErrorCode::ResourceLimit, "Unqualified geometry edit scratch capacity.");
        }
    }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::span<T> span() noexcept
    {
        return count_ <= leaf_capacity ? std::span<T>(inline_).first(count_) : std::span<T>(heap_);
    }
    auto begin() noexcept { return span().begin(); }
    auto end() noexcept { return span().end(); }
    T& operator[](std::size_t index) noexcept { return span()[index]; }

private:
    std::size_t count_;
    std::array<T, leaf_capacity> inline_{};
    std::vector<T> heap_;
};
bool before(OrderStamp a, OrderStamp b) noexcept
{
    return a.generation != b.generation ? a.generation > b.generation : a.position < b.position;
}
bool actual_owner(const std::shared_ptr<const void>& owner) noexcept
{
    const std::shared_ptr<const void> empty;
    return owner && (owner.owner_before(empty) || empty.owner_before(owner));
}
template<class T> struct Column
{
    struct Override
    {
        std::uint8_t row;
        T value;
    };
    T first{}, step{};
    std::uint8_t repeat = 1, period = leaf_capacity;
    std::variant<std::monostate, std::vector<Override>, std::vector<T>> exceptions;

    T pattern(std::size_t row) const noexcept { return T(first + step * T((row / repeat) % period)); }
    template<class Value> void build(std::size_t count, Value value, Admission admission)
    {
        first = value(0);
        std::size_t equal = 1;
        while (equal < count && value(equal) == first)
            ++equal;
        if (equal < count)
        {
            repeat = std::uint8_t(equal);
            step = T(value(equal) - first);
            for (std::size_t row = 2 * equal; row < count; row += equal)
                if (value(row) == first)
                {
                    period = std::uint8_t(row / equal);
                    break;
                }
        }
        std::size_t mismatches = 0;
        for (std::size_t row = 1; row < count; ++row)
            mismatches += value(row) != pattern(row);
        if (!mismatches)
            return;
        if (mismatches * sizeof(Override) < count * sizeof(T))
        {
            auto& overrides = exceptions.template emplace<std::vector<Override>>();
            admission.charge(mismatches, sizeof(Override));
            overrides.reserve(mismatches);
            if (overrides.capacity() != mismatches)
                throw GeometryError(GeometryErrorCode::ResourceLimit, "Unqualified demand override capacity.");
            for (std::size_t row = 1; row < count; ++row)
                if (value(row) != pattern(row))
                    overrides.push_back({std::uint8_t(row), value(row)});
        }
        else
        {
            auto& dense = exceptions.template emplace<std::vector<T>>();
            admission.charge(count, sizeof(T));
            dense.reserve(count);
            if (dense.capacity() != count)
                throw GeometryError(GeometryErrorCode::ResourceLimit, "Unqualified demand column capacity.");
            for (std::size_t row = 0; row < count; ++row)
                dense.push_back(value(row));
        }
    }
    T at(std::size_t row) const noexcept
    {
        if (const auto* dense = std::get_if<std::vector<T>>(&exceptions))
            return (*dense)[row];
        if (const auto* overrides = std::get_if<std::vector<Override>>(&exceptions))
        {
            const auto found = std::lower_bound(overrides->begin(), overrides->end(), row,
                [](const Override& value, std::size_t target) { return value.row < target; });
            if (found != overrides->end() && found->row == row)
                return found->value;
        }
        return pattern(row);
    }
    std::size_t payload_bytes() const noexcept
    {
        if (const auto* dense = std::get_if<std::vector<T>>(&exceptions))
            return dense->capacity() * sizeof(T);
        if (const auto* overrides = std::get_if<std::vector<Override>>(&exceptions))
            return overrides->capacity() * sizeof(Override);
        return 0;
    }
};
struct SourceOwner
{
    std::variant<std::shared_ptr<const void>, std::shared_ptr<const CartesianCellSource>> value;
    SourceKind kind() const noexcept { return value.index() == 0 ? SourceKind::Explicit : SourceKind::Cartesian; }
    const void* address() const noexcept
    {
        return std::visit([](const auto& owner) -> const void* { return owner.get(); }, value);
    }
    std::shared_ptr<const void> pin() const
    {
        return std::visit([](const auto& owner) -> std::shared_ptr<const void> { return owner; }, value);
    }
    bool matches(SourceKind type, const std::shared_ptr<const void>& owner) const noexcept
    {
        return kind() == type && std::visit([&](const auto& stored)
                                     { return !stored.owner_before(owner) && !owner.owner_before(stored); }, value);
    }
    static SourceOwner capture(const SourceSlot& slot)
    {
        SourceOwner result;
        if (slot.kind == SourceKind::Cartesian)
            result.value.emplace<1>(std::static_pointer_cast<const CartesianCellSource>(slot.source));
        else
            result.value.emplace<0>(slot.source);
        return result;
    }
};
struct SelectorPage
{
    std::vector<SourceOwner> sources;
    Column<std::uint8_t> owner;
    Column<std::uint64_t> row, generation, position;
    std::size_t count = 0;
    OrderedSourceSlot at(std::size_t slot, PageWork* work = nullptr) const
    {
        if (work)
            ++work->visited_source_rows;
        const auto& source = sources[owner.at(slot)];
        return {{generation.at(slot), position.at(slot)}, {source.pin(), row.at(slot), source.kind()}};
    }
    std::size_t payload_bytes() const noexcept
    {
        return sizeof(*this) + sources.capacity() * sizeof(sources[0]) + owner.payload_bytes() + row.payload_bytes() +
               generation.payload_bytes() + position.payload_bytes();
    }
    static std::shared_ptr<const SelectorPage> make(
        std::span<const OrderedSourceSlot> values, Admission admission, PageWork* work)
    {
        if (values.empty() || values.size() > leaf_capacity)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry selector page size.");
        std::array<SourceOwner, leaf_capacity> source_owners{};
        std::array<std::uint8_t, leaf_capacity> selected{};
        std::size_t sources = 0;
        for (std::size_t slot = 0; slot < values.size(); ++slot)
        {
            if (!actual_owner(values[slot].physical.source) ||
                (slot && !before(values[slot - 1].stamp, values[slot].stamp)))
                throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry selector page ownership/order.");
            std::size_t source = 0;
            while (source < sources && source_owners[source].address() != values[slot].physical.source.get())
                ++source;
            if (source == sources)
                source_owners[sources++] = SourceOwner::capture(values[slot].physical);
            else if (!source_owners[source].matches(values[slot].physical.kind, values[slot].physical.source))
                throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry selector address has foreign controls.");
            selected[slot] = std::uint8_t(source);
        }
        admission.charge(1, sizeof(SelectorPage) + control_bytes);
        auto result = std::make_shared<SelectorPage>();
        admission.charge(sources, sizeof(result->sources[0]));
        result->sources.reserve(sources);
        if (result->sources.capacity() != sources)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Unqualified geometry source directory capacity.");
        result->sources.assign(source_owners.begin(), source_owners.begin() + sources);
        result->count = values.size();
        result->owner.build(values.size(), [&](auto slot) { return selected[slot]; }, admission);
        result->row.build(values.size(), [&](auto slot) { return values[slot].physical.row; }, admission);
        result->generation.build(values.size(), [&](auto slot) { return values[slot].stamp.generation; }, admission);
        result->position.build(values.size(), [&](auto slot) { return values[slot].stamp.position; }, admission);
        if (work)
        {
            ++work->copied_pages;
            work->copied_source_rows += values.size();
            work->copied_payload_bytes = checked_sum(work->copied_payload_bytes, result->payload_bytes());
            work->copied_bytes = checked_sum(work->copied_bytes, result->payload_bytes() + control_bytes);
        }
        return result;
    }
};
struct Group
{
    std::shared_ptr<const SelectorPage> page;
    OrderStamp maximum;
    [[nodiscard]] std::size_t logical_size() const noexcept { return page->count; }
    [[nodiscard]] std::size_t exclusive_payload_bytes() const noexcept { return page->payload_bytes(); }
    [[nodiscard]] std::size_t exclusive_admission_bytes() const noexcept
    {
        return page->payload_bytes() + control_bytes;
    }
};
struct GroupLess
{
    bool operator()(const Group& a, const Group& b) const noexcept { return before(a.maximum, b.maximum); }
};
using Pages = OrderedPages<Group, GroupLess>;
Group group(std::shared_ptr<const SelectorPage> page)
{
    const auto last = page->at(page->count - 1).stamp;
    return {std::move(page), last};
}
const Group* containing(const Pages::Ptr& root, OrderStamp stamp, PageWork* work)
{
    return Pages::lower_bound(root, [&](const Group& value) { return before(value.maximum, stamp); }, work);
}
std::size_t position(const Group& value, OrderStamp stamp, PageWork* work = nullptr)
{
    std::size_t low = 0, high = value.page->count;
    while (low < high)
    {
        const auto mid = low + (high - low) / 2;
        if (before(value.page->at(mid, work).stamp, stamp))
            low = mid + 1;
        else
            high = mid;
    }
    if (low == value.page->count || value.page->at(low, work).stamp != stamp)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry selector stamp is absent.");
    return low;
}
Pages::Ptr replace(Pages::Ptr root, std::span<const Group> retired, std::span<const OrderedSourceSlot> values,
    Admission admission, PageWork* work)
{
    const Pages editor(admission, work);
    for (const auto& value : retired)
        root = editor.edit(root, value, nullptr, true);
    const auto groups = values.size() / leaf_capacity + (values.size() % leaf_capacity != 0);
    std::size_t start = 0;
    for (std::size_t index = 0; index < groups; ++index)
    {
        const auto count = values.size() / groups + (index < values.size() % groups);
        auto value = group(SelectorPage::make(values.subspan(start, count), admission, work));
        root = editor.edit(root, value, &value, false);
        start += count;
    }
    return root;
}
Pages::Ptr erased(Pages::Ptr root, OrderStamp stamp, Admission admission, PageWork* work)
{
    const auto* found = containing(root, stamp, work);
    if (!found)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry selector stamp is absent.");
    const Group old = *found;
    const auto removed = position(old, stamp, work);
    std::array<Group, 2> retired{old, {}};
    std::array<OrderedSourceSlot, 2 * leaf_capacity> values{};
    std::size_t count = 0, retiring = 1;
    for (std::size_t slot = 0; slot < old.page->count; ++slot)
        if (slot != removed)
            values[count++] = old.page->at(slot, work);
    if (count < leaf_capacity / 2 && root->size > 1)
    {
        const auto index = Pages::rank(root, [&](const Group& value) { return GroupLess{}(value, old); }, work);
        const bool previous = index > 0;
        const Group neighbor = Pages::at(root, previous ? index - 1 : index + 1, work);
        retired[retiring++] = neighbor;
        if (previous)
        {
            std::move_backward(values.begin(), values.begin() + count, values.begin() + count + neighbor.page->count);
            for (std::size_t slot = 0; slot < neighbor.page->count; ++slot)
                values[slot] = neighbor.page->at(slot, work);
        }
        else
            for (std::size_t slot = 0; slot < neighbor.page->count; ++slot)
                values[count + slot] = neighbor.page->at(slot, work);
        count += neighbor.page->count;
    }
    return replace(
        std::move(root), std::span(retired).first(retiring), std::span(values).first(count), admission, work);
}
} // namespace
struct GeometryDemandOrder::State
{
    Pages::Ptr root;
    std::uint64_t generation;
    State(Pages::Ptr value, std::uint64_t stamp) : root(std::move(value)), generation(stamp) {}
};
GeometryDemandOrder GeometryDemandOrder::prepended(
    std::span<const SourceSlot> prefix, std::span<const OrderStamp> retired, Admission admission, PageWork* work) const
{
    if (prefix.empty() && retired.empty())
        return *this;
    if (!prefix.empty() && generation() == std::numeric_limits<std::uint64_t>::max())
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry demand order generation overflows.");
    for (const auto& value : prefix)
        if (!actual_owner(value.source) || (value.kind != SourceKind::Explicit && value.kind != SourceKind::Cartesian))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry demand source owner/kind is invalid.");
    EditRows<OrderStamp> sorted(retired.size(), admission);
    std::copy(retired.begin(), retired.end(), sorted.begin());
    std::sort(sorted.begin(), sorted.end(), before);
    for (std::size_t index = 0; index < sorted.size(); ++index)
    {
        if (index && sorted[index] == sorted[index - 1])
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry demand retires a duplicate stamp.");
        const auto* value = state_ ? containing(state_->root, sorted[index], work) : nullptr;
        if (!value)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry selector stamp is absent.");
        (void) position(*value, sorted[index], work);
    }
    auto root = state_ ? state_->root : nullptr;
    for (const auto& stamp : sorted)
        root = erased(std::move(root), stamp, admission, work);
    const auto next_generation = generation() + std::uint64_t(!prefix.empty());
    if (!prefix.empty())
    {
        std::array<Group, 1> retired_front{};
        std::size_t front = 0;
        if (root)
        {
            retired_front[0] = Pages::at(root, 0, work);
            front = retired_front[0].page->count;
        }
        const auto count = checked_sum(prefix.size(), front);
        EditRows<OrderedSourceSlot> values(count, admission);
        for (std::size_t index = 0; index < prefix.size(); ++index)
            values[index] = {{next_generation, index}, prefix[index]};
        for (std::size_t index = 0; index < front; ++index)
            values[prefix.size() + index] = retired_front[0].page->at(index, work);
        root = replace(std::move(root), std::span(retired_front).first(front ? 1 : 0), values.span(), admission, work);
    }
    GeometryDemandOrder result;
    if (root || next_generation)
    {
        admission.charge(1, sizeof(State) + control_bytes);
        result.state_ = std::make_shared<const State>(std::move(root), next_generation);
    }
    return result;
}
GeometryDemandOrder GeometryDemandOrder::rebound(
    std::span<const OrderedSourceSlot> supplied, Admission admission, PageWork* work) const
{
    if (supplied.empty())
        return *this;
    EditRows<OrderedSourceSlot> updates(supplied.size(), admission);
    std::copy(supplied.begin(), supplied.end(), updates.begin());
    std::sort(updates.begin(), updates.end(), [](const auto& a, const auto& b) { return before(a.stamp, b.stamp); });
    for (std::size_t index = 0; index < updates.size(); ++index)
    {
        const auto& value = updates[index];
        if (!actual_owner(value.physical.source) ||
            (value.physical.kind != SourceKind::Explicit && value.physical.kind != SourceKind::Cartesian) ||
            (index && updates[index - 1].stamp == value.stamp))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry source rebound.");
        const auto* current = state_ ? containing(state_->root, value.stamp, work) : nullptr;
        if (!current)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Rebound geometry selector stamp is absent.");
        (void) position(*current, value.stamp, work);
    }
    auto root = state_->root;
    const Pages editor(admission, work);
    std::size_t cursor = 0;
    while (cursor < updates.size())
    {
        const Group old = *containing(root, updates[cursor].stamp, work);
        std::array<OrderedSourceSlot, leaf_capacity> rows{};
        for (std::size_t slot = 0; slot < old.page->count; ++slot)
            rows[slot] = old.page->at(slot, work);
        do
        {
            const auto& value = updates[cursor++];
            rows[position(old, value.stamp, work)].physical = value.physical;
        } while (cursor < updates.size() && !before(old.maximum, updates[cursor].stamp));
        auto replacement = group(SelectorPage::make(std::span(rows).first(old.page->count), admission, work));
        root = editor.edit(root, old, &replacement, true);
    }
    admission.charge(1, sizeof(State) + control_bytes);
    GeometryDemandOrder result;
    result.state_ = std::make_shared<const State>(std::move(root), generation());
    return result;
}
void GeometryDemandOrder::visit_dependencies(MetadataSink metadata, SourceSink sources) const
{
    if (!state_ || !metadata(state_, sizeof(State), sizeof(State) + control_bytes))
        return;
    Pages::visit_dependencies(state_->root, metadata,
        [&](const Group& value)
        {
            const auto payload = value.page->payload_bytes();
            if (metadata(value.page, payload, payload + control_bytes))
                for (const auto& source : value.page->sources)
                    sources(source.kind(), source.pin());
        });
}
void GeometryDemandOrder::visit_borrowed_dependencies(
    BorrowedMetadataSink metadata, BorrowedSourceSink sources, const void* parent) const
{
    if (!state_ || !metadata(state_.get(), parent, sizeof(State)))
        return;
    Pages::visit_borrowed_dependencies(
        state_->root, metadata,
        [&](const Group& value, const void* node)
        {
            if (metadata(value.page.get(), node, value.page->payload_bytes()))
                for (const auto& source : value.page->sources)
                    sources(value.page.get(), source.kind(), source.address());
        },
        state_.get());
}
std::size_t GeometryDemandOrder::size() const noexcept
{
    return state_ && state_->root ? state_->root->logical_size : 0;
}
OrderedSourceSlot GeometryDemandOrder::at(std::size_t slot) const
{
    const auto value = Pages::logical_at(state_ ? state_->root : nullptr, slot);
    return value.row->page->at(value.offset);
}
const std::shared_ptr<const CartesianCellSource>& GeometryDemandOrder::native_owner(std::size_t slot) const
{
    const auto value = Pages::logical_at(state_ ? state_->root : nullptr, slot);
    const auto& page = *value.row->page;
    return std::get<1>(page.sources[page.owner.at(value.offset)].value);
}
std::size_t GeometryDemandOrder::rank(OrderStamp stamp) const
{
    const auto* value = state_ ? containing(state_->root, stamp, nullptr) : nullptr;
    if (!value)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry selector stamp is absent.");
    return Pages::logical_rank(state_->root, [&](const Group& row) { return GroupLess{}(row, *value); }) +
           position(*value, stamp);
}
std::uint64_t GeometryDemandOrder::generation() const noexcept
{
    return state_ ? state_->generation : 0;
}
std::size_t GeometryDemandOrder::metadata_admission_bytes() const noexcept
{
    if (!state_)
        return 0;
    return sizeof(State) + control_bytes +
           (state_->root ? state_->root->subtree_bytes + state_->root->subtree_owned_admission_bytes : 0);
}
std::size_t GeometryDemandOrder::metadata_payload_bytes() const noexcept
{
    if (!state_)
        return 0;
    return sizeof(State) + (state_->root ? state_->root->subtree_payload_bytes + state_->root->subtree_owner_bytes : 0);
}
void GeometryDemandOrder::visit_metadata(const std::function<void(const void*, std::size_t)>& visitor) const
{
    if (!state_)
        return;
    visitor(state_.get(), sizeof(State));
    Pages::visit_pages(state_->root,
        [&](const auto& node)
        {
            visitor(node.get(),
                sizeof(*node) + node->rows.capacity() * sizeof(Group) + node->children.capacity() * sizeof(Pages::Ptr));
            for (const auto& row : node->rows)
                visitor(row.page.get(), row.page->payload_bytes());
        });
}
void GeometryDemandOrder::visit_source_pages(
    const std::function<void(SourceKind, const std::shared_ptr<const void>&)>& visitor) const
{
    if (state_)
        Pages::visit_from(
            state_->root, [](const auto&) { return false; },
            [&](const auto& group)
            {
                for (const auto& source : group.page->sources)
                    visitor(source.kind(), source.pin());
                return true;
            });
}
} // namespace SimpleFluid::PLIC::details::geometry_demand
