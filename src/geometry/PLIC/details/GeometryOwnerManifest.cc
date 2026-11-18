/** @file GeometryOwnerManifest.cc
 * @brief Compiled persistent exact owner bindings and first/last reference accounting.
 */
#include "geometry/PLIC/details/GeometryOwnerManifest.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryDemandPages.tcc"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include <limits>

namespace SimpleFluid::PLIC::details::geometry_demand
{
namespace
{
struct OwnerLess
{
    bool operator()(const OwnerRecord& a, const OwnerRecord& b) const noexcept
    {
        if (a.key.channel != b.key.channel)
            return a.key.channel < b.key.channel;
        if (a.key.kind != b.key.kind)
            return a.key.kind < b.key.kind;
        return std::less<const void*>{}(a.key.address, b.key.address);
    }
};
using Pages = OrderedPages<OwnerRecord, OwnerLess>;
struct TypedLess
{
    bool operator()(const TypedOwnerReference& a, const TypedOwnerReference& b) const noexcept
    { return std::less<const void*>{}(a.address, b.address); }
};
using TypedPages = OrderedPages<TypedOwnerReference, TypedLess>;
OwnerRecord lookup(std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>& owner)
{
    const std::shared_ptr<const void> empty;
    if (!owner || (!owner.owner_before(empty) && !empty.owner_before(owner)))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry manifest requires an actual owner control.");
    return {{channel, kind, owner.get()}, owner, 0, 0};
}
bool same_owner(const std::shared_ptr<const void>& first, const std::shared_ptr<const void>& second) noexcept
{
    return first.get() == second.get() && !first.owner_before(second) && !second.owner_before(first);
}
const OwnerRecord* find(const Pages::Ptr& root, const OwnerRecord& key, PageWork* work = nullptr)
{
    PageWork local;
    auto& measured = work ? *work : local;
    const auto before = measured.visited_pages;
    const auto* row = Pages::lower_bound(root, [&](const auto& value) { return OwnerLess{}(value, key); }, &measured);
    GeometryAllocationOwnerAccess::observe_work(
        GeometryAllocationOwnerAccess::Work::ManifestLookupPage, measured.visited_pages - before);
    if (!row || OwnerLess{}(key, *row))
        return nullptr;
    if (!same_owner(row->owner, key.owner))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry allocation address has conflicting controls.");
    return row;
}
std::size_t tree_page_bound(std::size_t rows)
{
    if (!rows)
        return 0;
    auto level = std::max(std::size_t{1}, rows / (leaf_capacity / 2));
    auto pages = level;
    while (level > 1)
    {
        level = std::max(std::size_t{1}, level / (branch_capacity / 2));
        pages = checked_sum(pages, level);
    }
    return pages;
}
template<class Node> std::size_t reservation_height(const Node* node)
{
    std::size_t height = 1;
    for (; node && !node->leaf(); node = node->children.front().get())
    {
        GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ReservationPathPage);
        height = checked_sum(height, 1);
    }
    if (node)
        GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ReservationPathPage);
    return height;
}
std::size_t height_scale(std::size_t current, std::size_t rows)
{
    auto level = std::max(std::size_t{1}, rows / (leaf_capacity / 2));
    std::size_t future = 1;
    while (level > 1)
    {
        level = std::max(std::size_t{1}, level / (branch_capacity / 2));
        future = checked_sum(future, 1);
    }
    const auto numerator = checked_sum(std::max(current, future), 1);
    const auto denominator = checked_sum(current, 1);
    return numerator / denominator + (numerator % denominator != 0);
}
std::size_t bound_product(std::size_t count, std::size_t width)
{
    if (width && count > std::numeric_limits<std::size_t>::max() / width)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry batch envelope overflows.");
    return count * width;
}
std::size_t typed_edit_nodes(std::size_t keys, std::size_t height)
{
    // Each affected search path can repack two pages and redistribute with a
    // neighbor into two pages at each level. Include one path for root growth.
    return bound_product(bound_product(4, checked_sum(keys, 1)), checked_sum(height, 1));
}
} // namespace
template<> inline constexpr bool stack_safe_default<TypedOwnerReference> = true;
template<> inline constexpr bool native_flag_in_owned_admission<TypedOwnerReference> = true;
struct GeometryOwnerManifest::State
{
    Pages::Ptr root;
    // The legacy inline record and the indexed tree's optional typed pointer
    // occupy the same storage. A nonempty primary root selects indexed form.
    union Contents
    {
        OwnerRecord single;
        TypedPages::Ptr typed;

        explicit Contents(OwnerRecord value) : single(std::move(value)) {}
        explicit Contents(TypedPages::Ptr roles) : typed(std::move(roles)) {}
        ~Contents() {}
    } contents;

    State(Pages::Ptr value, TypedPages::Ptr roles) : root(std::move(value)), contents(std::move(roles)) {}
    explicit State(OwnerRecord value) : contents(std::move(value)) {}
    ~State()
    {
        if (inline_row())
            std::destroy_at(&contents.single);
        else
            std::destroy_at(&contents.typed);
    }
    [[nodiscard]] bool inline_row() const noexcept { return !root; }
    [[nodiscard]] const OwnerRecord& single() const noexcept { return contents.single; }
    [[nodiscard]] TypedPages::Ptr typed() const noexcept { return root ? contents.typed : TypedPages::Ptr{}; }

    template<class Value> static std::shared_ptr<const State> created(
        Value value, Admission admission, TypedPages::Ptr roles = {})
    {
        static_assert(sizeof(State) == sizeof(Pages::Ptr) + sizeof(OwnerRecord));
        std::shared_ptr<const State> owner;
        if constexpr (std::is_same_v<Value, OwnerRecord>)
        {
            if (roles)
            {
                // A typed singleton still owns a real indexed primary leaf.
                auto primary = Pages(admission).build(std::span(&value, 1));
                return created(std::move(primary), admission, std::move(roles));
            }
            owner = admission.shared<State>(std::move(value));
        }
        else
        {
            if (!value)
                throw GeometryError(GeometryErrorCode::InvalidInput, "Indexed geometry manifest lacks a primary root.");
            owner = admission.shared<State>(std::move(value), std::move(roles));
        }

        const auto bytes = sizeof(State) + (admission.native ? metadata_control_bytes(owner) : 0);
        admission.created(owner, bytes, bytes + (admission.native ? 0 : control_bytes));
        return owner;
    }
};
bool GeometryOwnerManifest::same_root(const GeometryOwnerManifest& other) const noexcept
{
    return same_owner(state_, other.state_);
}
bool GeometryOwnerManifest::authenticates(const std::weak_ptr<const void>& other) const noexcept
{
    return other.lock().get() == state_.get() && !other.owner_before(state_) && !state_.owner_before(other);
}
std::shared_ptr<const void> GeometryOwnerManifest::root_control() const noexcept
{
    return state_;
}
std::size_t GeometryOwnerManifest::root_metadata_bytes() const noexcept
{
    return state_ ? sizeof(State) +
                        (geometry_demand::metadata_state(state_) ? geometry_demand::metadata_control_bytes(state_) : 0)
                  : 0;
}
void GeometryOwnerManifest::visit_added(const GeometryOwnerManifest& previous, RecordSink records, PageWork* work) const
{
    if (!state_ || same_root(previous))
        return;
    const auto row = [&](const OwnerRecord& value)
    {
        const auto count = previous.references(value.key.channel, value.key.kind, value.owner, work);
        if (count)
        {
            if (!previous.admits(value.key.channel, value.key.kind, value.owner, value.bytes, work))
                throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry delta changes an immutable owner size.");
        }
        else
            records(value);
    };
    const auto shared = [&](const Pages::Ptr& candidate)
    {
        auto old = previous.state_ && !previous.state_->inline_row() ? previous.state_->root : Pages::Ptr{};
        while (old)
        {
            if (work)
                ++work->visited_pages;
            GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ManifestLookupPage);
            if (same_owner(old, candidate))
                return true;
            if (old->leaf() || OwnerLess{}(*old->maximum, *candidate->maximum))
                return false;
            const auto child = std::lower_bound(old->children.begin(), old->children.end(), *candidate->maximum,
                [](const Pages::Ptr& item, const OwnerRecord& key) { return OwnerLess{}(*item->maximum, key); });
            if (child == old->children.end())
                return false;
            old = *child;
        }
        return false;
    };
    const auto visit = [&](const auto& self, const Pages::Ptr& node) -> void
    {
        if (work)
            ++work->visited_pages;
        GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ManifestLookupPage);
        if (shared(node))
            return;
        if (node->leaf())
            for (const auto& value : node->rows)
                row(value);
        else
            for (const auto& child : node->children)
                self(self, child);
    };
    if (state_->inline_row())
        row(state_->single());
    else
        visit(visit, state_->root);
}
void GeometryOwnerManifest::visit_dependencies(MetadataSink metadata, RecordSink records) const
{
    if (!state_ ||
        !metadata(state_, root_metadata_bytes(), root_metadata_bytes() + (metadata_state(state_) ? 0 : control_bytes)))
        return;
    if (geometry_demand::metadata_state(state_))
        for (unsigned edge = 0; edge != 2; ++edge)
            visit_native_state(geometry_demand::metadata_state(state_), metadata);
    TypedPages::visit_dependencies(state_->typed(), metadata, [](const auto&) {});
    if (state_->inline_row())
        records(state_->single());
    else
        Pages::visit_dependencies(state_->root, metadata, records);
}
bool GeometryOwnerManifest::has_native_administration() const noexcept
{
    return state_ &&
           (geometry_demand::metadata_state(state_) || (state_->root && state_->root->has_native_administration()) ||
               (state_->typed() && state_->typed()->has_native_administration()));
}
void GeometryOwnerManifest::visit_native_administration_delta(
    const GeometryOwnerManifest& previous, NativeAdministrationSink sink, PageWork* work) const
{
    if (same_root(previous))
        return;
    if (state_)
    {
        sink.owner(state_, true);
        sink(geometry_demand::metadata_state(state_), 2, true);
    }
    if (previous.state_)
    {
        sink.owner(previous.state_, false);
        sink(geometry_demand::metadata_state(previous.state_), 2, false);
    }
    Pages::visit_native_administration_delta(
        previous.state_ ? previous.state_->root : nullptr, state_ ? state_->root : nullptr, sink, work);
    TypedPages::visit_native_administration_delta(
        previous.state_ ? previous.state_->typed() : nullptr, state_ ? state_->typed() : nullptr, sink, work);
}
void GeometryOwnerManifest::visit_borrowed_dependencies(
    BorrowedMetadataSink metadata, BorrowedRecordSink records, const void* parent) const
{
    if (!state_ || !metadata(state_.get(), parent, sizeof(State)))
        return;
    TypedPages::visit_borrowed_dependencies(state_->typed(), metadata, [](const auto&, const void*) {}, state_.get());
    if (state_->inline_row())
        records(state_->single(), state_.get());
    else
        Pages::visit_borrowed_dependencies(state_->root, metadata, records, state_.get());
}
GeometryOwnerManifest GeometryOwnerManifest::built(
    std::span<const OwnerRecord> rows, Admission admission, PageWork* work)
{
    GeometryOwnerManifest result;
    if (rows.empty())
        return result;
    for (const auto& row : rows)
    {
        (void) lookup(row.key.channel, row.key.kind, row.owner);
        if (!row.references || row.bytes == std::numeric_limits<std::size_t>::max())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry owner set has an unusable row.");
        if (row.key.address != row.owner.get())
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry owner row address differs from its control.");
    }
    for (std::size_t index = 1; index < rows.size(); ++index)
        if (!OwnerLess{}(rows[index - 1], rows[index]))
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry owner set rows are not in strict canonical order.");
    admission.charge(1, sizeof(State) + control_bytes);
    if (rows.size() == 1)
        result.state_ = State::created(rows[0], admission);
    else
        result.state_ = State::created(Pages(admission, work).build(rows), admission);
    return result;
}
std::size_t GeometryOwnerManifest::metadata_owner_count_bound(std::size_t rows, std::size_t typed_rows)
{
    if (typed_rows > rows)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Typed geometry owner bound exceeds physical keys.");
    if (!rows)
        return 0;
    // Every non-root leaf has at least 32 rows, and every non-root
    // branch at least 16 children. The root alone may be underfull.
    return checked_sum(1, checked_sum(tree_page_bound(rows), tree_page_bound(typed_rows)));
}
std::size_t GeometryOwnerManifest::metadata_owner_count_bound() const
{
    return metadata_owner_count_bound(owner_count(), typed_owner_count());
}
std::size_t GeometryOwnerManifest::edit_height_scale_bound(
    std::size_t anticipated_additions, std::optional<std::size_t> anticipated_typed_additions) const
{
    // The legacy bound's only height terms are H+1 and H+2. This
    // ceiling ratio dominates both without changing that bound's policy.
    auto scale = height_scale(reservation_height(state_ ? state_->root.get() : nullptr),
        checked_sum(owner_count(), anticipated_additions));
    if (anticipated_typed_additions)
        scale = std::max(scale, height_scale(reservation_height(state_ ? state_->typed().get() : nullptr),
            checked_sum(typed_owner_count(), *anticipated_typed_additions)));
    return scale;
}
std::size_t GeometryOwnerManifest::batch_edit_metadata_owner_bound(
    std::size_t changed_keys, std::size_t typed_keys) const
{
    if (!changed_keys)
        return typed_keys ? checked_sum(state_ && state_->inline_row() ? 2 : 1,
            typed_edit_nodes(typed_keys, reservation_height(state_ ? state_->typed().get() : nullptr))) : 0;
    const auto height = reservation_height(state_ ? state_->root.get() : nullptr);
    const auto paths = checked_sum(changed_keys, 1);
    if (paths > (std::numeric_limits<std::size_t>::max() - 2) / 2 / checked_sum(height, 1))
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry index-owner envelope overflows.");
    auto owners = 2 * paths * (height + 1) + 2;
    if (typed_keys)
        owners = checked_sum(owners, typed_edit_nodes(typed_keys, reservation_height(state_ ? state_->typed().get() : nullptr)));
    return owners;
}
std::size_t GeometryOwnerManifest::batch_edit_byte_bound(std::size_t changed_keys, std::size_t typed_keys) const
{
    if (!changed_keys && !typed_keys)
        return 0;
    auto bytes = sizeof(State) + control_bytes;
    if (changed_keys)
    {
        const auto height = reservation_height(state_ ? state_->root.get() : nullptr);
        const auto maximum_node = sizeof(Pages::Node) + control_bytes +
            std::max(leaf_capacity * sizeof(OwnerRecord), branch_capacity * sizeof(Pages::Ptr));
        // A changed key can split its complete path and merge an adjacent path.
        const auto nodes = bound_product(bound_product(2, checked_sum(changed_keys, 1)), checked_sum(height, 1));
        bytes = checked_sum(bytes, bound_product(nodes, maximum_node));
        bytes = checked_sum(bytes, bound_product(checked_sum(leaf_capacity, changed_keys), sizeof(OwnerRecord)));
        bytes = checked_sum(bytes, bound_product(bound_product(checked_sum(height, 2),
            checked_sum(branch_capacity, changed_keys)), 3 * sizeof(Pages::Ptr)));
    }
    if (typed_keys)
    {
        const auto height = reservation_height(state_ ? state_->typed().get() : nullptr);
        const auto nodes = typed_edit_nodes(typed_keys, height);
        constexpr auto maximum_node = sizeof(TypedPages::Node) + control_bytes +
            std::max(leaf_capacity * sizeof(TypedOwnerReference), branch_capacity * sizeof(TypedPages::Ptr));
        bytes = checked_sum(bytes, bound_product(nodes, maximum_node));
        // The affected leaf merges reserve their old rows plus distinct keys.
        // At each branch level, each changed key can select one old branch.
        bytes = checked_sum(bytes, bound_product(bound_product(typed_keys, leaf_capacity + 1),
            sizeof(TypedOwnerReference)));
        const auto children = bound_product(bound_product(typed_keys, height), branch_capacity + 1);
        bytes = checked_sum(bytes, bound_product(checked_sum(children, nodes), sizeof(TypedPages::Ptr)));
        if (!changed_keys && state_ && state_->inline_row())
            bytes = checked_sum(bytes, sizeof(Pages::Node) + control_bytes + sizeof(OwnerRecord) + sizeof(Pages::Ptr));
    }
    return bytes;
}
GeometryOwnerManifest GeometryOwnerManifest::edited(
    std::span<const OwnerRecord> changes, Admission admission, PageWork* work, PageWork* lookup_work) const
{
    return edited_typed(changes, {}, admission, work, lookup_work);
}
std::size_t GeometryOwnerManifest::typed_references(const void* address, PageWork* work) const
{
    const TypedOwnerReference key{address};
    const auto* row = TypedPages::lower_bound(state_ ? state_->typed() : nullptr,
        [&](const auto& value) { return TypedLess{}(value, key); }, work);
    return row && row->address == address ? row->references : 0;
}
GeometryOwnerManifest GeometryOwnerManifest::edited_typed(std::span<const OwnerRecord> changes,
    std::span<const TypedOwnerReference> type_changes, Admission admission, PageWork* work, PageWork* lookup_work) const
{
    if (changes.empty() && type_changes.empty())
        return *this;
    for (const auto& row : changes)
    {
        (void) lookup(row.key.channel, row.key.kind, row.owner);
        if (row.key.address != row.owner.get() || row.bytes == std::numeric_limits<std::size_t>::max())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry batch row has an unusable address or size.");
        const auto references = this->references(row.key.channel, row.key.kind, row.owner, lookup_work);
        if ((!references && !row.references) ||
            (references && !admits(row.key.channel, row.key.kind, row.owner, row.bytes, lookup_work)))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry batch row changes its immutable owner.");
        if (row.key.kind == GeometryOwnerGraphEditor::external_kind() &&
            (!type_changes.empty() || (state_ && state_->typed())))
        {
            const TypedOwnerReference wanted{row.key.address};
            const auto found = std::lower_bound(type_changes.begin(), type_changes.end(), wanted, TypedLess{});
            const auto count = found != type_changes.end() && found->address == wanted.address ? found->references :
                typed_references(wanted.address, lookup_work);
            if (count > row.references)
                throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry batch retires a live typed owner.");
        }
    }
    auto typed = TypedPages(admission, work).patched(state_ ? state_->typed() : nullptr, type_changes,
        [](const auto& row) { return row.references != 0; });
    if (changes.empty())
    {
        if (!state_)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Typed references lack a physical geometry owner.");
        admission.charge(1, sizeof(State) + control_bytes);
        GeometryOwnerManifest result;
        result.state_ = state_->inline_row() ? State::created(state_->single(), admission, std::move(typed)) :
                                             State::created(state_->root, admission, std::move(typed));
        return result;
    }
    Pages::Ptr accepted;
    if (state_)
        accepted = state_->inline_row() ? Pages(admission, work).build(std::span(&state_->single(), 1)) : state_->root;
    const auto root =
        Pages(admission, work).patched(accepted, changes, [](const OwnerRecord& row) { return row.references != 0; });
    if (!root)
    {
        if (typed)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry manifest still has typed owners.");
        return {};
    }
    admission.charge(1, sizeof(State) + control_bytes);
    GeometryOwnerManifest result;
    if (root->size == 1 && !typed)
        result.state_ = State::created(Pages::at(root, 0), admission, std::move(typed));
    else
        result.state_ = State::created(root, admission, std::move(typed));
    return result;
}
GeometryOwnerManifest GeometryOwnerManifest::retained(std::uint8_t channel, std::uint8_t kind,
    std::shared_ptr<const void> owner, std::size_t bytes, Admission admission, PageWork* work) const
{
    return retained(channel, kind, std::move(owner), bytes, 1, admission, work);
}
GeometryOwnerManifest GeometryOwnerManifest::retained(std::uint8_t channel, std::uint8_t kind,
    std::shared_ptr<const void> owner, std::size_t bytes, std::size_t references, Admission admission,
    PageWork* work) const
{
    if (!references)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry owner admission lacks a reference.");
    auto row = lookup(channel, kind, owner);
    if (state_ && state_->inline_row())
    {
        const auto& accepted = state_->single();
        const bool same = !OwnerLess{}(accepted, row) && !OwnerLess{}(row, accepted);
        GeometryOwnerManifest result;
        if (same)
        {
            if (!same_owner(accepted.owner, row.owner))
                throw GeometryError(
                    GeometryErrorCode::InvalidInput, "Geometry allocation address has conflicting controls.");
            if (accepted.bytes != bytes || accepted.references > std::numeric_limits<std::size_t>::max() - references)
                throw GeometryError(
                    GeometryErrorCode::InvalidInput, "Geometry owner admission changes its immutable size.");
            row = accepted;
            row.bytes = bytes;
            row.references += references;
            admission.charge(1, sizeof(State) + control_bytes);
            result.state_ = State::created(std::move(row), admission, state_->typed());
            return result;
        }
        // A second distinct owner needs the indexed representation.
        row.bytes = bytes;
        row.references = references;
        std::array<OwnerRecord, 2> rows{accepted, row};
        if (OwnerLess{}(rows[1], rows[0]))
            std::swap(rows[0], rows[1]);
        admission.charge(1, sizeof(State) + control_bytes);
        result.state_ = State::created(Pages(admission, work).build(rows), admission, state_->typed());
        return result;
    }
    const auto* accepted = state_ ? find(state_->root, row, work) : nullptr;
    if (accepted)
    {
        if (accepted->bytes != bytes || accepted->references > std::numeric_limits<std::size_t>::max() - references)
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry owner admission changes its immutable size.");
        row = *accepted;
    }
    row.bytes = bytes;
    row.references += references;
    auto root = Pages(admission, work).edit(state_ ? state_->root : nullptr, row, &row, accepted != nullptr);
    admission.charge(1, sizeof(State) + control_bytes);
    GeometryOwnerManifest result;
    result.state_ = State::created(std::move(root), admission, state_ ? state_->typed() : nullptr);
    return result;
}
GeometryOwnerManifest GeometryOwnerManifest::retired(std::uint8_t channel, std::uint8_t kind,
    const std::shared_ptr<const void>& owner, Admission admission, PageWork* work) const
{
    return retired(channel, kind, owner, 1, admission, work);
}
GeometryOwnerManifest GeometryOwnerManifest::retired(std::uint8_t channel, std::uint8_t kind,
    const std::shared_ptr<const void>& owner, std::size_t references, Admission admission, PageWork* work) const
{
    if (!references)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner lacks a reference.");
    auto row = lookup(channel, kind, owner);
    if (state_ && state_->inline_row())
    {
        const auto& accepted = state_->single();
        if (OwnerLess{}(accepted, row) || OwnerLess{}(row, accepted))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner is absent.");
        if (!same_owner(accepted.owner, row.owner))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner has foreign controls.");
        if (accepted.references < references)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner is absent.");
        row = accepted;
        row.references -= references;
        if (kind == GeometryOwnerGraphEditor::external_kind() && row.references < typed_references(owner.get(), work))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Flat retirement consumes a typed geometry reference.");
        GeometryOwnerManifest result;
        if (row.references)
        {
            admission.charge(1, sizeof(State) + control_bytes);
            result.state_ = State::created(std::move(row), admission, state_->typed());
        }
        return result;
    }
    const auto* accepted = state_ ? find(state_->root, row, work) : nullptr;
    if (!accepted || accepted->references < references)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry owner is absent.");
    row = *accepted;
    row.references -= references;
    if (kind == GeometryOwnerGraphEditor::external_kind() && row.references < typed_references(owner.get(), work))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Flat retirement consumes a typed geometry reference.");
    auto root = Pages(admission, work).edit(state_->root, row, row.references ? &row : nullptr, true);
    GeometryOwnerManifest result;
    if (root)
    {
        admission.charge(1, sizeof(State) + control_bytes);
        result.state_ = State::created(std::move(root), admission, state_->typed());
    }
    return result;
}
bool GeometryOwnerManifest::admits(std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>& owner,
    std::size_t bytes, PageWork* work) const
{
    const auto wanted = lookup(channel, kind, owner);
    if (state_ && state_->inline_row())
    {
        if (work)
            ++work->visited_pages;
        GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ManifestLookupPage);
        if (OwnerLess{}(state_->single(), wanted) || OwnerLess{}(wanted, state_->single()))
            return false;
        if (!same_owner(state_->single().owner, wanted.owner))
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry allocation address has conflicting controls.");
        return state_->single().bytes == bytes;
    }
    const auto* row = state_ ? find(state_->root, wanted, work) : nullptr;
    return row && row->bytes == bytes;
}
std::size_t GeometryOwnerManifest::references(
    std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>& owner, PageWork* work) const
{
    const auto wanted = lookup(channel, kind, owner);
    if (state_ && state_->inline_row())
    {
        if (work)
            ++work->visited_pages;
        GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ManifestLookupPage);
        if (OwnerLess{}(state_->single(), wanted) || OwnerLess{}(wanted, state_->single()))
            return 0;
        if (!same_owner(state_->single().owner, wanted.owner))
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Geometry allocation address has conflicting controls.");
        return state_->single().references;
    }
    const auto* row = state_ ? find(state_->root, wanted, work) : nullptr;
    return row ? row->references : 0;
}
std::optional<std::uint8_t> GeometryOwnerManifest::uniform_channel(PageWork* work) const
{
    if (!state_)
        return std::nullopt;
    if (state_->inline_row())
    {
        if (work)
            ++work->visited_pages;
        GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ManifestLookupPage);
        return state_->single().key.channel;
    }
    PageWork local;
    auto& measured = work ? *work : local;
    const auto before = measured.visited_pages;
    const auto first = Pages::lower_bound(state_->root, [](const auto&) { return false; }, &measured);
    GeometryAllocationOwnerAccess::observe_work(
        GeometryAllocationOwnerAccess::Work::ManifestLookupPage, measured.visited_pages - before);
    if (first->key.channel != state_->root->maximum->key.channel)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry owner manifest mixes arithmetic channels.");
    return first->key.channel;
}
std::size_t GeometryOwnerManifest::payload_bytes() const noexcept
{
    if (!state_)
        return 0;
    return state_->inline_row() ? state_->single().bytes : state_->root->subtree_owner_bytes;
}
std::size_t GeometryOwnerManifest::metadata_admission_bytes() const noexcept
{
    if (!state_)
        return 0;
    const auto typed = state_->typed();
    return root_metadata_bytes() + (metadata_state(state_) ? 0 : control_bytes) +
           (state_->inline_row() ? 0 : state_->root->subtree_bytes) + (typed ? typed->subtree_bytes : 0);
}
std::size_t GeometryOwnerManifest::metadata_payload_bytes() const noexcept
{
    if (!state_)
        return 0;
    const auto typed = state_->typed();
    return root_metadata_bytes() + (state_->inline_row() ? 0 : state_->root->subtree_payload_bytes) +
           (typed ? typed->subtree_payload_bytes : 0);
}
std::size_t GeometryOwnerManifest::owner_count() const noexcept
{
    if (!state_)
        return 0;
    return state_->inline_row() ? 1 : state_->root->size;
}
std::size_t GeometryOwnerManifest::typed_owner_count() const noexcept
{
    const auto typed = state_ ? state_->typed() : nullptr;
    return typed ? typed->size : 0;
}
std::size_t GeometryOwnerManifest::physical_edge_count() const noexcept
{
    if (!state_)
        return 0;
    return state_->inline_row() ? state_->single().references : state_->root->logical_size;
}
void GeometryOwnerManifest::visit_records(RecordSink records) const
{
    if (!state_)
        return;
    const auto row = [&](const OwnerRecord& value)
    {
        GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ManifestVisitRow);
        records(value);
        return true;
    };
    if (state_->inline_row())
        row(state_->single());
    else
        Pages::visit_from(state_->root, [](const auto&) { return false; }, row);
}
void GeometryOwnerManifest::visit(const std::function<void(const OwnerRecord&)>& visitor) const
{
    if (state_ && state_->inline_row())
    {
        GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ManifestVisitRow);
        visitor(state_->single());
        return;
    }
    if (state_)
        Pages::visit_from(
            state_->root, [](const auto&) { return false; },
            [&](const auto& row)
            {
                GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ManifestVisitRow);
                visitor(row);
                return true;
            });
}
void GeometryOwnerManifest::visit_payload(const std::function<void(const void*, std::size_t)>& visitor) const
{
    if (!state_)
        return;
    visitor(state_.get(), sizeof(State));
    const auto expose = [&](const auto& page)
    {
        visitor(page.get(), sizeof(*page));
        if (page->rows.capacity())
            visitor(page->rows.data(), page->rows.capacity() * sizeof(*page->rows.data()));
        if (page->children.capacity())
            visitor(page->children.data(), page->children.capacity() * sizeof(*page->children.data()));
    };
    Pages::visit_pages(state_->root, expose);
    TypedPages::visit_pages(state_->typed(), expose);
}
void GeometryOwnerManifest::visit_metadata(const std::function<void(const void*, std::size_t)>& visitor) const
{
    if (!state_)
        return;
    visitor(state_.get(), sizeof(State));
    const auto expose = [&](const auto& page)
    {
        visitor(page.get(), sizeof(*page) + page->rows.capacity() * sizeof(*page->rows.data()) +
                                page->children.capacity() * sizeof(*page->children.data()));
    };
    Pages::visit_pages(state_->root, expose);
    TypedPages::visit_pages(state_->typed(), expose);
}
} // namespace SimpleFluid::PLIC::details::geometry_demand
