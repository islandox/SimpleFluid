#pragma once
#include "geometry/PLIC/details/GeometryDemandPages.tcc"
#include "geometry/PLIC/details/GeometryIndexControls.hh"
namespace SimpleFluid::PLIC::details
{
namespace index_control_detail
{
struct Less
{
    bool operator()(const IndexControlRecord& first, const IndexControlRecord& second) const noexcept
    {
        return std::less<const void*>{}(first.owner.get(), second.owner.get());
    }
};
using Pages = geometry_demand::OrderedPages<IndexControlRecord, Less>;
bool same(const std::shared_ptr<const void>& first, const std::shared_ptr<const void>& second)
{
    return first.get() == second.get() && !first.owner_before(second) && !second.owner_before(first);
}
struct EditEnvelope
{
    std::size_t height, owners;
};
EditEnvelope edit_envelope(const Pages::Ptr& root, std::size_t changes, std::size_t anticipated_additions)
{
    using namespace allocation_detail;
    std::size_t height = 1;
    for (auto node = root.get(); node; node = node->children.front().get())
    {
        GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ReservationPathPage);
        if (node->leaf())
            break;
        height = add(height, 1);
    }
    const auto rows = add(add(root ? root->size : 0, anticipated_additions), changes);
    auto level = std::max(std::size_t{1}, rows / (geometry_demand::leaf_capacity / 2));
    std::size_t future_height = 1;
    while (level > 1)
    {
        level = std::max(std::size_t{1}, level / (geometry_demand::branch_capacity / 2));
        future_height = add(future_height, 1);
    }
    height = std::max(height, future_height);
    // Per key and level: at most two rewritten/split pages and two
    // redistributed neighbor pages. One extra path covers root growth;
    // the final one is the new immutable State.
    const auto owners = add(product(product(4, add(changes, 1)), add(height, 1)), 1);
    return {height, owners};
}
} // namespace index_control_detail
struct GeometryIndexControls::State
{
    index_control_detail::Pages::Ptr root;
    explicit State(index_control_detail::Pages::Ptr value) : root(std::move(value)) {}
};
std::size_t GeometryIndexControls::control_bytes() const noexcept
{
    return state_ && state_->root ? state_->root->owned_admission_bytes() : 0;
}
std::size_t GeometryIndexControls::index_admission_bytes() const noexcept
{
    return state_ ? sizeof(State) + geometry_demand::metadata_control_bytes(state_) + state_->root->subtree_bytes : 0;
}
std::shared_ptr<const void> GeometryIndexControls::root_control() const noexcept
{
    return state_;
}
std::size_t GeometryIndexControls::edit_metadata_owner_bound(
    std::size_t changes, std::size_t anticipated_additions) const
{
    if (!changes)
        return 0;
    return index_control_detail::edit_envelope(state_ ? state_->root : nullptr, changes, anticipated_additions).owners;
}
std::size_t GeometryIndexControls::edit_admission_bound(std::size_t changes, std::size_t anticipated_additions) const
{
    if (!changes)
        return 0;
    using namespace allocation_detail;
    using namespace geometry_demand;
    using Pages = index_control_detail::Pages;
    const auto bound =
        index_control_detail::edit_envelope(state_ ? state_->root : nullptr, changes, anticipated_additions);
    const auto pages = bound.owners - 1;
    const auto page_bytes = add(sizeof(Pages::Node) + geometry_demand::control_bytes,
        std::max(product(leaf_capacity, sizeof(IndexControlRecord)), product(branch_capacity, sizeof(Pages::Ptr))));
    auto bytes = add(sizeof(State) + geometry_demand::control_bytes, product(pages, page_bytes));
    // Sum old-leaf capacity plus that leaf's assigned changes. There are
    // at most changes affected leaves; inline merges need no heap reserve.
    bytes = add(bytes, product(product(changes, leaf_capacity + 1), sizeof(IndexControlRecord)));
    // At each level, every changed key can select at most one old branch;
    // assigned change counts sum to changes. Include each result Ptr buffer
    // (one slot per constructed page) as well as root packing buffers.
    bytes = add(bytes, product(product(product(changes, bound.height), branch_capacity + 1), sizeof(Pages::Ptr)));
    return add(bytes, product(pages, sizeof(Pages::Ptr)));
}
bool GeometryIndexControls::holds(const std::shared_ptr<const void>& owner, std::size_t payload) const
{
    using namespace index_control_detail;
    if (!state_)
        return false;
    const IndexControlRecord key{owner, payload, 0};
    geometry_demand::PageWork work;
    const auto* row = Pages::lower_bound(state_->root, [&](const auto& value) { return Less{}(value, key); }, &work);
    GeometryAllocationOwnerAccess::observe_work(
        GeometryAllocationOwnerAccess::Work::IndexControlLookupPage, work.visited_pages);
    if (!row || Less{}(key, *row))
        return false;
    if (!same(row->owner, owner) || row->payload != payload)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Index admin role changes owner or payload.");
    return row->native || row->control;
}
bool GeometryIndexControls::contains(
    const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t control) const
{
    using namespace index_control_detail;
    if (!state_)
        return false;
    const IndexControlRecord key{owner, payload, control};
    geometry_demand::PageWork lookup;
    const auto* found = Pages::lower_bound(state_->root, [&](const auto& row) { return Less{}(row, key); }, &lookup);
    GeometryAllocationOwnerAccess::observe_work(
        GeometryAllocationOwnerAccess::Work::IndexControlLookupPage, lookup.visited_pages);
    if (!found || Less{}(key, *found))
        return false;
    if (!same(found->owner, owner) || found->payload != payload || found->control != control)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Immutable index-control credit changes owner or size.");
    return true;
}
GeometryIndexControls GeometryIndexControls::edited(std::span<const IndexControlRecord> changes,
    geometry_demand::Admission admission, geometry_demand::PageWork* work) const
{
    using namespace index_control_detail;
    if (changes.empty())
        return *this;
    for (const auto& row : changes)
    {
        const std::shared_ptr<const void> empty;
        if (!row.owner || (!row.owner.owner_before(empty) && !empty.owner_before(row.owner)))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Index control needs an actual immutable owner.");
        if (state_)
        {
            geometry_demand::PageWork lookup;
            const auto* old =
                Pages::lower_bound(state_->root, [&](const auto& old) { return Less{}(old, row); }, &lookup);
            GeometryAllocationOwnerAccess::observe_work(
                GeometryAllocationOwnerAccess::Work::IndexControlLookupPage, lookup.visited_pages);
            if (old && !Less{}(row, *old) && (!same(old->owner, row.owner) || old->payload != row.payload))
                throw GeometryError(GeometryErrorCode::InvalidInput, "Index control changes immutable storage.");
        }
    }
    const auto root = Pages(admission, work)
                          .patched(state_ ? state_->root : nullptr, changes,
                              [](const auto& row) { return row.native || row.control != 0; });
    if (!root)
        return {};
    admission.charge(1, sizeof(State) + geometry_demand::control_bytes);
    auto next = admission.shared<State>(root);

    const auto bytes = sizeof(State) + (admission.native ? geometry_demand::metadata_control_bytes(next) : 0);
    admission.created(next, bytes, bytes + (admission.native ? 0 : geometry_demand::control_bytes));
    GeometryIndexControls result;
    result.state_ = std::move(next);
    return result;
}
void GeometryIndexControls::visit_metadata(geometry_demand::MetadataSink metadata) const
{
    if (!state_)
        return;
    const auto bytes =
        sizeof(State) + (geometry_demand::metadata_state(state_) ? geometry_demand::metadata_control_bytes(state_) : 0);
    if (metadata(state_, bytes, bytes + (geometry_demand::metadata_state(state_) ? 0 : geometry_demand::control_bytes)))
    {
        if (geometry_demand::metadata_state(state_))
            for (unsigned edge = 0; edge != 2; ++edge)
                geometry_demand::visit_native_state(geometry_demand::metadata_state(state_), metadata);
        index_control_detail::Pages::visit_dependencies(state_->root, metadata, [](const auto&) {});
    }
}
bool GeometryIndexControls::has_native_administration() const noexcept
{
    return state_ &&
           (geometry_demand::metadata_state(state_) || (state_->root && state_->root->has_native_administration()));
}
void GeometryIndexControls::visit_native_administration_delta(const GeometryIndexControls& previous,
    geometry_demand::NativeAdministrationSink sink, geometry_demand::PageWork* work) const
{
    if (state_.get() == previous.state_.get() && !state_.owner_before(previous.state_) &&
        !previous.state_.owner_before(state_))
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
    index_control_detail::Pages::visit_native_administration_delta(
        previous.state_ ? previous.state_->root : nullptr, state_ ? state_->root : nullptr, sink, work);
}
void GeometryIndexControls::visit_records(RecordSink sink) const
{
    if (state_)
        index_control_detail::Pages::visit_from(
            state_->root, [](const auto&) { return false; },
            [&](const auto& row)
            {
                GeometryAllocationOwnerAccess::observe_work(
                    GeometryAllocationOwnerAccess::Work::ControlTableEnumerationRow);
                sink.visit(sink.context, row);
                return true;
            });
}
void GeometryIndexControls::visit(const std::function<void(const IndexControlRecord&)>& visitor) const
{
    if (state_)
        index_control_detail::Pages::visit_from(
            state_->root, [](const auto&) { return false; },
            [&](const auto& row)
            {
                GeometryAllocationOwnerAccess::observe_work(
                    GeometryAllocationOwnerAccess::Work::ControlTableEnumerationRow);
                visitor(row);
                return true;
            });
}
void GeometryIndexControls::visit_borrowed_dependencies(
    geometry_demand::BorrowedMetadataSink metadata, BorrowedRecordSink records) const
{
    if (state_ && metadata(state_.get(), nullptr, sizeof(State)))
        index_control_detail::Pages::visit_borrowed_dependencies(
            state_->root, metadata,
            [&](const auto& row, const void* parent)
            {
                GeometryAllocationOwnerAccess::observe_work(
                    GeometryAllocationOwnerAccess::Work::ControlTableEnumerationRow);
                records.visit(records.context, parent, row);
            },
            state_.get());
}
void GeometryIndexControls::visit_payload(const std::function<void(const void*, std::size_t)>& visitor) const
{
    if (!state_)
        return;
    visitor(state_.get(), sizeof(State));
    index_control_detail::Pages::visit_pages(state_->root,
        [&](const auto& node)
        {
            visitor(node.get(), sizeof(*node));
            if (node->rows.capacity())
                visitor(node->rows.data(), node->rows.capacity() * sizeof(node->rows[0]));
            if (node->children.capacity())
                visitor(node->children.data(), node->children.capacity() * sizeof(node->children[0]));
        });
}
} // namespace SimpleFluid::PLIC::details
