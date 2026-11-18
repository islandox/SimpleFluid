/** @file GeometryOwnerGraph.tcc
 * @brief Compiled persistent first/last immutable physical geometry graph edits.
 */
#pragma once
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/GeometryRetentionCensusAccess.hh"
#include "geometry/PLIC/details/NativeIdentityOwnerAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/details/NativeAllocationGraph.hh"
#include <algorithm>
#include <exception>
#include <iterator>
#include <string>
#include <tuple>

namespace SimpleFluid::PLIC::details
{
bool geometry_demand::MetadataSink::project_native_state(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& state) const
{
    if (!financial_graph)
        return false;
    financial_graph->native_state(state, financial_retain);
    return true;
}
GeometryOwnerGraphEditor::Impl::Call::Call(Impl& value) : editor(value)
{
    if (editor.failed || editor.finished)
        allocation_detail::exhausted("Geometry owner graph editor cannot resume rejected or completed edits.");
    if (editor.used < editor.last_counter || editor.used > editor.limit)
    {
        editor.failed = true;
        if (editor.ledger)
            GeometryAllocationOwnerAccess::poison(*editor.ledger);
        allocation_detail::exhausted("Geometry owner graph cumulative counter is invalid.");
    }
}
GeometryOwnerGraphEditor::Impl::Call::~Call()
{
    editor.last_counter = editor.used;
    if (std::uncaught_exceptions() > exceptions)
    {
        editor.failed = true;
        if (editor.ledger)
            GeometryAllocationOwnerAccess::poison(*editor.ledger);
    }
}
geometry_demand::Admission GeometryOwnerGraphEditor::Impl::admission()
{
    return {this,
        [](void* raw, std::size_t bytes)
        {
            auto& self = *static_cast<Impl*>(raw);
            if (self.used > self.limit || bytes > self.limit - self.used)
                throw GeometryError(GeometryErrorCode::ResourceLimit,
                    "Geometry owner graph index budget is exhausted: used=" + std::to_string(self.used) +
                        " request=" + std::to_string(bytes) + " limit=" + std::to_string(self.limit) + ".");
            if (self.native_administration)
                self.native_administration->record_ledger_work(bytes);
            self.used += bytes;
        },
        [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
        {
            auto& self = *static_cast<Impl*>(raw);
            if (self.ledger)
                GeometryAllocationOwnerAccess::retain_created_metadata(
                    *self.ledger, self.channel, owner, payload, admitted, self.used, self.limit);
        },
        native_administration,
        [](void* raw, const std::shared_ptr<const void>& owner, std::size_t bytes)
        {
            auto& self = *static_cast<Impl*>(raw);
            if (self.ledger)
                GeometryAllocationOwnerAccess::register_native_metadata(*self.ledger, self.channel, owner, bytes);
        }};
}
void GeometryOwnerGraphEditor::Impl::edit_paid(const NativePaidExtentCharge& charge, NativeOwnerReceipts::Action action)
{
    using Action = NativeOwnerReceipts::Action;
    const bool adding = action == Action::Record || action == Action::Retain;
    if (adding && !charge.valid())
    {
        if (charge.has_financial_pin())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Native financial receipt does not authenticate funding.");
        return;
    }
    if (action == Action::Record && !paid_reference_edits && paid.contains(charge))
        return;
    // Record is idempotent between reference edits. Fixed receipts have
    // immutable eligibility and often repeat for a shared State or keeper.
    // Borrow the staged charge, which stays in its block through sealing.
    const bool fixed = action == Action::Record && charge.proof_ == 1 &&
                       charge.fixed_owner_ != NativePaidExtentCharge::FixedOwner::None;
    if (fixed)
    {
        const auto previous = std::find_if(recent_fixed_receipts.begin(), recent_fixed_receipts.end(),
            [&](const NativeReceiptFact* value)
            {
                return value && value->fixed_owner_ == charge.fixed_owner_ &&
                       value->publication_h() == charge.publication_h() &&
                       value->state_address_ == charge.state_address_ &&
                       value->charge_snapshot_ == charge.charge_snapshot_ &&
                       value->bytes() == charge.bytes() && value->same_owner_fact(charge) &&
                       value->same_origin_fact(charge);
            });
        if (previous != recent_fixed_receipts.end())
        {
            std::rotate(recent_fixed_receipts.begin(), previous, std::next(previous));
            return;
        }
    }
    else if (action != Action::Record)
        recent_fixed_receipts.fill(nullptr);
    if (!paid_edits)
        paid_edits.emplace(admission());
    paid_edits->append(charge, action);
    if (fixed)
    {
        std::move_backward(recent_fixed_receipts.begin(), std::prev(recent_fixed_receipts.end()),
            recent_fixed_receipts.end());
        recent_fixed_receipts.front() = &paid_edits->rows().back().charge.fact();
    }
    paid_reference_edits |= action != Action::Record;
}
template<class Kind>
std::size_t GeometryOwnerGraphEditor::Impl::references(
    Kind kind, const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    const auto key = static_cast<std::uint8_t>(kind);
    if (const auto* found = edits.find({channel, key, owner.get()}))
    {
        if (found->row.bytes != bytes || owner.owner_before(found->row.owner) || found->row.owner.owner_before(owner))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Paired graph owner changes control or size.");
        return found->row.references;
    }
    const auto count = result.references(channel, key, owner, work ? &work->lookup : nullptr);
    if (count && !result.admits(channel, key, owner, bytes, work ? &work->lookup : nullptr))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Paired graph owner changes immutable size.");
    return count;
}
geometry_demand::OwnerEdits::Edit& GeometryOwnerGraphEditor::Impl::stage_owner(
    std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes)
{
    const geometry_demand::OwnerKey key{channel, kind, owner.get()};
    if (auto* found = edits.find(key))
    {
        if (found->row.bytes != bytes || owner.owner_before(found->row.owner) || found->row.owner.owner_before(owner))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry graph owner changes controls or bytes.");
        return *found;
    }
    auto* lookup = work ? &work->lookup : nullptr;
    const auto references = result.references(channel, kind, owner, lookup);
    if (bytes == std::numeric_limits<std::size_t>::max() ||
        (references && !result.admits(channel, kind, owner, bytes, lookup)))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry graph owner has an invalid or changed extent.");
    const auto typed = references ? result.typed_references(key, lookup) : 0;
    auto& staged = edits.insert({{key, owner, bytes, references}, references, typed, typed}, admission());
    if (work)
        ++work->dependency_entries;
    return staged;
}
template<class Kind>
bool GeometryOwnerGraphEditor::Impl::edge(Kind kind, const std::shared_ptr<const void>& owner, std::size_t bytes,
    bool retain, bool prepaid, std::size_t references)
{
    return owner && references &&
           apply_edge(stage_owner(static_cast<std::uint8_t>(kind), owner, bytes), retain, prepaid, references);
}
bool GeometryOwnerGraphEditor::Impl::apply_edge(
    geometry_demand::OwnerEdits::Edit& edit, bool retain, bool prepaid, std::size_t references)
{
    if (work)
        work->edge_visits = allocation_detail::add(work->edge_visits, references);
    auto& record = edit.row;
    const auto& owner = record.owner;
    const auto bytes = record.bytes;
    const auto kind = record.key.kind;
    bool transition = false;
    if (retain)
    {
        transition = record.references == 0;
        record.references = allocation_detail::add(record.references, references);
        if (transition)
        {
            if (work)
                ++work->first_transitions;
            if (ledger)
            {
                GeometryAllocationOwnerAccess::admit_graph_owner(
                    *ledger, channel, kind, owner, bytes, used, limit, prepaid);
                edit_paid(GeometryAllocationOwnerAccess::paid_charge(*ledger, channel, kind, owner, bytes));
            }
        }
    }
    else
    {
        if (references > record.references)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry graph owner is absent.");
        record.references -= references;
        if (record.references < edit.typed)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Flat graph retirement consumes a typed owner.");
        transition = record.references == 0;
        if (transition && work)
            ++work->last_transitions;
    }
    if (transition && partner && partner->references(kind, owner, bytes))
    {
        if (retain)
            *cross_payload_bytes = allocation_detail::add(*cross_payload_bytes, bytes);
        else
        {
            if (bytes > *cross_payload_bytes)
                throw GeometryError(GeometryErrorCode::InvalidInput, "Paired graph overlap underflows.");
            *cross_payload_bytes -= bytes;
        }
    }
    return transition;
}
GeometryOwnerGraphEditor::Impl::TypedTransition GeometryOwnerGraphEditor::Impl::typed_edge(
    allocation_detail::Kind kind, const std::shared_ptr<const void>& owner, std::size_t bytes, bool retain,
    bool prepaid, std::size_t references)
{
    auto& edit = stage_owner(static_cast<std::uint8_t>(kind), owner, bytes);
    const auto previous = edit.typed;
    if (!retain && references > previous)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Retired typed geometry edge is absent.");
    edit.typed = retain ? allocation_detail::add(previous, references) : previous - references;
    return {apply_edge(edit, retain, prepaid, references), retain ? previous == 0 : edit.typed == 0};
}
bool GeometryOwnerGraphEditor::Impl::direct_edge(
    allocation_detail::Kind kind, const std::shared_ptr<const void>& owner, std::size_t bytes, bool retain, bool prepaid)
{
    if (!owner)
        return false;
    if (separate_direct_edges)
        return typed_edge(kind, owner, bytes, retain, prepaid).children;
    return edge(kind, owner, bytes, retain, prepaid);
}
template<class Kind, class T>
void GeometryOwnerGraphEditor::Impl::push(Kind kind, const std::shared_ptr<const T>& owner, std::size_t&, std::size_t)
{
    if (!owner)
        return;
    if (pending_count < inline_pending.size())
    {
        inline_pending[pending_count++] = {static_cast<std::uint8_t>(kind), owner};
        return;
    }
    if (pending.size() == pending.capacity())
    {
        const auto next = pending.capacity() ? allocation_detail::product(pending.capacity(), 2) : 1;
        admission().charge(next, sizeof(PendingOwner));
        pending.reserve(next);
        if (pending.capacity() != next)
            allocation_detail::exhausted("Geometry owner graph scratch capacity is unqualified.");
    }
    pending.push_back({static_cast<std::uint8_t>(kind), owner});
}
bool GeometryOwnerGraphEditor::Impl::native_state(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& owner, bool retain, bool prepaid,
    std::size_t references)
{
    if (!references)
        return false;
    const auto counter_bytes = geometry_detail::NativeAllocationGraphAccess::validate_state(owner);
    if (retain)
    {
        edit_paid(NativePaidExtentCharge::state(owner));
        edit_paid(NativePaidExtentCharge::counter(owner));
        edit_paid(NativePaidExtentCharge::mirror(owner));
        if (ledger)
            GeometryAllocationOwnerAccess::authorize_native_state(*ledger, owner);
    }
    const bool paid = prepaid || owner->used_counter == &used;
    const auto kind = allocation_detail::Kind::ExternalOwner;
    const auto transition = typed_edge(kind, owner, owner->allocation_admission_bytes, retain, paid, references);
    if (transition.children)
    {
        const auto keeper_bytes = owner->mirror.owner_proof.bytes();
        if (owner->mirror.owner)
            edge(allocation_detail::Kind::ExternalOwner, owner->mirror.owner, keeper_bytes, retain, paid, 2);
        if (typed_edge(kind, owner->counter_owner, counter_bytes, retain, paid, 2).children && owner->mirror.owner)
            edge(allocation_detail::Kind::ExternalOwner, owner->mirror.owner, keeper_bytes, retain, paid, 2);
    }
    return transition.owner;
}
void GeometryOwnerGraphEditor::Impl::native_typed_receipt(allocation_detail::Kind kind,
    const std::shared_ptr<const void>& owner,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope,
    const geometry_detail::NativeConstructorStorageOwner& storage, std::size_t&, std::size_t)
{
    if (!adding) return;
    if (storage.owner)
        edit_paid(NativePaidExtentCharge::storage(storage));
    if (ledger)
    {
        GeometryAllocationOwnerAccess::authorize_native_semantic(*ledger, channel, std::uint8_t(kind), owner, envelope);
        if (storage.owner)
            GeometryAllocationOwnerAccess::authorize_native_storage(*ledger, storage.owner, storage.envelope);
    }
}
void GeometryOwnerGraphEditor::Impl::native_object_children(
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope,
    const geometry_detail::NativeConstructorStorageOwner& storage, std::size_t&, std::size_t)
{
    native_fixed(envelope, adding, false);
    if (!storage.owner) return;
    native_fixed(storage.envelope, adding, false);
    const bool prepaid = storage.owner->admission().state()->used_counter == &used;
    if (direct_edge(
            allocation_detail::Kind::ExternalOwner, storage.owner, storage.owner->exclusive_bytes(), adding, prepaid))
        native_state(storage.owner->admission().state(), adding, false);
}
bool GeometryOwnerGraphEditor::Impl::native_fixed(
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& owner, bool retain, bool prepaid)
{
    if (retain)
        edit_paid(NativePaidExtentCharge::fixed(owner));
    if (!owner)
        return false;
    if (!owner->authentic_owner(owner))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Native fixed envelope has a foreign shared control.");
    if (ledger && retain)
        GeometryAllocationOwnerAccess::authorize_native_fixed(*ledger, owner);
    const auto& state = owner->state();
    (void) geometry_detail::NativeAllocationGraphAccess::validate_state(state);
    const bool paid = prepaid || state->used_counter == &used;
    const auto transition = typed_edge(allocation_detail::Kind::ExternalOwner, owner, owner->fixed_allocation_bytes(),
        retain, paid);
    if (transition.children)
        native_state(state, retain, paid, 5);
    return transition.owner;
}
bool GeometryOwnerGraphEditor::Impl::weak_observation(const NativeWeakOwnerObservation& value, bool retain)
{
    if (const auto* native = value.native_envelope())
        return native_fixed(*native, retain, false);
    if (const auto* state = value.native_state())
        return native_state(*state, retain, false);
    const auto kind = allocation_detail::Kind::ExternalOwner;
    if (const auto* scalar = value.detached_scalar())
    {
        if (!scalar->keeper)
            return edge(kind, scalar->owner, scalar->bytes, retain);
        const auto transition = typed_edge(kind, scalar->owner, scalar->bytes, retain, false);
        // The witness owns its own keeper. The Counter control owns two
        // additional keeper edges through its installed allocator and deleter.
        edge(kind, scalar->keeper, scalar->keeper_bytes, retain, false, transition.children ? 3 : 1);
        return transition.owner;
    }
    if (const auto* contact = value.contact_control_group())
    {
        if (retain)
            edit_paid(NativePaidExtentCharge::contact_fixed(value));
        const auto transition = typed_edge(kind, contact->owner, contact->bytes, retain, false);
        // The witness and the actual installed group each hold a keeper.
        edge(kind, contact->keeper, contact->keeper_bytes, retain, false, transition.children ? 2 : 1);
        return transition.owner;
    }
    return false;
}
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
void GeometryOwnerGraphEditor::Impl::retain_identity(
    const GeometryIdentity<Kind, Mode>& identity, std::size_t&, std::size_t)
{
    if constexpr (GeometryIdentity<Kind, Mode>::shared_storage)
        for (auto owner = GeometryIdentityAccess::backing(identity); owner; owner = geometry_identity_parent(owner))
        {
            const auto* admitted = NativeIdentityOwnerAccess::backing(owner);
            const auto bytes = admitted ? 0 : geometry_identity_payload_bytes(owner);
            if (admitted && adding)
                edit_paid(NativePaidExtentCharge::storage(admitted->storage()));
            if (admitted && adding && ledger)
            {
                GeometryAllocationOwnerAccess::authorize_native_identity(*ledger, channel, owner);
                GeometryAllocationOwnerAccess::authorize_native_storage(
                    *ledger, admitted->storage().owner, admitted->storage().envelope);
            }
            if (!direct_edge(allocation_detail::Kind::IdentityPayload, owner, bytes, adding))
                break;
            if (admitted)
            {
                geometry_detail::NativeConstructorOwnerSink sink;
                sink.context = this;
                sink.owner_edge = [](void* raw, const std::shared_ptr<const void>& child,
                                      geometry_detail::NativeConstructorOwnerKind kind, std::size_t count)
                {
                    auto& self = *static_cast<Impl*>(raw);
                    bool paid = false;
                    if (kind == geometry_detail::NativeConstructorOwnerKind::Storage)
                    {
                        const auto storage =
                            std::static_pointer_cast<const geometry_detail::NativeConstructorStorage>(child);
                        paid = storage->admission().state()->used_counter == &self.used;
                    }
                    return self.direct_edge(allocation_detail::Kind::ExternalOwner, child, count, self.adding, paid);
                };
                sink.fixed_edge = [](void* raw, const auto& fixed)
                {
                    auto& self = *static_cast<Impl*>(raw);
                    self.native_fixed(fixed, self.adding, false);
                };
                sink.state_edge = [](void* raw, const auto& state)
                {
                    auto& self = *static_cast<Impl*>(raw);
                    self.native_state(state, self.adding, false);
                };
                // The outer loop edits the parent edge exactly once, retaining
                // iterative ancestry and first/last pruning for every mode.
                NativeIdentityOwnerAccess::visit_children(owner, sink, [](void*, const auto&) {});
            }
        }
}
void GeometryOwnerGraphEditor::Impl::retain_authorities(
    const std::vector<PlaneAuthority>& values, std::size_t& count, std::size_t cap)
{
    for (const auto& value : values)
        if (const auto* support = OriginalSupportAuthorityAccess::retained(value))
            retain_identity(*support, count, cap);
        else if (const auto owner = value.opaque_storage_owner())
            (void) edge(allocation_detail::Kind::AuthorityPayload, owner, value.opaque_payload_bytes(), adding);
}
template<ArithmeticMode Mode> void GeometryOwnerGraphEditor::Impl::drain(bool retain)
{
    adding = retain;
    while (pending_count || !pending.empty())
    {
        PendingOwner value;
        if (!pending.empty())
        {
            value = std::move(pending.back());
            pending.pop_back();
        }
        else
        {
            value = std::move(inline_pending[--pending_count]);
            inline_pending[pending_count] = {};
        }
        const allocation_detail::Item item{static_cast<allocation_detail::Kind>(value.kind), std::move(value.owner)};
        allocation_detail::visit_geometry_owner<Mode>(*this, item, used, limit,
            [&](const allocation_detail::Item& value, std::size_t bytes, std::size_t&, std::size_t)
            { return direct_edge(value.kind, value.owner, bytes, adding); });
    }
}

GeometryOwnerGraphEditor::GeometryOwnerGraphEditor(geometry_demand::GeometryOwnerManifest accepted,
    std::uint8_t channel, std::size_t& used, std::size_t limit, GeometryAllocationLedger* ledger,
    GeometryOwnerGraphWork* work, std::shared_ptr<const geometry_detail::NativeAllocationState> native_administration,
    const NativeOwnerReceipts* previous_paid)
    : state_{std::move(accepted), channel, used, limit, ledger, work,
          ledger ? GeometryAllocationOwnerAccess::native_state(*ledger) : std::move(native_administration),
          previous_paid ? *previous_paid : NativeOwnerReceipts{}, pending_}
{
    try
    {
        const auto native = impl_->native_administration;
        impl_->pending = decltype(impl_->pending)(geometry_detail::NativeAdministrationAllocator<PendingOwner>(native));
        if (channel > 1 || used > limit)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid geometry owner graph arithmetic channel.");
        if (const auto existing = impl_->result.uniform_channel(work ? &work->lookup : nullptr);
            existing && *existing != channel)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry owner graph has a foreign channel.");
        impl_->credited_predecessor = !ledger || GeometryOwnerImportAccess::credited(*ledger, impl_->result);
        // All fixed editor state and bounded scratch live on the caller stack.
        impl_->last_counter = used;
    }
    catch (...)
    {
        if (ledger)
            GeometryAllocationOwnerAccess::poison(*ledger);
        throw;
    }
}
void GeometryOwnerGraphEditor::pair_with(GeometryOwnerGraphEditor& other, std::size_t& overlap)
{
    const Impl::Call first(*impl_), second(*other.impl_);
    if (this == &other || impl_->channel == other.impl_->channel || &impl_->used != &other.impl_->used ||
        impl_->limit != other.impl_->limit || impl_->partner || other.impl_->partner ||
        overlap > std::min(impl_->result.payload_bytes(), other.impl_->result.payload_bytes()))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid paired graph overlap context.");
    impl_->partner = other.impl_;
    other.impl_->partner = impl_;
    impl_->cross_payload_bytes = &overlap;
    other.impl_->cross_payload_bytes = &overlap;
}
GeometryOwnerGraphEditor::~GeometryOwnerGraphEditor()
{
    if (impl_->partner)
    {
        impl_->partner->partner = nullptr;
        impl_->partner->cross_payload_bytes = nullptr;
    }
    // A catalog constructor or outer preflight can reject between editor calls.
    // Its unpublished candidate still shares this operation's admission ledger.
    if (impl_ && impl_->ledger && std::uncaught_exceptions() > impl_->construction_exceptions)
        GeometryAllocationOwnerAccess::poison(*impl_->ledger);
}
std::uint8_t GeometryOwnerGraphEditor::external_kind() noexcept
{
    return static_cast<std::uint8_t>(allocation_detail::Kind::ExternalOwner);
}
std::uint8_t GeometryOwnerGraphEditor::identity_payload_kind() noexcept
{
    return static_cast<std::uint8_t>(allocation_detail::Kind::IdentityPayload);
}
std::size_t GeometryOwnerGraphEditor::edit_byte_bound(
    const geometry_demand::GeometryOwnerManifest& accepted, std::size_t edges, std::size_t changed_keys,
    std::size_t typed_keys)
{
    using namespace allocation_detail;
    const auto excess = edges > geometry_demand::leaf_capacity ? edges - geometry_demand::leaf_capacity : 0;
    // Touched rows include cancelling edits that will never enter an index.
    auto bytes = geometry_demand::OwnerEdits::byte_bound(edges);
    // Account each full allocation in geometric vector growth, including the
    // previous buffer that remains live during reserve; no guessed multiplier.
    std::size_t capacity = 1, pending_slots = 0;
    if (excess)
    {
        while (capacity < excess)
        {
            pending_slots = add(pending_slots, capacity);
            capacity = product(capacity, 2);
        }
        pending_slots = add(pending_slots, capacity);
    }
    bytes = add(bytes, product(pending_slots, sizeof(PendingOwner)));
    bytes = add(bytes, accepted.batch_edit_byte_bound(changed_keys, typed_keys));
    // Each newly allocated index node/State produces at most one payload map
    // row and one metadata-credit registry row. Both are charged before insert.
    const auto bookkeeping_per_owner =
        sizeof(typename decltype(GeometryAllocationLedger::Impl::owners)::value_type) + 4 * sizeof(void*) +
        sizeof(typename decltype(GeometryAllocationLedger::Impl::metadata_credits)::value_type) + 4 * sizeof(void*);
    bytes = add(bytes,
        product(accepted.batch_edit_metadata_owner_bound(changed_keys, typed_keys), bookkeeping_per_owner));
    // A second distinct graph certificate materializes the ledger's bounded
    // overflow buffer. Reserve it before optional graph mutation can begin.
    return add(bytes, GeometryAllocationLedger::Impl::subset_overflow_bytes());
}
bool GeometryOwnerGraphEditor::external(
    const std::shared_ptr<const void>& owner, std::size_t bytes, bool retain, bool prepaid)
{
    const Impl::Call call(*impl_);
    return impl_->edge(allocation_detail::Kind::ExternalOwner, owner, bytes, retain, prepaid);
}
bool GeometryOwnerGraphEditor::external_with_children(
    const std::shared_ptr<const void>& owner, std::size_t bytes, bool retain, bool prepaid)
{
    const Impl::Call call(*impl_);
    return owner && impl_->typed_edge(allocation_detail::Kind::ExternalOwner, owner, bytes, retain, prepaid).children;
}
bool GeometryOwnerGraphEditor::owner(const geometry_demand::OwnerRecord& row, bool retain, bool prepaid)
{
    const Impl::Call call(*impl_);
    if (row.key.channel != impl_->channel || row.key.address != row.owner.get() || !row.references)
        throw GeometryError(
            GeometryErrorCode::InvalidInput, "Geometry owner graph record has a foreign channel or key.");
    return impl_->edge(static_cast<allocation_detail::Kind>(row.key.kind), row.owner, row.bytes, retain, prepaid);
}
std::size_t GeometryOwnerGraphEditor::references(const geometry_demand::OwnerRecord& row) const
{
    if (row.key.channel != impl_->channel || row.key.address != row.owner.get())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Graph reference query has a foreign key.");
    return impl_->references(static_cast<allocation_detail::Kind>(row.key.kind), row.owner, row.bytes);
}
template<ArithmeticMode M> void GeometryOwnerGraphEditor::geometry(const ConvexGeometryPtr<M>& owner, bool retain)
{
    const Impl::Call call(*impl_);
    require_mode(M, impl_->channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary);
    impl_->push(allocation_detail::Kind::TypedWrapper, owner, impl_->used, impl_->limit);
    impl_->template drain<M>(retain);
}
void GeometryOwnerGraphEditor::cartesian(const std::shared_ptr<const CartesianCellSource>& owner, bool retain)
{
    const Impl::Call call(*impl_);
    require_mode(
        ArithmeticMode::Ordinary, impl_->channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary);
    impl_->push(allocation_detail::Kind::CartesianSource, owner, impl_->used, impl_->limit);
    impl_->drain<ArithmeticMode::Ordinary>(retain);
}
template<ArithmeticMode M>
void GeometryOwnerGraphEditor::catalog(
    const std::shared_ptr<const CellGeometryCatalog<M>>& owner, bool retain, bool metadata_prepaid)
{
    const Impl::Call call(*impl_);
    if (!owner)
        return;
    require_mode(M, impl_->channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary);
    if (!impl_->edge(allocation_detail::Kind::CellCatalog, owner, owner->storage_bytes(), retain, metadata_prepaid))
        return;
    namespace G = geometry_demand;
    struct Context
    {
        GeometryOwnerGraphEditor& editor;
        bool retain, metadata_prepaid;
        void source(G::SourceKind kind, const std::shared_ptr<const void>& owner)
        {
            if (kind == G::SourceKind::Cartesian)
            {
                if constexpr (M == ArithmeticMode::Ordinary)
                    editor.cartesian(std::static_pointer_cast<const CartesianCellSource>(owner), retain);
                else
                    throw GeometryError(GeometryErrorCode::InvalidInput, "Foreign-mode procedural graph owner.");
                return;
            }
            const auto page = std::static_pointer_cast<const typename CellGeometryCatalog<M>::ExplicitPage>(owner);
            if (editor.impl_->edge(allocation_detail::Kind::CellShellPage, page, page->storage_bytes(), retain))
                for (const auto& geometry : page->cells)
                    editor.geometry(geometry, retain);
        }
    } context{*this, retain, metadata_prepaid};
    if (CellCatalogDemandAccess::persistent(*owner))
        CellCatalogDemandAccess::visit_dependencies(*owner,
            {&context,
                [](void* raw, const std::shared_ptr<const void>& owner, std::size_t bytes, std::size_t)
                {
                    auto& value = *static_cast<Context*>(raw);
                    return value.editor.external(owner, bytes, value.retain, value.metadata_prepaid);
                }},
            {&context, [](void* raw, G::SourceKind kind, const std::shared_ptr<const void>& owner)
                { static_cast<Context*>(raw)->source(kind, owner); }});
    else
    {
        if constexpr (M == ArithmeticMode::Ordinary)
            for (const auto& source : RetentionAccess::catalog_sources(*owner))
                cartesian(source, retain);
        for (const auto& page : RetentionAccess::catalog_pages(*owner))
            context.source(G::SourceKind::Explicit, page);
    }
}
template<ArithmeticMode M>
void GeometryOwnerGraphEditor::prepay_catalog_page(
    const std::shared_ptr<const typename CellGeometryCatalog<M>::ExplicitPage>& page)
{
    const Impl::Call call(*impl_);
    require_mode(M, impl_->channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary);
    if (!page || !impl_->ledger)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Catalog construction receipt lacks its typed ledger.");
    GeometryAllocationOwnerAccess::admit_graph_owner(*impl_->ledger, impl_->channel,
        static_cast<std::uint8_t>(allocation_detail::Kind::CellShellPage), page, page->storage_bytes(), impl_->used,
        impl_->limit, true);
}
void GeometryOwnerGraphEditor::planar(const PlanarFacePtr<ArithmeticMode::Ordinary>& owner, bool retain)
{
    const Impl::Call call(*impl_);
    if (!owner)
        return;
    require_mode(
        ArithmeticMode::Ordinary, impl_->channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary);
    impl_->adding = retain;
    const bool native = NativePlanarOwnerAccess::admitted(owner);
    if (native)
        impl_->native_typed_receipt(allocation_detail::Kind::PlanarWrapper, owner,
            NativePlanarOwnerAccess::wrapper_envelope(owner), {}, impl_->used, impl_->limit);
    if (!impl_->direct_edge(allocation_detail::Kind::PlanarWrapper, owner, native ? 0 : sizeof(*owner), retain))
        return;
    if (native)
        impl_->native_object_children(NativePlanarOwnerAccess::wrapper_envelope(owner), {}, impl_->used, impl_->limit);
    const auto payload = RetentionAccess::planar_payload_owner(*owner);
    if (native)
        impl_->native_typed_receipt(allocation_detail::Kind::PlanarPayload, payload,
            NativePlanarOwnerAccess::payload_envelope(owner), NativePlanarOwnerAccess::storage(owner),
            impl_->used, impl_->limit);
    if (impl_->direct_edge(
            allocation_detail::Kind::PlanarPayload, payload, native ? 0 : owner->exclusive_storage_bytes(), retain))
    {
        if (native)
            impl_->native_object_children(NativePlanarOwnerAccess::payload_envelope(owner),
                NativePlanarOwnerAccess::storage(owner), impl_->used, impl_->limit);
        if (const auto* support = RetentionAccess::planar_support_borrowed(*owner))
            impl_->retain_identity(*support, impl_->used, impl_->limit);
    }
}
void GeometryOwnerGraphEditor::identity(const PlaneSupportIdentity<ArithmeticMode::Ordinary>& identity, bool retain)
{
    const Impl::Call call(*impl_);
    impl_->adding = retain;
    impl_->retain_identity(identity, impl_->used, impl_->limit);
}
void GeometryOwnerGraphEditor::point_identity(const PointIdentity<ArithmeticMode::Ordinary>& value, bool retain)
{
    const Impl::Call call(*impl_);
    require_mode(
        ArithmeticMode::Ordinary, impl_->channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary);
    impl_->adding = retain;
    impl_->retain_identity(value, impl_->used, impl_->limit);
}
void GeometryOwnerGraphEditor::displacement_identity(
    const DisplacementIdentity<ArithmeticMode::Ordinary>& value, bool retain)
{
    const Impl::Call call(*impl_);
    require_mode(
        ArithmeticMode::Ordinary, impl_->channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary);
    impl_->adding = retain;
    impl_->retain_identity(value, impl_->used, impl_->limit);
}
void GeometryOwnerGraphEditor::native_fixed(
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& owner, bool retain, bool prepaid)
{
    const Impl::Call call(*impl_);
    impl_->native_fixed(owner, retain, prepaid);
}
bool GeometryOwnerGraphEditor::weak_observation(const NativeWeakOwnerObservation& value, bool retain)
{
    const Impl::Call call(*impl_);
    return impl_->weak_observation(value, retain);
}
void GeometryOwnerGraphEditor::native_state(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& owner, bool retain, bool prepaid)
{
    const Impl::Call call(*impl_);
    impl_->native_state(owner, retain, prepaid);
}
void GeometryOwnerGraphEditor::native_state_edges(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& owner, std::size_t references, bool retain)
{
    const Impl::Call call(*impl_);
    impl_->native_state(owner, retain, false, references);
}
void GeometryOwnerGraphEditor::record_paid_extent(const NativePaidExtentCharge& charge)
{
    const Impl::Call call(*impl_);
    impl_->edit_paid(impl_->financial_projection_depth ? charge.as_borrowed_financial() : charge);
}
void GeometryOwnerGraphEditor::financial_pin(const NativePaidExtentCharge& charge, bool retain)
{
    const Impl::Call call(*impl_);
    impl_->weak_observation(charge.physical_observation(), retain);
    impl_->weak_observation(charge.financial_observation(), retain);
    if (!charge.has_financial_pin())
        return;
    impl_->adding = retain;
    ++impl_->financial_projection_depth;
    struct Restore { Impl& value; ~Restore() { --value.financial_projection_depth; } } restore{*impl_};
    geometry_detail::NativeConstructorOwnerSink sink;
    sink.context = impl_;
    sink.financial_graph = this;
    sink.financial_retain = retain;
    sink.owner_edge = [](void* raw, const auto& owner, geometry_detail::NativeConstructorOwnerKind, std::size_t bytes)
    { auto& self = *static_cast<Impl*>(raw); return self.edge(allocation_detail::Kind::ExternalOwner, owner, bytes, self.adding); };
    sink.fixed_edge = [](void* raw, const auto& owner)
    { auto& self = *static_cast<Impl*>(raw); self.native_fixed(owner, self.adding, false); };
    sink.state_edge = [](void* raw, const auto& owner)
    { auto& self = *static_cast<Impl*>(raw); self.native_state(owner, self.adding, false); };
    sink.semantic_receipt = [](void* raw, const auto& owner, const auto& envelope)
    {
        auto& self = *static_cast<Impl*>(raw);
        self.native_typed_receipt(allocation_detail::Kind::ExternalOwner, owner, envelope, {}, self.used, self.limit);
    };
    sink.storage_receipt = [](void* raw, const auto& storage)
    {
        auto& self = *static_cast<Impl*>(raw);
        if (!self.adding) return;
        self.edit_paid(NativePaidExtentCharge::storage(storage));
        if (self.ledger)
            GeometryAllocationOwnerAccess::authorize_native_storage(*self.ledger, storage.owner, storage.envelope);
    };
    charge.project_financial(sink);
}
void GeometryOwnerGraphEditor::native_storage(const geometry_detail::NativeConstructorStorageOwner& value)
{
    record_paid_extent(NativePaidExtentCharge::storage(value));
    if (impl_->ledger)
        GeometryAllocationOwnerAccess::authorize_native_storage(*impl_->ledger, value.owner, value.envelope);
}
const NativeOwnerReceipts& GeometryOwnerGraphEditor::paid_receipts() const
{
    if (!impl_->finished)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry graph receipts require a finished edit.");
    return impl_->paid;
}
void GeometryOwnerGraphEditor::forget_paid_extent(const NativePaidExtentCharge& charge)
{
    const Impl::Call call(*impl_);
    if (charge.financial_address())
        impl_->edit_paid(charge, NativeOwnerReceipts::Action::RetireIfRecorded);
}
void GeometryOwnerGraphEditor::record_profile_charge(std::uint8_t kind, const std::shared_ptr<const void>& owner,
    std::size_t bytes, const std::shared_ptr<const void>& arena, const std::shared_ptr<const void>& lease,
    std::size_t snapshot, std::uint64_t revision, NativePaidExtentCharge::FinancialProjector projector,
    NativePaidExtentCharge::FinancialInspector inspector, NativePaidExtentCharge::FinancialValidator validator,
    const NativeWeakOwnerObservation& physical, const NativeWeakOwnerObservation& financial)
{
    record_paid_extent(NativePaidExtentCharge::source_profile(kind, owner, bytes, arena, lease, snapshot, revision,
        projector, inspector, validator).with_weak_observations(physical, financial));
}
void GeometryOwnerGraphEditor::record_subset_charge(std::uint8_t kind, const std::shared_ptr<const void>& owner,
    std::size_t bytes, const std::shared_ptr<const void>& arena, const std::shared_ptr<const void>& lease,
    std::size_t snapshot, std::uint64_t revision, NativePaidExtentCharge::FinancialProjector projector,
    NativePaidExtentCharge::FinancialInspector inspector, NativePaidExtentCharge::FinancialValidator validator,
    bool retain, const NativeWeakOwnerObservation& physical, const NativeWeakOwnerObservation& financial)
{
    const Impl::Call call(*impl_);
    auto charge = NativePaidExtentCharge::detached_subset(kind, owner, bytes, arena, lease, snapshot, revision,
        projector, inspector, validator).with_weak_observations(physical, financial);
    if (impl_->financial_projection_depth) charge = charge.as_borrowed_financial();
    impl_->edit_paid(charge, retain ? NativeOwnerReceipts::Action::Retain : NativeOwnerReceipts::Action::Retire);
}
geometry_demand::GeometryOwnerManifest GeometryOwnerGraphEditor::finish()
{
    const Impl::Call call(*impl_);
    const auto admission = impl_->admission();
    auto receipts = impl_->paid_edits ? impl_->paid_edits->rows() : std::span<NativeOwnerReceipts::Edit>{};
    impl_->paid = impl_->paid.edited_graph(receipts, admission, impl_->edits);
    impl_->recent_fixed_receipts.fill(nullptr);
    impl_->paid_edits.reset();
    impl_->result = impl_->result.edited_graph(impl_->edits, admission, impl_->work ? &impl_->work->index : nullptr);
    impl_->edits.clear();
    if (impl_->ledger && impl_->credited_predecessor && impl_->result.owner_count())
        GeometryOwnerImportAccess::certify_constructed(*impl_->ledger, impl_->result, impl_->used, impl_->limit);
    impl_->finished = true;
    return impl_->result;
}
template void GeometryOwnerGraphEditor::geometry<ArithmeticMode::Ordinary>(
    const ConvexGeometryPtr<ArithmeticMode::Ordinary>&, bool);
template void GeometryOwnerGraphEditor::geometry<ArithmeticMode::BoundedCertification>(
    const ConvexGeometryPtr<ArithmeticMode::BoundedCertification>&, bool);
template void GeometryOwnerGraphEditor::catalog<ArithmeticMode::Ordinary>(
    const std::shared_ptr<const CellGeometryCatalog<ArithmeticMode::Ordinary>>&, bool, bool);
template void GeometryOwnerGraphEditor::catalog<ArithmeticMode::BoundedCertification>(
    const std::shared_ptr<const CellGeometryCatalog<ArithmeticMode::BoundedCertification>>&, bool, bool);
template void GeometryOwnerGraphEditor::prepay_catalog_page<ArithmeticMode::Ordinary>(
    const std::shared_ptr<const CellGeometryCatalog<ArithmeticMode::Ordinary>::ExplicitPage>&);
template void GeometryOwnerGraphEditor::prepay_catalog_page<ArithmeticMode::BoundedCertification>(
    const std::shared_ptr<const CellGeometryCatalog<ArithmeticMode::BoundedCertification>::ExplicitPage>&);
} // namespace SimpleFluid::PLIC::details
