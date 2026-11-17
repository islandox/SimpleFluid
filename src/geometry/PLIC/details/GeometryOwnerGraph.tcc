/** @file GeometryOwnerGraph.tcc
 * @brief Compiled persistent first/last immutable physical geometry graph edits.
 */
#pragma once
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/GeometryRetentionCensusAccess.hh"
#include <algorithm>
#include <exception>

namespace SimpleFluid::PLIC::details
{
bool GeometryOwnerGraphEditor::Impl::KeyLess::operator()(
    const geometry_demand::OwnerKey& a, const geometry_demand::OwnerKey& b) const noexcept
{
    if (a.channel != b.channel)
        return a.channel < b.channel;
    if (a.kind != b.kind)
        return a.kind < b.kind;
    return std::less<const void*>{}(a.address, b.address);
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
                allocation_detail::exhausted("Geometry owner graph index budget is exhausted.");
            self.used += bytes;
        },
        [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
        {
            auto& self = *static_cast<Impl*>(raw);
            if (self.ledger)
                GeometryAllocationOwnerAccess::retain_manifest_metadata(
                    *self.ledger, self.channel, owner, payload, admitted, self.used, self.limit, true);
        }};
}
template<class Kind>
std::size_t GeometryOwnerGraphEditor::Impl::references(
    Kind kind, const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    const auto key = static_cast<std::uint8_t>(kind);
    const geometry_demand::OwnerKey lookup{channel, key, owner.get()};
    const geometry_demand::OwnerRecord* found = nullptr;
    for (std::size_t index = 0; index < row_count; ++index)
        if (rows[index].key.channel == channel && rows[index].key.kind == key && rows[index].key.address == owner.get())
        {
            found = &rows[index];
            break;
        }
    if (!found)
        if (const auto row = overlay.find(lookup); row != overlay.end())
            found = &row->second;
    if (found)
    {
        if (found->bytes != bytes || owner.owner_before(found->owner) || found->owner.owner_before(owner))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Paired graph owner changes control or size.");
        return found->references;
    }
    const auto count = result.references(channel, key, owner, work ? &work->lookup : nullptr);
    if (count && !result.admits(channel, key, owner, bytes, work ? &work->lookup : nullptr))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Paired graph owner changes immutable size.");
    return count;
}
template<class Kind>
bool GeometryOwnerGraphEditor::Impl::edge(
    Kind kind, const std::shared_ptr<const void>& owner, std::size_t bytes, bool retain, bool prepaid)
{
    if (!owner)
        return false;
    if (work)
        ++work->edge_visits;
    const auto key = static_cast<std::uint8_t>(kind);
    const geometry_demand::OwnerKey owner_key{channel, key, owner.get()};
    geometry_demand::OwnerRecord* record_pointer = nullptr;
    for (std::size_t index = 0; index < row_count; ++index)
        if (rows[index].key.channel == channel && rows[index].key.kind == key && rows[index].key.address == owner.get())
        {
            record_pointer = &rows[index];
            break;
        }
    if (!record_pointer)
        if (const auto found = overlay.find(owner_key); found != overlay.end())
            record_pointer = &found->second;
    if (!record_pointer)
    {
        const auto references = result.references(channel, key, owner, work ? &work->lookup : nullptr);
        if (references && !result.admits(channel, key, owner, bytes, work ? &work->lookup : nullptr))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry graph owner size changed.");
        geometry_demand::OwnerRecord value{owner_key, owner, bytes, references};
        if (row_count < rows.size())
        {
            record_pointer = &rows[row_count++];
            *record_pointer = std::move(value);
        }
        else
        {
            admission().charge(1, sizeof(typename decltype(overlay)::value_type) + 4 * sizeof(void*));
            record_pointer = &overlay.emplace(owner_key, std::move(value)).first->second;
        }
        if (work)
            ++work->dependency_entries;
    }
    auto& record = *record_pointer;
    if (record.bytes != bytes || owner.owner_before(record.owner) || record.owner.owner_before(owner))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry graph owner changes controls or bytes.");
    bool transition = false;
    if (retain)
    {
        transition = record.references == 0;
        record.references = allocation_detail::add(record.references, 1);
        if (transition)
        {
            if (work)
                ++work->first_transitions;
            if (ledger)
                GeometryAllocationOwnerAccess::admit_graph_owner(
                    *ledger, channel, key, owner, bytes, used, limit, prepaid);
        }
    }
    else
    {
        if (!record.references)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Retired geometry graph owner is absent.");
        transition = --record.references == 0;
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
template<GeometryIdentityKind Kind, ArithmeticMode Mode>
void GeometryOwnerGraphEditor::Impl::retain_identity(
    const GeometryIdentity<Kind, Mode>& identity, std::size_t&, std::size_t)
{
    visit_identity_allocations(identity, [&](const GeometryIdentityAllocation& value)
        { return edge(allocation_detail::Kind::IdentityPayload, value.owner, value.payload_bytes, adding); });
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
            { return edge(value.kind, value.owner, bytes, adding); });
    }
}

GeometryOwnerGraphEditor::GeometryOwnerGraphEditor(geometry_demand::GeometryOwnerManifest accepted,
    std::uint8_t channel, std::size_t& used, std::size_t limit, GeometryAllocationLedger* ledger,
    GeometryOwnerGraphWork* work)
    : state_{std::move(accepted), channel, used, limit, ledger, work, rows_, pending_}
{
    try
    {
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
std::size_t GeometryOwnerGraphEditor::edit_byte_bound(
    const geometry_demand::GeometryOwnerManifest& accepted, std::size_t edges, std::size_t changed_keys)
{
    using namespace allocation_detail;
    const auto excess = edges > geometry_demand::leaf_capacity ? edges - geometry_demand::leaf_capacity : 0;
    auto bytes = product(excess, sizeof(typename decltype(Impl::overlay)::value_type) + 4 * sizeof(void*));
    // Above the inline frontier, final rows need one contiguous charged buffer.
    if (excess)
        bytes = add(bytes, product(edges, sizeof(geometry_demand::OwnerRecord)));
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
    bytes = add(bytes, accepted.batch_edit_byte_bound(changed_keys));
    // Each newly allocated index node/State produces at most one payload map
    // row and one metadata-credit registry row. Both are charged before insert.
    const auto bookkeeping_per_owner =
        sizeof(typename decltype(GeometryAllocationLedger::Impl::owners)::value_type) + 4 * sizeof(void*) +
        sizeof(typename decltype(GeometryAllocationLedger::Impl::metadata_credits)::value_type) + 4 * sizeof(void*);
    bytes = add(bytes, product(accepted.batch_edit_metadata_owner_bound(changed_keys), bookkeeping_per_owner));
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
    if (!impl_->edge(allocation_detail::Kind::PlanarWrapper, owner, sizeof(*owner), retain))
        return;
    if (impl_->edge(allocation_detail::Kind::PlanarPayload, RetentionAccess::planar_payload_owner(*owner),
            owner->exclusive_storage_bytes(), retain))
        if (const auto* support = RetentionAccess::planar_support_borrowed(*owner))
        {
            impl_->adding = retain;
            impl_->retain_identity(*support, impl_->used, impl_->limit);
        }
}
geometry_demand::GeometryOwnerManifest GeometryOwnerGraphEditor::finish()
{
    const Impl::Call call(*impl_);
    const auto changed = [&](const geometry_demand::OwnerRecord& row)
    {
        return impl_->result.references(row.key.channel, row.key.kind, row.owner,
                   impl_->work ? &impl_->work->lookup : nullptr) != row.references;
    };
    const auto less = [](const auto& first, const auto& second) { return Impl::KeyLess{}(first.key, second.key); };
    const auto publish = [&](std::span<const geometry_demand::OwnerRecord> changes)
    {
        impl_->result = impl_->result.edited(changes, impl_->admission(), impl_->work ? &impl_->work->index : nullptr,
            impl_->work ? &impl_->work->lookup : nullptr);
    };
    if (impl_->overlay.empty())
    {
        // Compact and sort the caller's existing bounded buffer; no final heap
        // row allocation is needed for the common <=64 touched-owner edit.
        std::size_t count = 0;
        for (std::size_t index = 0; index < impl_->row_count; ++index)
            if (changed(impl_->rows[index]))
            {
                if (count != index)
                    impl_->rows[count] = std::move(impl_->rows[index]);
                ++count;
            }
        auto values = impl_->rows.first(count);
        std::sort(values.begin(), values.end(), less);
        publish(values);
    }
    else
    {
        const auto total = allocation_detail::add(impl_->row_count, impl_->overlay.size());
        impl_->admission().charge(total, sizeof(geometry_demand::OwnerRecord));
        std::vector<geometry_demand::OwnerRecord> changes;
        changes.reserve(total);
        if (changes.capacity() != total)
            allocation_detail::exhausted("Geometry graph final row capacity is unqualified.");
        for (std::size_t index = 0; index < impl_->row_count; ++index)
            if (changed(impl_->rows[index]))
                changes.push_back(impl_->rows[index]);
        for (const auto& [key, row] : impl_->overlay)
            if (changed(row))
                changes.push_back(row);
        std::sort(changes.begin(), changes.end(), less);
        publish(changes);
    }
    for (auto& row : impl_->rows.first(impl_->row_count))
        row = {};
    impl_->row_count = 0;
    impl_->overlay.clear();
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
