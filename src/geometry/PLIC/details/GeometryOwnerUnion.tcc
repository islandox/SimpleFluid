#pragma once
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"
#include <limits>
#include <map>
#include <string>
namespace SimpleFluid::PLIC::details
{
std::size_t OwnerManifestPair::logical_payload_bytes() const
{
    return allocation_detail::add(owners[0].payload_bytes(), owners[1].payload_bytes());
}
std::size_t OwnerManifestPair::overlap_bytes() const
{
    for (unsigned channel = 0; channel < 2; ++channel)
        if (const auto actual = owners[channel].uniform_channel(); actual && *actual != channel)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Owner pair has a foreign arithmetic channel.");
    if (sealed_)
    {
        if (!owners[0].authenticates(sealed_roots_[0]) || !owners[1].authenticates(sealed_roots_[1]))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Owner pair changed its sealed root controls.");
    }
    else if (owners[0].owner_count() && owners[1].owner_count())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Dual-channel owner pair lacks an overlap proof.");
    if (overlap_ > std::min(owners[0].payload_bytes(), owners[1].payload_bytes()))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Owner pair overlap exceeds its physical payload.");
    return overlap_;
}
void OwnerManifestPair::require_seal() const
{
    (void) overlap_bytes();
    if ((owners[0].owner_count() || owners[1].owner_count()) && !sealed_)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Import requires exact sealed pair root controls.");
}
std::size_t OwnerManifestPair::payload_bytes() const
{
    return logical_payload_bytes() - overlap_bytes();
}
std::size_t OwnerManifestPair::index_admission_bytes() const
{
    return allocation_detail::add(owners[0].metadata_admission_bytes(), owners[1].metadata_admission_bytes());
}
OwnerManifestPair OwnerManifestPair::sealed(
    std::array<geometry_demand::GeometryOwnerManifest, 2> roots, std::size_t overlap)
{
    OwnerManifestPair result;
    result.owners = std::move(roots);
    for (unsigned channel = 0; channel < 2; ++channel)
    {
        if (const auto actual = result.owners[channel].uniform_channel(); actual && *actual != channel)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Owner pair has a foreign arithmetic channel.");
        result.sealed_roots_[channel] = result.owners[channel].root_control();
    }
    result.sealed_ = true;
    result.overlap_ = overlap;
    (void) result.overlap_bytes();
    return result;
}
GeometryOwnerRootBinding::GeometryOwnerRootBinding(std::shared_ptr<const void> header, std::size_t bytes,
    OwnerManifestPair payload, const NativeOwnerReceipts* paid, const NativePaidExtentGroup* source_group)
    : header_(std::move(header)), header_bytes_(bytes), payload_(std::move(payload)), paid_(paid), group_(source_group)
{
    const std::shared_ptr<const void> empty;
    if (!header_ || (!header_.owner_before(empty) && !empty.owner_before(header_)))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Root binding needs an actual immutable header control.");
    if (!payload_.sealed_ && (!payload_.owners[0].owner_count() || !payload_.owners[1].owner_count()))
        payload_ = OwnerManifestPair::sealed(std::move(payload_.owners), 0);
    payload_.require_seal();
    if (group_ && !group_->valid())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Source group is not a closed valid certificate.");
}
namespace owner_union_detail
{
class OriginEdits
{
    using Row = NativePaidOriginTotals::Assignment;
    using Overflow = geometry_detail::NativeAdministrationMap<const void*, Row, std::less<const void*>>;

public:
    OriginEdits(NativePaidOriginTotals& totals, geometry_demand::Admission admission)
        : totals_(totals), admission_(std::move(admission)),
          overflow_(std::less<const void*>{}, admission_.allocator<Overflow::value_type>())
    {
    }
    void change(const std::shared_ptr<const void>& arena, std::size_t bytes, bool retain)
    {
        if (!bytes) return;
        if (!arena)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Native paid origin has no actual arena control.");
        Row& row = find_or_insert(arena);
        if (row.arena.owner_before(arena) || arena.owner_before(row.arena))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Native paid origin changes actual arena control.");
        if (retain)
            row.bytes = allocation_detail::add(row.bytes, bytes);
        else
        {
            if (bytes > row.bytes)
                throw GeometryError(GeometryErrorCode::InvalidInput, "Native paid origin extent underflows.");
            row.bytes -= bytes;
        }
    }
    void flush()
    {
        const auto less = [](const Row& a, const Row& b) { return std::less<const void*>{}(a.arena.get(), b.arena.get()); };
        auto rows = std::span(inline_).first(size_);
        if (overflow_.empty())
        {
            std::sort(rows.begin(), rows.end(), less);
            totals_ = totals_.edited(rows, admission_);
        }
        else
        {
            const auto count = allocation_detail::add(size_, overflow_.size());
            geometry_detail::NativeAdministrationVector<Row> merged(admission_.allocator<Row>());
            admission_.charge(count, sizeof(Row));
            merged.reserve(count);
            if (merged.capacity() != count)
                allocation_detail::exhausted("Paid origin edit capacity is unqualified.");
            merged.insert(merged.end(), rows.begin(), rows.end());
            for (const auto& [address, row] : overflow_) merged.push_back(row);
            std::sort(merged.begin(), merged.end(), less);
            totals_ = totals_.edited(merged, admission_);
            overflow_.clear();
        }
        for (auto& row : rows) row = {};
        size_ = 0;
    }

private:
    Row& find_or_insert(const std::shared_ptr<const void>& arena)
    {
        for (auto& row : std::span(inline_).first(size_))
            if (row.arena.get() == arena.get()) return row;
        if (size_ < inline_.size())
        {
            inline_[size_] = {arena, totals_.bytes(arena)};
            return inline_[size_++];
        }
        if (const auto found = overflow_.find(arena.get()); found != overflow_.end())
            return found->second;
        admission_.charge(1, sizeof(Overflow::value_type) + 4 * sizeof(void*));
        return overflow_.emplace(arena.get(), Row{arena, totals_.bytes(arena)}).first->second;
    }
    NativePaidOriginTotals& totals_;
    geometry_demand::Admission admission_;
    std::array<Row, 4> inline_{};
    std::size_t size_ = 0;
    Overflow overflow_;
};
struct ControlEdits
{
    const GeometryIndexControls& accepted;
    geometry_demand::Admission admission;
    std::array<IndexControlRecord, geometry_demand::leaf_capacity> inline_rows{};
    std::size_t size = 0;
    geometry_detail::NativeAdministrationMap<const void*, IndexControlRecord, std::less<const void*>> overflow;
    ControlEdits(const GeometryIndexControls& current, geometry_demand::Admission allocation)
        : accepted(current), admission(std::move(allocation)),
          overflow(std::less<const void*>{}, admission.allocator<std::pair<const void* const, IndexControlRecord>>())
    {
    }
    IndexControlRecord* changed(const std::shared_ptr<const void>& owner)
    {
        for (std::size_t i = 0; i < size; ++i)
            if (inline_rows[i].owner.get() == owner.get())
                return &inline_rows[i];
        if (auto found = overflow.find(owner.get()); found != overflow.end())
            return &found->second;
        return nullptr;
    }
    bool known(const std::shared_ptr<const void>& owner, std::size_t payload)
    {
        if (const auto* row = changed(owner))
        {
            if (row->payload != payload || owner.owner_before(row->owner) || row->owner.owner_before(owner))
                throw GeometryError(GeometryErrorCode::InvalidInput, "Index-control delta changes immutable storage.");
            return row->native || row->control != 0;
        }
        return accepted.holds(owner, payload);
    }
    void put(const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t control, bool native = false)
    {
        if (auto* row = changed(owner))
        {
            row->control = control;
            row->native = native;
            return;
        }
        if (size < inline_rows.size())
            inline_rows[size++] = {owner, payload, control, native};
        else
        {
            admission.charge(1, sizeof(typename decltype(overflow)::value_type) + 4 * sizeof(void*));
            overflow.emplace(owner.get(), IndexControlRecord{owner, payload, control, native});
        }
    }
    void classify(const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
    {
        if (!known(owner, payload))
            put(owner, payload, admitted - payload, admitted == payload);
    }
    void retire(const std::shared_ptr<const void>& owner, std::size_t payload)
    {
        if (known(owner, payload))
            put(owner, payload, 0);
    }
    GeometryIndexControls finish(geometry_demand::PageWork* work)
    {
        const auto less = [](const auto& a, const auto& b)
        { return std::less<const void*>{}(a.owner.get(), b.owner.get()); };
        if (overflow.empty())
        {
            auto rows = std::span(inline_rows).first(size);
            std::sort(rows.begin(), rows.end(), less);
            return accepted.edited(rows, admission, work);
        }
        const auto count = allocation_detail::add(size, overflow.size());
        admission.charge(count, sizeof(IndexControlRecord));
        geometry_detail::NativeAdministrationVector<IndexControlRecord> rows(admission.allocator<IndexControlRecord>());
        rows.reserve(count);
        if (rows.capacity() != count)
            allocation_detail::exhausted("Unqualified index-control delta capacity.");
        rows.insert(rows.end(), inline_rows.begin(), inline_rows.begin() + size);
        for (const auto& [key, row] : overflow)
            rows.push_back(row);
        std::sort(rows.begin(), rows.end(), less);
        return accepted.edited(rows, admission, work);
    }
};
} // namespace owner_union_detail
std::size_t GeometryNativeAdministration::resident_bytes() const noexcept
{
    return owners.payload_bytes() - body_credit + owners.metadata_admission_bytes() +
           domain_paid.metadata_admission_bytes() + paid.metadata_admission_bytes() +
           origins.metadata_admission_bytes() + direct_origins.metadata_admission_bytes() +
           publication_origins.metadata_admission_bytes() + group_overlap.metadata_admission_bytes();
}
std::size_t GeometryNativeAdministration::native_paid_bytes(const std::shared_ptr<const void>& arena) const
{
    if (source_group && arena && source_group->origin().get() == arena.get() &&
        (source_group->origin().owner_before(arena) || arena.owner_before(source_group->origin())))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Native source arena has foreign shared control.");
    auto bytes = origins.bytes(arena);
    if (arena && current && current->mirror.owner.get() == arena.get() &&
        !current->mirror.owner.owner_before(arena) && !arena.owner_before(current->mirror.owner))
        bytes = allocation_detail::add(bytes, current_paid_bytes);
    if (source_group && source_group->origin().get() == arena.get() &&
        !source_group->origin().owner_before(arena) && !arena.owner_before(source_group->origin()))
    {
        const auto overlap = group_overlap.bytes(arena);
        if (overlap > bytes || overlap > source_group->bytes())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Native source intersection exceeds paid extent.");
        bytes = allocation_detail::add(bytes - overlap, source_group->bytes());
    }
    return bytes;
}
std::size_t GeometryNativeAdministration::direct_h_paid_bytes(const std::shared_ptr<const void>& arena) const
{
    auto bytes = direct_origins.bytes(arena);
    if (arena && current && current->mirror.owner.get() == arena.get() &&
        !current->mirror.owner.owner_before(arena) && !arena.owner_before(current->mirror.owner))
        bytes = allocation_detail::add(bytes, current_paid_bytes);
    return bytes;
}
std::size_t GeometryOwnerUnion::direct_h_paid_bytes(const std::shared_ptr<const void>& arena) const
{
    return native_administration_.direct_h_paid_bytes(arena);
}
std::size_t GeometryNativeAdministration::publication_h_paid_bytes(const std::shared_ptr<const void>& arena) const
{
    auto bytes = publication_origins.bytes(arena);
    if (arena && current && current->mirror.owner.get() == arena.get())
    {
        if (current->mirror.owner.owner_before(arena) || arena.owner_before(current->mirror.owner))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Native publication arena has foreign shared control.");
        bytes = allocation_detail::add(bytes, current_publication_paid_bytes);
    }
    return bytes;
}
std::size_t GeometryOwnerUnion::publication_h_paid_bytes(const std::shared_ptr<const void>& arena) const
{
    return native_administration_.publication_h_paid_bytes(arena);
}
std::size_t GeometryOwnerUnion::native_paid_bytes(const std::shared_ptr<const void>& arena) const
{
    return native_administration_.native_paid_bytes(arena);
}
std::size_t GeometryOwnerUnion::native_group_covered_bytes(const std::shared_ptr<const void>& arena) const
{
    const auto* group = native_administration_.source_group;
    if (group && arena && group->origin().get() == arena.get() &&
        (group->origin().owner_before(arena) || arena.owner_before(group->origin())))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Native source arena has foreign shared control.");
    if (!group || !arena || group->origin().get() != arena.get() || group->origin().owner_before(arena) ||
        arena.owner_before(group->origin()))
        return 0;
    const auto intersection = native_administration_.group_overlap.bytes(arena);
    if (intersection > group->bytes())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Native paid source intersection exceeds group extent.");
    return group->bytes() - intersection;
}
bool GeometryNativeAdministration::admits(const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    return owners.admits(0, GeometryOwnerGraphEditor::external_kind(), owner, bytes);
}
std::size_t GeometryNativeAdministration::uncovered_owner_bytes(
    const std::shared_ptr<const void>& owner, std::size_t bytes, const OwnerManifestPair& body) const
{
    if (!admits(owner, bytes))
        return 0;
    for (unsigned channel = 0; channel != 2; ++channel)
        if (body.owners[channel].admits(channel, GeometryOwnerGraphEditor::external_kind(), owner, bytes))
            return 0;
    return bytes;
}
void GeometryNativeAdministration::visit(geometry_demand::MetadataSink metadata) const
{
    struct Context
    {
        geometry_demand::MetadataSink sink;
    } context{metadata};
    owners.visit_dependencies(metadata, {&context, [](void* raw, const geometry_demand::OwnerRecord& row)
                                            { static_cast<Context*>(raw)->sink.native_owner(row.owner, row.bytes); }});
    domain_paid.visit_administration(metadata);
    paid.visit_administration(metadata);
    origins.visit_administration(metadata);
    direct_origins.visit_administration(metadata);
    publication_origins.visit_administration(metadata);
    group_overlap.visit_administration(metadata);
    if (current)
        geometry_demand::visit_native_state(current, metadata);
}
bool GeometryOwnerUnion::admits_native_administration(const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    return native_administration_.admits(owner, bytes);
}
std::size_t GeometryOwnerUnion::native_administration_bytes() const noexcept
{
    return native_administration_.resident_bytes();
}
GeometryNativeAdministration GeometryNativeAdministration::updated(const OwnerManifestPair& previous_body,
    const OwnerManifestPair& body, const GeometryIndexControls& previous_controls,
    const GeometryIndexControls& controls, std::size_t& used, std::size_t limit,
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& state, GeometryOwnerUnionWork& work,
    const NativeOwnerReceipts* previous_receipts, const NativeOwnerReceipts* receipts,
    const NativeOwnerReceipts* previous_bounded_receipts, const NativeOwnerReceipts* bounded_receipts,
    std::shared_ptr<const void> new_source_group_header, const NativePaidExtentGroup* new_source_group) const
{
    using namespace geometry_demand;
    using namespace allocation_detail;
    if (!body.owners[0].owner_count() && !body.owners[1].owner_count() && !controls.root_control() &&
        (!receipts || receipts->empty()) && (!bounded_receipts || bounded_receipts->empty()) && !new_source_group)
        return {};
    if (!state && !owners.owner_count() && !body.owners[0].has_native_administration() &&
        !body.owners[1].has_native_administration() && !controls.has_native_administration() &&
        (!receipts || receipts->empty()) && (!bounded_receipts || bounded_receipts->empty()) && !new_source_group)
        return {};
    GeometryNativeAdministration result = *this;
    result.current = state;
    const auto same_control = [](const auto& a, const auto& b)
    { return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a); };
    const bool same_current = same_control(current, state);
    const auto old_arena = current ? current->mirror.owner : std::shared_ptr<const void>{};
    const auto new_arena = state ? state->mirror.owner : std::shared_ptr<const void>{};
    result.current_references = same_current ? current_references : 0;
    struct Edge
    {
        std::shared_ptr<const geometry_detail::NativeAllocationState> state;
        std::size_t references;
        bool retain;
    };
    geometry_detail::NativeAdministrationVector<Edge> pending{
        geometry_detail::NativeAdministrationAllocator<Edge>(state)};
    struct FinancialEdge { const NativePaidExtentCharge* charge; bool retain; };
    geometry_detail::NativeAdministrationVector<FinancialEdge> pending_financial{
        geometry_detail::NativeAdministrationAllocator<FinancialEdge>(state)};
    const auto charge_work = [&](std::size_t bytes)
    {
        if (used > limit || bytes > limit - used)
            throw GeometryError(GeometryErrorCode::ResourceLimit,
                "Native administration work budget is exhausted: used=" + std::to_string(used) +
                    " request=" + std::to_string(bytes) + " limit=" + std::to_string(limit) + ".");
        if (state)
            state->record_ledger_work(bytes);
        used += bytes;
    };
    const Admission admission{const_cast<void*>(static_cast<const void*>(&charge_work)),
        [](void* raw, std::size_t bytes) { (*static_cast<const decltype(charge_work)*>(raw))(bytes); },
        nullptr, state};
    owner_union_detail::OriginEdits all_origins(result.origins, admission);
    owner_union_detail::OriginEdits direct_h(result.direct_origins, admission);
    owner_union_detail::OriginEdits publication_h(result.publication_origins, admission);
    owner_union_detail::OriginEdits overlap(result.group_overlap, admission);
    geometry_demand::ReceiptEdits receipt_edits(admission);
    const auto flush_origins = [&]
    {
        all_origins.flush();
        direct_h.flush();
        publication_h.flush();
        overlap.flush();
    };
    const auto scalar_change = [](std::size_t& total, std::size_t bytes, bool retain)
    {
        if (retain)
            total = add(total, bytes);
        else
        {
            if (bytes > total)
                throw GeometryError(GeometryErrorCode::InvalidInput, "Native paid extent underflows.");
            total -= bytes;
        }
    };
    // The current arena's own auxiliary-index extent is a scalar. Updating it
    // cannot allocate another node in the index it is measuring. An arena
    // transition materializes that scalar once; exact old-node retirements
    // subsequently subtract from its cached origin row.
    if (!same_control(old_arena, new_arena))
    {
        if (old_arena && current_paid_bytes)
        {
            all_origins.change(old_arena, current_paid_bytes, true);
            direct_h.change(old_arena, current_paid_bytes, true);
        }
        if (old_arena && current_publication_paid_bytes)
            publication_h.change(old_arena, current_publication_paid_bytes, true);
        result.current_paid_bytes = 0;
        result.current_publication_paid_bytes = 0;
    }
    const auto origin_change = [&](const NativePaidExtentCharge& charge, bool retain)
    {
        const auto arena = charge.origin();
        const auto bytes = charge.bytes();
        all_origins.change(arena, bytes, retain);
        direct_h.change(arena, bytes, retain);
        if (charge.publication_h())
            publication_h.change(arena, bytes, retain);
    };
    const auto overlap_change = [&](const NativePaidExtentCharge& value, bool retain)
    {
        const auto owner = value.owner();
        if (owner && result.source_group && result.source_group->contains(value.kind(), owner, value.bytes()))
            overlap.change(value.origin(), value.bytes(), retain);
    };
    // All input channels and metadata roles meet in this physical receipt
    // index. First/last (owner, actual-origin) transitions update cached totals;
    // graph reference multiplicity never duplicates a physical byte extent.
    const auto paid_change = [&](const NativePaidExtentCharge& value, bool retain)
    {
        if (!value.source_only() && (!retain || value.valid()))
            receipt_edits.append(value,
                retain ? NativeOwnerReceipts::Action::Retain : NativeOwnerReceipts::Action::RetireIfRecorded);
    };
    const auto flush_paid = [&]
    {
        auto changes = receipt_edits.rows();
        const auto old_paid = result.paid;
        result.paid = old_paid.edited(changes, admission);
        if (!result.paid.same_root(old_paid))
            // Receipt keys are address-sorted for page edits. Origin credit
            // must still acquire replacement extents before it retires the
            // old graph's extents, including a same-arena metadata exchange.
            for (const bool retain : {true, false})
            {
                const NativePaidExtentCharge* previous = nullptr;
                for (const auto& edit : changes)
                {
                    const auto& value = *edit.charge;
                    if (previous && previous->same_physical_origin(value))
                        continue;
                    previous = &value;
                    const bool before = old_paid.contains_any(value);
                    const bool after = result.paid.contains_any(value);
                    if (before != after && after == retain)
                    {
                        all_origins.change(value.origin(), value.bytes(), retain);
                        overlap_change(value, retain);
                    }
                    const bool direct_before = old_paid.contains_direct_h(value);
                    const bool direct_after = result.paid.contains_direct_h(value);
                    if (direct_before != direct_after && direct_after == retain)
                        direct_h.change(value.origin(), value.bytes(), retain);
                    const bool publication_before = old_paid.contains_publication_h(value);
                    const bool publication_after = result.paid.contains_publication_h(value);
                    if (publication_before != publication_after && publication_after == retain)
                        publication_h.change(value.origin(), value.bytes(), retain);
                }
            }
        receipt_edits.clear();
    };
    const auto receipt_delta = [&](const NativeOwnerReceipts& before, const NativeOwnerReceipts& after)
    {
        struct Context
        {
            geometry_demand::ReceiptEdits& edits;
            bool retain;
        } add_context{receipt_edits, true}, retire_context{receipt_edits, false};
        const auto visit = [](void* raw, const NativePaidExtentCharge& value, std::size_t)
        {
            auto& context = *static_cast<Context*>(raw);
            if (!value.source_only() && (!context.retain || value.valid()))
                context.edits.append_borrowed(value, context.retain ? NativeOwnerReceipts::Action::Retain :
                    NativeOwnerReceipts::Action::RetireIfRecorded);
        };
        after.visit_added(before, {&add_context, visit}, &work.native_administration);
        before.visit_added(after, {&retire_context, visit}, &work.native_administration);
    };
    const auto append = [&](const std::shared_ptr<const geometry_detail::NativeAllocationState>& owner,
                            std::size_t count, bool retain)
    {
        if (!count || !owner)
            return;
        work.native_domain_edges = add(work.native_domain_edges, count);
        if (same_control(owner, result.current))
        {
            scalar_change(result.current_references, count, retain);
            return;
        }
        if (pending.size() == pending.capacity())
        {
            const auto count = pending.capacity() ? product(pending.capacity(), 2) : 1;
            charge_work(product(count, sizeof(Edge)));
            pending.reserve(count);
            if (pending.capacity() != count)
                exhausted("Native administration scratch capacity is unqualified.");
        }
        pending.push_back({owner, count, retain});
    };
    const auto append_financial = [&](const NativePaidExtentCharge& charge, bool retain)
    {
        if (!charge.has_lifetime_children()) return;
        if (pending_financial.size() == pending_financial.capacity())
        {
            const auto count = pending_financial.capacity() ? product(pending_financial.capacity(), 2) : 1;
            charge_work(product(count, sizeof(FinancialEdge)));
            pending_financial.reserve(count);
            if (pending_financial.capacity() != count)
                exhausted("Native financial projection scratch capacity is unqualified.");
        }
        pending_financial.push_back({&charge, retain});
    };
    struct Capture
    {
        const decltype(append)& append_edge;
        const decltype(paid_change)& paid_edge;
        const decltype(origin_change)& origin_edge;
        const decltype(scalar_change)& scalar;
        const decltype(same_control)& same;
        const decltype(append_financial)& financial_edge;
        GeometryNativeAdministration& result;
        bool auxiliary = false;
        void owner(const std::shared_ptr<const void>& owner, bool retain)
        {
            const auto state = metadata_state(owner);
            if (!state)
                return;
            const auto* receipt = std::get_deleter<NativeMetadataDelete>(owner);
            const auto bytes = receipt->extent_bytes(owner.get(), receipt->control_bytes);
            const auto paid = NativePaidExtentCharge::metadata(owner, bytes);
            if (!auxiliary)
                paid_edge(paid, retain);
            else if (paid.valid())
            {
                const auto arena = result.current ? result.current->mirror.owner : std::shared_ptr<const void>{};
                if (same(paid.origin(), arena))
                {
                    scalar(result.current_paid_bytes, bytes, retain);
                    if (paid.publication_h())
                        scalar(result.current_publication_paid_bytes, bytes, retain);
                }
                else
                    origin_edge(paid, retain);
            }
        }
    } external{append, paid_change, origin_change, scalar_change, same_control, append_financial, result};
    const auto sink = [](Capture& context)
    {
        return NativeAdministrationSink{&context,
            [](void* raw, const auto& state, std::size_t count, bool retain)
            { static_cast<Capture*>(raw)->append_edge(state, count, retain); },
            [](void* raw, const auto& owner, bool retain) { static_cast<Capture*>(raw)->owner(owner, retain); },
            [](void* raw, const NativePaidExtentCharge& charge, bool retain)
            { static_cast<Capture*>(raw)->financial_edge(charge, retain); }};
    };
    if (!same_current)
    {
        if (current)
        {
            append(current, current_references, true);
            append(current, 1, false);
        }
        if (state)
        {
            // The new fixed holder is not an index edge and must be stored.
            pending.push_back({state, 1, true});
        }
    }
    else if (state && !owners.owner_count())
        pending.push_back({state, 1, true});
    for (unsigned channel = 0; channel != 2; ++channel)
        body.owners[channel].visit_native_administration_delta(
            previous_body.owners[channel], sink(external), &work.native_administration);
    controls.visit_native_administration_delta(previous_controls, sink(external), &work.native_administration);
    const NativeOwnerReceipts empty;
    receipt_delta(previous_receipts ? *previous_receipts : empty, receipts ? *receipts : empty);
    receipt_delta(previous_bounded_receipts ? *previous_bounded_receipts : empty,
        bounded_receipts ? *bounded_receipts : empty);
    // Root receipt indexes are real children. A shared root reached through
    // both channels contributes two incoming edges but one canonical extent.
    (receipts ? *receipts : empty).visit_native_administration_delta(
        previous_receipts ? *previous_receipts : empty, sink(external), &work.native_administration);
    (bounded_receipts ? *bounded_receipts : empty).visit_native_administration_delta(
        previous_bounded_receipts ? *previous_bounded_receipts : empty, sink(external), &work.native_administration);
    flush_paid();

    // Membership deltas query only changed source owners in the canonical paid
    // index. Moving an unchanged membership to another arena reads its cached
    // origin intersection and does not remint one receipt per source member.
    struct Membership
    {
        GeometryNativeAdministration& result;
        owner_union_detail::OriginEdits& overlap;
    } membership{result, overlap};
    const NativePaidExtentGroup empty_group;
    (new_source_group ? *new_source_group : empty_group).visit_delta(source_group ? *source_group : empty_group,
        {&membership, [](void* raw, std::uint8_t kind, const auto& owner, std::size_t bytes, bool retain)
            {
                auto& context = *static_cast<Membership*>(raw);
                struct Owners
                {
                    Membership& context; std::size_t bytes; bool retain;
                    std::shared_ptr<const void> previous_origin;
                } owners{context, bytes, retain, {}};
                context.result.paid.visit_owner(kind, owner,
                    {&owners, [](void* raw, const NativePaidExtentCharge& value, std::size_t)
                        {
                            auto& context = *static_cast<Owners*>(raw);
                            const auto origin = value.origin();
                            if (context.previous_origin && origin.get() == context.previous_origin.get())
                            {
                                if (origin.owner_before(context.previous_origin) || context.previous_origin.owner_before(origin))
                                    throw GeometryError(GeometryErrorCode::InvalidInput, "Native receipt origin changed shared control.");
                                return;
                            }
                            context.previous_origin = origin;
                            if (value.bytes() != context.bytes)
                                throw GeometryError(GeometryErrorCode::InvalidInput,
                                    "Source membership changes canonical physical owner extent.");
                            context.context.overlap.change(value.origin(), value.bytes(), context.retain);
                        }});
            }});
    result.source_group_header = std::move(new_source_group_header);
    result.source_group = new_source_group;
    flush_origins();

    auto observed_owners = owners;
    auto observed_domain_paid = domain_paid;
    auto observed_paid = paid;
    // Financial edges borrow immutable receipt rows. The first batch borrows
    // the input roots. Subsequent batches also need the just-retired roots
    // through consumption at the start of the next round.
    NativeOwnerReceipts retired_domains, retired_paid;
    auto observed_origins = origins;
    auto observed_direct_origins = direct_origins;
    auto observed_publication_origins = publication_origins;
    auto observed_overlap = group_overlap;
    Capture auxiliary{append, paid_change, origin_change, scalar_change, same_control, append_financial, result, true};
    // Every noncurrent auxiliary retirement removes an actual old node. Fresh
    // nodes belong to the current domain and change only cached scalars. This
    // finite closure contains no predecessor-union ownership or full census.
    const auto rounds_bound = add(resident_bytes(), 16);
    for (std::size_t round = 0;; ++round)
    {
        if (round > rounds_bound)
            exhausted("Native paid administrative closure exceeds its old-owner retirement bound.");
        ++work.native_domain_batches;
        if (!pending.empty() || !pending_financial.empty())
        {
            GeometryOwnerGraphEditor editor(result.owners, 0, used, limit, nullptr, nullptr, state,
                &result.domain_paid);
            for (const auto& edge : pending)
                if (edge.retain)
                    editor.native_state_edges(edge.state, edge.references, true);
            for (const auto& edge : pending)
                if (!edge.retain)
                    editor.native_state_edges(edge.state, edge.references, false);
            for (const auto& edge : pending_financial)
                if (edge.retain) editor.financial_pin(*edge.charge, true);
            for (const auto& edge : pending_financial)
                if (!edge.retain) editor.financial_pin(*edge.charge, false);
            result.owners = editor.finish();
            auto next_receipts = editor.paid_receipts();
            receipt_delta(result.domain_paid, next_receipts);
            flush_paid();
            result.domain_paid = std::move(next_receipts);
            pending.clear();
            pending_financial.clear();
            retired_domains = {};
            retired_paid = {};
        }
        const auto prior_owners = result.owners;
        const auto prior_domains = result.domain_paid;
        const auto prior_paid = result.paid;
        const auto prior_origins = result.origins;
        const auto prior_direct_origins = result.direct_origins;
        const auto prior_publication_origins = result.publication_origins;
        const auto prior_overlap = result.group_overlap;
        prior_owners.visit_native_administration_delta(observed_owners, sink(auxiliary), &work.native_administration);
        prior_domains.visit_native_administration_delta(observed_domain_paid, sink(auxiliary), &work.native_administration);
        prior_paid.visit_native_administration_delta(observed_paid, sink(auxiliary), &work.native_administration);
        prior_origins.visit_native_administration_delta(observed_origins, sink(auxiliary), &work.native_administration);
        prior_direct_origins.visit_native_administration_delta(observed_direct_origins, sink(auxiliary), &work.native_administration);
        prior_publication_origins.visit_native_administration_delta(
            observed_publication_origins, sink(auxiliary), &work.native_administration);
        prior_overlap.visit_native_administration_delta(observed_overlap, sink(auxiliary), &work.native_administration);
        flush_origins();
        const auto unchanged = [](const auto& before, const auto& after)
        {
            bool changed = false;
            after.visit_native_administration_delta(before,
                {&changed, [](void* raw, const auto&, std::size_t, bool) { *static_cast<bool*>(raw) = true; }});
            return !changed;
        };
        const bool stable = pending.empty() && pending_financial.empty() &&
            unchanged(prior_origins, result.origins) && unchanged(prior_direct_origins, result.direct_origins) &&
            unchanged(prior_publication_origins, result.publication_origins) &&
            unchanged(prior_overlap, result.group_overlap);
        observed_owners = prior_owners;
        retired_domains = std::exchange(observed_domain_paid, prior_domains);
        retired_paid = std::exchange(observed_paid, prior_paid);
        observed_origins = prior_origins;
        observed_direct_origins = prior_direct_origins;
        observed_publication_origins = prior_publication_origins;
        observed_overlap = prior_overlap;
        if (stable)
            break;
    }
    // Domain scalar owners can also be explicit body owners. Credit their
    // physical intersection exactly once, independent of graph multiplicity.
    result.body_credit = body_credit;
    const auto covered = [](const OwnerManifestPair& roots, const OwnerRecord& row)
    {
        bool present = false;
        for (unsigned channel = 0; channel != 2; ++channel)
            if (roots.owners[channel].references(channel, row.key.kind, row.owner))
            {
                if (!roots.owners[channel].admits(channel, row.key.kind, row.owner, row.bytes))
                    throw GeometryError(GeometryErrorCode::InvalidInput, "Native domain/body owner changes size.");
                present = true;
            }
        return present;
    };
    const auto credit = [&](const OwnerRecord& row, bool retain)
    { scalar_change(result.body_credit, row.bytes, retain); };
    for (unsigned channel = 0; channel != 2; ++channel)
    {
        const auto change = [&](const OwnerRecord& row, bool retain)
        {
            if (!owners.references(0, row.key.kind, row.owner)) return;
            const auto& authority = retain ? body : previous_body;
            const unsigned canonical = authority.owners[0].references(0, row.key.kind, row.owner) ? 0 : 1;
            if (canonical == channel && covered(previous_body, row) != covered(body, row)) credit(row, retain);
        };
        struct Context { const decltype(change)& apply; bool retain; } add_context{change, true}, retire_context{change, false};
        const auto visit = [](void* raw, const OwnerRecord& row)
        { auto& context = *static_cast<Context*>(raw); context.apply(row, context.retain); };
        body.owners[channel].visit_added(previous_body.owners[channel], {&add_context, visit});
        previous_body.owners[channel].visit_added(body.owners[channel], {&retire_context, visit});
    }
    struct Domains
    {
        const decltype(covered)& covered_body;
        const decltype(credit)& credit_owner;
        const OwnerManifestPair& body;
        bool retain;
    } add_domains{covered, credit, body, true}, retire_domains{covered, credit, body, false};
    const auto domain = [](void* raw, const OwnerRecord& row)
    { auto& context = *static_cast<Domains*>(raw); if (context.covered_body(context.body, row)) context.credit_owner(row, context.retain); };
    result.owners.visit_added(owners, {&add_domains, domain});
    owners.visit_added(result.owners, {&retire_domains, domain});
    if (result.body_credit > result.owners.payload_bytes())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Native administration overlap exceeds its domain set.");
    return result;
}
void GeometryOwnerUnion::visit_administration(geometry_demand::MetadataSink metadata) const
{
    for (const auto& root : owners_.owners)
        root.visit_dependencies(metadata, {nullptr, [](void*, const auto&) {}});
    controls_.visit_metadata(metadata);
    for (const auto& receipts : paid_body_)
        receipts.visit_administration(metadata);
    native_administration_.visit(metadata);
}
void GeometryOwnerUnion::edit_sources(const GeometryOwnerRootBinding& binding, bool add)
{
    for (const auto& root : binding.payload_.owners)
    {
        if (!root.owner_count())
            continue;
        SourceRoot* found = nullptr;
        SourceRoot* free = nullptr;
        for (auto& source : sources_)
            if (!source.references)
            {
                if (!free)
                    free = &source;
            }
            else if (source.root.same_root(root))
                found = &source;
        if (add)
        {
            if (!found)
            {
                if (!free)
                    allocation_detail::exhausted("Owner union source inventory exhausted.");
                found = free;
                found->root = root;
            }
            found->references = allocation_detail::add(found->references, 1);
        }
        else
        {
            if (!found)
                throw GeometryError(GeometryErrorCode::InvalidInput, "Retired root index is outside its union.");
            if (!--found->references)
                found->root = {};
        }
    }
}
void GeometryOwnerUnion::validate(const GeometryOwnerRootBinding& binding) const
{
    if (!binding ||
        !owners_.owners[0].admits(0, GeometryOwnerGraphEditor::external_kind(), binding.header_, binding.header_bytes_))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Root header is outside its sealed owner union.");
    (void) binding.payload_.overlap_bytes();
    for (unsigned channel = 0; channel < 2; ++channel)
    {
        const auto& root = binding.payload_.owners[channel];
        if (!root.owner_count())
            continue;
        bool found = false;
        for (const auto& source : sources_)
            found = found || (source.references && source.root.same_root(root));
        if (!found || root.uniform_channel() != channel ||
            !owners_.owners[channel].admits(
                channel, GeometryOwnerGraphEditor::external_kind(), root.root_control(), root.root_metadata_bytes()))
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Root subset index differs from exact union controls.");
    }
}
void GeometryOwnerUnion::import(
    GeometryAllocationLedger& ordinary, GeometryAllocationLedger& bounded, std::size_t& used, std::size_t limit) const
{
    try
    {
        if (!owners_.owners[0].owner_count() && !owners_.owners[1].owner_count())
            return;
        GeometryOwnerAdmissionPair pair(ordinary, bounded);
        const bool fresh = !ordinary.impl_ || (!ordinary.impl_->imported && ordinary.impl_->owners.empty());
        GeometryOwnerImportAccess::import_pair(ordinary, bounded, owners_, used, limit);
        ordinary.impl_->set_native_grant(native_administration_.owners, used, limit);
        if (bounded.impl_)
            bounded.impl_->set_native_grant(native_administration_.owners, used, limit);
        if (owners_.owners[0].authenticates(ordinary.impl_->control_grant_authority))
            return;
        if (fresh)
        {
            auto admin = allocation_detail::add(
                allocation_detail::add(controls_.control_bytes(), controls_.index_admission_bytes()),
                native_administration_.resident_bytes());
            for (const auto& receipts : paid_body_)
                admin = allocation_detail::add(admin, receipts.metadata_admission_bytes());
            if (ordinary.impl_->prepaid_payload)
                ordinary.impl_->bookkeeping = allocation_detail::add(ordinary.impl_->bookkeeping, admin);
            else
                ordinary.impl_->account(admin, true, used, limit);
        }
        else
        {
            // Deliberately explicit uncommon fallback. It never runs for a
            // fresh pair or the exact previously staged authority head.
            struct Capture
            {
                GeometryAllocationLedger& ledger;
                std::size_t& used;
                std::size_t limit;
            } capture{ordinary, used, limit};
            controls_.visit_records({&capture, [](void* raw, const IndexControlRecord& row)
                {
                    auto& c = *static_cast<Capture*>(raw);
                    GeometryAllocationOwnerAccess::retain_verified_metadata(c.ledger, 0, row.owner, row.payload,
                        allocation_detail::add(row.payload, row.control), c.used, c.limit);
                }});
            ordinary.impl_->account(controls_.index_admission_bytes(), true, used, limit);
        }
        ordinary.impl_->control_grant = controls_;
        ordinary.impl_->control_grant_authority = owners_.owners[0].root_control();
        ordinary.impl_->last_counter = used;
    }
    catch (...)
    {
        GeometryAllocationOwnerAccess::poison(ordinary);
        GeometryAllocationOwnerAccess::poison(bounded);
        throw;
    }
}
GeometryOwnerUnion GeometryOwnerUnion::staged(const GeometryOwnerRootBinding* retire,
    const GeometryOwnerRootBinding* retain, GeometryAllocationLedger& ordinary_ledger,
    GeometryAllocationLedger& bounded_ledger, std::size_t& used, std::size_t limit,
    GeometryOwnerUnionWork* measured) const
{
    const std::array replacements{Replacement{retire, retain}};
    return staged(replacements, ordinary_ledger, bounded_ledger, used, limit, measured);
}
GeometryOwnerUnion GeometryOwnerUnion::staged(std::span<const Replacement> replacements,
    GeometryAllocationLedger& ordinary_ledger, GeometryAllocationLedger& bounded_ledger, std::size_t& used,
    std::size_t limit, GeometryOwnerUnionWork* measured) const
{
    try
    {
        GeometryOwnerAdmissionPair pair(ordinary_ledger, bounded_ledger);
        if (generation_ == std::numeric_limits<std::uint64_t>::max())
            allocation_detail::exhausted("Owner union generation overflows.");
        for (const auto& replacement : replacements)
            if (replacement.retire && *replacement.retire)
                validate(*replacement.retire);
        import(ordinary_ledger, bounded_ledger, used, limit);
        GeometryOwnerUnion result = *this;
        for (const auto& replacement : replacements)
        {
            const auto* retire = replacement.retire;
            if (!retire || !*retire) continue;
            result.edit_sources(*retire, false);
            if (retire->group_)
            {
                if (result.source_group_ != retire->group_ ||
                    result.source_group_header_.get() != retire->header_.get() ||
                    result.source_group_header_.owner_before(retire->header_) ||
                    retire->header_.owner_before(result.source_group_header_))
                    throw GeometryError(GeometryErrorCode::InvalidInput, "Retired source group is outside its exact root.");
                result.source_group_header_.reset();
                result.source_group_ = nullptr;
            }
        }
        for (const auto& replacement : replacements)
        {
            const auto* retain = replacement.retain;
            if (!retain || !*retain) continue;
            result.edit_sources(*retain, true);
            if (retain->group_)
            {
                if (result.source_group_)
                    throw GeometryError(GeometryErrorCode::InvalidInput, "Owner union has two unrelated source groups.");
                result.source_group_header_ = retain->header_;
                result.source_group_ = retain->group_;
            }
        }
        GeometryOwnerUnionWork local;
        auto& work = measured ? *measured : local;
        GeometryOwnerGraphEditor ordinary(owners_.owners[0], 0, used, limit, &ordinary_ledger, &work.channels[0]);
        GeometryOwnerGraphEditor bounded(owners_.owners[1], 1, used, limit, &bounded_ledger, &work.channels[1]);
        auto overlap = owners_.overlap_bytes();
        ordinary.pair_with(bounded, overlap);
        struct Allowance
        {
            std::size_t& used;
            std::size_t limit;
            GeometryAllocationLedger& ledger;
        } allowance{used, limit, ordinary_ledger};
        const geometry_demand::Admission admission{&allowance,
            [](void* raw, std::size_t bytes)
            {
                auto& a = *static_cast<Allowance*>(raw);
                if (a.used > a.limit || bytes > a.limit - a.used)
                    allocation_detail::exhausted("Owner union administration exceeds quota.");
                if (const auto state = GeometryAllocationOwnerAccess::native_state(a.ledger))
                    state->record_ledger_work(bytes);
                a.used += bytes;
            },
            [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
            {
                auto& a = *static_cast<Allowance*>(raw);
                GeometryAllocationOwnerAccess::retain_created_metadata(
                    a.ledger, 0, owner, payload, admitted, a.used, a.limit);
            },
            GeometryAllocationOwnerAccess::native_state(ordinary_ledger),
            [](void* raw, const std::shared_ptr<const void>& owner, std::size_t bytes)
            {
                auto& a = *static_cast<Allowance*>(raw);
                GeometryAllocationOwnerAccess::register_native_metadata(a.ledger, 0, owner, bytes);
            }};
        // Replacements traverse source-pruned receipt deltas; unchanged owners
        // retain their exact origin/lease facts without reissuing N rows.
        const NativeOwnerReceipts empty_receipts;
        const auto receipts = [&](const GeometryOwnerRootBinding* binding) -> const NativeOwnerReceipts&
        { return binding && *binding && binding->paid_ ? *binding->paid_ : empty_receipts; };
        geometry_demand::ReceiptEdits receipt_edits(admission);
        struct PaidEdits
        {
            geometry_demand::ReceiptEdits& edits;
            bool retain;
        } paid_add{receipt_edits, true}, paid_remove{receipt_edits, false};
        const auto paid_edit = [](void* raw, const NativePaidExtentCharge& value, std::size_t)
        {
            auto& context = *static_cast<PaidEdits*>(raw);
            if (!value.source_only() && (!context.retain || value.valid()))
                context.edits.append_borrowed(value, context.retain ? NativeOwnerReceipts::Action::Retain :
                    NativeOwnerReceipts::Action::RetireIfRecorded);
        };
        for (const auto& replacement : replacements)
            receipts(replacement.retain).visit_added(
                receipts(replacement.retire), {&paid_add, paid_edit}, &work.native_administration);
        for (const auto& replacement : replacements)
            receipts(replacement.retire).visit_added(
                receipts(replacement.retain), {&paid_remove, paid_edit}, &work.native_administration);
        owner_union_detail::ControlEdits controls{controls_, admission};
        const auto refs = [&](GeometryOwnerGraphEditor& editor, const geometry_demand::OwnerRecord& row)
        {
            auto key = row;
            key.key.channel = &editor == &ordinary ? 0 : 1;
            return editor.references(key);
        };
        const auto absent = [&](const geometry_demand::OwnerRecord& row)
        { return !refs(ordinary, row) && !refs(bounded, row); };
        const auto edit = [&](const GeometryOwnerRootBinding& binding, bool add)
        {
            (void) ordinary.external(binding.header_, binding.header_bytes_, add);
            for (unsigned channel = 0; channel < 2; ++channel)
            {
                struct Context
                {
                    GeometryOwnerGraphEditor& editor;
                    GeometryOwnerGraphEditor& ordinary;
                    GeometryOwnerGraphEditor& bounded;
                    owner_union_detail::ControlEdits& controls;
                    GeometryAllocationLedger& ledger;
                    geometry_demand::ReceiptEdits& paid;
                    std::size_t& used;
                    std::size_t limit;
                    unsigned channel;
                    bool add;
                } context{channel ? bounded : ordinary, ordinary, bounded, controls,
                    channel ? bounded_ledger : ordinary_ledger, receipt_edits, used, limit, channel, add};
                const auto metadata = geometry_demand::MetadataSink{&context,
                        [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload,
                            std::size_t admitted)
                        {
                            auto& c = *static_cast<Context*>(raw);
                            geometry_demand::OwnerRecord key{
                                {std::uint8_t(c.channel), GeometryOwnerGraphEditor::external_kind(), owner.get()},
                                owner, payload, 1};
                            auto other = key;
                            other.key.channel = std::uint8_t(!c.channel);
                            const bool known = c.controls.known(owner, payload);
                            if (c.add && !known &&
                                (c.editor.references(key) || (c.channel ? c.ordinary : c.bounded).references(other)))
                                throw GeometryError(GeometryErrorCode::InvalidTopology,
                                    "Root payload may not retain another administrative manifest value.");
                            if (c.add)
                                c.controls.classify(owner, payload, admitted);
                            GeometryAllocationOwnerAccess::retain_verified_metadata(
                                c.ledger, std::uint8_t(c.channel), owner, payload, admitted, c.used, c.limit);
                            const bool transition = c.editor.external(owner, payload, c.add);
                            if (transition)
                            {
                                const auto charge = NativePaidExtentCharge::metadata(owner, payload);
                                if (charge.valid())
                                    c.paid.append(charge, c.add ? NativeOwnerReceipts::Action::Retain :
                                        NativeOwnerReceipts::Action::Retire);
                            }
                            if (!c.add && transition && !c.editor.references(key) &&
                                !(c.channel ? c.ordinary : c.bounded).references(other))
                                c.controls.retire(owner, payload);
                            return transition;
                        },
                        [](void* raw, const std::shared_ptr<const void>& owner, std::size_t bytes)
                        {
                            auto& c = *static_cast<Context*>(raw);
                            return c.editor.external(owner, bytes, c.add);
                        }}.with_financial(context.editor, add);
                if (!channel && binding.paid_)
                    binding.paid_->visit_administration(metadata);
                binding.payload_.owners[channel].visit_dependencies(metadata,
                    {&context, [](void* raw, const geometry_demand::OwnerRecord& row)
                        {
                            auto& c = *static_cast<Context*>(raw);
                            if (c.add && c.controls.known(row.owner, row.bytes))
                                throw GeometryError(GeometryErrorCode::InvalidTopology,
                                    "Root body contains another root's administrative index.");
                            const bool transition = c.editor.owner(row, c.add);
                            if (!c.add && transition)
                            {
                                auto other = row;
                                other.key.channel = std::uint8_t(!c.channel);
                                if (!(c.channel ? c.ordinary : c.bounded).references(other))
                                    c.controls.retire(row.owner, row.bytes);
                            }
                        }});
            }
        };
        for (const auto& replacement : replacements)
            if (replacement.retain && *replacement.retain)
                edit(*replacement.retain, true);
        for (const auto& replacement : replacements)
            if (replacement.retire && *replacement.retire)
                edit(*replacement.retire, false);
        result.paid_body_[0] = result.paid_body_[0].edited(receipt_edits.rows(), admission);
        receipt_edits.clear();
        result.owners_ = OwnerManifestPair::sealed({ordinary.finish(), bounded.finish()}, overlap);
        result.controls_ = controls.finish(&work.controls);
        result.native_administration_ = native_administration_.updated(owners_, result.owners_, controls_,
            result.controls_, used, limit, GeometryAllocationOwnerAccess::native_state(ordinary_ledger), work,
            &paid_body_[0], &result.paid_body_[0], &paid_body_[1], &result.paid_body_[1],
            result.source_group_header_, result.source_group_);
        result.resident_bytes_ = allocation_detail::add(
            allocation_detail::add(result.owners_.payload_bytes(), result.owners_.index_admission_bytes()),
            allocation_detail::add(result.controls_.control_bytes(), result.controls_.index_admission_bytes()));
        for (const auto& receipts : result.paid_body_)
            result.resident_bytes_ = allocation_detail::add(result.resident_bytes_, receipts.metadata_admission_bytes());
        result.resident_bytes_ =
            allocation_detail::add(result.resident_bytes_, result.native_administration_.resident_bytes());
        ++result.generation_;
        work.imported_rows += GeometryOwnerImportAccess::import_derived(
            ordinary_ledger, owners_.owners[0], result.owners_.owners[0], used, limit);
        if (owners_.owners[1].owner_count() || result.owners_.owners[1].owner_count())
            work.imported_rows += GeometryOwnerImportAccess::import_derived(
                bounded_ledger, owners_.owners[1], result.owners_.owners[1], used, limit);
        ordinary_ledger.impl_->set_native_grant(result.native_administration_.owners, used, limit);
        if (bounded_ledger.impl_)
            bounded_ledger.impl_->set_native_grant(result.native_administration_.owners, used, limit);
        ordinary_ledger.impl_->control_grant = result.controls_;
        ordinary_ledger.impl_->control_grant_authority = result.owners_.owners[0].root_control();
        return result;
    }
    catch (...)
    {
        GeometryAllocationOwnerAccess::poison(ordinary_ledger);
        GeometryAllocationOwnerAccess::poison(bounded_ledger);
        throw;
    }
}
std::size_t GeometryOwnerUnion::administration_byte_bound(std::size_t edges, std::size_t changed_keys,
    std::size_t input_index_controls, std::size_t new_input_owners, std::size_t anticipated_additions,
    std::optional<std::size_t> anticipated_typed_additions) const
{
    using namespace allocation_detail;
    using namespace geometry_demand;
    // The operation's prefix has already imported this maintained union. This
    // envelope covers the Ordinary edit, not that prefix or incoming payload.
    const auto& accepted = owners_.owners[0];
    const auto growth = add(anticipated_additions, changed_keys);
    const auto scale = accepted.edit_height_scale_bound(growth, anticipated_typed_additions);
    // K bounds every touched identity, including a key whose final reference
    // delta cancels. Overlay/final scratch has one row per such identity;
    // this adapter never enqueues geometry children in the pending queue.
    const auto frontier = std::min(edges, changed_keys);
    const auto typed_keys = anticipated_typed_additions ?
        std::min(changed_keys, add(accepted.typed_owner_count(), *anticipated_typed_additions)) : 0;
    auto bytes = product(GeometryOwnerGraphEditor::edit_byte_bound(accepted, frontier, changed_keys, typed_keys), scale);
    const auto owner_row =
        sizeof(typename decltype(GeometryAllocationLedger::Impl::owners)::value_type) + 4 * sizeof(void*);
    const auto credit_row =
        sizeof(typename decltype(GeometryAllocationLedger::Impl::metadata_credits)::value_type) + 4 * sizeof(void*);
    bytes = add(bytes, product(new_input_owners, owner_row));
    // An incoming index owner may need a verified credit row and its control
    // allowance even when its exclusive payload was covered by the caller's A
    // envelope. Do not subtract hypothetical prior credits in this upper bound.
    bytes = add(bytes, product(input_index_controls, add(credit_row, control_bytes)));
    const auto excess = input_index_controls > leaf_capacity ? input_index_controls - leaf_capacity : 0;
    bytes =
        add(bytes, product(excess, sizeof(typename decltype(owner_union_detail::ControlEdits::overflow)::value_type) +
                                       4 * sizeof(void*)));
    if (excess)
        bytes = add(bytes, product(input_index_controls, sizeof(IndexControlRecord)));
    // Only input metadata controls change this table; geometry body rows cannot
    // alias administrative index owners. Earlier additions may enlarge it before
    // a future retirement. Every new table node/State creates one owner row and
    // one prepaid credit row; its own control is in edit_admission_bound already.
    bytes = add(bytes, controls_.edit_admission_bound(input_index_controls, anticipated_additions));
    bytes = add(bytes, product(controls_.edit_metadata_owner_bound(input_index_controls, anticipated_additions),
                           add(owner_row, credit_row)));
    // At most one new derived head per channel and one authority/subset pair
    // per channel. Certificate storage is fixed in Impl; only their weak shared
    // control lifetimes can require new retained allowance.
    constexpr std::size_t derived_heads = 2, certificate_controls = 2 * 2;
    return add(bytes, product(derived_heads + certificate_controls, control_bytes));
}
void GeometryOwnerUnion::certify_subset(const GeometryOwnerRootBinding& binding, GeometryAllocationLedger& ordinary,
    GeometryAllocationLedger& bounded, std::size_t& used, std::size_t limit) const
{
    try
    {
        validate(binding);
        for (unsigned channel = 0; channel < 2; ++channel)
            if (binding.payload_.owners[channel].owner_count())
                GeometryOwnerImportAccess::certify_subset(channel ? bounded : ordinary, owners_.owners[channel],
                    binding.payload_.owners[channel], generation_, used, limit);
    }
    catch (...)
    {
        GeometryAllocationOwnerAccess::poison(ordinary);
        GeometryAllocationOwnerAccess::poison(bounded);
        throw;
    }
}
void GeometryOwnerUnion::import_subset(const GeometryOwnerRootBinding& binding, GeometryAllocationLedger& ordinary,
    GeometryAllocationLedger& bounded, std::size_t& used, std::size_t limit) const
{
    try
    {
        validate(binding);
        GeometryOwnerAdmissionPair pair(ordinary, bounded);
        GeometryOwnerImportAccess::import_pair(ordinary, bounded, binding.payload_, used, limit);
        GeometryAllocationOwnerAccess::admit_graph_owner(ordinary, 0, GeometryOwnerGraphEditor::external_kind(),
            binding.header_, binding.header_bytes_, used, limit);
    }
    catch (...)
    {
        GeometryAllocationOwnerAccess::poison(ordinary);
        GeometryAllocationOwnerAccess::poison(bounded);
        throw;
    }
}
} // namespace SimpleFluid::PLIC::details
