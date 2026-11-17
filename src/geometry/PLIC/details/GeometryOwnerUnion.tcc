#pragma once
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerUnion.hh"
#include <limits>
#include <map>
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
GeometryOwnerRootBinding::GeometryOwnerRootBinding(
    std::shared_ptr<const void> header, std::size_t bytes, OwnerManifestPair payload)
    : header_(std::move(header)), header_bytes_(bytes), payload_(std::move(payload))
{
    const std::shared_ptr<const void> empty;
    if (!header_ || (!header_.owner_before(empty) && !empty.owner_before(header_)))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Root binding needs an actual immutable header control.");
    if (!payload_.sealed_ && (!payload_.owners[0].owner_count() || !payload_.owners[1].owner_count()))
        payload_ = OwnerManifestPair::sealed(std::move(payload_.owners), 0);
    payload_.require_seal();
}
namespace owner_union_detail
{
struct ControlEdits
{
    const GeometryIndexControls& accepted;
    geometry_demand::Admission admission;
    std::array<IndexControlRecord, geometry_demand::leaf_capacity> inline_rows{};
    std::size_t size = 0;
    std::map<const void*, IndexControlRecord, std::less<const void*>> overflow;
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
            return row->control != 0;
        }
        return accepted.contains(owner, payload, geometry_demand::control_bytes);
    }
    void put(const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t control)
    {
        if (auto* row = changed(owner))
        {
            row->control = control;
            return;
        }
        if (size < inline_rows.size())
            inline_rows[size++] = {owner, payload, control};
        else
        {
            admission.charge(1, sizeof(typename decltype(overflow)::value_type) + 4 * sizeof(void*));
            overflow.emplace(owner.get(), IndexControlRecord{owner, payload, control});
        }
    }
    void classify(const std::shared_ptr<const void>& owner, std::size_t payload)
    {
        if (!known(owner, payload))
            put(owner, payload, geometry_demand::control_bytes);
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
        std::vector<IndexControlRecord> rows;
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
void GeometryOwnerUnion::visit_administration(geometry_demand::MetadataSink metadata) const
{
    for (const auto& root : owners_.owners)
        root.visit_dependencies(metadata, {nullptr, [](void*, const auto&) {}});
    controls_.visit_metadata(metadata);
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
        if (owners_.owners[0].authenticates(ordinary.impl_->control_grant_authority))
            return;
        if (fresh)
        {
            const auto admin = allocation_detail::add(controls_.control_bytes(), controls_.index_admission_bytes());
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
    try
    {
        GeometryOwnerAdmissionPair pair(ordinary_ledger, bounded_ledger);
        if (generation_ == std::numeric_limits<std::uint64_t>::max())
            allocation_detail::exhausted("Owner union generation overflows.");
        import(ordinary_ledger, bounded_ledger, used, limit);
        if (retire && *retire)
            validate(*retire);
        GeometryOwnerUnion result = *this;
        if (retire && *retire)
            result.edit_sources(*retire, false);
        if (retain && *retain)
            result.edit_sources(*retain, true);
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
                a.used += bytes;
            },
            [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
            {
                auto& a = *static_cast<Allowance*>(raw);
                GeometryAllocationOwnerAccess::retain_manifest_metadata(
                    a.ledger, 0, owner, payload, admitted, a.used, a.limit, true);
            }};
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
                    std::size_t& used;
                    std::size_t limit;
                    unsigned channel;
                    bool add;
                } context{channel ? bounded : ordinary, ordinary, bounded, controls,
                    channel ? bounded_ledger : ordinary_ledger, used, limit, channel, add};
                binding.payload_.owners[channel].visit_dependencies(
                    {&context,
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
                                c.controls.classify(owner, payload);
                            GeometryAllocationOwnerAccess::retain_verified_metadata(
                                c.ledger, std::uint8_t(c.channel), owner, payload, admitted, c.used, c.limit);
                            const bool transition = c.editor.external(owner, payload, c.add);
                            if (!c.add && transition && !c.editor.references(key) &&
                                !(c.channel ? c.ordinary : c.bounded).references(other))
                                c.controls.retire(owner, payload);
                            return transition;
                        }},
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
        if (retain && *retain)
            edit(*retain, true);
        if (retire && *retire)
            edit(*retire, false);
        result.owners_ = OwnerManifestPair::sealed({ordinary.finish(), bounded.finish()}, overlap);
        result.controls_ = controls.finish(&work.controls);
        result.resident_bytes_ = allocation_detail::add(
            allocation_detail::add(result.owners_.payload_bytes(), result.owners_.index_admission_bytes()),
            allocation_detail::add(result.controls_.control_bytes(), result.controls_.index_admission_bytes()));
        ++result.generation_;
        work.imported_rows += GeometryOwnerImportAccess::import_derived(
            ordinary_ledger, owners_.owners[0], result.owners_.owners[0], used, limit);
        if (owners_.owners[1].owner_count() || result.owners_.owners[1].owner_count())
            work.imported_rows += GeometryOwnerImportAccess::import_derived(
                bounded_ledger, owners_.owners[1], result.owners_.owners[1], used, limit);
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
    std::size_t input_index_controls, std::size_t new_input_owners, std::size_t anticipated_additions) const
{
    using namespace allocation_detail;
    using namespace geometry_demand;
    // The operation's prefix has already imported this maintained union. This
    // envelope covers the Ordinary edit, not that prefix or incoming payload.
    const auto& accepted = owners_.owners[0];
    const auto growth = add(anticipated_additions, changed_keys);
    const auto scale = accepted.edit_height_scale_bound(growth);
    // K bounds every touched identity, including a key whose final reference
    // delta cancels. Overlay/final scratch has one row per such identity;
    // this adapter never enqueues geometry children in the pending queue.
    const auto frontier = std::min(edges, changed_keys);
    auto bytes = product(GeometryOwnerGraphEditor::edit_byte_bound(accepted, frontier, changed_keys), scale);
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
