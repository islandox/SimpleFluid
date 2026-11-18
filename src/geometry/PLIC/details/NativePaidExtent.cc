#include "geometry/PLIC/details/NativePaidExtent.hh"
#include "geometry/PLIC/details/GeometryDemandPages.tcc"
#include "geometry/PLIC/details/NativeGeometryAccounting.hh"
#include "geometry/details/NativeAllocationGraph.hh"
namespace SimpleFluid::PLIC::details
{
namespace
{
[[noreturn]] void invalid_paid(const char* message)
{
    throw GeometryError(GeometryErrorCode::InvalidInput, message);
}
bool same_paid(const std::weak_ptr<const void>& a, const std::shared_ptr<const void>& b)
{
    return b && a.lock().get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
}
} // namespace
NativePaidExtentCharge NativePaidExtentCharge::made(std::uint8_t kind, const std::shared_ptr<const void>& owner,
    std::size_t bytes, const std::shared_ptr<const geometry_detail::NativeAllocationState>& state)
{
    NativePaidExtentCharge result;
    if (!owner || !bytes || !state || !state->mirror.owner)
        return result;
    (void) geometry_detail::NativeAllocationGraphAccess::validate_state(state);
    result.address_ = owner.get();
    result.owner_ = owner;
    result.origin_address_ = state->mirror.owner.get();
    result.origin_ = state->mirror.owner;
    result.state_address_ = state.get();
    result.bytes_ = bytes;
    result.kind_ = kind;
    result.proof_ = 1;
    result.publication_h_ = state->mirror.owner_proof.publication_requests();
    return result;
}
bool NativePaidExtentCharge::valid() const
{
    const auto owner = owner_.lock(), origin = origin_.lock();
    if (!proof_ || !bytes_ || !owner || owner.get() != address_ || !origin || origin.get() != origin_address_)
        return false;
    if (proof_ == 2)
    {
        const auto value = profile_.lock();
        if (!value || value->state().get() != state_address_ || value->bytes() != bytes_)
            return false;
    }
    if (proof_ == 3)
    {
        const auto value = storage_.lock();
        if (!value || value->revision() != storage_revision_ || value->exclusive_bytes() != storage_bytes_)
            return false;
    }
    if (proof_ == 4 || proof_ == 6 || proof_ == 7)
    {
        const auto financial = borrowed_financial_ ? financial_observation_.lock() : lease_pin_;
        if (!financial || financial.get() != financial_address_ || !financial_projector_ || !financial_inspector_ ||
            !financial_validator_ || charge_snapshot_ < bytes_ ||
            !financial_validator_(financial, owner, origin, kind_, bytes_, charge_snapshot_, revision_))
            return false;
    }
    return true;
}
bool NativePaidExtentCharge::matches(
    std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    return valid() && kind == kind_ && bytes == bytes_ && same_paid(owner_, owner);
}
NativePaidExtentCharge NativePaidExtentCharge::fixed(
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& value)
{
    if (!value || !value->authentic_owner(value))
        invalid_paid("Paid fixed extent has a foreign actual shared control.");
    auto result = made(GeometryOwnerGraphEditor::external_kind(), value, value->fixed_allocation_bytes(), value->state());
    result.fixed_owner_ = FixedOwner::Envelope;
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::state(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& value)
{
    (void) geometry_detail::NativeAllocationGraphAccess::validate_state(value);
    auto result = made(GeometryOwnerGraphEditor::external_kind(), value, value->allocation_admission_bytes, value);
    result.fixed_owner_ = FixedOwner::State;
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::counter(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& value)
{
    const auto bytes = geometry_detail::NativeAllocationGraphAccess::validate_state(value);
    auto result = made(GeometryOwnerGraphEditor::external_kind(), value->counter_owner, bytes, value);
    result.fixed_owner_ = FixedOwner::Counter;
    result.charge_snapshot_ = value->mirror.owner_proof.bytes();
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::mirror(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& value)
{
    (void) geometry_detail::NativeAllocationGraphAccess::validate_state(value);
    auto result = made(
        GeometryOwnerGraphEditor::external_kind(), value->mirror.owner, value->mirror.owner_proof.bytes(), value);
    // The detached Budget keeper was already in the source arena when its
    // publication callbacks were installed.
    result.publication_h_ = false;
    result.fixed_owner_ = FixedOwner::Mirror;
    return result;
}
NativeWeakOwnerObservation NativePaidExtentCharge::detached_source_observation() const
{
    const auto physical = owner_.lock();
    // Retain the actual origin until both validation and witness issuance
    // have completed, including a counter's real control-resident keeper.
    const auto arena = origin_.lock();
    if (!physical || !arena || !valid()) return {};
    if (physical_fixed_.matches_live(physical)) return physical_fixed_;
    switch (fixed_owner_)
    {
    case FixedOwner::Envelope:
        return NativeWeakOwnerObservation::envelope_control(
            std::static_pointer_cast<const geometry_detail::NativeAllocationEnvelope>(physical));
    case FixedOwner::State:
        return NativeWeakOwnerObservation::state_control(
            std::static_pointer_cast<const geometry_detail::NativeAllocationState>(physical));
    case FixedOwner::Counter:
        return NativeWeakOwnerObservation::scalar_control(physical, bytes_, arena, charge_snapshot_);
    case FixedOwner::Mirror:
        return NativeWeakOwnerObservation::scalar_control(physical, bytes_, {}, 0);
    case FixedOwner::None:
        return {};
    }
    return {};
}
NativePaidExtentCharge NativePaidExtentCharge::storage(const geometry_detail::NativeConstructorStorageOwner& value)
{
    if (!value.owner || !value.envelope || !value.envelope->authentic_owner(value.envelope) ||
        !value.envelope->matches(value.owner) || value.owner->admission().state() != value.envelope->state())
        invalid_paid("Paid storage extent has a foreign actual receipt.");
    auto result = made(GeometryOwnerGraphEditor::external_kind(), value.owner, value.owner->exclusive_bytes(),
        value.envelope->state());
    result.storage_ = value.owner;
    result.physical_fixed_ = NativeWeakOwnerObservation::native(value.owner, value.envelope);
    result.storage_revision_ = value.owner->revision();
    result.storage_bytes_ = value.owner->exclusive_bytes();
    if (result.proof_)
        result.proof_ = 3;
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::metadata(const std::shared_ptr<const void>& owner, std::size_t bytes)
{
    const auto* proof = std::get_deleter<geometry_demand::NativeMetadataDelete>(owner);
    if (!proof)
        return {};
    if (proof->object != owner.get() || proof->extent_bytes(owner.get(), proof->control_bytes) != bytes)
        invalid_paid("Paid native metadata changes actual object/control or immutable extent.");
    return made(GeometryOwnerGraphEditor::external_kind(), owner, bytes, proof->state);
}
NativePaidExtentCharge NativePaidExtentCharge::buffer(const NativeActualBufferProof& proof)
{
    if (!proof.owner_ || !proof.envelope_ || !proof.envelope_->authentic_owner(proof.envelope_) ||
        !proof.envelope_->matches(proof.owner_) || proof.envelope_->state() != proof.state_ || !proof.extent_.address ||
        !proof.extent_.bytes)
        invalid_paid("Paid native buffer has a foreign typed allocation proof.");
    const std::shared_ptr<const void> alias(proof.owner_, proof.extent_.address);
    auto result = made(GeometryOwnerGraphEditor::external_kind(), alias, proof.extent_.bytes, proof.state_);
    result.physical_fixed_ = NativeWeakOwnerObservation::native_alias(alias, proof.envelope_);
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::retained_profile(const std::shared_ptr<const void>& owner,
    const std::shared_ptr<const geometry_detail::NativeRetainedExtent>& extent,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope)
{
    if (!extent || !extent->bytes() || !envelope || !envelope->shares_control(extent) ||
        envelope->state() != extent->state())
        invalid_paid("Paid retained profile lacks its exact holder and lifetime extent.");
    auto result = made(GeometryOwnerGraphEditor::external_kind(), owner, extent->bytes(), extent->state());
    result.physical_fixed_ = NativeWeakOwnerObservation::native_alias(owner, envelope);
    result.profile_ = extent;
    if (result.proof_)
        result.proof_ = 2;
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::source_profile(std::uint8_t kind,
    const std::shared_ptr<const void>& owner, std::size_t bytes, const std::shared_ptr<const void>& arena,
    const std::shared_ptr<const void>& lease, std::size_t charge_snapshot, std::uint64_t revision,
    FinancialProjector projector, FinancialInspector inspector, FinancialValidator validator)
{
    const std::shared_ptr<const void> empty;
    if (!owner || !arena || !lease || !projector || !inspector || !validator || !bytes || charge_snapshot < bytes ||
        (!owner.owner_before(empty) && !empty.owner_before(owner)) ||
        (!arena.owner_before(empty) && !empty.owner_before(arena)))
        invalid_paid("Source paid profile lacks exact detached lease authority.");
    NativePaidExtentCharge result;
    result.address_ = owner.get();
    result.owner_ = owner;
    result.origin_address_ = arena.get();
    result.origin_ = arena;
    result.lease_pin_ = lease;
    result.financial_address_ = lease.get();
    result.direct_h_ = false;
    result.financial_projector_ = projector;
    result.financial_inspector_ = inspector;
    result.financial_validator_ = validator;
    result.bytes_ = bytes;
    result.charge_snapshot_ = charge_snapshot;
    result.revision_ = revision;
    result.kind_ = kind;
    result.proof_ = 4;
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::detached_subset(std::uint8_t kind,
    const std::shared_ptr<const void>& owner, std::size_t bytes, const std::shared_ptr<const void>& arena,
    const std::shared_ptr<const void>& lease, std::size_t snapshot, std::uint64_t revision,
    FinancialProjector projector, FinancialInspector inspector, FinancialValidator validator)
{
    auto result = source_profile(kind, owner, bytes, arena, lease, snapshot, revision, projector, inspector, validator);
    result.proof_ = 6;
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::intrinsic(std::uint8_t kind,
    const std::shared_ptr<const void>& owner, std::size_t bytes, const std::shared_ptr<const void>& arena,
    const std::shared_ptr<const void>& financial, std::size_t snapshot, std::uint64_t revision,
    FinancialProjector projector, FinancialInspector inspector, FinancialValidator validator,
    const NativeWeakOwnerObservation& physical_observation, const NativeWeakOwnerObservation& financial_observation)
{
    auto result = detached_subset(kind, owner, bytes, arena, financial, snapshot, revision, projector, inspector, validator);
    result.proof_ = 7;
    result.direct_h_ = true;
    result = result.with_weak_observations(physical_observation, financial_observation);
    const auto* contact = physical_observation.contact_control_group();
    if ((contact && same_paid(contact->owner, financial)) ||
        (financial_observation && same_paid(owner, financial)))
        return result.as_borrowed_financial();
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::contact_fixed(const NativeWeakOwnerObservation& observation)
{
    const auto* group = observation.contact_control_group();
    if (!group)
        invalid_paid("Paid contact control extent has no concrete fixed witness.");
    NativePaidExtentCharge result;
    result.address_ = group->owner.get();
    result.owner_ = group->owner;
    result.origin_address_ = group->keeper.get();
    result.origin_ = group->keeper;
    result.bytes_ = group->bytes;
    result.kind_ = GeometryOwnerGraphEditor::external_kind();
    result.proof_ = 8;
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::with_weak_observations(
    const NativeWeakOwnerObservation& physical, const NativeWeakOwnerObservation& financial) const
{
    if ((physical && !physical.matches_live(owner())) || (financial && !financial.matches_live(lease_pin_)))
        invalid_paid("Native paid receipt has a foreign physical or financial fixed observation.");
    auto result = *this;
    result.physical_fixed_ = physical;
    result.financial_fixed_ = financial;
    return result;
}
NativePaidExtentCharge NativePaidExtentCharge::as_borrowed_financial() const
{
    auto result = *this;
    if (result.lease_pin_)
    {
        // A contact group's scalar financial token is already represented by
        // its physical observation. Native financial controls carry their own.
        const auto* group = result.physical_fixed_.contact_control_group();
        const bool represented = group && same_paid(group->owner, result.lease_pin_);
        if (!represented && !result.financial_fixed_.matches_live(result.lease_pin_))
            invalid_paid("Borrowed native financial receipt lacks its exact fixed witness.");
        result.financial_observation_ = result.lease_pin_;
        result.lease_pin_.reset();
        result.borrowed_financial_ = true;
    }
    return result;
}
bool NativePaidExtentCharge::same_financial_control(const NativePaidExtentCharge& other) const noexcept
{
    const std::weak_ptr<const void> a = borrowed_financial_ ? financial_observation_ : lease_pin_;
    const std::weak_ptr<const void> b = other.borrowed_financial_ ? other.financial_observation_ : other.lease_pin_;
    return financial_address_ == other.financial_address_ && !a.owner_before(b) && !b.owner_before(a);
}
bool NativePaidExtentCharge::same_owner_fact(const NativePaidExtentCharge& other) const noexcept
{
    return kind_ == other.kind_ && address_ == other.address_ && !owner_.owner_before(other.owner_) &&
           !other.owner_.owner_before(owner_);
}
bool NativePaidExtentCharge::same_origin_fact(const NativePaidExtentCharge& other) const noexcept
{
    return origin_address_ == other.origin_address_ && !origin_.owner_before(other.origin_) &&
           !other.origin_.owner_before(origin_);
}
bool NativePaidExtentCharge::same_receipt_fact(const NativePaidExtentCharge& other) const noexcept
{
    return same_owner_fact(other) && same_origin_fact(other) && bytes_ == other.bytes_ &&
           same_financial_control(other) && physical_fixed_.same_observation(other.physical_fixed_) &&
           financial_fixed_.same_observation(other.financial_fixed_);
}
void NativePaidExtentCharge::project_financial(const geometry_detail::NativeConstructorOwnerSink& sink) const
{
    if (!lease_pin_) return;
    if (!financial_projector_ || !financial_inspector_ || !financial_validator_)
        invalid_paid("Native financial pin lacks its closed typed projection.");
    financial_projector_(lease_pin_, sink);
}
void NativePaidExtentCharge::inspect_financial(const NativeFinancialInspectionSink& sink) const
{
    if (!lease_pin_) return;
    if (!financial_projector_ || !financial_inspector_ || !financial_validator_)
        invalid_paid("Native financial pin lacks its independent typed inspection.");
    financial_inspector_(lease_pin_, sink);
}
void geometry_demand::MetadataSink::financial(const NativePaidExtentCharge& value) const
{
    if (!value.has_lifetime_children()) return;
    if (financial_pin) financial_pin(context, value);
    else if (financial_graph) financial_graph->financial_pin(value, financial_retain);
    else invalid_paid("Financial receipt leaf lacks its dedicated body projection adapter.");
}
bool NativePaidExtentGroup::valid() const noexcept
{
    const std::shared_ptr<const void> empty;
    return membership_ && coverage_ && origin_ && contains_ && same_ && delta_ &&
           (origin_.owner_before(empty) || empty.owner_before(origin_));
}
bool NativePaidExtentGroup::contains(
    std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    if (!valid())
        invalid_paid("Source group lacks its closed membership/coverage authority.");
    return contains_(membership_.get(), kind, owner, bytes);
}
bool NativePaidExtentGroup::same_membership(const NativePaidExtentGroup& other) const noexcept
{
    // Both certificates bind the same closed membership type. The issuing
    // code can have a different address in separately linked library images.
    return valid() && other.valid() && same_(membership_.get(), other.membership_.get());
}
void NativePaidExtentGroup::visit_delta(const NativePaidExtentGroup& previous, MemberSink sink) const
{
    if (!valid() && !previous.valid())
        return;
    if (valid() && previous.valid() && same_membership(previous))
        return;
    const auto callback = valid() ? delta_ : previous.delta_;
    callback(previous.valid() ? previous.membership_.get() : nullptr,
        valid() ? membership_.get() : nullptr, sink);
}
namespace paid_extent_detail
{
struct Row
{
    NativePaidExtentCharge charge;
    std::size_t references = 0;
    [[nodiscard]] std::size_t exclusive_payload_bytes() const noexcept { return charge.source_only(); }
};
// The accepted leaf or the edit batch owns each referenced receipt. Borrow
// through sorting and leaf merge, then copy only the published row payload.
struct RowChange
{
    const NativePaidExtentCharge* charge = nullptr;
    std::size_t references = 0;
    RowChange() = default;
    RowChange(const NativePaidExtentCharge* value, std::size_t count) : charge(value), references(count) {}
    RowChange(const Row& value) : RowChange(&value.charge, value.references) {}
    operator Row() const noexcept { return {*charge, references}; }
};
struct Less
{
    bool operator()(const NativePaidExtentCharge& a, const NativePaidExtentCharge& b) const noexcept
    {
        if (a.kind() != b.kind()) return a.kind() < b.kind();
        if (a.owner_address() != b.owner_address())
            return std::less<const void*>{}(a.owner_address(), b.owner_address());
        if (a.origin_address() != b.origin_address())
            return std::less<const void*>{}(a.origin_address(), b.origin_address());
        if (a.direct_h() != b.direct_h()) return a.direct_h() < b.direct_h();
        if (a.publication_h() != b.publication_h()) return a.publication_h() < b.publication_h();
        if (a.financial_address() != b.financial_address())
            return std::less<const void*>{}(a.financial_address(), b.financial_address());
        return a.borrowed_financial() < b.borrowed_financial();
    }
    bool operator()(const Row& a, const Row& b) const noexcept { return (*this)(a.charge, b.charge); }
    bool operator()(const Row& a, const RowChange& b) const noexcept { return (*this)(a.charge, *b.charge); }
    bool operator()(const RowChange& a, const Row& b) const noexcept { return (*this)(*a.charge, b.charge); }
    bool operator()(const RowChange& a, const RowChange& b) const noexcept { return (*this)(*a.charge, *b.charge); }
};
using Pages = geometry_demand::OrderedPages<Row, Less>;
} // namespace paid_extent_detail
namespace geometry_demand
{
template<> inline constexpr bool stack_safe_default<paid_extent_detail::Row> = true;
template<> inline constexpr bool stack_safe_default<paid_extent_detail::RowChange> = true;
template<> inline constexpr bool native_flag_in_owned_admission<paid_extent_detail::Row> = true;
} // namespace geometry_demand
struct NativeOwnerReceipts::State
{
    paid_extent_detail::Pages::Ptr root;
    explicit State(paid_extent_detail::Pages::Ptr value) : root(std::move(value)) {}
};
bool NativeOwnerReceipts::empty() const noexcept
{
    return !state_;
}
bool NativeOwnerReceipts::same_root(const NativeOwnerReceipts& other) const noexcept
{
    return state_.get() == other.state_.get() && !state_.owner_before(other.state_) &&
           !other.state_.owner_before(state_);
}
bool NativeOwnerReceipts::contains(const NativePaidExtentCharge& charge) const
{
    using namespace paid_extent_detail;
    if (!state_)
        return false;
    const Row key{charge, 0};
    const auto* old = Pages::lower_bound(state_->root, [&](const Row& value) { return Less{}(value, key); });
    return old && !Less{}(key, *old) && old->charge.same_receipt_fact(charge);
}
bool NativeOwnerReceipts::contains_any(const NativePaidExtentCharge& charge) const
{
    return contains_category(charge, Category::Any);
}
bool NativeOwnerReceipts::contains_direct_h(const NativePaidExtentCharge& charge) const
{
    return contains_category(charge, Category::DirectH);
}
bool NativeOwnerReceipts::contains_publication_h(const NativePaidExtentCharge& charge) const
{
    return contains_category(charge, Category::PublicationH);
}
bool NativeOwnerReceipts::contains_category(const NativePaidExtentCharge& charge, Category category) const
{
    if (!state_)
        return false;
    bool found = false;
    paid_extent_detail::Pages::visit_from(state_->root,
        [&](const auto& row)
        {
            const auto& value = row.charge;
            return value.kind() < charge.kind() || (value.kind() == charge.kind() &&
                std::less<const void*>{}(value.owner_address(), charge.owner_address()));
        },
        [&](const auto& row)
        {
            const auto& value = row.charge;
            if (value.kind() != charge.kind() || value.owner_address() != charge.owner_address())
                return false;
            const bool in_category = category == Category::Any ||
                (category == Category::DirectH ? value.direct_h() : value.publication_h());
            if (in_category && value.same_owner_fact(charge) && value.same_origin_fact(charge))
            {
                if (value.bytes() != charge.bytes())
                    invalid_paid("Canonical paid physical extent changed between categories.");
                found = true;
            }
            return true;
        });
    return found;
}
bool NativeOwnerReceipts::source_only() const noexcept
{
    return state_ && state_->root->subtree_owner_bytes;
}
void NativeOwnerReceipts::visit_owner(std::uint8_t kind, const std::shared_ptr<const void>& owner, Sink sink) const
{
    using namespace paid_extent_detail;
    if (!state_)
        return;
    Pages::visit_from(
        state_->root,
        [&](const Row& row)
        {
            return row.charge.kind() < kind ||
                   (row.charge.kind() == kind && std::less<const void*>{}(row.charge.owner_address(), owner.get()));
        },
        [&](const Row& row)
        {
            if (row.charge.kind() != kind || row.charge.owner_address() != owner.get())
                return false;
            if (row.charge.owner_.expired())
                return true;
            if (!same_paid(row.charge.owner_, owner))
                invalid_paid("Paid owner query has foreign controls.");
            if (row.charge.valid())
                sink.visit(sink.context, row.charge, row.references);
            return true;
        });
}
NativeOwnerReceipts NativeOwnerReceipts::recorded(
    const NativePaidExtentCharge& charge, geometry_demand::Admission admission) const
{
    using namespace paid_extent_detail;
    if (!charge.valid())
    {
        if (charge.has_financial_pin()) invalid_paid("Native financial receipt no longer authenticates live funding.");
        return *this;
    }
    const Row key{charge, 0};
    const auto* old =
        Pages::lower_bound(state_ ? state_->root : nullptr, [&](const Row& value) { return Less{}(value, key); });
    if (old && !Less{}(key, *old))
    {
        if (!old->charge.same_receipt_fact(charge))
            invalid_paid("Native receipt replaces an immutable owner/control or extent.");
        return *this;
    }
    return retained(charge, admission);
}
NativeOwnerReceipts NativeOwnerReceipts::retained(
    const NativePaidExtentCharge& charge, geometry_demand::Admission admission) const
{
    using namespace paid_extent_detail;
    if (!charge.valid())
    {
        if (charge.has_financial_pin()) invalid_paid("Native financial receipt no longer authenticates live funding.");
        return *this;
    }
    Row row{charge, 1};
    const auto* old =
        Pages::lower_bound(state_ ? state_->root : nullptr, [&](const Row& value) { return Less{}(value, row); });
    const bool present = old && !Less{}(row, *old);
    if (present)
    {
        if (!old->charge.same_receipt_fact(charge))
            invalid_paid("Paid owner changes immutable control/extent/origin.");
        row.references = geometry_demand::checked_sum(old->references, 1);
    }
    auto root = Pages(admission).edit(state_ ? state_->root : nullptr, row, &row, present);
    admission.charge(1, sizeof(State) + geometry_demand::control_bytes);
    auto next = admission.shared<State>(std::move(root));
    const auto bytes = sizeof(State) + (admission.native ? geometry_demand::metadata_control_bytes(next) : 0);
    admission.created(next, bytes, bytes + (admission.native ? 0 : geometry_demand::control_bytes));
    NativeOwnerReceipts result;
    result.state_ = std::move(next);
    return result;
}
NativeOwnerReceipts NativeOwnerReceipts::retired(
    const NativePaidExtentCharge& charge, geometry_demand::Admission admission) const
{
    using namespace paid_extent_detail;
    if (!state_)
        invalid_paid("Retired paid owner is absent.");
    Row key{charge, 0};
    const auto* old = Pages::lower_bound(state_->root, [&](const Row& value) { return Less{}(value, key); });
    if (!old || Less{}(key, *old) || !old->references)
        invalid_paid("Retired paid owner has no exact origin row.");
    if (!old->charge.same_receipt_fact(charge))
        invalid_paid("Retired paid owner changes immutable extent or financial control.");
    auto row = *old;
    --row.references;
    const auto root = Pages(admission).edit(state_->root, row, row.references ? &row : nullptr, true);
    if (!root)
        return {};
    admission.charge(1, sizeof(State) + geometry_demand::control_bytes);
    auto next = admission.shared<State>(root);
    const auto bytes = sizeof(State) + (admission.native ? geometry_demand::metadata_control_bytes(next) : 0);
    admission.created(next, bytes, bytes + (admission.native ? 0 : geometry_demand::control_bytes));
    NativeOwnerReceipts result;
    result.state_ = std::move(next);
    return result;
}
NativeOwnerReceipts NativeOwnerReceipts::edited(std::span<Edit> edits, geometry_demand::Admission admission,
    std::span<const geometry_demand::OwnerRecord> owner_changes) const
{
    return edited_with(edits, admission, owner_changes,
        [&](const NativePaidExtentCharge& key) -> const geometry_demand::OwnerRecord*
        {
            const auto pos = std::lower_bound(owner_changes.begin(), owner_changes.end(), key,
                [](const auto& row, const auto& charge)
                {
                    return row.key.kind < charge.kind() || (row.key.kind == charge.kind() &&
                        std::less<const void*>{}(row.key.address, charge.owner_address()));
                });
            if (pos != owner_changes.end() && pos->key.kind == key.kind() && pos->key.address == key.owner_address())
                return &*pos;
            return nullptr;
        });
}
NativeOwnerReceipts NativeOwnerReceipts::edited_graph(std::span<Edit> edits, geometry_demand::Admission admission,
    const geometry_demand::OwnerEdits& owner_changes) const
{
    const auto owners = owner_changes | std::views::transform(
        [](const auto& edit) -> const geometry_demand::OwnerRecord& { return edit.row; });
    return edited_with(edits, admission, owners,
        [&](const NativePaidExtentCharge& key) -> const geometry_demand::OwnerRecord*
        {
            const auto* edit = owner_changes.find({owner_changes.channel(), key.kind(), key.owner_address()});
            return edit ? &edit->row : nullptr;
        });
}
template<class Owners, class Find>
NativeOwnerReceipts NativeOwnerReceipts::edited_with(std::span<Edit> edits, geometry_demand::Admission admission,
    const Owners& owner_changes, const Find& find_owner) const
{
    using namespace paid_extent_detail;
    for (std::size_t i = 0; i != edits.size(); ++i)
    {
        if (!edits[i].charge)
            invalid_paid("Paid receipt edit has no charge reference.");
        edits[i].sequence = i;
    }
    const Less less;
    std::sort(edits.begin(), edits.end(), [less](const Edit& a, const Edit& b)
        {
            if (less(*a.charge, *b.charge)) return true;
            if (less(*b.charge, *a.charge)) return false;
            return a.sequence < b.sequence;
        });
    const auto accepted = state_ ? state_->root : Pages::Ptr{};
    const auto owner_before = [](const NativePaidExtentCharge& charge, const geometry_demand::OwnerKey& key)
    {
        return charge.kind() < key.kind || (charge.kind() == key.kind &&
            std::less<const void*>{}(charge.owner_address(), key.address));
    };
    const auto check_removed_control = [](const NativePaidExtentCharge& charge, const auto& owner)
    {
        if (!same_paid(charge.owner_, owner.owner))
            invalid_paid("Removed paid owner changes its exact shared control.");
    };
    const auto visit_removed = [&](const auto& visit)
    {
        if (!accepted) return;
        for (const auto& owner : owner_changes)
        {
            if (owner.references) continue;
            Pages::visit_from(accepted, [&](const Row& row) { return owner_before(row.charge, owner.key); },
                [&](const Row& row)
                {
                    if (row.charge.kind() != owner.key.kind || row.charge.owner_address() != owner.key.address)
                        return false;
                    check_removed_control(row.charge, owner);
                    visit(row);
                    return true;
                });
        }
    };
    std::size_t maximum = 0;
    for (std::size_t i = 0; i != edits.size(); ++i)
        if (!i || less(*edits[i - 1].charge, *edits[i].charge))
            ++maximum;
    visit_removed([&](const Row&) { maximum = geometry_demand::checked_sum(maximum, 1); });
    if (!maximum)
        return *this;
    std::array<RowChange, 8> local{};
    geometry_detail::NativeAdministrationVector<RowChange> overflow(admission.allocator<RowChange>());
    std::span<RowChange> rows = local;
    if (maximum > local.size())
    {
        admission.charge(maximum, sizeof(RowChange));
        overflow.resize(maximum);
        if (overflow.capacity() != maximum)
            invalid_paid("Paid receipt batch capacity is unqualified.");
        rows = overflow;
    }
    std::size_t count = 0;
    for (std::size_t begin = 0; begin != edits.size();)
    {
        const auto& key = *edits[begin].charge;
        const auto* old = Pages::lower_bound(accepted, [&](const Row& row) { return less(row.charge, key); });
        if (old && less(key, old->charge)) old = nullptr;
        RowChange row = old ? RowChange(*old) : RowChange(&key, 0);
        std::size_t end = begin;
        do
        {
            const auto& edit = edits[end++];
            const auto& charge = *edit.charge;
            if (edit.action == Action::Retire || edit.action == Action::RetireIfRecorded)
            {
                if (!row.references || !row.charge->same_receipt_fact(charge))
                {
                    if (edit.action == Action::Retire)
                        invalid_paid("Retired paid owner has no exact origin row.");
                    continue;
                }
                --row.references;
            }
            else if (charge.valid())
            {
                if ((old && !old->charge.same_receipt_fact(charge)) ||
                    (row.references && !row.charge->same_receipt_fact(charge)))
                    invalid_paid("Paid owner changes immutable control/extent/origin.");
                if (!row.references) row.charge = &charge;
                if (edit.action == Action::Retain || !row.references)
                    row.references = geometry_demand::checked_sum(row.references, 1);
            }
            else if (charge.has_financial_pin())
                invalid_paid("Native financial receipt no longer authenticates live funding.");
        } while (end != edits.size() && !less(key, *edits[end].charge));
        if (row.references != (old ? old->references : 0))
        {
            const auto* owner = find_owner(key);
            if (owner && !owner->references)
                check_removed_control(*row.charge, *owner); // Already covered by the last-owner pass.
            else
                rows[count++] = std::move(row);
        }
        begin = end;
    }
    const auto explicit_changes = count;
    visit_removed([&](const Row& row) { rows[count++] = {&row.charge, 0}; });
    if (!count)
        return *this;
    auto changes = rows.first(count);
    if (count != explicit_changes)
        std::sort(changes.begin(), changes.end(), less);
    const auto input = std::span<const RowChange>(changes);
    auto root = accepted ? Pages(admission).patched_from(accepted, input,
                               [](const RowChange& row) { return row.references; }) :
                           Pages(admission).build_from(input);
    if (!root)
        return {};
    admission.charge(1, sizeof(State) + geometry_demand::control_bytes);
    auto next = admission.shared<State>(std::move(root));
    const auto bytes = sizeof(State) + (admission.native ? geometry_demand::metadata_control_bytes(next) : 0);
    admission.created(next, bytes, bytes + (admission.native ? 0 : geometry_demand::control_bytes));
    NativeOwnerReceipts result;
    result.state_ = std::move(next);
    return result;
}
void NativeOwnerReceipts::visit(Sink sink) const
{
    if (!state_)
        return;
    paid_extent_detail::Pages::visit_from(
        state_->root, [](const auto&) { return false; },
        [&](const auto& row)
        {
            sink.visit(sink.context, row.charge, row.references);
            return true;
        });
}
void NativeOwnerReceipts::visit_added(
    const NativeOwnerReceipts& previous, Sink sink, geometry_demand::PageWork* work) const
{
    using namespace paid_extent_detail;
    if (same_root(previous) || !state_)
        return;
    // Source-pruned traversal: exact shared subtrees prove unchanged origin rows.
    const auto shared = [&](const Pages::Ptr& candidate)
    {
        auto node = previous.state_ ? previous.state_->root : Pages::Ptr{};
        while (node)
        {
            if (work)
                ++work->visited_pages;
            if (node.get() == candidate.get() && !node.owner_before(candidate) && !candidate.owner_before(node))
                return true;
            if (node->leaf() || Less{}(*node->maximum, *candidate->maximum))
                return false;
            auto at = std::lower_bound(node->children.begin(), node->children.end(), *candidate->maximum,
                [](const auto& child, const auto& key) { return Less{}(*child->maximum, key); });
            if (at == node->children.end())
                return false;
            node = *at;
        }
        return false;
    };
    const auto walk = [&](const auto& self, const Pages::Ptr& node) -> void
    {
        if (shared(node))
            return;
        if (node->leaf())
            for (const auto& row : node->rows)
            {
                const auto* old = Pages::lower_bound(
                    previous.state_ ? previous.state_->root : nullptr,
                    [&](const auto& value) { return Less{}(value, row); }, work);
                if (!old || Less{}(row, *old))
                    sink.visit(sink.context, row.charge, row.references);
            }
        else
            for (const auto& child : node->children)
                self(self, child);
    };
    walk(walk, state_->root);
}
void NativeOwnerReceipts::visit_administration(geometry_demand::MetadataSink sink) const
{
    if (!state_)
        return;
    const auto bytes =
        sizeof(State) + (geometry_demand::metadata_state(state_) ? geometry_demand::metadata_control_bytes(state_) : 0);
    if (!sink(state_, bytes, bytes + (geometry_demand::metadata_state(state_) ? 0 : geometry_demand::control_bytes)))
        return;
    if (geometry_demand::metadata_state(state_))
        for (unsigned edge = 0; edge != 2; ++edge)
            geometry_demand::visit_native_state(geometry_demand::metadata_state(state_), sink);
    paid_extent_detail::Pages::visit_dependencies(state_->root, sink,
        [&](const auto& row) { sink.financial(row.charge); });
}
void NativeOwnerReceipts::visit_native_administration_delta(const NativeOwnerReceipts& previous,
    geometry_demand::NativeAdministrationSink sink, geometry_demand::PageWork* work) const
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
    paid_extent_detail::Pages::visit_native_administration_delta(
        previous.state_ ? previous.state_->root : nullptr, state_ ? state_->root : nullptr, sink, work, true,
        [&](const auto& row, bool retain) { sink.financial(row.charge, retain); });
}
std::size_t NativeOwnerReceipts::metadata_admission_bytes() const noexcept
{
    return state_ ? sizeof(State) + geometry_demand::metadata_control_bytes(state_) + state_->root->subtree_bytes : 0;
}
namespace paid_origin_detail
{
struct Row
{
    const void* address = nullptr;
    std::weak_ptr<const void> arena;
    std::size_t bytes = 0;
};
struct Less
{
    bool operator()(const Row& a, const Row& b) const noexcept
    {
        return std::less<const void*>{}(a.address, b.address);
    }
};
using Pages = geometry_demand::OrderedPages<Row, Less>;
} // namespace paid_origin_detail
namespace geometry_demand
{
template<> inline constexpr bool stack_safe_default<paid_origin_detail::Row> = true;
template<> inline constexpr bool native_flag_in_owned_admission<paid_origin_detail::Row> = true;
} // namespace geometry_demand
struct NativePaidOriginTotals::State
{
    paid_origin_detail::Pages::Ptr root;
    explicit State(paid_origin_detail::Pages::Ptr value) : root(std::move(value)) {}
};
std::size_t NativePaidOriginTotals::bytes(const std::shared_ptr<const void>& arena) const
{
    using namespace paid_origin_detail;
    if (!state_ || !arena)
        return 0;
    const Row key{arena.get(), arena, 0};
    const auto* row = Pages::lower_bound(state_->root, [&](const Row& value) { return Less{}(value, key); });
    if (!row || Less{}(key, *row))
        return 0;
    if (!same_paid(row->arena, arena))
        invalid_paid("Native paid query has a foreign reused arena control.");
    return row->bytes;
}
NativePaidOriginTotals NativePaidOriginTotals::edited(
    std::span<const Assignment> values, geometry_demand::Admission admission) const
{
    using namespace paid_origin_detail;
    if (values.empty())
        return *this;
    std::array<Row, 8> local{};
    geometry_detail::NativeAdministrationVector<Row> overflow(admission.allocator<Row>());
    std::span<Row> rows = local;
    if (values.size() > local.size())
    {
        admission.charge(values.size(), sizeof(Row));
        overflow.resize(values.size());
        if (overflow.capacity() != values.size())
            invalid_paid("Paid origin batch capacity is unqualified.");
        rows = overflow;
    }
    std::size_t count = 0;
    for (const auto& value : values)
    {
        if (!value.arena)
            invalid_paid("Native paid origin has no actual arena control.");
        if (value.bytes != bytes(value.arena))
            rows[count++] = {value.arena.get(), value.arena, value.bytes};
    }
    if (!count)
        return *this;
    const auto root = state_ ? Pages(admission).patched(state_->root, rows.first(count),
                                   [](const Row& row) { return row.bytes; }) :
                               Pages(admission).build(rows.first(count));
    if (!root)
        return {};
    admission.charge(1, sizeof(State) + geometry_demand::control_bytes);
    auto next = admission.shared<State>(root);
    const auto state_bytes = sizeof(State) + (admission.native ? geometry_demand::metadata_control_bytes(next) : 0);
    admission.created(next, state_bytes, state_bytes + (admission.native ? 0 : geometry_demand::control_bytes));
    NativePaidOriginTotals result;
    result.state_ = std::move(next);
    return result;
}
void NativePaidOriginTotals::visit_administration(geometry_demand::MetadataSink sink) const
{
    if (!state_)
        return;
    const auto bytes =
        sizeof(State) + (geometry_demand::metadata_state(state_) ? geometry_demand::metadata_control_bytes(state_) : 0);
    if (!sink(state_, bytes, bytes + (geometry_demand::metadata_state(state_) ? 0 : geometry_demand::control_bytes)))
        return;
    if (geometry_demand::metadata_state(state_))
        for (unsigned edge = 0; edge != 2; ++edge)
            geometry_demand::visit_native_state(geometry_demand::metadata_state(state_), sink);
    paid_origin_detail::Pages::visit_dependencies(state_->root, sink, [](const auto&) {});
}
void NativePaidOriginTotals::visit_native_administration_delta(const NativePaidOriginTotals& previous,
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
    paid_origin_detail::Pages::visit_native_administration_delta(
        previous.state_ ? previous.state_->root : nullptr, state_ ? state_->root : nullptr, sink, work);
}
std::size_t NativePaidOriginTotals::metadata_admission_bytes() const noexcept
{
    return state_ ? sizeof(State) + geometry_demand::metadata_control_bytes(state_) + state_->root->subtree_bytes : 0;
}
} // namespace SimpleFluid::PLIC::details
