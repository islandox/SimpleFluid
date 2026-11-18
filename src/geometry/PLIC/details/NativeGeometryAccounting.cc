#include "geometry/PLIC/details/NativeGeometryAccounting.hh"
#include "geometry/PLIC/details/GeometryAllocationLedgerInternal.hh"
#include "geometry/details/NativeAllocationGraph.hh"
namespace SimpleFluid::PLIC::details
{
namespace
{
bool same(const std::weak_ptr<const void>& a, const std::shared_ptr<const void>& b) noexcept
{
    return !a.expired() && a.lock().get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
}
[[noreturn]] void invalid(const char* text)
{
    throw GeometryError(GeometryErrorCode::InvalidInput, text);
}
} // namespace
SealedSourcePrefix::SealedSourcePrefix(const GeometryOwnerUnion& current, std::size_t residual)
    : current_(&current), generation_(current.generation()),
      bytes_(allocation_detail::add(current.resident_bytes(), residual))
{
    (void) current.manifests().overlap_bytes();
    for (unsigned channel = 0; channel != 2; ++channel)
        roots_[channel] = current.manifests().owners[channel].root_control();
}
void SealedSourcePrefix::require_current() const
{
    if (!current_ || current_->generation() != generation_)
        invalid("Native prefix changed its accepted union generation.");
    for (unsigned channel = 0; channel != 2; ++channel)
        if (!current_->manifests().owners[channel].authenticates(roots_[channel]))
            invalid("Native prefix changed its accepted union controls.");
}
bool SealedSourcePrefix::admits_library_context(const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    for (std::size_t i = 0; i != library_owner_count_; ++i)
    {
        const auto& row = library_owners_[i];
        if (row.owner.get() == owner.get())
            return row.bytes == bytes && !row.owner.owner_before(owner) && !owner.owner_before(row.owner);
    }
    return false;
}
void SealedSourcePrefix::include_library_context_owner(
    const std::shared_ptr<const void>& owner, std::size_t bytes, bool funded_by_source)
{
    if (!owner)
        invalid("Native library prefix is missing its actual owner control.");
    for (std::size_t i = 0; i != library_owner_count_; ++i)
        if (library_owners_[i].owner.get() == owner.get())
        {
            if (!admits_library_context(owner, bytes))
                invalid("Native library prefix changes an existing owner control or extent.");
            return;
        }
    if (library_owner_count_ == library_owners_.size())
        invalid("Native library prefix exceeds its typed owner inventory.");
    const auto kind = GeometryOwnerGraphEditor::external_kind();
    bool covered = current_->admits_native_administration(owner, bytes);
    for (unsigned channel = 0; channel != 2; ++channel)
    {
        const auto& root = current_->manifests().owners[channel];
        if (root.references(channel, kind, owner))
        {
            if (!root.admits(channel, kind, owner, bytes))
                invalid("Native library prefix changes an accepted owner extent.");
            covered = true;
        }
    }
    if (!covered && !funded_by_source)
        bytes_ = allocation_detail::add(bytes_, bytes);
    library_owners_[library_owner_count_++] = {owner, bytes};
}
bool NativePaidOwnerAuthority::Key::operator<(const Key& other) const noexcept
{
    if (channel != other.channel)
        return channel < other.channel;
    if (kind != other.kind)
        return kind < other.kind;
    return std::less<const void*>{}(address, other.address);
}
NativePaidOwnerAuthority::NativePaidOwnerAuthority(
    geometry_detail::NativeConstructionAdmission admission, const SealedSourcePrefix& prefix)
    : admission_(std::move(admission)), prefix_(prefix), lifetime_(admission_.shared<unsigned char>()),
      receipts_(std::less<Key>{}, admission_.allocator<std::pair<const Key, Receipt>>())
{
}
NativePaidExtentRef NativePaidOwnerAuthority::charge(const Receipt& receipt) noexcept
{
    return std::visit([](const auto& value) -> NativePaidExtentRef
        {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(value)>, std::monostate>) return {};
            else if constexpr (std::is_same_v<typename std::remove_cvref_t<decltype(value)>::element_type,
                                   NativePaidExtentCharge>) return value.get();
            else return NativePaidExtentRef(*value);
        }, receipt.charge);
}
void NativePaidOwnerAuthority::record(std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>& owner,
    std::size_t bytes, NativePaidExtentCharge charge)
{
    const std::shared_ptr<const void> empty;
    if (channel > 1 || !owner || (!owner.owner_before(empty) && !empty.owner_before(owner)))
        invalid("Native receipt has no actual owner control or valid channel.");
    const Key key{channel, kind, owner.get()};
    const auto update_charge = [&](Receipt& receipt)
    {
        if (!charge.valid())
            return;
        const auto store = [&]<class Value>(const Value& value)
        {
            if (auto* existing = std::get_if<std::shared_ptr<Value>>(&receipt.charge))
                **existing = value;
            else
                receipt.charge = admission_.shared<Value>(value);
        };
        const NativePaidExtentRef value(&charge);
        if (value.has_native_witness())
            store(value.native_witness());
        else if (const auto* proof = value.proof())
            store(*proof);
        else
            store(value.fact());
    };
    if (const auto found = receipts_.find(key); found != receipts_.end())
    {
        if (found->second.owner.expired())
        {
            // Address reuse is a new physical allocation. The stale weak
            // receipt cannot authenticate it, but its new native receipt can.
            Receipt replacement{owner, bytes};
            update_charge(replacement);
            found->second = std::move(replacement);
            return;
        }
        if (!same(found->second.owner, owner) || found->second.bytes != bytes)
            invalid("Native receipt changes its actual shared control or extent.");
        update_charge(found->second);
        return;
    }
    Receipt receipt{owner, bytes};
    update_charge(receipt);
    receipts_.emplace(key, std::move(receipt));
}
bool NativePaidOwnerAuthority::admits(
    std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes) const
{
    prefix_.require_current();
    if (channel > 1 || !owner)
        return false;
    if ((kind == GeometryOwnerGraphEditor::external_kind() &&
            (prefix_.current_->admits_native_administration(owner, bytes) ||
                prefix_.admits_library_context(owner, bytes))) ||
        prefix_.current_->manifests().owners[channel].admits(channel, kind, owner, bytes))
        return true;
    const auto found = receipts_.find({channel, kind, owner.get()});
    if (found == receipts_.end() || !same(found->second.owner, owner) || found->second.bytes != bytes)
        return false;
    if (found->second.concrete)
    {
        const auto storage = found->second.storage.lock();
        if (!storage || storage->revision() != found->second.storage_revision ||
            storage->exclusive_bytes() != found->second.storage_bytes)
            return false;
    }
    if (!found->second.profiled)
        return true;
    const auto profile = found->second.profile.lock();
    return profile && profile->state() == admission_.state() && profile->bytes() == bytes;
}
bool NativePaidOwnerAuthority::admits_root(const geometry_demand::GeometryOwnerManifest& root) const
{
    prefix_.require_current();
    if (!root.owner_count())
        return true;
    return root.same_root(prefix_.current_->manifests().owners[0]) ||
           root.same_root(prefix_.current_->manifests().owners[1]);
}
void NativePaidOwnerAuthority::state(const std::shared_ptr<const geometry_detail::NativeAllocationState>& value)
{
    const auto counter_bytes = geometry_detail::NativeAllocationGraphAccess::validate_state(value);
    if (value != admission_.state() || value.owner_before(admission_.state()) || admission_.state().owner_before(value))
    {
        // Accepted old domains are paid only through their exact prefix rows.
        for (unsigned channel = 0; channel != 2; ++channel)
            if (admits(channel, GeometryOwnerGraphEditor::external_kind(), value, value->allocation_admission_bytes))
                return;
        invalid("Native allocation receipt belongs to a foreign admission State.");
    }
    for (std::uint8_t channel = 0; channel != 2; ++channel)
    {
        record(channel, GeometryOwnerGraphEditor::external_kind(), value, value->allocation_admission_bytes,
            NativePaidExtentCharge::state(value));
        record(channel, GeometryOwnerGraphEditor::external_kind(), value->counter_owner, counter_bytes,
            NativePaidExtentCharge::counter(value));
        if (value->mirror.owner)
            record(channel, GeometryOwnerGraphEditor::external_kind(), value->mirror.owner,
                value->mirror.owner_proof.bytes(), NativePaidExtentCharge::mirror(value));
    }
}
void NativePaidOwnerAuthority::fixed(const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& value)
{
    if (!value || !value->authentic_owner(value))
        invalid("Native fixed receipt has a foreign shared control.");
    const auto bytes = value->fixed_allocation_bytes();
    if (value->state() != admission_.state())
    {
        for (unsigned channel = 0; channel != 2; ++channel)
            if (admits(channel, GeometryOwnerGraphEditor::external_kind(), value, bytes))
                return;
        invalid("Native fixed receipt belongs to a foreign admission State.");
    }
    state(value->state());
    for (std::uint8_t channel = 0; channel != 2; ++channel)
        record(channel, GeometryOwnerGraphEditor::external_kind(), value, bytes, NativePaidExtentCharge::fixed(value));
}
void NativePaidOwnerAuthority::storage(const std::shared_ptr<geometry_detail::NativeConstructorStorage>& value,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope)
{
    if (!value || !envelope || !envelope->authentic_owner(envelope) || !envelope->matches(value) ||
        value->admission().state() != envelope->state())
        invalid("Native storage receipt changes its actual target/control or domain.");
    fixed(envelope);
    if (value->admission().state() != admission_.state())
        return;
    if (value->revision() == std::numeric_limits<std::size_t>::max())
        invalid("Native storage has exhausted its immutable receipt revision.");
    for (std::uint8_t channel = 0; channel != 2; ++channel)
    {
        record(channel, GeometryOwnerGraphEditor::external_kind(), value, value->exclusive_bytes(),
            NativePaidExtentCharge::storage({value, envelope}));
        auto& receipt = receipts_.at({channel, GeometryOwnerGraphEditor::external_kind(), value.get()});
        receipt.concrete = true;
        receipt.storage = value;
        receipt.storage_revision = value->revision();
        receipt.storage_bytes = value->exclusive_bytes();
    }
}
void NativePaidOwnerAuthority::semantic(std::uint8_t channel, std::uint8_t kind,
    const std::shared_ptr<const void>& owner,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope)
{
    if (!envelope || !envelope->authentic_owner(envelope) || !envelope->matches(owner))
        invalid("Native semantic owner has a foreign actual target/control.");
    if (kind > std::uint8_t(allocation_detail::Kind::CellShellPage) &&
        kind != std::uint8_t(allocation_detail::Kind::PlanarWrapper) &&
        kind != std::uint8_t(allocation_detail::Kind::PlanarPayload))
        invalid("Native semantic owner has no canonical allocation kind.");
    fixed(envelope);
    if (envelope->state() != admission_.state())
        return;
    record(channel, kind, owner, 0);
}
void NativePaidOwnerAuthority::identity(std::uint8_t channel, const GeometryIdentityBackingPtr& owner)
{
    const auto* admitted = NativeIdentityOwnerAccess::backing(owner);
    if (!admitted)
        invalid("Native paid identity lacks its admitted backing.");
    // The body key is the exact base subobject. The checked alias uses the
    // concrete backing address and the same actual shared control.
    fixed(admitted->envelope());
    if (admitted->envelope()->state() == admission_.state())
        record(channel, std::uint8_t(allocation_detail::Kind::IdentityPayload), owner, 0);
}
void NativePaidOwnerAuthority::metadata(
    std::uint8_t channel, const std::shared_ptr<const void>& owner, std::size_t bytes)
{
    state(admission_.state());
    record(channel, GeometryOwnerGraphEditor::external_kind(), owner, bytes,
        NativePaidExtentCharge::metadata(owner, bytes));
}
const SealedSourcePrefix& NativeGeometryAccounting::empty_prefix()
{
    static const GeometryOwnerUnion current;
    static const SealedSourcePrefix prefix(current, 0);
    return prefix;
}
NativeGeometryAccounting::NativeGeometryAccounting(geometry_detail::NativeConstructionAdmission admission)
    : NativeGeometryAccounting(std::move(admission), empty_prefix())
{
}
NativeGeometryAccounting::NativeGeometryAccounting(
    geometry_detail::NativeConstructionAdmission admission, const SealedSourcePrefix& prefix)
    : admission_(std::move(admission)), prefix_(prefix), paid_(admission_, prefix), pair_(ordinary_, bounded_)
{
    prefix_.require_current();
    if (admission_.statistics().imported_live != prefix_.bytes())
        invalid("Native domain was not born against the exact sealed physical prefix.");
    GeometryAllocationOwnerAccess::begin_native_capture(ordinary_, admission_, paid_, cursor_, limit());
    GeometryAllocationOwnerAccess::begin_native_capture(bounded_, admission_, paid_, cursor_, limit(), 1);
    paid_.state(admission_.state());
}
NativeGeometryAccounting::~NativeGeometryAccounting() = default;
void NativeGeometryAccounting::import_current(const GeometryOwnerUnion& current)
{
    prefix_.require_current();
    if (&current != prefix_.current_)
        invalid("Native operation cannot import a foreign accepted union.");
    current.import(ordinary_, bounded_, cursor_, limit());
}
OwnerManifestPair NativeGeometryAccounting::finish_pair(
    GeometryOwnerGraphEditor& ordinary, GeometryOwnerGraphEditor& bounded, std::size_t overlap)
{
    try
    {
        if (ordinary.impl_->ledger != &ordinary_ || bounded.impl_->ledger != &bounded_ ||
            ordinary.impl_->channel != 0 || bounded.impl_->channel != 1 || &ordinary.impl_->used != &cursor_ ||
            &bounded.impl_->used != &cursor_ || ordinary.impl_->partner != bounded.impl_ ||
            bounded.impl_->partner != ordinary.impl_ || !ordinary.impl_->cross_payload_bytes ||
            *ordinary.impl_->cross_payload_bytes != overlap)
            invalid("Native manifest pair has foreign editors or overlap authority.");
        return OwnerManifestPair::sealed({ordinary.finish(), bounded.finish()}, overlap);
    }
    catch (...)
    {
        GeometryAllocationOwnerAccess::poison(ordinary_);
        GeometryAllocationOwnerAccess::poison(bounded_);
        throw;
    }
}
void NativeGeometryAccounting::authorize_fixed(
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& value)
{
    paid_.fixed(value);
}
void NativeGeometryAccounting::authorize_state(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& value)
{
    paid_.state(value);
}
void NativeGeometryAccounting::authorize_storage(
    const std::shared_ptr<geometry_detail::NativeConstructorStorage>& value,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope)
{
    paid_.storage(value, envelope);
}
void NativeGeometryAccounting::authorize_semantic(std::uint8_t channel, std::uint8_t kind,
    const std::shared_ptr<const void>& owner,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope)
{
    if (kind != GeometryOwnerGraphEditor::external_kind())
        invalid("Opaque native semantic receipt requires the canonical external kind.");
    paid_.semantic(channel, kind, owner, envelope);
}
void NativeGeometryAccounting::authorize_schema(const std::shared_ptr<const geometry_detail::NativeSchemaOwner>& owner)
{
    if (!owner)
        invalid("Native schema receipt has no owner.");
    const auto& envelope = owner->allocation();
    paid_.semantic(0, GeometryOwnerGraphEditor::external_kind(), owner, envelope);
    if (owner->capacity_bytes() && envelope->state() == admission_.state())
    {
        const std::shared_ptr<const void> alias(owner, owner->span().data());
        if (const auto* retained = owner->retained_input_extent())
        {
            const std::shared_ptr<const geometry_detail::NativeRetainedExtent> extent(owner, retained);
            paid_.record(0, GeometryOwnerGraphEditor::external_kind(), alias, owner->capacity_bytes(),
                NativePaidExtentCharge::retained_profile(alias, extent, envelope));
            auto& receipt = paid_.receipts_.at({0, GeometryOwnerGraphEditor::external_kind(), alias.get()});
            receipt.profile = extent;
            receipt.profiled = true;
        }
        else
        {
            const NativeActualBufferProof proof(owner, envelope, envelope->state(),
                {owner->span().data(), owner->capacity_bytes()});
            paid_.record(0, GeometryOwnerGraphEditor::external_kind(), alias, owner->capacity_bytes(),
                NativePaidExtentCharge::buffer(proof));
        }
    }
}
void NativeGeometryAccounting::authorize_buffer(const NativeActualBufferProof& proof)
{
    if (!proof.owner_ || !proof.envelope_ || !proof.envelope_->authentic_owner(proof.envelope_) ||
        !proof.envelope_->matches(proof.owner_) || proof.state_ != proof.envelope_->state() ||
        proof.state_.owner_before(proof.envelope_->state()) || proof.envelope_->state().owner_before(proof.state_) ||
        !proof.extent_.address || !proof.extent_.bytes)
        invalid("Typed native buffer receipt changes its actual target/control or allocation extent.");
    const std::shared_ptr<const void> alias(proof.owner_, proof.extent_.address);
    if (proof.state_ != admission_.state())
    {
        if (!paid_.admits(0, GeometryOwnerGraphEditor::external_kind(), alias, proof.extent_.bytes))
            invalid("Typed native buffer belongs to a foreign non-prefix domain.");
        return;
    }
    paid_.fixed(proof.envelope_);
    paid_.record(0, GeometryOwnerGraphEditor::external_kind(), alias, proof.extent_.bytes,
        NativePaidExtentCharge::buffer(proof));
}
void NativeGeometryAccounting::authorize_buffer(const std::shared_ptr<const void>& alias, std::size_t bytes,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& target,
    const geometry_detail::NativeConstructorStorageOwner& storage)
{
    if (!target || !target->authentic_owner(target) || !target->shares_control(alias) || !storage.owner ||
        storage.owner->admission().state() != target->state() || target->state() != admission_.state())
        invalid("Native buffer receipt has foreign target/control or allocation State.");
    paid_.storage(storage.owner, storage.envelope);
    bool actual = false;
    storage.owner->visit_allocations([&](const geometry_detail::NativeAllocationExtent& extent)
        { actual = actual || (extent.address == alias.get() && extent.bytes == bytes); });
    if (!actual)
        invalid("Native buffer alias lacks an exact actual allocation extent.");
    paid_.record(0, GeometryOwnerGraphEditor::external_kind(), alias, bytes);
    auto& receipt = paid_.receipts_.at({0, GeometryOwnerGraphEditor::external_kind(), alias.get()});
    receipt.concrete = true;
    receipt.storage = storage.owner;
    receipt.storage_revision = storage.owner->revision();
    receipt.storage_bytes = storage.owner->exclusive_bytes();
}
void NativeGeometryAccounting::authorize_profile_owner(std::uint8_t channel, const std::shared_ptr<const void>& owner,
    const std::shared_ptr<const geometry_detail::NativeRetainedExtent>& extent,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope)
{
    if (!envelope || !envelope->matches(extent))
        invalid("Native retained library profile has a foreign holder envelope.");
    const auto charge = NativePaidExtentCharge::retained_profile(owner, extent, envelope);
    if (extent->state() != admission_.state())
    {
        if (!paid_.admits(channel, GeometryOwnerGraphEditor::external_kind(), owner, extent->bytes()))
            invalid("Native retained library profile has a foreign non-prefix State.");
        return;
    }
    paid_.record(channel, GeometryOwnerGraphEditor::external_kind(), owner, extent->bytes(), charge);
    auto& receipt = paid_.receipts_.at({channel, GeometryOwnerGraphEditor::external_kind(), owner.get()});
    receipt.profile = extent;
    receipt.profiled = true;
}
void GeometryAllocationOwnerAccess::begin_native_bookkeeping(GeometryAllocationLedger& ledger,
    const geometry_detail::NativeConstructionAdmission& admission, std::size_t& cursor, std::size_t limit)
{
    try
    {
        const auto bytes =
            sizeof(GeometryAllocationLedger::Impl) + sizeof(GeometryAllocationLedger::Impl::NativeContext);
        if (ledger.impl_ || ledger.failed_ || &cursor == admission.state()->used_counter || cursor > limit ||
            bytes > limit - cursor)
            invalid("Native proof bookkeeping requires a fresh ledger and independent finite logical cursor.");
        admission.record_ledger_work(bytes);
        cursor += bytes;
        auto storage = admission.unique<GeometryAllocationLedger::Impl>(admission.state(), nullptr);
        ledger.impl_ =
            decltype(ledger.impl_)(storage.release(), GeometryAllocationLedger::ImplDelete{admission.state()});
        ledger.impl_->native->cursor = &cursor;
        ledger.impl_->native->logical_limit = limit;
        ledger.impl_->bookkeeping = bytes;
        ledger.impl_->last_counter = cursor;
        ledger.impl_->peer = paired_peer(ledger);
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}
void GeometryAllocationOwnerAccess::begin_native_capture(GeometryAllocationLedger& ledger,
    const geometry_detail::NativeConstructionAdmission& admission, NativePaidOwnerAuthority& paid, std::size_t& used,
    std::size_t limit, std::uint8_t channel)
{
    try
    {
        if (channel > 1 || ledger.impl_ || ledger.failed_ || used > limit ||
            sizeof(GeometryAllocationLedger::Impl) + sizeof(GeometryAllocationLedger::Impl::NativeContext) >
                limit - used)
            invalid("Native capture requires a fresh ledger and finite bookkeeping budget.");
        admission.record_ledger_work(
            sizeof(GeometryAllocationLedger::Impl) + sizeof(GeometryAllocationLedger::Impl::NativeContext));
        used += sizeof(GeometryAllocationLedger::Impl) + sizeof(GeometryAllocationLedger::Impl::NativeContext);
        auto storage = admission.unique<GeometryAllocationLedger::Impl>(admission.state(), &paid);
        ledger.impl_ =
            decltype(ledger.impl_)(storage.release(), GeometryAllocationLedger::ImplDelete{admission.state()});
        ledger.impl_->mode = channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary;
        ledger.impl_->native->lifetime = paid.lifetime_;
        ledger.impl_->native->cursor = &used;
        ledger.impl_->native->logical_limit = limit;
        ledger.impl_->prepaid_payload = true;
        ledger.impl_->bookkeeping =
            sizeof(GeometryAllocationLedger::Impl) + sizeof(GeometryAllocationLedger::Impl::NativeContext);
        ledger.impl_->last_counter = used;
        ledger.impl_->peer = paired_peer(ledger);
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}
NativePaidExtentCharge GeometryAllocationOwnerAccess::paid_charge(const GeometryAllocationLedger& ledger,
    std::uint8_t channel, std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes)
{
    if (!ledger.impl_ || !ledger.impl_->native_paid())
        return {};
    auto* authority = ledger.impl_->native_authority();
    const auto at = authority->receipts_.find({channel, kind, owner.get()});
    if (at == authority->receipts_.end())
        return {};
    const auto recorded = NativePaidOwnerAuthority::charge(at->second);
    if (!recorded)
        return {};
    auto charge = recorded.value();
    return charge.matches(kind, owner, bytes) ? std::move(charge) : NativePaidExtentCharge{};
}
std::shared_ptr<const geometry_detail::NativeAllocationState> GeometryAllocationOwnerAccess::native_state(
    const GeometryAllocationLedger& ledger) noexcept
{
    return ledger.impl_ ? ledger.impl_->native_state() : nullptr;
}
void GeometryAllocationOwnerAccess::require_native_paid(const GeometryAllocationLedger& ledger, std::uint8_t channel,
    std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes)
{
    if (ledger.impl_ && ledger.impl_->native_paid() &&
        !ledger.impl_->native_authority()->admits(channel, kind, owner, bytes))
    {
        if (ledger.impl_->peer)
            poison(*ledger.impl_->peer);
        invalid("Native owner lacks exact prefix/allocation receipt authority.");
    }
}
void GeometryAllocationOwnerAccess::register_native_metadata(
    GeometryAllocationLedger& ledger, std::uint8_t channel, const std::shared_ptr<const void>& owner, std::size_t bytes)
{
    if (ledger.impl_ && ledger.impl_->native_paid())
        ledger.impl_->native_authority()->metadata(channel, owner, bytes);
}
void GeometryAllocationOwnerAccess::authorize_native_fixed(
    GeometryAllocationLedger& ledger, const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& value)
{
    if (ledger.impl_ && ledger.impl_->native_paid())
        ledger.impl_->native_authority()->fixed(value);
}
void GeometryAllocationOwnerAccess::authorize_native_state(
    GeometryAllocationLedger& ledger, const std::shared_ptr<const geometry_detail::NativeAllocationState>& value)
{
    if (ledger.impl_ && ledger.impl_->native_paid())
        ledger.impl_->native_authority()->state(value);
}
void GeometryAllocationOwnerAccess::authorize_native_storage(GeometryAllocationLedger& ledger,
    const std::shared_ptr<geometry_detail::NativeConstructorStorage>& value,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope)
{
    if (ledger.impl_ && ledger.impl_->native_paid())
        ledger.impl_->native_authority()->storage(value, envelope);
}
void GeometryAllocationOwnerAccess::authorize_native_semantic(GeometryAllocationLedger& ledger, std::uint8_t channel,
    std::uint8_t kind, const std::shared_ptr<const void>& value,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope)
{
    if (ledger.impl_ && ledger.impl_->native_paid())
        ledger.impl_->native_authority()->semantic(channel, kind, value, envelope);
}
void GeometryAllocationOwnerAccess::authorize_native_identity(
    GeometryAllocationLedger& ledger, std::uint8_t channel, const GeometryIdentityBackingPtr& owner)
{
    if (ledger.impl_ && ledger.impl_->native_paid())
        ledger.impl_->native_authority()->identity(channel, owner);
}
} // namespace SimpleFluid::PLIC::details
namespace SimpleFluid::PLIC
{
void GeometryAllocationLedger::Impl::NativeContextDelete::operator()(NativeContext* value) const noexcept
{
    if (!value)
        return;
    const auto state = value->state;
    std::destroy_at(value);
    std::allocator<NativeContext>{}.deallocate(value, 1);
    if (state)
        state->retire(sizeof(NativeContext));
}
std::unique_ptr<GeometryAllocationLedger::Impl::NativeContext, GeometryAllocationLedger::Impl::NativeContextDelete>
GeometryAllocationLedger::Impl::make_context(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& state, details::NativePaidOwnerAuthority* paid)
{
    geometry_detail::NativeConstructionAllocator<NativeContext> allocator(state);
    auto* memory = allocator.allocate(1);
    try
    {
        std::construct_at(memory, state, paid);
    }
    catch (...)
    {
        allocator.deallocate(memory, 1);
        throw;
    }
    return {memory, NativeContextDelete{}};
}
void GeometryAllocationLedger::Impl::set_native_grant(
    const details::geometry_demand::GeometryOwnerManifest& value, std::size_t& used, std::size_t limit)
{
    if (!native && value.owner_count())
    {
        account(sizeof(NativeContext), true, used, limit);
        auto* memory = std::allocator<NativeContext>{}.allocate(1);
        std::construct_at(memory);
        native = decltype(native)(memory, NativeContextDelete{});
    }
    if (native)
        native->administration_grant = value;
}
details::NativePaidOwnerAuthority* GeometryAllocationLedger::Impl::native_authority() const
{
    if (native_paid() && native->lifetime.expired())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Native ledger operation authority has expired.");
    return native_paid();
}
bool GeometryAllocationLedger::Impl::native_import_authorized(
    const details::geometry_demand::GeometryOwnerManifest& root) const
{
    return native_paid() && native_authority()->admits_root(root);
}
bool GeometryAllocationLedger::Impl::native_paid_admits(
    std::uint8_t channel, const allocation_detail::Item& item, std::size_t bytes) const
{
    try
    {
        const bool valid = native_authority()->admits(channel, static_cast<std::uint8_t>(item.kind), item.owner, bytes);
        if (!valid && peer)
            details::GeometryAllocationOwnerAccess::poison(*peer);
        return valid;
    }
    catch (...)
    {
        if (peer)
            details::GeometryAllocationOwnerAccess::poison(*peer);
        throw;
    }
}
void GeometryAllocationLedger::Impl::authorize_native_fixed_receipt(
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& owner)
{
    native_authority()->fixed(owner);
}
void GeometryAllocationLedger::Impl::native_typed_receipt(allocation_detail::Kind kind,
    const std::shared_ptr<const void>& owner,
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& envelope,
    const geometry_detail::NativeConstructorStorageOwner& storage, std::size_t&, std::size_t)
{
    if (!native_paid()) return;
    const auto channel = mode == ArithmeticMode::BoundedCertification ? 1 : 0;
    native_authority()->semantic(channel, std::uint8_t(kind), owner, envelope);
    if (storage.owner)
        native_authority()->storage(storage.owner, storage.envelope);
}
void GeometryAllocationLedger::Impl::authorize_native_identity_receipt(const details::GeometryIdentityBackingPtr& owner)
{
    const auto* admitted = details::NativeIdentityOwnerAccess::backing(owner);
    if (!admitted)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Native paid identity lacks its admitted backing.");
    const auto channel = mode == ArithmeticMode::BoundedCertification ? 1 : 0;
    native_authority()->identity(channel, owner);
    native_authority()->storage(admitted->storage().owner, admitted->storage().envelope);
}
void GeometryAllocationLedger::Impl::authorize_native_state_receipt(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& owner)
{
    native_authority()->state(owner);
}
} // namespace SimpleFluid::PLIC
