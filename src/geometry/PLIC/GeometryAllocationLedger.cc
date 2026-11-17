/** @file GeometryAllocationLedger.cc
 * @brief Alias-aware immutable geometry ownership traversal and cumulative admission.
 */
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/GeometryAllocationLedgerInternal.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryDemandPages.tcc"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.tcc"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.tcc"
#include <type_traits>

namespace SimpleFluid::PLIC
{
namespace
{
thread_local details::GeometryAllocationWork allocation_work;
}
details::GeometryAllocationWork details::GeometryAllocationOwnerAccess::work_statistics() noexcept
{
    return allocation_work;
}
void details::GeometryAllocationOwnerAccess::reset_work_statistics() noexcept
{
    allocation_work = {};
}
void details::GeometryAllocationOwnerAccess::observe_work(Work kind, std::size_t count) noexcept
{
    switch (kind)
    {
        case Work::Push:
            allocation_work.pushes += count;
            break;
        case Work::Admit:
            allocation_work.admissions += count;
            break;
        case Work::PendingVisit:
            allocation_work.pending_visits += count;
            break;
        case Work::Import:
            allocation_work.import_calls += count;
            break;
        case Work::ImportRow:
            allocation_work.import_rows += count;
            break;
        case Work::Export:
            allocation_work.export_calls += count;
            break;
        case Work::ExportRow:
            allocation_work.export_rows += count;
            break;
        case Work::ManifestVisitRow:
            allocation_work.manifest_visit_rows += count;
            break;
        case Work::MetadataCreditQuery:
            allocation_work.metadata_credit_queries += count;
            break;
        case Work::MetadataCreditSeedRow:
            allocation_work.metadata_credit_seed_rows += count;
            break;
        case Work::MetadataCreditCreated:
            allocation_work.metadata_credit_created += count;
            break;
    }
}
std::size_t details::identity_ledger_bookkeeping_bound(std::size_t owners)
{
    return allocation_detail::add(sizeof(GeometryAllocationLedger::Impl),
        allocation_detail::product(
            owners, sizeof(typename decltype(GeometryAllocationLedger::Impl::owners)::value_type) + 4 * sizeof(void*)));
}

GeometryAllocationLedger::GeometryAllocationLedger() noexcept = default;
GeometryAllocationLedger::~GeometryAllocationLedger() = default;
GeometryAllocationLedger::GeometryAllocationLedger(GeometryAllocationLedger&&) noexcept = default;
GeometryAllocationLedger& GeometryAllocationLedger::operator=(GeometryAllocationLedger&&) noexcept = default;
std::size_t GeometryAllocationLedger::owner_count() const noexcept
{
    if (!impl_)
        return 0;
    return impl_->owners.size() + (impl_->imported ? impl_->imported->owner_count() : 0);
}
std::size_t GeometryAllocationLedger::payload_bytes() const noexcept
{
    return impl_ ? impl_->payload : 0;
}
std::size_t GeometryAllocationLedger::bookkeeping_bytes() const noexcept
{
    return impl_ ? impl_->bookkeeping : 0;
}

void GeometryAllocationLedger::begin_admission(std::size_t& used, std::size_t limit, bool allocate_state)
{
    if (failed_)
        allocation_detail::exhausted("Geometry allocation ledger cannot resume rejected admission.");
    if (impl_ && used < impl_->last_counter)
        allocation_detail::exhausted("Geometry allocation cumulative byte count decreased.");
    if (used > limit)
        allocation_detail::exhausted("Geometry allocation byte budget is already exhausted.");
    if (!impl_ && allocate_state)
    {
        if (sizeof(Impl) > limit - used)
            allocation_detail::exhausted("Geometry allocation ledger byte budget is exhausted.");
        used += sizeof(Impl);
        impl_ = std::make_unique<Impl>();
        impl_->bookkeeping = sizeof(Impl);
        impl_->last_counter = used;
    }
}

void details::GeometryAllocationOwnerAccess::begin(
    GeometryAllocationLedger& ledger, std::size_t& cumulative, std::size_t bookkeeping_limit)
{
    if (ledger.impl_ || bookkeeping_limit < sizeof(GeometryAllocationLedger::Impl))
        allocation_detail::exhausted("Geometry owner census bookkeeping budget is exhausted.");
    ledger.begin_admission(cumulative, std::numeric_limits<std::size_t>::max());
    ledger.impl_->scratch_limit = bookkeeping_limit;
}

void details::GeometryAllocationOwnerAccess::begin_prepaid_capture(
    GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
{
    try
    {
        if (ledger.impl_)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Prepaid geometry capture requires a fresh ledger.");
        ledger.begin_admission(used, limit);
        ledger.impl_->prepaid_payload = true;
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}

void details::GeometryAllocationOwnerAccess::admit_prepaid(GeometryAllocationLedger& ledger,
    const std::shared_ptr<const void>& owner, std::size_t exclusive_bytes, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!owner)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null prepaid allocation owner.");
        ledger.begin_admission(used, limit);
        (void) ledger.impl_->admit({allocation_detail::Kind::ExternalOwner, owner}, exclusive_bytes, used, limit, true);
        ledger.impl_->last_counter = used;
    }
    catch (...)
    {
        // A rejected admission poisons the ledger exactly like every other
        // mutator: a partial payload/owner split must never be reused.
        ledger.failed_ = true;
        throw;
    }
}

void details::GeometryAllocationOwnerAccess::poison(GeometryAllocationLedger& ledger) noexcept
{
    ledger.failed_ = true;
}

void details::GeometryAllocationOwnerAccess::seed_imported_metadata(
    GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
{
    if (!ledger.impl_->imported || ledger.impl_->imported_metadata_seeded)
        return;
    struct Context
    {
        GeometryAllocationLedger& ledger;
        std::size_t& used;
        std::size_t limit;
    } context{ledger, used, limit};
    // Explicit Stage2 compatibility census, once per imported root/ledger. A
    // later incremental union protocol can authenticate shared index frontiers.
    ledger.impl_->imported->visit_dependencies(
        {&context,
            [](void* raw, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted)
            {
                auto& value = *static_cast<Context*>(raw);
                observe_work(Work::MetadataCreditSeedRow);
                const allocation_detail::Key key{allocation_detail::Kind::ExternalOwner, owner.get()};
                auto& credits = value.ledger.impl_->metadata_credits;
                if (const auto found = credits.find(key); found != credits.end())
                {
                    if (found->second.payload != payload || found->second.admitted != admitted ||
                        owner.owner_before(found->second.owner) || found->second.owner.owner_before(owner))
                        throw GeometryError(
                            GeometryErrorCode::InvalidInput, "Imported index credit changes owner or size.");
                }
                else
                {
                    value.ledger.impl_->account(
                        sizeof(typename std::remove_reference_t<decltype(credits)>::value_type) + 4 * sizeof(void*),
                        true, value.used, value.limit);
                    credits.emplace(
                        key, GeometryAllocationLedger::Impl::MetadataCredit{owner, payload, admitted, true});
                }
                return true;
            }},
        {nullptr, [](void*, const geometry_demand::OwnerRecord&) {}});
    ledger.impl_->imported_metadata_seeded = true;
    ledger.impl_->last_counter = used;
}
void details::GeometryAllocationOwnerAccess::retain_manifest_metadata(GeometryAllocationLedger& ledger,
    std::uint8_t channel, const std::shared_ptr<const void>& owner, std::size_t payload, std::size_t admitted,
    std::size_t& used, std::size_t limit, bool prepaid)
{
    observe_work(Work::MetadataCreditQuery);
    try
    {
        const std::shared_ptr<const void> empty;
        if (channel > 1 || !owner || admitted < payload || admitted - payload != geometry_demand::control_bytes ||
            (!owner.owner_before(empty) && !empty.owner_before(owner)))
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid own-index metadata receipt.");
        ledger.begin_admission(used, limit);
        const auto mode = channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary;
        if (ledger.impl_->mode)
            details::require_mode(*ledger.impl_->mode, mode);
        else
            ledger.impl_->mode = mode;
        // An actual new allocation cannot alias still-pinned imported index
        // storage. Its trusted constructor receipt needs no old-root census.
        if (!prepaid)
            seed_imported_metadata(ledger, used, limit);
        const allocation_detail::Key key{allocation_detail::Kind::ExternalOwner, owner.get()};
        auto& credits = ledger.impl_->metadata_credits;
        if (const auto found = credits.find(key); found != credits.end())
        {
            if (found->second.payload != payload || found->second.admitted != admitted ||
                owner.owner_before(found->second.owner) || found->second.owner.owner_before(owner))
                throw GeometryError(GeometryErrorCode::InvalidInput, "Immutable index credit changes owner or size.");
            ledger.impl_->last_counter = used;
            return;
        }
        (void) ledger.impl_->admit({allocation_detail::Kind::ExternalOwner, owner}, payload, used, limit, prepaid);
        const auto control = admitted - payload;
        if (prepaid || ledger.impl_->prepaid_payload)
            ledger.impl_->bookkeeping = allocation_detail::add(ledger.impl_->bookkeeping, control);
        else
            ledger.impl_->account(control, true, used, limit);
        ledger.impl_->account(
            sizeof(typename std::remove_reference_t<decltype(credits)>::value_type) + 4 * sizeof(void*), true, used,
            limit);
        credits.emplace(key, GeometryAllocationLedger::Impl::MetadataCredit{owner, payload, admitted, false});
        observe_work(Work::MetadataCreditCreated);
        ledger.impl_->last_counter = used;
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}
bool details::GeometryAllocationOwnerAccess::imported_metadata_seeded(const GeometryAllocationLedger& ledger) noexcept
{
    return !ledger.impl_ || !ledger.impl_->imported || ledger.impl_->imported_metadata_seeded;
}
void details::GeometryAllocationOwnerAccess::visit_metadata_credits(const GeometryAllocationLedger& ledger,
    const std::function<void(const std::shared_ptr<const void>&, std::size_t, std::size_t, bool)>& visitor)
{
    if (ledger.failed_)
        allocation_detail::exhausted("Cannot inspect rejected index metadata credits.");
    if (ledger.impl_)
        for (const auto& [key, credit] : ledger.impl_->metadata_credits)
            visitor(credit.owner, credit.payload, credit.admitted, credit.imported_bookkeeping);
}

void details::GeometryAllocationOwnerAccess::admit_graph_owner(GeometryAllocationLedger& ledger, std::uint8_t channel,
    std::uint8_t kind, const std::shared_ptr<const void>& owner, std::size_t bytes, std::size_t& used,
    std::size_t limit, bool prepaid)
{
    try
    {
        if (channel > 1 || !owner)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid typed geometry graph owner.");
        ledger.begin_admission(used, limit);
        const auto mode = channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary;
        if (ledger.impl_->mode)
            details::require_mode(*ledger.impl_->mode, mode);
        else
            ledger.impl_->mode = mode;
        (void) ledger.impl_->admit({static_cast<allocation_detail::Kind>(kind), owner}, bytes, used, limit, prepaid);
        ledger.impl_->last_counter = used;
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}
void details::GeometryAllocationOwnerAccess::visit(
    const GeometryAllocationLedger& ledger, const std::function<void(std::uint8_t, const void*, std::size_t)>& visitor)
{
    if (ledger.failed_)
        allocation_detail::exhausted("Cannot inspect rejected geometry owner census.");
    if (ledger.impl_)
        for (const auto& [key, entry] : ledger.impl_->owners)
            visitor(static_cast<std::uint8_t>(key.kind), entry.pin.get(), entry.payload_bytes);
}

void GeometryAllocationLedger::retain_external_owner(
    const std::shared_ptr<const void>& owner, std::size_t exclusive_bytes, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!owner)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null external allocation owner.");
        begin_admission(used, limit);
        if (const auto credit = impl_->metadata_credits.find({allocation_detail::Kind::ExternalOwner, owner.get()});
            credit != impl_->metadata_credits.end())
        {
            if (credit->second.payload != exclusive_bytes || owner.owner_before(credit->second.owner) ||
                credit->second.owner.owner_before(owner))
                throw GeometryError(GeometryErrorCode::InvalidInput, "External index credit changes owner or size.");
            impl_->last_counter = used;
            return;
        }
        // An imported owner is already in this ledger's count and pin set. An
        // exact size match proves identity; a changed size is a real conflict.
        if (impl_->imported)
        {
            const auto references = impl_->imported->references(
                impl_->imported_channel, static_cast<std::uint8_t>(allocation_detail::Kind::ExternalOwner), owner);
            if (references)
            {
                if (!impl_->imported->admits(impl_->imported_channel,
                        static_cast<std::uint8_t>(allocation_detail::Kind::ExternalOwner), owner, exclusive_bytes))
                    throw GeometryError(GeometryErrorCode::InvalidInput, "Immutable external allocation size changed.");
                impl_->last_counter = used;
                return;
            }
        }
        if (impl_->owners.contains({allocation_detail::Kind::ExternalOwner, owner.get()}))
        {
            // Authenticate existing mutable credit before any compatibility
            // allocation; a foreign control/size rejection spends no new bytes.
            impl_->admit({allocation_detail::Kind::ExternalOwner, owner}, exclusive_bytes, used, limit);
            impl_->last_counter = used;
            return;
        }
        details::GeometryAllocationOwnerAccess::seed_imported_metadata(*this, used, limit);
        if (const auto credit = impl_->metadata_credits.find({allocation_detail::Kind::ExternalOwner, owner.get()});
            credit != impl_->metadata_credits.end())
        {
            if (credit->second.payload != exclusive_bytes || owner.owner_before(credit->second.owner) ||
                credit->second.owner.owner_before(owner))
                throw GeometryError(GeometryErrorCode::InvalidInput, "External index credit changes owner or size.");
            impl_->last_counter = used;
            return;
        }
        impl_->admit({allocation_detail::Kind::ExternalOwner, owner}, exclusive_bytes, used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

void DynamicGeometryAccess::retain_native_body(const std::shared_ptr<const details::NativeGeometryBody>& body,
    GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!body)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null native source representation admission.");
        ledger.begin_admission(used, limit);
        if (ledger.impl_->mode)
            details::require_mode(*ledger.impl_->mode, ArithmeticMode::Ordinary);
        else
            ledger.impl_->mode = ArithmeticMode::Ordinary;
        ledger.impl_->push(allocation_detail::Kind::NativeSourceBody, body, used, limit);
        ledger.impl_->walk<ArithmeticMode::Ordinary>(used, limit);
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}

template<ArithmeticMode Mode>
void GeometryAllocationLedger::retain(const ConvexGeometryPtr<Mode>& geometry, std::size_t& used, std::size_t limit)
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    try
    {
        if (!geometry)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null geometry allocation admission.");
        details::require_mode(geometry->arithmetic_mode(), Mode);
        begin_admission(used, limit);
        if (impl_->mode)
            details::require_mode(*impl_->mode, Mode);
        else
            impl_->mode = Mode;
        impl_->push(allocation_detail::Kind::TypedWrapper, geometry, used, limit);
        impl_->walk<Mode>(used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

template<ArithmeticMode Mode>
void GeometryAllocationLedger::retain(const DynamicConvexGeometryPtr& geometry, std::size_t& used, std::size_t limit)
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    try
    {
        if (!geometry)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null dynamic geometry allocation admission.");
        details::require_mode(geometry->arithmetic_mode(), Mode);
        begin_admission(used, limit);
        if (impl_->mode)
            details::require_mode(*impl_->mode, Mode);
        else
            impl_->mode = Mode;
        impl_->push(allocation_detail::Kind::DynamicWrapper, geometry, used, limit);
        impl_->walk<Mode>(used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

template<ArithmeticMode Mode>
void DynamicGeometryAccess::retain_prepaid_native_binding(
    const ConvexGeometryPtr<Mode>& geometry, GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
{
    static_assert(Mode == ArithmeticMode::Ordinary);
    try
    {
        if (!geometry)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null prepaid native binding.");
        details::require_mode(geometry->arithmetic_mode(), Mode);
        const auto& dynamic = geometry->dynamic_geometry();
        const auto storage = details::checked_geometry_storage<Mode>(implementation(*dynamic));
        const allocation_detail::NativeBindingPrepaid<Mode> prepaid{
            geometry.get(), dynamic.get(), storage.get(), storage->topology.get()};
        ledger.begin_admission(used, limit);
        if (ledger.impl_->mode)
            details::require_mode(*ledger.impl_->mode, Mode);
        else
            ledger.impl_->mode = Mode;
        ledger.impl_->push(allocation_detail::Kind::TypedWrapper, geometry, used, limit);
        ledger.impl_->walk<Mode>(used, limit, &prepaid);
        ledger.impl_->last_counter = used;
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}
template void DynamicGeometryAccess::retain_prepaid_native_binding<ArithmeticMode::Ordinary>(
    const ConvexGeometryPtr<ArithmeticMode::Ordinary>&, GeometryAllocationLedger&, std::size_t&, std::size_t);

template<ArithmeticMode Mode>
void GeometryAllocationLedger::retain_snapshot(
    const std::shared_ptr<const PointSnapshotContextBase>& context, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!context || !context->frozen)
            throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Only frozen point snapshots may be admitted.");
        details::require_mode(context->mode, Mode);
        begin_admission(used, limit);
        if (impl_->mode)
            details::require_mode(*impl_->mode, Mode);
        else
            impl_->mode = Mode;
        impl_->push(allocation_detail::Kind::PointSnapshot, context, used, limit);
        impl_->walk<Mode>(used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}
template void GeometryAllocationLedger::retain_snapshot<ArithmeticMode::Ordinary>(
    const std::shared_ptr<const PointSnapshotContextBase>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain_snapshot<ArithmeticMode::BoundedCertification>(
    const std::shared_ptr<const PointSnapshotContextBase>&, std::size_t&, std::size_t);

template<ArithmeticMode Mode>
void DynamicGeometryAccess::retain_geometry_storage(const std::shared_ptr<const DynamicConvexGeometry::Impl>& storage,
    GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!storage)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null geometry storage allocation admission.");
        details::require_mode(storage->limits.arithmetic_mode, Mode);
        ledger.begin_admission(used, limit);
        if (ledger.impl_->mode)
            details::require_mode(*ledger.impl_->mode, Mode);
        else
            ledger.impl_->mode = Mode;
        ledger.impl_->push(allocation_detail::Kind::Geometry, storage, used, limit);
        ledger.impl_->walk<Mode>(used, limit);
        ledger.impl_->last_counter = used;
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}
template void DynamicGeometryAccess::retain_geometry_storage<ArithmeticMode::Ordinary>(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&, GeometryAllocationLedger&, std::size_t&, std::size_t);
template void DynamicGeometryAccess::retain_geometry_storage<ArithmeticMode::BoundedCertification>(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&, GeometryAllocationLedger&, std::size_t&, std::size_t);

void GeometryAllocationLedger::retain_point_provenance(const DynamicPoint& point, std::size_t& used, std::size_t limit)
{
    try
    {
        const auto& context = point.provenance_.context_pin();
        if (!context)
        {
            begin_admission(used, limit, false);
            return;
        }
        if (context->mode == ArithmeticMode::Ordinary)
            retain_snapshot<ArithmeticMode::Ordinary>(context, used, limit);
        else
            retain_snapshot<ArithmeticMode::BoundedCertification>(context, used, limit);
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

void GeometryAllocationLedger::retain(const DynamicPointExport& points, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!points.payload_)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Empty dynamic point export admission.");
        begin_admission(used, limit);
        const auto mode = points.payload_->mode;
        if (impl_->mode)
            details::require_mode(*impl_->mode, mode);
        else
            impl_->mode = mode;
        impl_->push(allocation_detail::Kind::DynamicPointExport, points.payload_, used, limit);
        if (mode == ArithmeticMode::Ordinary)
            impl_->walk<ArithmeticMode::Ordinary>(used, limit);
        else
            impl_->walk<ArithmeticMode::BoundedCertification>(used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

template<ArithmeticMode Mode>
void GeometryAllocationLedger::retain(const PointExport<Mode>& points, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!points.payload_)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Empty typed point export admission.");
        begin_admission(used, limit);
        if (impl_->mode)
            details::require_mode(*impl_->mode, Mode);
        else
            impl_->mode = Mode;
        impl_->push(allocation_detail::Kind::TypedPointExport, points.payload_, used, limit);
        impl_->walk<Mode>(used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}
template void GeometryAllocationLedger::retain<ArithmeticMode::Ordinary>(
    const PointExport<ArithmeticMode::Ordinary>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<ArithmeticMode::BoundedCertification>(
    const PointExport<ArithmeticMode::BoundedCertification>&, std::size_t&, std::size_t);

template<ArithmeticMode Mode>
void GeometryAllocationLedger::retain(const PointSnapshot<Mode>& points, std::size_t& used, std::size_t limit)
{
    try
    {
        begin_admission(used, limit, bool(points.payload_));
        if (!points.payload_)
            return;
        details::require_mode(points.payload_->context->mode, Mode);
        if (impl_->mode)
            details::require_mode(*impl_->mode, Mode);
        else
            impl_->mode = Mode;
        impl_->push(allocation_detail::Kind::CompactPointBatch, points.payload_, used, limit);
        impl_->walk<Mode>(used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}
template void GeometryAllocationLedger::retain<ArithmeticMode::Ordinary>(
    const PointSnapshot<ArithmeticMode::Ordinary>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<ArithmeticMode::BoundedCertification>(
    const PointSnapshot<ArithmeticMode::BoundedCertification>&, std::size_t&, std::size_t);

void GeometryAllocationLedger::retain(const PlaneAuthority& authority, std::size_t& used, std::size_t limit)
{
    try
    {
        begin_admission(used, limit, !authority.empty());
        if (!impl_)
            return;
        impl_->retain_authority(authority, used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

template<GeometryIdentityKind Kind, ArithmeticMode Mode>
    requires(GeometryIdentity<Kind, Mode>::shared_storage)
void GeometryAllocationLedger::retain(
    const GeometryIdentity<Kind, Mode>& identity, std::size_t& used, std::size_t limit)
{
    try
    {
        begin_admission(used, limit, identity.storage_identity() != nullptr);
        if (!impl_)
            return;
        impl_->retain_identity(identity, used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}

template void GeometryAllocationLedger::retain<ArithmeticMode::Ordinary>(
    const ConvexGeometryPtr<ArithmeticMode::Ordinary>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<ArithmeticMode::BoundedCertification>(
    const ConvexGeometryPtr<ArithmeticMode::BoundedCertification>&, std::size_t&, std::size_t);

template void GeometryAllocationLedger::retain<ArithmeticMode::Ordinary>(
    const DynamicConvexGeometryPtr&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<ArithmeticMode::BoundedCertification>(
    const DynamicConvexGeometryPtr&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<GeometryIdentityKind::PlaneSupport, ArithmeticMode::Ordinary>(
    const PlaneSupportIdentity<ArithmeticMode::Ordinary>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<GeometryIdentityKind::Point, ArithmeticMode::Exact>(
    const PointIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<GeometryIdentityKind::Line, ArithmeticMode::Exact>(
    const LineIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<GeometryIdentityKind::PlaneSupport, ArithmeticMode::Exact>(
    const PlaneSupportIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<GeometryIdentityKind::Displacement, ArithmeticMode::Exact>(
    const DisplacementIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);

namespace details
{
template<ArithmeticMode Mode>
std::size_t geometry_allocation_footprint(const std::shared_ptr<const DynamicConvexGeometry::Impl>& geometry)
{
    require_mode(geometry->limits.arithmetic_mode, Mode);
    GeometryAllocationLedger::Impl accounting;
    accounting.mode = Mode;
    accounting.scratch_limit = geometry->limits.byte_budget;
    std::size_t used = 0;
    accounting.account(sizeof(accounting), true, used, std::numeric_limits<std::size_t>::max());
    accounting.push(allocation_detail::Kind::Geometry, geometry, used, std::numeric_limits<std::size_t>::max());
    accounting.walk<Mode>(used, std::numeric_limits<std::size_t>::max());
    return accounting.payload;
}
template std::size_t geometry_allocation_footprint<ArithmeticMode::Ordinary>(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&);
template std::size_t geometry_allocation_footprint<ArithmeticMode::BoundedCertification>(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&);
} // namespace details

std::size_t details::GeometryOwnerImportAccess::import(GeometryAllocationLedger& ledger,
    const geometry_demand::GeometryOwnerManifest& owners, std::size_t& cumulative_bytes, std::size_t byte_limit)
{
    GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::Import);
    try
    {
        ledger.begin_admission(cumulative_bytes, byte_limit);
        const auto channel = owners.uniform_channel();
        if (channel)
        {
            if (*channel > 1)
                throw GeometryError(GeometryErrorCode::InvalidInput, "Unknown geometry owner arithmetic channel.");
            const auto mode = *channel ? ArithmeticMode::BoundedCertification : ArithmeticMode::Ordinary;
            if (ledger.impl_->mode)
                details::require_mode(*ledger.impl_->mode, mode);
            else
                ledger.impl_->mode = mode;
        }
        if (ledger.impl_->owners.empty() && !ledger.impl_->imported)
        {
            // Capture shares an already admitted manifest index. Its retained
            // total remains complete while only new ledger storage is charged.
            if (ledger.impl_->prepaid_payload)
                ledger.impl_->bookkeeping =
                    allocation_detail::add(ledger.impl_->bookkeeping, owners.metadata_admission_bytes());
            else
                ledger.impl_->account(owners.metadata_admission_bytes(), true, cumulative_bytes, byte_limit);
            ledger.impl_->account(owners.payload_bytes(), false, cumulative_bytes, byte_limit);
            ledger.impl_->imported = owners;
            ledger.impl_->imported_channel = channel.value_or(0);
            ledger.impl_->last_counter = cumulative_bytes;
            return 0;
        }
        // A populated ledger needs the exact union, including overlap with an
        // earlier immutable import. Any failure below poisons partial rows.
        std::size_t enumerated = 0;
        owners.visit(
            [&](const geometry_demand::OwnerRecord& row)
            {
                GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ImportRow);
                ++enumerated;
                const auto kind = static_cast<allocation_detail::Kind>(row.key.kind);
                if (kind == allocation_detail::Kind::ExternalOwner)
                {
                    // Only an unknown external destination can alias imported
                    // index bookkeeping. Validate known payload credits first,
                    // preserving no-op imports and reject-before-spend behavior.
                    const allocation_detail::Key key{kind, row.owner.get()};
                    const bool imported =
                        ledger.impl_->imported &&
                        ledger.impl_->imported->references(ledger.impl_->imported_channel, row.key.kind, row.owner);
                    if (!imported && !ledger.impl_->owners.contains(key) &&
                        !ledger.impl_->metadata_credits.contains(key))
                        details::GeometryAllocationOwnerAccess::seed_imported_metadata(
                            ledger, cumulative_bytes, byte_limit);
                }
                (void) ledger.impl_->admit(
                    {kind, row.owner}, row.exclusive_payload_bytes(), cumulative_bytes, byte_limit);
                ledger.impl_->last_counter = cumulative_bytes;
            });
        return enumerated;
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}

geometry_demand::GeometryOwnerManifest details::GeometryOwnerImportAccess::export_manifest(
    const GeometryAllocationLedger& ledger, geometry_demand::Admission admission, geometry_demand::PageWork* work,
    std::uint8_t channel)
{
    GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::Export);
    using geometry_demand::OwnerRecord;
    if (ledger.failed_)
        allocation_detail::exhausted("Cannot export a rejected geometry owner census.");
    if (!ledger.impl_)
        return {};
    if (channel > 1 || (ledger.impl_->mode && *ledger.impl_->mode != (channel ? ArithmeticMode::BoundedCertification
                                                                              : ArithmeticMode::Ordinary)))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Geometry owner export changes its arithmetic channel.");
    // The contiguous construction buffer overlaps the source ledger and the
    // persistent pages. Admit all of it before the first heap allocation.
    const auto count = ledger.owner_count();
    admission.charge(count, sizeof(OwnerRecord));
    std::vector<OwnerRecord> rows;
    rows.reserve(count);
    if (rows.capacity() != count)
        allocation_detail::exhausted("Geometry owner export row capacity is unqualified.");
    if (ledger.impl_->imported)
        ledger.impl_->imported->visit(
            [&](const OwnerRecord& row)
            {
                GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ExportRow);
                rows.push_back({{channel, row.key.kind, row.owner.get()}, row.owner, row.bytes, row.references});
            });
    for (const auto& [key, entry] : ledger.impl_->owners)
    {
        GeometryAllocationOwnerAccess::observe_work(GeometryAllocationOwnerAccess::Work::ExportRow);
        rows.push_back(
            {{channel, static_cast<std::uint8_t>(key.kind), entry.pin.get()}, entry.pin, entry.payload_bytes, 1});
    }
    std::sort(rows.begin(), rows.end(),
        [](const OwnerRecord& first, const OwnerRecord& second)
        {
            return first.key.kind != second.key.kind ? first.key.kind < second.key.kind
                                                     : std::less<const void*>{}(first.key.address, second.key.address);
        });
    return geometry_demand::GeometryOwnerManifest::built(rows, admission, work);
}

std::size_t details::GeometryOwnerImportAccess::imported_owners(const GeometryAllocationLedger& ledger) noexcept
{
    return ledger.impl_ && ledger.impl_->imported ? ledger.impl_->imported->owner_count() : 0;
}

void details::GeometryOwnerImportAccess::visit_imported(
    const GeometryAllocationLedger& ledger, const std::function<void(std::uint8_t, const void*, std::size_t)>& visitor)
{
    if (ledger.failed_)
        allocation_detail::exhausted("Cannot inspect rejected geometry owner import.");
    if (!ledger.impl_ || !ledger.impl_->imported)
        return;
    ledger.impl_->imported->visit([&](const geometry_demand::OwnerRecord& row)
        { visitor(row.key.kind, row.owner.get(), row.exclusive_payload_bytes()); });
}
} // namespace SimpleFluid::PLIC
