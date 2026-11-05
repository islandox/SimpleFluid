/** @file GeometryAllocationLedger.cc
 * @brief Alias-aware immutable geometry ownership traversal and cumulative admission.
 */
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/GeometryAllocationLedgerInternal.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.tcc"

namespace SimpleFluid::PLIC
{
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
    return impl_ ? impl_->owners.size() : 0;
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

void GeometryAllocationLedger::retain_external_owner(const std::shared_ptr<const void>& owner,
    std::size_t exclusive_bytes, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!owner)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null external allocation owner.");
        begin_admission(used, limit);
        const auto found = impl_->owners.find({allocation_detail::Kind::ExternalOwner, owner.get()});
        if (found != impl_->owners.end() && found->second.payload_bytes != exclusive_bytes)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Immutable external allocation size changed.");
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
} // namespace SimpleFluid::PLIC
