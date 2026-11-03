#pragma once
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.hh"
#include "geometry/PLIC/details/PointSnapshot.hh"
namespace SimpleFluid::PLIC::details
{
/** Raw borrows only. The caller holds every enclosing root for the whole probe. */
class RetentionAccess
{
public:
    static const GeometryAllocationLedger::Impl* ledger(const GeometryAllocationLedger& p) noexcept
    {
        return p.impl_.get();
    }
    static bool failed(const GeometryAllocationLedger& p) noexcept { return p.failed_; }
    static PointProvenanceView provenance(const DynamicPoint& p) { return p.provenance_.view(); }
    template<ArithmeticMode M> static PointProvenanceView provenance(const Point<M>& p) { return p.provenance_.view(); }
    static const DynamicPointExportPayload* payload(const DynamicPointExport& p) noexcept { return p.payload_.get(); }
    template<ArithmeticMode M> static const PointExportPayload<M>* payload(const PointExport<M>& p) noexcept
    {
        return p.payload_.get();
    }
    template<ArithmeticMode M> static const PointSnapshotPayload<M>* payload(const PointSnapshot<M>& p) noexcept
    {
        return p.payload_.get();
    }
    static const std::vector<std::uint64_t>* opaque(const PlaneAuthority& p) noexcept
    {
        const auto* value = std::get_if<PlaneAuthority::Opaque>(&p.storage_);
        return value ? value->get() : nullptr;
    }
    static const ConvexCellGeometry<ArithmeticMode::Ordinary>* boundary_geometry(const OrdinaryBoundary& p) noexcept
    {
        return p.boundary_.geometry.get();
    }
    static const PlaneAuthority& boundary_authority(const OrdinaryBoundary& p) noexcept { return p.native_authority_; }
    static std::size_t planar_metadata_bytes() noexcept;
    static const PlaneSupportIdentity<ArithmeticMode::Ordinary>* planar_support_borrowed(
        const PlanarFace<ArithmeticMode::Ordinary>&) noexcept;
};
} // namespace SimpleFluid::PLIC::details
