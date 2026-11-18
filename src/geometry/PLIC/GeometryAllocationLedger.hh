/** @file GeometryAllocationLedger.hh
 * @brief Cumulative admission of unique immutable finite-geometry payloads.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"

namespace SimpleFluid::PLIC
{
template<ArithmeticMode> struct PointSnapshotContext;
class CartesianCellSource;
template<ArithmeticMode> class CellGeometryCatalog;
namespace details
{
class GeometryAllocationOwnerAccess;
class GeometryOwnerImportAccess;
class GeometryOwnerUnion;
class GeometryOwnerAdmissionPair;
class NativeGeometryAccounting;
} // namespace details
/** Counts immutable owners and their exclusively owned buffers once. Strong
 * pins prevent allocator address reuse from hiding later allocations. Rejected
 * admission poisons the ledger; create a new ledger for a new transaction.
 * These are retained-construction accounting bytes, not allocator/RSS totals. */
class GeometryAllocationLedger
{
public:
    struct Impl;
    GeometryAllocationLedger() noexcept;
    ~GeometryAllocationLedger();
    GeometryAllocationLedger(GeometryAllocationLedger&&) noexcept;
    GeometryAllocationLedger& operator=(GeometryAllocationLedger&&) noexcept;
    GeometryAllocationLedger(const GeometryAllocationLedger&) = delete;
    GeometryAllocationLedger& operator=(const GeometryAllocationLedger&) = delete;
    template<ArithmeticMode Mode>
    void retain(const ConvexGeometryPtr<Mode>&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    template<ArithmeticMode Mode>
    void retain(const DynamicConvexGeometryPtr&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    template<ArithmeticMode Mode>
    void retain_snapshot(
        const std::shared_ptr<const PointSnapshotContextBase>&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    template<ArithmeticMode Mode>
    void retain_snapshot(
        const std::shared_ptr<const PointSnapshotContext<Mode>>& context, std::size_t& used, std::size_t limit)
    {
        retain_snapshot<Mode>(std::static_pointer_cast<const PointSnapshotContextBase>(context), used, limit);
    }
    /** Admit only physically held provenance owners; the caller owns/counts
     * the point object or buffer. Mutation invalidates authority, not storage. */
    void retain_point_provenance(const DynamicPoint&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    template<ArithmeticMode Mode>
    void retain_point_provenance(const Point<Mode>& point, std::size_t& used, std::size_t limit)
    {
        retain_point_provenance(point.dynamic(), used, limit);
    }
    void retain(const DynamicPointExport&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    template<ArithmeticMode Mode>
    void retain(const PointExport<Mode>&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    template<ArithmeticMode Mode>
    void retain(const PointSnapshot<Mode>&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    /** Admit a shared opaque authority payload or the complete typed support
     * ancestry. Content equality does not merge independently owned handles. */
    void retain(const PlaneAuthority&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    /** Shared identity owners join the same admission scope as geometry.
     * Content equality never deduplicates independently allocated owners. */
    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
        requires(GeometryIdentity<Kind, Mode>::shared_storage)
    void retain(const GeometryIdentity<Kind, Mode>&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    template<ArithmeticMode Mode>
        requires(Mode != ArithmeticMode::Exact)
    void retain(
        const std::shared_ptr<const CellGeometryCatalog<Mode>>&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    /** Admit one immutable procedural context and its shared original supports. */
    void retain(
        const std::shared_ptr<const CartesianCellSource>&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    /** Admit detached immutable application metadata by actual owner identity.
     * The byte count covers exclusively owned storage; admit child owners separately. */
    void retain_external_owner(const std::shared_ptr<const void>&, std::size_t exclusive_bytes,
        std::size_t& cumulative_bytes, std::size_t byte_limit);
    /** Admit one Ordinary face shell, its immutable polygon owner and support
     * ancestry by physical allocation identity shared with retention probes. */
    void retain_planar_face(const std::shared_ptr<const PlanarFace<ArithmeticMode::Ordinary>>&,
        std::size_t& cumulative_bytes, std::size_t byte_limit);
    [[nodiscard]] std::size_t owner_count() const noexcept;
    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t bookkeeping_bytes() const noexcept;

private:
    friend class DynamicGeometryAccess;
    friend class details::RetentionAccess;
    friend class details::GeometryAllocationOwnerAccess;
    friend class details::GeometryOwnerImportAccess;
    friend class details::GeometryOwnerUnion;
    friend class details::GeometryOwnerAdmissionPair;
    void begin_admission(std::size_t&, std::size_t, bool allocate_state = true);
    struct ImplDelete
    {
        // Type erasure keeps the private native allocator out of the public ABI.
        std::shared_ptr<const void> native_state;
        void operator()(Impl*) const noexcept;
    };
    std::unique_ptr<Impl, ImplDelete> impl_;
    bool failed_ = false;
};
namespace details
{
/** Identity-only admission uses no pending graph vector. Fixed-shape
 * preflights include the ledger object and ordered owner-node allocations. */
[[nodiscard]] std::size_t identity_ledger_bookkeeping_bound(std::size_t owners);
} // namespace details

extern template void GeometryAllocationLedger::retain<ArithmeticMode::Ordinary>(
    const ConvexGeometryPtr<ArithmeticMode::Ordinary>&, std::size_t&, std::size_t);
extern template void GeometryAllocationLedger::retain<ArithmeticMode::BoundedCertification>(
    const ConvexGeometryPtr<ArithmeticMode::BoundedCertification>&, std::size_t&, std::size_t);
extern template void GeometryAllocationLedger::retain<ArithmeticMode::Ordinary>(
    const DynamicConvexGeometryPtr&, std::size_t&, std::size_t);
extern template void GeometryAllocationLedger::retain<ArithmeticMode::BoundedCertification>(
    const DynamicConvexGeometryPtr&, std::size_t&, std::size_t);
extern template void GeometryAllocationLedger::retain<GeometryIdentityKind::PlaneSupport, ArithmeticMode::Ordinary>(
    const PlaneSupportIdentity<ArithmeticMode::Ordinary>&, std::size_t&, std::size_t);
extern template void GeometryAllocationLedger::retain<GeometryIdentityKind::Point, ArithmeticMode::Exact>(
    const PointIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);
extern template void GeometryAllocationLedger::retain<GeometryIdentityKind::Line, ArithmeticMode::Exact>(
    const LineIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);
extern template void GeometryAllocationLedger::retain<GeometryIdentityKind::PlaneSupport, ArithmeticMode::Exact>(
    const PlaneSupportIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);
extern template void GeometryAllocationLedger::retain<GeometryIdentityKind::Displacement, ArithmeticMode::Exact>(
    const DisplacementIdentity<ArithmeticMode::Exact>&, std::size_t&, std::size_t);
} // namespace SimpleFluid::PLIC
