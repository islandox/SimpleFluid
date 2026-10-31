/** @file GeometryAllocationLedger.hh
 * @brief Cumulative admission of unique immutable finite-geometry payloads.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"

namespace SimpleFluid::PLIC
{
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
    void retain(const DynamicPointExport&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    template<ArithmeticMode Mode>
    void retain(const PointExport<Mode>&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    /** Admit a shared opaque authority payload or the complete typed support
     * ancestry. Content equality does not merge independently owned handles. */
    void retain(const PlaneAuthority&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    /** Shared identity owners join the same admission scope as geometry.
     * Content equality never deduplicates independently allocated owners. */
    template<GeometryIdentityKind Kind, ArithmeticMode Mode>
        requires(GeometryIdentity<Kind, Mode>::shared_storage)
    void retain(const GeometryIdentity<Kind, Mode>&, std::size_t& cumulative_bytes, std::size_t byte_limit);
    [[nodiscard]] std::size_t owner_count() const noexcept;
    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t bookkeeping_bytes() const noexcept;

private:
    friend class DynamicGeometryAccess;
    void begin_admission(std::size_t&, std::size_t, bool allocate_state = true);
    std::unique_ptr<Impl> impl_;
    bool failed_ = false;
};

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
