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
    [[nodiscard]] std::size_t owner_count() const noexcept;
    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t bookkeeping_bytes() const noexcept;

private:
    std::unique_ptr<Impl> impl_;
    bool failed_ = false;
};

extern template void GeometryAllocationLedger::retain<ArithmeticMode::Ordinary>(
    const ConvexGeometryPtr<ArithmeticMode::Ordinary>&, std::size_t&, std::size_t);
extern template void GeometryAllocationLedger::retain<ArithmeticMode::BoundedCertification>(
    const ConvexGeometryPtr<ArithmeticMode::BoundedCertification>&, std::size_t&, std::size_t);
} // namespace SimpleFluid::PLIC
