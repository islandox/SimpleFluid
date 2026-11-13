/** @file GeometryAllocationOwnerAccess.hh
 * @brief Internal scoped census of admitted geometry allocation owners.
 */
#pragma once
#include "geometry/PLIC/GeometryAllocationLedger.hh"

#include <cstdint>
#include <functional>

namespace SimpleFluid::PLIC::details
{
/** Restricts a fresh ledger's bookkeeping independently of its payload cap.
 * The existing retain calls still check complete payload size and ownership.
 * Borrowed identities remain valid while the ledger pins the allocations. */
class GeometryAllocationOwnerAccess
{
public:
    static void begin(GeometryAllocationLedger&, std::size_t& cumulative, std::size_t bookkeeping_limit);
    /** Capture a fully prepaid payload graph under the caller's finite budget.
     * Every newly constructed ledger map/scratch allocation is still charged;
     * exclusive payload totals and strong owner pins remain complete. Imported
     * manifests must also have prepaid their shared index metadata. */
    static void begin_prepaid_capture(GeometryAllocationLedger&, std::size_t& cumulative, std::size_t byte_limit);
    static void visit(const GeometryAllocationLedger&,
        const std::function<void(std::uint8_t kind, const void* owner, std::size_t bytes)>&);
    /** Record an immutable owner whose allocation was already charged in the
     * same cumulative operation. Only ledger bookkeeping is charged here; a
     * later normal retain of the same owner then deduplicates instead of
     * charging the identical persistent bytes twice. */
    static void admit_prepaid(GeometryAllocationLedger&, const std::shared_ptr<const void>& owner,
        std::size_t exclusive_bytes, std::size_t& used, std::size_t limit);
};
} // namespace SimpleFluid::PLIC::details
