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
    static void visit(const GeometryAllocationLedger&,
        const std::function<void(std::uint8_t kind, const void* owner, std::size_t bytes)>&);
};
} // namespace SimpleFluid::PLIC::details
