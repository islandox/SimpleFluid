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
struct GeometryAllocationWork
{
    std::size_t pushes = 0, admissions = 0, pending_visits = 0;
    std::size_t import_calls = 0, import_rows = 0, export_calls = 0, export_rows = 0, manifest_visit_rows = 0;
};
class GeometryAllocationOwnerAccess
{
public:
    enum class Work
    {
        Push,
        Admit,
        PendingVisit,
        Import,
        ImportRow,
        Export,
        ExportRow,
        ManifestVisitRow
    };
    /** Thread-local actual event counts; separate from logical retained owner counts. */
    [[nodiscard]] static GeometryAllocationWork work_statistics() noexcept;
    static void reset_work_statistics() noexcept;
    static void observe_work(Work, std::size_t count = 1) noexcept;
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
