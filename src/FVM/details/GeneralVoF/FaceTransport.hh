/** @file FaceTransport.hh
 * @brief Admitted donor-cell support without candidate publication.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
/** Private numerical baseline. Receipts and endpoint capacities have not been
 * narrowed into candidate inventory arrays or imported into candidate ghosts.
 * All donor, cell and independent global phase admission gates have run. */
struct AdmittedLowOrderLedger
{
    std::vector<GeneralVoFDirectedStream> streams;
    std::vector<GeneralVoFFiniteTransfer> transfers;
    std::vector<GeneralVoFFiniteDonorBudget> budgets;
    std::map<std::uint64_t, std::array<long double, 3>> endpoint_capacity;
    GeneralVoFTransportDiagnostics diagnostics;
};
SIMPLEFLUID_FVM_LOCAL AdmittedLowOrderLedger admit_low_order_ledger(
    const GeneralVoF::Data&, std::span<const GeneralVoFFaceVolume>, double timestep);
} // namespace SimpleFluid::VoF::general_vof_detail
