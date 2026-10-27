/** @file OrdinaryCurvature.hh
 * @brief Connected catalog fits and canonical owner publication for Ordinary capillarity.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "FVM/GeneralVoFSurface.hh"
#include "SimpleFluidExport.hh"
#include <vector>

namespace SimpleFluid::detail
{
class GeneralVoFFlowWorkspace;
}
namespace SimpleFluid::detail::general_vof
{
/** Collective, including empty owners. The geometry-only catalog never changes
 * accepted phase inventories. Returned values follow native owned-cell order.
 * Diagnostics publish only after fits, halo exchange and extension all succeed.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL std::vector<double> assemble_ordinary_curvature(const MeshHandle<>&,
    const GeneralVoFFlowWorkspace&, const VoF::GeneralVoF&, const VoF::GeneralVoFSurfaceOptions&,
    VoF::GeneralVoFCapillaryDiagnostics*, std::size_t);
} // namespace SimpleFluid::detail::general_vof
