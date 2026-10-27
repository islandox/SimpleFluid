/** @file FilteredSurface.hh
 * @brief Read-only, finite-support binary capillary geometry.
 */
#pragma once
#include "FVM/GeneralVoFSurface.hh"
#include "FVM/details/GeneralVoF/CapillaryNodeFans.hh"
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
// Local builder. The caller owns collective admission/failure synchronization.
SIMPLEFLUID_FVM_LOCAL GeneralVoFSurfaceCatalog filtered_binary_surface(const GeneralVoF::Data&,
    const GeneralVoFSurfaceOptions&, const CapillaryNodeFans&, std::size_t remaining_budget = 0);
} // namespace SimpleFluid::VoF::general_vof_detail
