/** @file Reconstruction.hh
 * @brief Private observed Ordinary owner reconstruction used by sparse plane imports.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
SIMPLEFLUID_FVM_LOCAL PLIC::PolyhedralInterface reconstruct_observed_ordinary_cell(const GeneralVoF::Data& data,
    const Key& key, GeneralVoFOwnerPlaneRequestDiagnostics* diagnostics, GeneralVoFWorkCell* row);
} // namespace SimpleFluid::VoF::general_vof_detail
