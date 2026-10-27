/** @file CapillaryPlanes.hh
 * @brief Detached native-face plane witnesses for capillary geometry only.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
/** Collective owner selection and sparse import. Original transport planes,
 * accepted inventories, histories, declarations and geometry remain immutable.
 * An incomplete or ambiguous witness retains the original plane. */
SIMPLEFLUID_FVM_LOCAL std::map<Key, PLIC::PolyhedralInterface> import_capillary_planes(
    const GeneralVoF::Data&, const std::map<Key, PLIC::PolyhedralInterface>& original);
} // namespace SimpleFluid::VoF::general_vof_detail
