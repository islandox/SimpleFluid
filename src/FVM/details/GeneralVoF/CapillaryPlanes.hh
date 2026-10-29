/** @file CapillaryPlanes.hh
 * @brief Detached native-face plane witnesses for capillary geometry only.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
/** Collective full-support owner reconstruction and sparse import. Original
 * transport planes, inventories, histories, declarations and geometry remain
 * immutable. Adopted geometry remains authoritative; an inadmissible or ambiguous
 * native-face witness retains the full-support capillary plane. */
SIMPLEFLUID_FVM_LOCAL std::map<Key, PLIC::PolyhedralInterface> import_capillary_planes(const GeneralVoF::Data&);
} // namespace SimpleFluid::VoF::general_vof_detail
