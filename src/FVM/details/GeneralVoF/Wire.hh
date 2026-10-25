/** @file Wire.hh
 * @brief Shared phase records; transport/remap own their exchange protocols.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
SIMPLEFLUID_FVM_LOCAL void finite_phases(wire::Writer&, const PLIC::DynamicPhaseMeasures&);
SIMPLEFLUID_FVM_LOCAL PLIC::DynamicPhaseMeasures finite_phases(wire::Reader&, ArithmeticMode);
} // namespace SimpleFluid::VoF::general_vof_detail
