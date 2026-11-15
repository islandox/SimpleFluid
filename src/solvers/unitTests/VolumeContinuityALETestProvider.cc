/** @file VolumeContinuityALETestProvider.cc
 * @brief Executable-local ALE validation for the continuity observation mesh.
 */
#include "VolumeContinuityObservedMesh.hh"
#include "FVM/ALEControlVolumeState.tcc"

namespace SimpleFluid::FVM
{

template void ALEControlVolumeState::validate(const test::VolumeContinuityObservedMesh&) const;

} // namespace SimpleFluid::FVM
