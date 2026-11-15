/**
 * @file ALEControlVolumeState.cc
 * @brief  Explicit ALE validation instantiations.
 */
#include "utils/export_helper.hh"

#include "FVM/ALEControlVolumeState.tcc"
#include "geometry/MeshHandle.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/SolidSubdomain.hh"

namespace SimpleFluid::FVM
{
template void ALEControlVolumeState::validate(const MeshHandle<DefaultTpetraTypes>&) const;
template void ALEControlVolumeState::validate(const SolidSubdomain<DefaultTpetraTypes>&) const;
} // namespace SimpleFluid::FVM
