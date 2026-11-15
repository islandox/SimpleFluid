/**
 * @file ALEControlVolumeState.cc
 * @brief  Explicit ALE validation instantiations.
 */
#include "utils/export_helper.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "FVM/ALEControlVolumeState.tcc"
#include "geometry/Mesh.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/SolidSubdomain.hh"

namespace SimpleFluid::FVM
{
template void ALEControlVolumeState::validate(const Mesh<DefaultTpetraTypes>&) const;
template void ALEControlVolumeState::validate(const MeshHandle<DefaultTpetraTypes>&) const;
template void ALEControlVolumeState::validate(const SolidSubdomain<DefaultTpetraTypes>&) const;
} // namespace SimpleFluid::FVM

SIMPLEFLUID_LEGACY_API_END
