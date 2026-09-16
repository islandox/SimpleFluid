/** @file ALEControlVolumeState.cc @brief Explicit ALE validation instantiations. */
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
