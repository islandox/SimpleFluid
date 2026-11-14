/** @brief Geometry and ALE implementation provider for the generic reference mesh. */
#include "FVM/ALEControlVolumeState.tcc"
#include "TestMeshes.hh"
#include "geometry/MeshTraverse.tcc"

namespace SimpleFluid
{
template class MeshTraversePlan<test::transport_providers::ReferenceHandle>;
} // namespace SimpleFluid
namespace SimpleFluid::FVM
{
template class TransportGeometryCache<test::transport_providers::ReferenceHandle>;
template void ALEControlVolumeState::validate(const test::transport_providers::ReferenceHandle&) const;
} // namespace SimpleFluid::FVM
