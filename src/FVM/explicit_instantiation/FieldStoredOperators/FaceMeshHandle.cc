/** @brief Compiled mapped face provider for MeshHandle. */
#include "FVM/FieldStoredFaceFlux.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = MeshHandle<Pack>;
} // namespace

template FieldStoredVelocityBoundaryCache<Pack, MeshType> cache_velocity_boundary_conditions<Pack, MeshType>(
    SP<const MeshType> mesh, const BoundaryConditionSet& boundary_conditions);

template void face_velocities<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, VectorFaceFieldStored<Pack, MeshType>& face_velocity);

template void face_velocities<Pack, MeshType>(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    VectorFaceFieldStored<Pack, MeshType>& face_velocity);

template void normal_face_fluxes<Pack, MeshType>(
    const VectorFaceFieldStored<Pack, MeshType>& face_velocity, ScalarFaceFieldStored<Pack, MeshType>& fluxes);

template void face_fluxes<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, ScalarFaceFieldStored<Pack, MeshType>& fluxes);

template void face_fluxes<Pack, MeshType>(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

template void mesh_relative_face_fluxes<Pack, MeshType>(const ScalarFaceFieldStored<Pack, MeshType>& absolute_fluxes,
    const ALEControlVolumeState& ale, ScalarFaceFieldStored<Pack, MeshType>& relative_fluxes);

} // namespace SimpleFluid::FVM
