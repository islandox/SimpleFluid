/** @brief Compiled mapped face provider for MeshHandle. */
#include "FVM/FieldStoredFaceFlux.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Mesh = MeshHandle<Pack>;
} // namespace

template FieldStoredVelocityBoundaryCache<Pack, Mesh> cache_velocity_boundary_conditions<Pack, Mesh>(
    SP<const Mesh> mesh, const BoundaryConditionSet& boundary_conditions);

template void face_velocities<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, VectorFaceFieldStored<Pack, Mesh>& face_velocity);

template void face_velocities<Pack, Mesh>(const VectorCellFieldStored<Pack, Mesh>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    VectorFaceFieldStored<Pack, Mesh>& face_velocity);

template void normal_face_fluxes<Pack, Mesh>(
    const VectorFaceFieldStored<Pack, Mesh>& face_velocity, ScalarFaceFieldStored<Pack, Mesh>& fluxes);

template void face_fluxes<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, ScalarFaceFieldStored<Pack, Mesh>& fluxes);

template void face_fluxes<Pack, Mesh>(const VectorCellFieldStored<Pack, Mesh>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes);

template void mesh_relative_face_fluxes<Pack, Mesh>(const ScalarFaceFieldStored<Pack, Mesh>& absolute_fluxes,
    const ALEControlVolumeState& ale, ScalarFaceFieldStored<Pack, Mesh>& relative_fluxes);

} // namespace SimpleFluid::FVM
