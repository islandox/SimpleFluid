/** @brief Compiled mapped face provider for PartitionedCartesian. */
#include "FVM/FieldStoredFaceFlux.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"

// Only these public static-mesh specializations override hidden argument
// visibility. GNU attributes on instantiations are accepted by GCC and Clang;
// standard [[gnu::visibility]] attributes are not valid there in Clang.
#if defined(__GNUC__) || defined(__clang__)
#define SIMPLEFLUID_STATIC_OPERATOR_EXPORT __attribute__((visibility("default")))
#else
#define SIMPLEFLUID_STATIC_OPERATOR_EXPORT
#endif

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Mesh = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
} // namespace

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT FieldStoredVelocityBoundaryCache<Pack, Mesh>
cache_velocity_boundary_conditions<Pack, Mesh>(
    SP<const Mesh> mesh, const BoundaryConditionSet& boundary_conditions);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void face_velocities<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, VectorFaceFieldStored<Pack, Mesh>& face_velocity);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void face_velocities<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    VectorFaceFieldStored<Pack, Mesh>& face_velocity);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void normal_face_fluxes<Pack, Mesh>(
    const VectorFaceFieldStored<Pack, Mesh>& face_velocity, ScalarFaceFieldStored<Pack, Mesh>& fluxes);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void face_fluxes<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, ScalarFaceFieldStored<Pack, Mesh>& fluxes);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void face_fluxes<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void mesh_relative_face_fluxes<Pack, Mesh>(
    const ScalarFaceFieldStored<Pack, Mesh>& absolute_fluxes, const ALEControlVolumeState& ale,
    ScalarFaceFieldStored<Pack, Mesh>& relative_fluxes);

} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_OPERATOR_EXPORT
