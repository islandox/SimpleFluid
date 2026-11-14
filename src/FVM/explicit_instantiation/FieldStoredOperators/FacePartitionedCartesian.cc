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
using MeshType = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
} // namespace

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT FieldStoredVelocityBoundaryCache<Pack, MeshType>
cache_velocity_boundary_conditions<Pack, MeshType>(
    SP<const MeshType> mesh, const BoundaryConditionSet& boundary_conditions);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void face_velocities<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, VectorFaceFieldStored<Pack, MeshType>& face_velocity);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void face_velocities<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    VectorFaceFieldStored<Pack, MeshType>& face_velocity);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void normal_face_fluxes<Pack, MeshType>(
    const VectorFaceFieldStored<Pack, MeshType>& face_velocity, ScalarFaceFieldStored<Pack, MeshType>& fluxes);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void face_fluxes<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, ScalarFaceFieldStored<Pack, MeshType>& fluxes);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void face_fluxes<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void mesh_relative_face_fluxes<Pack, MeshType>(
    const ScalarFaceFieldStored<Pack, MeshType>& absolute_fluxes, const ALEControlVolumeState& ale,
    ScalarFaceFieldStored<Pack, MeshType>& relative_fluxes);

} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_OPERATOR_EXPORT
