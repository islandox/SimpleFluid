/** @brief Compiled mapped pressure provider for PartitionedCartesian. */
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

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_fluxes<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes, CellGradientScheme gradient_scheme);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_fluxes<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_fluxes<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes, CellGradientScheme gradient_scheme);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_flux_components<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& total, ScalarFaceFieldStored<Pack, MeshType>& velocity_component,
    ScalarFaceFieldStored<Pack, MeshType>& pressure_component, CellGradientScheme gradient_scheme);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_fluxes<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, const ScalarCellFieldStored<Pack, MeshType>& pressure,
    VectorCellFieldStored<Pack, MeshType>& pressure_gradient, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_fluxes<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions, ScalarFaceFieldStored<Pack, MeshType>& fluxes);

} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_OPERATOR_EXPORT
