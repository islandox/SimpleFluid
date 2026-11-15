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
using Mesh = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
} // namespace

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_fluxes<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, const ScalarCellFieldStored<Pack, Mesh>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Mesh>& workspace,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes, CellGradientScheme gradient_scheme);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_fluxes<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, const ScalarCellFieldStored<Pack, Mesh>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_fluxes<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, const ScalarCellFieldStored<Pack, Mesh>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Mesh>& workspace,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes, CellGradientScheme gradient_scheme);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_flux_components<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, const ScalarCellFieldStored<Pack, Mesh>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Mesh>& workspace,
    ScalarFaceFieldStored<Pack, Mesh>& total, ScalarFaceFieldStored<Pack, Mesh>& velocity_component,
    ScalarFaceFieldStored<Pack, Mesh>& pressure_component, CellGradientScheme gradient_scheme);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_fluxes<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, const ScalarCellFieldStored<Pack, Mesh>& pressure,
    VectorCellFieldStored<Pack, Mesh>& pressure_gradient, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Mesh>& workspace,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT void pressure_weighted_face_fluxes<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, const ScalarCellFieldStored<Pack, Mesh>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions, ScalarFaceFieldStored<Pack, Mesh>& fluxes);

} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_OPERATOR_EXPORT
