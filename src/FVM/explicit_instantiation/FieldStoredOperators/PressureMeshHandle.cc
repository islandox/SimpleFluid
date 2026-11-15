/** @brief Compiled mapped pressure provider for MeshHandle. */
#include "FVM/FieldStoredFaceFlux.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Mesh = MeshHandle<Pack>;
} // namespace

template void pressure_weighted_face_fluxes<Pack, Mesh>(const VectorCellFieldStored<Pack, Mesh>& velocity,
    const ScalarCellFieldStored<Pack, Mesh>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Mesh>& workspace,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes, CellGradientScheme gradient_scheme);

template void pressure_weighted_face_fluxes<Pack, Mesh>(const VectorCellFieldStored<Pack, Mesh>& velocity,
    const ScalarCellFieldStored<Pack, Mesh>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes);

template void pressure_weighted_face_fluxes<Pack, Mesh>(const VectorCellFieldStored<Pack, Mesh>& velocity,
    const ScalarCellFieldStored<Pack, Mesh>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Mesh>& workspace,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes, CellGradientScheme gradient_scheme);

template void pressure_weighted_face_flux_components<Pack, Mesh>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, const ScalarCellFieldStored<Pack, Mesh>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Mesh>& workspace,
    ScalarFaceFieldStored<Pack, Mesh>& total, ScalarFaceFieldStored<Pack, Mesh>& velocity_component,
    ScalarFaceFieldStored<Pack, Mesh>& pressure_component, CellGradientScheme gradient_scheme);

template void pressure_weighted_face_fluxes<Pack, Mesh>(const VectorCellFieldStored<Pack, Mesh>& velocity,
    const ScalarCellFieldStored<Pack, Mesh>& pressure, VectorCellFieldStored<Pack, Mesh>& pressure_gradient,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Mesh>& workspace,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes);

template void pressure_weighted_face_fluxes<Pack, Mesh>(const VectorCellFieldStored<Pack, Mesh>& velocity,
    const ScalarCellFieldStored<Pack, Mesh>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions, ScalarFaceFieldStored<Pack, Mesh>& fluxes);

} // namespace SimpleFluid::FVM
