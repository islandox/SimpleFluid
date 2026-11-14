/** @brief Compiled mapped pressure provider for MeshHandle. */
#include "FVM/FieldStoredFaceFlux.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = MeshHandle<Pack>;
} // namespace

template void pressure_weighted_face_fluxes<Pack, MeshType>(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes, CellGradientScheme gradient_scheme);

template void pressure_weighted_face_fluxes<Pack, MeshType>(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

template void pressure_weighted_face_fluxes<Pack, MeshType>(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes, CellGradientScheme gradient_scheme);

template void pressure_weighted_face_flux_components<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, const ScalarCellFieldStored<Pack, MeshType>& pressure,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& total, ScalarFaceFieldStored<Pack, MeshType>& velocity_component,
    ScalarFaceFieldStored<Pack, MeshType>& pressure_component, CellGradientScheme gradient_scheme);

template void pressure_weighted_face_fluxes<Pack, MeshType>(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, VectorCellFieldStored<Pack, MeshType>& pressure_gradient,
    typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes);

template void pressure_weighted_face_fluxes<Pack, MeshType>(const VectorCellFieldStored<Pack, MeshType>& velocity,
    const ScalarCellFieldStored<Pack, MeshType>& pressure, typename Pack::scalar_type pressure_coefficient,
    const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap& pressure_boundary_conditions, ScalarFaceFieldStored<Pack, MeshType>& fluxes);

} // namespace SimpleFluid::FVM
