/**
 * @file DiagnosticPressureFaceFlux.cc
 * @brief Private stored-pressure provider for frozen-gradient diagnostic components.
 */

#include "FVM/FaceFlux.hh"
#include "FVM/details/FieldStoredPressureWeightedFaceFlux.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM::detail
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = MeshHandle<Pack>;
} // namespace

// The public component API reconstructs its own gradient. Diagnostics retain a
// separately captured physical-pressure gradient while changing ALE flux BCs.
// Keep that combined signature private to the equation/solver image.
template void pressure_weighted_stored_face_fluxes_impl<Pack, MeshType,
    FieldStoredVelocityBoundaryCache<Pack, MeshType>, FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>>(
    const VectorCellFieldStored<Pack, MeshType>& velocity, const ScalarCellFieldStored<Pack, MeshType>& pressure,
    Pack::scalar_type pressure_coefficient, const FieldStoredVelocityBoundaryCache<Pack, MeshType>& boundary_cache,
    const BoundaryConditionMap* pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>& workspace,
    VectorCellFieldStored<Pack, MeshType>* precomputed_pressure_gradient,
    ScalarFaceFieldStored<Pack, MeshType>& fluxes, CellGradientScheme gradient_scheme,
    ScalarFaceFieldStored<Pack, MeshType>* velocity_component,
    ScalarFaceFieldStored<Pack, MeshType>* pressure_component);
} // namespace SimpleFluid::FVM::detail
