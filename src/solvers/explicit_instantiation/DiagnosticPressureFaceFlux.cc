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
using Mesh = MeshHandle<Pack>;
} // namespace

// The public component API reconstructs its own gradient. Diagnostics retain a
// separately captured physical-pressure gradient while changing ALE flux BCs.
// Keep that combined signature private to the equation/solver image.
template void pressure_weighted_stored_face_fluxes_impl<Pack, Mesh,
    FieldStoredVelocityBoundaryCache<Pack, Mesh>, FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Mesh>>(
    const VectorCellFieldStored<Pack, Mesh>& velocity, const ScalarCellFieldStored<Pack, Mesh>& pressure,
    Pack::scalar_type pressure_coefficient, const FieldStoredVelocityBoundaryCache<Pack, Mesh>& boundary_cache,
    const BoundaryConditionMap* pressure_boundary_conditions,
    FieldStoredPressureWeightedFaceFluxWorkspace<Pack, Mesh>& workspace,
    VectorCellFieldStored<Pack, Mesh>* precomputed_pressure_gradient,
    ScalarFaceFieldStored<Pack, Mesh>& fluxes, CellGradientScheme gradient_scheme,
    ScalarFaceFieldStored<Pack, Mesh>* velocity_component,
    ScalarFaceFieldStored<Pack, Mesh>* pressure_component);
} // namespace SimpleFluid::FVM::detail
