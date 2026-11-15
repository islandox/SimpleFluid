/** @brief Test-owned pressure kernels for custom boundary and failure fixtures. */
#include "PressureFluxFixtures.hh"
#include "FVM/details/FieldStoredPressureWeightedFaceFlux.tcc"

namespace SimpleFluid::FVM::detail
{
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Workspace = FieldStoredPressureWeightedFaceFluxWorkspace<Pack>;
using BoundaryCache = FieldStoredVelocityBoundaryCache<Pack, Handle>;
using CountingCache = test::pressure_flux::CountingVelocityBoundaryCache;
using FailingWorkspace = test::pressure_flux::FailingWorkspace;

template void pressure_weighted_stored_face_fluxes_impl<Pack, Handle, CountingCache, Workspace>(
    const VectorCellFieldStored<Pack, Handle>&, const ScalarCellFieldStored<Pack, Handle>&,
    Pack::scalar_type, const CountingCache&, const BoundaryConditionMap*, Workspace&,
    VectorCellFieldStored<Pack, Handle>*, ScalarFaceFieldStored<Pack, Handle>&, CellGradientScheme,
    ScalarFaceFieldStored<Pack, Handle>*, ScalarFaceFieldStored<Pack, Handle>*);

template void pressure_weighted_stored_face_fluxes_impl<Pack, Handle, BoundaryCache, FailingWorkspace>(
    const VectorCellFieldStored<Pack, Handle>&, const ScalarCellFieldStored<Pack, Handle>&,
    Pack::scalar_type, const BoundaryCache&, const BoundaryConditionMap*, FailingWorkspace&,
    VectorCellFieldStored<Pack, Handle>*, ScalarFaceFieldStored<Pack, Handle>&, CellGradientScheme,
    ScalarFaceFieldStored<Pack, Handle>*, ScalarFaceFieldStored<Pack, Handle>*);
} // namespace SimpleFluid::FVM::detail
