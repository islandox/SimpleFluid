/**
 * @file FVM/explicit_instantiation/MappedWeightedTransportSystem.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Shared mapped weighted and physical-temperature assembly provider.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "FVM/TransportSystem.tcc"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Scalar = Pack::scalar_type;
using LocalOrdinal = Pack::local_ordinal_type;
using Matrix = Pack::matrix_type;
using ScalarField = CellField<Pack>;
using VectorField = VectorCellField<Pack>;
using FluxField = FaceField<Pack>;
using GeometryCache = TransportGeometryCache<Mesh<Pack>>;
using MappedGeometryCache = TransportGeometryCache<MeshHandle<Pack>>;
using MappedScalarField = ScalarCellFieldStored<Pack>;
using MappedFluxField = ScalarFaceFieldStored<Pack>;
using MappedBoundaryCache = FieldStoredBoundaryCache<Pack, MeshHandle<Pack>>;
} // namespace

// Compiled mapped entry point shared by the equation translation units that
// advance FieldStored transport on the runtime mesh handle.
template TransportSystem<Pack> weighted_scalar_transport_system<Pack, MeshHandle<Pack>>(
    FieldStoredWeightedScalarTransportRequest<Pack, MeshHandle<Pack>>);

template TransportSystem<Pack> physical_temperature_transport_system<Pack, MeshHandle<Pack>>(
    const MappedScalarField&, const MappedFluxField&, Scalar, const MappedScalarField&, const MappedScalarField&,
    const MappedScalarField&, ScalarBoundaryConditionProvider<Pack>, ScalarBoundaryValueProvider<Pack>,
    ScalarCellValueProvider<Pack>, NonOrthogonalTreatment, const MappedScalarField*, Teuchos::RCP<Matrix>,
    const MappedBoundaryCache*, const MappedGeometryCache*, FaceCoefficientInterpolation, const ALEControlVolumeState*,
    const MappedScalarField*, const MappedScalarField*, bool, AssemblyMode);

template TransportSystem<Pack> physical_temperature_transport_system<Pack, MeshHandle<Pack>>(
    const MappedScalarField&, const MappedFluxField&, Scalar, const MappedScalarField&, const MappedScalarField&,
    const MappedScalarField&, ScalarBoundaryConditionProvider<Pack>, ScalarBoundaryValueProvider<Pack>,
    ScalarCellValueProvider<Pack>, ScalarTransportDiscretization, NonOrthogonalTreatment, const MappedScalarField*,
    const MappedScalarField*, Teuchos::RCP<Matrix>, const MappedBoundaryCache*, const MappedGeometryCache*,
    FaceCoefficientInterpolation, const ALEControlVolumeState*, const MappedScalarField*, const MappedScalarField*);

} // namespace SimpleFluid::FVM
