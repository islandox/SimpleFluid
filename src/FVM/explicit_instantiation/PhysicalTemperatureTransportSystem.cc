/**
 * @file FVM/explicit_instantiation/PhysicalTemperatureTransportSystem.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit instantiations for compiled transport-system assembly.
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
} // namespace

template TransportSystem<Pack> physical_temperature_transport_system<Pack>(const ScalarField&, const FluxField&, Scalar,
    const ScalarField&, const ScalarField&, const ScalarField&, ScalarBoundaryConditionProvider<Pack>,
    ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>, NonOrthogonalTreatment, const ScalarField*,
    Teuchos::RCP<Matrix>, const BoundaryCache<Pack>*, const GeometryCache*, FaceCoefficientInterpolation, AssemblyMode);

template TransportSystem<Pack> physical_temperature_transport_system<Pack>(const ScalarField&, const FluxField&, Scalar,
    const ScalarField&, const ScalarField&, const ScalarField&, ScalarBoundaryConditionProvider<Pack>,
    ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>, NonOrthogonalTreatment, const ScalarField*,
    Teuchos::RCP<Matrix>, const BoundaryCache<Pack>*, const GeometryCache*, FaceCoefficientInterpolation, bool, AssemblyMode);

} // namespace SimpleFluid::FVM
