/**
 * @file FVM/explicit_instantiation/TransportSystem.cc
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

template TransportSystem<Pack> transport_system<Pack>(const ScalarField&, const FluxField&, Scalar, Scalar,
    ScalarBoundaryConditionProvider<Pack>, ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>,
    Teuchos::RCP<Matrix>);

template TransportSystem<Pack> transport_system<Pack>(const ScalarField&, const FluxField&, Scalar, Scalar,
    ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>, Teuchos::RCP<Matrix>);

template TransportSystem<Pack> transport_system<Pack>(const ScalarField&, const FluxField&, Scalar, Scalar,
    ScalarBoundaryConditionProvider<Pack>, ScalarBoundaryValueProvider<Pack>, Teuchos::RCP<Matrix>);

template TransportSystem<Pack> transport_system<Pack>(
    const ScalarField&, const FluxField&, Scalar, Scalar, ScalarBoundaryValueProvider<Pack>, Teuchos::RCP<Matrix>);

template TransportSystem<Pack> non_orthogonal_transport_system<Pack>(const ScalarField&, const FluxField&, Scalar,
    Scalar, ScalarBoundaryConditionProvider<Pack>, ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>,
    NonOrthogonalTreatment, const ScalarField*, Teuchos::RCP<Matrix>, const GeometryCache*);

template VectorTransportSystem<Pack> transport_system<Pack>(const VectorField&, const FluxField&, Scalar, Scalar,
    VectorBoundaryConditionProvider<Pack>, VectorBoundaryValueProvider<Pack>, VectorCellValueProvider<Pack>,
    Teuchos::RCP<Matrix>);

template VectorTransportSystem<Pack> transport_system<Pack>(const VectorField&, const FluxField&, Scalar, Scalar,
    VectorBoundaryValueProvider<Pack>, VectorCellValueProvider<Pack>, Teuchos::RCP<Matrix>);

template VectorTransportSystem<Pack> transport_system<Pack>(const VectorField&, const FluxField&, Scalar, Scalar,
    VectorBoundaryConditionProvider<Pack>, VectorBoundaryValueProvider<Pack>, Teuchos::RCP<Matrix>);

template VectorTransportSystem<Pack> non_orthogonal_transport_system<Pack>(const VectorField&, const FluxField&, Scalar,
    Scalar, VectorBoundaryValueProvider<Pack>, VectorCellValueProvider<Pack>, NonOrthogonalTreatment,
    const VectorField*, Teuchos::RCP<Matrix>, BoundaryFaceSelector, const GeometryCache*);

template VectorTransportSystem<Pack> transport_system<Pack>(
    const VectorField&, const FluxField&, Scalar, Scalar, VectorBoundaryValueProvider<Pack>, Teuchos::RCP<Matrix>);

} // namespace SimpleFluid::FVM
