/**
 * @file FVM/TransportSystem.cc
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
} // namespace

template class TransportGeometryCache<Mesh<Pack>>;

template TransportSystem<Pack> transport_system<Pack>(const ScalarField&, const FluxField&, Scalar, Scalar,
    ScalarBoundaryConditionProvider<Pack>, ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>,
    Teuchos::RCP<Matrix>);

template TransportSystem<Pack> transport_system<Pack>(const ScalarField&, const FluxField&, Scalar, Scalar,
    ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>, Teuchos::RCP<Matrix>);

template TransportSystem<Pack> transport_system<Pack>(const ScalarField&, const FluxField&, Scalar, Scalar,
    ScalarBoundaryConditionProvider<Pack>, ScalarBoundaryValueProvider<Pack>, Teuchos::RCP<Matrix>);

template TransportSystem<Pack> transport_system<Pack>(
    const ScalarField&, const FluxField&, Scalar, Scalar, ScalarBoundaryValueProvider<Pack>, Teuchos::RCP<Matrix>);

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

template TransportSystem<Pack> weighted_scalar_transport_system<Pack>(const ScalarField&, const FluxField&, Scalar,
    const ScalarField&, const ScalarField&, const ScalarField&, ScalarBoundaryConditionProvider<Pack>,
    ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>, NonOrthogonalTreatment, const ScalarField*,
    Teuchos::RCP<Matrix>, std::function<Scalar(LocalOrdinal)>, std::function<std::optional<Scalar>(LocalOrdinal)>,
    const BoundaryCache<Pack>*, const GeometryCache*, FaceCoefficientInterpolation);

template TransportSystem<Pack> physical_temperature_transport_system<Pack>(const ScalarField&, const FluxField&, Scalar,
    const ScalarField&, const ScalarField&, const ScalarField&, ScalarBoundaryConditionProvider<Pack>,
    ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>, NonOrthogonalTreatment, const ScalarField*,
    Teuchos::RCP<Matrix>, const BoundaryCache<Pack>*, const GeometryCache*, FaceCoefficientInterpolation);

template VectorTransportSystem<Pack> physical_momentum_transport_system<Pack>(const VectorField&, const FluxField&,
    Scalar, const ScalarField&, Scalar, VectorBoundaryValueProvider<Pack>, VectorCellValueProvider<Pack>,
    NonOrthogonalTreatment, const VectorField*, Teuchos::RCP<Matrix>, BoundaryFaceSelector, const BoundaryCache<Pack>*,
    const GeometryCache*, FaceCoefficientInterpolation);

template VectorTransportSystem<Pack> transport_system<Pack>(
    const VectorField&, const FluxField&, Scalar, Scalar, VectorBoundaryValueProvider<Pack>, Teuchos::RCP<Matrix>);

} // namespace SimpleFluid::FVM
