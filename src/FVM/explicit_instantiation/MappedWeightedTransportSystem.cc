/**
 * @file FVM/explicit_instantiation/MappedWeightedTransportSystem.cc
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

// Compiled mapped entry point shared by the equation translation units that
// advance FieldStored transport on the runtime mesh handle.
template TransportSystem<Pack> weighted_scalar_transport_system<Pack, MeshHandle<Pack>>(
    FieldStoredWeightedScalarTransportRequest<Pack, MeshHandle<Pack>>);

} // namespace SimpleFluid::FVM
