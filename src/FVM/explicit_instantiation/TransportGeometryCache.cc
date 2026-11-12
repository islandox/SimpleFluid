/**
 * @file FVM/explicit_instantiation/TransportGeometryCache.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit instantiations for compiled transport-system assembly.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "FVM/TransportSystem.hh"

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

template class TransportGeometryCache<Mesh<Pack>>;
template class TransportGeometryCache<MeshHandle<Pack>>;

} // namespace SimpleFluid::FVM
