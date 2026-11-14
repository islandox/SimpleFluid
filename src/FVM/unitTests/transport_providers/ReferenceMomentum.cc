/** @file ReferenceMomentum.cc
 * @brief Momentum factory provider for the generic reference mesh.
 */
#include "FVM/TransportSystemFactory.tcc"
#include "TestMeshes.hh"
#include "fields/FieldStored.tcc"

namespace SimpleFluid::FVM
{
using Pack = DefaultTpetraTypes;
using MeshType = test::transport_providers::ReferenceHandle;
using Factory = TransportSystemFactory<Pack, MeshType>;
template VectorTransportSystem<Pack> Factory::assemble(
    Factory::MomentumTerms, Factory::MomentumBoundary, Factory::MomentumOptions, Factory::MomentumContext);
} // namespace SimpleFluid::FVM
