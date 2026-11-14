/** @file ReferenceWeightedScalar.cc
 * @brief Weighted-scalar factory provider for the generic reference mesh.
 */
#include "FVM/TransportSystemFactory.tcc"
#include "TestMeshes.hh"
#include "fields/FieldStored.tcc"

namespace SimpleFluid::FVM
{
using Pack = DefaultTpetraTypes;
using Reference = test::transport_providers::ReferenceHandle;
using Factory = TransportSystemFactory<Pack, Reference>;
template TransportSystem<Pack> Factory::assemble(Factory::WeightedScalarTerms, Factory::WeightedScalarBoundary,
    Factory::WeightedScalarOptions, Factory::WeightedScalarContext);
} // namespace SimpleFluid::FVM
