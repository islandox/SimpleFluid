/** @brief Weighted-scalar implementation provider for the generic reference mesh. */
#include "FVM/FieldStoredTransportSystem.tcc"
#include "TestMeshes.hh"
#include "fields/FieldStored.tcc"

namespace SimpleFluid::FVM
{
using Pack = DefaultTpetraTypes;
using Reference = test::transport_providers::ReferenceHandle;
template TransportSystem<Pack> weighted_scalar_transport_system<Pack, Reference>(
    FieldStoredWeightedScalarTransportRequest<Pack, Reference>);
} // namespace SimpleFluid::FVM
