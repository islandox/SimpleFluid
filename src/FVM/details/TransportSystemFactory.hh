/** @file FVM/details/TransportSystemFactory.hh
 * @brief Private declarations shared by generic and exported transport factory providers.
 */
#pragma once

#include "FVM/TransportSystemFactory.hh"

namespace SimpleFluid::FVM::detail
{
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_LOCAL TransportSystem<Pack> assemble_factory_scalar(
    typename TransportSystemFactory<Pack, MeshType>::ScalarTerms terms,
    typename TransportSystemFactory<Pack, MeshType>::ScalarBoundary boundary,
    typename TransportSystemFactory<Pack, MeshType>::ScalarOptions options,
    typename TransportSystemFactory<Pack, MeshType>::ScalarContext context);

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_LOCAL VectorTransportSystem<Pack> assemble_factory_vector(
    typename TransportSystemFactory<Pack, MeshType>::VectorTerms terms,
    typename TransportSystemFactory<Pack, MeshType>::VelocityBoundary boundary,
    typename TransportSystemFactory<Pack, MeshType>::VectorOptions options,
    typename TransportSystemFactory<Pack, MeshType>::VectorContext context);

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_LOCAL TransportSystem<Pack> assemble_factory_weighted(
    typename TransportSystemFactory<Pack, MeshType>::WeightedScalarTerms terms,
    typename TransportSystemFactory<Pack, MeshType>::WeightedScalarBoundary boundary,
    typename TransportSystemFactory<Pack, MeshType>::WeightedScalarOptions options,
    typename TransportSystemFactory<Pack, MeshType>::WeightedScalarContext context);

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_LOCAL TransportSystem<Pack> assemble_factory_temperature(
    typename TransportSystemFactory<Pack, MeshType>::TemperatureTerms terms,
    typename TransportSystemFactory<Pack, MeshType>::TemperatureBoundary boundary,
    typename TransportSystemFactory<Pack, MeshType>::TemperatureOptions options,
    typename TransportSystemFactory<Pack, MeshType>::TemperatureContext context,
    bool rhs_only);

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_LOCAL VectorTransportSystem<Pack> assemble_factory_momentum(
    typename TransportSystemFactory<Pack, MeshType>::MomentumTerms terms,
    typename TransportSystemFactory<Pack, MeshType>::MomentumBoundary boundary,
    typename TransportSystemFactory<Pack, MeshType>::MomentumOptions options,
    typename TransportSystemFactory<Pack, MeshType>::MomentumContext context);
} // namespace SimpleFluid::FVM::detail
