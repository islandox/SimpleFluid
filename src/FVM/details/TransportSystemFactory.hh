/** @file FVM/details/TransportSystemFactory.hh
 * @brief Private declarations shared by generic and exported transport factory providers.
 */
#pragma once

#include "FVM/TransportSystemFactory.hh"

SIMPLEFLUID_LEGACY_API_BEGIN

namespace SimpleFluid::FVM::detail
{
// Legacy compatibility builders are supplied by the legacy factory providers.
template<TpetraTypePack Pack>
SIMPLEFLUID_LOCAL TransportSystem<Pack> assemble_legacy_basic_scalar(
    ScalarTerms<Pack, Mesh<Pack>> terms,
    typename TransportSystemFactory<Pack>::ScalarBoundary boundary,
    typename TransportSystemFactory<Pack>::ScalarOptions,
    typename TransportSystemFactory<Pack>::ScalarContext context);

template<TpetraTypePack Pack>
SIMPLEFLUID_LOCAL TransportSystem<Pack> assemble_legacy_corrected_scalar(
    ScalarTerms<Pack, Mesh<Pack>> terms,
    typename TransportSystemFactory<Pack>::ScalarBoundary boundary,
    typename TransportSystemFactory<Pack>::ScalarOptions options,
    typename TransportSystemFactory<Pack>::ScalarContext context);

template<TpetraTypePack Pack>
SIMPLEFLUID_LOCAL VectorTransportSystem<Pack> assemble_legacy_basic_vector(
    VectorTerms<Pack, Mesh<Pack>> terms,
    typename TransportSystemFactory<Pack>::BasicVectorBoundary boundary,
    typename TransportSystemFactory<Pack>::BasicVectorOptions,
    typename TransportSystemFactory<Pack>::BasicVectorContext context);

template<TpetraTypePack Pack>
SIMPLEFLUID_LOCAL VectorTransportSystem<Pack> assemble_legacy_corrected_vector(
    VectorTerms<Pack, Mesh<Pack>> terms,
    typename TransportSystemFactory<Pack>::VelocityBoundary boundary,
    typename TransportSystemFactory<Pack>::VectorOptions options,
    typename TransportSystemFactory<Pack>::VectorContext context);

template<TpetraTypePack Pack>
SIMPLEFLUID_LOCAL TransportSystem<Pack> assemble_legacy_weighted(
    WeightedScalarTerms<Pack, Mesh<Pack>> terms,
    typename TransportSystemFactory<Pack>::WeightedScalarBoundary boundary,
    typename TransportSystemFactory<Pack>::WeightedScalarOptions options,
    typename TransportSystemFactory<Pack>::WeightedScalarContext context);

template<TpetraTypePack Pack>
SIMPLEFLUID_LOCAL TransportSystem<Pack> assemble_legacy_temperature(
    TemperatureTerms<Pack, Mesh<Pack>> terms,
    typename TransportSystemFactory<Pack>::TemperatureBoundary boundary,
    typename TransportSystemFactory<Pack>::TemperatureOptions options,
    typename TransportSystemFactory<Pack>::TemperatureContext context, bool rhs_only);

template<TpetraTypePack Pack>
SIMPLEFLUID_LOCAL VectorTransportSystem<Pack> assemble_legacy_momentum(
    MomentumTerms<Pack, Mesh<Pack>> terms,
    typename TransportSystemFactory<Pack>::MomentumBoundary boundary,
    typename TransportSystemFactory<Pack>::MomentumOptions options,
    typename TransportSystemFactory<Pack>::MomentumContext context);

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

SIMPLEFLUID_LEGACY_API_END
