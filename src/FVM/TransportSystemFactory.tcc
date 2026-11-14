/** @file FVM/TransportSystemFactory.tcc
 * @brief Compiled factory definitions; opt in for custom mesh/type providers.
 */
#pragma once

#include "FVM/details/TransportSystemFactory.tcc"

namespace SimpleFluid::FVM
{
template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    ScalarTerms terms, ScalarBoundary boundary, ScalarOptions options, ScalarContext context)
{
    return detail::assemble_factory_scalar<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, class MeshType>
VectorTransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    VectorTerms terms, BasicVectorBoundary boundary, BasicVectorOptions options, BasicVectorContext context)
    requires std::is_same_v<MeshType, Mesh<Pack>>
{
    if (!terms.source)
        terms.source = [](typename Pack::local_ordinal_type) { return typename VectorCellField<Pack>::vec_type{}; };
    return detail::assemble_legacy_basic_vector<Pack>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, class MeshType>
VectorTransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    VectorTerms terms, VelocityBoundary boundary, VectorOptions options, VectorContext context)
{
    return detail::assemble_factory_vector<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(WeightedScalarTerms terms,
    WeightedScalarBoundary boundary, WeightedScalarOptions options, WeightedScalarContext context)
{
    return detail::assemble_factory_weighted<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    return detail::assemble_factory_temperature<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context), false);
}

template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::refresh_temperature_rhs(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    return detail::assemble_factory_temperature<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context), true);
}

template<TpetraTypePack Pack, class MeshType>
VectorTransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    MomentumTerms terms, MomentumBoundary boundary, MomentumOptions options, MomentumContext context)
{
    return detail::assemble_factory_momentum<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}
} // namespace SimpleFluid::FVM
