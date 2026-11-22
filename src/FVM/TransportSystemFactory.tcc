/** @file FVM/TransportSystemFactory.tcc
 * @brief Compiled factory definitions; opt in for custom mesh/type providers.
 */
#pragma once
#include "utils/export_helper.hh"

#include "geometry/MeshConcepts.hh"
#include "FVM/details/TransportSystemFactory.tcc"

namespace SimpleFluid::FVM
{
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    ScalarTerms terms, ScalarBoundary boundary, ScalarOptions options, ScalarContext context)
{
    return detail::assemble_factory_scalar<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
VectorTransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    VectorTerms terms, BasicVectorBoundary boundary, BasicVectorOptions options, BasicVectorContext context)
{
    if (!terms.source)
        terms.source = [](typename Pack::local_ordinal_type) { return typename VectorCellFieldStored<Pack, MeshType>::value_type{}; };
    return detail::assemble_stored_basic_vector<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
VectorTransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    VectorTerms terms, VelocityBoundary boundary, VectorOptions options, VectorContext context)
{
    return detail::assemble_factory_vector<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(WeightedScalarTerms terms,
    WeightedScalarBoundary boundary, WeightedScalarOptions options, WeightedScalarContext context)
{
    return detail::assemble_factory_weighted<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
MultiColumnTransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble_multi_column(
    MultiColumnWeightedTerms terms, MultiColumnWeightedBoundary boundary, WeightedScalarOptions options,
    MultiColumnWeightedContext context)
{
    return detail::assemble_factory_multi_column_weighted<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context), false);
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
MultiColumnTransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::refresh_multi_column_rhs(
    MultiColumnWeightedTerms terms, MultiColumnWeightedBoundary boundary, WeightedScalarOptions options,
    MultiColumnWeightedContext context)
{
    return detail::assemble_factory_multi_column_weighted<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context), true);
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    return detail::assemble_factory_temperature<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context), false);
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::refresh_temperature_rhs(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    return detail::assemble_factory_temperature<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context), true);
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
VectorTransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    MomentumTerms terms, MomentumBoundary boundary, MomentumOptions options, MomentumContext context)
{
    return detail::assemble_factory_momentum<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}
} // namespace SimpleFluid::FVM
