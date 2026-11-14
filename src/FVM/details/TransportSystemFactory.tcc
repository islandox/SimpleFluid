/** @file FVM/details/TransportSystemFactory.tcc
 * @brief Shared implementation for generic and exported transport factory entries.
 * Implementation-only: included by provider TUs, never by the public header.
 */
#pragma once

#include "FVM/TransportSystem.tcc"
#include "FVM/TransportSystemFactory.hh"
#include "FVM/details/FieldStoredTransportSystem.tcc"

namespace SimpleFluid::FVM::detail
{
template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> assemble_factory_scalar(
    typename TransportSystemFactory<Pack, MeshType>::ScalarTerms terms,
    typename TransportSystemFactory<Pack, MeshType>::ScalarBoundary boundary,
    typename TransportSystemFactory<Pack, MeshType>::ScalarOptions options,
    typename TransportSystemFactory<Pack, MeshType>::ScalarContext context)
{
    const int selected = options.correction.has_value() ? 1 : 0;
    const auto correction_selection = reduce_transport_validation_state(
        terms.history.old_values.mesh(), std::array<int, 2>{selected, -selected});
    if (correction_selection[0] != -correction_selection[1])
        throw std::invalid_argument("Scalar transport requires every rank to select the same correction category.");
    if (!terms.source)
        terms.source = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
    {
        if (options.correction)
            return assemble_legacy_corrected_scalar<Pack>(
                std::move(terms), std::move(boundary), options, std::move(context));
        return assemble_legacy_basic_scalar<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context));
    }
    else
    {
        if (options.correction)
            return assemble_stored_corrected_scalar<Pack, MeshType>(
                std::move(terms), std::move(boundary), options, std::move(context));
        return assemble_stored_scalar<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context));
    }
}

template<TpetraTypePack Pack, class MeshType>
VectorTransportSystem<Pack> assemble_factory_vector(
    typename TransportSystemFactory<Pack, MeshType>::VectorTerms terms,
    typename TransportSystemFactory<Pack, MeshType>::VelocityBoundary boundary,
    typename TransportSystemFactory<Pack, MeshType>::VectorOptions options,
    typename TransportSystemFactory<Pack, MeshType>::VectorContext context)
{
    if (!terms.source)
        terms.source = [](typename Pack::local_ordinal_type) { return typename VectorCellField<Pack>::vec_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
        return assemble_legacy_corrected_vector<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context));
    else
        return assemble_stored_vector<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> assemble_factory_weighted(
    typename TransportSystemFactory<Pack, MeshType>::WeightedScalarTerms terms,
    typename TransportSystemFactory<Pack, MeshType>::WeightedScalarBoundary boundary,
    typename TransportSystemFactory<Pack, MeshType>::WeightedScalarOptions options,
    typename TransportSystemFactory<Pack, MeshType>::WeightedScalarContext context)
{
    if (!terms.sources.source)
        terms.sources.source = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
        return assemble_legacy_weighted<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context));
    else
        return assemble_stored_weighted<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> assemble_factory_temperature(
    typename TransportSystemFactory<Pack, MeshType>::TemperatureTerms terms,
    typename TransportSystemFactory<Pack, MeshType>::TemperatureBoundary boundary,
    typename TransportSystemFactory<Pack, MeshType>::TemperatureOptions options,
    typename TransportSystemFactory<Pack, MeshType>::TemperatureContext context,
    bool rhs_only)
{
    if (!terms.power_density)
        terms.power_density = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
        return assemble_legacy_temperature<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context), rhs_only);
    else
        return assemble_stored_temperature<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context), rhs_only);
}

template<TpetraTypePack Pack, class MeshType>
VectorTransportSystem<Pack> assemble_factory_momentum(
    typename TransportSystemFactory<Pack, MeshType>::MomentumTerms terms,
    typename TransportSystemFactory<Pack, MeshType>::MomentumBoundary boundary,
    typename TransportSystemFactory<Pack, MeshType>::MomentumOptions options,
    typename TransportSystemFactory<Pack, MeshType>::MomentumContext context)
{
    if (!terms.acceleration_source)
        terms.acceleration_source = [](typename Pack::local_ordinal_type)
        { return typename VectorCellField<Pack>::vec_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
        return assemble_legacy_momentum<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context));
    else
        return assemble_stored_momentum<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context));
}
} // namespace SimpleFluid::FVM::detail
