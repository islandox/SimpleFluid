/** @file FVM/TransportSystemFactory.tcc
 * @brief Compiled factory definitions; opt in for custom mesh/type providers.
 */
#pragma once

#include "FVM/TransportSystem.tcc"
#include "FVM/TransportSystemFactory.hh"
#include "FVM/details/FieldStoredTransportSystem.tcc"

namespace SimpleFluid::FVM
{
template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    ScalarTerms terms, ScalarBoundary boundary, ScalarOptions options, ScalarContext context)
{
    const int selected = options.correction.has_value() ? 1 : 0;
    const auto correction_selection = detail::reduce_transport_validation_state(
        terms.history.old_values.mesh(), std::array<int, 2>{selected, -selected});
    if (correction_selection[0] != -correction_selection[1])
        throw std::invalid_argument("Scalar transport requires every rank to select the same correction category.");
    if (!terms.source)
        terms.source = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
    {
        if (options.correction)
            return detail::assemble_legacy_corrected_scalar<Pack>(
                std::move(terms), std::move(boundary), options, std::move(context));
        return detail::assemble_legacy_basic_scalar<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context));
    }
    else
    {
        if (options.correction)
            return detail::assemble_stored_corrected_scalar<Pack, MeshType>(
                std::move(terms), std::move(boundary), options, std::move(context));
        return detail::assemble_stored_scalar<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context));
    }
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
    if (!terms.source)
        terms.source = [](typename Pack::local_ordinal_type) { return typename VectorCellField<Pack>::vec_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
        return detail::assemble_legacy_corrected_vector<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context));
    else
        return detail::assemble_stored_vector<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(WeightedScalarTerms terms,
    WeightedScalarBoundary boundary, WeightedScalarOptions options, WeightedScalarContext context)
{
    if (!terms.sources.source)
        terms.sources.source = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
        return detail::assemble_legacy_weighted<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context));
    else
        return detail::assemble_stored_weighted<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context));
}

template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    if (!terms.power_density)
        terms.power_density = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
        return detail::assemble_legacy_temperature<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context), false);
    else
        return detail::assemble_stored_temperature<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context), false);
}

template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::refresh_temperature_rhs(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    if (!terms.power_density)
        terms.power_density = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
        return detail::assemble_legacy_temperature<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context), true);
    else
        return detail::assemble_stored_temperature<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context), true);
}

template<TpetraTypePack Pack, class MeshType>
VectorTransportSystem<Pack> TransportSystemFactory<Pack, MeshType>::assemble(
    MomentumTerms terms, MomentumBoundary boundary, MomentumOptions options, MomentumContext context)
{
    if (!terms.acceleration_source)
        terms.acceleration_source = [](typename Pack::local_ordinal_type)
        { return typename VectorCellField<Pack>::vec_type{}; };
    if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
        return detail::assemble_legacy_momentum<Pack>(
            std::move(terms), std::move(boundary), options, std::move(context));
    else
        return detail::assemble_stored_momentum<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context));
}
} // namespace SimpleFluid::FVM
