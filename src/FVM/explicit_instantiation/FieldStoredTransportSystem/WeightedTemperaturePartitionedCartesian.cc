/** @brief Exported static-Cartesian transport factory specializations. */
#include "FVM/details/FieldStoredTransportSystem.tcc"
#include "FVM/TransportSystemFactory.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Mesh = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
using Factory = TransportSystemFactory<Pack, Mesh>;
} // namespace

template<>
TransportSystem<Pack> Factory::assemble(WeightedScalarTerms terms,
    WeightedScalarBoundary boundary, WeightedScalarOptions options, WeightedScalarContext context)
{
    return detail::assemble_factory_weighted<Pack, Mesh>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<>
TransportSystem<Pack> Factory::assemble(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    return detail::assemble_factory_temperature<Pack, Mesh>(
        std::move(terms), std::move(boundary), options, std::move(context), false);
}

template<>
TransportSystem<Pack> Factory::refresh_temperature_rhs(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    return detail::assemble_factory_temperature<Pack, Mesh>(
        std::move(terms), std::move(boundary), options, std::move(context), true);
}

namespace detail
{
template TransportSystem<Pack> assemble_stored_weighted<Pack, Mesh>(
    Factory::WeightedScalarTerms, Factory::WeightedScalarBoundary, Factory::WeightedScalarOptions,
    Factory::WeightedScalarContext);

template TransportSystem<Pack> assemble_stored_temperature<Pack, Mesh>(
    Factory::TemperatureTerms, Factory::TemperatureBoundary, Factory::TemperatureOptions,
    Factory::TemperatureContext, bool);
} // namespace detail
} // namespace SimpleFluid::FVM
