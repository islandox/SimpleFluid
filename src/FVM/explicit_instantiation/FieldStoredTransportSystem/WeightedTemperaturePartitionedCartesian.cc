/** @brief Exported static-Cartesian transport factory specializations. */
#include "FVM/TransportSystemFactory.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
using Factory = TransportSystemFactory<Pack, MeshType>;
} // namespace

template<>
TransportSystem<Pack> Factory::assemble(WeightedScalarTerms terms,
    WeightedScalarBoundary boundary, WeightedScalarOptions options, WeightedScalarContext context)
{
    if (!terms.sources.source)
        terms.sources.source = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    return detail::assemble_stored_weighted<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<>
TransportSystem<Pack> Factory::assemble(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    if (!terms.power_density)
        terms.power_density = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    return detail::assemble_stored_temperature<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context), false);
}

template<>
TransportSystem<Pack> Factory::refresh_temperature_rhs(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    if (!terms.power_density)
        terms.power_density = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    return detail::assemble_stored_temperature<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context), true);
}

} // namespace SimpleFluid::FVM
