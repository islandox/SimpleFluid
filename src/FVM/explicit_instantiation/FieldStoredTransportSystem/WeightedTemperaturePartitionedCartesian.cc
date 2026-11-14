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
    return detail::assemble_factory_weighted<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

template<>
TransportSystem<Pack> Factory::assemble(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    return detail::assemble_factory_temperature<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context), false);
}

template<>
TransportSystem<Pack> Factory::refresh_temperature_rhs(
    TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context)
{
    return detail::assemble_factory_temperature<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context), true);
}

} // namespace SimpleFluid::FVM
