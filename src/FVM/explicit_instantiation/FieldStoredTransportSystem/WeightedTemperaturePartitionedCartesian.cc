/** @brief Individual compiled transport factory overloads. */
#include "FVM/TransportSystemFactory.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"

// Static mesh types are hidden; export the supported public overloads individually.
#if defined(__GNUC__) || defined(__clang__)
#define SIMPLEFLUID_STATIC_TRANSPORT_EXPORT __attribute__((visibility("default")))
#else
#define SIMPLEFLUID_STATIC_TRANSPORT_EXPORT
#endif

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
using Factory = TransportSystemFactory<Pack, MeshType>;
} // namespace

template SIMPLEFLUID_STATIC_TRANSPORT_EXPORT TransportSystem<Pack> Factory::assemble(Factory::WeightedScalarTerms,
    Factory::WeightedScalarBoundary, Factory::WeightedScalarOptions, Factory::WeightedScalarContext);

template SIMPLEFLUID_STATIC_TRANSPORT_EXPORT TransportSystem<Pack> Factory::assemble(
    Factory::TemperatureTerms, Factory::TemperatureBoundary, Factory::TemperatureOptions, Factory::TemperatureContext);

template SIMPLEFLUID_STATIC_TRANSPORT_EXPORT TransportSystem<Pack> Factory::refresh_temperature_rhs(
    Factory::TemperatureTerms, Factory::TemperatureBoundary, Factory::TemperatureOptions, Factory::TemperatureContext);

} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_TRANSPORT_EXPORT
