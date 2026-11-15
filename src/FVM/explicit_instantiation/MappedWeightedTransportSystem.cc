/** @brief Individual compiled transport factory overloads. */
#include "FVM/details/FieldStoredTransportSystem.tcc"
#include "FVM/TransportSystemFactory.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = MeshHandle<Pack>;
using Factory = TransportSystemFactory<Pack, MeshType>;
} // namespace

template TransportSystem<Pack> Factory::assemble(Factory::WeightedScalarTerms, Factory::WeightedScalarBoundary,
    Factory::WeightedScalarOptions, Factory::WeightedScalarContext);

template TransportSystem<Pack> Factory::assemble(
    Factory::TemperatureTerms, Factory::TemperatureBoundary, Factory::TemperatureOptions, Factory::TemperatureContext);

template TransportSystem<Pack> Factory::refresh_temperature_rhs(
    Factory::TemperatureTerms, Factory::TemperatureBoundary, Factory::TemperatureOptions, Factory::TemperatureContext);

namespace detail
{
template TransportSystem<Pack> assemble_stored_weighted<Pack, MeshType>(
    Factory::WeightedScalarTerms, Factory::WeightedScalarBoundary, Factory::WeightedScalarOptions,
    Factory::WeightedScalarContext);

template TransportSystem<Pack> assemble_stored_temperature<Pack, MeshType>(
    Factory::TemperatureTerms, Factory::TemperatureBoundary, Factory::TemperatureOptions,
    Factory::TemperatureContext, bool);
} // namespace detail
} // namespace SimpleFluid::FVM
