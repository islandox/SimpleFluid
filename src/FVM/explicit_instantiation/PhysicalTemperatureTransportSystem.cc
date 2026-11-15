/** @brief Individual compiled transport factory overloads. */
#include "utils/export_helper.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "FVM/TransportSystemFactory.tcc"
#include "FVM/TransportSystem.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = Mesh<Pack>;
using Factory = TransportSystemFactory<Pack, MeshType>;
} // namespace

template TransportSystem<Pack> Factory::assemble(
    Factory::TemperatureTerms, Factory::TemperatureBoundary, Factory::TemperatureOptions, Factory::TemperatureContext);

template TransportSystem<Pack> Factory::refresh_temperature_rhs(
    Factory::TemperatureTerms, Factory::TemperatureBoundary, Factory::TemperatureOptions, Factory::TemperatureContext);

namespace detail
{
template TransportSystem<Pack> assemble_legacy_temperature<Pack>(
    Factory::TemperatureTerms, Factory::TemperatureBoundary,
    Factory::TemperatureOptions, Factory::TemperatureContext, bool);
} // namespace detail

} // namespace SimpleFluid::FVM

SIMPLEFLUID_LEGACY_API_END
