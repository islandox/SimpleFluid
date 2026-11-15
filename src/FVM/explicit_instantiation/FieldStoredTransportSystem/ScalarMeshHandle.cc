/** @brief Individual compiled transport factory overloads. */
#include "FVM/details/FieldStoredTransportSystem.tcc"
#include "FVM/TransportSystemFactory.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Mesh = MeshHandle<Pack>;
using Factory = TransportSystemFactory<Pack, Mesh>;
} // namespace

template TransportSystem<Pack> Factory::assemble(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);

namespace detail
{
template TransportSystem<Pack> assemble_stored_scalar<Pack, Mesh>(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);

template TransportSystem<Pack> assemble_stored_corrected_scalar<Pack, Mesh>(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);
} // namespace detail
} // namespace SimpleFluid::FVM
