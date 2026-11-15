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

template TransportSystem<Pack> Factory::assemble(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);

namespace detail
{
template TransportSystem<Pack> assemble_stored_scalar<Pack, MeshType>(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);

template TransportSystem<Pack> assemble_stored_corrected_scalar<Pack, MeshType>(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);
} // namespace detail
} // namespace SimpleFluid::FVM
