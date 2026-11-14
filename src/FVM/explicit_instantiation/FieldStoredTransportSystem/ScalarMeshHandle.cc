/** @brief Individual compiled transport factory overloads. */
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

} // namespace SimpleFluid::FVM
