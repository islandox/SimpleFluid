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

template VectorTransportSystem<Pack> Factory::assemble(
    Factory::VectorTerms, Factory::VelocityBoundary, Factory::VectorOptions, Factory::VectorContext);

} // namespace SimpleFluid::FVM
