/** @brief Individual compiled transport factory overloads. */
#include "FVM/TransportSystemFactory.tcc"
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
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);

template VectorTransportSystem<Pack> Factory::assemble(
    Factory::VectorTerms, Factory::BasicVectorBoundary, Factory::BasicVectorOptions, Factory::BasicVectorContext);

template VectorTransportSystem<Pack> Factory::assemble(
    Factory::VectorTerms, Factory::VelocityBoundary, Factory::VectorOptions, Factory::VectorContext);

} // namespace SimpleFluid::FVM
