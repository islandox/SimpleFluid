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

template VectorTransportSystem<Pack> Factory::assemble(
    Factory::MomentumTerms, Factory::MomentumBoundary, Factory::MomentumOptions, Factory::MomentumContext);

} // namespace SimpleFluid::FVM
