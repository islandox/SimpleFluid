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

template VectorTransportSystem<Pack> Factory::assemble(
    Factory::MomentumTerms, Factory::MomentumBoundary, Factory::MomentumOptions, Factory::MomentumContext);

namespace detail
{
template VectorTransportSystem<Pack> assemble_stored_momentum<Pack, MeshType>(
    Factory::MomentumTerms, Factory::MomentumBoundary, Factory::MomentumOptions, Factory::MomentumContext);
} // namespace detail
} // namespace SimpleFluid::FVM
