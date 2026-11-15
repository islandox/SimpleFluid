/** @brief Individual compiled transport factory overloads. */
#include "SimpleFluidExport.hh"
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

template VectorTransportSystem<Pack> Factory::assemble(
    Factory::MomentumTerms, Factory::MomentumBoundary, Factory::MomentumOptions, Factory::MomentumContext);

namespace detail
{
template VectorTransportSystem<Pack> assemble_legacy_momentum<Pack>(
    Factory::MomentumTerms, Factory::MomentumBoundary, Factory::MomentumOptions, Factory::MomentumContext);
} // namespace detail

} // namespace SimpleFluid::FVM

SIMPLEFLUID_LEGACY_API_END
