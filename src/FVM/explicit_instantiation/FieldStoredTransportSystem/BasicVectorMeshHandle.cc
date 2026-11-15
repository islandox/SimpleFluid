/** @brief Native full-boundary single-matrix vector factory provider. */
#include "FVM/TransportSystemFactory.tcc"
#include "FVM/details/BasicVectorTransportSystem.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Mesh = MeshHandle<Pack>;
using Factory = TransportSystemFactory<Pack, Mesh>;
} // namespace

template VectorTransportSystem<Pack> Factory::assemble(
    Factory::VectorTerms, Factory::BasicVectorBoundary, Factory::BasicVectorOptions, Factory::BasicVectorContext);

namespace detail
{
template VectorTransportSystem<Pack> assemble_stored_basic_vector<Pack, Mesh>(
    Factory::VectorTerms, Factory::BasicVectorBoundary, Factory::BasicVectorOptions, Factory::BasicVectorContext);
} // namespace detail
} // namespace SimpleFluid::FVM
