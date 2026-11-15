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
    Factory::VectorTerms, Factory::VelocityBoundary, Factory::VectorOptions, Factory::VectorContext);

namespace detail
{
template VectorTransportSystem<Pack> assemble_stored_vector<Pack, MeshType>(
    Factory::VectorTerms, Factory::VelocityBoundary, Factory::VectorOptions, Factory::VectorContext);
} // namespace detail
} // namespace SimpleFluid::FVM
