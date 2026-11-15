/** @brief Individual compiled transport factory overloads. */
#include "utils/export_helper.hh"
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

template TransportSystem<Pack> Factory::assemble(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);

template VectorTransportSystem<Pack> Factory::assemble(
    Factory::VectorTerms, Factory::BasicVectorBoundary, Factory::BasicVectorOptions, Factory::BasicVectorContext);

template VectorTransportSystem<Pack> Factory::assemble(
    Factory::VectorTerms, Factory::VelocityBoundary, Factory::VectorOptions, Factory::VectorContext);

namespace detail
{
template TransportSystem<Pack> assemble_legacy_basic_scalar<Pack>(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);

template TransportSystem<Pack> assemble_legacy_corrected_scalar<Pack>(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);

template VectorTransportSystem<Pack> assemble_legacy_basic_vector<Pack>(
    Factory::VectorTerms, Factory::BasicVectorBoundary, Factory::BasicVectorOptions, Factory::BasicVectorContext);

template VectorTransportSystem<Pack> assemble_legacy_corrected_vector<Pack>(
    Factory::VectorTerms, Factory::VelocityBoundary, Factory::VectorOptions, Factory::VectorContext);
} // namespace detail

} // namespace SimpleFluid::FVM

SIMPLEFLUID_LEGACY_API_END
