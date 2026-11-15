/** @brief Exported static-Cartesian transport factory specializations. */
#include "FVM/details/FieldStoredTransportSystem.tcc"
#include "FVM/TransportSystemFactory.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
using Factory = TransportSystemFactory<Pack, MeshType>;
} // namespace

template<>
TransportSystem<Pack> Factory::assemble(
    ScalarTerms terms, ScalarBoundary boundary, ScalarOptions options, ScalarContext context)
{
    return detail::assemble_factory_scalar<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

namespace detail
{
template TransportSystem<Pack> assemble_stored_scalar<Pack, MeshType>(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);

template TransportSystem<Pack> assemble_stored_corrected_scalar<Pack, MeshType>(
    Factory::ScalarTerms, Factory::ScalarBoundary, Factory::ScalarOptions, Factory::ScalarContext);
} // namespace detail
} // namespace SimpleFluid::FVM
