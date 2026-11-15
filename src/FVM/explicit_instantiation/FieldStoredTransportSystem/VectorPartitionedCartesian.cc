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
VectorTransportSystem<Pack> Factory::assemble(
    VectorTerms terms, VelocityBoundary boundary, VectorOptions options, VectorContext context)
{
    return detail::assemble_factory_vector<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

namespace detail
{
template VectorTransportSystem<Pack> assemble_stored_vector<Pack, MeshType>(
    Factory::VectorTerms, Factory::VelocityBoundary, Factory::VectorOptions, Factory::VectorContext);
} // namespace detail
} // namespace SimpleFluid::FVM
