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
    MomentumTerms terms, MomentumBoundary boundary, MomentumOptions options, MomentumContext context)
{
    return detail::assemble_factory_momentum<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

namespace detail
{
template VectorTransportSystem<Pack> assemble_stored_momentum<Pack, MeshType>(
    Factory::MomentumTerms, Factory::MomentumBoundary, Factory::MomentumOptions, Factory::MomentumContext);
} // namespace detail
} // namespace SimpleFluid::FVM
