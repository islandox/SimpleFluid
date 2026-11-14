/** @brief Exported static-Cartesian transport factory specializations. */
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
    if (!terms.source)
        terms.source = [](typename Pack::local_ordinal_type) { return typename VectorCellField<Pack>::vec_type{}; };
    return detail::assemble_stored_vector<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

} // namespace SimpleFluid::FVM
