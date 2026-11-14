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
TransportSystem<Pack> Factory::assemble(
    ScalarTerms terms, ScalarBoundary boundary, ScalarOptions options, ScalarContext context)
{
    const int selected = options.correction.has_value() ? 1 : 0;
    const auto correction_selection = detail::reduce_transport_validation_state(
        terms.history.old_values.mesh(), std::array<int, 2>{selected, -selected});
    if (correction_selection[0] != -correction_selection[1])
        throw std::invalid_argument("Scalar transport requires every rank to select the same correction category.");
    if (!terms.source)
        terms.source = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    if (options.correction)
        return detail::assemble_stored_corrected_scalar<Pack, MeshType>(
            std::move(terms), std::move(boundary), options, std::move(context));
    return detail::assemble_stored_scalar<Pack, MeshType>(
        std::move(terms), std::move(boundary), options, std::move(context));
}

} // namespace SimpleFluid::FVM
