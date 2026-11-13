/** @brief Compiled mapped transport provider: VectorPartitionedCartesian. */
#include "FVM/FieldStoredTransportSystem.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"

// This static mesh type is intentionally hidden. Export only these public
// function specializations; standard attributes on explicit instantiations
// are rejected by Clang, so use its supported GNU attribute spelling here.
#if defined(__GNUC__) || defined(__clang__)
#define SIMPLEFLUID_STATIC_TRANSPORT_EXPORT __attribute__((visibility("default")))
#else
#define SIMPLEFLUID_STATIC_TRANSPORT_EXPORT
#endif

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
} // namespace

template SIMPLEFLUID_STATIC_TRANSPORT_EXPORT VectorTransportSystem<Pack>
non_orthogonal_transport_system<Pack, MeshType>(const VectorCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, VectorBoundaryValueProvider<Pack> boundary_value,
    VectorCellValueProvider<Pack> right_hand_source, NonOrthogonalTreatment treatment,
    const std::type_identity_t<VectorCellFieldStored<Pack, MeshType>>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, BoundaryFaceSelector boundary_diffusion,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache, const ALEControlVolumeState* ale);

} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_TRANSPORT_EXPORT
