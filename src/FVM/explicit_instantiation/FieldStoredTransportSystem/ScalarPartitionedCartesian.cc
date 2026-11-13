/** @brief Compiled mapped transport provider: ScalarPartitionedCartesian. */
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

template SIMPLEFLUID_STATIC_TRANSPORT_EXPORT TransportSystem<Pack> transport_system<Pack, MeshType>(
    const ScalarCellFieldStored<Pack, MeshType>& old_values, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> right_hand_source, Teuchos::RCP<typename Pack::matrix_type> cached_matrix);

template SIMPLEFLUID_STATIC_TRANSPORT_EXPORT TransportSystem<Pack> non_orthogonal_transport_system<Pack, MeshType>(
    const ScalarCellFieldStored<Pack, MeshType>& old_values, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> right_hand_source, NonOrthogonalTreatment treatment,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache);

} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_TRANSPORT_EXPORT
