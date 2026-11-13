/** @brief Compiled mapped transport provider: VectorMeshHandle. */
#include "FVM/FieldStoredTransportSystem.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = MeshHandle<Pack>;
} // namespace

template VectorTransportSystem<Pack> non_orthogonal_transport_system<Pack, MeshType>(
    const VectorCellFieldStored<Pack, MeshType>& old_values, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryValueProvider<Pack> boundary_value, VectorCellValueProvider<Pack> right_hand_source,
    NonOrthogonalTreatment treatment,
    const std::type_identity_t<VectorCellFieldStored<Pack, MeshType>>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, BoundaryFaceSelector boundary_diffusion,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache, const ALEControlVolumeState* ale);

} // namespace SimpleFluid::FVM
