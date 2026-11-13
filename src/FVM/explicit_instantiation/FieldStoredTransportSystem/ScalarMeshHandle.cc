/** @brief Compiled mapped transport provider: ScalarMeshHandle. */
#include "FVM/FieldStoredTransportSystem.tcc"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = MeshHandle<Pack>;
} // namespace

template TransportSystem<Pack> transport_system<Pack, MeshType>(const ScalarCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarBoundaryValueProvider<Pack> boundary_value, ScalarCellValueProvider<Pack> right_hand_source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix);

template TransportSystem<Pack> non_orthogonal_transport_system<Pack, MeshType>(
    const ScalarCellFieldStored<Pack, MeshType>& old_values, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> right_hand_source, NonOrthogonalTreatment treatment,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache);

} // namespace SimpleFluid::FVM
