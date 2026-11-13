/** @brief Compiled mapped transport provider: WeightedTemperaturePartitionedCartesian. */
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

template SIMPLEFLUID_STATIC_TRANSPORT_EXPORT TransportSystem<Pack> weighted_scalar_transport_system<Pack, MeshType>(
    FieldStoredWeightedScalarTransportRequest<Pack, MeshType> request);

template SIMPLEFLUID_STATIC_TRANSPORT_EXPORT TransportSystem<Pack>
physical_temperature_transport_system<Pack, MeshType>(const ScalarCellFieldStored<Pack, MeshType>& old_temperature,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    const ScalarCellFieldStored<Pack, MeshType>& density,
    const ScalarCellFieldStored<Pack, MeshType>& specific_heat_capacity,
    const ScalarCellFieldStored<Pack, MeshType>& thermal_conductivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> power_density, NonOrthogonalTreatment treatment,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    const std::type_identity_t<FieldStoredBoundaryCache<Pack, MeshType>>* boundary_thermal_conductivity,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation, const ALEControlVolumeState* ale,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* old_density,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* old_specific_heat_capacity, bool rhs_only,
    AssemblyMode mode);

template SIMPLEFLUID_STATIC_TRANSPORT_EXPORT TransportSystem<Pack>
physical_temperature_transport_system<Pack, MeshType>(const ScalarCellFieldStored<Pack, MeshType>& old_temperature,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    const ScalarCellFieldStored<Pack, MeshType>& density,
    const ScalarCellFieldStored<Pack, MeshType>& specific_heat_capacity,
    const ScalarCellFieldStored<Pack, MeshType>& thermal_conductivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> power_density, ScalarTransportDiscretization discretization,
    NonOrthogonalTreatment treatment,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* older_temperature,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    const std::type_identity_t<FieldStoredBoundaryCache<Pack, MeshType>>* boundary_thermal_conductivity,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation, const ALEControlVolumeState* ale,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* old_density,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* old_specific_heat_capacity);

} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_TRANSPORT_EXPORT
