/** @file FVM/FieldStoredTransportSystem.tcc
 * @brief Public mapped transport definitions for owning explicit-instantiation units.
 */
#pragma once

#include "FVM/TransportSystem.hh"
#include "FVM/details/FieldStoredTransportSystem.tcc"

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT TransportSystem<Pack> transport_system(const ScalarCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarBoundaryValueProvider<Pack> boundary_value, ScalarCellValueProvider<Pack> right_hand_source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix)
{
    return detail::stored_scalar_transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity,
        std::move(boundary_condition), std::move(boundary_value), std::move(right_hand_source),
        std::move(cached_matrix));
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT TransportSystem<Pack> non_orthogonal_transport_system(
    const ScalarCellFieldStored<Pack, MeshType>& old_values, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> right_hand_source, NonOrthogonalTreatment treatment,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache)
{
    return detail::stored_scalar_non_orthogonal_transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity,
        std::move(boundary_condition), std::move(boundary_value), std::move(right_hand_source), treatment,
        correction_field, std::move(cached_matrix), geometry_cache);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT TransportSystem<Pack> weighted_scalar_transport_system(
    FieldStoredWeightedScalarTransportRequest<Pack, MeshType> request)
{
    return detail::stored_weighted_scalar_transport_system<Pack>(request.old_values, request.face_fluxes,
        request.time_step, request.storage_weight, request.advection_weight, request.diffusivity,
        std::move(request.boundary_condition), std::move(request.boundary_value), std::move(request.source),
        request.treatment, request.correction_field, std::move(request.cached_matrix), std::move(request.implicit_sink),
        std::move(request.fixed_cell_value), request.boundary_diffusivity, request.geometry_cache,
        request.coefficient_interpolation, request.discretization, request.older_values, request.old_storage_weight,
        request.ale, request.symbolic_plan, std::move(request.cached_rhs));
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT TransportSystem<Pack> physical_temperature_transport_system(
    const ScalarCellFieldStored<Pack, MeshType>& old_temperature,
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
    AssemblyMode mode)
{
    return detail::stored_physical_temperature_transport_system<Pack>(old_temperature, face_fluxes, time_step, density,
        specific_heat_capacity, thermal_conductivity, std::move(boundary_condition), std::move(boundary_value),
        std::move(power_density), treatment, correction_field, std::move(cached_matrix), boundary_thermal_conductivity,
        geometry_cache, coefficient_interpolation, ScalarTransportDiscretization{.mode = mode},
        static_cast<const ScalarCellFieldStored<Pack, MeshType>*>(nullptr), old_density, old_specific_heat_capacity,
        ale, rhs_only);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT TransportSystem<Pack> physical_temperature_transport_system(
    const ScalarCellFieldStored<Pack, MeshType>& old_temperature,
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
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* old_specific_heat_capacity)
{
    return detail::stored_physical_temperature_transport_system<Pack>(old_temperature, face_fluxes, time_step, density,
        specific_heat_capacity, thermal_conductivity, std::move(boundary_condition), std::move(boundary_value),
        std::move(power_density), treatment, correction_field, std::move(cached_matrix), boundary_thermal_conductivity,
        geometry_cache, coefficient_interpolation, discretization, older_temperature, old_density,
        old_specific_heat_capacity, ale);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT VectorTransportSystem<Pack> non_orthogonal_transport_system(
    const VectorCellFieldStored<Pack, MeshType>& old_values, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryValueProvider<Pack> boundary_value, VectorCellValueProvider<Pack> right_hand_source,
    NonOrthogonalTreatment treatment,
    const std::type_identity_t<VectorCellFieldStored<Pack, MeshType>>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, BoundaryFaceSelector boundary_diffusion,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache, const ALEControlVolumeState* ale)
{
    return detail::stored_vector_transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity,
        std::move(boundary_value), std::move(right_hand_source), treatment, correction_field, std::move(cached_matrix),
        std::move(boundary_diffusion), geometry_cache, ale);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT VectorTransportSystem<Pack> physical_momentum_transport_system(
    const VectorCellFieldStored<Pack, MeshType>& old_velocity, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, const ScalarCellFieldStored<Pack, MeshType>& dynamic_viscosity,
    typename Pack::scalar_type reference_density, VectorBoundaryValueProvider<Pack> boundary_value,
    VectorCellValueProvider<Pack> acceleration_source, NonOrthogonalTreatment treatment,
    const std::type_identity_t<VectorCellFieldStored<Pack, MeshType>>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, BoundaryFaceSelector boundary_diffusion,
    const std::type_identity_t<FieldStoredBoundaryCache<Pack, MeshType>>* boundary_dynamic_viscosity,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation, const ALEControlVolumeState* ale,
    std::type_identity_t<TensorCellFieldStored<Pack, MeshType>>* gradient_workspace,
    BoundaryFaceSelector zero_gradient_stress, AssemblyMode mode)
{
    return detail::stored_physical_momentum_transport_system<Pack>(old_velocity, face_fluxes, time_step,
        dynamic_viscosity, reference_density, std::move(boundary_value), std::move(acceleration_source), treatment,
        correction_field, std::move(cached_matrix), std::move(boundary_diffusion), boundary_dynamic_viscosity,
        geometry_cache, coefficient_interpolation, ale, gradient_workspace, std::move(zero_gradient_stress), mode);
}

} // namespace SimpleFluid::FVM
