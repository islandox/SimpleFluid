/**
 * @file SolidHeatConductionEquation.tcc
 * @brief Template implementations for solid heat conduction.
 */

#include "SolidHeatConductionEquation.hh"

namespace SimpleFluid
{

template<TpetraTypePack Pack, class MeshType>
SolidHeatConductionEquation<Pack, MeshType>::SolidHeatConductionEquation(
    SP<const mesh_type> mesh, const BoundaryConditionSet& boundary_conditions)
    : d_mesh(EquationValidation::require_non_null_mesh(std::move(mesh), "SolidHeatConductionEquation")),
      d_zero_face_flux(d_mesh, scalar_type{}, "solid_zero_face_flux"),
      d_temperature_equation(
          std::make_unique<TemperatureDiffusionEquation<Pack, mesh_type>>(d_mesh, boundary_conditions))
{
}

template<TpetraTypePack Pack, class MeshType>
void SolidHeatConductionEquation<Pack, MeshType>::set_boundary_conditions(
    const BoundaryConditionSet& boundary_conditions)
{
    d_temperature_equation =
        std::make_unique<TemperatureDiffusionEquation<Pack, mesh_type>>(d_mesh, boundary_conditions);
}

template<TpetraTypePack Pack, class MeshType>
auto SolidHeatConductionEquation<Pack, MeshType>::advance(const field_type& old_temperature, scalar_type time_step,
    const material_type& material, field_type& temperature, FVM::NonOrthogonalTreatment treatment,
    const LinearSolverOptions& linear_options, const field_type* thermal_conductivity_override,
    const boundary_cache_type* boundary_thermal_conductivity,
    FVM::FaceCoefficientInterpolation coefficient_interpolation) const -> LinearSolveStatistics
{
    const auto zero_power_density = [](local_ordinal_type) -> scalar_type { return scalar_type{}; };
    return advance(old_temperature, time_step, material, temperature, zero_power_density, treatment, linear_options,
        thermal_conductivity_override, boundary_thermal_conductivity, coefficient_interpolation);
}

template<TpetraTypePack Pack, class MeshType>
auto SolidHeatConductionEquation<Pack, MeshType>::advance(const field_type& old_temperature, scalar_type time_step,
    const material_type& material, field_type& temperature, const source_type& power_density,
    FVM::NonOrthogonalTreatment treatment, const LinearSolverOptions& linear_options,
    const field_type* thermal_conductivity_override, const boundary_cache_type* boundary_thermal_conductivity,
    FVM::FaceCoefficientInterpolation coefficient_interpolation) const -> LinearSolveStatistics
{
    return d_temperature_equation->advance_physical(old_temperature, d_zero_face_flux, time_step, material, temperature,
        power_density, treatment, linear_options, thermal_conductivity_override, boundary_thermal_conductivity,
        coefficient_interpolation);
}

} // namespace SimpleFluid
