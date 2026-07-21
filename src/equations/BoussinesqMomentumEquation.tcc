/**
 * @file BoussinesqMomentumEquation.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementations for BoussinesqMomentumEquation.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "BoussinesqMomentumEquation.hh"

#include <cmath>
#include <utility>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
auto BoussinesqMomentumEquation<Pack>::select_dynamic_viscosity(
    const MaterialPropertyFields<Pack>& material,
    const field_type* dynamic_viscosity_override) const -> const field_type&
{
    if (dynamic_viscosity_override == nullptr)
    {
        return material.dynamic_viscosity;
    }

    EquationValidation::require_mesh_match(
        this->mesh(), *dynamic_viscosity_override,
        "BoussinesqMomentumEquation");
    for (size_t owned = 0; owned < this->mesh().num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto value = dynamic_viscosity_override->value(cell_lid);
        if (!std::isfinite(value) || value < scalar_type{})
        {
            throw std::invalid_argument(
                "BoussinesqMomentumEquation dynamic-viscosity override "
                "must contain finite non-negative values.");
        }
    }
    return *dynamic_viscosity_override;
}

template<TpetraTypePack Pack>
BoussinesqMomentumEquation<Pack>::BoussinesqMomentumEquation(
    SP<const mesh_type> mesh)
    : base_type(std::move(mesh))
{
}

template<TpetraTypePack Pack>
auto BoussinesqMomentumEquation<Pack>::advance_velocity(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const field_type& temperature,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    velocity_field_type& velocity,
    const LinearSolverOptions& linear_options) const -> LinearSolveSummary
{
    auto zero_source =
        [](local_ordinal_type) -> typename velocity_field_type::vec_type
    {
        return {};
    };
    return advance_velocity(
        old_velocity, face_fluxes, temperature, velocity_boundary_cache,
        options, velocity, zero_source, linear_options);
}

template<TpetraTypePack Pack>
auto BoussinesqMomentumEquation<Pack>::assemble_system(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const field_type& temperature,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const velocity_field_type* correction_field) const -> system_type
{
    auto zero_source =
        [](local_ordinal_type) -> typename velocity_field_type::vec_type
    {
        return {};
    };
    return assemble_system(
        old_velocity, face_fluxes, temperature, velocity_boundary_cache,
        options, zero_source, correction_field);
}

template<TpetraTypePack Pack>
auto BoussinesqMomentumEquation<Pack>::assemble_system(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const field_type& temperature,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const source_type& right_hand_source,
    const velocity_field_type* correction_field) const -> system_type
{
    EquationValidation::require_mesh_match(
        this->mesh(), temperature, "BoussinesqMomentumEquation");

    const auto gravity = options.gravity_vector();
    auto combined_source =
        [&](local_ordinal_type cell_lid)
            -> typename velocity_field_type::vec_type
    {
        const auto source_scale =
            options.thermal_expansion
          * (temperature.value(cell_lid)
             - options.reference_temperature);
        return gravity * (-source_scale) + right_hand_source(cell_lid);
    };

    return base_type::assemble_system(
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        combined_source, correction_field);
}

template<TpetraTypePack Pack>
auto BoussinesqMomentumEquation<Pack>::advance_velocity(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const field_type& temperature,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    velocity_field_type& velocity,
    const source_type& right_hand_source,
    const LinearSolverOptions& linear_options) const -> LinearSolveSummary
{
    EquationValidation::require_mesh_match(
        this->mesh(), temperature, "BoussinesqMomentumEquation");

    const auto gravity = options.gravity_vector();
    auto combined_source =
        [&](local_ordinal_type cell_lid)
            -> typename velocity_field_type::vec_type
    {
        const auto source_scale =
            options.thermal_expansion
          * (temperature.value(cell_lid)
             - options.reference_temperature);
        return gravity * (-source_scale) + right_hand_source(cell_lid);
    };

    return base_type::advance_velocity(
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        velocity, combined_source, linear_options);
}

template<TpetraTypePack Pack>
auto BoussinesqMomentumEquation<Pack>::assemble_physical_system(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const field_type& temperature,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const MaterialPropertyFields<Pack>& material,
    scalar_type reference_density,
    bool density_feedback_enabled,
    const source_type& right_hand_source,
    const velocity_field_type* correction_field,
    const field_type* dynamic_viscosity_override,
    const FVM::BoundaryCache<Pack>* boundary_dynamic_viscosity) const
    -> system_type
{
    EquationValidation::require_mesh_match(
        this->mesh(), temperature, "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(
        this->mesh(), material.density, "BoussinesqMomentumEquation");
    const auto& dynamic_viscosity = select_dynamic_viscosity(
        material, dynamic_viscosity_override);

    const auto gravity = options.gravity_vector();
    auto acceleration =
        [&](local_ordinal_type cell_lid)
            -> typename velocity_field_type::vec_type
    {
        typename velocity_field_type::vec_type buoyancy{};
        if (density_feedback_enabled)
        {
            const auto scale =
                (material.density.value(cell_lid) - reference_density)
              / reference_density;
            buoyancy = gravity * scale;
        }
        else
        {
            const auto scale =
                options.thermal_expansion
              * (temperature.value(cell_lid)
                 - options.reference_temperature);
            buoyancy = gravity * (-scale);
        }
        return buoyancy + right_hand_source(cell_lid);
    };

    return base_type::assemble_physical_system(
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        dynamic_viscosity, reference_density, acceleration,
        correction_field, boundary_dynamic_viscosity);
}

template<TpetraTypePack Pack>
auto BoussinesqMomentumEquation<Pack>::advance_velocity_physical(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const field_type& temperature,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const MaterialPropertyFields<Pack>& material,
    scalar_type reference_density,
    bool density_feedback_enabled,
    velocity_field_type& velocity,
    const source_type& right_hand_source,
    const LinearSolverOptions& linear_options,
    const field_type* dynamic_viscosity_override,
    const FVM::BoundaryCache<Pack>* boundary_dynamic_viscosity) const
    -> LinearSolveSummary
{
    EquationValidation::require_mesh_match(
        this->mesh(), temperature, "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(
        this->mesh(), material.density, "BoussinesqMomentumEquation");
    const auto& dynamic_viscosity = select_dynamic_viscosity(
        material, dynamic_viscosity_override);

    const auto gravity = options.gravity_vector();
    auto acceleration =
        [&](local_ordinal_type cell_lid)
            -> typename velocity_field_type::vec_type
    {
        typename velocity_field_type::vec_type buoyancy{};
        if (density_feedback_enabled)
        {
            const auto scale =
                (material.density.value(cell_lid) - reference_density)
              / reference_density;
            buoyancy = gravity * scale;
        }
        else
        {
            const auto scale =
                options.thermal_expansion
              * (temperature.value(cell_lid)
                 - options.reference_temperature);
            buoyancy = gravity * (-scale);
        }
        return buoyancy + right_hand_source(cell_lid);
    };

    return base_type::advance_velocity_physical(
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        dynamic_viscosity, reference_density, velocity,
        acceleration, linear_options, boundary_dynamic_viscosity);
}

} // namespace SimpleFluid
