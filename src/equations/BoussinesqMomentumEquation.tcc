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

/**
 * @brief Select and validate the dynamic-viscosity field used by momentum.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param material Material fields supplying the default viscosity.
 * @param dynamic_viscosity_override Optional effective-viscosity field.
 * @return Selected viscosity field.
 * @throws std::invalid_argument if the override is mismatched or invalid.
 */
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

/**
 * @brief Construct a Boussinesq momentum equation on a mesh.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param mesh Computational mesh.
 * @throws std::invalid_argument if @p mesh is null.
 */
template<TpetraTypePack Pack>
BoussinesqMomentumEquation<Pack>::BoussinesqMomentumEquation(
    SP<const mesh_type> mesh)
    : base_type(std::move(mesh))
{
}

/**
 * @brief Advance velocity with buoyancy and no additional source.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param temperature Cell temperature driving buoyancy.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Time-step and Boussinesq parameters.
 * @param[out] velocity Updated velocity field.
 * @param linear_options Linear-solver configuration.
 * @return Aggregated linear-solve statistics.
 * @throws std::invalid_argument if input fields or options are inconsistent.
 * @throws std::runtime_error if a velocity solve fails.
 */
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

/**
 * @brief Assemble the Boussinesq momentum system without an extra source.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param temperature Cell temperature driving buoyancy.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Time-step and Boussinesq parameters.
 * @param correction_field Optional lagged non-orthogonal correction field.
 * @return Assembled vector transport system.
 * @throws std::invalid_argument if input fields or options are inconsistent.
 */
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

/**
 * @brief Assemble the Boussinesq momentum system with a caller source.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param temperature Cell temperature driving buoyancy.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Time-step and Boussinesq parameters.
 * @param right_hand_source Additional per-cell acceleration.
 * @param correction_field Optional lagged non-orthogonal correction field.
 * @return Assembled vector transport system.
 * @throws std::invalid_argument if input fields or options are inconsistent.
 */
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

/**
 * @brief Advance velocity with buoyancy and a caller acceleration source.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param temperature Cell temperature driving buoyancy.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Time-step and Boussinesq parameters.
 * @param[out] velocity Updated velocity field.
 * @param right_hand_source Additional per-cell acceleration.
 * @param linear_options Linear-solver configuration.
 * @return Aggregated linear-solve statistics.
 * @throws std::invalid_argument if input fields or options are inconsistent.
 * @throws std::runtime_error if a velocity solve fails.
 */
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

/**
 * @brief Assemble physical momentum transport with material feedback.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param temperature Cell temperature driving Boussinesq buoyancy.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Time-step and Boussinesq parameters.
 * @param material Physical material-property fields.
 * @param reference_density Density used to normalize momentum diffusion.
 * @param density_feedback_enabled Whether density replaces thermal buoyancy.
 * @param right_hand_source Additional per-cell acceleration.
 * @param correction_field Optional lagged non-orthogonal correction field.
 * @param dynamic_viscosity_override Optional effective-viscosity field.
 * @param boundary_dynamic_viscosity Optional boundary viscosity cache.
 * @return Assembled physical momentum system.
 * @throws std::invalid_argument if fields or physical inputs are invalid.
 */
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

/**
 * @brief Advance physical momentum with material-dependent buoyancy.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param temperature Cell temperature driving Boussinesq buoyancy.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Time-step and Boussinesq parameters.
 * @param material Physical material-property fields.
 * @param reference_density Density used to normalize momentum diffusion.
 * @param density_feedback_enabled Whether density replaces thermal buoyancy.
 * @param[out] velocity Updated velocity field.
 * @param right_hand_source Additional per-cell acceleration.
 * @param linear_options Linear-solver configuration.
 * @param dynamic_viscosity_override Optional effective-viscosity field.
 * @param boundary_dynamic_viscosity Optional boundary viscosity cache.
 * @return Aggregated linear-solve statistics.
 * @throws std::invalid_argument if fields or physical inputs are invalid.
 * @throws std::runtime_error if a velocity solve fails.
 */
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
