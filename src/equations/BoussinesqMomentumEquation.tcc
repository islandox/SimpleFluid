/**
 * @file BoussinesqMomentumEquation.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template method implementations for BoussinesqMomentumEquation.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "BoussinesqMomentumEquation.hh"

namespace SimpleFluid
{

template<TpetraTypePack Pack>
BoussinesqMomentumEquation<Pack>::BoussinesqMomentumEquation(
    SP<const mesh_type> mesh)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "BoussinesqMomentumEquation"))
{
}

template<TpetraTypePack Pack>
void BoussinesqMomentumEquation<Pack>::advance_velocity(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const field_type& temperature,
    const BoundaryConditionSet& boundary_conditions,
    const TimeStepperOptions& options,
    velocity_field_type& velocity,
    const LinearSolverOptions& linear_options) const
{
    const auto cache =
        FvmOperators::cache_velocity_boundary_conditions<Pack>(
            d_mesh, boundary_conditions);
    advance_velocity(old_velocity, face_fluxes, temperature, cache, options,
                     velocity, linear_options);
}

template<TpetraTypePack Pack>
void BoussinesqMomentumEquation<Pack>::advance_velocity(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const field_type& temperature,
    const FvmOperators::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    velocity_field_type& velocity,
    const LinearSolverOptions& linear_options) const
{
    EquationValidation::require_mesh_match(*d_mesh, old_velocity,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(*d_mesh, temperature,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.time_step, "time step",
                                             "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.kinematic_viscosity, "viscosity",
                                             "BoussinesqMomentumEquation");
    if (velocity_boundary_cache.value.size() != d_mesh->boundary_patches().size()
        || velocity_boundary_cache.mesh != d_mesh)
    {
        throw std::invalid_argument(
            "BoussinesqMomentumEquation received the wrong boundary-cache size.");
    }
    auto boundary_value =
        [&](int boundary_id,
            local_ordinal_type boundary_face_id)
    {
        const auto face = static_cast<std::size_t>(boundary_face_id);

        return velocity_boundary_cache.value.at(boundary_id)[face];
    };

    auto system = FvmOperators::transport_system<Pack>(
        old_velocity, face_fluxes, options.time_step,
        options.kinematic_viscosity, boundary_value,
        d_cached_transport_matrix);

    if (d_cached_transport_matrix.is_null())
    {
        d_cached_transport_matrix = system.matrix;
    }

    const auto gravity = options.gravity_vector();
    for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto temperature_delta =
            temperature.value(cell_lid) - options.reference_temperature;
        const auto volume = d_mesh->cell_volume(cell_lid);
        for (std::size_t component = 0;
             component < velocity_field_type::num_components;
             ++component)
        {
            const auto gravity_component =
                FvmOperators::detail::component_value(gravity, component);
            const auto buoyancy =
                options.thermal_expansion
              * temperature_delta
              * (-gravity_component);
            system.rhs->sumIntoLocalValue(cell_lid, component,
                                          volume * buoyancy);
        }
    }

    bool has_nonzero_rhs = false;
    for (std::size_t component = 0;
         component < velocity_field_type::num_components && !has_nonzero_rhs;
         ++component)
    {
        const auto rhs_data = system.rhs->getData(component);
        for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            if (std::abs(rhs_data[cell_lid]) > 0.0)
            {
                has_nonzero_rhs = true;
                break;
            }
        }
    }

    velocity.owned_data().putScalar(0.0);
    if (!has_nonzero_rhs)
    {
        velocity.sync_ghosts();
        return;
    }

    Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
    const auto converged =
        solve_linear_system<Pack>(matrix, *system.rhs,
                                  velocity.owned_data(), linear_options);
    if (!converged)
    {
        for (std::size_t component = 0;
             component < velocity_field_type::num_components;
             ++component)
        {
            const auto solution_data = velocity.owned_data().getData(component);
            for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid = static_cast<local_ordinal_type>(owned);
                if (!std::isfinite(solution_data[cell_lid]))
                {
                    throw std::runtime_error(
                        "BoussinesqMomentumEquation velocity transport solve produced a non-finite value.");
                }
            }
        }
    }

    velocity.sync_ghosts();
}

template<TpetraTypePack Pack>
void BoussinesqMomentumEquation<Pack>::advance_velocity(
    const velocity_field_type& old_velocity,
    const face_velocity_field_type& face_velocity,
    const field_type& temperature,
    const BoundaryConditionSet& boundary_conditions,
    const TimeStepperOptions& options,
    velocity_field_type& velocity,
    const LinearSolverOptions& linear_options) const
{
    const auto cache =
        FvmOperators::cache_velocity_boundary_conditions<Pack>(
            d_mesh, boundary_conditions);
    advance_velocity(old_velocity, face_velocity, temperature, cache, options,
                     velocity, linear_options);
}

template<TpetraTypePack Pack>
void BoussinesqMomentumEquation<Pack>::advance_velocity(
    const velocity_field_type& old_velocity,
    const face_velocity_field_type& face_velocity,
    const field_type& temperature,
    const FvmOperators::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    velocity_field_type& velocity,
    const LinearSolverOptions& linear_options) const
{
    const auto face_fluxes =
        FvmOperators::normal_face_fluxes<Pack>(*d_mesh, face_velocity);
    advance_velocity(old_velocity, face_fluxes, temperature, velocity_boundary_cache,
                     options, velocity, linear_options);
}

} // namespace SimpleFluid
