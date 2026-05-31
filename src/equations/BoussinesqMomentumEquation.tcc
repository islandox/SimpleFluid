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
    const std::vector<vec_type>& old_velocity,
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
    const std::vector<vec_type>& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const field_type& temperature,
    const FvmOperators::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    velocity_field_type& velocity,
    const LinearSolverOptions& linear_options) const
{
    EquationValidation::require_mesh_match(*d_mesh, temperature,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.time_step, "time step",
                                             "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.kinematic_viscosity, "viscosity",
                                             "BoussinesqMomentumEquation");
    EquationValidation::assert_sufficient_cache_size(old_velocity.size(),
                                                     d_mesh->num_local_cells());
    if (velocity_boundary_cache.value.size() != d_mesh->boundary_patches().size()
        || velocity_boundary_cache.mesh != d_mesh)
    {
        throw std::invalid_argument(
            "BoussinesqMomentumEquation received the wrong boundary-cache size.");
    }
    const auto gravity = options.gravity_vector();

    for (std::size_t component = 0;
         component < velocity_field_type::num_components;
         ++component)
    {
        if (d_cached_old_component.size() != d_mesh->num_local_cells())
        {
            d_cached_old_component.resize(d_mesh->num_local_cells());
        }
        for (std::size_t cell = 0; cell < d_mesh->num_local_cells(); ++cell)
        {
            d_cached_old_component[cell] =
                FvmOperators::detail::component_value(old_velocity[cell], component);
        }

        auto boundary_value =
            [&](int boundary_id,
                local_ordinal_type boundary_face_id)
        {
            const auto face = static_cast<std::size_t>(boundary_face_id);

            return FvmOperators::detail::component_value(
                velocity_boundary_cache.value.at(boundary_id)[face], component);
        };

        auto system = FvmOperators::transport_system<Pack>(
            *d_mesh, d_cached_old_component, face_fluxes, options.time_step,
            options.kinematic_viscosity, boundary_value,
            d_cached_transport_matrix);

        if (d_cached_transport_matrix.is_null())
        {
            d_cached_transport_matrix = system.matrix;
        }

        const auto gravity_component =
            FvmOperators::detail::component_value(gravity, component);

        // Build buoyancy as a Tpetra vector and update RHS in one operation.
        typename Pack::vector_type buoyancy_vec(d_mesh->owned_cell_map(), true);
        for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto buoyancy =
                options.thermal_expansion
              * (temperature.value(cell_lid) - options.reference_temperature)
              * (-gravity_component);
            buoyancy_vec.replaceLocalValue(cell_lid,
                                           d_mesh->cell_volume(cell_lid) * buoyancy);
        }
        system.rhs.update(1.0, buoyancy_vec, 1.0);

        if (system.rhs.norm2() <= 0.0)
        {
            for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                velocity.set_owned_component_value(
                    static_cast<local_ordinal_type>(owned), component, 0.0);
            }
            continue;
        }

        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        if (d_cached_solution.is_null())
        {
            d_cached_solution = Teuchos::rcp(
                new typename Pack::vector_type(d_mesh->owned_cell_map(), true));
        }
        else
        {
            d_cached_solution->putScalar(0.0);
        }
        const auto converged =
            solve_linear_system<Pack>(matrix, system.rhs, *d_cached_solution,
                                      linear_options);
        const auto solution_data = d_cached_solution->getData();
        for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto value = solution_data[cell_lid];
            if (!converged && !std::isfinite(value))
            {
                throw std::runtime_error(
                    "BoussinesqMomentumEquation velocity transport solve produced a non-finite value.");
            }
            velocity.set_owned_component_value(cell_lid, component, value);
        }
    }

    velocity.sync_ghosts();
}

template<TpetraTypePack Pack>
void BoussinesqMomentumEquation<Pack>::advance_velocity(
    const std::vector<vec_type>& old_velocity,
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
    const std::vector<vec_type>& old_velocity,
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
