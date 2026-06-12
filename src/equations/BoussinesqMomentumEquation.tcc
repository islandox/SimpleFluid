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

/**
 * @brief Construct a BoussinesqMomentumEquation on the given mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the computational mesh.
 * @throws std::invalid_argument if @p mesh is null.
 */
template<TpetraTypePack Pack>
BoussinesqMomentumEquation<Pack>::BoussinesqMomentumEquation(
    SP<const mesh_type> mesh)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "BoussinesqMomentumEquation"))
{
}

/**
 * @brief Advance the velocity field by one time step with a zero source
 *        term.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_velocity Velocity from the previous time step.
 * @param face_fluxes Pre-computed volumetric face fluxes.
 * @param temperature Scalar temperature field (Boussinesq coupling).
 * @param velocity_boundary_cache Cached velocity boundary conditions.
 * @param options Time-stepping and physical parameters.
 * @param[out] velocity Updated velocity on return.
 * @param linear_options Linear solver configuration.
 */
template<TpetraTypePack Pack>
auto BoussinesqMomentumEquation<Pack>::advance_velocity(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const field_type& temperature,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    velocity_field_type& velocity,
    const LinearSolverOptions& linear_options) const
    -> LinearSolveSummary
{
    auto zero_source =
        [](local_ordinal_type) -> typename velocity_field_type::vec_type
    {
        return {};
    };

    return advance_velocity(old_velocity, face_fluxes, temperature,
                            velocity_boundary_cache, options, velocity,
                            zero_source, linear_options);
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

    return assemble_system(old_velocity, face_fluxes, temperature,
                           velocity_boundary_cache, options, zero_source,
                           correction_field);
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
    EquationValidation::require_mesh_match(*d_mesh, old_velocity,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(*d_mesh, temperature,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.time_step, "time step",
                                             "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.kinematic_viscosity, "viscosity",
                                             "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.n_non_orthogonal_correctors,
                                             "non-orthogonal correctors",
                                             "BoussinesqMomentumEquation");
    if (correction_field != nullptr)
    {
        EquationValidation::require_mesh_match(
            *d_mesh, *correction_field, "BoussinesqMomentumEquation");
    }
    if (velocity_boundary_cache.value.size() != d_mesh->boundary_batches().size()
        || velocity_boundary_cache.type.size() != d_mesh->boundary_batches().size()
        || velocity_boundary_cache.mesh != d_mesh)
    {
        throw std::invalid_argument(
            "BoussinesqMomentumEquation received the wrong boundary-cache size.");
    }

    auto boundary_value =
        [&](int boundary_id,
            local_ordinal_type boundary_face_id)
    {
        const auto face = static_cast<size_t>(boundary_face_id);
        const auto boundary_type =
            velocity_boundary_cache.type.at(boundary_id);
        if (boundary_type == BoundaryConditionType::Slip)
        {
            const auto face_lid =
                d_mesh->boundary_face_batch(boundary_id).face_lids[face];
            return FVM::detail::slip_face_velocity(
                old_velocity, face_lid);
        }

        return velocity_boundary_cache.value.at(boundary_id)[face];
    };

    const auto gravity = options.gravity_vector();
    auto combined_source =
        [&](local_ordinal_type cell_lid) -> typename velocity_field_type::vec_type
    {
        const auto temperature_delta =
            temperature.value(cell_lid) - options.reference_temperature;
        const auto source_scale =
            options.thermal_expansion * temperature_delta;
        const auto external_source = right_hand_source(cell_lid);
        return {
            source_scale * (-gravity.x) + external_source.x,
            source_scale * (-gravity.y) + external_source.y,
            source_scale * (-gravity.z) + external_source.z
        };
    };

    auto system = FVM::non_orthogonal_transport_system<Pack>(
        old_velocity, face_fluxes, options.time_step,
        options.kinematic_viscosity, boundary_value,
        combined_source,
        options.non_orthogonal_treatment,
        correction_field,
        d_cached_transport_matrix);
    d_cached_transport_matrix = system.matrix;
    return system;
}

/**
 * @brief Advance the velocity field by one time step with an explicit
 *        right-hand source term.
 *
 * Validates mesh consistency and parameter bounds, then assembles and
 * solves the momentum transport system including Boussinesq buoyancy.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_velocity Velocity from the previous time step.
 * @param face_fluxes Pre-computed volumetric face fluxes.
 * @param temperature Scalar temperature field (Boussinesq coupling).
 * @param velocity_boundary_cache Cached velocity boundary conditions.
 * @param options Time-stepping and physical parameters.
 * @param[out] velocity Updated velocity on return.
 * @param right_hand_source Per-cell source vector provider.
 * @param linear_options Linear solver configuration.
 * @throws std::invalid_argument on mesh mismatch, negative time step,
 *         negative viscosity, or wrong boundary-cache size.
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
    const LinearSolverOptions& linear_options) const
    -> LinearSolveSummary
{
    EquationValidation::require_mesh_match(*d_mesh, velocity,
                                           "BoussinesqMomentumEquation");

    velocity_field_type old_velocity_snapshot(
        old_velocity.mesh_ptr(), "momentum_old_velocity_snapshot", false);
    const velocity_field_type* transport_old_velocity = &old_velocity;
    if (&old_velocity == &velocity
        && options.n_non_orthogonal_correctors > 0)
    {
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            old_velocity_snapshot.set_value(cell_lid,
                                            old_velocity.value(cell_lid));
        }
        old_velocity_snapshot.sync_ghosts();
        transport_old_velocity = &old_velocity_snapshot;
    }

    LinearSolveSummary summary;
    auto solve_system =
        [&](const auto& system)
    {
        Teuchos::Array<scalar_type> rhs_norms(
            velocity_field_type::num_components);
        system.rhs->norm2(rhs_norms());
        const auto has_nonzero_rhs =
            std::any_of(
                rhs_norms.begin(), rhs_norms.end(),
                [](scalar_type norm)
                {
                    return norm > scalar_type{};
                });
        velocity.owned_data().putScalar(0.0);
        if (!has_nonzero_rhs)
        {
            d_mesh->sync_periodic_boundaries(velocity);
            return;
        }

        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        const auto statistics =
            d_linear_solver.solve_with_statistics(
                matrix, *system.rhs,
                velocity.owned_data(), linear_options);
        summary.add(statistics);
        if (!statistics.converged)
        {
            throw std::runtime_error(
                "BoussinesqMomentumEquation velocity transport solve did not converge.");
        }

        for (size_t component = 0;
             component < velocity_field_type::num_components;
             ++component)
        {
            const auto solution_data = velocity.owned_data().getData(component);
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid = static_cast<local_ordinal_type>(owned);
                if (!std::isfinite(solution_data[cell_lid]))
                {
                    throw std::runtime_error(
                        "BoussinesqMomentumEquation velocity transport solve produced a non-finite value.");
                }
            }
        }

        d_mesh->sync_periodic_boundaries(velocity);
    };

    for (int corrector = 0;
         corrector <= options.n_non_orthogonal_correctors;
         ++corrector)
    {
        const auto* correction_field =
            corrector == 0 ? nullptr : &velocity;
        auto system = assemble_system(
            *transport_old_velocity, face_fluxes, temperature,
            velocity_boundary_cache, options, right_hand_source,
            correction_field);
        solve_system(system);

        if (options.non_orthogonal_treatment
            == FVM::NonOrthogonalTreatment::Implicit)
        {
            break;
        }
    }

    return summary;
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
    const velocity_field_type* correction_field) const -> system_type
{
    EquationValidation::require_mesh_match(
        *d_mesh, old_velocity, "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(
        *d_mesh, temperature, "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(
        *d_mesh, material.density, "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(
        *d_mesh, material.dynamic_viscosity,
        "BoussinesqMomentumEquation");
    if (reference_density <= scalar_type{})
    {
        throw std::invalid_argument(
            "BoussinesqMomentumEquation requires positive reference density.");
    }

    auto boundary_value =
        [&](int boundary_id,
            local_ordinal_type boundary_face_id)
    {
        const auto face = static_cast<size_t>(boundary_face_id);
        if (velocity_boundary_cache.type.at(boundary_id)
            == BoundaryConditionType::Slip)
        {
            const auto face_lid =
                d_mesh->boundary_face_batch(
                    boundary_id).face_lids[face];
            return FVM::detail::slip_face_velocity(
                old_velocity, face_lid);
        }
        return velocity_boundary_cache.value.at(
            boundary_id)[face];
    };

    const auto gravity = options.gravity_vector();
    auto acceleration =
        [&](local_ordinal_type cell_lid)
            -> typename velocity_field_type::vec_type
    {
        typename velocity_field_type::vec_type buoyancy{};
        if (density_feedback_enabled)
        {
            const auto scale =
                (material.density.value(cell_lid)
                 - reference_density)
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

    auto system = FVM::physical_momentum_transport_system<Pack>(
        old_velocity,
        face_fluxes,
        options.time_step,
        material.dynamic_viscosity,
        reference_density,
        boundary_value,
        acceleration,
        options.non_orthogonal_treatment,
        correction_field,
        d_cached_transport_matrix);
    d_cached_transport_matrix = system.matrix;
    return system;
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
    const LinearSolverOptions& linear_options) const
    -> LinearSolveSummary
{
    EquationValidation::require_mesh_match(
        *d_mesh, velocity, "BoussinesqMomentumEquation");

    velocity_field_type old_snapshot(
        old_velocity.mesh_ptr(),
        "physical_momentum_old_velocity",
        false);
    const velocity_field_type* transport_old = &old_velocity;
    if (&old_velocity == &velocity
        && options.n_non_orthogonal_correctors > 0)
    {
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            old_snapshot.set_value(
                cell_lid, old_velocity.value(cell_lid));
        }
        old_snapshot.sync_ghosts();
        transport_old = &old_snapshot;
    }

    LinearSolveSummary summary;
    for (int corrector = 0;
         corrector <= options.n_non_orthogonal_correctors;
         ++corrector)
    {
        const auto* correction_field =
            corrector == 0 ? nullptr : &velocity;
        auto system = assemble_physical_system(
            *transport_old,
            face_fluxes,
            temperature,
            velocity_boundary_cache,
            options,
            material,
            reference_density,
            density_feedback_enabled,
            right_hand_source,
            correction_field);
        velocity.owned_data().putScalar(0.0);
        Teuchos::RCP<const typename Pack::matrix_type> matrix =
            system.matrix;
        const auto statistics =
            d_linear_solver.solve_with_statistics(
                matrix,
                *system.rhs,
                velocity.owned_data(),
                linear_options);
        summary.add(statistics);
        if (!statistics.converged)
        {
            throw std::runtime_error(
                "BoussinesqMomentumEquation physical transport solve did not converge.");
        }
        d_mesh->sync_periodic_boundaries(velocity);
        if (options.non_orthogonal_treatment
            == FVM::NonOrthogonalTreatment::Implicit)
        {
            break;
        }
    }
    return summary;
}

} // namespace SimpleFluid
