/**
 * @file IncompressibleMomentumEquation.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementations for IncompressibleMomentumEquation.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "IncompressibleMomentumEquation.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace SimpleFluid
{

/**
 * @brief Construct an incompressible momentum equation on a mesh.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param mesh Computational mesh.
 * @throws std::invalid_argument if @p mesh is null.
 */
template<TpetraTypePack Pack, class MeshType>
IncompressibleMomentumEquation<Pack, MeshType>::IncompressibleMomentumEquation(
    SP<const mesh_type> mesh)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "IncompressibleMomentumEquation")),
      d_transport_geometry_cache(*d_mesh),
      d_candidate_velocity(
          d_mesh, "momentum_velocity_candidate", false)
{
}

/**
 * @brief Validate fields, time-step options, and cached boundary data.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Transport and non-orthogonal-correction options.
 * @param correction_field Optional lagged correction field.
 * @throws std::invalid_argument if an input violates the transport contract.
 */
template<TpetraTypePack Pack, class MeshType>
void IncompressibleMomentumEquation<Pack, MeshType>::validate_transport_inputs(
    const velocity_field_type& old_velocity,
    const face_flux_field_type& face_fluxes,
    const velocity_boundary_cache_type& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const velocity_field_type* correction_field) const
{
    EquationValidation::require_mesh_match(
        *d_mesh, old_velocity, "IncompressibleMomentumEquation");
    EquationValidation::require_mesh_match(
        *d_mesh, face_fluxes, "IncompressibleMomentumEquation");
    EquationValidation::require_non_negative(
        options.time_step, "time step", "IncompressibleMomentumEquation");
    EquationValidation::require_non_negative(
        options.kinematic_viscosity, "viscosity",
        "IncompressibleMomentumEquation");
    EquationValidation::require_non_negative(
        options.n_non_orthogonal_correctors, "non-orthogonal correctors",
        "IncompressibleMomentumEquation");
    if (correction_field != nullptr)
    {
        EquationValidation::require_mesh_match(
            *d_mesh, *correction_field, "IncompressibleMomentumEquation");
    }
    if (velocity_boundary_cache.value.size()
            != d_mesh->boundary_batches().size()
        || velocity_boundary_cache.type.size()
            != d_mesh->boundary_batches().size()
        || velocity_boundary_cache.mesh != d_mesh)
    {
        throw std::invalid_argument(
            "IncompressibleMomentumEquation received the wrong "
            "boundary-cache size.");
    }
}

/**
 * @brief Assemble velocity transport with no explicit acceleration source.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Transport and non-orthogonal-correction options.
 * @param correction_field Optional lagged correction field.
 * @return Assembled vector transport system.
 * @throws std::invalid_argument if an input violates the transport contract.
 */
template<TpetraTypePack Pack, class MeshType>
auto IncompressibleMomentumEquation<Pack, MeshType>::assemble_system(
    const velocity_field_type& old_velocity,
    const face_flux_field_type& face_fluxes,
    const velocity_boundary_cache_type& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const velocity_field_type* correction_field) const -> system_type
{
    auto zero_source =
        [](local_ordinal_type) -> typename velocity_field_type::vec_type
    {
        return {};
    };
    return assemble_system(
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        zero_source, correction_field);
}

/**
 * @brief Assemble velocity transport with a caller acceleration source.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Transport and non-orthogonal-correction options.
 * @param right_hand_source Per-cell acceleration provider.
 * @param correction_field Optional lagged correction field.
 * @return Assembled vector transport system.
 * @throws std::invalid_argument if an input violates the transport contract.
 */
template<TpetraTypePack Pack, class MeshType>
auto IncompressibleMomentumEquation<Pack, MeshType>::assemble_system(
    const velocity_field_type& old_velocity,
    const face_flux_field_type& face_fluxes,
    const velocity_boundary_cache_type& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const source_type& right_hand_source,
    const velocity_field_type* correction_field) const -> system_type
{
    validate_transport_inputs(
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        correction_field);

    const auto old_velocity_values = old_velocity.local_read_view();
    auto boundary_value =
        [&](int boundary_id, local_ordinal_type boundary_face_id)
    {
        const auto face = static_cast<size_t>(boundary_face_id);
        const auto type = velocity_boundary_cache.type.at(boundary_id);
        if (type == BoundaryConditionType::Slip)
        {
            const auto face_lid =
                d_mesh->boundary_face_batch(boundary_id).face_lids[face];
            if constexpr (std::same_as<mesh_type, Mesh<Pack>>)
            {
                return FVM::detail::slip_face_velocity(
                    old_velocity, face_lid);
            }
            else
            {
                return FVM::detail::stored_slip_face_velocity(
                    old_velocity, face_lid);
            }
        }
        if (type == BoundaryConditionType::Neumann)
        {
            const auto face_lid =
                d_mesh->boundary_face_batch(boundary_id).face_lids[face];
            const auto owner = d_mesh->owner_cell(face_lid);
            return typename velocity_field_type::vec_type{
                old_velocity_values(owner, 0),
                old_velocity_values(owner, 1),
                old_velocity_values(owner, 2)};
        }
        return velocity_boundary_cache.value.at(boundary_id)[face];
    };
    auto boundary_diffusion =
        [&](int boundary_id, local_ordinal_type)
    {
        const auto type = velocity_boundary_cache.type.at(boundary_id);
        // Homogeneous velocity Neumann is the open zero-gradient contract,
        // so it contributes no component-Laplacian boundary diagonal.
        return type != BoundaryConditionType::Slip
            && type != BoundaryConditionType::Neumann;
    };

    const auto requires_non_orthogonal_graph =
        options.non_orthogonal_treatment
            != FVM::NonOrthogonalTreatment::Explicit;
    if (requires_non_orthogonal_graph
        && !d_cached_graph_supports_non_orthogonal_correction)
    {
        d_cached_transport_matrix = Teuchos::null;
    }
    auto system = [&]() -> system_type
    {
        try
        {
            return FVM::non_orthogonal_transport_system<Pack>(
                old_velocity, face_fluxes, options.time_step,
                options.kinematic_viscosity, boundary_value,
                right_hand_source,
                options.non_orthogonal_treatment, correction_field,
                d_cached_transport_matrix, boundary_diffusion,
                &d_transport_geometry_cache);
        }
        catch (...)
        {
            // A reused Tpetra matrix is left in resume-fill mode if
            // assembly exits early.  Do not offer that partial matrix, or
            // its graph-support claim, to the next assembly.
            d_cached_transport_matrix = Teuchos::null;
            d_cached_graph_supports_non_orthogonal_correction = false;
            throw;
        }
    }();
    d_cached_transport_matrix = system.matrix;
    if (requires_non_orthogonal_graph
        && options.kinematic_viscosity > scalar_type{})
    {
        d_cached_graph_supports_non_orthogonal_correction = true;
    }
    return system;
}

/**
 * @brief Advance velocity with no explicit acceleration source.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Transport and non-orthogonal-correction options.
 * @param[out] velocity Updated velocity field.
 * @param linear_options Linear-solver configuration.
 * @return Aggregated linear-solve statistics.
 * @throws std::invalid_argument if an input violates the transport contract.
 * @throws std::runtime_error if a velocity solve fails or is non-finite.
 */
template<TpetraTypePack Pack, class MeshType>
auto IncompressibleMomentumEquation<Pack, MeshType>::advance_velocity(
    const velocity_field_type& old_velocity,
    const face_flux_field_type& face_fluxes,
    const velocity_boundary_cache_type& velocity_boundary_cache,
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
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        velocity, zero_source, linear_options);
}

/**
 * @brief Advance velocity with a caller acceleration source.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Transport and non-orthogonal-correction options.
 * @param[out] velocity Updated velocity field.
 * @param right_hand_source Per-cell acceleration provider.
 * @param linear_options Linear-solver configuration.
 * @return Aggregated linear-solve statistics.
 * @throws std::invalid_argument if an input violates the transport contract.
 * @throws std::runtime_error if a velocity solve fails or is non-finite.
 */
template<TpetraTypePack Pack, class MeshType>
auto IncompressibleMomentumEquation<Pack, MeshType>::advance_velocity(
    const velocity_field_type& old_velocity,
    const face_flux_field_type& face_fluxes,
    const velocity_boundary_cache_type& velocity_boundary_cache,
    const TimeStepperOptions& options,
    velocity_field_type& velocity,
    const source_type& right_hand_source,
    const LinearSolverOptions& linear_options) const -> LinearSolveSummary
{
    EquationValidation::require_mesh_match(
        *d_mesh, velocity, "IncompressibleMomentumEquation");

    // Solve into a warm-started candidate so a rejected solve cannot modify
    // the caller's last accepted velocity, even when both arguments alias.
    auto& candidate_velocity = d_candidate_velocity;
    candidate_velocity.owned_data().update(
        scalar_type{1}, old_velocity.owned_data(), scalar_type{0});

    LinearSolveSummary summary;
    auto solve_system =
        [&](const system_type& system)
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
        if (!has_nonzero_rhs)
        {
            candidate_velocity.owned_data().putScalar(0.0);
            return;
        }

        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        const auto statistics =
            d_linear_solver.solve_with_statistics(
                matrix, *system.rhs, candidate_velocity.owned_data(),
                linear_options);
        summary.add(statistics);
        if (!statistics.converged)
        {
            throw RetryableMomentumNonconvergence(
                "IncompressibleMomentumEquation velocity transport solve "
                "did not converge.");
        }

        {
            const auto solution_values =
                candidate_velocity.owned_read_view();
            for (size_t component = 0;
                 component < velocity_field_type::num_components;
                 ++component)
            {
                for (size_t owned = 0;
                     owned < d_mesh->num_owned_cells(); ++owned)
                {
                    const auto cell_lid =
                        static_cast<local_ordinal_type>(owned);
                    if (!std::isfinite(
                            solution_values(cell_lid, component)))
                    {
                        throw std::runtime_error(
                            "IncompressibleMomentumEquation velocity "
                            "transport solve produced a non-finite value.");
                    }
                }
            }
        }
    };

    for (int corrector = 0;
         corrector <= options.n_non_orthogonal_correctors;
         ++corrector)
    {
        if (corrector > 0)
        {
            d_mesh->sync_periodic_boundaries(candidate_velocity);
        }
        const auto* correction_field =
            corrector == 0 ? nullptr : &candidate_velocity;
        const auto system = assemble_system(
            old_velocity, face_fluxes, velocity_boundary_cache,
            options, right_hand_source, correction_field);
        solve_system(system);

        if (options.non_orthogonal_treatment
            == FVM::NonOrthogonalTreatment::Implicit)
        {
            break;
        }
    }
    velocity.owned_data().update(
        scalar_type{1}, candidate_velocity.owned_data(), scalar_type{0});
    d_mesh->sync_periodic_boundaries(velocity);
    return summary;
}

/**
 * @brief Assemble material-dependent incompressible momentum transport.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Transport and non-orthogonal-correction options.
 * @param dynamic_viscosity Cell dynamic-viscosity field.
 * @param reference_density Constant density used to normalize momentum.
 * @param acceleration_source Per-cell acceleration provider.
 * @param correction_field Optional lagged correction field.
 * @param boundary_dynamic_viscosity Optional boundary viscosity cache.
 * @return Assembled physical momentum system.
 * @throws std::invalid_argument if fields or physical inputs are invalid.
 */
template<TpetraTypePack Pack, class MeshType>
auto IncompressibleMomentumEquation<Pack, MeshType>::assemble_physical_system(
    const velocity_field_type& old_velocity,
    const face_flux_field_type& face_fluxes,
    const velocity_boundary_cache_type& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const field_type& dynamic_viscosity,
    scalar_type reference_density,
    const source_type& acceleration_source,
    const velocity_field_type* correction_field,
    const boundary_cache_type* boundary_dynamic_viscosity) const
    -> system_type
{
    validate_transport_inputs(
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        correction_field);
    EquationValidation::require_mesh_match(
        *d_mesh, dynamic_viscosity, "IncompressibleMomentumEquation");
    if (reference_density <= scalar_type{})
    {
        throw std::invalid_argument(
            "IncompressibleMomentumEquation requires positive reference "
            "density.");
    }

    const auto old_velocity_values = old_velocity.local_read_view();
    auto boundary_value =
        [&](int boundary_id, local_ordinal_type boundary_face_id)
    {
        const auto face = static_cast<size_t>(boundary_face_id);
        const auto type = velocity_boundary_cache.type.at(boundary_id);
        if (type == BoundaryConditionType::Slip)
        {
            const auto face_lid =
                d_mesh->boundary_face_batch(boundary_id).face_lids[face];
            if constexpr (std::same_as<mesh_type, Mesh<Pack>>)
            {
                return FVM::detail::slip_face_velocity(
                    old_velocity, face_lid);
            }
            else
            {
                return FVM::detail::stored_slip_face_velocity(
                    old_velocity, face_lid);
            }
        }
        if (type == BoundaryConditionType::Neumann)
        {
            const auto face_lid =
                d_mesh->boundary_face_batch(boundary_id).face_lids[face];
            const auto owner = d_mesh->owner_cell(face_lid);
            return typename velocity_field_type::vec_type{
                old_velocity_values(owner, 0),
                old_velocity_values(owner, 1),
                old_velocity_values(owner, 2)};
        }
        return velocity_boundary_cache.value.at(boundary_id)[face];
    };
    auto boundary_diffusion =
        [&](int boundary_id, local_ordinal_type)
    {
        const auto type = velocity_boundary_cache.type.at(boundary_id);
        // Homogeneous velocity Neumann is the open/traction-free outlet
        // contract. It contributes neither a component-Laplacian boundary
        // diagonal nor a lagged transpose-stress traction.
        return type != BoundaryConditionType::Slip
            && type != BoundaryConditionType::Neumann;
    };

    const auto requires_non_orthogonal_graph =
        options.non_orthogonal_treatment
            != FVM::NonOrthogonalTreatment::Explicit;
    bool all_viscosities_positive = true;
    const auto viscosity_values =
        dynamic_viscosity.local_read_view();
    for (size_t local = 0; local < d_mesh->num_local_cells(); ++local)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(local);
        all_viscosities_positive = all_viscosities_positive
            && viscosity_values(cell_lid, 0) > scalar_type{};
    }
    if (requires_non_orthogonal_graph
        && !d_cached_physical_graph_supports_non_orthogonal_correction)
    {
        d_cached_physical_transport_matrix = Teuchos::null;
    }

    auto system = [&]() -> system_type
    {
        try
        {
            return FVM::physical_momentum_transport_system<Pack>(
                old_velocity, face_fluxes, options.time_step,
                dynamic_viscosity, reference_density, boundary_value,
                acceleration_source, options.non_orthogonal_treatment,
                correction_field, d_cached_physical_transport_matrix,
                boundary_diffusion, boundary_dynamic_viscosity,
                &d_transport_geometry_cache,
                options.coefficient_interpolation);
        }
        catch (...)
        {
            d_cached_physical_transport_matrix = Teuchos::null;
            d_cached_physical_graph_supports_non_orthogonal_correction =
                false;
            throw;
        }
    }();
    d_cached_physical_transport_matrix = system.matrix;
    if (requires_non_orthogonal_graph && all_viscosities_positive)
    {
        d_cached_physical_graph_supports_non_orthogonal_correction = true;
    }
    return system;
}

/**
 * @brief Advance material-dependent incompressible momentum transport.
 * @tparam Pack Tpetra type pack used by the equation.
 * @param old_velocity Accepted velocity from the previous step.
 * @param face_fluxes Oriented volumetric face fluxes.
 * @param velocity_boundary_cache Cached boundary velocities.
 * @param options Transport and non-orthogonal-correction options.
 * @param dynamic_viscosity Cell dynamic-viscosity field.
 * @param reference_density Constant density used to normalize momentum.
 * @param[out] velocity Updated velocity field.
 * @param acceleration_source Per-cell acceleration provider.
 * @param linear_options Linear-solver configuration.
 * @param boundary_dynamic_viscosity Optional boundary viscosity cache.
 * @return Aggregated linear-solve statistics.
 * @throws std::invalid_argument if fields or physical inputs are invalid.
 * @throws std::runtime_error if a velocity solve fails or is non-finite.
 */
template<TpetraTypePack Pack, class MeshType>
auto IncompressibleMomentumEquation<Pack, MeshType>::advance_velocity_physical(
    const velocity_field_type& old_velocity,
    const face_flux_field_type& face_fluxes,
    const velocity_boundary_cache_type& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const field_type& dynamic_viscosity,
    scalar_type reference_density,
    velocity_field_type& velocity,
    const source_type& acceleration_source,
    const LinearSolverOptions& linear_options,
    const boundary_cache_type* boundary_dynamic_viscosity) const
    -> LinearSolveSummary
{
    EquationValidation::require_mesh_match(
        *d_mesh, velocity, "IncompressibleMomentumEquation");

    // Keep the accepted velocity intact until every correction solve
    // succeeds.  The old state is also the physically relevant warm start.
    auto& candidate_velocity = d_candidate_velocity;
    candidate_velocity.owned_data().update(
        scalar_type{1}, old_velocity.owned_data(), scalar_type{0});

    LinearSolveSummary summary;
    for (int corrector = 0;
         corrector <= options.n_non_orthogonal_correctors;
         ++corrector)
    {
        if (corrector > 0)
        {
            d_mesh->sync_periodic_boundaries(candidate_velocity);
        }
        const auto* correction_field =
            corrector == 0 ? nullptr : &candidate_velocity;
        const auto system = assemble_physical_system(
            old_velocity, face_fluxes, velocity_boundary_cache, options,
            dynamic_viscosity, reference_density, acceleration_source,
            correction_field, boundary_dynamic_viscosity);
        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        const auto statistics =
            d_linear_solver.solve_with_statistics(
                matrix, *system.rhs, candidate_velocity.owned_data(),
                linear_options);
        summary.add(statistics);
        if (!statistics.converged)
        {
            throw RetryableMomentumNonconvergence(
                "IncompressibleMomentumEquation physical transport solve "
                "did not converge.");
        }
        if (options.non_orthogonal_treatment
            == FVM::NonOrthogonalTreatment::Implicit)
        {
            break;
        }
    }
    velocity.owned_data().update(
        scalar_type{1}, candidate_velocity.owned_data(), scalar_type{0});
    d_mesh->sync_periodic_boundaries(velocity);
    return summary;
}

} // namespace SimpleFluid
