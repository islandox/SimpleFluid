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

template<TpetraTypePack Pack>
IncompressibleMomentumEquation<Pack>::IncompressibleMomentumEquation(
    SP<const mesh_type> mesh)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "IncompressibleMomentumEquation"))
{
}

template<TpetraTypePack Pack>
void IncompressibleMomentumEquation<Pack>::validate_transport_inputs(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
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

template<TpetraTypePack Pack>
auto IncompressibleMomentumEquation<Pack>::assemble_system(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
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
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        zero_source, correction_field);
}

template<TpetraTypePack Pack>
auto IncompressibleMomentumEquation<Pack>::assemble_system(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const source_type& right_hand_source,
    const velocity_field_type* correction_field) const -> system_type
{
    validate_transport_inputs(
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        correction_field);

    auto boundary_value =
        [&](int boundary_id, local_ordinal_type boundary_face_id)
    {
        const auto face = static_cast<size_t>(boundary_face_id);
        const auto type = velocity_boundary_cache.type.at(boundary_id);
        if (type == BoundaryConditionType::Slip)
        {
            const auto face_lid =
                d_mesh->boundary_face_batch(boundary_id).face_lids[face];
            return FVM::detail::slip_face_velocity(
                old_velocity, face_lid);
        }
        if (type == BoundaryConditionType::Neumann)
        {
            const auto face_lid =
                d_mesh->boundary_face_batch(boundary_id).face_lids[face];
            return old_velocity.local_value(
                d_mesh->owner_cell(face_lid));
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

    auto system = FVM::non_orthogonal_transport_system<Pack>(
        old_velocity, face_fluxes, options.time_step,
        options.kinematic_viscosity, boundary_value, right_hand_source,
        options.non_orthogonal_treatment, correction_field,
        d_cached_transport_matrix, boundary_diffusion);
    d_cached_transport_matrix = system.matrix;
    return system;
}

template<TpetraTypePack Pack>
auto IncompressibleMomentumEquation<Pack>::advance_velocity(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
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
        old_velocity, face_fluxes, velocity_boundary_cache, options,
        velocity, zero_source, linear_options);
}

template<TpetraTypePack Pack>
auto IncompressibleMomentumEquation<Pack>::advance_velocity(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    velocity_field_type& velocity,
    const source_type& right_hand_source,
    const LinearSolverOptions& linear_options) const -> LinearSolveSummary
{
    EquationValidation::require_mesh_match(
        *d_mesh, velocity, "IncompressibleMomentumEquation");

    velocity_field_type old_velocity_snapshot(
        old_velocity.mesh_ptr(), "momentum_old_velocity_snapshot", false);
    const velocity_field_type* transport_old_velocity = &old_velocity;
    if (&old_velocity == &velocity
        && options.n_non_orthogonal_correctors > 0)
    {
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            old_velocity_snapshot.set_value(
                cell_lid, old_velocity.value(cell_lid));
        }
        old_velocity_snapshot.sync_ghosts();
        transport_old_velocity = &old_velocity_snapshot;
    }

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
        velocity.owned_data().putScalar(0.0);
        if (!has_nonzero_rhs)
        {
            d_mesh->sync_periodic_boundaries(velocity);
            return;
        }

        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        const auto statistics =
            d_linear_solver.solve_with_statistics(
                matrix, *system.rhs, velocity.owned_data(), linear_options);
        summary.add(statistics);
        if (!statistics.converged)
        {
            throw std::runtime_error(
                "IncompressibleMomentumEquation velocity transport solve "
                "did not converge.");
        }

        for (size_t component = 0;
             component < velocity_field_type::num_components;
             ++component)
        {
            const auto solution_data =
                velocity.owned_data().getData(component);
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                if (!std::isfinite(solution_data[cell_lid]))
                {
                    throw std::runtime_error(
                        "IncompressibleMomentumEquation velocity transport "
                        "solve produced a non-finite value.");
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
        const auto system = assemble_system(
            *transport_old_velocity, face_fluxes, velocity_boundary_cache,
            options, right_hand_source, correction_field);
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
auto IncompressibleMomentumEquation<Pack>::assemble_physical_system(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const field_type& dynamic_viscosity,
    scalar_type reference_density,
    const source_type& acceleration_source,
    const velocity_field_type* correction_field,
    const FVM::BoundaryCache<Pack>* boundary_dynamic_viscosity) const
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

    auto boundary_value =
        [&](int boundary_id, local_ordinal_type boundary_face_id)
    {
        const auto face = static_cast<size_t>(boundary_face_id);
        const auto type = velocity_boundary_cache.type.at(boundary_id);
        if (type == BoundaryConditionType::Slip)
        {
            const auto face_lid =
                d_mesh->boundary_face_batch(boundary_id).face_lids[face];
            return FVM::detail::slip_face_velocity(
                old_velocity, face_lid);
        }
        if (type == BoundaryConditionType::Neumann)
        {
            const auto face_lid =
                d_mesh->boundary_face_batch(boundary_id).face_lids[face];
            return old_velocity.local_value(
                d_mesh->owner_cell(face_lid));
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
    for (size_t local = 0; local < d_mesh->num_local_cells(); ++local)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(local);
        all_viscosities_positive = all_viscosities_positive
            && dynamic_viscosity.local_value(cell_lid) > scalar_type{};
    }
    if (requires_non_orthogonal_graph
        && !d_cached_physical_graph_supports_non_orthogonal_correction)
    {
        d_cached_physical_transport_matrix = Teuchos::null;
    }

    auto system = FVM::physical_momentum_transport_system<Pack>(
        old_velocity, face_fluxes, options.time_step, dynamic_viscosity,
        reference_density, boundary_value, acceleration_source,
        options.non_orthogonal_treatment, correction_field,
        d_cached_physical_transport_matrix, boundary_diffusion,
        boundary_dynamic_viscosity);
    d_cached_physical_transport_matrix = system.matrix;
    if (requires_non_orthogonal_graph && all_viscosities_positive)
    {
        d_cached_physical_graph_supports_non_orthogonal_correction = true;
    }
    return system;
}

template<TpetraTypePack Pack>
auto IncompressibleMomentumEquation<Pack>::advance_velocity_physical(
    const velocity_field_type& old_velocity,
    const FaceField<Pack>& face_fluxes,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    const TimeStepperOptions& options,
    const field_type& dynamic_viscosity,
    scalar_type reference_density,
    velocity_field_type& velocity,
    const source_type& acceleration_source,
    const LinearSolverOptions& linear_options,
    const FVM::BoundaryCache<Pack>* boundary_dynamic_viscosity) const
    -> LinearSolveSummary
{
    EquationValidation::require_mesh_match(
        *d_mesh, velocity, "IncompressibleMomentumEquation");

    velocity_field_type old_snapshot(
        old_velocity.mesh_ptr(), "physical_momentum_old_velocity", false);
    const velocity_field_type* transport_old = &old_velocity;
    if (&old_velocity == &velocity
        && options.n_non_orthogonal_correctors > 0)
    {
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            old_snapshot.set_value(cell_lid, old_velocity.value(cell_lid));
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
        const auto system = assemble_physical_system(
            *transport_old, face_fluxes, velocity_boundary_cache, options,
            dynamic_viscosity, reference_density, acceleration_source,
            correction_field, boundary_dynamic_viscosity);
        velocity.owned_data().putScalar(0.0);
        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        const auto statistics =
            d_linear_solver.solve_with_statistics(
                matrix, *system.rhs, velocity.owned_data(), linear_options);
        summary.add(statistics);
        if (!statistics.converged)
        {
            throw std::runtime_error(
                "IncompressibleMomentumEquation physical transport solve "
                "did not converge.");
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
