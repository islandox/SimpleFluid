/** @file VariableDensityMomentumEquation.tcc
 * @brief Compiled conservative variable-density momentum implementation.
 */
#pragma once
#include "equations/VariableDensityMomentumEquation.hh"

#include <cmath>
#include <stdexcept>

namespace SimpleFluid
{
template<TpetraTypePack Pack, class MeshType>
VariableDensityMomentumEquation<Pack, MeshType>::VariableDensityMomentumEquation(SP<const MeshType> mesh)
    : d_mesh(EquationValidation::require_non_null_mesh(std::move(mesh), "VariableDensityMomentumEquation")),
      d_old_component(d_mesh, "old_velocity_component"), d_component(d_mesh, "velocity_component"),
      d_unit(d_mesh, scalar_type{1}, "mass_flux_unit_weight"), d_candidate(d_mesh, "momentum_candidate"),
      d_force(d_mesh, "momentum_force"), d_geometry(*d_mesh)
{
}

template<TpetraTypePack Pack, class MeshType> void VariableDensityMomentumEquation<Pack, MeshType>::refresh_geometry()
{
    d_geometry.refresh();
    d_solver.notify_operator_values_changed();
}

template<TpetraTypePack Pack, class MeshType>
LinearSolveSummary VariableDensityMomentumEquation<Pack, MeshType>::advance(const velocity_field_type& old_velocity,
    const field_type& old_density, const field_type& density, const face_field_type& mass_flux,
    const field_type& dynamic_viscosity, scalar_type time_step, velocity_field_type& velocity, force_type force,
    boundary_type boundary, boundary_value_type boundary_value, const LinearSolverOptions& linear,
    FVM::NonOrthogonalTreatment treatment, const FVM::ALEControlVolumeState* ale)
{
    collective_detail::collective_local_validation(*d_mesh, "Variable-density momentum inputs",
        [&]
        {
            if (!std::isfinite(time_step) || time_step <= 0)
                throw std::invalid_argument("Variable-density momentum requires a positive finite timestep.");
            BelosLinearSolver<Pack>::validate_options(linear);
            if constexpr (std::is_same_v<MeshType, Mesh<Pack>>)
                if (ale)
                    throw std::invalid_argument("Legacy variable-density momentum does not own ALE geometry.");
            const auto same_mesh = [&](const auto& field)
            { EquationValidation::require_mesh_match(*d_mesh, field, "VariableDensityMomentumEquation"); };
            same_mesh(old_velocity);
            same_mesh(old_density);
            same_mesh(density);
            same_mesh(mass_flux);
            same_mesh(dynamic_viscosity);
            same_mesh(velocity);
            for (const auto face : mass_flux.owned_face_ids())
                if (!std::isfinite(mass_flux.value(face)))
                    throw std::invalid_argument("Momentum mass flux must be finite.");
            for (size_t c = 0; c < d_mesh->num_owned_cells(); ++c)
            {
                const auto cell = static_cast<local_ordinal_type>(c);
                if (!std::isfinite(old_density.value(cell)) || old_density.value(cell) <= 0 ||
                    !std::isfinite(density.value(cell)) || density.value(cell) <= 0 ||
                    !std::isfinite(dynamic_viscosity.value(cell)) || dynamic_viscosity.value(cell) < 0)
                    throw std::invalid_argument("Momentum density must be positive and viscosity nonnegative.");
                const auto u = old_velocity.value(cell);
                const auto f = force ? force(cell) : vector_type{};
                for (size_t a = 0; a < 3; ++a)
                    if (!std::isfinite(u.component(a)) || !std::isfinite(f.component(a)))
                        throw std::invalid_argument("Momentum velocity and force must be finite.");
                d_force.set_owned_value(cell, f);
            }
        });
    collective_detail::require_uniform_value(*d_mesh, time_step, "Variable-density momentum timestep");
    collective_detail::require_uniform_value(*d_mesh, int(ale != nullptr), "Variable-density momentum ALE selection");
    collective_detail::require_uniform_value(*d_mesh, int(treatment), "Variable-density momentum correction policy");
    collective_detail::require_uniform_value(*d_mesh, linear.max_iterations, "Momentum linear iteration limit");
    collective_detail::require_uniform_value(*d_mesh, linear.tolerance, "Momentum linear tolerance");
    collective_detail::require_uniform_value(*d_mesh, linear.verbosity, "Momentum linear verbosity");
    collective_detail::require_uniform_value(*d_mesh, int(linear.backend), "Momentum linear backend");
    collective_detail::require_uniform_value(*d_mesh, int(linear.preconditioner), "Momentum preconditioner");
    collective_detail::require_uniform_value(
        *d_mesh, int(linear.reuse_preconditioner), "Momentum preconditioner reuse");
    d_geometry.refresh();
    d_candidate.owned_data().update(scalar_type{1}, old_velocity.owned_data(), scalar_type{});
    LinearSolveSummary summary;
    for (size_t component = 0; component < 3; ++component)
    {
        for (size_t c = 0; c < d_mesh->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<local_ordinal_type>(c);
            d_old_component.set_owned_value(cell, old_velocity.value(cell).component(component));
        }
        d_old_component.sync_ghosts();
        d_component.owned_data().update(scalar_type{1}, d_old_component.owned_data(), scalar_type{});
        auto system = FVM::weighted_scalar_transport_system<Pack>(
            FVM::MeshWeightedScalarTransportRequest<Pack, MeshType>{.old_values = d_old_component,
                .face_fluxes = mass_flux,
                .time_step = time_step,
                .storage_weight = density,
                .advection_weight = d_unit,
                .diffusivity = dynamic_viscosity,
                .boundary_condition =
                    [&](int patch, size_t face)
                {
                    return boundary ? boundary(patch, face, component)
                                    : BoundaryCondition{BoundaryConditionType::Neumann, 0};
                },
                .boundary_value = [&](int patch, size_t face)
                { return boundary_value ? boundary_value(patch, face, component) : scalar_type{}; },
                .source = [&](local_ordinal_type cell) { return d_force.value(cell).component(component); },
                .treatment = treatment,
                .correction_field = &d_old_component,
                .geometry_cache = &d_geometry,
                .old_storage_weight = &old_density,
                .ale = ale});
        d_solver.notify_operator_values_changed();
        const auto statistics =
            d_solver.solve_with_statistics(system.matrix, *system.rhs, d_component.owned_data(), linear);
        summary.add(statistics);
        if (!statistics.converged)
            throw std::runtime_error("Variable-density momentum failed its true-residual convergence gate.");
        collective_detail::collective_local_validation(*d_mesh, "Variable-density momentum candidate",
            [&]
            {
                for (size_t c = 0; c < d_mesh->num_owned_cells(); ++c)
                {
                    const auto cell = static_cast<local_ordinal_type>(c);
                    if (!std::isfinite(d_component.value(cell)))
                        throw std::runtime_error("Variable-density momentum produced nonfinite velocity.");
                    auto u = d_candidate.value(cell);
                    u.component(component) = d_component.value(cell);
                    d_candidate.set_owned_value(cell, u);
                }
            });
    }
    velocity.owned_data().update(scalar_type{1}, d_candidate.owned_data(), scalar_type{});
    velocity.sync_ghosts();
    return summary;
}
} // namespace SimpleFluid
