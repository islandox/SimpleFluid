/**
 * @file BoussinesqMomentumEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Boussinesq buoyancy and velocity-transport updates.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "equations/EquationValidation.hh"
#include "equations/TimeStepperOptions.hh"
#include "fields/CellField.hh"
#include "fields/VectorCellField.hh"
#include "operators/FvmOperators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Boussinesq momentum update for component-wise velocity fields.
 *
 * The solver stores velocity as a three-column MultiVector-backed field.
 * This equation class advances all velocity components together.
 *
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class BoussinesqMomentumEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename mesh_type::Vec3;

    explicit BoussinesqMomentumEquation(SP<const mesh_type> mesh);

    void advance_velocity(
        const std::vector<vec_type>& old_velocity,
        const std::vector<scalar_type>& face_fluxes,
        const field_type& temperature,
        const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const LinearSolverOptions& linear_options = {}) const;

    void advance_velocity(
        const std::vector<vec_type>& old_velocity,
        const std::vector<scalar_type>& face_fluxes,
        const field_type& temperature,
        const FvmOperators::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const LinearSolverOptions& linear_options = {}) const;

private:
    SP<const mesh_type> d_mesh;
    mutable std::vector<scalar_type> d_cached_old_component;
    mutable Teuchos::RCP<typename Pack::vector_type> d_cached_solution;
};

/**
 * @brief Construct a Boussinesq momentum equation with the given mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the assembled mesh.
 * @throws std::invalid_argument if the mesh is null.
 */
template<TpetraTypePack Pack>
BoussinesqMomentumEquation<Pack>::BoussinesqMomentumEquation(
    SP<const mesh_type> mesh)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "BoussinesqMomentumEquation"))
{
}

/**
 * @brief Advance all velocity components with semi-implicit convection/diffusion.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_velocity Previous local velocity values.
 * @param face_fluxes Owner-oriented integrated fluxes from the previous velocity.
 * @param temperature Temperature field used for Boussinesq buoyancy.
 * @param boundary_conditions Velocity boundary conditions.
 * @param options Time and physical coefficients.
 * @param velocity Output velocity field.
 * @param linear_options Belos solver options for component transport solves.
 */
template<TpetraTypePack Pack>
void BoussinesqMomentumEquation<Pack>::advance_velocity(
    const std::vector<vec_type>& old_velocity,
    const std::vector<scalar_type>& face_fluxes,
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

/**
 * @brief Advance all velocity components with cached velocity boundary values.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_velocity Previous local velocity values.
 * @param face_fluxes Owner-oriented integrated fluxes from the previous velocity.
 * @param temperature Temperature field used for Boussinesq buoyancy.
 * @param velocity_boundary_cache Per-face resolved velocity boundary values.
 * @param options Time and physical coefficients.
 * @param velocity Output velocity field.
 * @param linear_options Belos solver options for component transport solves.
 */
template<TpetraTypePack Pack>
void BoussinesqMomentumEquation<Pack>::advance_velocity(
    const std::vector<vec_type>& old_velocity,
    const std::vector<scalar_type>& face_fluxes,
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
    if (velocity_boundary_cache.has_value.size() != d_mesh->num_faces()
        || &velocity_boundary_cache.value.mesh() != d_mesh.get())
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
            [&](local_ordinal_type face_lid,
                scalar_type /*fallback*/) -> std::optional<scalar_type>
        {
            const auto face = static_cast<std::size_t>(face_lid);
            if (!velocity_boundary_cache.has_value[face])
            {
                return std::nullopt;
            }

            return FvmOperators::detail::component_value(
                velocity_boundary_cache.value.value(face_lid), component);
        };

        auto system = FvmOperators::transport_system<Pack>(
            *d_mesh, d_cached_old_component, face_fluxes, options.time_step,
            options.kinematic_viscosity, boundary_value);

        const auto gravity_component =
            FvmOperators::detail::component_value(gravity, component);
        for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto buoyancy =
                options.thermal_expansion
              * (temperature.value(cell_lid) - options.reference_temperature)
              * (-gravity_component);
            system.rhs.sumIntoLocalValue(cell_lid,
                                         d_mesh->cell_volume(cell_lid) * buoyancy);
        }

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

} // namespace SimpleFluid
