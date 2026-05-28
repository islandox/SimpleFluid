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
#include "operators/FvmOperators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <array>
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
 * The solver stores velocity component-wise. This equation class provides the
 * legacy vertical update plus the current 3-component transport update.
 *
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class BoussinesqMomentumEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    explicit BoussinesqMomentumEquation(SP<const mesh_type> mesh);

    void advance_vertical_velocity(const std::vector<scalar_type>& old_velocity_z,
                                   const field_type& temperature,
                                   const TimeStepperOptions& options,
                                   field_type& velocity_z) const;

    void advance_velocity(
        const std::vector<scalar_type>& old_velocity_x,
        const std::vector<scalar_type>& old_velocity_y,
        const std::vector<scalar_type>& old_velocity_z,
        const std::vector<scalar_type>& face_fluxes,
        const field_type& temperature,
        const BoundaryConditionSet& boundary_conditions,
        const TimeStepperOptions& options,
        field_type& velocity_x,
        field_type& velocity_y,
        field_type& velocity_z,
        const LinearSolverOptions& linear_options = {}) const;

private:
    SP<const mesh_type> d_mesh;
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
 * @brief Advance vertical velocity using Boussinesq buoyancy and linear damping.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_velocity_z Local-cell velocity values from the previous time level.
 * @param temperature Updated temperature field.
 * @param options Physical and time-stepping parameters.
 * @param velocity_z Output vertical velocity field over owned cells.
 */
template<TpetraTypePack Pack>
void BoussinesqMomentumEquation<Pack>::advance_vertical_velocity(
    const std::vector<scalar_type>& old_velocity_z,
    const field_type& temperature,
    const TimeStepperOptions& options,
    field_type& velocity_z) const
{
    EquationValidation::require_mesh_match(*d_mesh, temperature,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity_z,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.time_step, "time step",
                                             "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.kinematic_viscosity, "viscosity",
                                             "BoussinesqMomentumEquation");
    EquationValidation::assert_sufficient_cache_size(old_velocity_z.size(),
                                                     d_mesh->num_local_cells());

    const auto damping =
        1.0 / (1.0 + options.time_step * options.kinematic_viscosity);

    for (std::size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto buoyancy =
            options.thermal_expansion
          * (temperature.value(cell_lid) - options.reference_temperature)
          * (-options.gravity_z);

        velocity_z.set_owned_value(cell_lid,
                                   (old_velocity_z[cell]
                                  + options.time_step * buoyancy) * damping);
    }
}

/**
 * @brief Advance all velocity components with semi-implicit convection/diffusion.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_velocity_x Previous local x velocity values.
 * @param old_velocity_y Previous local y velocity values.
 * @param old_velocity_z Previous local z velocity values.
 * @param face_fluxes Owner-oriented integrated fluxes from the previous velocity.
 * @param temperature Temperature field used for Boussinesq buoyancy.
 * @param boundary_conditions Velocity boundary conditions.
 * @param options Time and physical coefficients.
 * @param velocity_x Output x velocity.
 * @param velocity_y Output y velocity.
 * @param velocity_z Output z velocity.
 * @param linear_options Belos solver options for component transport solves.
 */
template<TpetraTypePack Pack>
void BoussinesqMomentumEquation<Pack>::advance_velocity(
    const std::vector<scalar_type>& old_velocity_x,
    const std::vector<scalar_type>& old_velocity_y,
    const std::vector<scalar_type>& old_velocity_z,
    const std::vector<scalar_type>& face_fluxes,
    const field_type& temperature,
    const BoundaryConditionSet& boundary_conditions,
    const TimeStepperOptions& options,
    field_type& velocity_x,
    field_type& velocity_y,
    field_type& velocity_z,
    const LinearSolverOptions& linear_options) const
{
    EquationValidation::require_mesh_match(*d_mesh, temperature,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity_x,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity_y,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity_z,
                                           "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.time_step, "time step",
                                             "BoussinesqMomentumEquation");
    EquationValidation::require_non_negative(options.kinematic_viscosity, "viscosity",
                                             "BoussinesqMomentumEquation");
    EquationValidation::assert_sufficient_cache_size(old_velocity_x.size(),
                                                     d_mesh->num_local_cells());
    EquationValidation::assert_sufficient_cache_size(old_velocity_y.size(),
                                                     d_mesh->num_local_cells());
    EquationValidation::assert_sufficient_cache_size(old_velocity_z.size(),
                                                     d_mesh->num_local_cells());

    const std::array<const std::vector<scalar_type>*, 3> old_velocity{
        &old_velocity_x, &old_velocity_y, &old_velocity_z
    };
    const std::array<field_type*, 3> output_velocity{
        &velocity_x, &velocity_y, &velocity_z
    };
    const auto gravity = options.gravity_vector();

    for (std::size_t component = 0; component < output_velocity.size(); ++component)
    {
        auto boundary_value =
            [&](local_ordinal_type face_lid,
                scalar_type /*fallback*/) -> std::optional<scalar_type>
        {
            if (!d_mesh->is_boundary_face(face_lid))
            {
                return std::nullopt;
            }

            const auto iter = boundary_conditions.velocity.find(
                d_mesh->boundary_name(face_lid));
            if (iter == boundary_conditions.velocity.end())
            {
                return std::nullopt;
            }

            if (iter->second.type == BoundaryConditionType::NoSlip)
            {
                return scalar_type{};
            }
            if (iter->second.type == BoundaryConditionType::Dirichlet)
            {
                return FvmOperators::detail::component_value(iter->second.value,
                                                             component);
            }
            return std::nullopt;
        };

        auto system = FvmOperators::transport_system<Pack>(
            *d_mesh, *old_velocity[component], face_fluxes, options.time_step,
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
            output_velocity[component]->owned_data().putScalar(0.0);
            continue;
        }

        Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
        const auto converged =
            solve_linear_system<Pack>(matrix, system.rhs,
                                      output_velocity[component]->owned_data(),
                                      linear_options);
        if (!converged)
        {
            for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid = static_cast<local_ordinal_type>(owned);
                if (!std::isfinite(output_velocity[component]->value(cell_lid)))
                {
                    throw std::runtime_error(
                        "BoussinesqMomentumEquation velocity transport solve produced a non-finite value.");
                }
            }
        }
    }

    velocity_x.sync_ghosts();
    velocity_y.sync_ghosts();
    velocity_z.sync_ghosts();
}

} // namespace SimpleFluid
