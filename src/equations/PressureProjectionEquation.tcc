/**
 * @file PressureProjectionEquation.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template method implementations for PressureProjectionEquation.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "PressureProjectionEquation.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>

namespace SimpleFluid
{

/**
 * @brief Construct a PressureProjectionEquation on the given mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the computational mesh.
 * @param linear_options Linear solver configuration.
 * @throws std::invalid_argument if @p mesh is null.
 * @throws std::runtime_error if the mesh has no owned-cell map.
 */
template<TpetraTypePack Pack>
PressureProjectionEquation<Pack>::PressureProjectionEquation(
    SP<const mesh_type> mesh,
    LinearSolverOptions linear_options)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "PressureProjectionEquation")),
      d_linear_options(linear_options),
      d_cached_face_velocity(d_mesh, "pressure_projection_face_velocity"),
      d_cached_face_fluxes(d_mesh, "pressure_projection_face_flux"),
      d_cached_gradient(d_mesh, "pressure_projection_gradient")
{
    require_owned_cell_map(d_mesh);
}

/**
 * @brief Require and return the mesh owned-cell map.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the computational mesh.
 * @return The mesh's owned-cell map.
 * @throws std::runtime_error if the mesh has no owned-cell map.
 */
template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::require_owned_cell_map(
    const SP<const mesh_type>& mesh) -> Teuchos::RCP<const map_type>
{
    auto map = mesh->owned_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error(
            "PressureProjectionEquation requires an assembled mesh with an owned-cell map.");
    }
    if (map->getGlobalNumElements() == 0)
    {
        throw std::runtime_error(
            "PressureProjectionEquation requires at least one global cell.");
    }

    return map;
}

/**
 * @brief (Re)build the cached pressure-Poisson matrix.
 *
 * Uses the globally smallest owned-row ID as the gauge-fixing row.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void PressureProjectionEquation<Pack>::rebuild_matrix() const
{
    const auto owned_map = require_owned_cell_map(d_mesh);
    const auto gauge_gid = owned_map->getMinAllGlobalIndex();
    d_cached_pressure_matrix =
        FVM::pressure_poisson_matrix<Pack>(*d_mesh, gauge_gid);
}

/**
 * @brief Solve for the pressure field (initialise with zero and sync
 *        periodic boundaries).
 *
 * @tparam Pack Tpetra type pack.
 * @param[out] pressure Pressure field to initialise.
 * @throws std::invalid_argument if the pressure field mesh does not match.
 */
template<TpetraTypePack Pack>
void PressureProjectionEquation<Pack>::solve(field_type& pressure)
{
    EquationValidation::require_mesh_match(*d_mesh, pressure,
                                           "PressureProjectionEquation");

    pressure.owned_data().putScalar(0.0);
    d_mesh->sync_periodic_boundaries(pressure);
}

/**
 * @brief Perform the pressure projection step with a zero source term.
 *
 * @tparam Pack Tpetra type pack.
 * @param[in,out] pressure Pressure field (updated on output).
 * @param time_step Time-step size.
 * @param velocity_boundary_cache Cached velocity boundary conditions.
 * @param[in,out] velocity Velocity field corrected by the pressure
 *        gradient on output.
 */
template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::project(
    field_type& pressure,
    scalar_type time_step,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    velocity_field_type& velocity) -> ProjectionResult
{
    auto zero_source =
        [](local_ordinal_type) -> scalar_type
    {
        return scalar_type{};
    };

    return project(pressure, time_step, velocity_boundary_cache, velocity,
                   zero_source);
}

/**
 * @brief Perform the pressure projection step: compute face velocities,
 *        assemble the pressure-Poisson RHS, solve for pressure, and
 *        correct the velocity field.
 *
 * @tparam Pack Tpetra type pack.
 * @param[in,out] pressure Pressure field (solved on output).
 * @param time_step Time-step size (must be positive).
 * @param velocity_boundary_cache Cached velocity boundary conditions.
 * @param[in,out] velocity Velocity field corrected by the pressure
 *        gradient on output.
 * @param right_hand_source Per-cell scalar source provider.
 * @throws std::invalid_argument on mesh mismatch or non-positive time
 *         step.
 */
template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::project(
    field_type& pressure,
    scalar_type time_step,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    velocity_field_type& velocity,
    const source_type& right_hand_source) -> ProjectionResult
{
    EquationValidation::require_mesh_match(*d_mesh, pressure,
                                           "PressureProjectionEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity,
                                           "PressureProjectionEquation");
    if (time_step <= 0.0)
    {
        throw std::invalid_argument("PressureProjectionEquation requires a positive time step.");
    }
    FVM::face_velocities(velocity, velocity_boundary_cache,
                         d_cached_face_velocity);
    FVM::normal_face_fluxes(d_cached_face_velocity,
                            d_cached_face_fluxes);
    const auto owned_map = require_owned_cell_map(d_mesh);
    const auto gauge_gid = owned_map->getMinAllGlobalIndex();
    if (d_cached_pressure_matrix.is_null())
    {
        rebuild_matrix();
    }
    if (d_cached_rhs.is_null())
    {
        d_cached_rhs = Teuchos::rcp(
            new typename Pack::vector_type(d_mesh->owned_cell_map(), true));
    }
    else
    {
        d_cached_rhs->putScalar(0.0);
    }

    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = owned_map->getGlobalElement(cell_lid);
        const auto rhs_value = row_gid == gauge_gid
                             ? scalar_type{}
                             : -FVM::cell_flux_balance<Pack>(
                                   *d_mesh, d_cached_face_fluxes, cell_lid)
                               / time_step
                               + d_mesh->cell_volume(cell_lid)
                               * right_hand_source(cell_lid);
        d_cached_rhs->replaceLocalValue(cell_lid, rhs_value);
    }

    pressure.owned_data().putScalar(0.0);
    Teuchos::RCP<const typename Pack::matrix_type> const_matrix =
        d_cached_pressure_matrix;
    const auto linear_statistics =
        d_linear_solver.solve_with_statistics(
            const_matrix, *d_cached_rhs, pressure.owned_data(),
            d_linear_options);
    if (!linear_statistics.converged)
    {
        throw std::runtime_error("PressureProjectionEquation projection solve did not converge.");
    }
    d_mesh->sync_periodic_boundaries(pressure);

    scalar_type pressure_norm_squared = {};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto correction = pressure.value(cell_lid);
        pressure_norm_squared += correction * correction
                               * d_mesh->cell_volume(cell_lid);
    }

    FVM::cell_gradient(pressure, d_cached_gradient);
    velocity.owned_data().update(
        -time_step, d_cached_gradient.owned_data(), 1.0);

    d_mesh->sync_periodic_boundaries(velocity);

    FVM::pressure_weighted_face_fluxes(
        velocity, pressure, time_step, velocity_boundary_cache,
        d_cached_face_fluxes);

    scalar_type continuity_norm_squared = {};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto balance =
            FVM::cell_flux_balance<Pack>(*d_mesh, d_cached_face_fluxes, cell_lid);
        continuity_norm_squared += balance * balance;
    }

    const std::array<scalar_type, 2> local_norms_squared{
        pressure_norm_squared,
        continuity_norm_squared};
    std::array<scalar_type, 2> global_norms_squared{};
    Teuchos::reduceAll(
        *owned_map->getComm(),
        Teuchos::REDUCE_SUM,
        2,
        local_norms_squared.data(),
        global_norms_squared.data());

    using std::sqrt;
    return {
        sqrt(global_norms_squared[0]),
        sqrt(global_norms_squared[1]),
        linear_statistics};
}

} // namespace SimpleFluid
