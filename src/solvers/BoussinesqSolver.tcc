/**
 * @file BoussinesqSolver.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template method implementations for BoussinesqSolver.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "BoussinesqSolver.hh"

namespace SimpleFluid
{

/**
 * @brief Validate and store a non-null mesh pointer.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the mesh.
 * @return The validated mesh shared pointer.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::require_mesh(SP<const mesh_type> mesh)
    -> SP<const mesh_type>
{
    return EquationValidation::require_non_null_mesh(
        std::move(mesh), "BoussinesqSolver");
}

/**
 * @brief Construct a Boussinesq solver with mesh, boundary conditions, and solver options.
 *
 * Initialises all field and equation objects.  Validates that the time step is positive.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Computational mesh.
 * @param boundary_conditions Boundary condition set for velocity and temperature.
 * @param time_options Time-stepping parameters.
 * @param linear_options Linear solver parameters.
 * @throws std::invalid_argument If the time step is non-positive.
 */
template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(
    SP<const mesh_type> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : d_mesh(require_mesh(std::move(mesh))),
      d_boundary_conditions(std::move(boundary_conditions)),
      d_time_options(time_options),
      d_linear_options(linear_options),
      d_velocity_boundary_cache(
          FVM::cache_velocity_boundary_conditions<Pack>(
              d_mesh, d_boundary_conditions)),
      d_temperature_equation(d_mesh, d_boundary_conditions),
      d_momentum_equation(d_mesh),
      d_pressure_projection(d_mesh, d_linear_options),
      d_temperature(d_mesh, "temperature"),
      d_pressure(d_mesh, "pressure"),
      d_pressure_correction(d_mesh, "pressure_correction"),
      d_velocity(d_mesh, "velocity"),
      d_predictor_velocity(d_mesh, "pressure_velocity_predictor"),
      d_old_face_fluxes(d_mesh, "old_face_flux"),
      d_projected_face_fluxes(d_mesh, "projected_face_flux")
{
    if (d_time_options.time_step <= 0.0)
    {
        throw std::invalid_argument("BoussinesqSolver requires a positive time step.");
    }
}

/**
 * @brief Compute the volume-weighted L2 norm of a velocity-field update.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::velocity_update_norm(
    const velocity_field_type& before,
    const velocity_field_type& after) const -> scalar_type
{
    EquationValidation::require_mesh_match(*d_mesh, before, "BoussinesqSolver");
    EquationValidation::require_mesh_match(*d_mesh, after, "BoussinesqSolver");

    scalar_type norm_squared = {};
    for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto delta = after.value(cell_lid) - before.value(cell_lid);
        norm_squared += delta.dot(delta) * d_mesh->cell_volume(cell_lid);
    }

    using std::sqrt;
    return sqrt(norm_squared);
}

/**
 * @brief Solve the semi-implicit momentum predictor and report its update norm.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::run_momentum_predictor()
{
    for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        d_predictor_velocity.set_value(cell_lid, d_velocity.value(cell_lid));
    }
    d_mesh->sync_periodic_boundaries(d_predictor_velocity);

    FVM::face_fluxes(d_velocity, d_velocity_boundary_cache,
                              d_old_face_fluxes);
    d_momentum_equation.advance_velocity(d_velocity,
                                         d_old_face_fluxes,
                                         d_temperature,
                                         d_velocity_boundary_cache,
                                         d_time_options,
                                         d_velocity,
                                         d_linear_options);
    d_last_pressure_velocity_residuals.momentum =
        velocity_update_norm(d_predictor_velocity, d_velocity);
}

/**
 * @brief Run one pressure-correction solve and accumulate pressure.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::run_pressure_correction()
    -> typename PressureProjectionEquation<Pack>::ProjectionResult
{
    const auto result =
        d_pressure_projection.project(d_pressure_correction,
                                      d_time_options.time_step,
                                      d_velocity_boundary_cache,
                                      d_velocity);
    d_pressure.owned_data().update(1.0, d_pressure_correction.owned_data(), 1.0);
    d_mesh->sync_periodic_boundaries(d_pressure);
    return result;
}

/**
 * @brief Dispatch SIMPLE, PISO, or PIMPLE pressure-velocity coupling loops.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::solve_pressure_velocity_coupling()
{
    if (d_time_options.n_pressure_correctors < 1)
    {
        throw std::invalid_argument(
            "BoussinesqSolver requires at least one pressure corrector.");
    }
    if (d_time_options.n_outer_correctors < 1)
    {
        throw std::invalid_argument(
            "BoussinesqSolver requires at least one outer corrector.");
    }

    d_last_pressure_velocity_residuals = {};

    const auto pressure_corrections =
        d_time_options.pressure_velocity_coupling == PressureVelocityCoupling::SIMPLE
      ? 1
      : d_time_options.n_pressure_correctors;
    const auto outer_corrections =
        d_time_options.pressure_velocity_coupling == PressureVelocityCoupling::PIMPLE
      ? d_time_options.n_outer_correctors
      : 1;

    for (int outer = 0; outer < outer_corrections; ++outer)
    {
        run_momentum_predictor();

        typename PressureProjectionEquation<Pack>::ProjectionResult result;
        for (int corrector = 0; corrector < pressure_corrections; ++corrector)
        {
            result = run_pressure_correction();
        }

        d_last_pressure_velocity_residuals.pressure =
            result.pressure_correction;
        d_last_pressure_velocity_residuals.continuity =
            result.continuity;
    }

    FVM::face_fluxes(d_velocity, d_velocity_boundary_cache,
                              d_projected_face_fluxes);
}

/**
 * @brief Initialise the temperature field as a linear ramp along a given direction.
 *
 * Sets temperature to hot_at_min at the minimum projection onto direction and
 * cold_at_max at the maximum.  Also zeroes velocity and sets a uniform initial
 * pressure.
 *
 * @tparam Pack Tpetra type pack.
 * @param direction Unit direction vector for the temperature gradient.
 * @param hot_at_min Temperature at the minimum projection point.
 * @param cold_at_max Temperature at the maximum projection point.
 * @param initial_pressure Uniform initial pressure value.
 * @throws std::invalid_argument If direction has zero norm.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::initialize_linear_temperature(
    const vec_type& direction,
    scalar_type hot_at_min,
    scalar_type cold_at_max,
    scalar_type initial_pressure)
{
    if (direction.norm() <= 0.0)
    {
        throw std::invalid_argument("BoussinesqSolver requires a nonzero initialization direction.");
    }
    if (d_mesh->num_owned_cells() == 0)
    {
        return;
    }

    auto min_projected = d_mesh->cell_centroid(0).dot(direction);
    auto max_projected = min_projected;
    for (std::size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto projected = d_mesh->cell_centroid(cell_lid).dot(direction);
        min_projected = std::min(min_projected, projected);
        max_projected = std::max(max_projected, projected);
    }

    const auto width = max_projected > min_projected
                     ? max_projected - min_projected
                     : 1.0;
    for (std::size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto projected = d_mesh->cell_centroid(cell_lid).dot(direction);
        const auto blend = (projected - min_projected) / width;
        d_temperature.set_value(cell_lid,
                                hot_at_min * (1.0 - blend)
                              + cold_at_max * blend);
        d_pressure.set_value(cell_lid, initial_pressure);
        d_velocity.set_value(cell_lid, {});
    }

    d_mesh->sync_periodic_boundaries(d_temperature);
    d_mesh->sync_periodic_boundaries(d_pressure);
    d_mesh->sync_periodic_boundaries(d_velocity);
}

/**
 * @brief Initialise fields for a heated-box problem with temperature gradient along X.
 *
 * Delegates to initialize_linear_temperature with direction (1, 0, 0).
 *
 * @tparam Pack Tpetra type pack.
 * @param hot_temperature Temperature at the hot (xmin) boundary.
 * @param cold_temperature Temperature at the cold (xmax) boundary.
 * @param initial_pressure Uniform initial pressure value.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::initialize_heated_box(
    scalar_type hot_temperature,
    scalar_type cold_temperature,
    scalar_type initial_pressure)
{
    initialize_linear_temperature({1.0, 0.0, 0.0},
                                  hot_temperature,
                                  cold_temperature,
                                  initial_pressure);
}

/**
 * @brief Initialise fields for a bottom-hot, top-cold problem with temperature gradient along Z.
 *
 * Delegates to initialize_linear_temperature with direction (0, 0, 1).
 *
 * @tparam Pack Tpetra type pack.
 * @param hot_temperature Temperature at the bottom (zmin) boundary.
 * @param cold_temperature Temperature at the top (zmax) boundary.
 * @param initial_pressure Uniform initial pressure value.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::initialize_bottom_hot_top_cold(
    scalar_type hot_temperature,
    scalar_type cold_temperature,
    scalar_type initial_pressure)
{
    initialize_linear_temperature({0.0, 0.0, 1.0},
                                  hot_temperature,
                                  cold_temperature,
                                  initial_pressure);
}

/**
 * @brief Advance the solution by one time step.
 *
 * Performs: face-flux computation, momentum advance, pressure projection,
 * corrected face fluxes, and semi-implicit temperature advance.  Periodic
 * boundary synchronisation is applied at the start of the first step and
 * after the updates.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::step()
{
    if (d_step_index == 0)
    {
        d_mesh->sync_periodic_boundaries(d_temperature);
        d_mesh->sync_periodic_boundaries(d_velocity);
    }

    solve_pressure_velocity_coupling();
    d_temperature_equation.advance_semi_implicit(d_temperature,
                                                 d_projected_face_fluxes,
                                                 d_time_options.time_step,
                                                 d_time_options.thermal_diffusivity,
                                                 d_temperature,
                                                 d_linear_options);

    d_mesh->sync_periodic_boundaries(d_temperature);
    d_mesh->sync_periodic_boundaries(d_velocity);

    d_time += d_time_options.time_step;
    ++d_step_index;
}

/**
 * @brief Run the solver for a specified number of time steps.
 *
 * @tparam Pack Tpetra type pack.
 * @param steps Number of time steps to execute (defaults to the value configured
 *              in TimeStepperOptions).
 * @throws std::invalid_argument If steps is negative.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::run(int steps)
{
    if (steps < 0)
    {
        throw std::invalid_argument("BoussinesqSolver::run steps cannot be negative.");
    }

    for (int step_id = 0; step_id < steps; ++step_id)
    {
        step();
    }
}

/**
 * @brief Write the current solution fields to a VTU file.
 *
 * Converts the internal mesh and cell fields (temperature, pressure, velocity)
 * into VTU format and writes to the specified path.
 *
 * @tparam Pack Tpetra type pack.
 * @param filename Output .vtu file path.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::write_solution_vtu(const std::string& filename) const
{
    std::unordered_map<global_ordinal_type, global_index_t> node_lid;
    VTUWriter::VectorData node_coords;
    VTUWriter::Int64Data cell_node_offsets;
    VTUWriter::Int64Data cell_node_ids;
    VTUWriter::UInt8Data cell_types;

    auto append_node = [&](global_ordinal_type node_gid) -> global_index_t
    {
        const auto iter = node_lid.find(node_gid);
        if (iter != node_lid.end())
        {
            return iter->second;
        }

        const auto lid = static_cast<global_index_t>(node_coords.size());
        node_lid.emplace(node_gid, lid);
        node_coords.push_back(d_mesh->node_coord(node_gid));
        return lid;
    };

    for (std::size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        const auto& cell_info = d_mesh->cell(static_cast<local_ordinal_type>(lid));
        for (const auto node_gid : cell_info.node_gids)
        {
            cell_node_ids.push_back(append_node(node_gid));
        }
        cell_node_offsets.push_back(static_cast<global_index_t>(cell_node_ids.size()));
        cell_types.push_back(static_cast<std::uint8_t>(
            MeshUtils::vtu_cell_type_code(cell_info.type)));
    }

    VTUWriter::ScalarData temperature_values;
    VTUWriter::ScalarData pressure_values;
    VTUWriter::VectorData velocity_values;
    temperature_values.reserve(d_mesh->num_local_cells());
    pressure_values.reserve(d_mesh->num_local_cells());
    velocity_values.reserve(d_mesh->num_local_cells());
    for (std::size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(lid);
        temperature_values.push_back(static_cast<real_t>(
            d_temperature.local_value(cell_lid)));
        pressure_values.push_back(static_cast<real_t>(
            d_pressure.local_value(cell_lid)));
        velocity_values.push_back(d_velocity.local_value(cell_lid));
    }

    VTUWriter writer;
    writer.set_points(std::move(node_coords));
    writer.set_cells(std::move(cell_node_ids),
                     std::move(cell_node_offsets),
                     std::move(cell_types));
    writer.add_scalar_cell_data("temperature", std::move(temperature_values));
    writer.add_scalar_cell_data("pressure", std::move(pressure_values));
    writer.add_vector_cell_data("velocity", std::move(velocity_values));
    writer.write(filename);
}

} // namespace SimpleFluid
