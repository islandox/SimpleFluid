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
#include "geometry/MeshFactory.hh"

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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::require_legacy_mesh(
    const SP<const MeshHandle<Pack>>& mesh) -> SP<const mesh_type>
{
    if (!mesh)
    {
        throw std::invalid_argument(
            "BoussinesqSolver requires a non-null mesh handle.");
    }
    auto legacy = mesh->legacy_mesh();
    if (legacy)
    {
        return legacy;
    }

    const auto* cartesian =
        std::get_if<typename MeshHandle<Pack>::CartesianPtr>(
            &mesh->variant());
    if (!cartesian)
    {
        throw std::invalid_argument(
            "BoussinesqSolver currently supports STK and Cartesian "
            "mesh handles.");
    }
    if (mesh->num_owned_cells() != (*cartesian)->num_cells())
    {
        throw std::invalid_argument(
            "BoussinesqSolver Cartesian compatibility currently "
            "requires a serial mesh.");
    }

    auto database = std::make_shared<Database>();
    database->set("dimension", 3);
    database->set("mesh_size", real_t{1.0});
    database->set(
        "domain_type",
        static_cast<int>(MeshFactory::DomainType::BOX));
    database->set("X", ArrReal((*cartesian)->cell_edges()[0]));
    database->set("Y", ArrReal((*cartesian)->cell_edges()[1]));
    database->set("Z", ArrReal((*cartesian)->cell_edges()[2]));
    database->set(
        "domain_exterior_face_types",
        ArrString{
            "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    return MeshFactory(database).template build<Pack>();
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
    : BoussinesqSolver(
          std::make_shared<MeshHandle<Pack>>(
              require_mesh(std::move(mesh))),
          std::move(boundary_conditions),
          time_options,
          linear_options)
{
}

template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(
    SP<const MeshHandle<Pack>> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : d_mesh(require_legacy_mesh(mesh)),
      d_problem(std::make_shared<MeshHandle<Pack>>(d_mesh),
                std::move(boundary_conditions),
                time_options,
                linear_options)
{
    if (d_problem.time_options().time_step <= 0.0)
    {
        throw std::invalid_argument("BoussinesqSolver requires a positive time step.");
    }

    d_problem.template emplace_object<FVM::VelocityBoundaryCache<Pack>>(
        "velocity_boundary_cache",
        FVM::cache_velocity_boundary_conditions<Pack>(
            d_mesh, d_problem.boundary_conditions()));
    d_problem.template emplace_object<TemperatureDiffusionEquation<Pack>>(
        "temperature_equation",
        d_mesh,
        d_problem.boundary_conditions());
    d_problem.template emplace_object<BoussinesqMomentumEquation<Pack>>(
        "momentum_equation", d_mesh);
    d_problem.template emplace_object<PressureProjectionEquation<Pack>>(
        "pressure_projection", d_mesh, d_problem.linear_options());
    d_problem.template emplace_object<field_type>(
        "temperature", d_mesh, "temperature");
    d_problem.template emplace_object<field_type>(
        "pressure", d_mesh, "pressure");
    d_problem.template emplace_object<field_type>(
        "pressure_correction", d_mesh, "pressure_correction");
    d_problem.template emplace_object<velocity_field_type>(
        "velocity", d_mesh, "velocity");
    d_problem.template emplace_object<velocity_field_type>(
        "pressure_velocity_predictor",
        d_mesh,
        "pressure_velocity_predictor");
    d_problem.template emplace_object<face_flux_field_type>(
        "old_face_flux", d_mesh, "old_face_flux");
    d_problem.template emplace_object<face_flux_field_type>(
        "projected_face_flux", d_mesh, "projected_face_flux");
    d_problem.template emplace_object<residual_type>(
        "pressure_velocity_residuals");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature() const noexcept
    -> const field_type&
{
    return d_problem.template object<field_type>("temperature");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure() const noexcept
    -> const field_type&
{
    return d_problem.template object<field_type>("pressure");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::velocity() const noexcept
    -> const velocity_field_type&
{
    return d_problem.template object<velocity_field_type>("velocity");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature() noexcept -> field_type&
{
    return d_problem.template object<field_type>("temperature");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure() noexcept -> field_type&
{
    return d_problem.template object<field_type>("pressure");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::velocity() noexcept
    -> velocity_field_type&
{
    return d_problem.template object<velocity_field_type>("velocity");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::last_pressure_velocity_residuals()
    const noexcept -> const residual_type&
{
    return pressure_velocity_residuals();
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::temperature_equation()
    -> TemperatureDiffusionEquation<Pack>&
{
    return d_problem.template object<
        TemperatureDiffusionEquation<Pack>>("temperature_equation");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::momentum_equation()
    -> BoussinesqMomentumEquation<Pack>&
{
    return d_problem.template object<
        BoussinesqMomentumEquation<Pack>>("momentum_equation");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure_projection()
    -> PressureProjectionEquation<Pack>&
{
    return d_problem.template object<
        PressureProjectionEquation<Pack>>("pressure_projection");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::velocity_boundary_cache()
    -> FVM::VelocityBoundaryCache<Pack>&
{
    return d_problem.template object<
        FVM::VelocityBoundaryCache<Pack>>("velocity_boundary_cache");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure_correction() -> field_type&
{
    return d_problem.template object<field_type>("pressure_correction");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::predictor_velocity()
    -> velocity_field_type&
{
    return d_problem.template object<velocity_field_type>(
        "pressure_velocity_predictor");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::old_face_fluxes()
    -> face_flux_field_type&
{
    return d_problem.template object<face_flux_field_type>("old_face_flux");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::projected_face_fluxes()
    -> face_flux_field_type&
{
    return d_problem.template object<face_flux_field_type>(
        "projected_face_flux");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure_velocity_residuals()
    -> residual_type&
{
    return d_problem.template object<residual_type>(
        "pressure_velocity_residuals");
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::pressure_velocity_residuals()
    const -> const residual_type&
{
    return d_problem.template object<residual_type>(
        "pressure_velocity_residuals");
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
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
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
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        predictor_velocity().set_value(cell_lid, velocity().value(cell_lid));
    }
    d_mesh->sync_periodic_boundaries(predictor_velocity());

    FVM::face_fluxes(velocity(), velocity_boundary_cache(),
                              old_face_fluxes());
    momentum_equation().advance_velocity(velocity(),
                                         old_face_fluxes(),
                                         temperature(),
                                         velocity_boundary_cache(),
                                         d_problem.time_options(),
                                         velocity(),
                                         d_problem.linear_options());
    pressure_velocity_residuals().momentum =
        velocity_update_norm(predictor_velocity(), velocity());
}

/**
 * @brief Run one pressure-correction solve and accumulate pressure.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::run_pressure_correction()
    -> typename PressureProjectionEquation<Pack>::ProjectionResult
{
    const auto result =
        pressure_projection().project(pressure_correction(),
                                      d_problem.time_options().time_step,
                                      velocity_boundary_cache(),
                                      velocity());
    pressure().owned_data().update(1.0, pressure_correction().owned_data(), 1.0);
    d_mesh->sync_periodic_boundaries(pressure());
    return result;
}

/**
 * @brief Dispatch SIMPLE, PISO, or PIMPLE pressure-velocity coupling loops.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::solve_pressure_velocity_coupling()
{
    if (d_problem.time_options().n_pressure_correctors < 1)
    {
        throw std::invalid_argument(
            "BoussinesqSolver requires at least one pressure corrector.");
    }
    if (d_problem.time_options().n_outer_correctors < 1)
    {
        throw std::invalid_argument(
            "BoussinesqSolver requires at least one outer corrector.");
    }

    pressure_velocity_residuals() = {};

    const auto pressure_corrections =
        d_problem.time_options().pressure_velocity_coupling == PressureVelocityCoupling::SIMPLE
      ? 1
      : d_problem.time_options().n_pressure_correctors;
    const auto outer_corrections =
        d_problem.time_options().pressure_velocity_coupling == PressureVelocityCoupling::PIMPLE
      ? d_problem.time_options().n_outer_correctors
      : 1;

    for (int outer = 0; outer < outer_corrections; ++outer)
    {
        run_momentum_predictor();

        typename PressureProjectionEquation<Pack>::ProjectionResult result;
        for (int corrector = 0; corrector < pressure_corrections; ++corrector)
        {
            result = run_pressure_correction();
        }

        pressure_velocity_residuals().pressure =
            result.pressure_correction;
        pressure_velocity_residuals().continuity =
            result.continuity;
    }

    FVM::face_fluxes(velocity(), velocity_boundary_cache(),
                              projected_face_fluxes());
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
    for (size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto projected = d_mesh->cell_centroid(cell_lid).dot(direction);
        min_projected = std::min(min_projected, projected);
        max_projected = std::max(max_projected, projected);
    }

    const auto width = max_projected > min_projected
                     ? max_projected - min_projected
                     : 1.0;
    for (size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto projected = d_mesh->cell_centroid(cell_lid).dot(direction);
        const auto blend = (projected - min_projected) / width;
        temperature().set_value(cell_lid,
                                hot_at_min * (1.0 - blend)
                              + cold_at_max * blend);
        pressure().set_value(cell_lid, initial_pressure);
        velocity().set_value(cell_lid, {});
    }

    d_mesh->sync_periodic_boundaries(temperature());
    d_mesh->sync_periodic_boundaries(pressure());
    d_mesh->sync_periodic_boundaries(velocity());
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
        d_mesh->sync_periodic_boundaries(temperature());
        d_mesh->sync_periodic_boundaries(velocity());
    }

    solve_pressure_velocity_coupling();
    temperature_equation().advance_semi_implicit(temperature(),
                                                 projected_face_fluxes(),
                                                 d_problem.time_options().time_step,
                                                 d_problem.time_options().thermal_diffusivity,
                                                 temperature(),
                                                 d_problem.linear_options());

    d_mesh->sync_periodic_boundaries(temperature());
    d_mesh->sync_periodic_boundaries(velocity());

    d_time += d_problem.time_options().time_step;
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

    for (size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
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
    for (size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(lid);
        temperature_values.push_back(static_cast<real_t>(
            temperature().local_value(cell_lid)));
        pressure_values.push_back(static_cast<real_t>(
            pressure().local_value(cell_lid)));
        velocity_values.push_back(velocity().local_value(cell_lid));
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
