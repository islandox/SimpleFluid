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

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::require_mesh(SP<const mesh_type> mesh)
    -> SP<const mesh_type>
{
    return EquationValidation::require_non_null_mesh(
        std::move(mesh), "BoussinesqSolver");
}

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
          FvmOperators::cache_velocity_boundary_conditions<Pack>(
              d_mesh, d_boundary_conditions)),
      d_temperature_equation(d_mesh, d_boundary_conditions),
      d_momentum_equation(d_mesh),
      d_pressure_projection(d_mesh, d_linear_options),
      d_temperature(d_mesh, "temperature"),
      d_pressure(d_mesh, "pressure"),
      d_velocity(d_mesh, "velocity"),
      d_old_face_velocities(d_mesh, "old_face_velocity"),
      d_projected_face_velocities(d_mesh, "projected_face_velocity")
{
    if (d_time_options.time_step <= 0.0)
    {
        throw std::invalid_argument("BoussinesqSolver requires a positive time step.");
    }
}

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

    d_temperature.sync_ghosts();
    d_pressure.sync_ghosts();
    d_velocity.sync_ghosts();
}

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

template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::step()
{
    if (d_step_index == 0)
    {
        d_temperature.sync_ghosts();
        d_velocity.sync_ghosts();
    }

    FvmOperators::face_velocities(*d_mesh, d_velocity, d_velocity_boundary_cache,
                                  d_old_face_velocities);
    d_momentum_equation.advance_velocity(d_velocity,
                                         d_old_face_velocities,
                                         d_temperature,
                                         d_velocity_boundary_cache,
                                         d_time_options,
                                         d_velocity,
                                         d_linear_options);
    d_pressure_projection.project(d_pressure,
                                  d_time_options.time_step,
                                  d_velocity_boundary_cache,
                                  d_velocity);
    FvmOperators::face_velocities(*d_mesh, d_velocity, d_velocity_boundary_cache,
                                  d_projected_face_velocities);
    d_temperature_equation.advance_semi_implicit(d_temperature,
                                                 d_projected_face_velocities,
                                                 d_time_options.time_step,
                                                 d_time_options.thermal_diffusivity,
                                                 d_temperature,
                                                 d_linear_options);

    d_temperature.sync_ghosts();
    d_velocity.sync_ghosts();

    d_time += d_time_options.time_step;
    ++d_step_index;
}

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
