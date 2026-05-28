/**
 * @file BoussinesqSolver.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Minimal transient Boussinesq natural-convection driver.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "equations/BoussinesqMomentumEquation.hh"
#include "equations/PressureProjectionEquation.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "equations/TimeStepperOptions.hh"
#include "fields/CellField.hh"
#include "fields/VectorCellField.hh"
#include "geometry/MeshUtils.hh"
#include "io/VTUWriter.hh"
#include "operators/FvmOperators.hh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Minimal transient Boussinesq natural-convection solver.
 *
 * @tparam Pack Tpetra type pack used for vector storage and communication.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class BoussinesqSolver
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using vec_type = typename mesh_type::Vec3;
    using cell_type = typename mesh_type::CellType;

    BoussinesqSolver(SP<const mesh_type> mesh,
                     BoundaryConditionSet boundary_conditions,
                     TimeStepperOptions time_options = {},
                     LinearSolverOptions linear_options = {});

    void initialize_linear_temperature(const vec_type& direction,
                                       scalar_type hot_at_min,
                                       scalar_type cold_at_max,
                                       scalar_type initial_pressure = 0.0);

    void initialize_heated_box(scalar_type hot_temperature,
                               scalar_type cold_temperature,
                               scalar_type initial_pressure = 0.0);

    void initialize_bottom_hot_top_cold(scalar_type hot_temperature,
                                        scalar_type cold_temperature,
                                        scalar_type initial_pressure = 0.0);

    void step();
    void run(int steps);
    void run() { run(d_time_options.steps); }

    scalar_type time() const noexcept { return d_time; }
    int step_index() const noexcept { return d_step_index; }

    const field_type& temperature() const noexcept { return d_temperature; }
    const field_type& pressure() const noexcept { return d_pressure; }
    const velocity_field_type& velocity() const noexcept { return d_velocity; }

    field_type& temperature() noexcept { return d_temperature; }
    field_type& pressure() noexcept { return d_pressure; }
    velocity_field_type& velocity() noexcept { return d_velocity; }

    void write_vtu(const std::string& filename) const { d_mesh->export_vtu(filename); }
    void write_solution_vtu(const std::string& filename) const;

private:
    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh);

    SP<const mesh_type> d_mesh;
    BoundaryConditionSet d_boundary_conditions;
    TimeStepperOptions d_time_options;
    LinearSolverOptions d_linear_options;
    FvmOperators::VelocityBoundaryCache<Pack> d_velocity_boundary_cache;

    TemperatureDiffusionEquation<Pack> d_temperature_equation;
    BoussinesqMomentumEquation<Pack> d_momentum_equation;
    PressureProjectionEquation<Pack> d_pressure_projection;

    field_type d_temperature;
    field_type d_pressure;
    velocity_field_type d_velocity;

    std::vector<scalar_type> d_old_temperature;
    std::vector<vec_type> d_old_velocity;
    std::vector<scalar_type> d_old_face_fluxes;
    std::vector<scalar_type> d_projected_face_fluxes;

    scalar_type d_time = 0.0;
    int d_step_index = 0;
};

/**
 * @brief Validate and return the mesh pointer, throwing if null.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the mesh.
 * @return The validated mesh pointer.
 * @throws std::invalid_argument if the mesh is null.
 */
template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::require_mesh(SP<const mesh_type> mesh)
    -> SP<const mesh_type>
{
    if (!mesh)
    {
        throw std::invalid_argument("BoussinesqSolver requires a non-null mesh.");
    }

    return mesh;
}

/**
 * @brief Construct a Boussinesq solver with mesh, boundary conditions, and solver options.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the assembled mesh.
 * @param boundary_conditions Boundary condition set for temperature, velocity, and pressure.
 * @param time_options Time stepping and physical parameters.
 * @param linear_options Linear solver convergence options.
 * @throws std::invalid_argument if the mesh is null or the time step is non-positive.
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
          FvmOperators::cache_velocity_boundary_conditions<Pack>(
              *d_mesh, d_boundary_conditions)),
      d_temperature_equation(d_mesh, d_boundary_conditions),
      d_momentum_equation(d_mesh),
      d_pressure_projection(d_mesh, d_linear_options),
      d_temperature(d_mesh, "temperature"),
      d_pressure(d_mesh, "pressure"),
      d_velocity(d_mesh, "velocity")
{
    if (d_time_options.time_step <= 0.0)
    {
        throw std::invalid_argument("BoussinesqSolver requires a positive time step.");
    }

    d_old_temperature.resize(d_mesh->num_local_cells());
    d_old_velocity.resize(d_mesh->num_local_cells());
}

/**
 * @brief Initialize temperature field as a linear blend along a direction vector.
 *
 * @tparam Pack Tpetra type pack.
 * @param direction Spatial direction used to project cell centers.
 * @param hot_at_min Temperature at the minimum projected coordinate.
 * @param cold_at_max Temperature at the maximum projected coordinate.
 * @param initial_pressure Uniform initial pressure value.
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

    d_temperature.sync_ghosts();
    d_pressure.sync_ghosts();
    d_velocity.sync_ghosts();
}

/**
 * @brief Initialize temperature field as a linear x-direction box profile.
 *
 * @tparam Pack Tpetra type pack.
 * @param hot_temperature Temperature at the x-min boundary.
 * @param cold_temperature Temperature at the x-max boundary.
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
 * @brief Initialize temperature field as a bottom-hot/top-cold vertical vessel profile.
 *
 * @tparam Pack Tpetra type pack.
 * @param hot_temperature Temperature at the z-min side of the vessel.
 * @param cold_temperature Temperature at the z-max side of the vessel.
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
 * Performs semi-implicit velocity transport, pressure projection, and
 * semi-implicit temperature convection-diffusion.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::step()
{
    if (d_step_index == 0)
    {
        d_temperature.sync_ghosts();
        d_velocity.sync_ghosts();
    }

    for (std::size_t cell = 0; cell < d_mesh->num_local_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        d_old_temperature[cell] = d_temperature.local_value(cell_lid);
        d_old_velocity[cell] = d_velocity.local_value(cell_lid);
    }

    FvmOperators::face_fluxes(*d_mesh, d_velocity, d_velocity_boundary_cache,
                              d_old_face_fluxes);
    d_momentum_equation.advance_velocity(d_old_velocity,
                                         d_old_face_fluxes,
                                         d_temperature,
                                         d_boundary_conditions,
                                         d_time_options,
                                         d_velocity,
                                         d_linear_options);
    d_pressure_projection.project(d_pressure,
                                  d_time_options.time_step,
                                  d_velocity_boundary_cache,
                                  d_velocity);
    FvmOperators::face_fluxes(*d_mesh, d_velocity, d_velocity_boundary_cache,
                              d_projected_face_fluxes);
    d_temperature_equation.advance_semi_implicit(d_old_temperature,
                                                 d_projected_face_fluxes,
                                                 d_time_options.time_step,
                                                 d_time_options.thermal_diffusivity,
                                                 d_temperature,
                                                 d_linear_options);

    d_temperature.sync_ghosts();
    d_velocity.sync_ghosts();

    d_time += d_time_options.time_step;
    ++d_step_index;
}

/**
 * @brief Advance the solution by a specified number of time steps.
 *
 * @tparam Pack Tpetra type pack.
 * @param steps Number of time steps to perform.
 * @throws std::invalid_argument if steps is negative.
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
 * @brief Write mesh geometry plus Boussinesq cell fields to VTU.
 *
 * @tparam Pack Tpetra type pack.
 * @param filename Output VTU filename.
 * @throws std::runtime_error if file writing fails.
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
