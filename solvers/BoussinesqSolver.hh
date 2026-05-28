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
#include "operators/FvmOperators.hh"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
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
    const field_type& velocity_x() const noexcept { return d_velocity_x; }
    const field_type& velocity_y() const noexcept { return d_velocity_y; }
    const field_type& velocity_z() const noexcept { return d_velocity_z; }

    field_type& temperature() noexcept { return d_temperature; }
    field_type& pressure() noexcept { return d_pressure; }
    field_type& velocity_x() noexcept { return d_velocity_x; }
    field_type& velocity_y() noexcept { return d_velocity_y; }
    field_type& velocity_z() noexcept { return d_velocity_z; }

    void write_vtu(const std::string& filename) const { d_mesh->export_vtu(filename); }
    void write_solution_vtu(const std::string& filename) const;

private:
    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh);
    static int vtu_cell_type(cell_type type);

    SP<const mesh_type> d_mesh;
    BoundaryConditionSet d_boundary_conditions;
    TimeStepperOptions d_time_options;
    LinearSolverOptions d_linear_options;

    TemperatureDiffusionEquation<Pack> d_temperature_equation;
    BoussinesqMomentumEquation<Pack> d_momentum_equation;
    PressureProjectionEquation<Pack> d_pressure_projection;

    field_type d_temperature;
    field_type d_pressure;
    field_type d_velocity_x;
    field_type d_velocity_y;
    field_type d_velocity_z;

    std::vector<scalar_type> d_old_temperature;
    std::vector<scalar_type> d_old_velocity_x;
    std::vector<scalar_type> d_old_velocity_y;
    std::vector<scalar_type> d_old_velocity_z;

    scalar_type d_time = 0.0;
    int d_step_index = 0;
};

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
      d_temperature_equation(d_mesh, d_boundary_conditions),
      d_momentum_equation(d_mesh),
      d_pressure_projection(d_mesh, d_linear_options),
      d_temperature(d_mesh, "temperature"),
      d_pressure(d_mesh, "pressure"),
      d_velocity_x(d_mesh, "velocity_x"),
      d_velocity_y(d_mesh, "velocity_y"),
      d_velocity_z(d_mesh, "velocity_z")
{
    if (d_time_options.time_step <= 0.0)
    {
        throw std::invalid_argument("BoussinesqSolver requires a positive time step.");
    }

    d_old_temperature.resize(d_mesh->num_local_cells());
    d_old_velocity_x.resize(d_mesh->num_local_cells());
    d_old_velocity_y.resize(d_mesh->num_local_cells());
    d_old_velocity_z.resize(d_mesh->num_local_cells());
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
        d_velocity_x.set_value(cell_lid, 0.0);
        d_velocity_y.set_value(cell_lid, 0.0);
        d_velocity_z.set_value(cell_lid, 0.0);
    }

    d_temperature.sync_ghosts();
    d_pressure.sync_ghosts();
    d_velocity_x.sync_ghosts();
    d_velocity_y.sync_ghosts();
    d_velocity_z.sync_ghosts();
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
 * Performs explicit thermal diffusion, buoyancy-driven velocity update,
 * and a pressure projection solve.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::step()
{
    if (d_step_index == 0)
    {
        d_temperature.sync_ghosts();
        d_velocity_x.sync_ghosts();
        d_velocity_y.sync_ghosts();
        d_velocity_z.sync_ghosts();
    }

    for (std::size_t cell = 0; cell < d_mesh->num_local_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        d_old_temperature[cell] = d_temperature.local_value(cell_lid);
        d_old_velocity_x[cell] = d_velocity_x.local_value(cell_lid);
        d_old_velocity_y[cell] = d_velocity_y.local_value(cell_lid);
        d_old_velocity_z[cell] = d_velocity_z.local_value(cell_lid);
    }

    const auto old_face_fluxes =
        FvmOperators::face_fluxes(*d_mesh, d_velocity_x, d_velocity_y,
                                  d_velocity_z, &d_boundary_conditions);
    d_momentum_equation.advance_velocity(d_old_velocity_x,
                                         d_old_velocity_y,
                                         d_old_velocity_z,
                                         old_face_fluxes,
                                         d_temperature,
                                         d_boundary_conditions,
                                         d_time_options,
                                         d_velocity_x,
                                         d_velocity_y,
                                         d_velocity_z,
                                         d_linear_options);
    d_pressure_projection.project(d_pressure,
                                  d_time_options.time_step,
                                  d_boundary_conditions,
                                  d_velocity_x,
                                  d_velocity_y,
                                  d_velocity_z);
    const auto projected_face_fluxes =
        FvmOperators::face_fluxes(*d_mesh, d_velocity_x, d_velocity_y,
                                  d_velocity_z, &d_boundary_conditions);
    d_temperature_equation.advance_semi_implicit(d_old_temperature,
                                                 projected_face_fluxes,
                                                 d_time_options.time_step,
                                                 d_time_options.thermal_diffusivity,
                                                 d_temperature,
                                                 d_linear_options);

    d_temperature.sync_ghosts();
    d_velocity_x.sync_ghosts();
    d_velocity_y.sync_ghosts();
    d_velocity_z.sync_ghosts();

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

template<TpetraTypePack Pack>
int BoussinesqSolver<Pack>::vtu_cell_type(cell_type type)
{
    switch (type)
    {
        case cell_type::HEXAHEDRON:
            return 12;
        case cell_type::TRIPRISM:
            return 13;
        default:
            break;
    }

    throw std::runtime_error("Solution VTU export encountered an unsupported cell type.");
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
    std::unordered_map<global_ordinal_type, local_ordinal_type> node_lid;
    std::vector<vec_type> node_coords;
    std::vector<local_ordinal_type> cell_node_offset{0};
    std::vector<local_ordinal_type> cell_node_ids;

    auto append_node = [&](global_ordinal_type node_gid) -> local_ordinal_type
    {
        const auto iter = node_lid.find(node_gid);
        if (iter != node_lid.end())
        {
            return iter->second;
        }

        const auto lid = static_cast<local_ordinal_type>(node_coords.size());
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
        cell_node_offset.push_back(static_cast<local_ordinal_type>(cell_node_ids.size()));
    }

    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error("Failed to open solution VTU output file: " + filename);
    }

    auto write_scalar_field = [&](const std::string& name, const field_type& field)
    {
        out << "        <DataArray type=\"Float64\" Name=\"" << name
            << "\" format=\"ascii\">\n";
        out << "          ";
        for (std::size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
        {
            out << field.local_value(static_cast<local_ordinal_type>(lid))
                << (lid + 1 == d_mesh->num_local_cells() ? "" : " ");
        }
        out << "\n        </DataArray>\n";
    };

    out << std::setprecision(std::numeric_limits<real_t>::max_digits10);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << node_coords.size()
        << "\" NumberOfCells=\"" << d_mesh->num_local_cells() << "\">\n";

    out << "      <PointData/>\n";
    out << "      <CellData>\n";
    write_scalar_field("temperature", d_temperature);
    write_scalar_field("pressure", d_pressure);
    write_scalar_field("velocity_x", d_velocity_x);
    write_scalar_field("velocity_y", d_velocity_y);
    write_scalar_field("velocity_z", d_velocity_z);
    out << "      </CellData>\n";

    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& coord : node_coords)
    {
        out << "          " << coord.x << " " << coord.y << " " << coord.z << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    out << "      <Cells>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (std::size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        const auto begin = static_cast<std::size_t>(cell_node_offset[lid]);
        const auto end = static_cast<std::size_t>(cell_node_offset[lid + 1]);
        out << "          ";
        for (std::size_t i = begin; i < end; ++i)
        {
            out << cell_node_ids[i] << (i + 1 == end ? "" : " ");
        }
        out << "\n";
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        out << cell_node_offset[lid + 1]
            << (lid + 1 == d_mesh->num_local_cells() ? "" : " ");
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (std::size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        out << vtu_cell_type(d_mesh->cell(static_cast<local_ordinal_type>(lid)).type)
            << (lid + 1 == d_mesh->num_local_cells() ? "" : " ");
    }
    out << "\n        </DataArray>\n";
    out << "      </Cells>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    if (!out)
    {
        throw std::runtime_error("Failed while writing solution VTU output file: " + filename);
    }
}

} // namespace SimpleFluid
