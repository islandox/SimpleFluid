/**
 * @file BoussinesqSolver.hh
 * @brief Minimal transient Boussinesq natural-convection driver.
 */
#pragma once

#include "fields/CellField.hh"
#include "operators/FvmOperators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace SimpleFluid
{

enum class BoundaryConditionType : uint8_t
{
    Dirichlet = 0,
    Neumann = 1,
    NoSlip = 2
};

struct BoundaryCondition
{
    BoundaryConditionType type = BoundaryConditionType::Neumann;
    real_t value = 0.0;
};

struct BoundaryConditionSet
{
    std::unordered_map<std::string, BoundaryCondition> temperature;
    std::unordered_map<std::string, BoundaryCondition> velocity;
    std::unordered_map<std::string, BoundaryCondition> pressure;
};

struct TimeStepperOptions
{
    real_t time_step = 1.0e-3;
    int steps = 1;
    real_t thermal_diffusivity = 1.0e-3;
    real_t kinematic_viscosity = 1.0e-3;
    real_t thermal_expansion = 3.0e-3;
    real_t gravity_z = -9.81;
    real_t reference_temperature = 0.5;
};

template<TpetraTypePack Pack = DefaultTpetraTypes>
class BoussinesqSolver
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using matrix_type = typename Pack::matrix_type;
    using vector_type = typename Pack::vector_type;

    BoussinesqSolver(SP<const mesh_type> mesh,
                     BoundaryConditionSet boundary_conditions,
                     TimeStepperOptions time_options = {},
                     LinearSolverOptions linear_options = {});

    void initialize_heated_box(scalar_type hot_temperature,
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

private:
    void refresh_temperature_boundary_cache();
    scalar_type cached_temperature_boundary_value(local_ordinal_type face_lid,
                                                  scalar_type fallback) const;
    void solve_pressure_projection();

    SP<const mesh_type> d_mesh;
    BoundaryConditionSet d_boundary_conditions;
    TimeStepperOptions d_time_options;
    LinearSolverOptions d_linear_options;

    field_type d_temperature;
    field_type d_pressure;
    field_type d_velocity_x;
    field_type d_velocity_y;
    field_type d_velocity_z;

    Teuchos::RCP<matrix_type> d_pressure_matrix;
    vector_type d_pressure_rhs;

    std::vector<scalar_type> d_old_temperature;
    std::vector<scalar_type> d_old_velocity_z;
    std::vector<scalar_type> d_face_boundary_temperature;
    std::vector<std::uint8_t> d_face_has_dirichlet_temperature;

    scalar_type d_time = 0.0;
    int d_step_index = 0;
};

template<TpetraTypePack Pack>
BoussinesqSolver<Pack>::BoussinesqSolver(
    SP<const mesh_type> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : d_mesh(std::move(mesh)),
      d_boundary_conditions(std::move(boundary_conditions)),
      d_time_options(time_options),
      d_linear_options(linear_options),
      d_temperature(d_mesh, "temperature"),
      d_pressure(d_mesh, "pressure"),
      d_velocity_x(d_mesh, "velocity_x"),
      d_velocity_y(d_mesh, "velocity_y"),
      d_velocity_z(d_mesh, "velocity_z"),
      d_pressure_rhs(d_mesh->owned_cell_map(), true)
{
    if (!d_mesh)
    {
        throw std::invalid_argument("BoussinesqSolver requires a non-null mesh.");
    }
    if (d_time_options.time_step <= 0.0)
    {
        throw std::invalid_argument("BoussinesqSolver requires a positive time step.");
    }

    d_old_temperature.resize(d_mesh->num_local_cells());
    d_old_velocity_z.resize(d_mesh->num_local_cells());
    refresh_temperature_boundary_cache();
}

template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::initialize_heated_box(
    scalar_type hot_temperature,
    scalar_type cold_temperature,
    scalar_type initial_pressure)
{
    if (d_mesh->num_owned_cells() == 0)
    {
        return;
    }

    auto xmin = d_mesh->cell_centroid(0).x;
    auto xmax = xmin;
    for (std::size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto x = d_mesh->cell_centroid(cell_lid).x;
        xmin = std::min(xmin, x);
        xmax = std::max(xmax, x);
    }

    const auto width = xmax > xmin ? xmax - xmin : 1.0;
    for (std::size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto x = d_mesh->cell_centroid(cell_lid).x;
        const auto blend = (x - xmin) / width;
        d_temperature.set_value(cell_lid,
                                hot_temperature * (1.0 - blend)
                              + cold_temperature * blend);
        d_pressure.set_value(cell_lid, initial_pressure);
        d_velocity_x.set_value(cell_lid, 0.0);
        d_velocity_y.set_value(cell_lid, 0.0);
        d_velocity_z.set_value(cell_lid, 0.0);
    }

    d_temperature.sync_ghosts();
    d_pressure.sync_ghosts();
    d_velocity_z.sync_ghosts();
}

template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::refresh_temperature_boundary_cache()
{
    d_face_boundary_temperature.assign(d_mesh->num_faces(), scalar_type{});
    d_face_has_dirichlet_temperature.assign(d_mesh->num_faces(), std::uint8_t{0});

    for (std::size_t face = 0; face < d_mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<local_ordinal_type>(face);
        if (!d_mesh->is_boundary_face(face_lid))
        {
            continue;
        }

        const auto iter =
            d_boundary_conditions.temperature.find(d_mesh->boundary_name(face_lid));
        if (iter == d_boundary_conditions.temperature.end()
            || iter->second.type != BoundaryConditionType::Dirichlet)
        {
            continue;
        }

        d_face_boundary_temperature[face] = iter->second.value;
        d_face_has_dirichlet_temperature[face] = 1;
    }
}

template<TpetraTypePack Pack>
auto BoussinesqSolver<Pack>::cached_temperature_boundary_value(
    local_ordinal_type face_lid,
    scalar_type fallback) const -> scalar_type
{
    const auto index = static_cast<std::size_t>(face_lid);
    return d_face_has_dirichlet_temperature[index]
         ? d_face_boundary_temperature[index]
         : fallback;
}

template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::solve_pressure_projection()
{
    if (d_pressure_matrix == Teuchos::null)
    {
        d_pressure_matrix =
            FvmOperators::identity_matrix<Pack>(d_mesh->owned_cell_map());
    }

    d_pressure_rhs.putScalar(0.0);
    d_pressure.owned_data().putScalar(0.0);

    auto pressure_operator =
        Teuchos::rcp_implicit_cast<const typename Pack::operator_type>(
            d_pressure_matrix);
    const bool converged =
        solve_linear_system<Pack>(pressure_operator, d_pressure_rhs,
                                  d_pressure.owned_data(), d_linear_options);
    if (!converged)
    {
        throw std::runtime_error("Boussinesq pressure projection did not converge.");
    }

    d_pressure.sync_ghosts();
}

template<TpetraTypePack Pack>
void BoussinesqSolver<Pack>::step()
{
    d_temperature.sync_ghosts();
    d_velocity_z.sync_ghosts();

    if (d_old_temperature.size() != d_mesh->num_local_cells())
    {
        d_old_temperature.resize(d_mesh->num_local_cells());
        d_old_velocity_z.resize(d_mesh->num_local_cells());
    }

    for (std::size_t cell = 0; cell < d_mesh->num_local_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        d_old_temperature[cell] = d_temperature.local_value(cell_lid);
        d_old_velocity_z[cell] = d_velocity_z.local_value(cell_lid);
    }

    for (std::size_t cell = 0; cell < d_mesh->num_owned_cells(); ++cell)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(cell);
        const auto temp_p = d_old_temperature[cell];
        scalar_type laplacian = 0.0;

        const auto& faces = d_mesh->faces(cell_lid);
        const auto& face_distances = d_mesh->face_distances(cell_lid);
        for (std::size_t face_index = 0; face_index < faces.size(); ++face_index)
        {
            const auto face_lid = faces[face_index];

            if (d_mesh->is_interior_face(face_lid))
            {
                const auto other = d_mesh->opposite_cell(face_lid, cell_lid);
                const auto distance = d_mesh->face_cell_center_distance(face_lid);
                if (distance > 0.0)
                {
                    laplacian += (d_old_temperature[static_cast<std::size_t>(other)] - temp_p)
                               * d_mesh->face_area(face_lid)
                               / distance;
                }
            }
            else if (const auto distance_to_face = face_distances[face_index];
                     distance_to_face > 0.0)
            {
                const auto boundary_temperature =
                    cached_temperature_boundary_value(face_lid, temp_p);
                laplacian += (boundary_temperature - temp_p)
                           * d_mesh->face_area(face_lid)
                           / distance_to_face;
            }
        }

        laplacian /= d_mesh->cell_volume(cell_lid);
        const auto new_temperature =
            temp_p + d_time_options.time_step
                   * d_time_options.thermal_diffusivity
                   * laplacian;
        d_temperature.set_value(cell_lid, new_temperature);

        const auto buoyancy =
            d_time_options.thermal_expansion
          * (new_temperature - d_time_options.reference_temperature)
          * (-d_time_options.gravity_z);
        const auto damping =
            1.0 / (1.0 + d_time_options.time_step
                         * d_time_options.kinematic_viscosity);
        d_velocity_z.set_value(cell_lid,
                               (d_old_velocity_z[cell]
                              + d_time_options.time_step * buoyancy) * damping);
    }

    solve_pressure_projection();

    d_temperature.sync_ghosts();
    d_velocity_z.sync_ghosts();

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

} // namespace SimpleFluid
