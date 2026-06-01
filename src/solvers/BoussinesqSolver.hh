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
#include "equations/EquationValidation.hh"
#include "equations/PressureProjectionEquation.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "equations/TimeStepperOptions.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "geometry/MeshUtils.hh"
#include "io/VTUWriter.hh"
#include "FVM/FvmOperators.hh"

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
    using face_flux_field_type = FaceField<Pack>;
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
    face_flux_field_type d_old_face_fluxes;
    face_flux_field_type d_projected_face_fluxes;

    scalar_type d_time = 0.0;
    int d_step_index = 0;
};

} // namespace SimpleFluid
