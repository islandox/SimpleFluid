/**
 * @file FluidSolver.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Transient incompressible pressure-velocity solver.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "equations/IncompressibleMomentumEquation.hh"
#include "equations/PressureProjectionEquation.hh"
#include "equations/TimeStepperOptions.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "geometry/MeshUtils.hh"
#include "io/VTUWriter.hh"
#include "problems/Problem.hh"
#include "solvers/CoupledPressureVelocitySolver.hh"
#include "solvers/SolverProgress.hh"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Transient solver for incompressible momentum and pressure.
 *
 * Derived fluid solvers can override the momentum hooks while reusing field
 * ownership, pressure-velocity coupling, time stepping, and solution output.
 *
 * @tparam Pack Tpetra type pack used for distributed solver storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class FluidSolver
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using face_flux_field_type = FaceField<Pack>;
    using face_flux_workspace_type =
        FVM::PressureWeightedFaceFluxWorkspace<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using vec_type = typename mesh_type::Vec3;
    using residual_type = PressureVelocityResiduals<scalar_type>;
    using step_statistics_type = FluidStepStatistics<scalar_type>;
    using coupled_system_type = CoupledPressureVelocitySystem<Pack>;

    FluidSolver(SP<const mesh_type> mesh,
                BoundaryConditionSet boundary_conditions,
                TimeStepperOptions time_options = {},
                LinearSolverOptions linear_options = {});

    FluidSolver(SP<const MeshHandle<Pack>> mesh,
                BoundaryConditionSet boundary_conditions,
                TimeStepperOptions time_options = {},
                LinearSolverOptions linear_options = {});

    virtual ~FluidSolver() = default;

    virtual void step();
    /** @brief Advance one step and print rank-zero convergence progress. */
    void step(ProgressStream& progress_output);
    void run(int steps);
    /** @brief Advance @p steps and print one rank-zero line per step. */
    void run(int steps, ProgressStream& progress_output);
    void run() { run(d_problem.time_options().steps); }
    void run(ProgressStream& progress_output)
    {
        run(d_problem.time_options().steps, progress_output);
    }

    scalar_type time() const noexcept { return d_time; }
    int step_index() const noexcept { return d_step_index; }

    /** @brief Physical gauge-pressure field in Pa. */
    const field_type& pressure() const noexcept;
    const velocity_field_type& velocity() const noexcept;
    /** @brief Mutable physical gauge-pressure field in Pa. */
    field_type& pressure() noexcept;
    velocity_field_type& velocity() noexcept;

    const residual_type& last_pressure_velocity_residuals() const noexcept;
    const step_statistics_type& last_step_statistics() const noexcept
    {
        return d_last_step_statistics;
    }

    void write_vtu(const std::string& filename) const
    {
        d_mesh->export_vtu(filename);
    }
    void write_solution_vtu(const std::string& filename) const;

protected:
    /** @brief Tag selecting deferred registration of a momentum equation. */
    struct DeferredMomentumEquationTag {};

    FluidSolver(SP<const MeshHandle<Pack>> mesh,
                BoundaryConditionSet boundary_conditions,
                TimeStepperOptions time_options,
                LinearSolverOptions linear_options,
                DeferredMomentumEquationTag);

    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh);
    static SP<const mesh_type> require_legacy_mesh(
        const SP<const MeshHandle<Pack>>& mesh);

    virtual IncompressibleMomentumEquation<Pack>& momentum_equation();
    virtual LinearSolveSummary advance_momentum();
    virtual coupled_system_type assemble_coupled_system();
    /** @return Pressure normalization density in kg/m^3. */
    virtual scalar_type pressure_reference_density() const noexcept
    {
        return scalar_type{1};
    }

    void begin_step();
    void finish_step();
    void solve_pressure_velocity_coupling();

    PressureProjectionEquation<Pack>& pressure_projection();
    CoupledPressureVelocitySolver<Pack>& coupled_pressure_velocity_solver();
    FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache();
    face_flux_workspace_type& pressure_face_flux_workspace();
    field_type& pressure_correction();
    velocity_field_type& predictor_pressure_gradient();
    velocity_field_type& predictor_velocity();
    face_flux_field_type& old_face_fluxes();
    face_flux_field_type& projected_face_fluxes();
    residual_type& pressure_velocity_residuals();
    const residual_type& pressure_velocity_residuals() const;

    /**
     * @brief Compute a global relative velocity update norm.
     * @return Communicator-wide relative L2 update norm.
     */
    scalar_type velocity_update_norm(
        const velocity_field_type& before,
        const velocity_field_type& after) const;
    VTUWriter fluid_solution_writer() const;
    /**
     * @return Owned scalar values in mesh order.
     */
    VTUWriter::ScalarData collect_scalar_field(
        const field_type& field) const;

    SP<const mesh_type> d_mesh;
    Problem<Pack> d_problem;
    scalar_type d_time = 0.0;
    int d_step_index = 0;
    step_statistics_type d_last_step_statistics;
    mutable VTUWriter::TopologyHandle d_vtu_topology;

private:
    FluidSolver(SP<const MeshHandle<Pack>> mesh,
                BoundaryConditionSet boundary_conditions,
                TimeStepperOptions time_options,
                LinearSolverOptions linear_options,
                bool register_momentum_equation);

    LinearSolveSummary run_momentum_predictor();
    /**
     * @return Communicator-wide sum.
     */
    scalar_type global_sum(scalar_type local_value) const;
    void write_step_progress(
        ProgressStream& progress_output,
        int total_steps) const;
    typename PressureProjectionEquation<Pack>::ProjectionResult
    run_pressure_correction(bool reuse_cached_predictor_flux);
    void solve_coupled_krylov();
};

} // namespace SimpleFluid
