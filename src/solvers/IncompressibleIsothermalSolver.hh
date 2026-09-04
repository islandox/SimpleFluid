/**
 * @file IncompressibleIsothermalSolver.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Transient constant-density isothermal incompressible flow solver.
 * @version 0.1
 * @date 2026-08-20
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoussinesqModel.hh"
#include "equations/turbulence/TurbulenceModel.hh"
#include "solvers/FluidSolver.hh"

#include <string>

namespace SimpleFluid
{

/**
 * @brief Transient isothermal incompressible solver with optional RANS transport.
 *
 * Molecular dynamic viscosity is initialized as
 * `reference_density * TimeStepperOptions::kinematic_viscosity`. The solver
 * advances only pressure, velocity, and optional turbulence variables; it does
 * not allocate or solve a temperature equation.
 *
 * @tparam Pack Tpetra type pack used for distributed solver storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class SIMPLEFLUID_SOLVERS_EXPORT IncompressibleIsothermalSolver : public FluidSolver<Pack>
{
public:
    using base_type = FluidSolver<Pack>;
    using typename base_type::coupled_system_type;
    using typename base_type::face_flux_field_type;
    using typename base_type::field_type;
    using typename base_type::legacy_mesh_type;
    using typename base_type::local_ordinal_type;
    using typename base_type::mesh_type;
    using typename base_type::momentum_equation_type;
    using typename base_type::scalar_type;
    using typename base_type::vec_type;
    using typename base_type::velocity_field_type;
    using canonical_velocity_boundary_cache_type = typename base_type::velocity_boundary_cache_type;
    using canonical_face_flux_workspace_type = typename base_type::native_face_flux_workspace_type;
    using canonical_coupled_solver_type = typename base_type::native_coupled_solver_type;
    using material_type = MaterialPropertyFields<Pack, mesh_type>;
    using turbulence_model_type = TurbulenceModel<Pack, mesh_type>;

    IncompressibleIsothermalSolver(SP<const legacy_mesh_type> mesh, BoundaryConditionSet boundary_conditions,
        TimeStepperOptions time_options = {}, LinearSolverOptions linear_options = {},
        scalar_type reference_density = scalar_type{1});

    /** @brief Construct directly on a runtime mesh handle. */
    IncompressibleIsothermalSolver(SP<const mesh_type> mesh, BoundaryConditionSet boundary_conditions,
        TimeStepperOptions time_options = {}, LinearSolverOptions linear_options = {},
        scalar_type reference_density = scalar_type{1});

    using base_type::pressure;
    using base_type::step;
    using base_type::velocity;
    using base_type::write_solution_vtu;

    void step() override;

    /** @brief Constant density used to normalize pressure and momentum. */
    scalar_type reference_density() const noexcept { return d_reference_density; }

    material_type& material_properties();
    const material_type& material_properties() const;

    /** @brief Configure the Problem-owned turbulence model. */
    turbulence_model_type& configure_turbulence(const TurbulenceModelOptions& options);
    /** @brief Configure turbulence from flat database keys. */
    turbulence_model_type& configure_turbulence(const Database& database);
    /** @brief Disable active turbulence and return whether it was enabled. */
    bool remove_turbulence_model() noexcept;
    /** @brief Return the active model, or nullptr in laminar mode. */
    turbulence_model_type* find_turbulence_model() noexcept;
    /** @brief Return the active model, or nullptr in laminar mode. */
    const turbulence_model_type* find_turbulence_model() const noexcept;

    /** @brief Write core fields and requested isothermal model fields. */
    void write_solution_vtu(const std::string& filename, const SolutionOutputOptions& output_options) const;

    /** @brief Write rank pieces and a rank-zero PVTU index. */
    void write_parallel_solution_vtu(
        const std::string& filename, const SolutionOutputOptions& output_options = {}) const;

protected:
    // Retain FluidSolver's established derived-test/extension inspection seam.
    using base_type::d_problem;

private:
    using base_type::begin_step;
    using base_type::collect_scalar_field;
    using base_type::d_last_step_statistics;
    using base_type::d_mesh;
    using base_type::finish_step;
    using base_type::fluid_solution_writer;
    using base_type::native_coupled_pressure_velocity_solver;
    using base_type::native_momentum_equation;
    using base_type::native_pressure_face_flux_workspace;
    using base_type::native_velocity_boundary_cache;
    using base_type::old_face_fluxes;
    using base_type::predictor_pressure_gradient;
    using base_type::pressure_velocity_residuals;
    using base_type::projected_face_fluxes;
    using base_type::require_mesh;
    using base_type::solve_pressure_velocity_coupling;
    using base_type::uses_legacy_backend;

    SIMPLEFLUID_SOLVERS_LOCAL
    momentum_equation_type& isothermal_momentum_equation();
    SIMPLEFLUID_SOLVERS_LOCAL
    canonical_velocity_boundary_cache_type& isothermal_velocity_boundary_cache();
    SIMPLEFLUID_SOLVERS_LOCAL
    canonical_face_flux_workspace_type& isothermal_pressure_face_flux_workspace();
    SIMPLEFLUID_SOLVERS_LOCAL
    canonical_coupled_solver_type& isothermal_coupled_pressure_velocity_solver();
    SIMPLEFLUID_SOLVERS_LOCAL
    material_type& stored_material_properties();
    SIMPLEFLUID_SOLVERS_LOCAL
    const material_type& stored_material_properties() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    turbulence_model_type& stored_turbulence_model();
    SIMPLEFLUID_SOLVERS_LOCAL
    const turbulence_model_type& stored_turbulence_model() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    VTUWriter solution_writer(const SolutionOutputOptions& output_options) const;

    LinearSolveSummary advance_momentum() override;
    coupled_system_type assemble_coupled_system() override;
    scalar_type pressure_reference_density() const noexcept override { return d_reference_density; }

    scalar_type d_reference_density;
};

extern template class IncompressibleIsothermalSolver<DefaultTpetraTypes>;

} // namespace SimpleFluid
