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

#include "FVM/TransportSystem.hh"
#include "equations/BoussinesqMomentumEquation.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "equations/turbulence/TurbulenceModel.hh"
#include "solvers/FluidSolver.hh"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid
{
template<TpetraTypePack Pack> class MultiphysicsSolverCoupler;

/**
 * @brief Minimal transient Boussinesq natural-convection solver.
 *
 * @tparam Pack Tpetra type pack used for vector storage and communication.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class SIMPLEFLUID_SOLVERS_EXPORT BoussinesqSolver : public FluidSolver<Pack>
{
public:
    using base_type = FluidSolver<Pack>;
    using typename base_type::coupled_system_type;
    using typename base_type::face_flux_field_type;
    using typename base_type::field_type;
    using typename base_type::global_ordinal_type;
    using typename base_type::legacy_face_flux_field_type;
    using typename base_type::legacy_field_type;
    using typename base_type::legacy_mesh_type;
    using typename base_type::legacy_velocity_field_type;
    using typename base_type::local_ordinal_type;
    using typename base_type::mesh_type;
    using typename base_type::momentum_equation_type;
    using typename base_type::residual_type;
    using typename base_type::scalar_type;
    using typename base_type::step_statistics_type;
    using typename base_type::vec_type;
    using typename base_type::velocity_field_type;
    using canonical_velocity_boundary_cache_type = typename base_type::velocity_boundary_cache_type;
    using canonical_face_flux_workspace_type = typename base_type::native_face_flux_workspace_type;
    using canonical_coupled_solver_type = typename base_type::native_coupled_solver_type;
    using material_type = MaterialPropertyFields<Pack, mesh_type>;
    using temperature_source_registry_type = TemperatureSourceRegistry<Pack, mesh_type>;
    using volumetric_source_type = VolumetricScalarSource<Pack, mesh_type>;
    using update_context_type = BoussinesqUpdateContext<Pack, mesh_type>;
    using temperature_equation_type = TemperatureDiffusionEquation<Pack, mesh_type>;
    using boussinesq_momentum_equation_type = BoussinesqMomentumEquation<Pack, mesh_type>;
    using turbulence_model_type = TurbulenceModel<Pack, mesh_type>;
    using turbulence_buoyancy_context_type = TurbulenceBuoyancyContext<Pack, mesh_type>;
    using continuity_target_type = typename base_type::continuity_target_type;
    using continuity_residual_type = typename base_type::continuity_residual_type;

    // Retain the historical legacy cell alias for downstream code that uses
    // it for STK-only model helpers.
    using cell_type = typename legacy_mesh_type::CellType;
    using base_type::pressure;
    using base_type::velocity;

    BoussinesqSolver(SP<const legacy_mesh_type> mesh, BoundaryConditionSet boundary_conditions,
        TimeStepperOptions time_options = {}, LinearSolverOptions linear_options = {});

    BoussinesqSolver(SP<const legacy_mesh_type> mesh, BoundaryConditionSet boundary_conditions,
        TimeStepperOptions time_options, LinearSolverOptions linear_options, BoussinesqModelOptions model_options);

    BoussinesqSolver(SP<const MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
        TimeStepperOptions time_options = {}, LinearSolverOptions linear_options = {});

    /** @brief Retain a mutable native handle for controlled mesh motion. */
    BoussinesqSolver(SP<MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
        TimeStepperOptions time_options = {}, LinearSolverOptions linear_options = {});

    /**
     * @brief Construct with explicit physical models on a runtime handle.
     *
     * Physical material fields, heat sources, turbulence, and optional models
     * retain their native MeshHandle/FieldStored representation when @p mesh
     * does not wrap a legacy mesh.
     */
    BoussinesqSolver(SP<const MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
        TimeStepperOptions time_options, LinearSolverOptions linear_options, BoussinesqModelOptions model_options);

    /** @brief Construct physical Boussinesq transport on a mutable native handle. */
    BoussinesqSolver(SP<MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
        TimeStepperOptions time_options, LinearSolverOptions linear_options, BoussinesqModelOptions model_options);

    /**
     * @brief Initialize a linear temperature profile and uniform pressure.
     * @p initial_pressure is physical gauge pressure in Pa.
     */
    void initialize_linear_temperature(
        const vec_type& direction, scalar_type hot_at_min, scalar_type cold_at_max, scalar_type initial_pressure = 0.0);

    /**
     * @brief Initialize the standard side-heated-box state.
     * @p initial_pressure is physical gauge pressure in Pa.
     */
    void initialize_heated_box(
        scalar_type hot_temperature, scalar_type cold_temperature, scalar_type initial_pressure = 0.0);

    /**
     * @param initial_pressure Physical gauge pressure in Pa.
     * @param hot_temperature Hot-side temperature in kelvin.
     * @param cold_temperature Cold-side temperature in kelvin.
     */
    void initialize_bottom_hot_top_cold(
        scalar_type hot_temperature, scalar_type cold_temperature, scalar_type initial_pressure = 0.0);

    using base_type::step;
    /**
     * @brief Advance standalone flow, heat, and turbulence.
     * An active external timestep controller must perform coupled advancement.
     */
    void step() override;

    const field_type& temperature() const noexcept;
    field_type& temperature() noexcept;

    material_type& material_properties();
    const material_type& material_properties() const;

    /** Configure a Problem-owned two-equation turbulence model. */
    turbulence_model_type& configure_turbulence(const TurbulenceModelOptions& options);
    /** Configure turbulence from flat database keys. */
    turbulence_model_type& configure_turbulence(const Database& database);
    /**
     * Disable the active turbulence model and restore laminar transport.
     * @note Invoke consistently on every mesh rank.
     */
    bool remove_turbulence_model() noexcept;
    /** Return the active turbulence model, or nullptr in laminar mode. */
    turbulence_model_type* find_turbulence_model() noexcept;
    /** Return the active turbulence model, or nullptr in laminar mode. */
    const turbulence_model_type* find_turbulence_model() const noexcept;

    /**
     * @param name Must be unique and non-reserved.
     * @param initial_power_density Volumetric power density in W/m^3.
     */
    volumetric_source_type& add_temperature_source(std::string name, scalar_type initial_power_density = {});
    bool remove_temperature_source(const std::string& name);
    volumetric_source_type* find_temperature_source(const std::string& name) noexcept;
    const volumetric_source_type* find_temperature_source(const std::string& name) const noexcept;
    temperature_source_registry_type& temperature_sources();
    const temperature_source_registry_type& temperature_sources() const;

    void set_material_updater(typename material_type::updater_type updater);
    void clear_material_updater() noexcept;

    void write_solution_vtu(const std::string& filename) const;
    void write_solution_vtu(const std::string& filename, const SolutionOutputOptions& output_options) const;
    void write_parallel_solution_vtu(
        const std::string& filename, const SolutionOutputOptions& output_options = {}) const;

protected:
    /** Refresh Boussinesq momentum/temperature plus shared solver geometry. */
    void refresh_geometry_dependent_state() override;

private:
    friend class MultiphysicsSolverCoupler<Pack>;

    using base_type::begin_step;
    using base_type::clear_volume_continuity_target;
    using base_type::collect_scalar_field;
    using base_type::coupled_pressure_velocity_solver;
    using base_type::d_last_step_statistics;
    using base_type::d_legacy_mesh;
    using base_type::d_mesh;
    using base_type::d_problem;
    using base_type::d_step_index;
    using base_type::d_time;
    using base_type::finish_step;
    using base_type::fluid_solution_writer;
    using base_type::legacy_old_face_fluxes;
    using base_type::legacy_predictor_pressure_gradient;
    using base_type::legacy_pressure;
    using base_type::legacy_pressure_face_flux_workspace;
    using base_type::legacy_projected_face_fluxes;
    using base_type::legacy_velocity;
    using base_type::native_coupled_pressure_velocity_solver;
    using base_type::native_momentum_equation;
    using base_type::native_pressure_face_flux_workspace;
    using base_type::native_pressure_projection;
    using base_type::native_velocity_boundary_cache;
    using base_type::old_face_fluxes;
    using base_type::predictor_pressure_gradient;
    using base_type::pressure_face_flux_workspace;
    using base_type::pressure_velocity_residuals;
    using base_type::projected_face_fluxes;
    using base_type::refine_volume_continuity;
    using base_type::require_mesh;
    using base_type::set_volume_continuity_target;
    using base_type::solve_pressure_velocity_coupling;
    using base_type::sync_primary_fields_from_legacy;
    using base_type::sync_primary_fields_to_legacy;
    using base_type::uses_legacy_backend;
    using base_type::velocity_boundary_cache;
    using base_type::volume_continuity_target;

    /** @brief Tag selecting the physical-model constructor implementation. */
    struct PhysicalModelTag
    {
    };

    SIMPLEFLUID_SOLVERS_LOCAL
    BoussinesqSolver(SP<const MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
        TimeStepperOptions time_options, LinearSolverOptions linear_options, BoussinesqModelOptions model_options,
        bool physical_model_enabled, PhysicalModelTag);

    SIMPLEFLUID_SOLVERS_LOCAL
    BoussinesqSolver(SP<MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
        TimeStepperOptions time_options, LinearSolverOptions linear_options, BoussinesqModelOptions model_options,
        bool physical_model_enabled, PhysicalModelTag);

    SIMPLEFLUID_SOLVERS_LOCAL
    temperature_equation_type& temperature_equation();
    SIMPLEFLUID_SOLVERS_LOCAL
    boussinesq_momentum_equation_type& boussinesq_momentum_equation();
    SIMPLEFLUID_SOLVERS_LOCAL
    canonical_velocity_boundary_cache_type& boussinesq_velocity_boundary_cache();
    SIMPLEFLUID_SOLVERS_LOCAL
    canonical_face_flux_workspace_type& boussinesq_pressure_face_flux_workspace();
    SIMPLEFLUID_SOLVERS_LOCAL
    canonical_coupled_solver_type& boussinesq_coupled_pressure_velocity_solver();
    SIMPLEFLUID_SOLVERS_LOCAL
    legacy_field_type& legacy_temperature();
    SIMPLEFLUID_SOLVERS_LOCAL
    const legacy_field_type& legacy_temperature() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void sync_temperature_to_legacy();
    BoussinesqMomentumEquation<Pack>& momentum_equation() override;
    LinearSolveSummary advance_momentum() override;
    coupled_system_type assemble_coupled_system() override;
    bool supports_coupled_nonlinear() const noexcept override;
    std::unique_ptr<CoupledNonlinearProblem> make_coupled_nonlinear_problem() override;
    scalar_type pressure_reference_density() const noexcept override;
    SIMPLEFLUID_SOLVERS_LOCAL
    material_type& stored_material_properties();
    SIMPLEFLUID_SOLVERS_LOCAL
    const material_type& stored_material_properties() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    turbulence_model_type& stored_turbulence_model();
    SIMPLEFLUID_SOLVERS_LOCAL
    const turbulence_model_type& stored_turbulence_model() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    bool physical_transport_enabled() const noexcept;
    SIMPLEFLUID_SOLVERS_LOCAL
    VTUWriter solution_writer(const SolutionOutputOptions& output_options,
        const std::map<std::string, const field_type*>* extra_material_fields = nullptr) const;
    SIMPLEFLUID_SOLVERS_LOCAL
    temperature_source_registry_type& stored_temperature_sources();
    SIMPLEFLUID_SOLVERS_LOCAL
    const temperature_source_registry_type& stored_temperature_sources() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void advance_turbulence(scalar_type time_step);
    struct FlowTransportInputs
    {
        const FVM::ALEControlVolumeState* ale = nullptr;
        face_flux_field_type* relative_flux = nullptr;
    };
    struct TemperatureTransportInputs
    {
        const FVM::ALEControlVolumeState* ale = nullptr;
        const field_type* liquid_mass_inventory = nullptr;
        const field_type* old_heat_capacity = nullptr;
        const field_type* temperature_density = nullptr;
        const face_flux_field_type* flux = nullptr;
        const field_type* power_sink = nullptr;
        const field_type* power_release = nullptr;
    };
    SIMPLEFLUID_SOLVERS_LOCAL void advance_temperature_transport(
        scalar_type time_step, const TemperatureTransportInputs& inputs);
    SIMPLEFLUID_SOLVERS_LOCAL void solve_flow(const FlowTransportInputs& inputs);
    SIMPLEFLUID_SOLVERS_LOCAL const face_flux_field_type& momentum_transport_flux();
    SIMPLEFLUID_SOLVERS_LOCAL void refresh_physical_models();
    SIMPLEFLUID_SOLVERS_LOCAL void validate_step_state() const;

    BoussinesqModelOptions d_model_options;
    bool d_physical_model_enabled = false;
    bool d_primary_fields_initialized = false;
    bool d_step_failed = false;
    std::optional<FVM::TransportGeometryCache<MeshHandle<Pack>>> d_shared_transport_geometry;
    FlowTransportInputs d_flow_transport;
    // A generic lifetime lease prevents bypassing an external timestep controller.
    std::weak_ptr<const void> d_external_step_controller;
};

extern template class BoussinesqSolver<DefaultTpetraTypes>;

} // namespace SimpleFluid
