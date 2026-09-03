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
#include "equations/BoilingSourceModel.hh"
#include "equations/BoussinesqMomentumEquation.hh"
#include "equations/DelayedNeutronPrecursorModel.hh"
#include "equations/FissionPowerSource.hh"
#include "equations/MaterialFeedbackModel.hh"
#include "equations/RadiolyticGasModel.hh"
#include "equations/ScalarVoidFractionModel.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "equations/turbulence/TurbulenceModel.hh"
#include "solvers/FluidSolver.hh"
#include "solvers/PlanarFreeSurfaceModel.hh"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
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
    using fission_power_source_type = FissionPowerSource<Pack, mesh_type>;
    using radiolytic_gas_model_type = RadiolyticGasModel<Pack, mesh_type>;
    using boiling_source_model_type = BoilingSourceModel<Pack, mesh_type>;
    using scalar_void_fraction_model_type = ScalarVoidFractionModel<Pack, mesh_type>;
    using material_feedback_model_type = MaterialFeedbackModel<Pack, mesh_type>;
    using precursor_model_type = DelayedNeutronPrecursorModel<Pack, mesh_type>;
    using free_surface_model_type = PlanarFreeSurfaceModel;
    using liquid_mass_inventory_type = LiquidMassInventory<Pack, mesh_type>;
    using free_surface_diagnostics_type = FreeSurfaceDiagnostics;
    using liquid_mass_diagnostics_type = typename liquid_mass_inventory_type::diagnostics_type;
    using boiling_diagnostics_type = typename boiling_source_model_type::diagnostics_type;
    struct FreeSurfaceHistoryRecord
    {
        free_surface_diagnostics_type free_surface;
        liquid_mass_diagnostics_type liquid_mass;
        scalar_type pool_occupancy_volume_error = {};
        scalar_type microbubble_hydrogen_moles = {};
        scalar_type large_bubble_hydrogen_moles = {};
        std::optional<boiling_diagnostics_type> boiling;
    };
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

    /**
     * @brief Construct with explicit physical models on a runtime handle.
     *
     * Physical material fields, heat sources, turbulence, and optional models
     * retain their native MeshHandle/FieldStored representation when @p mesh
     * does not wrap a legacy mesh.
     */
    BoussinesqSolver(SP<const MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
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
     */
    void initialize_bottom_hot_top_cold(
        scalar_type hot_temperature, scalar_type cold_temperature, scalar_type initial_pressure = 0.0);

    using base_type::step;
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

    /**
     * @brief Create the reserved qdot_fission source if needed.
     */
    fission_power_source_type& add_fission_power_source();
    /**
     * @brief Configure the reserved fission power-density source.
     */
    void configure_fission_power_source(const FissionPowerSourceOptions& options);
    /**
     * @brief Remove the reserved fission power source, if present.
     */
    bool remove_fission_power_source() noexcept;
    /**
     * @brief Return the mutable fission source, or nullptr when absent.
     */
    fission_power_source_type* find_fission_power_source() noexcept;
    /**
     * @brief Return the fission source, or nullptr when absent.
     */
    const fission_power_source_type* find_fission_power_source() const noexcept;

    /**
     * @brief Configure the optional radiolytic gas model from explicit options.
     */
    radiolytic_gas_model_type& configure_radiolytic_gas(const RadiolyticGasOptions& options);
    /**
     * @brief Configure the optional radiolytic gas model from database keys.
     */
    radiolytic_gas_model_type& configure_radiolytic_gas(const Database& database);
    /**
     * @brief Remove the optional radiolytic gas model, if present.
     */
    bool remove_radiolytic_gas_model() noexcept;
    /**
     * @brief Return the mutable radiolytic gas model, or nullptr when absent.
     */
    radiolytic_gas_model_type* find_radiolytic_gas_model() noexcept;
    /**
     * @brief Return the radiolytic gas model, or nullptr when absent.
     */
    const radiolytic_gas_model_type* find_radiolytic_gas_model() const noexcept;

    /**
     * @brief Configure the optional boiling source from explicit options.
     */
    boiling_source_model_type& configure_boiling_source(const BoilingSourceOptions& options);
    /**
     * @brief Configure the optional boiling source from database keys.
     */
    boiling_source_model_type& configure_boiling_source(const Database& database);
    /**
     * @brief Remove the optional boiling source model, if present.
     */
    bool remove_boiling_source_model() noexcept;
    /**
     * @brief Return the mutable boiling source model, or nullptr when absent.
     */
    boiling_source_model_type* find_boiling_source_model() noexcept;
    /**
     * @brief Return the boiling source model, or nullptr when absent.
     */
    const boiling_source_model_type* find_boiling_source_model() const noexcept;

    /**
     * @brief Configure the scalar void-fraction model from explicit options.
     */
    scalar_void_fraction_model_type& configure_scalar_void_fraction(const ScalarVoidFractionOptions& options);
    /**
     * @brief Configure the scalar void-fraction model from database keys.
     */
    scalar_void_fraction_model_type& configure_scalar_void_fraction(const Database& database);
    /**
     * @brief Return the mutable scalar void-fraction model, or nullptr.
     */
    scalar_void_fraction_model_type* find_scalar_void_fraction_model() noexcept;
    /**
     * @brief Return the scalar void-fraction model, or nullptr when absent.
     */
    const scalar_void_fraction_model_type* find_scalar_void_fraction_model() const noexcept;

    /**
     * @brief Configure material-property feedback from explicit options.
     */
    material_feedback_model_type& configure_material_feedback(const MaterialFeedbackOptions& options);
    /**
     * @brief Configure material-property feedback from database keys.
     */
    material_feedback_model_type& configure_material_feedback(const Database& database);
    /**
     * @brief Remove the optional material-feedback model, if present.
     */
    bool remove_material_feedback_model() noexcept;
    /**
     * @brief Return the mutable material-feedback model, or nullptr.
     */
    material_feedback_model_type* find_material_feedback_model() noexcept;
    /**
     * @brief Return the material-feedback model, or nullptr when absent.
     */
    const material_feedback_model_type* find_material_feedback_model() const noexcept;

    /**
     * @brief Configure delayed-neutron precursor groups from explicit options.
     * @note Invoke collectively on every mesh rank.
     */
    precursor_model_type& configure_precursors(const DelayedNeutronPrecursorOptions& options);
    /**
     * @brief Configure delayed-neutron precursor groups from database keys.
     * @note Invoke collectively on every mesh rank.
     */
    precursor_model_type& configure_precursors(const Database& database);
    /**
     * @brief Remove the optional precursor model, if present.
     * @note Invoke consistently on every mesh rank. A later step rejects a
     *       rank-divergent precursor state collectively.
     */
    bool remove_precursor_model() noexcept;
    /**
     * @brief Return the mutable precursor model, or nullptr when absent.
     */
    precursor_model_type* find_precursor_model() noexcept;
    /**
     * @brief Return the precursor model, or nullptr when absent.
     */
    const precursor_model_type* find_precursor_model() const noexcept;

    /**
     * @brief Configure the optional fixed-grid planar free-surface budget.
     *
     * Disabled options remove any existing model and return nullptr.  The
     * model is initialized after the primary fields, or lazily at the first
     * step when callers use the fields' default initial state.
     *
     * @note Invoke collectively on every mesh rank.
     */
    free_surface_model_type* configure_free_surface(const FreeSurfaceOptions& options);
    /** Configure the optional fixed-grid planar budget from flat keys. */
    free_surface_model_type* configure_free_surface(const Database& database);
    /**
     * Remove the free-surface model, inventory, and published fields.
     *
     * @note Invoke consistently on every mesh rank before the next collective
     *       solver step. Rank-divergent removal is rejected collectively.
     */
    bool remove_free_surface_model() noexcept;
    /** Return the mutable configured free-surface model, or nullptr. */
    free_surface_model_type* find_free_surface_model() noexcept;
    /** Return the configured free-surface model, or nullptr. */
    const free_surface_model_type* find_free_surface_model() const noexcept;

    /** Return the accepted free-surface diagnostics snapshot. */
    free_surface_diagnostics_type free_surface_diagnostics() const;
    /** Return the mutable liquid-mass inventory; throws when disabled. */
    liquid_mass_inventory_type& liquid_mass_inventory();
    /** Return the liquid-mass inventory; throws when disabled. */
    const liquid_mass_inventory_type& liquid_mass_inventory() const;
    /** Return the liquid-mass inventory, or nullptr when disabled. */
    liquid_mass_inventory_type* find_liquid_mass_inventory() noexcept;
    /** Return the liquid-mass inventory, or nullptr when disabled. */
    const liquid_mass_inventory_type* find_liquid_mass_inventory() const noexcept;

    /** Pure (bubble-free) liquid-density field [kg/m^3]. */
    const field_type& rho_liquid() const;
    /** Spatially constant diagnostic clear-level field [m]. */
    const field_type& clear_level() const;
    /** Spatially constant diagnostic pool-level field [m]. */
    const field_type& pool_level() const;
    /** Spatially constant absolute headspace-pressure field [Pa]. */
    const field_type& headspace_pressure() const;
    /**
     * Cell-centre planar pool indicator (0 or 1).
     *
     * This is a visualization/feedback approximation, not a conservative
     * cut-cell volume fraction.
     */
    const field_type& pool_occupancy() const;
    /** Global signed occupancy-volume error relative to pool volume [m^3]. */
    scalar_type pool_occupancy_volume_error() const;
    /** Accepted initialization and per-step global free-surface history. */
    const std::vector<FreeSurfaceHistoryRecord>& free_surface_history() const noexcept;
    /** Write the accepted history on mesh rank zero using a fixed CSV schema. */
    void write_free_surface_history_csv(const std::string& filename) const;

    void set_material_updater(typename material_type::updater_type updater);
    void clear_material_updater() noexcept;

    void write_solution_vtu(const std::string& filename) const;
    void write_solution_vtu(const std::string& filename, const SolutionOutputOptions& output_options) const;
    void write_parallel_solution_vtu(
        const std::string& filename, const SolutionOutputOptions& output_options = {}) const;

private:
    using base_type::begin_step;
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
    using base_type::native_velocity_boundary_cache;
    using base_type::old_face_fluxes;
    using base_type::predictor_pressure_gradient;
    using base_type::pressure_face_flux_workspace;
    using base_type::pressure_velocity_residuals;
    using base_type::projected_face_fluxes;
    using base_type::require_mesh;
    using base_type::solve_pressure_velocity_coupling;
    using base_type::sync_primary_fields_from_legacy;
    using base_type::sync_primary_fields_to_legacy;
    using base_type::uses_legacy_backend;
    using base_type::velocity_boundary_cache;

    /** @brief Tag selecting the physical-model constructor implementation. */
    struct PhysicalModelTag
    {
    };

    SIMPLEFLUID_SOLVERS_LOCAL
    BoussinesqSolver(SP<const MeshHandle<Pack>> mesh, BoundaryConditionSet boundary_conditions,
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
    VTUWriter solution_writer(const SolutionOutputOptions& output_options) const;
    SIMPLEFLUID_SOLVERS_LOCAL
    temperature_source_registry_type& stored_temperature_sources();
    SIMPLEFLUID_SOLVERS_LOCAL
    const temperature_source_registry_type& stored_temperature_sources() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void refresh_physical_models();
    SIMPLEFLUID_SOLVERS_LOCAL
    void refresh_material_feedback(scalar_type time);
    SIMPLEFLUID_SOLVERS_LOCAL
    void initialize_radiolytic_gas_state(bool force = false);
    SIMPLEFLUID_SOLVERS_LOCAL
    void update_void_fraction_models(scalar_type time_step);
    SIMPLEFLUID_SOLVERS_LOCAL
    const field_type* active_alpha_g_field() const noexcept;
    SIMPLEFLUID_SOLVERS_LOCAL
    const field_type* active_alpha_l_field() const noexcept;
    SIMPLEFLUID_SOLVERS_LOCAL
    void ensure_scalar_void_fraction_model();
    SIMPLEFLUID_SOLVERS_LOCAL
    void validate_step_coupling() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void advance_turbulence(scalar_type time_step);
    SIMPLEFLUID_SOLVERS_LOCAL
    bool advance_pre_temperature_models(scalar_type time_step);
    SIMPLEFLUID_SOLVERS_LOCAL
    void advance_temperature_transport(scalar_type time_step);
    SIMPLEFLUID_SOLVERS_LOCAL
    void advance_post_temperature_models(scalar_type time_step, bool sheng_after_temperature);
    SIMPLEFLUID_SOLVERS_LOCAL
    void validate_collective_model_state() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void validate_free_surface_configuration(const FreeSurfaceOptions& options) const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void initialize_free_surface_if_needed(
        bool allow_default_fields = false, bool dependencies_already_refreshed = false);
    SIMPLEFLUID_SOLVERS_LOCAL
    void advance_free_surface(scalar_type time_step);
    struct FreeSurfaceAccountingPreview
    {
        typename liquid_mass_inventory_type::diagnostics_type liquid;
        scalar_type liquid_volume_deficit = {};
    };
    SIMPLEFLUID_SOLVERS_LOCAL
    FreeSurfaceUpdate make_free_surface_update(
        scalar_type time, bool initializing, const FreeSurfaceAccountingPreview* preview = nullptr);
    SIMPLEFLUID_SOLVERS_LOCAL
    scalar_type pure_liquid_density(local_ordinal_type cell_lid) const;
    SIMPLEFLUID_SOLVERS_LOCAL
    scalar_type headspace_temperature(scalar_type time) const;
    SIMPLEFLUID_SOLVERS_LOCAL
    scalar_type scalar_void_volume() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void publish_free_surface_fields();
    SIMPLEFLUID_SOLVERS_LOCAL
    void record_free_surface_history();

    BoussinesqModelOptions d_model_options;
    bool d_physical_model_enabled = false;
    bool d_primary_fields_initialized = false;
    std::unique_ptr<fission_power_source_type> d_fission_power_source;
    std::unique_ptr<radiolytic_gas_model_type> d_radiolytic_gas_model;
    std::unique_ptr<boiling_source_model_type> d_boiling_source_model;
    std::unique_ptr<scalar_void_fraction_model_type> d_scalar_void_fraction_model;
    bool d_scalar_void_fraction_explicitly_configured = false;
    std::unique_ptr<material_feedback_model_type> d_material_feedback_model;
    std::unique_ptr<precursor_model_type> d_precursor_model;
    std::unique_ptr<free_surface_model_type> d_free_surface_model;
    std::unique_ptr<liquid_mass_inventory_type> d_liquid_mass_inventory;
    std::unique_ptr<field_type> d_clear_level;
    std::unique_ptr<field_type> d_pool_level;
    std::unique_ptr<field_type> d_headspace_pressure;
    std::unique_ptr<field_type> d_pool_occupancy;
    FreeSurfaceOptions d_free_surface_options;
    scalar_type d_pool_occupancy_volume_error = {};
    std::vector<FreeSurfaceHistoryRecord> d_free_surface_history;
    bool d_free_surface_step_failed = false;
};

extern template class BoussinesqSolver<DefaultTpetraTypes>;

} // namespace SimpleFluid
