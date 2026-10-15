/**
 * @file MultiphysicsSolverCoupler.hh
 * @brief Own optional physics and coordinate their transactions with a fluid solver.
 */
#pragma once

#include "FVM/ALEControlVolumeState.hh"
#include "equations/BoilingSourceModel.hh"
#include "equations/ScalarSourceDecayTransport.hh"
#include "equations/PrescribedHeatSource.hh"
#include "equations/MaterialFeedbackModel.hh"
#include "equations/ReactingGasModel.hh"
#include "equations/ScalarVoidFractionModel.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/PlanarALEBoundary.hh"
#include "solvers/PlanarFreeSurfaceModel.hh"
#include "solvers/ThermalStepParticipant.hh"
#include "solvers/VolumeContinuityModel.hh"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Coordinate flow, heat, reaction_source, and constrained planar mesh motion.
 *
 * The bound Boussinesq solver owns the mesh, equations, primary fields, clocks,
 * and linear solvers. This coupler owns optional models and their accepted
 * diagnostics, advances them in physical dependency order, and coordinates ALE
 * trials and interval checkpoint replay. Configuration and advancement follow
 * the same collective mesh-communicator contracts as the bound solver.
 *
 * Construct with shared ownership of an existing BoussinesqSolver and advance
 * through this controller. The coupler retains the solver. Only one external
 * timestep controller may bind a solver at a time. The solver owns no coupler.
 * @tparam Pack Tpetra storage and communication types.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes> class SIMPLEFLUID_SOLVERS_EXPORT MultiphysicsSolverCoupler
{
private:
    struct CouplingState;
    struct SIMPLEFLUID_SOLVERS_LOCAL PlanarALEStep;

public:
    using solver_type = BoussinesqSolver<Pack>;
    using solver_pointer = SP<solver_type>;
    using base_type = FluidSolver<Pack>;
    using face_flux_field_type = typename base_type::face_flux_field_type;
    using field_type = typename base_type::field_type;
    using local_ordinal_type = typename base_type::local_ordinal_type;
    using mesh_type = typename base_type::mesh_type;
    using residual_type = typename base_type::residual_type;
    using scalar_type = typename base_type::scalar_type;
    using step_statistics_type = typename base_type::step_statistics_type;
    using vec_type = typename base_type::vec_type;
    using velocity_field_type = typename base_type::velocity_field_type;
    using material_type = MaterialPropertyFields<Pack, mesh_type>;
    using temperature_source_registry_type = TemperatureSourceRegistry<Pack, mesh_type>;
    using volumetric_source_type = VolumetricScalarSource<Pack, mesh_type>;
    using update_context_type = BoussinesqUpdateContext<Pack, mesh_type>;
    using thermal_participant_type = ThermalStepParticipant<Pack>;
    using temperature_equation_type = TemperatureDiffusionEquation<Pack, mesh_type>;
    using prescribed_heat_source_type = PrescribedHeatSource<Pack, mesh_type>;
    using reacting_gas_model_type = ReactingGasModel<Pack, mesh_type>;
    using boiling_source_model_type = BoilingSourceModel<Pack, mesh_type>;
    using scalar_void_fraction_model_type = ScalarVoidFractionModel<Pack, mesh_type>;
    using material_feedback_model_type = MaterialFeedbackModel<Pack, mesh_type>;
    using scalar_group_model_type = ScalarSourceDecayTransport<Pack, mesh_type>;
    using free_surface_model_type = PlanarFreeSurfaceModel;
    using liquid_mass_inventory_type = LiquidMassInventory<Pack, mesh_type>;
    using free_surface_diagnostics_type = FreeSurfaceDiagnostics;
    using liquid_mass_diagnostics_type = typename liquid_mass_inventory_type::diagnostics_type;
    using boiling_diagnostics_type = typename boiling_source_model_type::diagnostics_type;
    using ale_motion_type = PlanarALEMeshMotion<Pack>;
    using ale_boundary_type = PlanarALEBoundary<Pack>;
    using volume_continuity_model_type = VolumeContinuityModel<Pack, mesh_type>;
    using continuity_target_type = typename base_type::continuity_target_type;
    using continuity_residual_type = typename base_type::continuity_residual_type;
    using volume_source_diagnostics_type = typename volume_continuity_model_type::diagnostics_type;

    /** Accepted diagnostics for the constrained flat-surface ALE path. */
    struct PlanarALEStepDiagnostics
    {
        bool initialized = false;
        std::uint64_t old_geometry_epoch = 0;
        std::uint64_t new_geometry_epoch = 0;
        scalar_type old_mesh_volume = {};      ///< [m^3]
        scalar_type new_mesh_volume = {};      ///< [m^3]
        scalar_type mesh_vessel_mismatch = {}; ///< Actual mesh minus closure pool [m^3]
        scalar_type maximum_gcl_residual = {}; ///< [m^3/s]
        scalar_type maximum_normalized_gcl_residual = {};
        MeshQualityMetrics mesh_quality;
        int outer_correctors = 0;
        scalar_type level_residual = {};          ///< [m]
        scalar_type pressure_residual = {};       ///< [Pa]
        scalar_type material_state_residual = {}; ///< normalized Picard change
        scalar_type gas_state_residual = {};      ///< normalized Picard change
        scalar_type thermal_state_residual = {};  ///< normalized thermal participant change
        scalar_type outward_boundary_energy = {}; ///< prescribed heat leaving liquid [J]
        scalar_type energy_residual = {};         ///< [J]
        scalar_type normalized_energy_residual = {};
        scalar_type liquid_mass_residual = {};   ///< [kg]
        scalar_type gas_inventory_residual = {}; ///< [mol]
        volume_source_diagnostics_type volume_source;
        continuity_residual_type continuity;
        std::vector<scalar_type> level_residual_history;     ///< [m]
        std::vector<scalar_type> target_change_history;      ///< [m^3/s]
        std::vector<scalar_type> continuity_maximum_history; ///< [m^3/s]
        std::vector<scalar_type> material_state_residual_history;
        std::vector<scalar_type> gas_state_residual_history;
        std::size_t rejected_transactions = 0;
        std::string last_rejection_reason;
    };
    struct FreeSurfaceHistoryRecord
    {
        free_surface_diagnostics_type free_surface;
        liquid_mass_diagnostics_type liquid_mass;
        scalar_type pool_occupancy_volume_error = {};
        scalar_type microbubble_hydrogen_moles = {};
        scalar_type large_bubble_hydrogen_moles = {};
        std::optional<boiling_diagnostics_type> boiling;
        std::optional<PlanarALEStepDiagnostics> planar_ale;
    };
    /**
     * Retain an existing solver and bind its timestep-controller lease collectively.
     * Every participating rank must supply a non-null solver. Null ownership is
     * rejected before mesh communication can begin.
     */
    explicit MultiphysicsSolverCoupler(solver_pointer solver);
    solver_type& boussinesq_solver() noexcept { return *d_solver; }
    const solver_type& boussinesq_solver() const noexcept { return *d_solver; }

    /** Initialize primary fields and dependent gas/free-surface inventories. */
    void initialize_linear_temperature(
        const vec_type& direction, scalar_type hot_at_min, scalar_type cold_at_max, scalar_type initial_pressure = {});
    void initialize_heated_box(
        scalar_type hot_temperature, scalar_type cold_temperature, scalar_type initial_pressure = {});
    void initialize_bottom_hot_top_cold(
        scalar_type hot_temperature, scalar_type cold_temperature, scalar_type initial_pressure = {});
    void set_material_updater(typename material_type::updater_type updater);
    void clear_material_updater() noexcept { d_solver->clear_material_updater(); }
    /** Advance the configured coupled schedule, optionally printing accepted progress. */
    void step(ProgressStream& progress);
    void run(int steps);
    void run(int steps, ProgressStream& progress);
    void run();
    void run(ProgressStream& progress);
    /** Maximum Courant number using the accepted carrier-relative ALE flux. */
    scalar_type maximum_courant_number() const;
    void write_solution_vtu(const std::string& filename) const;
    void write_solution_vtu(const std::string& filename, const SolutionOutputOptions& options) const;
    void write_parallel_solution_vtu(const std::string& filename, const SolutionOutputOptions& options = {}) const;

    MultiphysicsSolverCoupler(const MultiphysicsSolverCoupler&) = delete;
    MultiphysicsSolverCoupler& operator=(const MultiphysicsSolverCoupler&) = delete;
    MultiphysicsSolverCoupler(MultiphysicsSolverCoupler&&) = delete;
    MultiphysicsSolverCoupler& operator=(MultiphysicsSolverCoupler&&) = delete;

    /** Install a typed external thermal transaction for mapped physical planar ALE.
     * Base thermal patches remain adiabatic; participant fluxes override selected
     * faces on each candidate. Installation and removal are collective and idle.
     */
    void set_thermal_step_participant(SP<thermal_participant_type> participant);
    SP<thermal_participant_type> thermal_step_participant() const noexcept { return d_thermal_participant; }
    /**
     * @brief Advance buoyant flow, temperature, and configured physical models.
     *
     * The fixed-grid path below shows one timestep. Planar ALE delegates to
     * `step_planar_ale()`, whose geometry-trial correctors and rollback contract
     * are separate from this fixed-grid ordering. A configured
     * PrescribedHeatSource supplies a local power-density field or interval
     * energy budget; the coupler does not execute an external neutronics code.
     *
     * @par Algorithm
     * @if SIMPLEFLUID_DIAGRAMS
     * @startuml
     * start
     * :Validate coupling mode and collective model state;
     * if (Interval prescribed_heat energy supplied?) then (yes)
     *   :Refresh local power density for current interval;
     * endif
     * if (FreeSurfaceMode::PlanarALE?) then (yes)
     *   :begin_step();
     *   :step_planar_ale()\n(geometry trials, coupled acceptance, rollback);
     *   stop
     * endif
     * if (active SST-SAS?) then (yes)
     *   :Snapshot accepted multiphysics state;
     *   note right
     *     The enclosing catch restores these snapshots
     *     before propagating an exception.
     *   end note
     * endif
     * if (CoupledNonlinear?) then (yes)
     *   :Preserve accepted reports until private flow solve commits;
     * else (no)
     *   :begin_step();
     * endif
     * if (first accepted step?) then (yes)
     *   :Synchronize temperature ghosts;
     * endif
     * if (physical model enabled?) then (yes)
     *   :refresh_physical_models();
     *   note right
     *     Material updater, gas initialization,
     *     material feedback, and registered
     *     temperature-source refresh use accepted fields.
     *   end note
     * endif
     * :Initialize free-surface state if configured;
     * if (turbulence enabled?) then (yes)
     *   :Refresh effective viscosity/conductivity;
     * endif
     * :solve_pressure_velocity_coupling()\nwith temperature buoyancy;
     * :advance_turbulence(time_step);
     * :advance_pre_temperature_models(time_step);
     * :advance_temperature_transport(time_step);
     * :advance_post_temperature_models(time_step);
     * if (legacy backend?) then (yes)
     *   :Publish temperature to legacy storage;
     * endif
     * :finish_step();
     * if (free surface configured?) then (yes)
     *   :Record accepted free-surface history;
     * endif
     * stop
     * @enduml
     * @endif
     *
     * @see FluidSolver::solve_pressure_velocity_coupling()
     * @see advance_pre_temperature_models()
     */
    void step();

    /** Move-only, reusable interval checkpoint. No destructor performs MPI work. */
    class CouplingCheckpoint
    {
    public:
        CouplingCheckpoint(CouplingCheckpoint&&) noexcept = default;
        CouplingCheckpoint& operator=(CouplingCheckpoint&&) noexcept = default;
        CouplingCheckpoint(const CouplingCheckpoint&) = delete;
        CouplingCheckpoint& operator=(const CouplingCheckpoint&) = delete;

    private:
        friend class MultiphysicsSolverCoupler;
        explicit CouplingCheckpoint(std::shared_ptr<const CouplingState> state) : d_state(std::move(state)) {}
        std::shared_ptr<const CouplingState> d_state;
    };
    /** Collective ALE interval transaction. Configuration/mesh topology must stay fixed.
     * Restore permits repeated trials after multiple accepted physical substeps;
     * accept keeps the current fields and invalidates the checkpoint. Only one
     * checkpoint may remain live. Destroying one does not restore physical state.
     */
    [[nodiscard]] CouplingCheckpoint create_coupling_checkpoint();
    void restore_coupling_checkpoint(const CouplingCheckpoint& checkpoint);
    /** Collective read-only acceptance preflight for an external coupled commit.
     * Performs the same checks as accept_coupling_checkpoint without releasing
     * the checkpoint or changing fields, geometry, clocks, or ledgers. Call
     * before committing another solver, then accept without intervening SF
     * state/configuration changes. This check does not reserve the candidate.
     */
    void validate_coupling_checkpoint_acceptance(const CouplingCheckpoint& checkpoint) const;
    void accept_coupling_checkpoint(CouplingCheckpoint& checkpoint);
    /** Assign a material-following extensive prescribed_heat budget [J/owned cell].
     * Uniform in time over [time(), time()+duration]; remapped by current cell
     * volume on every ALE trial. Configure a prescribed_heat source before checkpointing.
     */
    void set_coupling_interval_energy(
        std::span<const scalar_type> owned_cell_energy_joules, scalar_type interval_duration);

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
     * @brief Create the reserved qdot_external source if needed.
     */
    prescribed_heat_source_type& add_prescribed_heat_source();
    /**
     * @brief Configure the reserved prescribed_heat power-density source.
     */
    void configure_prescribed_heat_source(const PrescribedHeatSourceOptions& options);
    /**
     * @brief Remove the reserved prescribed_heat power source, if present.
     */
    bool remove_prescribed_heat_source() noexcept;
    /**
     * @brief Return the mutable prescribed_heat source, or nullptr when absent.
     */
    prescribed_heat_source_type* find_prescribed_heat_source() noexcept;
    /**
     * @brief Return the prescribed_heat source, or nullptr when absent.
     */
    const prescribed_heat_source_type* find_prescribed_heat_source() const noexcept;

    /**
     * @brief Configure the optional reacting gas model from explicit options.
     */
    reacting_gas_model_type& configure_reacting_gas(const ReactingGasOptions& options);
    /**
     * @brief Configure the optional reacting gas model from database keys.
     */
    reacting_gas_model_type& configure_reacting_gas(const Database& database);
    /**
     * @brief Remove the optional reacting gas model, if present.
     */
    bool remove_reacting_gas_model();
    /**
     * @brief Return the mutable reacting gas model, or nullptr when absent.
     */
    reacting_gas_model_type* find_reacting_gas_model() noexcept;
    /**
     * @brief Return the reacting gas model, or nullptr when absent.
     */
    const reacting_gas_model_type* find_reacting_gas_model() const noexcept;

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
    bool remove_boiling_source_model();
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
    bool remove_material_feedback_model();
    /**
     * @brief Return the mutable material-feedback model, or nullptr.
     */
    material_feedback_model_type* find_material_feedback_model() noexcept;
    /**
     * @brief Return the material-feedback model, or nullptr when absent.
     */
    const material_feedback_model_type* find_material_feedback_model() const noexcept;

    /**
     * @brief Configure delayed-neutron scalar_group groups from explicit options.
     * @note Invoke collectively on every mesh rank.
     */
    scalar_group_model_type& configure_scalar_groups(const ScalarSourceDecayOptions& options);
    /**
     * @brief Configure delayed-neutron scalar_group groups from database keys.
     * @note Invoke collectively on every mesh rank.
     */
    scalar_group_model_type& configure_scalar_groups(const Database& database);
    /**
     * @brief Remove the optional scalar_group model, if present.
     * @note Invoke consistently on every mesh rank. A later step rejects a
     *       rank-divergent scalar_group state collectively.
     */
    bool remove_scalar_group_model() noexcept;
    /**
     * @brief Return the mutable scalar_group model, or nullptr when absent.
     */
    scalar_group_model_type* find_scalar_group_model() noexcept;
    /**
     * @brief Return the scalar_group model, or nullptr when absent.
     */
    const scalar_group_model_type* find_scalar_group_model() const noexcept;

    /**
     * @brief Configure optional fixed-grid or solver-integrated planar free-surface coupling.
     *
     * Disabled options remove any existing model and return nullptr.  The
     * model is initialized after the primary fields, or lazily at the first
     * step when callers use the fields' default initial state.
     *
     * @note Invoke collectively on every mesh rank.
     */
    free_surface_model_type* configure_free_surface(const FreeSurfaceOptions& options);
    /** Configure optional planar free-surface coupling from flat keys. */
    free_surface_model_type* configure_free_surface(const Database& database);
    /**
     * Remove the free-surface model, inventory, and published fields.
     *
     * @note Invoke consistently on every mesh rank before the next collective
     *       solver step. Rank-divergent removal is rejected collectively.
     */
    bool remove_free_surface_model();
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
    /** Whether the fully transactional planar-ALE path is configured. */
    bool planar_ale_enabled() const noexcept
    {
        return d_free_surface_model && d_free_surface_options.mode == FreeSurfaceMode::PlanarALE;
    }
    /** Honest free-surface capability label for diagnostics and applications. */
    std::string_view free_surface_approximation_label() const noexcept
    {
        return planar_ale_enabled() ? "solver-integrated flat planar ALE (not VOF or Euler-Euler)"
                                    : free_surface_model_type::approximationLabel();
    }
    /** Diagnostics from the latest accepted ALE step (plus rejection count). */
    const PlanarALEStepDiagnostics& planar_ale_diagnostics() const noexcept { return d_planar_ale_diagnostics; }
    /** Accepted absolute-minus-mesh carrier flux; throws when ALE is inactive. */
    const face_flux_field_type& mesh_relative_face_fluxes() const;
    /** Accepted initialization and per-step global free-surface history. */
    const std::vector<FreeSurfaceHistoryRecord>& free_surface_history() const noexcept;
    /** Write the accepted history on mesh rank zero using a fixed CSV schema. */
    void write_free_surface_history_csv(const std::string& filename) const;

private:
    SIMPLEFLUID_SOLVERS_LOCAL void refresh_geometry_dependent_state();
    SIMPLEFLUID_SOLVERS_LOCAL void advance_temperature_transport(scalar_type time_step);
    SIMPLEFLUID_SOLVERS_LOCAL VTUWriter solution_writer(const SolutionOutputOptions& options) const;
    SIMPLEFLUID_SOLVERS_LOCAL void initialize_primary_state();
    SIMPLEFLUID_SOLVERS_LOCAL void refresh_geometry_dependent_models();
    SIMPLEFLUID_SOLVERS_LOCAL void append_solution_fields(
        VTUWriter& writer, const SolutionOutputOptions& options) const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void refresh_physical_models();
    SIMPLEFLUID_SOLVERS_LOCAL
    void refresh_material_feedback(scalar_type time);
    SIMPLEFLUID_SOLVERS_LOCAL
    void initialize_reacting_gas_state(bool force = false);
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
    /**
     * @brief Order optional model updates around temperature transport.
     *
     * This diagram combines this helper, the bound temperature transport, and
     * advance_post_temperature_models(). Ideal reaction_source and boiling consume
     * the temperature accepted at the start of the step. Sheng reaction_source is
     * intentionally delayed until after temperature transport because it owns
     * the authoritative gas void fraction. All branches are sequential.
     *
     * @par Algorithm
     * @if SIMPLEFLUID_DIAGRAMS
     * @startuml
     * start
     * if (Reacting gas enabled and\nnot Sheng void owner?) then (yes)
     *   :ReactingGasModel::advance()\nwith lagged temperature;
     * endif
     * if (Sheng void owner?) then (yes)
     *   :Defer gas/void update;
     * else (no)
     *   if (BoilingSourceModel configured?) then (yes)
     *     :update() boiling source\nfrom lagged temperature;
     *   endif
     *   if (ScalarVoidFractionModel configured?) then (yes)
     *     :Mirror reacting alpha_g or\nupdate_explicit(gas + boiling sources);
     *     :Complete boiling void/inventory update;
     *   endif
     * endif
     * :Solve temperature transport with projected flux,\nregistered sources, and optional boiling sink;
     * note right
     *   Uses effective thermal conductivity when turbulent;
     *   otherwise material conductivity or legacy diffusivity.
     * end note
     * if (Sheng void owner?) then (yes)
     *   :ReactingGasModel::advance()\nwith updated temperature;
     *   :Mirror authoritative alpha_g into scalar void model;
     * endif
     * if (ScalarSourceDecayTransport configured?) then (yes)
     *   :Advance scalar_groups with liquid fraction,\nprojected flux, and optional prescribed_heat field;
     * endif
     * if (MaterialFeedbackModel configured?) then (yes)
     *   :Apply feedback at time + time_step\nusing updated temperature/void state;
     * endif
     * if (free surface configured?) then (yes)
     *   :Advance liquid inventory and free-surface state;
     * endif
     * if (turbulence enabled?) then (yes)
     *   :Refresh effective properties for next step;
     * endif
     * stop
     * @enduml
     * @endif
     */
    bool advance_pre_temperature_models(scalar_type time_step);
    SIMPLEFLUID_SOLVERS_LOCAL
    void advance_post_temperature_models(scalar_type time_step, bool sheng_after_temperature);
    SIMPLEFLUID_SOLVERS_LOCAL
    void validate_collective_model_state() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void validate_free_surface_configuration(const FreeSurfaceOptions& options) const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void validate_planar_ale_support(const FreeSurfaceOptions& options) const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void validate_planar_ale_runtime_parity() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void initialize_planar_ale_if_needed();
    SIMPLEFLUID_SOLVERS_LOCAL
    void step_planar_ale();
    SIMPLEFLUID_SOLVERS_LOCAL
    void form_mesh_relative_flux(face_flux_field_type& absolute_flux);
    SIMPLEFLUID_SOLVERS_LOCAL
    std::vector<scalar_type> material_volumes(const field_type& liquid_mass_density,
        const field_type& pure_liquid_density, std::span<const real_t> cell_volumes) const;
    SIMPLEFLUID_SOLVERS_LOCAL
    scalar_type volume_weighted_mean_pressure() const;
    SIMPLEFLUID_SOLVERS_LOCAL
    scalar_type total_sensible_energy(
        const field_type& density, const field_type& heat_capacity, std::span<const real_t> cell_volumes) const;
    SIMPLEFLUID_SOLVERS_LOCAL
    void configure_ale_pressure_boundary(std::uint64_t generation);
    SIMPLEFLUID_SOLVERS_LOCAL
    void clear_ale_pressure_boundary() noexcept;
    SIMPLEFLUID_SOLVERS_LOCAL
    void initialize_free_surface_if_needed(bool allow_default_fields = false,
        bool dependencies_already_refreshed = false, bool preserve_on_failure = false);
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

    SIMPLEFLUID_SOLVERS_LOCAL std::string coupling_configuration_signature() const;
    SIMPLEFLUID_SOLVERS_LOCAL void validate_coupling_checkpoint(const CouplingCheckpoint& checkpoint) const;
    solver_pointer d_solver;
    std::shared_ptr<const void> d_controller_lease;
    SP<thermal_participant_type> d_thermal_participant;
    std::weak_ptr<const CouplingState> d_active_coupling_checkpoint;
    std::unique_ptr<prescribed_heat_source_type> d_prescribed_heat_source;
    std::unique_ptr<reacting_gas_model_type> d_reacting_gas_model;
    std::unique_ptr<boiling_source_model_type> d_boiling_source_model;
    std::unique_ptr<scalar_void_fraction_model_type> d_scalar_void_fraction_model;
    bool d_scalar_void_fraction_explicitly_configured = false;
    std::unique_ptr<material_feedback_model_type> d_material_feedback_model;
    std::unique_ptr<scalar_group_model_type> d_scalar_group_model;
    std::unique_ptr<free_surface_model_type> d_free_surface_model;
    std::unique_ptr<liquid_mass_inventory_type> d_liquid_mass_inventory;
    std::unique_ptr<field_type> d_clear_level;
    std::unique_ptr<field_type> d_pool_level;
    std::unique_ptr<field_type> d_headspace_pressure;
    std::unique_ptr<field_type> d_pool_occupancy;
    std::unique_ptr<ale_motion_type> d_ale_motion;
    std::unique_ptr<ale_boundary_type> d_ale_boundary;
    std::unique_ptr<volume_continuity_model_type> d_volume_continuity_model;
    std::unique_ptr<face_flux_field_type> d_mesh_relative_face_flux;
    std::unique_ptr<face_flux_field_type> d_bubble_slip_volume_flux;
    std::unique_ptr<field_type> d_mesh_volume_rate;
    std::unique_ptr<field_type> d_continuity_residual;
    std::unique_ptr<field_type> d_ale_old_density;
    std::unique_ptr<field_type> d_ale_old_heat_capacity;
    std::optional<FVM::ALEControlVolumeState> d_active_ale;
    const field_type* d_ale_temperature_density = nullptr;
    std::uint64_t d_ale_target_generation = 0;
    PlanarALEStepDiagnostics d_planar_ale_diagnostics;
    FreeSurfaceOptions d_free_surface_options;
    scalar_type d_pool_occupancy_volume_error = {};
    std::vector<FreeSurfaceHistoryRecord> d_free_surface_history;
    bool d_free_surface_step_failed = false;
};

extern template class MultiphysicsSolverCoupler<DefaultTpetraTypes>;

} // namespace SimpleFluid
