/**
 * @file RadiolyticGasModel.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Runtime-selectable ideal-gas and two-population radiolysis model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoussinesqModel.hh"
#include "equations/CollectiveValidation.hh"
#include "equations/RadiolyticGasProperties.hh"
#include "fields/FaceField.hh"
#include "fields/MeshFieldTraits.hh"
#include "FVM/TransportSystem.hh"
#include "solvers/BelosLinearSolver.hh"

#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Per-step global diagnostics from a radiolytic gas update.
 *
 * @tparam Scalar Floating-point scalar type used by the solver fields.
 */
template<class Scalar>
struct RadiolyticGasStepStatistics
{
    Scalar hydrogen_before = {};
    Scalar hydrogen_produced = {};
    Scalar dissolved_hydrogen_outflow = {};
    Scalar microbubble_hydrogen_escaped = {};
    Scalar large_bubble_hydrogen_escaped = {};
    Scalar submerged_bubble_hydrogen_escaped = {};
    Scalar hydrogen_escaped = {};
    Scalar hydrogen_after = {};
    Scalar inventory_error = {}; ///< Hydrogen conservation residual.
    Scalar escaped_microbubble_count = {};
    Scalar escaped_large_bubble_count = {};
    Scalar escaped_bubble_count = {};
    Scalar cumulative_hydrogen_produced = {};
    Scalar cumulative_dissolved_hydrogen_outflow = {};
    Scalar cumulative_submerged_bubble_hydrogen_escaped = {};
    Scalar cumulative_hydrogen_escaped = {};
    Scalar cumulative_escaped_bubble_count = {};
    Scalar void_volume = {};
    int maximum_subcycles = 0;
    int clipped_cells = 0;
    int pressure_floor_cells = 0;
    int radius_solver_failures = 0;
};

/**
 * @brief Runtime radiolytic gas model with ideal and two-population modes.
 *
 * @tparam Pack Tpetra type pack used for mesh, field, and communicator types.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes,
         class MeshType = Mesh<Pack>>
class RadiolyticGasModel
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using velocity_field_type = typename field_traits::vector_cell_type;
    using face_flux_field_type = typename field_traits::scalar_face_type;
    using material_type = MaterialPropertyFields<Pack, mesh_type>;
    using statistics_type = RadiolyticGasStepStatistics<scalar_type>;

    /** Opaque complete model snapshot for an ALE outer-step transaction. */
    class StateSnapshot
    {
    private:
        friend class RadiolyticGasModel;
        const RadiolyticGasModel* d_owner = nullptr;
        std::vector<std::vector<scalar_type>> d_fields;
        std::vector<local_ordinal_type> d_transport_slip_face_ids;
        std::vector<scalar_type> d_transport_slip_face_values;
        std::vector<scalar_type> d_transport_carrier_face_values;
        statistics_type d_statistics;
        bool d_history_initialized = false;
        bool d_initial_state_initialized = false;
        scalar_type d_absolute_pressure_offset = {};
        scalar_type d_cumulative_hydrogen_produced = {};
        scalar_type d_cumulative_dissolved_hydrogen_outflow = {};
        scalar_type d_cumulative_submerged_bubble_hydrogen_escaped = {};
        scalar_type d_cumulative_hydrogen_escaped = {};
        scalar_type d_cumulative_escaped_bubble_count = {};
    };

    /**
     * @brief Construct a radiolysis model on a mesh.
     */
    explicit RadiolyticGasModel(
        SP<const mesh_type> mesh,
        RadiolyticGasOptions options = {});

    /**
     * @brief Replace model options and reinitialize dependent fields.
     */
    void configure(const RadiolyticGasOptions& options);
    /**
     * @brief Return the active radiolysis options.
     */
    const RadiolyticGasOptions& options() const noexcept
    {
        return d_options;
    }
    /**
     * @brief Return the selected radiolysis model family.
     */
    RadiolyticGasMode mode() const noexcept { return d_options.mode; }
    /**
     * @brief True when the model is configured to produce radiolysis fields.
     */
    bool enabled() const noexcept
    {
        return mode() != RadiolyticGasMode::Disabled;
    }
    /**
     * @brief True when the model owns alpha_g/alpha_l state directly.
     */
    bool supplies_void_fraction() const noexcept
    {
        return mode() == RadiolyticGasMode::Sheng2024TwoPopulation;
    }

    /**
     * @brief Reconstruct and publish the configured initial Sheng state.
     *
     * Bubble radii and void require the initial thermodynamic fields and
     * therefore cannot be completed by the mesh-only constructor.  This call
     * preserves the configured dissolved concentration by initializing its
     * conserved inventory as `alpha_l * C_H2`, and seeds the history fields so
     * the initial reconstructed void is not reported as a timestep source.
     * Repeated calls are no-ops unless @p force is true; a forced call resets
     * every radiolytic field to the configured initial state before rebuilding
     * its derived fields.
     */
    void initialize_state(
        scalar_type time,
        const field_type& temperature,
        const field_type& gauge_pressure,
        const velocity_field_type& velocity,
        const material_type& material,
        bool force = false);

    /**
     * @brief True after the configured initial Sheng state was reconstructed.
     */
    bool initial_state_initialized() const noexcept
    {
        return d_initial_state_initialized;
    }

    /**
     * @brief Advance disabled or two-population radiolysis state.
     *
     * Reconstructed pressure mode uses the supplied physical gauge-pressure
     * variation in Pa.
     *
     * Ideal-gas mode requires the overload carrying authoritative scalar void.
     */
    void advance(
        scalar_type time,
        scalar_type time_step,
        const field_type& temperature,
        const field_type& gauge_pressure,
        const velocity_field_type& velocity,
        const face_flux_field_type& liquid_face_flux,
        const material_type& material,
        const field_type* fission_power_density,
        const FVM::ALEControlVolumeState* ale = nullptr,
        Dimension slip_axis = Dimension::Z);

    /**
     * @brief Advance using the solver's authoritative scalar void state.
     *
     * Ideal-gas source limiting uses @p alpha_g and @p alpha_max.  The
     * two-population model continues to reconstruct void from its conserved
     * bubble inventories.
     */
    void advance(
        scalar_type time,
        scalar_type time_step,
        const field_type& temperature,
        const field_type& gauge_pressure,
        const velocity_field_type& velocity,
        const face_flux_field_type& liquid_face_flux,
        const material_type& material,
        const field_type* fission_power_density,
        const field_type& alpha_g,
        scalar_type alpha_max,
        const FVM::ALEControlVolumeState* ale = nullptr,
        Dimension slip_axis = Dimension::Z);

    /**
     * @brief Synchronize ideal-mode diagnostics with canonical scalar void.
     *
     * The solver calls this after applying the aggregate scalar-void update so
     * the legacy radiolysis alpha fields remain current without owning the
     * evolving ideal-mode state.
     */
    void synchronize_void_fraction(
        const field_type& alpha_g,
        scalar_type alpha_max);

    /**
     * @brief Model-local gas void field reconstructed or mirrored on advance.
     */
    const field_type& alpha_g() const noexcept { return d_alpha_g; }
    /**
     * @brief Liquid fraction field.
     */
    const field_type& alpha_l() const noexcept { return d_alpha_l; }
    /**
     * @brief Ideal alpha source or advanced net reconstructed-void rate.
     */
    const field_type& source_alpha_rad() const noexcept
    {
        return d_source_alpha_rad;
    }
    /**
     * @brief Absolute pressure field used by radiolysis correlations.
     */
    const field_type& absolute_pressure() const noexcept
    {
        return d_absolute_pressure;
    }
    /**
     * @brief Dissolved hydrogen concentration field.
     */
    const field_type& dissolved_hydrogen() const noexcept
    {
        return d_dissolved_hydrogen;
    }
    /**
     * @brief Dissolved hydrogen inventory field.
     */
    const field_type& dissolved_hydrogen_inventory() const noexcept
    {
        return d_dissolved_hydrogen_inventory;
    }
    /**
     * @brief Microbubble number-density field.
     */
    const field_type& micro_number_density() const noexcept
    {
        return d_micro_number;
    }
    /**
     * @brief Microbubble gas inventory field.
     */
    const field_type& micro_moles() const noexcept
    {
        return d_micro_moles;
    }
    /**
     * @brief Large-bubble number-density field.
     */
    const field_type& large_number_density() const noexcept
    {
        return d_large_number;
    }
    /**
     * @brief Large-bubble gas inventory field.
     */
    const field_type& large_moles() const noexcept
    {
        return d_large_moles;
    }

    /** EOS-derived unbounded gas-volume fraction used by conservative closure. */
    const field_type& raw_bubble_volume_fraction() const noexcept { return d_alpha_g_raw; }
    /** EOS-derived microbubble contribution before representational bounds. */
    const field_type& raw_microbubble_volume_fraction() const noexcept { return d_alpha_g_micro; }
    /** EOS-derived large-bubble contribution before representational bounds. */
    const field_type& raw_large_bubble_volume_fraction() const noexcept { return d_alpha_g_large; }
    /** Raw bubble fraction after transport and before this step's local kinetics. */
    const field_type& transported_raw_bubble_volume_fraction() const noexcept
    {
        return d_transport_alpha_g_raw;
    }
    /** Slip-volume flux actually paired with the transported bubble state. */
    const face_flux_field_type& transported_bubble_slip_volume_flux() const noexcept
    {
        return d_transport_bubble_slip_volume_flux;
    }
    /** Carrier part of the same implicit bubble transport face flux. */
    const face_flux_field_type& transported_bubble_carrier_volume_flux() const noexcept
    {
        return d_transport_bubble_carrier_volume_flux;
    }

    [[nodiscard]] StateSnapshot snapshot() const;
    void restore(const StateSnapshot& snapshot);
    /** Refresh reconstruction geometry and discard retained numeric transport state. */
    void refresh_geometry();

    /** Build the owner-oriented raw bubble-volume slip flux [m^3/s]. */
    void bubble_slip_volume_flux(const field_type& temperature,
        const material_type& material, Dimension slip_axis,
        face_flux_field_type& output) const;

    /** @brief Global dissolved H2 inventory in moles. */
    scalar_type global_dissolved_hydrogen_moles() const;
    /** @brief Global submerged microbubble H2 inventory in moles. */
    scalar_type global_microbubble_hydrogen_moles() const;
    /** @brief Global submerged large-bubble H2 inventory in moles. */
    scalar_type global_large_bubble_hydrogen_moles() const;
    /** @brief Global gas-phase H2 inventory still submerged in the liquid. */
    scalar_type global_submerged_bubble_hydrogen_moles() const;
    /** @brief Global dissolved plus gas-phase H2 inventory still submerged. */
    scalar_type global_submerged_hydrogen_moles() const;
    /** @brief Total H2 generated over all accepted model advances. */
    scalar_type cumulative_hydrogen_produced() const noexcept { return d_cumulative_hydrogen_produced; }
    /** @brief Total dissolved H2 transported out of the modeled pool. */
    scalar_type cumulative_dissolved_hydrogen_outflow() const noexcept
    {
        return d_cumulative_dissolved_hydrogen_outflow;
    }
    /** @brief Total H2 transferred out of submerged bubble populations. */
    scalar_type cumulative_submerged_bubble_hydrogen_escaped() const noexcept
    {
        return d_cumulative_submerged_bubble_hydrogen_escaped;
    }

    /**
     * @brief Current EOS-derived submerged bubble volume before alpha bounds.
     *
     * This is the volume integral of `alpha_g_raw`, not the bounded
     * hydrodynamic void fraction.
     */
    scalar_type global_submerged_bubble_volume() const;
    /** @brief Raw bubble volume hidden by the configured upper alpha bound. */
    scalar_type global_unrepresented_bubble_volume() const;
    /**
     * @brief Re-evaluate raw bubble volume at an absolute-pressure offset.
     *
     * The candidate replaces the uniform thermodynamic pressure offset while
     * retaining the accepted reconstructed gauge-pressure variation.  Current
     * temperatures, population number densities, gas moles, the configured
     * surface-tension correlation, and the existing Laplace/EOS radius solve
     * are reused.  The accepted model fields are not modified.
     */
    scalar_type evaluate_submerged_bubble_volume(scalar_type candidate_absolute_pressure_offset) const;

    /**
     * @brief Lowest offset that keeps every reconstructed pressure valid.
     *
     * The returned collective value accounts for both the configured absolute
     * pressure floor and the most negative accepted gauge-pressure variation.
     * Constant and reconstructed pressure modes are supported.  All ranks in
     * the model communicator must call this query together.
     */
    scalar_type minimum_valid_absolute_pressure_offset() const;

    /** @brief Uniform absolute-pressure offset used by gas thermodynamics. */
    scalar_type absolute_pressure_offset() const noexcept { return d_absolute_pressure_offset; }
    /**
     * @brief Shift the thermodynamic pressure offset and published pressure.
     *
     * Only constant and reconstructed pressure modes support an externally
     * coupled offset.  Reconstructed mode retains every accepted gauge-pressure
     * variation exactly.  For initialized two-population state, every
     * pressure-dependent bubble diagnostic is reconstructed at the accepted
     * offset without modifying conserved inventories or escape ledgers.  This
     * is a collective operation.
     */
    void set_absolute_pressure_offset(scalar_type pressure_offset);

    /**
     * @brief Diagnostics from the most recent advance call.
     */
    const statistics_type& last_statistics() const noexcept
    {
        return d_last_statistics;
    }

    /**
     * @brief Fields that can be published to solution output.
     */
    const std::map<std::string, const field_type*>& output_fields() const
        noexcept
    {
        return d_output_fields;
    }

private:
    /** @brief Per-cell conserved inventories advanced by local kinetics. */
    struct CellKineticsState
    {
        scalar_type dissolved_inventory = {};
        scalar_type micro_number = {};
        scalar_type micro_moles = {};
        scalar_type large_number = {};
        scalar_type large_moles = {};
    };

    /** @brief Thermophysical inputs derived for one local kinetics solve. */
    struct CellProperties
    {
        scalar_type pressure = {};
        scalar_type temperature = {};
        scalar_type density = {};
        scalar_type viscosity = {};
        scalar_type surface_tension = {};
        scalar_type diffusivity = {};
        scalar_type nucleation_radius = {};
        scalar_type nucleation_moles = {};
    };

    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh);
    void initialize_fields();
    void register_output_fields();
    void reconstruct_absolute_pressure(
        scalar_type time,
        const field_type& gauge_pressure);
    scalar_type prescribed_pressure(scalar_type time) const;
    void update_ideal_gas_source(
        scalar_type time_step,
        const field_type& temperature,
        const field_type* fission_power_density,
        const field_type& alpha_g,
        scalar_type alpha_max);
    void advance_two_population(
        scalar_type time_step,
        const field_type& temperature,
        const velocity_field_type& velocity,
        const face_flux_field_type& liquid_face_flux,
        const material_type& material,
        const field_type* fission_power_density,
        const FVM::ALEControlVolumeState* ale,
        Dimension slip_axis);
    void transport_populations(
        scalar_type time_step,
        const field_type& temperature,
        const velocity_field_type& velocity,
        const face_flux_field_type& liquid_face_flux,
        const material_type& material,
        const FVM::ALEControlVolumeState* ale,
        Dimension slip_axis);
    void transport_scalar(
        field_type& field,
        scalar_type time_step,
        const face_flux_field_type& liquid_face_flux,
        field_type* slip_velocity,
        scalar_type diffusivity,
        bool diffuse,
        bool liquid_weighted,
        field_type& escape_rate,
        const FVM::ALEControlVolumeState* ale,
        Dimension slip_axis);
    CellProperties cell_properties(local_ordinal_type cell_lid, const field_type& temperature,
        const field_type& density, const field_type& dynamic_viscosity) const;
    CellKineticsState integrate_cell_kinetics(
        local_ordinal_type cell_lid,
        scalar_type time_step,
        scalar_type power_density,
        const CellProperties& properties);
    void reconstruct_derived_fields(const field_type& temperature, const field_type& density,
        const field_type& dynamic_viscosity, bool record_event_statistics = true);
    void update_inertial_pressure(
        scalar_type time_step,
        const field_type& temperature,
        const face_flux_field_type& liquid_face_flux,
        const material_type& material);
    void sync_all_fields();
    /** @brief Compute a globally reduced volume integral. */
    scalar_type global_integral(const field_type& field,
        std::span<const real_t> cell_volumes = {}) const;
    /** @brief Sum a rank-local scalar and replicate it on every rank. */
    scalar_type global_sum(scalar_type local_value) const;
    /** @brief Compute and replicate the communicator-wide scalar minimum. */
    scalar_type global_min(scalar_type local_value) const;
    /** @brief Compute and replicate the communicator-wide integer maximum. */
    int global_max(int local_value) const;
    void reduce_event_statistics();
    scalar_type total_hydrogen_inventory(
        std::span<const real_t> cell_volumes = {}) const;
    scalar_type rise_velocity(
        scalar_type radius,
        scalar_type liquid_density,
        scalar_type dynamic_viscosity,
        scalar_type surface_tension) const;
    scalar_type concentration(
        const CellKineticsState& state,
        scalar_type liquid_fraction) const;
    void assign_cell_state(
        local_ordinal_type cell_lid,
        const CellKineticsState& state);
    std::vector<field_type*> mutable_state_fields();
    std::vector<const field_type*> state_fields() const;

    SP<const mesh_type> d_mesh;
    FVM::TransportGeometryCache<mesh_type> d_transport_geometry_cache;
    RadiolyticGasOptions d_options;

    field_type d_alpha_g;
    field_type d_alpha_l;
    field_type d_source_alpha_rad;
    field_type d_absolute_pressure;
    field_type d_previous_temperature;
    field_type d_previous_density;
    field_type d_previous_dynamic_viscosity;
    field_type d_previous_alpha_g;

    field_type d_dissolved_hydrogen;
    field_type d_dissolved_hydrogen_inventory;
    field_type d_excluded_dissolved_inventory;
    field_type d_micro_number;
    field_type d_micro_moles;
    field_type d_large_number;
    field_type d_large_moles;
    field_type d_nucleation_radius;
    field_type d_micro_radius;
    field_type d_large_radius;
    field_type d_critical_concentration;
    field_type d_equilibrium_concentration;
    field_type d_mass_transfer_coefficient;
    field_type d_alpha_g_micro;
    field_type d_alpha_g_large;
    field_type d_alpha_g_raw;
    field_type d_transport_alpha_g_raw;
    field_type d_alpha_g_excess;
    field_type d_characteristic_radius;

    field_type d_hydrogen_production_rate;
    field_type d_micro_to_large_number_rate;
    field_type d_micro_to_large_molar_rate;
    field_type d_large_growth_rate;
    field_type d_dissolution_rate;
    field_type d_escape_molar_rate;
    field_type d_escape_number_rate;
    field_type d_inventory_error;
    face_flux_field_type d_transport_bubble_slip_volume_flux;
    face_flux_field_type d_transport_bubble_carrier_volume_flux;

    bool d_history_initialized = false;
    bool d_initial_state_initialized = false;
    scalar_type d_absolute_pressure_offset = {};
    scalar_type d_cumulative_hydrogen_produced = {};
    scalar_type d_cumulative_dissolved_hydrogen_outflow = {};
    scalar_type d_cumulative_submerged_bubble_hydrogen_escaped = {};
    scalar_type d_cumulative_hydrogen_escaped = {};
    scalar_type d_cumulative_escaped_bubble_count = {};
    BelosLinearSolver<Pack> d_transport_solver;
    statistics_type d_last_statistics;
    std::map<std::string, const field_type*> d_output_fields;
};

} // namespace SimpleFluid

#include "equations/RadiolyticGasModel.tcc"
