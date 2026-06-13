/**
 * @file RadiolyticGasModel.hh
 * @brief Runtime-selectable ideal-gas and two-population radiolysis model.
 */
#pragma once

#include "equations/BoussinesqModel.hh"
#include "equations/RadiolyticGasProperties.hh"
#include "fields/FaceField.hh"
#include "solvers/BelosLinearSolver.hh"

#include <map>
#include <string>
#include <string_view>

namespace SimpleFluid
{

template<class Scalar>
struct RadiolyticGasStepStatistics
{
    Scalar hydrogen_before = {};
    Scalar hydrogen_produced = {};
    Scalar hydrogen_escaped = {};
    Scalar hydrogen_after = {};
    Scalar inventory_error = {};
    Scalar escaped_bubble_count = {};
    Scalar cumulative_hydrogen_escaped = {};
    Scalar cumulative_escaped_bubble_count = {};
    Scalar void_volume = {};
    int maximum_subcycles = 0;
    int clipped_cells = 0;
    int pressure_floor_cells = 0;
    int radius_solver_failures = 0;
};

template<TpetraTypePack Pack = DefaultTpetraTypes>
class RadiolyticGasModel
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using face_flux_field_type = FaceField<Pack>;
    using material_type = MaterialPropertyFields<Pack>;
    using statistics_type = RadiolyticGasStepStatistics<scalar_type>;

    explicit RadiolyticGasModel(
        SP<const mesh_type> mesh,
        RadiolyticGasOptions options = {});

    void configure(const RadiolyticGasOptions& options);
    const RadiolyticGasOptions& options() const noexcept
    {
        return d_options;
    }
    RadiolyticGasMode mode() const noexcept { return d_options.mode; }
    bool enabled() const noexcept
    {
        return mode() != RadiolyticGasMode::Disabled;
    }
    bool supplies_void_fraction() const noexcept
    {
        return mode() == RadiolyticGasMode::Sheng2024TwoPopulation;
    }

    void advance(
        scalar_type time,
        scalar_type time_step,
        const field_type& temperature,
        const field_type& gauge_pressure,
        const velocity_field_type& velocity,
        const face_flux_field_type& liquid_face_flux,
        const material_type& material,
        const field_type* fission_power_density);

    field_type& alpha_g() noexcept { return d_alpha_g; }
    const field_type& alpha_g() const noexcept { return d_alpha_g; }
    const field_type& alpha_l() const noexcept { return d_alpha_l; }
    const field_type& source_alpha_rad() const noexcept
    {
        return d_source_alpha_rad;
    }
    const field_type& absolute_pressure() const noexcept
    {
        return d_absolute_pressure;
    }
    const field_type& dissolved_hydrogen() const noexcept
    {
        return d_dissolved_hydrogen;
    }
    const field_type& dissolved_hydrogen_inventory() const noexcept
    {
        return d_dissolved_hydrogen_inventory;
    }
    const field_type& micro_number_density() const noexcept
    {
        return d_micro_number;
    }
    const field_type& micro_moles() const noexcept
    {
        return d_micro_moles;
    }
    const field_type& large_number_density() const noexcept
    {
        return d_large_number;
    }
    const field_type& large_moles() const noexcept
    {
        return d_large_moles;
    }
    const statistics_type& last_statistics() const noexcept
    {
        return d_last_statistics;
    }

    const std::map<std::string, const field_type*>& output_fields() const
        noexcept
    {
        return d_output_fields;
    }

private:
    struct CellKineticsState
    {
        scalar_type dissolved_inventory = {};
        scalar_type micro_number = {};
        scalar_type micro_moles = {};
        scalar_type large_number = {};
        scalar_type large_moles = {};
    };

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
        const field_type* fission_power_density);
    void advance_two_population(
        scalar_type time_step,
        const field_type& temperature,
        const velocity_field_type& velocity,
        const face_flux_field_type& liquid_face_flux,
        const material_type& material,
        const field_type* fission_power_density);
    void transport_populations(
        scalar_type time_step,
        const field_type& temperature,
        const velocity_field_type& velocity,
        const face_flux_field_type& liquid_face_flux,
        const material_type& material);
    void transport_scalar(
        field_type& field,
        scalar_type time_step,
        const face_flux_field_type& liquid_face_flux,
        field_type* slip_velocity,
        scalar_type diffusivity,
        bool diffuse,
        bool liquid_weighted,
        scalar_type& escaped_inventory);
    CellProperties cell_properties(
        local_ordinal_type cell_lid,
        const field_type& temperature,
        const velocity_field_type& velocity,
        const material_type& material) const;
    CellKineticsState integrate_cell_kinetics(
        local_ordinal_type cell_lid,
        scalar_type time_step,
        scalar_type power_density,
        const CellProperties& properties);
    void reconstruct_derived_fields(
        const field_type& temperature,
        const velocity_field_type& velocity,
        const material_type& material);
    void update_inertial_pressure(
        scalar_type time_step,
        const field_type& temperature,
        const face_flux_field_type& liquid_face_flux,
        const material_type& material);
    void sync_all_fields();
    scalar_type global_integral(const field_type& field) const;
    scalar_type global_sum(scalar_type local_value) const;
    scalar_type total_hydrogen_inventory() const;
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

    SP<const mesh_type> d_mesh;
    RadiolyticGasOptions d_options;

    field_type d_alpha_g;
    field_type d_alpha_l;
    field_type d_source_alpha_rad;
    field_type d_absolute_pressure;
    field_type d_previous_temperature;
    field_type d_previous_density;
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

    bool d_history_initialized = false;
    scalar_type d_cumulative_hydrogen_escaped = {};
    scalar_type d_cumulative_escaped_bubble_count = {};
    BelosLinearSolver<Pack> d_transport_solver;
    statistics_type d_last_statistics;
    std::map<std::string, const field_type*> d_output_fields;
};

} // namespace SimpleFluid

#include "equations/RadiolyticGasModel.tcc"
