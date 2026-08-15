/**
 * @file RadiolyticGasModel.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementations for the runtime radiolytic gas model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "FVM/CellOperators.hh"
#include "FVM/TransportSystem.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

namespace SimpleFluid
{

/**
 * @brief Require a mesh before constructing model-owned fields.
 * @tparam Pack Tpetra type pack used by the model.
 * @param mesh Candidate computational mesh.
 * @return The validated mesh pointer.
 * @throws std::invalid_argument if @p mesh is null.
 */
template<TpetraTypePack Pack, class MeshType>
SP<const typename RadiolyticGasModel<Pack, MeshType>::mesh_type>
RadiolyticGasModel<Pack, MeshType>::require_mesh(SP<const mesh_type> mesh)
{
    if (!mesh)
    {
        throw std::invalid_argument(
            "RadiolyticGasModel requires a non-null mesh.");
    }
    return mesh;
}

/**
 * @brief Construct and initialize a radiolytic gas model.
 * @tparam Pack Tpetra type pack used by the model.
 * @param mesh Computational mesh.
 * @param options Validated runtime physics options.
 * @throws std::invalid_argument if the mesh or options are invalid.
 */
template<TpetraTypePack Pack, class MeshType>
RadiolyticGasModel<Pack, MeshType>::RadiolyticGasModel(
    SP<const mesh_type> mesh,
    RadiolyticGasOptions options)
    : d_mesh(require_mesh(std::move(mesh))),
      d_transport_geometry_cache(*d_mesh),
      d_options(std::move(options)),
      d_alpha_g(d_mesh, "alpha_g"),
      d_alpha_l(d_mesh, "alpha_l"),
      d_source_alpha_rad(d_mesh, "S_alpha_rad"),
      d_absolute_pressure(d_mesh, "p_abs"),
      d_previous_temperature(d_mesh, "radiolytic_previous_temperature"),
      d_previous_density(d_mesh, "radiolytic_previous_density"),
      d_previous_alpha_g(d_mesh, "radiolytic_previous_alpha_g"),
      d_dissolved_hydrogen(d_mesh, "C_H2"),
      d_dissolved_hydrogen_inventory(d_mesh, "I_H2"),
      d_excluded_dissolved_inventory(d_mesh, "I_H2_excluded"),
      d_micro_number(d_mesh, "N_micro"),
      d_micro_moles(d_mesh, "M_micro"),
      d_large_number(d_mesh, "N_large"),
      d_large_moles(d_mesh, "M_large"),
      d_nucleation_radius(d_mesh, "r_nuc"),
      d_micro_radius(d_mesh, "r_micro"),
      d_large_radius(d_mesh, "r_large"),
      d_critical_concentration(d_mesh, "C_H2_crit"),
      d_equilibrium_concentration(d_mesh, "C_H2_eq"),
      d_mass_transfer_coefficient(d_mesh, "K_L"),
      d_alpha_g_micro(d_mesh, "alpha_g_micro"),
      d_alpha_g_large(d_mesh, "alpha_g_large"),
      d_alpha_g_raw(d_mesh, "alpha_g_raw"),
      d_alpha_g_excess(d_mesh, "alpha_g_excess"),
      d_characteristic_radius(d_mesh, "r_characteristic"),
      d_hydrogen_production_rate(d_mesh, "H2_production_rate"),
      d_micro_to_large_number_rate(
          d_mesh, "micro_to_large_number_rate"),
      d_micro_to_large_molar_rate(
          d_mesh, "micro_to_large_molar_rate"),
      d_large_growth_rate(d_mesh, "large_growth_rate"),
      d_dissolution_rate(d_mesh, "H2_dissolution_rate"),
      d_escape_molar_rate(d_mesh, "H2_escape_molar_rate"),
      d_escape_number_rate(d_mesh, "bubble_escape_number_rate"),
      d_inventory_error(d_mesh, "H2_inventory_error")
{
    validate_radiolytic_gas_options(d_options);
    if (d_options.mode
            == RadiolyticGasMode::Sheng2024TwoPopulation
        && d_options.micro_to_large_conversion_coefficient == 1.0e-4
        && d_mesh->owned_cell_map()->getComm()->getRank() == 0)
    {
        std::clog
            << "SimpleFluid radiolysis: F=1e-4 Pa^-1 s^-1 is "
               "SILENE-calibrated and requires case-specific validation.\n";
    }
    register_output_fields();
    initialize_fields();
}

/**
 * @brief Reconfigure the model and reset its evolving state.
 * @tparam Pack Tpetra type pack used by the model.
 * @param options Replacement runtime physics options.
 * @throws std::invalid_argument if @p options is inconsistent.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::configure(
    const RadiolyticGasOptions& options)
{
    validate_radiolytic_gas_options(options);
    d_options = options;
    if (d_options.mode
            == RadiolyticGasMode::Sheng2024TwoPopulation
        && d_options.micro_to_large_conversion_coefficient == 1.0e-4
        && d_mesh->owned_cell_map()->getComm()->getRank() == 0)
    {
        std::clog
            << "SimpleFluid radiolysis: F=1e-4 Pa^-1 s^-1 is "
               "SILENE-calibrated and requires case-specific validation.\n";
    }
    d_history_initialized = false;
    d_cumulative_hydrogen_escaped = 0.0;
    d_cumulative_escaped_bubble_count = 0.0;
    d_last_statistics = {};
    initialize_fields();
}

/**
 * @brief Register stable output names for all model-owned fields.
 * @tparam Pack Tpetra type pack used by the model.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::register_output_fields()
{
    d_output_fields = {
        {"alpha_g", &d_alpha_g},
        {"alpha_l", &d_alpha_l},
        {"S_alpha_rad", &d_source_alpha_rad},
        {"p_abs", &d_absolute_pressure},
        {"C_H2", &d_dissolved_hydrogen},
        {"I_H2", &d_dissolved_hydrogen_inventory},
        {"I_H2_excluded", &d_excluded_dissolved_inventory},
        {"N_micro", &d_micro_number},
        {"M_micro", &d_micro_moles},
        {"N_large", &d_large_number},
        {"M_large", &d_large_moles},
        {"r_nuc", &d_nucleation_radius},
        {"r_micro", &d_micro_radius},
        {"r_large", &d_large_radius},
        {"C_H2_crit", &d_critical_concentration},
        {"C_H2_eq", &d_equilibrium_concentration},
        {"K_L", &d_mass_transfer_coefficient},
        {"alpha_g_micro", &d_alpha_g_micro},
        {"alpha_g_large", &d_alpha_g_large},
        {"alpha_g_raw", &d_alpha_g_raw},
        {"alpha_g_excess", &d_alpha_g_excess},
        {"r_characteristic", &d_characteristic_radius},
        {"H2_production_rate", &d_hydrogen_production_rate},
        {"micro_to_large_number_rate",
         &d_micro_to_large_number_rate},
        {"micro_to_large_molar_rate",
         &d_micro_to_large_molar_rate},
        {"large_growth_rate", &d_large_growth_rate},
        {"H2_dissolution_rate", &d_dissolution_rate},
        {"H2_escape_molar_rate", &d_escape_molar_rate},
        {"bubble_escape_number_rate", &d_escape_number_rate},
        {"H2_inventory_error", &d_inventory_error}};
}

/**
 * @brief Reset primary, history, diagnostic, and cumulative model fields.
 * @tparam Pack Tpetra type pack used by the model.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::initialize_fields()
{
    d_history_initialized = false;
    d_initial_state_initialized = false;
    d_cumulative_hydrogen_escaped = scalar_type{};
    d_cumulative_escaped_bubble_count = scalar_type{};
    d_last_statistics = {};
    d_alpha_g.put_scalar(d_options.alpha_min);
    d_alpha_l.put_scalar(1.0 - d_options.alpha_min);
    d_source_alpha_rad.put_scalar(0.0);
    d_absolute_pressure.put_scalar(d_options.reference_pressure);
    d_previous_temperature.put_scalar(0.0);
    d_previous_density.put_scalar(0.0);
    d_previous_alpha_g.put_scalar(d_options.alpha_min);
    d_dissolved_hydrogen.put_scalar(
        d_options.initial_dissolved_hydrogen);
    d_dissolved_hydrogen_inventory.put_scalar(
        (1.0 - d_options.alpha_min)
        * d_options.initial_dissolved_hydrogen);
    d_excluded_dissolved_inventory.put_scalar(0.0);
    d_micro_number.put_scalar(
        d_options.initial_micro_number_density);
    d_micro_moles.put_scalar(d_options.initial_micro_moles);
    d_large_number.put_scalar(
        d_options.initial_large_number_density);
    d_large_moles.put_scalar(d_options.initial_large_moles);

    d_nucleation_radius.put_scalar(0.0);
    d_micro_radius.put_scalar(0.0);
    d_large_radius.put_scalar(0.0);
    d_critical_concentration.put_scalar(0.0);
    d_equilibrium_concentration.put_scalar(0.0);
    d_mass_transfer_coefficient.put_scalar(0.0);
    d_alpha_g_micro.put_scalar(0.0);
    d_alpha_g_large.put_scalar(0.0);
    d_alpha_g_raw.put_scalar(d_options.alpha_min);
    d_alpha_g_excess.put_scalar(0.0);
    d_characteristic_radius.put_scalar(0.0);
    d_hydrogen_production_rate.put_scalar(0.0);
    d_micro_to_large_number_rate.put_scalar(0.0);
    d_micro_to_large_molar_rate.put_scalar(0.0);
    d_large_growth_rate.put_scalar(0.0);
    d_dissolution_rate.put_scalar(0.0);
    d_escape_molar_rate.put_scalar(0.0);
    d_escape_number_rate.put_scalar(0.0);
    d_inventory_error.put_scalar(0.0);
    sync_all_fields();
}

/**
 * @brief Build the derived initial state for two-population radiolysis.
 * @tparam Pack Tpetra type pack used by the model.
 * @param time Initial physical time.
 * @param temperature Initial cell temperature.
 * @param gauge_pressure Initial gauge pressure in Pa.
 * @param velocity Initial liquid velocity.
 * @param material Initial material-property fields.
 * @param force Whether to discard an already initialized state.
 * @throws std::logic_error if the active mode is not two-population.
 * @throws std::invalid_argument if time or input meshes are invalid.
 * @throws std::runtime_error if a derived property cannot be reconstructed.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::initialize_state(
    scalar_type time,
    const field_type& temperature,
    const field_type& gauge_pressure,
    const velocity_field_type& velocity,
    const material_type& material,
    bool force)
{
    if (mode() != RadiolyticGasMode::Sheng2024TwoPopulation)
    {
        throw std::logic_error(
            "Only Sheng two-population radiolysis has a derived initial state.");
    }
    if (!std::isfinite(time))
    {
        throw std::invalid_argument(
            "Radiolytic gas initialization time must be finite.");
    }
    if (&temperature.mesh() != d_mesh.get()
        || &gauge_pressure.mesh() != d_mesh.get()
        || &velocity.mesh() != d_mesh.get()
        || &material.density.mesh() != d_mesh.get())
    {
        throw std::invalid_argument(
            "Radiolytic gas initial fields are on the wrong mesh.");
    }
    if (d_initial_state_initialized && !force)
    {
        return;
    }
    if (force)
    {
        initialize_fields();
    }

    reconstruct_absolute_pressure(time, gauge_pressure);
    reconstruct_derived_fields(temperature, velocity, material);

    const auto published_concentration = std::min(
        static_cast<scalar_type>(d_options.initial_dissolved_hydrogen),
        static_cast<scalar_type>(d_options.max_concentration));
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto liquid_fraction = d_alpha_l.value(cell_lid);
        const auto dissolved_inventory =
            liquid_fraction * d_options.initial_dissolved_hydrogen;
        d_dissolved_hydrogen_inventory.set_owned_value(
            cell_lid, dissolved_inventory);
        d_dissolved_hydrogen.set_owned_value(
            cell_lid, published_concentration);
        d_excluded_dissolved_inventory.set_owned_value(
            cell_lid,
            std::max(
                dissolved_inventory
                  - liquid_fraction * published_concentration,
                scalar_type{}));
        d_source_alpha_rad.set_owned_value(cell_lid, scalar_type{});
        d_previous_temperature.set_owned_value(
            cell_lid, temperature.value(cell_lid));
        d_previous_density.set_owned_value(
            cell_lid, material.density.value(cell_lid));
        d_previous_alpha_g.set_owned_value(
            cell_lid, d_alpha_g.value(cell_lid));
    }

    d_history_initialized = true;
    d_initial_state_initialized = true;
    d_last_statistics = {};
    sync_all_fields();
}

/**
 * @brief Sum one scalar value over the model communicator.
 * @tparam Pack Tpetra type pack used by the model.
 * @param local_value Rank-local contribution.
 * @return Communicator-wide sum.
 */
template<TpetraTypePack Pack, class MeshType>
auto RadiolyticGasModel<Pack, MeshType>::global_sum(
    scalar_type local_value) const -> scalar_type
{
    scalar_type global_value{};
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_SUM,
        1,
        &local_value,
        &global_value);
    return global_value;
}

/**
 * @brief Compute the communicator-wide maximum of an integer diagnostic.
 * @tparam Pack Tpetra type pack used by the model.
 * @param local_value Rank-local diagnostic.
 * @return Communicator-wide maximum.
 */
template<TpetraTypePack Pack, class MeshType>
int RadiolyticGasModel<Pack, MeshType>::global_max(int local_value) const
{
    int global_value{};
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_value,
        &global_value);
    return global_value;
}

/**
 * @brief Reduce local event counters into rank-consistent statistics.
 * @tparam Pack Tpetra type pack used by the model.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::reduce_event_statistics()
{
    const int local_counts[]{
        d_last_statistics.clipped_cells,
        d_last_statistics.pressure_floor_cells,
        d_last_statistics.radius_solver_failures};
    int global_counts[3]{};
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_SUM,
        3,
        local_counts,
        global_counts);
    d_last_statistics.maximum_subcycles = global_max(
        d_last_statistics.maximum_subcycles);
    d_last_statistics.clipped_cells = global_counts[0];
    d_last_statistics.pressure_floor_cells = global_counts[1];
    d_last_statistics.radius_solver_failures = global_counts[2];
}

/**
 * @brief Integrate an owned cell field over the distributed mesh volume.
 * @tparam Pack Tpetra type pack used by the model.
 * @param field Cell field to integrate.
 * @return Global volume integral.
 */
template<TpetraTypePack Pack, class MeshType>
auto RadiolyticGasModel<Pack, MeshType>::global_integral(
    const field_type& field) const -> scalar_type
{
    scalar_type local_integral{};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        local_integral +=
            field.value(cell_lid) * d_mesh->cell_volume(cell_lid);
    }
    return global_sum(local_integral);
}

/**
 * @brief Compute dissolved plus bubble hydrogen inventory over the mesh.
 * @tparam Pack Tpetra type pack used by the model.
 * @return Global hydrogen inventory in moles.
 */
template<TpetraTypePack Pack, class MeshType>
auto RadiolyticGasModel<Pack, MeshType>::total_hydrogen_inventory() const
    -> scalar_type
{
    return global_integral(d_dissolved_hydrogen_inventory)
         + global_integral(d_micro_moles)
         + global_integral(d_large_moles);
}

/**
 * @brief Interpolate the prescribed absolute-pressure history.
 * @tparam Pack Tpetra type pack used by the model.
 * @param time Physical time at which to sample the history.
 * @return Clamped, linearly interpolated absolute pressure.
 */
template<TpetraTypePack Pack, class MeshType>
auto RadiolyticGasModel<Pack, MeshType>::prescribed_pressure(
    scalar_type time) const -> scalar_type
{
    const auto& times = d_options.pressure_history_times;
    const auto& values = d_options.pressure_history_values;
    if (times.empty())
        return d_options.reference_pressure;
    if (time <= times.front())
        return values.front();
    if (time >= times.back())
        return values.back();

    const auto upper =
        std::upper_bound(times.begin(), times.end(), time);
    const auto high =
        static_cast<size_t>(std::distance(times.begin(), upper));
    const auto low = high - 1;
    const auto fraction =
        (time - times[low]) / (times[high] - times[low]);
    return values[low]
         + fraction * (values[high] - values[low]);
}

/**
 * @brief Reconstruct absolute pressure for the configured pressure mode.
 * @tparam Pack Tpetra type pack used by the model.
 * @param time Current physical time.
 * @param gauge_pressure Cell gauge pressure in Pa.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::reconstruct_absolute_pressure(
    scalar_type time,
    const field_type& gauge_pressure)
{
    if (d_options.pressure_mode == RadiolyticPressureMode::Inertial
        && d_history_initialized)
    {
        return;
    }

    if (d_options.pressure_mode == RadiolyticPressureMode::Constant
        || d_options.pressure_mode
            == RadiolyticPressureMode::PrescribedHistory)
    {
        d_absolute_pressure.put_scalar(
            d_options.pressure_mode
                    == RadiolyticPressureMode::Constant
                ? d_options.reference_pressure
                : prescribed_pressure(time));
        d_absolute_pressure.sync_ghosts();
        return;
    }

    scalar_type local_pressure_volume{};
    scalar_type local_volume{};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto volume = d_mesh->cell_volume(cell_lid);
        local_pressure_volume +=
            gauge_pressure.value(cell_lid) * volume;
        local_volume += volume;
    }
    const auto volume = global_sum(local_volume);
    const auto mean_pressure =
        volume > 0.0
            ? global_sum(local_pressure_volume) / volume
            : 0.0;

    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        auto pressure =
            d_options.reference_pressure
          + gauge_pressure.value(cell_lid) - mean_pressure;
        if (pressure < d_options.minimum_absolute_pressure)
        {
            ++d_last_statistics.pressure_floor_cells;
            pressure = d_options.minimum_absolute_pressure;
        }
        d_absolute_pressure.set_owned_value(cell_lid, pressure);
    }
    d_absolute_pressure.sync_ghosts();
}

/**
 * @brief Update the ideal-gas void-fraction source from fission power.
 * @tparam Pack Tpetra type pack used by the model.
 * @param time_step Positive physical time step.
 * @param temperature Cell temperature field.
 * @param fission_power_density Required fission power-density field.
 * @param alpha_g Authoritative gas void fraction.
 * @param alpha_max Upper bound for gas void fraction.
 * @throws std::invalid_argument if required fields or values are invalid.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::update_ideal_gas_source(
    scalar_type time_step,
    const field_type& temperature,
    const field_type* fission_power_density,
    const field_type& alpha_g,
    scalar_type alpha_max)
{
    if (!fission_power_density)
    {
        throw std::invalid_argument(
            "Enabled radiolysis requires a fission power source.");
    }
    if (!std::isfinite(time_step) || time_step <= 0.0)
    {
        throw std::invalid_argument(
            "Radiolytic gas timestep must be finite and positive.");
    }

    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto alpha = alpha_g.value(cell_lid);
        RadiolyticGasPhysics::require_positive(
            temperature.value(cell_lid), "temperature");
        RadiolyticGasPhysics::require_positive(
            d_absolute_pressure.value(cell_lid),
            "absolute pressure");
        scalar_type source{};
        if (alpha < alpha_max
            && fission_power_density->value(cell_lid) > 0.0)
        {
            source = RadiolyticGasPhysics::ideal_gas_alpha_source(
                1.0 - alpha,
                d_options.gas_release_efficiency,
                d_options.hydrogen_yield_mol_per_j,
                fission_power_density->value(cell_lid),
                d_options.gas_constant,
                temperature.value(cell_lid),
                d_absolute_pressure.value(cell_lid),
                d_options.max_source_alpha_rate);
            source = std::min(
                source,
                (alpha_max - alpha) / time_step);
        }
        d_source_alpha_rad.set_owned_value(cell_lid, source);
    }
    d_source_alpha_rad.sync_ghosts();
}

/**
 * @brief Evaluate the configured bubble slip velocity.
 * @tparam Pack Tpetra type pack used by the model.
 * @param radius Bubble radius.
 * @param liquid_density Liquid density.
 * @param dynamic_viscosity Liquid dynamic viscosity.
 * @param surface_tension Liquid-gas surface tension.
 * @return Non-negative rise speed.
 * @throws std::invalid_argument if correlation inputs are invalid.
 * @throws std::runtime_error if the Celata iteration does not converge.
 */
template<TpetraTypePack Pack, class MeshType>
auto RadiolyticGasModel<Pack, MeshType>::rise_velocity(
    scalar_type radius,
    scalar_type liquid_density,
    scalar_type dynamic_viscosity,
    scalar_type surface_tension) const -> scalar_type
{
    switch (d_options.rise_velocity_mode)
    {
        case BubbleRiseVelocityMode::ZeroSlip:
            return 0.0;
        case BubbleRiseVelocityMode::ConstantSlip:
            return d_options.constant_slip_velocity;
        case BubbleRiseVelocityMode::Celata2007:
        {
            const auto result =
                RadiolyticGasPhysics::celata2007_bubble_rise_velocity(
                    radius,
                    liquid_density,
                    d_options.bubble_gas_density,
                    dynamic_viscosity,
                    surface_tension,
                    d_options.bubble_gravity,
                    d_options.max_rise_velocity_iterations,
                    d_options.rise_velocity_tolerance);
            if (!result.converged)
            {
                throw std::runtime_error(
                    "Celata 2007 bubble-rise solve did not converge.");
            }
            return result.velocity;
        }
    }
    return 0.0;
}

/**
 * @brief Transport one non-negative radiolytic inventory field.
 * @tparam Pack Tpetra type pack used by the model.
 * @param[in,out] field Inventory or population field to transport.
 * @param time_step Positive physical time step.
 * @param liquid_face_flux Oriented liquid volumetric flux.
 * @param slip_velocity Optional cell slip speed added in the axial direction.
 * @param diffusivity Molecular diffusivity.
 * @param diffuse Whether diffusion is active.
 * @param liquid_weighted Whether storage and advection use liquid fraction.
 * @param[in,out] escape_rate Accumulated free-surface escape rate.
 * @throws std::invalid_argument if a field or transport input is invalid.
 * @throws std::runtime_error if the transport solve does not converge.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::transport_scalar(
    field_type& field,
    scalar_type time_step,
    const face_flux_field_type& liquid_face_flux,
    field_type* slip_velocity,
    scalar_type diffusivity,
    bool diffuse,
    bool liquid_weighted,
    field_type& escape_rate)
{
    if (&escape_rate.mesh() != d_mesh.get())
    {
        throw std::invalid_argument(
            "Radiolytic escape rate is on the wrong mesh.");
    }
    field.sync_ghosts();
    d_alpha_l.sync_ghosts();
    if (slip_velocity)
        slip_velocity->sync_ghosts();
    auto is_free_surface = [&](local_ordinal_type face_lid)
    {
        if (!d_mesh->is_boundary_face(face_lid))
            return false;
        const auto& name = d_mesh->boundary_batch_name(
            d_mesh->boundary_id(face_lid));
        return std::ranges::find(
                   d_options.free_surface_patches, name)
            != d_options.free_surface_patches.end();
    };

    face_flux_field_type transport_flux(
        d_mesh, 0.0, "radiolytic_transport_flux");
    for (size_t face = 0; face < d_mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<local_ordinal_type>(face);
        if (!transport_flux.is_owned_face(face_lid))
            continue;
        auto flux = liquid_face_flux.is_owned_face(face_lid)
            ? liquid_face_flux.value(face_lid)
            : scalar_type{};
        if (slip_velocity)
        {
            const auto owner = d_mesh->owner_cell(face_lid);
            auto face_slip = slip_velocity->local_value(owner);
            if (!d_mesh->is_boundary_face(face_lid))
            {
                const auto neighbor =
                    d_mesh->opposite_or_periodic_neighbor_cell(
                        face_lid, owner);
                face_slip = 0.5
                    * (face_slip
                       + slip_velocity->local_value(neighbor));
            }
            flux += face_slip
                  * d_mesh->face_area_vector(face_lid).z;
        }
        if (d_mesh->is_boundary_face(face_lid))
        {
            flux = is_free_surface(face_lid)
                ? std::max(flux, scalar_type{})
                : scalar_type{};
        }
        transport_flux.set_value(face_lid, flux);
    }
    if constexpr (requires { transport_flux.sync_ghosts(); })
    {
        transport_flux.sync_ghosts();
    }

    field_type old_values(d_mesh, "radiolytic_transport_old");
    field_type storage_weight(
        d_mesh, 1.0, "radiolytic_storage_weight");
    field_type advection_weight(
        d_mesh, 1.0, "radiolytic_advection_weight");
    field_type diffusion_weight(
        d_mesh, 0.0, "radiolytic_diffusion_weight");
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto liquid_fraction =
            std::max(d_alpha_l.value(cell_lid), 1.0e-15);
        old_values.set_owned_value(
            cell_lid,
            liquid_weighted
                ? field.value(cell_lid) / liquid_fraction
                : field.value(cell_lid));
        if (liquid_weighted)
        {
            storage_weight.set_owned_value(
                cell_lid, liquid_fraction);
            advection_weight.set_owned_value(
                cell_lid, liquid_fraction);
        }
        diffusion_weight.set_owned_value(
            cell_lid,
            diffuse ? liquid_fraction * diffusivity : 0.0);
    }
    old_values.sync_ghosts();
    storage_weight.sync_ghosts();
    advection_weight.sync_ghosts();
    diffusion_weight.sync_ghosts();

    auto boundary_condition =
        [](int, size_t)
    {
        return BoundaryCondition{
            BoundaryConditionType::Neumann, 0.0};
    };
    auto boundary_value =
        [](int, size_t) -> scalar_type { return 0.0; };
    auto source =
        [](local_ordinal_type) -> scalar_type { return 0.0; };
    auto system = FVM::weighted_scalar_transport_system<Pack>(
        old_values,
        transport_flux,
        time_step,
        storage_weight,
        advection_weight,
        diffusion_weight,
        boundary_condition,
        boundary_value,
        source,
        FVM::NonOrthogonalTreatment::Hybrid,
        &old_values,
        Teuchos::null,
        {},
        {},
        nullptr,
        &d_transport_geometry_cache);

    field_type solution(d_mesh, "radiolytic_transport_solution");
    const auto solve_statistics =
        d_transport_solver.solve_with_statistics(
            system.matrix,
            *system.rhs,
            solution.owned_data(),
            LinearSolverOptions{});
    if (!solve_statistics.converged)
    {
        throw std::runtime_error(
            "Radiolytic weighted transport solve did not converge.");
    }

    for (size_t owned = 0;
         owned < d_mesh->num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        auto transported = solution.value(cell_lid);
        if (liquid_weighted)
            transported *= storage_weight.value(cell_lid);
        if (!std::isfinite(transported) || transported < 0.0)
        {
            ++d_last_statistics.clipped_cells;
            transported = 0.0;
        }
        field.set_owned_value(cell_lid, transported);
    }
    field.sync_ghosts();

    for (size_t face = 0; face < d_mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<local_ordinal_type>(face);
        if (!transport_flux.is_owned_face(face_lid)
            || !is_free_surface(face_lid))
        {
            continue;
        }
        const auto flux =
            std::max(transport_flux.value(face_lid), scalar_type{});
        const auto owner = d_mesh->owner_cell(face_lid);
        const auto primary_value = liquid_weighted
            ? field.value(owner) / storage_weight.value(owner)
            : field.value(owner);
        const auto boundary_rate =
            flux
          * advection_weight.local_value(owner)
          * primary_value;
        escape_rate.sum_into_value(
            owner,
            boundary_rate / d_mesh->cell_volume(owner));
    }
}

/**
 * @brief Transport dissolved hydrogen and both bubble populations.
 * @tparam Pack Tpetra type pack used by the model.
 * @param time_step Positive physical time step.
 * @param temperature Cell temperature field.
 * @param velocity Liquid velocity field.
 * @param liquid_face_flux Oriented liquid volumetric flux.
 * @param material Material-property fields.
 * @throws std::invalid_argument if property or transport inputs are invalid.
 * @throws std::runtime_error if a radius, rise, or transport solve fails.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::transport_populations(
    scalar_type time_step,
    const field_type& temperature,
    const velocity_field_type& velocity,
    const face_flux_field_type& liquid_face_flux,
    const material_type& material)
{
    d_escape_molar_rate.put_scalar(0.0);
    d_escape_number_rate.put_scalar(0.0);
    scalar_type diffusivity = d_options.hydrogen_diffusivity;
    if (d_options.diffusivity_mode
        == HydrogenDiffusivityMode::Sheng2024)
    {
        scalar_type local_temperature_volume{};
        scalar_type local_volume{};
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto volume = d_mesh->cell_volume(cell_lid);
            local_temperature_volume +=
                temperature.value(cell_lid) * volume;
            local_volume += volume;
        }
        const auto volume = global_sum(local_volume);
        const auto mean_temperature =
            global_sum(local_temperature_volume) / volume;
        diffusivity =
            RadiolyticGasPhysics::sheng2024_hydrogen_diffusivity(
                mean_temperature);
    }

    face_flux_field_type zero_flux(d_mesh, 0.0, "radiolytic_zero_flux");
    const auto& dissolved_flux =
        d_options.dissolved_transport
                == RadiolyticTransportMode::Advective
            ? liquid_face_flux
            : zero_flux;
    transport_scalar(
        d_dissolved_hydrogen_inventory,
        time_step,
        dissolved_flux,
        nullptr,
        diffusivity,
        true,
        true,
        d_escape_molar_rate);

    face_flux_field_type axial_bubble_flux(
        d_mesh, 0.0, "radiolytic_axial_bubble_flux");
    const face_flux_field_type* bubble_liquid_flux =
        &liquid_face_flux;
    if (d_options.bubble_transport == BubbleTransportMode::Axial)
    {
        for (size_t face = 0; face < d_mesh->num_faces(); ++face)
        {
            const auto face_lid =
                static_cast<local_ordinal_type>(face);
            if (!axial_bubble_flux.is_owned_face(face_lid))
                continue;
            const auto owner = d_mesh->owner_cell(face_lid);
            auto axial_velocity = velocity.local_value(owner).z;
            if (!d_mesh->is_boundary_face(face_lid))
            {
                const auto neighbor =
                    d_mesh->opposite_or_periodic_neighbor_cell(
                        face_lid, owner);
                axial_velocity = 0.5
                    * (axial_velocity
                       + velocity.local_value(neighbor).z);
            }
            axial_bubble_flux.set_value(
                face_lid,
                axial_velocity
              * d_mesh->face_area_vector(face_lid).z);
        }
        if constexpr (requires { axial_bubble_flux.sync_ghosts(); })
        {
            axial_bubble_flux.sync_ghosts();
        }
        bubble_liquid_flux = &axial_bubble_flux;
    }

    field_type micro_slip(d_mesh, 0.0, "microbubble_slip_velocity");
    field_type large_slip(d_mesh, 0.0, "large_bubble_slip_velocity");
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto properties =
            cell_properties(cell_lid, temperature, velocity, material);
        const auto category_slip =
            [&](scalar_type number, scalar_type moles)
        {
            if (number <= d_options.min_population || moles <= 0.0)
                return scalar_type{};
            const auto radius =
                RadiolyticGasPhysics::solve_bubble_radius(
                    moles / number,
                    properties.pressure,
                    properties.surface_tension,
                    d_options.gas_constant,
                    properties.temperature,
                    d_options.min_radius,
                    d_options.max_radius,
                    d_options.max_radius_iterations,
                    d_options.local_ode_tolerance);
            if (!radius.converged)
            {
                ++d_last_statistics.radius_solver_failures;
                return scalar_type{};
            }
            return rise_velocity(
                radius.radius,
                properties.density,
                properties.viscosity,
                properties.surface_tension);
        };
        micro_slip.set_owned_value(
            cell_lid,
            category_slip(
                d_micro_number.value(cell_lid),
                d_micro_moles.value(cell_lid)));
        large_slip.set_owned_value(
            cell_lid,
            category_slip(
                d_large_number.value(cell_lid),
                d_large_moles.value(cell_lid)));
    }
    micro_slip.sync_ghosts();
    large_slip.sync_ghosts();

    transport_scalar(
        d_micro_number,
        time_step,
        *bubble_liquid_flux,
        &micro_slip,
        0.0,
        false,
        false,
        d_escape_number_rate);
    transport_scalar(
        d_micro_moles,
        time_step,
        *bubble_liquid_flux,
        &micro_slip,
        0.0,
        false,
        false,
        d_escape_molar_rate);

    transport_scalar(
        d_large_number,
        time_step,
        *bubble_liquid_flux,
        &large_slip,
        0.0,
        false,
        false,
        d_escape_number_rate);
    transport_scalar(
        d_large_moles,
        time_step,
        *bubble_liquid_flux,
        &large_slip,
        0.0,
        false,
        false,
        d_escape_molar_rate);

    d_escape_molar_rate.sync_ghosts();
    d_escape_number_rate.sync_ghosts();
    d_last_statistics.hydrogen_escaped =
        time_step * global_integral(d_escape_molar_rate);
    d_last_statistics.escaped_bubble_count =
        time_step * global_integral(d_escape_number_rate);
    d_cumulative_hydrogen_escaped +=
        d_last_statistics.hydrogen_escaped;
    d_cumulative_escaped_bubble_count +=
        d_last_statistics.escaped_bubble_count;
    d_last_statistics.cumulative_hydrogen_escaped =
        d_cumulative_hydrogen_escaped;
    d_last_statistics.cumulative_escaped_bubble_count =
        d_cumulative_escaped_bubble_count;
}

/**
 * @brief Derive thermophysical and nucleation properties for one cell.
 * @tparam Pack Tpetra type pack used by the model.
 * @param cell_lid Local cell identifier.
 * @param temperature Cell temperature field.
 * @param velocity Liquid velocity field reserved for property extensions.
 * @param material Material-property fields.
 * @return Derived cell properties used by kinetics.
 * @throws std::invalid_argument if a correlation input is invalid.
 * @throws std::runtime_error if the nucleation radius leaves configured bounds.
 */
template<TpetraTypePack Pack, class MeshType>
auto RadiolyticGasModel<Pack, MeshType>::cell_properties(
    local_ordinal_type cell_lid,
    const field_type& temperature,
    const velocity_field_type&,
    const material_type& material) const -> CellProperties
{
    CellProperties properties;
    properties.pressure = d_absolute_pressure.value(cell_lid);
    properties.temperature = temperature.value(cell_lid);
    properties.density = material.density.value(cell_lid);
    properties.viscosity =
        material.dynamic_viscosity.value(cell_lid);
    properties.surface_tension =
        d_options.surface_tension_mode == SurfaceTensionMode::Constant
            ? d_options.surface_tension
            : RadiolyticGasPhysics::sheng2024_surface_tension(
                  properties.temperature - 273.15,
                  d_options.uranium_concentration_mol_per_m3);
    properties.diffusivity =
        d_options.diffusivity_mode
                == HydrogenDiffusivityMode::Constant
            ? d_options.hydrogen_diffusivity
            : RadiolyticGasPhysics::sheng2024_hydrogen_diffusivity(
                  properties.temperature);
    properties.nucleation_radius =
        RadiolyticGasPhysics::sheng2024_nucleation_radius(
            properties.temperature,
            d_options.uranium_concentration_mol_per_m3,
            d_options.hydrogen_yield_molecules_per_100_ev,
            properties.pressure,
            d_options.atmospheric_pressure);
    if (!std::isfinite(properties.nucleation_radius)
        || properties.nucleation_radius < d_options.min_radius
        || properties.nucleation_radius > d_options.max_radius)
    {
        throw std::runtime_error(
            "Sheng 2024 nucleation radius is outside configured bounds.");
    }

    const auto radius = properties.nucleation_radius;
    properties.nucleation_moles =
        4.0 * std::numbers::pi / 3.0
        * (properties.pressure * radius * radius * radius
           + 2.0 * properties.surface_tension
                 * radius * radius)
        / (d_options.gas_constant * properties.temperature);
    return properties;
}

/**
 * @brief Recover dissolved concentration from liquid-volume inventory.
 * @tparam Pack Tpetra type pack used by the model.
 * @param state Conserved per-cell kinetics state.
 * @param liquid_fraction Local liquid volume fraction.
 * @return Dissolved concentration, or zero without liquid volume.
 */
template<TpetraTypePack Pack, class MeshType>
auto RadiolyticGasModel<Pack, MeshType>::concentration(
    const CellKineticsState& state,
    scalar_type liquid_fraction) const -> scalar_type
{
    if (liquid_fraction <= 0.0)
        return 0.0;
    return state.dissolved_inventory / liquid_fraction;
}

/**
 * @brief Clamp and publish one updated cell kinetics state.
 * @tparam Pack Tpetra type pack used by the model.
 * @param cell_lid Local cell identifier.
 * @param state Updated conserved inventories and populations.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::assign_cell_state(
    local_ordinal_type cell_lid,
    const CellKineticsState& state)
{
    const auto micro_number = std::clamp(
        state.micro_number, scalar_type{}, d_options.max_population);
    const auto large_number = std::clamp(
        state.large_number, scalar_type{}, d_options.max_population);
    if (micro_number != state.micro_number
        || large_number != state.large_number)
    {
        ++d_last_statistics.clipped_cells;
    }
    d_dissolved_hydrogen_inventory.set_owned_value(
        cell_lid, std::max(state.dissolved_inventory, scalar_type{}));
    d_micro_number.set_owned_value(cell_lid, micro_number);
    d_micro_moles.set_owned_value(
        cell_lid, std::max(state.micro_moles, scalar_type{}));
    d_large_number.set_owned_value(cell_lid, large_number);
    d_large_moles.set_owned_value(
        cell_lid, std::max(state.large_moles, scalar_type{}));
}

/**
 * @brief Integrate local hydrogen production, conversion, and dissolution.
 * @tparam Pack Tpetra type pack used by the model.
 * @param cell_lid Local cell identifier.
 * @param time_step Positive physical time step.
 * @param power_density Local fission power density.
 * @param properties Derived thermophysical cell properties.
 * @return Updated conserved cell kinetics state.
 * @throws std::invalid_argument if a correlation input is invalid.
 * @throws std::runtime_error if the selected rise-velocity solve fails.
 */
template<TpetraTypePack Pack, class MeshType>
auto RadiolyticGasModel<Pack, MeshType>::integrate_cell_kinetics(
    local_ordinal_type cell_lid,
    scalar_type time_step,
    scalar_type power_density,
    const CellProperties& properties) -> CellKineticsState
{
    CellKineticsState state{
        d_dissolved_hydrogen_inventory.value(cell_lid),
        d_micro_number.value(cell_lid),
        d_micro_moles.value(cell_lid),
        d_large_number.value(cell_lid),
        d_large_moles.value(cell_lid)};
    const auto initial = state;
    const auto production_rate =
        d_options.gas_release_efficiency
      * d_options.hydrogen_yield_mol_per_j
      * std::max(power_density, scalar_type{});
    d_hydrogen_production_rate.set_owned_value(
        cell_lid, production_rate);

    const auto controlling_time =
        std::min(
            d_options.microbubble_lifetime,
            d_options.large_bubble_dissolution_time);
    auto subcycles = static_cast<int>(
        std::ceil(time_step / (0.2 * controlling_time)));
    subcycles = std::clamp(
        subcycles, 1, d_options.max_subcycles);
    d_last_statistics.maximum_subcycles = std::max(
        d_last_statistics.maximum_subcycles, subcycles);
    const auto substep = time_step / subcycles;

    scalar_type converted_number{};
    scalar_type converted_moles{};
    scalar_type large_growth{};
    scalar_type dissolved_moles{};
    const auto critical_concentration =
        RadiolyticGasPhysics::henry_equilibrium_concentration(
            d_options.henry_coefficient,
            properties.pressure,
            properties.surface_tension,
            properties.nucleation_radius);

    for (int cycle = 0; cycle < subcycles; ++cycle)
    {
        const auto produced = production_rate * substep;
        state.micro_moles += produced;
        state.micro_number +=
            produced / properties.nucleation_moles;

        const auto micro_decay_fraction =
            1.0 - std::exp(
                -substep / d_options.microbubble_lifetime);
        const auto micro_decay_moles =
            state.micro_moles * micro_decay_fraction;
        state.micro_moles -= micro_decay_moles;
        state.micro_number *= 1.0 - micro_decay_fraction;
        state.dissolved_inventory += micro_decay_moles;
        dissolved_moles += micro_decay_moles;

        const auto liquid_fraction =
            std::max(
                1.0 - d_alpha_g.value(cell_lid),
                scalar_type{1.0e-15});
        const auto dissolved_concentration =
            concentration(state, liquid_fraction);
        const auto supersaturation =
            dissolved_concentration / critical_concentration - 1.0;
        const auto activation =
            RadiolyticGasPhysics::smoothed_heaviside(
                supersaturation,
                d_options.heaviside_mode,
                d_options.smooth_heaviside_width);
        const auto conversion_frequency =
            d_options.micro_to_large_conversion_coefficient
          * properties.pressure
          * std::max(supersaturation, scalar_type{})
          * activation;
        const auto conversion_fraction =
            1.0 - std::exp(-conversion_frequency * substep);
        const auto number_to_large =
            state.micro_number * conversion_fraction;
        const auto moles_to_large =
            std::min(
                state.micro_moles,
                properties.nucleation_moles * number_to_large);
        state.micro_number -= number_to_large;
        state.micro_moles -= moles_to_large;
        state.large_number += number_to_large;
        state.large_moles += moles_to_large;
        converted_number += number_to_large;
        converted_moles += moles_to_large;

        if (state.large_number <= d_options.min_population
            || state.large_moles <= 0.0)
        {
            continue;
        }
        const auto radius_result =
            RadiolyticGasPhysics::solve_bubble_radius(
                state.large_moles / state.large_number,
                properties.pressure,
                properties.surface_tension,
                d_options.gas_constant,
                properties.temperature,
                d_options.min_radius,
                d_options.max_radius,
                d_options.max_radius_iterations,
                d_options.local_ode_tolerance);
        if (!radius_result.converged)
        {
            ++d_last_statistics.radius_solver_failures;
            continue;
        }

        const auto radius = radius_result.radius;
        const auto equilibrium =
            RadiolyticGasPhysics::henry_equilibrium_concentration(
                d_options.henry_coefficient,
                properties.pressure,
                properties.surface_tension,
                radius);
        const auto relative_speed =
            rise_velocity(
                radius,
                properties.density,
                properties.viscosity,
                properties.surface_tension);
        const auto transfer_coefficient =
            RadiolyticGasPhysics::hughmark_mass_transfer_coefficient(
                properties.diffusivity,
                radius,
                properties.density,
                properties.viscosity,
                relative_speed);
        const auto interfacial_area =
            4.0 * std::numbers::pi * radius * radius
          * state.large_number;
        const auto transfer_rate =
            transfer_coefficient * interfacial_area
          * (dissolved_concentration - equilibrium);
        if (transfer_rate > 0.0)
        {
            const auto transfer =
                std::min(
                    state.dissolved_inventory,
                    transfer_rate * substep);
            state.dissolved_inventory -= transfer;
            state.large_moles += transfer;
            large_growth += transfer;
        }
        else if (transfer_rate < 0.0)
        {
            const auto analytic_decay =
                state.large_moles
              * (1.0 - std::exp(
                    -substep
                    / d_options.large_bubble_dissolution_time));
            const auto transfer = std::min(
                state.large_moles,
                -transfer_rate * substep + analytic_decay);
            state.large_moles -= transfer;
            state.large_number *= std::exp(
                -substep
                / d_options.large_bubble_dissolution_time);
            state.dissolved_inventory += transfer;
            dissolved_moles += transfer;
        }
    }

    d_micro_to_large_number_rate.set_owned_value(
        cell_lid, converted_number / time_step);
    d_micro_to_large_molar_rate.set_owned_value(
        cell_lid, converted_moles / time_step);
    d_large_growth_rate.set_owned_value(
        cell_lid, large_growth / time_step);
    d_dissolution_rate.set_owned_value(
        cell_lid, dissolved_moles / time_step);

    const auto before =
        initial.dissolved_inventory
      + initial.micro_moles + initial.large_moles;
    const auto after =
        state.dissolved_inventory
      + state.micro_moles + state.large_moles;
    d_inventory_error.set_owned_value(
        cell_lid,
        after - before - production_rate * time_step);
    return state;
}

/**
 * @brief Reconstruct radii, void fractions, concentrations, and diagnostics.
 * @tparam Pack Tpetra type pack used by the model.
 * @param temperature Cell temperature field.
 * @param velocity Liquid velocity field.
 * @param material Material-property fields.
 * @throws std::invalid_argument if a correlation input is invalid.
 * @throws std::runtime_error if a derived-property solve fails.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::reconstruct_derived_fields(
    const field_type& temperature,
    const velocity_field_type& velocity,
    const material_type& material)
{
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto properties =
            cell_properties(cell_lid, temperature, velocity, material);
        const auto solve_radius =
            [&](scalar_type number, scalar_type moles)
        {
            if (number <= d_options.min_population || moles <= 0.0)
                return scalar_type{};
            const auto result =
                RadiolyticGasPhysics::solve_bubble_radius(
                    moles / number,
                    properties.pressure,
                    properties.surface_tension,
                    d_options.gas_constant,
                    properties.temperature,
                    d_options.min_radius,
                    d_options.max_radius,
                    d_options.max_radius_iterations,
                    d_options.local_ode_tolerance);
            if (!result.converged)
            {
                ++d_last_statistics.radius_solver_failures;
                return scalar_type{};
            }
            return result.radius;
        };

        const auto micro_radius = solve_radius(
            d_micro_number.value(cell_lid),
            d_micro_moles.value(cell_lid));
        const auto large_radius = solve_radius(
            d_large_number.value(cell_lid),
            d_large_moles.value(cell_lid));
        const auto micro_void =
            RadiolyticGasPhysics::bubble_void_fraction(
                d_micro_number.value(cell_lid), micro_radius);
        const auto large_void =
            RadiolyticGasPhysics::bubble_void_fraction(
                d_large_number.value(cell_lid), large_radius);
        const auto raw_void = micro_void + large_void;
        const auto bounded_void = std::clamp(
            raw_void, d_options.alpha_min, d_options.alpha_max);
        if (bounded_void != raw_void)
            ++d_last_statistics.clipped_cells;
        const auto liquid_fraction = 1.0 - bounded_void;
        const auto raw_concentration =
            d_dissolved_hydrogen_inventory.value(cell_lid)
            / liquid_fraction;
        const auto published_concentration =
            std::min(raw_concentration, d_options.max_concentration);
        const auto characteristic_radius =
            RadiolyticGasPhysics::characteristic_radius(
                d_micro_number.value(cell_lid),
                micro_radius,
                d_large_number.value(cell_lid),
                large_radius);
        const auto equilibrium_radius =
            large_radius > 0.0
                ? large_radius : properties.nucleation_radius;

        d_nucleation_radius.set_owned_value(
            cell_lid, properties.nucleation_radius);
        d_micro_radius.set_owned_value(cell_lid, micro_radius);
        d_large_radius.set_owned_value(cell_lid, large_radius);
        d_alpha_g_micro.set_owned_value(cell_lid, micro_void);
        d_alpha_g_large.set_owned_value(cell_lid, large_void);
        d_alpha_g_raw.set_owned_value(cell_lid, raw_void);
        d_alpha_g.set_owned_value(cell_lid, bounded_void);
        d_alpha_l.set_owned_value(cell_lid, liquid_fraction);
        d_alpha_g_excess.set_owned_value(
            cell_lid, std::max(raw_void - bounded_void, scalar_type{}));
        d_characteristic_radius.set_owned_value(
            cell_lid, characteristic_radius);
        d_dissolved_hydrogen.set_owned_value(
            cell_lid, published_concentration);
        d_excluded_dissolved_inventory.set_owned_value(
            cell_lid,
            std::max(
                d_dissolved_hydrogen_inventory.value(cell_lid)
                  - liquid_fraction * published_concentration,
                scalar_type{}));
        d_critical_concentration.set_owned_value(
            cell_lid,
            RadiolyticGasPhysics::henry_equilibrium_concentration(
                d_options.henry_coefficient,
                properties.pressure,
                properties.surface_tension,
                properties.nucleation_radius));
        d_equilibrium_concentration.set_owned_value(
            cell_lid,
            RadiolyticGasPhysics::henry_equilibrium_concentration(
                d_options.henry_coefficient,
                properties.pressure,
                properties.surface_tension,
                equilibrium_radius));

        if (characteristic_radius > 0.0)
        {
            d_mass_transfer_coefficient.set_owned_value(
                cell_lid,
                RadiolyticGasPhysics::hughmark_mass_transfer_coefficient(
                    properties.diffusivity,
                    characteristic_radius,
                    properties.density,
                    properties.viscosity,
                    rise_velocity(
                        characteristic_radius,
                        properties.density,
                        properties.viscosity,
                        properties.surface_tension)));
        }
        else
        {
            d_mass_transfer_coefficient.set_owned_value(cell_lid, 0.0);
        }
    }
}

/**
 * @brief Advance the full two-population transport and kinetics model.
 * @tparam Pack Tpetra type pack used by the model.
 * @param time_step Positive physical time step.
 * @param temperature Cell temperature field.
 * @param velocity Liquid velocity field.
 * @param liquid_face_flux Oriented liquid volumetric flux.
 * @param material Material-property fields.
 * @param fission_power_density Required fission power-density field.
 * @throws std::invalid_argument if required fields or values are invalid.
 * @throws std::runtime_error if a transport or property solve fails.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::advance_two_population(
    scalar_type time_step,
    const field_type& temperature,
    const velocity_field_type& velocity,
    const face_flux_field_type& liquid_face_flux,
    const material_type& material,
    const field_type* fission_power_density)
{
    if (!fission_power_density)
    {
        throw std::invalid_argument(
            "Sheng 2024 radiolysis requires a fission power source.");
    }

    d_last_statistics.hydrogen_before =
        total_hydrogen_inventory();
    transport_populations(
        time_step,
        temperature,
        velocity,
        liquid_face_flux,
        material);

    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto properties =
            cell_properties(cell_lid, temperature, velocity, material);
        assign_cell_state(
            cell_lid,
            integrate_cell_kinetics(
                cell_lid,
                time_step,
                fission_power_density->value(cell_lid),
                properties));
    }
    d_dissolved_hydrogen_inventory.sync_ghosts();
    d_micro_number.sync_ghosts();
    d_micro_moles.sync_ghosts();
    d_large_number.sync_ghosts();
    d_large_moles.sync_ghosts();

    reconstruct_derived_fields(temperature, velocity, material);
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        d_source_alpha_rad.set_owned_value(
            cell_lid,
            (d_alpha_g.value(cell_lid)
             - d_previous_alpha_g.value(cell_lid))
            / time_step);
    }

    d_last_statistics.hydrogen_produced =
        global_integral(d_hydrogen_production_rate) * time_step;
    d_last_statistics.hydrogen_after =
        total_hydrogen_inventory();
    d_last_statistics.inventory_error =
        d_last_statistics.hydrogen_after
      + d_last_statistics.hydrogen_escaped
      - d_last_statistics.hydrogen_before
      - d_last_statistics.hydrogen_produced;
    d_last_statistics.void_volume = global_integral(d_alpha_g);
}

/**
 * @brief Advance reconstructed absolute pressure with the inertial closure.
 * @tparam Pack Tpetra type pack used by the model.
 * @param time_step Positive physical time step.
 * @param temperature Cell temperature field.
 * @param liquid_face_flux Oriented liquid volumetric flux.
 * @param material Material-property fields.
 * @throws std::invalid_argument if the selected property correlation is invalid.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::update_inertial_pressure(
    scalar_type time_step,
    const field_type& temperature,
    const face_flux_field_type& liquid_face_flux,
    const material_type& material)
{
    if (d_options.pressure_mode != RadiolyticPressureMode::Inertial)
        return;

    if (!d_history_initialized)
    {
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            d_previous_temperature.set_owned_value(
                cell_lid, temperature.value(cell_lid));
            d_previous_density.set_owned_value(
                cell_lid, material.density.value(cell_lid));
            d_previous_alpha_g.set_owned_value(
                cell_lid, d_alpha_g.value(cell_lid));
        }
        d_history_initialized = true;
        return;
    }

    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto pressure = d_absolute_pressure.value(cell_lid);
        const auto temperature_value = temperature.value(cell_lid);
        const auto density = material.density.value(cell_lid);
        const auto alpha = d_alpha_g.value(cell_lid);
        const auto liquid_fraction =
            std::max(1.0 - alpha, scalar_type{1.0e-15});
        const auto radius =
            d_characteristic_radius.value(cell_lid);
        const auto surface_tension =
            d_options.surface_tension_mode
                    == SurfaceTensionMode::Constant
                ? d_options.surface_tension
                : RadiolyticGasPhysics::sheng2024_surface_tension(
                      temperature_value - 273.15,
                      d_options.uranium_concentration_mol_per_m3);
        const auto laplace_term =
            radius > 0.0
                ? 4.0 * surface_tension / (3.0 * radius)
                : 0.0;
        const auto bubble_pressure =
            std::max(pressure + laplace_term,
                     d_options.minimum_absolute_pressure);
        const auto mixture_compressibility =
            d_options.liquid_compressibility * liquid_fraction
          + alpha / bubble_pressure;
        const auto mixture_expansion =
            d_options.liquid_thermal_expansion * liquid_fraction
          + (alpha > 0.0
                ? alpha / temperature_value
                  * (pressure
                     + (radius > 0.0
                            ? 2.0 * surface_tension / radius
                            : 0.0))
                  / bubble_pressure
                : 0.0);
        const auto divergence =
            FVM::cell_flux_balance<Pack>(
                *d_mesh, liquid_face_flux, cell_lid)
            / d_mesh->cell_volume(cell_lid);
        const auto alpha_rate =
            (alpha - d_previous_alpha_g.value(cell_lid))
            / time_step;
        const auto density_rate =
            -density * divergence
          + density * alpha_rate / liquid_fraction;
        const auto temperature_rate =
            (temperature_value
             - d_previous_temperature.value(cell_lid))
            / time_step;
        const auto pressure_rate =
            (density_rate / density
             + mixture_expansion * temperature_rate)
            / mixture_compressibility;
        auto updated_pressure =
            pressure + time_step * pressure_rate;
        if (!std::isfinite(updated_pressure)
            || updated_pressure
                < d_options.minimum_absolute_pressure)
        {
            ++d_last_statistics.pressure_floor_cells;
            updated_pressure = d_options.minimum_absolute_pressure;
        }
        d_absolute_pressure.set_owned_value(
            cell_lid, updated_pressure);
        d_previous_temperature.set_owned_value(
            cell_lid, temperature_value);
        d_previous_density.set_owned_value(cell_lid, density);
        d_previous_alpha_g.set_owned_value(cell_lid, alpha);
    }
}

/**
 * @brief Advance non-ideal radiolysis using the model-owned void fraction.
 * @tparam Pack Tpetra type pack used by the model.
 * @param time Current physical time.
 * @param time_step Positive physical time step.
 * @param temperature Cell temperature field.
 * @param gauge_pressure Cell gauge pressure in Pa.
 * @param velocity Liquid velocity field.
 * @param liquid_face_flux Oriented liquid volumetric flux.
 * @param material Material-property fields.
 * @param fission_power_density Optional source field required when active.
 * @throws std::logic_error if called for ideal-gas source mode.
 * @throws std::invalid_argument if fields or physical inputs are invalid.
 * @throws std::runtime_error if a transport or property solve fails.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::advance(
    scalar_type time,
    scalar_type time_step,
    const field_type& temperature,
    const field_type& gauge_pressure,
    const velocity_field_type& velocity,
    const face_flux_field_type& liquid_face_flux,
    const material_type& material,
    const field_type* fission_power_density)
{
    if (mode() == RadiolyticGasMode::IdealGasSource)
    {
        throw std::logic_error(
            "Ideal radiolysis advance requires authoritative scalar void.");
    }
    advance(
        time,
        time_step,
        temperature,
        gauge_pressure,
        velocity,
        liquid_face_flux,
        material,
        fission_power_density,
        d_alpha_g,
        d_options.alpha_max);
}

/**
 * @brief Advance the active radiolysis mode with authoritative void bounds.
 * @tparam Pack Tpetra type pack used by the model.
 * @param time Current physical time.
 * @param time_step Positive physical time step.
 * @param temperature Cell temperature field.
 * @param gauge_pressure Cell gauge pressure in Pa.
 * @param velocity Liquid velocity field.
 * @param liquid_face_flux Oriented liquid volumetric flux.
 * @param material Material-property fields.
 * @param fission_power_density Optional source field required when active.
 * @param alpha_g Authoritative void fraction for ideal-gas source mode.
 * @param alpha_max Upper bound for authoritative void fraction.
 * @throws std::logic_error if model and authoritative state are incompatible.
 * @throws std::invalid_argument if fields or physical inputs are invalid.
 * @throws std::runtime_error if a transport or property solve fails.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::advance(
    scalar_type time,
    scalar_type time_step,
    const field_type& temperature,
    const field_type& gauge_pressure,
    const velocity_field_type& velocity,
    const face_flux_field_type& liquid_face_flux,
    const material_type& material,
    const field_type* fission_power_density,
    const field_type& alpha_g,
    scalar_type alpha_max)
{
    d_last_statistics = {};
    if (!enabled())
    {
        d_source_alpha_rad.put_scalar(0.0);
        return;
    }
    if (!std::isfinite(time_step) || time_step <= 0.0)
    {
        throw std::invalid_argument(
            "Radiolytic gas timestep must be finite and positive.");
    }

    if (mode() == RadiolyticGasMode::Sheng2024TwoPopulation
        && !d_initial_state_initialized)
    {
        initialize_state(
            time - time_step,
            temperature,
            gauge_pressure,
            velocity,
            material);
    }
    reconstruct_absolute_pressure(time, gauge_pressure);
    if (mode() == RadiolyticGasMode::IdealGasSource)
    {
        synchronize_void_fraction(alpha_g, alpha_max);
        update_ideal_gas_source(
            time_step,
            temperature,
            fission_power_density,
            alpha_g,
            alpha_max);
        reduce_event_statistics();
        return;
    }

    advance_two_population(
        time_step,
        temperature,
        velocity,
        liquid_face_flux,
        material,
        fission_power_density);
    update_inertial_pressure(
        time_step, temperature, liquid_face_flux, material);
    reduce_event_statistics();

    if (d_options.pressure_mode != RadiolyticPressureMode::Inertial)
    {
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            d_previous_temperature.set_owned_value(
                cell_lid, temperature.value(cell_lid));
            d_previous_density.set_owned_value(
                cell_lid, material.density.value(cell_lid));
            d_previous_alpha_g.set_owned_value(
                cell_lid, d_alpha_g.value(cell_lid));
        }
        d_history_initialized = true;
    }
    sync_all_fields();
}

/**
 * @brief Mirror an externally authoritative scalar void-fraction field.
 * @tparam Pack Tpetra type pack used by the model.
 * @param alpha_g Authoritative gas void fraction.
 * @param alpha_max Maximum permitted gas void fraction.
 * @throws std::logic_error unless ideal-gas source mode is active.
 * @throws std::invalid_argument if the field, bound, or values are invalid.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::synchronize_void_fraction(
    const field_type& alpha_g,
    scalar_type alpha_max)
{
    if (mode() != RadiolyticGasMode::IdealGasSource)
    {
        throw std::logic_error(
            "Only ideal radiolysis can mirror authoritative scalar void.");
    }
    if (&alpha_g.mesh() != d_mesh.get())
    {
        throw std::invalid_argument(
            "Radiolytic gas authoritative alpha_g is on the wrong mesh.");
    }
    if (!std::isfinite(alpha_max)
        || alpha_max <= scalar_type{}
        || alpha_max > scalar_type{1})
    {
        throw std::invalid_argument(
            "Radiolytic gas authoritative alpha_max must be in (0, 1].");
    }
    for (size_t owned = 0;
         owned < d_mesh->num_owned_cells();
         ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto alpha = alpha_g.value(cell_lid);
        if (!std::isfinite(alpha)
            || alpha < scalar_type{}
            || alpha > alpha_max)
        {
            throw std::invalid_argument(
                "Radiolytic gas authoritative alpha_g is outside its bounds.");
        }
        d_alpha_g.set_owned_value(cell_lid, alpha);
        d_alpha_l.set_owned_value(cell_lid, scalar_type{1} - alpha);
    }
    d_alpha_g.sync_ghosts();
    d_alpha_l.sync_ghosts();
}

/**
 * @brief Synchronize overlap storage for every model-owned cell field.
 * @tparam Pack Tpetra type pack used by the model.
 */
template<TpetraTypePack Pack, class MeshType>
void RadiolyticGasModel<Pack, MeshType>::sync_all_fields()
{
    field_type* fields[] = {
        &d_alpha_g,
        &d_alpha_l,
        &d_source_alpha_rad,
        &d_absolute_pressure,
        &d_previous_temperature,
        &d_previous_density,
        &d_previous_alpha_g,
        &d_dissolved_hydrogen,
        &d_dissolved_hydrogen_inventory,
        &d_excluded_dissolved_inventory,
        &d_micro_number,
        &d_micro_moles,
        &d_large_number,
        &d_large_moles,
        &d_nucleation_radius,
        &d_micro_radius,
        &d_large_radius,
        &d_critical_concentration,
        &d_equilibrium_concentration,
        &d_mass_transfer_coefficient,
        &d_alpha_g_micro,
        &d_alpha_g_large,
        &d_alpha_g_raw,
        &d_alpha_g_excess,
        &d_characteristic_radius,
        &d_hydrogen_production_rate,
        &d_micro_to_large_number_rate,
        &d_micro_to_large_molar_rate,
        &d_large_growth_rate,
        &d_dissolution_rate,
        &d_escape_molar_rate,
        &d_escape_number_rate,
        &d_inventory_error};
    for (auto* field : fields)
        field->sync_ghosts();
}

} // namespace SimpleFluid
