/**
 * @file ReactingGasProperties.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Implements reacting gas option parsing and property calculations.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "equations/ReactingGasProperties.hh"
#include "equations/BoussFiniteVapor.hh"

#include "dataclass/DatabaseOptionReader.hh"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace SimpleFluid
{
namespace
{

/**
 * @brief Convert an option token to lower case for case-insensitive parsing.
 * @param value Token to normalize.
 * @return Lower-case copy of @p value.
 */
std::string normalized(std::string value)
{
    std::ranges::transform(
        value,
        value.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

void require_history(const ReactingGasOptions& options)
{
    const auto size = options.pressure_history_times.size();
    if (size < 2 || size != options.pressure_history_values.size())
    {
        throw std::invalid_argument(
            "Prescribed pressure history requires matching time and "
            "pressure arrays with at least two entries.");
    }
    for (size_t index = 0; index < size; ++index)
    {
        if (!std::isfinite(options.pressure_history_times[index])
            || !std::isfinite(options.pressure_history_values[index])
            || options.pressure_history_values[index]
                < options.minimum_absolute_pressure)
        {
            throw std::invalid_argument(
                "Prescribed pressure history must contain finite times "
                "and positive absolute pressures.");
        }
        if (index > 0
            && options.pressure_history_times[index]
                <= options.pressure_history_times[index - 1])
        {
            throw std::invalid_argument(
                "Prescribed pressure history times must be strictly "
                "increasing.");
        }
    }
}

} // namespace

ReactingGasOptions reacting_gas_options_from_database(const Database& database)
{
    // Flat configuration has no chemistry catalog/provider factory. Enabling it
    // must fail explicitly rather than manufacture an unnamed source/material.
    for (const auto* key : {"enable_radiolysis", "radiolytic_bubble_model", "radiolytic_pressure_mode",
             "radiolytic_heaviside_mode", "radiolytic_kinetics_mode", "radiolytic_transport_solver_tolerance",
             "max_radiolytic_subcycles", "radiolytic_free_surface_patches", "precursor_group_count",
             "precursor_decay_constants", "precursor_initial_concentrations", "precursor_source_terms",
             "precursor_power_yields", "precursor_effective_diffusivity", "reacting_bubble_model",
             "dissolved_hydrogen_transport_mode", "reacting_heaviside_mode", "reacting_kinetics_mode",
             "hydrogen_diffusivity_model", "hydrogen_yield_mol_per_j", "gas_release_efficiency",
             "max_source_alpha_rate", "henry_coefficient", "hydrogen_diffusivity", "microbubble_lifetime",
             "large_bubble_dissolution_time", "micro_to_large_conversion_coefficient", "smooth_heaviside_width",
             "bubble_gas_density", "initial_dissolved_hydrogen", "initial_micro_number_density", "initial_micro_moles",
             "initial_large_number_density", "initial_large_moles", "min_population", "max_population",
             "max_concentration", "local_ode_tolerance", "liquid_compressibility", "liquid_thermal_expansion",
             "uranium_concentration_mol_per_m3", "hydrogen_yield_molecules_per_100_ev", "atmospheric_pressure",
             "nucleation_radius", "nucleation_radius_model", "gas_constant", "max_radius_iterations"})
        if (database.contains(key))
            throw std::invalid_argument(
                std::string(key) +
                " is retired; supply explicit schema-owned ReactingGasOptions and actual inventories.");
    const detail::DatabaseOptionReader reader(database, "Reacting species");
    ReactingGasOptions options;
    const auto mode = normalized(reader.value_or<std::string>("reacting_species_model", "disabled"));
    if (mode != "disabled" && mode != "species")
        throw std::invalid_argument("reacting_species_model requires disabled or species; old modes are retired.");
    if (reader.value_or<bool>("enable_reacting_gas", false) || mode == "species")
        throw std::invalid_argument("Database enable requires an explicit C++ ReactionMechanism, EOS/reference "
                                    "binding and actual species/count inventories; no source default exists.");
    const auto pressure = normalized(reader.value_or<std::string>("reacting_pressure_mode", "constant"));
    if (pressure == "constant")
        options.pressure_mode = ReactingPressureMode::Constant;
    else if (pressure == "reconstructed")
        options.pressure_mode = ReactingPressureMode::Reconstructed;
    else if (pressure == "prescribedhistory" || pressure == "prescribed_history")
        options.pressure_mode = ReactingPressureMode::PrescribedHistory;
    else
        throw std::invalid_argument("Unsupported reacting pressure mode; inertial is retired.");
    const auto transport = normalized(reader.value_or<std::string>("solution_transport_mode", "noAdvection"));
    if (transport == "noadvection" || transport == "no_advection")
        options.solution_transport = ReactingTransportMode::NoAdvection;
    else if (transport == "advective")
        options.solution_transport = ReactingTransportMode::Advective;
    else
        throw std::invalid_argument("solution_transport_mode requires noAdvection or advective.");
    const auto bubble = normalized(reader.value_or<std::string>("bubble_transport_mode", "general"));
    if (bubble != "general")
        throw std::invalid_argument("Schema-owned bubble transport requires general; axial is retired.");
    const auto surface = normalized(reader.value_or<std::string>("surface_tension_model", "constant"));
    if (surface != "constant")
        throw std::invalid_argument("Database surface tension requires constant; supplied laws belong to C++ options.");
    const auto rise = normalized(reader.value_or<std::string>("bubble_rise_velocity_model", "zeroSlip"));
    if (rise == "zeroslip" || rise == "zero_slip")
        options.rise_velocity_mode = BubbleRiseVelocityMode::ZeroSlip;
    else if (rise == "constantslip" || rise == "constant_slip")
        options.rise_velocity_mode = BubbleRiseVelocityMode::ConstantSlip;
    else if (rise == "celata2007")
        options.rise_velocity_mode = BubbleRiseVelocityMode::Celata2007;
    else
        throw std::invalid_argument("Unknown bubble rise velocity law.");
    const auto transfer = normalized(reader.value_or<std::string>("bubble_mass_transfer_model", "hughmark"));
    if (transfer == "hughmark")
        options.mass_transfer_mode = BubbleMassTransferMode::Hughmark;
    else if (transfer == "sphericaldiffusion" || transfer == "spherical_diffusion")
        options.mass_transfer_mode = BubbleMassTransferMode::SphericalDiffusion;
    else if (transfer == "fengmichaelidesclean" || transfer == "feng_michaelides_clean")
        options.mass_transfer_mode = BubbleMassTransferMode::FengMichaelidesClean;
    else if (transfer == "fengmichaelidesrigid" || transfer == "feng_michaelides_rigid")
        options.mass_transfer_mode = BubbleMassTransferMode::FengMichaelidesRigid;
    else
        throw std::invalid_argument("Unknown/retired bubble transfer law; use a neutral named correlation.");
#define READ_REAL(member, key) options.member = reader.value_or<real_t>(key, options.member)
    READ_REAL(reference_pressure, "reference_pressure");
    READ_REAL(alpha_min, "alpha_min");
    READ_REAL(alpha_max, "alpha_max");
    READ_REAL(minimum_absolute_pressure, "minimum_absolute_pressure");
    READ_REAL(transport_solver_tolerance, "reacting_transport_solver_tolerance");
    READ_REAL(surface_tension, "surface_tension");
    READ_REAL(constant_slip_velocity, "constant_slip_velocity");
    READ_REAL(min_radius, "min_radius");
    READ_REAL(max_radius, "max_radius");
    READ_REAL(bubble_gravity, "bubble_gravity");
    READ_REAL(rise_velocity_tolerance, "rise_velocity_tolerance");
#undef READ_REAL
    options.max_subcycles = reader.value_or<int>("max_reacting_subcycles", options.max_subcycles);
    options.max_rise_velocity_iterations =
        reader.value_or<int>("max_rise_velocity_iterations", options.max_rise_velocity_iterations);
    options.free_surface_patches =
        reader.value_or<ArrString>("reacting_free_surface_patches", options.free_surface_patches);
    options.pressure_history_times = reader.value_or<ArrReal>("pressure_history_times", {});
    options.pressure_history_values = reader.value_or<ArrReal>("pressure_history_values", {});
    BubbleGasPhysics::require_positive(options.transport_solver_tolerance, "reacting transport solver tolerance");
    BubbleGasPhysics::require_positive(options.minimum_absolute_pressure, "minimum absolute pressure");
    // Disabled flat setup retains these explicit public controls and their
    // domains; it must not silently discard a malformed future configuration.
    BubbleGasPhysics::require_non_negative(options.surface_tension, "surface tension");
    BubbleGasPhysics::require_non_negative(options.constant_slip_velocity, "slip velocity");
    BubbleGasPhysics::require_positive(options.min_radius, "minimum radius");
    BubbleGasPhysics::require_positive(options.max_radius, "maximum radius");
    BubbleGasPhysics::require_non_negative(options.bubble_gravity, "bubble gravity");
    BubbleGasPhysics::require_positive(options.rise_velocity_tolerance, "rise-velocity tolerance");
    if (options.max_radius <= options.min_radius || options.max_subcycles <= 0 ||
        options.max_rise_velocity_iterations <= 0)
        throw std::invalid_argument("Flat reacting radius bounds/subcycle/rise-iteration limits are invalid.");
    if (options.rise_velocity_mode == BubbleRiseVelocityMode::Celata2007)
        BubbleGasPhysics::require_positive(options.bubble_gravity, "bubble gravity");
    if (options.pressure_mode == ReactingPressureMode::PrescribedHistory)
        require_history(options);
    validate_reacting_gas_options(options);
    return options;
}

void validate_reacting_gas_options(const ReactingGasOptions& options)
{
    using namespace BubbleGasPhysics;
    if ((options.bouss_reference || options.bouss_sources) && !options.reaction_mechanism)
        throw std::invalid_argument("Bouss finite material sources require an explicit canonical reaction schema.");
    if (options.source_schedule != ReactingSourceSchedule::BeforeTemperature &&
        options.source_schedule != ReactingSourceSchedule::PrescribedCarrierAfterTemperature)
        throw std::invalid_argument("Unknown reacting source schedule.");
    if (options.source_schedule == ReactingSourceSchedule::PrescribedCarrierAfterTemperature &&
        !options.reaction_mechanism)
        throw std::invalid_argument("Post-temperature reference scheduling requires an explicit species mechanism.");
    if (options.source_schedule == ReactingSourceSchedule::PrescribedCarrierAfterTemperature &&
        (options.bouss_reference || options.bouss_sources))
        throw std::invalid_argument("Post-temperature reference scheduling cannot own finite Bouss source/heat laws.");
    require_positive(options.reference_pressure, "reference pressure");
    require_non_negative(options.alpha_min, "minimum void fraction");
    require_non_negative(options.alpha_max, "maximum void fraction");
    if (options.alpha_max <= options.alpha_min
        || options.alpha_max >= 1.0)
    {
        throw std::invalid_argument(
            "Reacting alpha bounds require 0 <= alpha_min < "
            "alpha_max < 1.");
    }
    if (options.mode != ReactingGasMode::Disabled && options.mode != ReactingGasMode::Species)
        throw std::invalid_argument("Unknown reacting species mode.");
    if (options.pressure_mode != ReactingPressureMode::Constant &&
        options.pressure_mode != ReactingPressureMode::PrescribedHistory &&
        options.pressure_mode != ReactingPressureMode::Reconstructed)
        throw std::invalid_argument("Unknown/retired reacting pressure mode; inertial is unsupported.");
    if (options.mode == ReactingGasMode::Disabled && !options.reaction_mechanism)
        return;

    require_positive(options.transport_solver_tolerance, "reacting transport solver tolerance");

    if (options.reaction_mechanism)
    {
        const auto& catalog = options.reaction_mechanism->catalog();
        if (options.bulk_diffusion != ReactingBulkDiffusionMode::Disabled)
            throw std::invalid_argument("Typed bulk species diffusion requires finite counterflux and canonical face enthalpy; this field backend currently rejects it.");
        if ((options.pressure_mode != ReactingPressureMode::Constant &&
                options.pressure_mode != ReactingPressureMode::PrescribedHistory &&
                options.pressure_mode != ReactingPressureMode::Reconstructed) ||
            (options.solution_transport != ReactingTransportMode::NoAdvection &&
                options.solution_transport != ReactingTransportMode::Advective) ||
            options.bubble_transport != BubbleTransportMode::General ||
            (options.rise_velocity_mode != BubbleRiseVelocityMode::ZeroSlip &&
                options.rise_velocity_mode != BubbleRiseVelocityMode::ConstantSlip &&
                options.rise_velocity_mode != BubbleRiseVelocityMode::Celata2007) ||
            (options.surface_tension_mode != SurfaceTensionMode::Constant &&
                options.surface_tension_mode != SurfaceTensionMode::External) ||
            (options.surface_tension_mode == SurfaceTensionMode::External && !options.surface_tension_correlation))
            throw std::invalid_argument(
                "General reacting transport/process selection is invalid or lacks its callback.");
        (void) select_bubble_mass_transfer_evaluator(options.mass_transfer_mode);
        if (options.bouss_reference)
        {
            if (!options.bouss_sources || options.mode != ReactingGasMode::Species || options.alpha_min != 0 ||
                options.gas_thermodynamics || options.liquid_thermodynamics || options.population_process ||
                options.surface_tension_mode != SurfaceTensionMode::Constant || options.surface_tension != 0 ||
                options.rise_velocity_mode == BubbleRiseVelocityMode::Celata2007 ||
                !options.free_surface_patches.empty())
                throw std::invalid_argument(
                    "Bouss reference volumes require a paired source binding, sigma=0, "
                    "prescribed/zero slip, closed boundaries and no supplied phase EOS/general EOS process; use "
                    "the distinct Bouss populationProcess callback.");
            Reacting::validate_bouss_sources(
                *options.reaction_mechanism, *options.bouss_reference, *options.bouss_sources);
            for (const auto& property : options.species_properties)
                if (property.diffusivity != 0 || property.diffusivity_provider || property.henry_coefficient != 0 ||
                    property.equilibrium_concentration)
                    throw std::invalid_argument(
                        "Bouss reference-volume binding does not supply EOS film/fugacity laws.");
            if (options.bouss_sources->vapor &&
                options.bouss_sources->vapor->vaporPhase == Reacting::SpeciesPhase::Dispersed &&
                std::ranges::find(options.population_ids, options.bouss_sources->vapor->population) ==
                    options.population_ids.end())
                throw std::invalid_argument("Bouss vapor population must be an explicit configured identity.");
        }
        else if (options.bouss_sources)
            throw std::invalid_argument("Bouss sources require their explicit reference-volume/thermal owner.");
        else if (options.mode != ReactingGasMode::Species || !options.gas_thermodynamics ||
                 !options.liquid_thermodynamics || options.gas_thermodynamics->kind() != Reacting::EOSKind::Gas ||
                 options.liquid_thermodynamics->kind() != Reacting::EOSKind::Liquid ||
                 options.gas_thermodynamics->catalog().fingerprint() != catalog.fingerprint() ||
                 options.liquid_thermodynamics->catalog().fingerprint() != catalog.fingerprint() ||
                 options.gas_thermodynamics->energy_convention() != options.reaction_mechanism->energy_convention() ||
                 options.liquid_thermodynamics->energy_convention() != options.reaction_mechanism->energy_convention())
            throw std::invalid_argument(
                "General reacting fields require matching species/energy schemas and supplied liquid/gas EOS.");
        if (options.pressure_mode == ReactingPressureMode::PrescribedHistory)
            require_history(options);
        require_positive(options.minimum_absolute_pressure, "minimum absolute pressure");
        require_positive(options.min_radius, "minimum radius");
        require_positive(options.max_radius, "maximum radius");
        if (options.max_radius <= options.min_radius || options.max_subcycles <= 0)
            throw std::invalid_argument("General population radius bounds/subcycle limit are invalid.");
        require_non_negative(options.surface_tension, "surface tension");
        require_non_negative(options.constant_slip_velocity, "slip velocity");
        require_non_negative(options.micro_dissolution_frequency, "micro dissolution frequency");
        require_non_negative(options.large_dissolution_frequency, "large dissolution frequency");
        require_non_negative(options.micro_conversion_frequency, "micro conversion frequency");
        std::vector<Reacting::SpeciesId> ids;
        for (const auto& property : options.species_properties)
        {
            (void) catalog.index(property.species);
            if (std::ranges::find(ids, property.species) != ids.end())
                throw std::invalid_argument("Species properties must have unique stable species IDs.");
            ids.push_back(property.species);
            require_non_negative(property.diffusivity, "species diffusivity");
            require_non_negative(property.henry_coefficient, "species Henry coefficient");
        }
        if (options.source_schedule == ReactingSourceSchedule::PrescribedCarrierAfterTemperature)
        {
            if (options.micro_dissolution_frequency != 0. || options.large_dissolution_frequency != 0. ||
                options.micro_conversion_frequency != 0.)
                throw std::invalid_argument("Post-temperature gas references reject implicit interphase processes.");
            for (const auto& property : options.species_properties)
                if (property.diffusivity != 0. || property.diffusivity_provider || property.henry_coefficient != 0. ||
                    property.equilibrium_concentration)
                    throw std::invalid_argument("Post-temperature gas references reject finite-liquid film laws.");
        }
        std::vector<std::string> populations;
        for (const auto& id : options.population_ids)
        {
            if (id.empty() || std::ranges::find(populations, id) != populations.end())
                throw std::invalid_argument("Population IDs must be nonempty and unique.");
            populations.push_back(id);
        }
        return;
    }
    throw std::invalid_argument(
        "Enabled reacting species require an explicit ReactionMechanism and actual inventories.");
}

} // namespace SimpleFluid
