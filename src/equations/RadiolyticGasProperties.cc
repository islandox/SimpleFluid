/**
 * @file RadiolyticGasProperties.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Implements radiolytic gas option parsing and property calculations.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "equations/RadiolyticGasProperties.hh"

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

/**
 * @brief Parse the configured radiolytic gas model.
 * @param value Model name.
 * @return Parsed model mode.
 * @throws std::invalid_argument if @p value is unknown.
 */
RadiolyticGasMode parse_mode(std::string value)
{
    value = normalized(std::move(value));
    if (value == "disabled")
        return RadiolyticGasMode::Disabled;
    if (value == "idealgassource" || value == "ideal_gas_source")
        return RadiolyticGasMode::IdealGasSource;
    if (value == "sheng2024twopopulation"
        || value == "sheng2024_two_population")
    {
        return RadiolyticGasMode::Sheng2024TwoPopulation;
    }
    throw std::invalid_argument(
        "Unknown radiolytic_bubble_model; expected disabled, "
        "idealGasSource, or sheng2024TwoPopulation.");
}

/**
 * @brief Parse the absolute-pressure closure used by the gas model.
 * @param value Pressure-mode name.
 * @return Parsed pressure mode.
 * @throws std::invalid_argument if @p value is unknown.
 */
RadiolyticPressureMode parse_pressure_mode(std::string value)
{
    value = normalized(std::move(value));
    if (value == "constant")
        return RadiolyticPressureMode::Constant;
    if (value == "prescribedhistory" || value == "prescribed_history")
        return RadiolyticPressureMode::PrescribedHistory;
    if (value == "reconstructed")
        return RadiolyticPressureMode::Reconstructed;
    if (value == "inertial")
        return RadiolyticPressureMode::Inertial;
    throw std::invalid_argument(
        "Unknown radiolytic_pressure_mode; expected constant, "
        "prescribedHistory, reconstructed, or inertial.");
}

/**
 * @brief Parse the dissolved-hydrogen transport mode.
 * @param value Transport-mode name.
 * @return Parsed dissolved transport mode.
 * @throws std::invalid_argument if @p value is unknown.
 */
RadiolyticTransportMode parse_dissolved_transport(std::string value)
{
    value = normalized(std::move(value));
    if (value == "noadvection" || value == "no_advection")
        return RadiolyticTransportMode::NoAdvection;
    if (value == "advective")
        return RadiolyticTransportMode::Advective;
    throw std::invalid_argument(
        "Unknown dissolved_hydrogen_transport_mode; expected "
        "noAdvection or advective.");
}

/**
 * @brief Parse the bubble transport direction model.
 * @param value Transport-mode name.
 * @return Parsed bubble transport mode.
 * @throws std::invalid_argument if @p value is unknown.
 */
BubbleTransportMode parse_bubble_transport(std::string value)
{
    value = normalized(std::move(value));
    if (value == "general")
        return BubbleTransportMode::General;
    if (value == "axial")
        return BubbleTransportMode::Axial;
    throw std::invalid_argument(
        "Unknown bubble_transport_mode; expected general or axial.");
}

/**
 * @brief Parse the switching function used by population kinetics.
 * @param value Heaviside-mode name.
 * @return Parsed Heaviside mode.
 * @throws std::invalid_argument if @p value is unknown.
 */
RadiolyticHeavisideMode parse_heaviside(std::string value)
{
    value = normalized(std::move(value));
    if (value == "exact")
        return RadiolyticHeavisideMode::Exact;
    if (value == "smoothed")
        return RadiolyticHeavisideMode::Smoothed;
    throw std::invalid_argument(
        "Unknown radiolytic_heaviside_mode; expected exact or smoothed.");
}

/**
 * @brief Parse the bubble rise-velocity correlation.
 * @param value Correlation name.
 * @return Parsed rise-velocity mode.
 * @throws std::invalid_argument if @p value is unknown.
 */
BubbleRiseVelocityMode parse_rise_velocity(std::string value)
{
    value = normalized(std::move(value));
    if (value == "zeroslip" || value == "zero_slip")
        return BubbleRiseVelocityMode::ZeroSlip;
    if (value == "constantslip" || value == "constant_slip")
        return BubbleRiseVelocityMode::ConstantSlip;
    if (value == "celata2007")
        return BubbleRiseVelocityMode::Celata2007;
    throw std::invalid_argument(
        "Unknown bubble_rise_velocity_model; expected zeroSlip, "
        "constantSlip, or celata2007.");
}

/**
 * @brief Parse the surface-tension property model.
 * @param value Property-model name.
 * @return Parsed surface-tension mode.
 * @throws std::invalid_argument if @p value is unknown.
 */
SurfaceTensionMode parse_surface_tension(std::string value)
{
    value = normalized(std::move(value));
    if (value == "constant")
        return SurfaceTensionMode::Constant;
    if (value == "sheng2024")
        return SurfaceTensionMode::Sheng2024;
    throw std::invalid_argument(
        "Unknown surface_tension_model; expected constant or sheng2024.");
}

/**
 * @brief Parse the dissolved-hydrogen diffusivity model.
 * @param value Property-model name.
 * @return Parsed diffusivity mode.
 * @throws std::invalid_argument if @p value is unknown.
 */
HydrogenDiffusivityMode parse_diffusivity(std::string value)
{
    value = normalized(std::move(value));
    if (value == "constant")
        return HydrogenDiffusivityMode::Constant;
    if (value == "sheng2024")
        return HydrogenDiffusivityMode::Sheng2024;
    throw std::invalid_argument(
        "Unknown hydrogen_diffusivity_model; expected constant or "
        "sheng2024.");
}

/**
 * @brief Validate a prescribed absolute-pressure history.
 * @param options Options containing the history arrays and pressure floor.
 * @throws std::invalid_argument if the arrays are inconsistent or invalid.
 */
void require_history(const RadiolyticGasOptions& options)
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

/**
 * @brief Parse and validate radiolytic gas options from a database.
 * @param database Source configuration database.
 * @return Validated radiolytic gas options.
 * @throws std::invalid_argument if an option is ill-typed or invalid.
 */
RadiolyticGasOptions radiolytic_gas_options_from_database(
    const Database& database)
{
    RadiolyticGasOptions options;
    const detail::DatabaseOptionReader reader(
        database, "Radiolytic gas model");
    const auto enabled =
        reader.value_or<bool>("enable_radiolysis", false);
    if (reader.contains("radiolytic_bubble_model"))
    {
        options.mode = parse_mode(reader.required<std::string>(
            "radiolytic_bubble_model"));
    }
    else if (enabled)
    {
        options.mode = RadiolyticGasMode::IdealGasSource;
    }

    options.pressure_mode = parse_pressure_mode(reader.value_or<std::string>(
        "radiolytic_pressure_mode", "constant"));
    options.dissolved_transport =
        parse_dissolved_transport(reader.value_or<std::string>(
            "dissolved_hydrogen_transport_mode",
            "noAdvection"));
    options.bubble_transport = parse_bubble_transport(
        reader.value_or<std::string>(
            "bubble_transport_mode", "general"));
    options.heaviside_mode = parse_heaviside(reader.value_or<std::string>(
        "radiolytic_heaviside_mode", "exact"));
    options.rise_velocity_mode =
        parse_rise_velocity(reader.value_or<std::string>(
            "bubble_rise_velocity_model", "zeroSlip"));
    options.surface_tension_mode =
        parse_surface_tension(reader.value_or<std::string>(
            "surface_tension_model", "constant"));
    options.diffusivity_mode = parse_diffusivity(reader.value_or<std::string>(
        "hydrogen_diffusivity_model", "constant"));

#define SIMPLEFLUID_RADIOLYTIC_REAL(member, key) \
    options.member = reader.value_or<real_t>(key, options.member)
    SIMPLEFLUID_RADIOLYTIC_REAL(
        hydrogen_yield_mol_per_j, "hydrogen_yield_mol_per_j");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        gas_release_efficiency, "gas_release_efficiency");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        reference_pressure, "reference_pressure");
    SIMPLEFLUID_RADIOLYTIC_REAL(gas_constant, "gas_constant");
    SIMPLEFLUID_RADIOLYTIC_REAL(alpha_min, "alpha_min");
    SIMPLEFLUID_RADIOLYTIC_REAL(alpha_max, "alpha_max");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        max_source_alpha_rate, "max_source_alpha_rate");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        henry_coefficient, "henry_coefficient");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        surface_tension, "surface_tension");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        hydrogen_diffusivity, "hydrogen_diffusivity");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        atmospheric_pressure, "atmospheric_pressure");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        uranium_concentration_mol_per_m3,
        "uranium_concentration_mol_per_m3");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        hydrogen_yield_molecules_per_100_ev,
        "hydrogen_yield_molecules_per_100_ev");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        microbubble_lifetime, "microbubble_lifetime");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        large_bubble_dissolution_time,
        "large_bubble_dissolution_time");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        micro_to_large_conversion_coefficient,
        "micro_to_large_conversion_coefficient");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        smooth_heaviside_width, "smooth_heaviside_width");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        constant_slip_velocity, "constant_slip_velocity");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        bubble_gas_density, "bubble_gas_density");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        bubble_gravity, "bubble_gravity");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        rise_velocity_tolerance, "rise_velocity_tolerance");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        initial_dissolved_hydrogen,
        "initial_dissolved_hydrogen");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        initial_micro_number_density,
        "initial_micro_number_density");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        initial_micro_moles, "initial_micro_moles");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        initial_large_number_density,
        "initial_large_number_density");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        initial_large_moles, "initial_large_moles");
    SIMPLEFLUID_RADIOLYTIC_REAL(min_radius, "min_radius");
    SIMPLEFLUID_RADIOLYTIC_REAL(max_radius, "max_radius");
    SIMPLEFLUID_RADIOLYTIC_REAL(min_population, "min_population");
    SIMPLEFLUID_RADIOLYTIC_REAL(max_population, "max_population");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        max_concentration, "max_concentration");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        local_ode_tolerance, "local_ode_tolerance");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        liquid_compressibility, "liquid_compressibility");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        liquid_thermal_expansion, "liquid_thermal_expansion");
    SIMPLEFLUID_RADIOLYTIC_REAL(
        minimum_absolute_pressure, "minimum_absolute_pressure");
#undef SIMPLEFLUID_RADIOLYTIC_REAL

    options.max_subcycles =
        reader.value_or<int>(
            "max_radiolytic_subcycles", options.max_subcycles);
    options.max_radius_iterations =
        reader.value_or<int>(
            "max_radius_iterations", options.max_radius_iterations);
    options.max_rise_velocity_iterations =
        reader.value_or<int>(
            "max_rise_velocity_iterations",
            options.max_rise_velocity_iterations);
    options.pressure_history_times =
        reader.value_or<ArrReal>(
            "pressure_history_times",
            options.pressure_history_times);
    options.pressure_history_values =
        reader.value_or<ArrReal>(
            "pressure_history_values",
            options.pressure_history_values);
    options.free_surface_patches =
        reader.value_or<ArrString>(
            "radiolytic_free_surface_patches",
            options.free_surface_patches);

    validate_radiolytic_gas_options(options);
    return options;
}

/**
 * @brief Validate radiolytic gas options for the selected model modes.
 * @param options Options to validate.
 * @throws std::invalid_argument if any active option is inconsistent.
 */
void validate_radiolytic_gas_options(
    const RadiolyticGasOptions& options)
{
    using namespace RadiolyticGasPhysics;
    require_positive(options.reference_pressure, "reference pressure");
    require_positive(options.gas_constant, "gas constant");
    require_non_negative(options.alpha_min, "minimum void fraction");
    require_non_negative(options.alpha_max, "maximum void fraction");
    if (options.alpha_max <= options.alpha_min
        || options.alpha_max >= 1.0)
    {
        throw std::invalid_argument(
            "Radiolytic alpha bounds require 0 <= alpha_min < "
            "alpha_max < 1.");
    }
    if (options.mode == RadiolyticGasMode::Disabled)
        return;

    require_positive(
        options.hydrogen_yield_mol_per_j,
        "hydrogen yield");
    require_non_negative(
        options.gas_release_efficiency,
        "gas release efficiency");
    if (options.gas_release_efficiency > 1.0)
    {
        throw std::invalid_argument(
            "Gas release efficiency cannot exceed one.");
    }
    if (!std::isfinite(options.max_source_alpha_rate)
        || options.max_source_alpha_rate < 0.0)
    {
        throw std::invalid_argument(
            "Maximum source alpha rate must be explicitly finite and "
            "non-negative.");
    }
    require_positive(
        options.minimum_absolute_pressure,
        "minimum absolute pressure");

    if (options.pressure_mode
        == RadiolyticPressureMode::PrescribedHistory)
    {
        require_history(options);
    }
    if (options.mode == RadiolyticGasMode::IdealGasSource)
        return;

    require_positive(options.henry_coefficient, "Henry coefficient");
    if (options.surface_tension_mode == SurfaceTensionMode::Constant)
        require_positive(options.surface_tension, "surface tension");
    if (options.diffusivity_mode == HydrogenDiffusivityMode::Constant)
        require_positive(
            options.hydrogen_diffusivity,
            "hydrogen diffusivity");
    require_positive(
        options.atmospheric_pressure,
        "atmospheric pressure");
    require_positive(
        options.uranium_concentration_mol_per_m3,
        "uranyl nitrate concentration");
    require_positive(
        options.hydrogen_yield_molecules_per_100_ev,
        "hydrogen yield in molecules per 100 eV");
    if (options.hydrogen_yield_molecules_per_100_ev <= 0.5
        || options.hydrogen_yield_molecules_per_100_ev >= 4.5)
    {
        throw std::invalid_argument(
            "Winter's nucleation-radius yield correction requires "
            "0.5 < G_H2 < 4.5 molecules per 100 eV.");
    }
    require_positive(
        options.microbubble_lifetime,
        "microbubble lifetime");
    require_positive(
        options.large_bubble_dissolution_time,
        "large-bubble dissolution time");
    require_non_negative(
        options.micro_to_large_conversion_coefficient,
        "micro-to-large conversion coefficient");
    if (options.heaviside_mode == RadiolyticHeavisideMode::Smoothed)
        require_positive(
            options.smooth_heaviside_width,
            "smoothed Heaviside width");
    require_non_negative(
        options.constant_slip_velocity,
        "constant slip velocity");
    require_non_negative(
        options.initial_dissolved_hydrogen,
        "initial dissolved hydrogen");
    require_non_negative(
        options.initial_micro_number_density,
        "initial microbubble number density");
    require_non_negative(
        options.initial_micro_moles,
        "initial microbubble inventory");
    require_non_negative(
        options.initial_large_number_density,
        "initial large-bubble number density");
    require_non_negative(
        options.initial_large_moles,
        "initial large-bubble inventory");
    require_positive(options.min_radius, "minimum radius");
    require_positive(options.max_radius, "maximum radius");
    if (options.max_radius <= options.min_radius)
        throw std::invalid_argument(
            "Maximum bubble radius must exceed minimum radius.");
    require_positive(options.min_population, "minimum population");
    require_positive(options.max_population, "maximum population");
    if (options.max_population <= options.min_population)
        throw std::invalid_argument(
            "Maximum population must exceed minimum population.");
    require_positive(
        options.max_concentration,
        "maximum hydrogen concentration");
    require_positive(
        options.local_ode_tolerance,
        "local ODE tolerance");
    if (options.max_subcycles < 1
        || options.max_radius_iterations < 1
        || options.max_rise_velocity_iterations < 1)
    {
        throw std::invalid_argument(
            "Radiolytic iteration limits must be positive.");
    }
    require_positive(
        options.liquid_compressibility,
        "liquid compressibility");
    require_non_negative(
        options.liquid_thermal_expansion,
        "liquid thermal expansion");

    if (options.rise_velocity_mode
        == BubbleRiseVelocityMode::Celata2007)
    {
        require_non_negative(
            options.bubble_gas_density,
            "bubble gas density");
        require_positive(options.bubble_gravity, "bubble gravity");
        require_positive(
            options.rise_velocity_tolerance,
            "rise-velocity tolerance");
    }
}

} // namespace SimpleFluid
