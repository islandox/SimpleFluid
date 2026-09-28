/**
 * @file RadiolyticGasPhysics.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Pure validation and property correlations for radiolytic gas models.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "equations/RadiolyticGasProperties.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid::RadiolyticGasPhysics
{

real_t require_positive(real_t value, std::string_view label)
{
    if (!std::isfinite(value) || value <= 0.0)
    {
        throw std::invalid_argument(std::string(label) + " must be finite and positive.");
    }
    return value;
}

real_t require_non_negative(real_t value, std::string_view label)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::invalid_argument(std::string(label) + " must be finite and non-negative.");
    }
    return value;
}

real_t ideal_gas_alpha_source(real_t liquid_fraction, real_t release_efficiency,
                              real_t yield_mol_per_j, real_t power_density, real_t gas_constant,
                              real_t temperature, real_t absolute_pressure, real_t max_source_rate)
{
    require_non_negative(liquid_fraction, "liquid fraction");
    require_non_negative(release_efficiency, "release efficiency");
    if (release_efficiency > 1.0)
    {
        throw std::invalid_argument("release efficiency cannot exceed one.");
    }
    require_non_negative(yield_mol_per_j, "hydrogen yield");
    require_non_negative(power_density, "fission power density");
    require_positive(gas_constant, "gas constant");
    require_positive(temperature, "temperature");
    require_positive(absolute_pressure, "absolute pressure");
    if (max_source_rate < 0.0 || std::isnan(max_source_rate))
    {
        throw std::invalid_argument("maximum alpha source rate must be non-negative.");
    }

    const auto raw = liquid_fraction * release_efficiency * yield_mol_per_j * power_density *
                     gas_constant * temperature / absolute_pressure;
    if (!std::isfinite(raw))
    {
        throw std::invalid_argument("ideal-gas alpha source is non-finite.");
    }
    return std::min(raw, max_source_rate);
}

real_t henry_equilibrium_concentration(real_t henry_coefficient, real_t liquid_pressure,
                                       real_t surface_tension, real_t bubble_radius)
{
    require_non_negative(henry_coefficient, "Henry coefficient");
    require_positive(liquid_pressure, "liquid pressure");
    require_non_negative(surface_tension, "surface tension");
    require_positive(bubble_radius, "bubble radius");
    return henry_coefficient * (liquid_pressure + 2.0 * surface_tension / bubble_radius);
}

real_t surface_tension(const RadiolyticGasOptions& options, real_t temperature_kelvin)
{
    switch (options.surface_tension_mode)
    {
    case SurfaceTensionMode::Constant:
        return require_positive(options.surface_tension, "surface tension");
    case SurfaceTensionMode::External:
        require_positive(temperature_kelvin, "temperature");
        if (!options.surface_tension_correlation)
            throw std::invalid_argument("External surface tension requires a correlation callback.");
        return require_positive(options.surface_tension_correlation(temperature_kelvin), "surface tension");
    }
    throw std::invalid_argument("Unknown surface tension mode.");
}

real_t nucleation_radius(const RadiolyticGasOptions& options, real_t temperature_kelvin, real_t absolute_pressure)
{
    switch (options.nucleation_radius_mode)
    {
    case NucleationRadiusMode::Constant:
        return require_positive(options.nucleation_radius, "nucleation radius");
    case NucleationRadiusMode::External:
        require_positive(temperature_kelvin, "temperature");
        require_positive(absolute_pressure, "absolute pressure");
        if (!options.nucleation_radius_correlation)
            throw std::invalid_argument("External nucleation radius requires a correlation callback.");
        return require_positive(options.nucleation_radius_correlation(temperature_kelvin, absolute_pressure), "nucleation radius");
    }
    throw std::invalid_argument("Unknown nucleation radius mode.");
}

real_t hydrogen_diffusivity(const RadiolyticGasOptions& options, real_t temperature_kelvin)
{
    switch (options.diffusivity_mode)
    {
    case HydrogenDiffusivityMode::Constant:
        return require_positive(options.hydrogen_diffusivity, "hydrogen diffusivity");
    case HydrogenDiffusivityMode::External:
        require_positive(temperature_kelvin, "temperature");
        if (!options.hydrogen_diffusivity_correlation)
            throw std::invalid_argument("External hydrogen diffusivity requires a correlation callback.");
        return require_positive(options.hydrogen_diffusivity_correlation(temperature_kelvin), "hydrogen diffusivity");
    }
    throw std::invalid_argument("Unknown hydrogen diffusivity mode.");
}

real_t hughmark_sherwood(real_t reynolds, real_t schmidt)
{
    require_non_negative(reynolds, "Reynolds number");
    require_non_negative(schmidt, "Schmidt number");
    if (schmidt >= 250.0)
    {
        throw std::invalid_argument("Hughmark correlation requires 0 <= Sc < 250.");
    }
    if (reynolds < 776.06)
    {
        return 2.0 + 0.6 * std::sqrt(reynolds) * std::cbrt(schmidt);
    }
    return 2.0 + 0.27 * std::pow(reynolds, 0.63) * std::cbrt(schmidt);
}

real_t hughmark_mass_transfer_coefficient(real_t diffusivity, real_t radius, real_t liquid_density,
                                          real_t dynamic_viscosity, real_t relative_speed)
{
    require_positive(diffusivity, "hydrogen diffusivity");
    require_positive(radius, "bubble radius");
    require_positive(liquid_density, "liquid density");
    require_positive(dynamic_viscosity, "dynamic viscosity");
    require_non_negative(relative_speed, "relative speed");
    const auto schmidt = dynamic_viscosity / (liquid_density * diffusivity);
    const auto reynolds = 2.0 * liquid_density * relative_speed * radius / dynamic_viscosity;
    return hughmark_sherwood(reynolds, schmidt) * diffusivity / (2.0 * radius);
}

real_t celata2007_drag_coefficient(real_t reynolds, real_t eotvos)
{
    require_positive(reynolds, "bubble Reynolds number");
    require_non_negative(eotvos, "bubble Eotvos number");
    const auto spherical = 24.0 * (1.0 + 0.15 * std::pow(reynolds, 0.687)) / reynolds;
    const auto deformed = 8.0 * eotvos / (3.0 * (eotvos + 4.0));
    return std::max(spherical, deformed);
}

BubbleRiseVelocityResult
celata2007_bubble_rise_velocity(real_t radius, real_t liquid_density, real_t gas_density,
                                real_t dynamic_viscosity, real_t surface_tension, real_t gravity,
                                int max_iterations, real_t relative_tolerance)
{
    if (radius <= 0.0)
        return {};
    require_positive(liquid_density, "liquid density");
    require_non_negative(gas_density, "gas density");
    if (gas_density >= liquid_density)
    {
        throw std::invalid_argument("bubble gas density must be less than liquid density.");
    }
    require_positive(dynamic_viscosity, "dynamic viscosity");
    require_positive(surface_tension, "surface tension");
    require_positive(gravity, "gravity");
    require_positive(relative_tolerance, "rise-velocity tolerance");
    if (max_iterations < 1)
    {
        throw std::invalid_argument("rise-velocity iteration limit must be positive.");
    }

    const auto density_difference = liquid_density - gas_density;
    const auto eotvos = 4.0 * gravity * radius * radius * density_difference / surface_tension;
    const auto velocity_scale = 8.0 * radius * gravity / 3.0;
    auto evaluate = [&](real_t velocity)
    {
        const auto reynolds = 2.0 * liquid_density * velocity * radius / dynamic_viscosity;
        const auto drag_coefficient = celata2007_drag_coefficient(reynolds, eotvos);
        return std::pair{drag_coefficient * velocity * velocity - velocity_scale, drag_coefficient};
    };

    auto lower = std::numeric_limits<real_t>::min();
    auto upper = std::max(std::sqrt(velocity_scale), 1.0e-8);
    auto upper_evaluation = evaluate(upper);
    for (int expansion = 0; upper_evaluation.first < 0.0 && expansion < 64; ++expansion)
    {
        upper *= 2.0;
        upper_evaluation = evaluate(upper);
    }

    BubbleRiseVelocityResult result;
    result.eotvos = eotvos;
    if (upper_evaluation.first < 0.0)
    {
        result.velocity = upper;
        result.residual = upper_evaluation.first;
        result.drag_coefficient = upper_evaluation.second;
        result.converged = false;
        return result;
    }

    for (int iteration = 1; iteration <= max_iterations; ++iteration)
    {
        result.velocity = 0.5 * (lower + upper);
        const auto evaluation = evaluate(result.velocity);
        result.residual = evaluation.first;
        result.drag_coefficient = evaluation.second;
        result.iterations = iteration;
        if (std::abs(result.residual) <= relative_tolerance * velocity_scale)
        {
            result.reynolds = 2.0 * liquid_density * result.velocity * radius / dynamic_viscosity;
            const auto diameter = 2.0 * radius;
            result.within_experimental_range =
                diameter >= 0.5e-3 && diameter <= 4.0e-3 && result.reynolds >= 200.0 &&
                result.reynolds <= 1500.0 && result.eotvos >= 0.1 && result.eotvos <= 3.5;
            return result;
        }
        if (result.residual < 0.0)
            lower = result.velocity;
        else
            upper = result.velocity;
    }
    result.reynolds = 2.0 * liquid_density * result.velocity * radius / dynamic_viscosity;
    result.converged = false;
    return result;
}

BubbleRadiusResult solve_bubble_radius(real_t moles_per_bubble, real_t liquid_pressure,
                                       real_t surface_tension, real_t gas_constant,
                                       real_t temperature, real_t min_radius, real_t max_radius,
                                       int max_iterations, real_t relative_tolerance)
{
    if (moles_per_bubble <= 0.0)
    {
        return {};
    }
    require_positive(liquid_pressure, "liquid pressure");
    require_non_negative(surface_tension, "surface tension");
    require_positive(gas_constant, "gas constant");
    require_positive(temperature, "temperature");
    require_positive(min_radius, "minimum radius");
    require_positive(max_radius, "maximum radius");
    if (max_radius <= min_radius || max_iterations < 1)
    {
        throw std::invalid_argument("invalid bubble-radius solver bounds.");
    }

    const auto rhs = moles_per_bubble * gas_constant * temperature;
    auto residual = [&](real_t radius)
    {
        return 4.0 * std::numbers::pi / 3.0 *
                   (liquid_pressure * radius * radius * radius +
                    2.0 * surface_tension * radius * radius) -
               rhs;
    };
    auto lower = min_radius;
    auto upper = max_radius;
    const auto lower_residual = residual(lower);
    const auto upper_residual = residual(upper);
    if (lower_residual > 0.0 || upper_residual < 0.0)
    {
        return {lower_residual > 0.0 ? lower : upper,
                lower_residual > 0.0 ? lower_residual : upper_residual, 0, false};
    }

    BubbleRadiusResult result;
    for (int iteration = 1; iteration <= max_iterations; ++iteration)
    {
        result.radius = 0.5 * (lower + upper);
        result.residual = residual(result.radius);
        result.iterations = iteration;
        if (std::abs(result.residual) <= relative_tolerance * std::max(rhs, 1.0e-300))
        {
            return result;
        }
        if (result.residual < 0.0)
            lower = result.radius;
        else
            upper = result.radius;
    }
    result.converged = false;
    return result;
}

real_t bubble_void_fraction(real_t number_density, real_t radius)
{
    require_non_negative(number_density, "bubble number density");
    require_non_negative(radius, "bubble radius");
    return 4.0 * std::numbers::pi / 3.0 * number_density * radius * radius * radius;
}

real_t characteristic_radius(real_t micro_number, real_t micro_radius, real_t large_number,
                             real_t large_radius)
{
    const auto denominator =
        micro_number * micro_radius * micro_radius + large_number * large_radius * large_radius;
    if (denominator <= 0.0)
        return 0.0;
    return (micro_number * std::pow(micro_radius, 3) + large_number * std::pow(large_radius, 3)) /
           denominator;
}

real_t smoothed_heaviside(real_t value, RadiolyticHeavisideMode mode, real_t width)
{
    if (mode == RadiolyticHeavisideMode::Exact)
        return value > 0.0 ? 1.0 : 0.0;
    require_positive(width, "smoothed Heaviside width");
    return 0.5 * (1.0 + std::tanh(value / width));
}

} // namespace SimpleFluid::RadiolyticGasPhysics
