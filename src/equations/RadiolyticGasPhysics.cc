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

real_t pressure_nucleation_correction(real_t liquid_pressure, real_t atmospheric_pressure)
{
    require_positive(liquid_pressure, "liquid pressure");
    require_positive(atmospheric_pressure, "atmospheric pressure");
    const auto ratio = liquid_pressure / atmospheric_pressure;
    return 5.165e-5 * std::pow(ratio, 4) - 1.732e-3 * std::pow(ratio, 3) +
           0.02245 * std::pow(ratio, 2) - 0.1554 * ratio + 1.134;
}

real_t mean_fission_fragment_let(real_t temperature_kelvin, real_t uranium_concentration_mol_per_m3)
{
    require_positive(temperature_kelvin, "temperature");
    require_non_negative(uranium_concentration_mol_per_m3, "uranyl nitrate concentration");
    return (-1.3387e-6 * temperature_kelvin - 3.4319e-5) * uranium_concentration_mol_per_m3 -
           6.6431e-3 * temperature_kelvin + 8.8142;
}

real_t pure_water_nucleation_radius(real_t temperature_kelvin, real_t mean_let)
{
    require_positive(temperature_kelvin, "temperature");
    require_positive(mean_let, "mean LET");
    const auto t = temperature_kelvin;
    return (-2.862e-15 * t * t + 7.3996e-13 * t - 9.9925e-11) * mean_let * mean_let +
           (8.7909e-14 * t * t - 9.7928e-13 * t + 3.4558e-9) * mean_let + 9.7683e-14 * t * t -
           4.0125e-11 * t + 4.9092e-9;
}

real_t atmospheric_nucleation_radius(real_t pure_water_radius,
                                     real_t hydrogen_yield_molecules_per_100_ev)
{
    require_positive(pure_water_radius, "pure-water nucleation radius");
    require_non_negative(hydrogen_yield_molecules_per_100_ev, "hydrogen yield");
    if (hydrogen_yield_molecules_per_100_ev <= 0.5 || hydrogen_yield_molecules_per_100_ev >= 4.5)
    {
        throw std::invalid_argument("Winter's yield correction requires "
                                    "0.5 < G_H2 < 4.5 molecules per 100 eV.");
    }
    const auto yield = hydrogen_yield_molecules_per_100_ev;
    return (0.3554 + 0.4264 * yield - 0.0400 * yield * yield) * pure_water_radius;
}

real_t sheng2024_nucleation_radius(real_t temperature_kelvin,
                                   real_t uranium_concentration_mol_per_m3,
                                   real_t hydrogen_yield_molecules_per_100_ev,
                                   real_t liquid_pressure, real_t atmospheric_pressure)
{
    const auto mean_let =
        mean_fission_fragment_let(temperature_kelvin, uranium_concentration_mol_per_m3);
    const auto water_radius = pure_water_nucleation_radius(temperature_kelvin, mean_let);
    const auto atmospheric_radius =
        atmospheric_nucleation_radius(water_radius, hydrogen_yield_molecules_per_100_ev);
    return pressure_nucleation_correction(liquid_pressure, atmospheric_pressure) *
           atmospheric_radius;
}

real_t sheng2024_surface_tension(real_t temperature_celsius,
                                 real_t uranium_concentration_mol_per_m3)
{
    if (!std::isfinite(temperature_celsius))
    {
        throw std::invalid_argument("temperature must be finite.");
    }
    require_non_negative(uranium_concentration_mol_per_m3, "uranyl nitrate concentration");
    return 1.7160e-7 * temperature_celsius * temperature_celsius - 1.4427e-4 * temperature_celsius +
           2.0163e-6 * uranium_concentration_mol_per_m3 + 7.5725e-2;
}

real_t sheng2024_hydrogen_diffusivity(real_t temperature_kelvin)
{
    require_positive(temperature_kelvin, "temperature");
    return std::pow(10.0, -1.46551 - 8.4259e2 / temperature_kelvin) * 1.0e-4;
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
