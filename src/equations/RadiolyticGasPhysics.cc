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
#include <sstream>
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

namespace
{
BubbleMassTransferResult initialize_bubble_mass_transfer(
    BubbleMassTransferMode model, real_t diffusivity, real_t radius,
    real_t liquid_density, real_t dynamic_viscosity, real_t relative_speed)
{
    require_positive(diffusivity, "molecular hydrogen diffusivity");
    require_positive(radius, "bubble radius");
    require_positive(liquid_density, "liquid density");
    require_positive(dynamic_viscosity, "liquid dynamic viscosity");
    if (!std::isfinite(relative_speed))
        throw std::invalid_argument("bubble-liquid relative speed must be finite.");

    BubbleMassTransferResult result;
    result.model = model;
    result.diffusivity = diffusivity;
    result.radius = radius;
    result.diameter = 2.0 * radius;
    result.liquid_density = liquid_density;
    result.dynamic_viscosity = dynamic_viscosity;
    result.relative_speed = std::abs(relative_speed);
    result.reynolds = 2.0 * liquid_density * result.relative_speed * radius
                    / dynamic_viscosity;
    result.schmidt = dynamic_viscosity / (liquid_density * diffusivity);
    result.peclet = result.reynolds * result.schmidt;
    if (!std::isfinite(result.diameter) || !std::isfinite(result.reynolds)
        || !std::isfinite(result.schmidt) || !std::isfinite(result.peclet))
        throw std::invalid_argument("bubble transfer dimensionless groups must be finite.");
    return result;
}

BubbleMassTransferResult finalize_bubble_mass_transfer(BubbleMassTransferResult result)
{
    result.coefficient = result.sherwood * result.diffusivity / result.diameter;
    if (!std::isfinite(result.sherwood) || !std::isfinite(result.coefficient)
        || result.sherwood <= 0.0 || result.coefficient <= 0.0)
        throw std::invalid_argument("bubble mass-transfer result must be finite and positive.");
    return result;
}

BubbleMassTransferResult feng_michaelides_bubble_mass_transfer(
    BubbleMassTransferResult result)
{
    // Feng and Michaelides (2000), Powder Technology 112: p. 64 Eq. (4)
    // quotes the earlier low-Pe steady asymptote; p. 67 Eq. (19) is the
    // high-Pe numerical fit. Eq. (11) uses radius Pe_a = U R / D, while
    // Eq. (A-12) uses diameter-averaged Sh.
    // sigma = mu_l/mu_s: sigma -> infinity gives the inviscid mobile
    // endpoint; sigma = 0 gives the rigid endpoint. The paper supplies
    // no bridge between the small-Pe asymptote and high-Pe fit.
    const auto radius_peclet = result.peclet / 2.0;
    const bool clean = result.model == BubbleMassTransferMode::FengMichaelidesClean;
    if (result.reynolds >= 1.0
        || (result.peclet >= 1.0
            && (radius_peclet < 100.0 || radius_peclet > 5000.0)))
    {
        std::ostringstream message;
        message.precision(17);
        message << (clean ? "fengMichaelidesClean" : "fengMichaelidesRigid")
                << " requires Re_d < 1 and either 0 <= Pe_d < 1 "
                   "(low-Pe asymptote) or 100 <= Pe_a <= 5000 "
                   "(high-Pe fit, Pe_a = Pe_d/2): R="
                << result.radius << " m, d=" << result.diameter << " m, u_rel="
                << result.relative_speed << " m/s, rho_l=" << result.liquid_density
                << " kg/m3, mu_l=" << result.dynamic_viscosity << " Pa s, D_m="
                << result.diffusivity << " m2/s, Re_d=" << result.reynolds
                << ", Sc=" << result.schmidt << ", Pe_d=" << result.peclet
                << ", Pe_a=" << radius_peclet;
        throw std::domain_error(message.str());
    }
    if (result.peclet < 1.0)
    {
        // Eq. (4) with sigma -> infinity (clean) or sigma = 0 (rigid).
        // lim_{Pe->0} Pe^2 log(Pe) = 0; avoid log(0) explicitly.
        const auto logarithmic = result.peclet == 0.0
            ? 0.0 : result.peclet * result.peclet * std::log(result.peclet);
        result.sherwood = 2.0 + 0.5 * result.peclet
            + (clean ? 1.0 / 12.0 : 1.0 / 8.0) * logarithmic;
        result.applicability = BubbleMassTransferApplicability::SourceLowPeAsymptotic;
    }
    else
    {
        // The Pe_a >= 100 gate is inferred from Table 1 fit residuals;
        // Re_d < 1 is an operational Stokes assumption, not a source
        // validation over a rectangular Re-by-Sc domain.
        result.sherwood = 1.49 * std::pow(radius_peclet, clean ? 0.435 : 0.322);
        result.applicability = BubbleMassTransferApplicability::SourceFitOperationalWindow;
    }
    return result;
}

template<BubbleMassTransferMode Model>
BubbleMassTransferResult selected_bubble_mass_transfer(
    real_t diffusivity, real_t radius, real_t liquid_density,
    real_t dynamic_viscosity, real_t relative_speed);

template<>
BubbleMassTransferResult
selected_bubble_mass_transfer<BubbleMassTransferMode::LegacyHughmark>(
    real_t diffusivity, real_t radius, real_t liquid_density,
    real_t dynamic_viscosity, real_t relative_speed)
{
    auto result = initialize_bubble_mass_transfer(
        BubbleMassTransferMode::LegacyHughmark, diffusivity, radius,
        liquid_density, dynamic_viscosity, relative_speed);
    if (result.schmidt >= 250.0)
    {
        std::ostringstream message;
        message.precision(17);
        message << "legacyHughmark requires Sc < 250: R=" << result.radius
                << " m, d=" << result.diameter << " m, u_rel="
                << result.relative_speed << " m/s, rho_l=" << result.liquid_density
                << " kg/m3, mu_l=" << result.dynamic_viscosity << " Pa s, D_m="
                << result.diffusivity << " m2/s, Re=" << result.reynolds
                << ", Sc=" << result.schmidt << ", Pe=" << result.peclet;
        throw std::domain_error(message.str());
    }
    result.sherwood = hughmark_sherwood(result.reynolds, result.schmidt);
    return finalize_bubble_mass_transfer(result);
}

template<>
BubbleMassTransferResult
selected_bubble_mass_transfer<BubbleMassTransferMode::SphericalDiffusion>(
    real_t diffusivity, real_t radius, real_t liquid_density,
    real_t dynamic_viscosity, real_t relative_speed)
{
    auto result = initialize_bubble_mass_transfer(
        BubbleMassTransferMode::SphericalDiffusion, diffusivity, radius,
        liquid_density, dynamic_viscosity, relative_speed);
    if (result.relative_speed != 0.0)
    {
        std::ostringstream message;
        message.precision(17);
        message << "sphericalDiffusion is a quiescent-liquid reference requiring Pe = 0: "
                << "R=" << result.radius << " m, d=" << result.diameter
                << " m, u_rel=" << result.relative_speed << " m/s, rho_l="
                << result.liquid_density << " kg/m3, mu_l="
                << result.dynamic_viscosity << " Pa s, D_m=" << result.diffusivity
                << " m2/s, Re=" << result.reynolds << ", Sc=" << result.schmidt
                << ", Pe=" << result.peclet;
        throw std::domain_error(message.str());
    }
    result.applicability = BubbleMassTransferApplicability::DiffusionOnlyReference;
    result.sherwood = 2.0;
    return finalize_bubble_mass_transfer(result);
}

template<>
BubbleMassTransferResult
selected_bubble_mass_transfer<BubbleMassTransferMode::FengMichaelidesClean>(
    real_t diffusivity, real_t radius, real_t liquid_density,
    real_t dynamic_viscosity, real_t relative_speed)
{
    const auto result = initialize_bubble_mass_transfer(
        BubbleMassTransferMode::FengMichaelidesClean, diffusivity, radius,
        liquid_density, dynamic_viscosity, relative_speed);
    return finalize_bubble_mass_transfer(
        feng_michaelides_bubble_mass_transfer(result));
}

template<>
BubbleMassTransferResult
selected_bubble_mass_transfer<BubbleMassTransferMode::FengMichaelidesRigid>(
    real_t diffusivity, real_t radius, real_t liquid_density,
    real_t dynamic_viscosity, real_t relative_speed)
{
    const auto result = initialize_bubble_mass_transfer(
        BubbleMassTransferMode::FengMichaelidesRigid, diffusivity, radius,
        liquid_density, dynamic_viscosity, relative_speed);
    return finalize_bubble_mass_transfer(
        feng_michaelides_bubble_mass_transfer(result));
}
} // namespace

BubbleMassTransferEvaluator select_bubble_mass_transfer_evaluator(
    BubbleMassTransferMode model)
{
    switch (model)
    {
    case BubbleMassTransferMode::LegacyHughmark:
        return &selected_bubble_mass_transfer<BubbleMassTransferMode::LegacyHughmark>;
    case BubbleMassTransferMode::SphericalDiffusion:
        return &selected_bubble_mass_transfer<BubbleMassTransferMode::SphericalDiffusion>;
    case BubbleMassTransferMode::FengMichaelidesClean:
        return &selected_bubble_mass_transfer<BubbleMassTransferMode::FengMichaelidesClean>;
    case BubbleMassTransferMode::FengMichaelidesRigid:
        return &selected_bubble_mass_transfer<BubbleMassTransferMode::FengMichaelidesRigid>;
    }
    throw std::invalid_argument("Unknown bubble mass-transfer mode.");
}

BubbleMassTransferResult bubble_mass_transfer(
    BubbleMassTransferMode model, real_t diffusivity, real_t radius,
    real_t liquid_density, real_t dynamic_viscosity, real_t relative_speed)
{
    return select_bubble_mass_transfer_evaluator(model)(
        diffusivity, radius, liquid_density, dynamic_viscosity, relative_speed);
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
