/**
 * @file RadiolyticGasProperties.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Configuration and pure property functions for radiolytic gas models.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "SimpleFluidExport.hh"
#include "dataclass/Database.hh"
#include "dataclass/typedefs.hh"

#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Top-level radiolysis model selection.
 */
enum class RadiolyticGasMode
{
    Disabled,
    IdealGasSource,        ///< Convert hydrogen production directly to void source.
    Sheng2024TwoPopulation ///< Evolve dissolved, micro-, and large-bubble inventories.
};

/**
 * @brief Strategy used to provide absolute pressure to radiolysis physics.
 */
enum class RadiolyticPressureMode
{
    Constant,          ///< Use a constant reference pressure.
    PrescribedHistory, ///< Interpolate pressure from a supplied time history.
    Reconstructed,     ///< Add solver gauge pressure to the reference pressure.
    Inertial           ///< Include transient inertial pressure corrections.
};

/**
 * @brief Dissolved-hydrogen transport selection.
 */
enum class RadiolyticTransportMode
{
    NoAdvection, ///< Keep dissolved inventory cell-local.
    Advective    ///< Transport dissolved inventory with liquid flux.
};

/**
 * @brief Bubble-population transport selection.
 */
enum class BubbleTransportMode
{
    General, ///< Transport bubbles with vector rise velocity.
    Axial    ///< Restrict bubble rise to the configured axial direction.
};

/**
 * @brief Switch used by thresholded conversion and source terms.
 */
enum class RadiolyticHeavisideMode
{
    Exact,   ///< Use a discontinuous threshold switch.
    Smoothed ///< Use a hyperbolic-tangent transition.
};

/**
 * @brief Bubble slip/rise-velocity correlation selection.
 */
enum class BubbleRiseVelocityMode
{
    ZeroSlip,     ///< Advect bubbles with liquid velocity.
    ConstantSlip, ///< Apply a prescribed slip speed.
    Celata2007    ///< Solve the Celata drag relation.
};

/**
 * @brief Surface-tension correlation selection.
 */
enum class SurfaceTensionMode
{
    Constant,
    Sheng2024 ///< Evaluate the Sheng temperature/concentration correlation.
};

/**
 * @brief Hydrogen diffusivity correlation selection.
 */
enum class HydrogenDiffusivityMode
{
    Constant,
    Sheng2024 ///< Evaluate the Sheng temperature correlation.
};

/**
 * @brief Runtime configuration for ideal and two-population radiolysis.
 */
struct RadiolyticGasOptions
{
    RadiolyticGasMode mode = RadiolyticGasMode::Disabled;
    RadiolyticPressureMode pressure_mode = RadiolyticPressureMode::Constant;
    RadiolyticTransportMode dissolved_transport = RadiolyticTransportMode::NoAdvection;
    BubbleTransportMode bubble_transport = BubbleTransportMode::General;
    RadiolyticHeavisideMode heaviside_mode = RadiolyticHeavisideMode::Exact;
    BubbleRiseVelocityMode rise_velocity_mode = BubbleRiseVelocityMode::ZeroSlip;
    SurfaceTensionMode surface_tension_mode = SurfaceTensionMode::Constant;
    HydrogenDiffusivityMode diffusivity_mode = HydrogenDiffusivityMode::Constant;

    real_t hydrogen_yield_mol_per_j = 0.0;
    real_t gas_release_efficiency = 1.0;  ///< Fraction of generated hydrogen released to gas.
    real_t reference_pressure = 101325.0; ///< Reference absolute pressure.
    real_t gas_constant = 8.31446261815324;
    real_t alpha_min = 0.0;  ///< Lower gas void-fraction bound.
    real_t alpha_max = 0.95; ///< Upper gas void-fraction bound.
    real_t max_source_alpha_rate =
        std::numeric_limits<real_t>::infinity(); ///< Ideal-mode void-source cap.

    // H = C/p in mol/(m^3 Pa), matching Sheng et al. Eq. (9).
    real_t henry_coefficient = 0.0;
    real_t surface_tension = 0.0;           ///< Constant gas-liquid surface tension.
    real_t hydrogen_diffusivity = 0.0;      ///< Constant dissolved-hydrogen diffusivity.
    real_t atmospheric_pressure = 101325.0; ///< Pressure reference for nucleation correction.
    real_t uranium_concentration_mol_per_m3 = 0.0;
    real_t hydrogen_yield_molecules_per_100_ev =
        0.0; ///< Radiation yield used by nucleation correlation.

    real_t microbubble_lifetime = 10.0e-6;
    real_t large_bubble_dissolution_time = 50.0e-6;
    real_t micro_to_large_conversion_coefficient = 1.0e-4;
    real_t smooth_heaviside_width = 0.0;
    real_t constant_slip_velocity = 0.0;
    real_t bubble_gas_density = 0.0;
    real_t bubble_gravity = 9.80665;          ///< Gravity magnitude in rise correlations.
    real_t rise_velocity_tolerance = 1.0e-10; ///< Relative terminal-speed solver tolerance.
    int max_rise_velocity_iterations = 100;

    real_t initial_dissolved_hydrogen = 0.0;
    real_t initial_micro_number_density = 0.0;
    real_t initial_micro_moles = 0.0;
    real_t initial_large_number_density = 0.0;
    real_t initial_large_moles = 0.0;

    real_t min_radius = 1.0e-10;
    real_t max_radius = 1.0e-2;
    real_t min_population = 1.0e-30;
    real_t max_population = 1.0e30;
    real_t max_concentration = 1.0e6;
    real_t local_ode_tolerance = 1.0e-8; ///< Local kinetics relative tolerance.
    int max_subcycles = 10000;
    int max_radius_iterations = 100;

    real_t liquid_compressibility = 4.5e-10;
    real_t liquid_thermal_expansion = 0.0;
    real_t minimum_absolute_pressure = 1.0;

    std::vector<real_t> pressure_history_times;
    std::vector<real_t>
        pressure_history_values; ///< Absolute pressures corresponding to history times.
    std::vector<std::string> free_surface_patches; ///< Patches through which bubbles may escape.
};

/**
 * @brief Pure validation helpers and gas-property correlations.
 */
namespace RadiolyticGasPhysics
{

/** @brief Require a finite strictly positive value. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
require_positive(real_t value, std::string_view label);

/** @brief Require a finite non-negative value. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
require_non_negative(real_t value, std::string_view label);

/** @brief Convert fission power density to an ideal-gas alpha source rate. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
ideal_gas_alpha_source(real_t liquid_fraction, real_t release_efficiency,
                       real_t yield_mol_per_j, real_t power_density,
                       real_t gas_constant, real_t temperature,
                       real_t absolute_pressure, real_t max_source_rate);

/** @brief Henry-law equilibrium H2 concentration including Laplace pressure. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
henry_equilibrium_concentration(real_t henry_coefficient,
                                real_t liquid_pressure,
                                real_t surface_tension,
                                real_t bubble_radius);

/** @brief Sheng pressure correction for the nucleation radius. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
pressure_nucleation_correction(real_t liquid_pressure,
                               real_t atmospheric_pressure);

/** @brief Mean fission-fragment LET from Sheng Eq. (13). */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
mean_fission_fragment_let(real_t temperature_kelvin,
                          real_t uranium_concentration_mol_per_m3);

/** @brief Pure-water nucleation radius correlation from Winter. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
pure_water_nucleation_radius(real_t temperature_kelvin, real_t mean_let);

/** @brief Correct pure-water nucleation radius for H2 radiation yield. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
atmospheric_nucleation_radius(
    real_t pure_water_radius,
    real_t hydrogen_yield_molecules_per_100_ev);

/** @brief Sheng 2024 nucleation radius assembled from LET and pressure terms. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
sheng2024_nucleation_radius(
    real_t temperature_kelvin,
    real_t uranium_concentration_mol_per_m3,
    real_t hydrogen_yield_molecules_per_100_ev,
    real_t liquid_pressure, real_t atmospheric_pressure);

/**
 * @brief Sheng 2024 surface-tension correlation from Table 2.
 *
 * Sheng et al. (2024) Table 2 prints a positive quadratic term. Its cited
 * Winter sources print a negative term; this selector follows the Sheng table.
 */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
sheng2024_surface_tension(real_t temperature_celsius,
                          real_t uranium_concentration_mol_per_m3);

/** @brief Sheng 2024 hydrogen diffusivity correlation. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
sheng2024_hydrogen_diffusivity(real_t temperature_kelvin);

/** @brief Hughmark Sherwood-number correlation. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
hughmark_sherwood(real_t reynolds, real_t schmidt);

/** @brief Liquid-side mass-transfer coefficient from Hughmark. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
hughmark_mass_transfer_coefficient(
    real_t diffusivity, real_t radius, real_t liquid_density,
    real_t dynamic_viscosity, real_t relative_speed);

/** @brief Celata 2007 drag coefficient from Reynolds and Eotvos numbers. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
celata2007_drag_coefficient(real_t reynolds, real_t eotvos);

/**
 * @brief Result and diagnostics from the bubble rise-velocity solve.
 */
struct BubbleRiseVelocityResult
{
    real_t velocity = 0.0;
    real_t reynolds = 0.0;
    real_t eotvos = 0.0;
    real_t drag_coefficient = 0.0;
    real_t residual = 0.0;
    int iterations = 0;
    bool converged = true;
    bool within_experimental_range = false;
};

/**
 * @brief Solve terminal bubble rise speed with the Celata drag relation.
 *
 * Winter et al. (2022) Eq. (15), using the Celata et al. (2007) drag
 * relation. The bracketed solve avoids the zero-velocity root introduced by
 * rearranging the terminal-velocity equation.
 */
SIMPLEFLUID_EQUATIONS_EXPORT BubbleRiseVelocityResult
celata2007_bubble_rise_velocity(real_t radius, real_t liquid_density, real_t gas_density,
                                real_t dynamic_viscosity, real_t surface_tension,
                                real_t gravity = 9.80665, int max_iterations = 100,
                                real_t relative_tolerance = 1.0e-10);

/**
 * @brief Result and diagnostics from the bubble-radius solve.
 */
struct BubbleRadiusResult
{
    real_t radius = 0.0;
    real_t residual = 0.0;
    int iterations = 0;
    bool converged = true;
};

/** @brief Solve bubble radius from moles and capillary pressure. */
SIMPLEFLUID_EQUATIONS_EXPORT BubbleRadiusResult
solve_bubble_radius(real_t moles_per_bubble, real_t liquid_pressure,
                    real_t surface_tension, real_t gas_constant,
                    real_t temperature, real_t min_radius,
                    real_t max_radius, int max_iterations = 100,
                    real_t relative_tolerance = 1.0e-12);

/** @brief Compute void fraction from bubble number density and radius. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
bubble_void_fraction(real_t number_density, real_t radius);

/** @brief Compute the area-weighted characteristic radius of two populations. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
characteristic_radius(real_t micro_number, real_t micro_radius,
                      real_t large_number, real_t large_radius);

/** @brief Evaluate either exact or smoothed Heaviside activation. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
smoothed_heaviside(real_t value, RadiolyticHeavisideMode mode, real_t width);

} // namespace RadiolyticGasPhysics

/**
 * @brief Parse radiolytic gas options from a flat database.
 */
SIMPLEFLUID_EQUATIONS_EXPORT RadiolyticGasOptions
radiolytic_gas_options_from_database(const Database& database);

/**
 * @brief Validate radiolytic gas options and throw on inconsistent settings.
 */
SIMPLEFLUID_EQUATIONS_EXPORT void
validate_radiolytic_gas_options(const RadiolyticGasOptions& options);

} // namespace SimpleFluid
