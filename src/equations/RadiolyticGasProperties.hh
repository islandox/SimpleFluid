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

#include <functional>
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

/** @brief Local two-population kinetics integration policy. */
enum class RadiolyticKineticsMode
{
    LegacySubcycled, ///< Retain the original source/decay splitting and subcycles.
    ExactInactive   ///< Integrate subcritical microbubble production/decay exactly; otherwise use legacy subcycles.
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
    External ///< Evaluate a caller-supplied surface-tension correlation.
};

/**
 * @brief Hydrogen diffusivity correlation selection.
 */
enum class HydrogenDiffusivityMode
{
    Constant,
    External   ///< Evaluate a caller-supplied temperature correlation.
};

/** @brief Liquid-side transfer law for the representative bubbles. */
enum class BubbleMassTransferMode
{
    LegacyHughmark,   ///< Historical model, requiring Sc < 250.
    SphericalDiffusion, ///< Steady diffusion-only reference: Sh = 2.
    FengMichaelidesClean, ///< Feng 2000 inviscid, mobile-interface endpoint.
    FengMichaelidesRigid ///< Feng 2000 rigid, immobilized-interface endpoint.
};

/** @brief Nucleation-radius property selection. */
enum class NucleationRadiusMode
{
    Constant,
    External
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
    RadiolyticKineticsMode kinetics_mode = RadiolyticKineticsMode::LegacySubcycled;
    BubbleRiseVelocityMode rise_velocity_mode = BubbleRiseVelocityMode::ZeroSlip;
    SurfaceTensionMode surface_tension_mode = SurfaceTensionMode::Constant;
    HydrogenDiffusivityMode diffusivity_mode = HydrogenDiffusivityMode::Constant;
    BubbleMassTransferMode mass_transfer_mode = BubbleMassTransferMode::LegacyHughmark;
    NucleationRadiusMode nucleation_radius_mode = NucleationRadiusMode::Constant;

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
    // External mode: input K, output m2/s, finite and positive. The function
    // must be reentrant and implement the same correlation on every MPI rank.
    real_t (*hydrogen_diffusivity_correlation)(real_t) = nullptr;
    // Material callbacks own any composition/model parameters they capture.
    // Inputs are K and Pa; outputs are N/m and m. Captures must remain valid,
    // reentrant and physically identical on every MPI rank.
    std::function<real_t(real_t)> surface_tension_correlation;
    std::function<real_t(real_t, real_t)> nucleation_radius_correlation;
    real_t nucleation_radius = 0.0; ///< Required supplied value for Constant mode.

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
    real_t transport_solver_tolerance = 1.0e-10; ///< Relative FV solve tolerance; default preserves existing behavior.
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

/** @brief Evaluate supplied surface tension (N/m) at temperature (K). */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
surface_tension(const RadiolyticGasOptions& options, real_t temperature_kelvin);

/** @brief Evaluate supplied nucleation radius (m) at temperature (K), pressure (Pa). */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
nucleation_radius(const RadiolyticGasOptions& options, real_t temperature_kelvin, real_t absolute_pressure);

/** @brief Evaluate and validate the selected dissolved-hydrogen diffusivity. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
hydrogen_diffusivity(const RadiolyticGasOptions& options, real_t temperature_kelvin);

/** @brief Hughmark Sherwood-number correlation. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
hughmark_sherwood(real_t reynolds, real_t schmidt);

/** @brief Liquid-side mass-transfer coefficient from Hughmark. */
SIMPLEFLUID_EQUATIONS_EXPORT real_t
hughmark_mass_transfer_coefficient(
    real_t diffusivity, real_t radius, real_t liquid_density,
    real_t dynamic_viscosity, real_t relative_speed);

/** @brief How to interpret the selected transfer result. */
enum class BubbleMassTransferApplicability
{
    LegacyScGuardPassed, ///< Existing Sc guard passed; no new Re qualification.
    DiffusionOnlyReference, ///< Steady radial diffusion with Pe = 0.
    SourceLowPeAsymptotic, ///< Feng 2000 p. 64 Eq. (4), Pe_d < 1.
    SourceFitOperationalWindow, ///< Feng 2000 p. 67 Eq. (19), gated high Pe.
    MixedSourceRegimes ///< Aggregate only: low-Pe and high-Pe evaluations occurred.
};

/** @brief Diameter-based dimensionless groups and liquid-side coefficient. */
struct BubbleMassTransferResult
{
    BubbleMassTransferMode model = BubbleMassTransferMode::LegacyHughmark;
    BubbleMassTransferApplicability applicability =
        BubbleMassTransferApplicability::LegacyScGuardPassed;
    real_t radius = 0.0;
    real_t diameter = 0.0;
    real_t relative_speed = 0.0;
    real_t liquid_density = 0.0;
    real_t dynamic_viscosity = 0.0;
    real_t diffusivity = 0.0;
    real_t reynolds = 0.0;
    real_t schmidt = 0.0;
    real_t peclet = 0.0;
    real_t sherwood = 0.0;
    real_t coefficient = 0.0;
};

/** @brief Pure, SI-input transfer evaluator selected once for a field operation. */
using BubbleMassTransferEvaluator = BubbleMassTransferResult (*)(
    real_t diffusivity, real_t radius, real_t liquid_density,
    real_t dynamic_viscosity, real_t relative_speed);

/**
 * @brief Resolve the immutable model selector before traversing cells.
 *
 * The returned evaluator still validates each cell's physical state and
 * correlation domain. Unknown model values throw invalid_argument here.
 */
SIMPLEFLUID_EQUATIONS_EXPORT BubbleMassTransferEvaluator
select_bubble_mass_transfer_evaluator(BubbleMassTransferMode model);

/**
 * @brief Evaluate the selected liquid-side law using SI inputs and bubble-liquid slip.
 *
 * Re, Sc, Pe and Sh use bubble diameter. Invalid inputs throw invalid_argument;
 * a finite state outside a correlation domain throws domain_error.
 */
SIMPLEFLUID_EQUATIONS_EXPORT BubbleMassTransferResult
bubble_mass_transfer(BubbleMassTransferMode model, real_t diffusivity,
                     real_t radius, real_t liquid_density,
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
