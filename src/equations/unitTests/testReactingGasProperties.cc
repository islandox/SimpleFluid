/**
 * @file testReactingGasProperties.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests reacting gas option parsing and pure property functions.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/ReactingGasProperties.hh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

namespace Physics = SimpleFluid::BubbleGasPhysics;

TEST(ReactingGasPropertiesTest, ExternalDiffusivityChecksCallbackAndValues)
{
    SimpleFluid::ReactingGasOptions options;
    options.diffusivity_mode = SimpleFluid::HydrogenDiffusivityMode::External;
    EXPECT_THROW(Physics::hydrogen_diffusivity(options, 300), std::invalid_argument);
    options.hydrogen_diffusivity_correlation = +[](double temperature) { return temperature * 1e-11; };
    EXPECT_DOUBLE_EQ(Physics::hydrogen_diffusivity(options, 350), 350 * 1e-11);
    EXPECT_THROW(Physics::hydrogen_diffusivity(options, 0), std::invalid_argument);
    options.hydrogen_diffusivity_correlation = +[](double) { return -1.0; };
    EXPECT_THROW(Physics::hydrogen_diffusivity(options, 300), std::invalid_argument);
    options.hydrogen_diffusivity_correlation = +[](double) { return std::numeric_limits<double>::quiet_NaN(); };
    EXPECT_THROW(Physics::hydrogen_diffusivity(options, 300), std::invalid_argument);
}

/** @brief Verifies the expected scaling of the ideal-gas reacting source. */
TEST(ReactingGasPropertiesTest, IdealGasSourceHasExpectedScaling)
{
    const auto base = Physics::ideal_gas_alpha_source(
        0.8, 0.5, 2.0e-7, 4.0e6, 8.314462618, 300.0, 1.0e5, 10.0);
    const auto double_temperature = Physics::ideal_gas_alpha_source(
        0.8, 0.5, 2.0e-7, 4.0e6, 8.314462618, 600.0, 1.0e5, 10.0);
    const auto double_pressure = Physics::ideal_gas_alpha_source(
        0.8, 0.5, 2.0e-7, 4.0e6, 8.314462618, 300.0, 2.0e5, 10.0);
    const auto double_power = Physics::ideal_gas_alpha_source(
        0.8, 0.5, 2.0e-7, 8.0e6, 8.314462618, 300.0, 1.0e5, 10.0);
    const auto double_yield = Physics::ideal_gas_alpha_source(
        0.8, 0.5, 4.0e-7, 4.0e6, 8.314462618, 300.0, 1.0e5, 10.0);

    EXPECT_GT(base, 0.0);
    EXPECT_NEAR(double_temperature, 2.0 * base, 1.0e-14);
    EXPECT_NEAR(double_pressure, 0.5 * base, 1.0e-14);
    EXPECT_NEAR(double_power, 2.0 * base, 1.0e-14);
    EXPECT_NEAR(double_yield, 2.0 * base, 1.0e-14);
    EXPECT_DOUBLE_EQ(
        Physics::ideal_gas_alpha_source(
            0.8, 0.5, 2.0e-7, 0.0, 8.314462618, 300.0, 1.0e5, 10.0),
        0.0);
    EXPECT_DOUBLE_EQ(
        Physics::ideal_gas_alpha_source(
            0.8, 0.5, 2.0e-7, 4.0e6, 8.314462618, 300.0, 1.0e5,
            base / 3.0),
        base / 3.0);
}

/** @brief Verifies rejection of invalid thermodynamic property inputs. */
TEST(ReactingGasPropertiesTest, RejectsInvalidThermodynamicInputs)
{
    EXPECT_THROW(
        Physics::ideal_gas_alpha_source(
            1.0, 1.0, 1.0e-7, 1.0, 8.314, 0.0, 1.0e5, 1.0),
        std::invalid_argument);
    EXPECT_THROW(
        Physics::ideal_gas_alpha_source(
            1.0, 1.0, 1.0e-7, 1.0, 8.314, 300.0, 0.0, 1.0),
        std::invalid_argument);
    EXPECT_THROW(
        Physics::ideal_gas_alpha_source(
            1.0, 1.2, 1.0e-7, 1.0, 8.314, 300.0, 1.0e5, 1.0),
        std::invalid_argument);
}

/** @brief Verifies monotonic Henry-law and Laplace-pressure terms. */
TEST(ReactingGasPropertiesTest, HenryAndLaplaceTermsAreMonotone)
{
    const auto low_pressure =
        Physics::henry_equilibrium_concentration(
            1.0e-5, 1.0e5, 0.07, 1.0e-5);
    const auto high_pressure =
        Physics::henry_equilibrium_concentration(
            1.0e-5, 2.0e5, 0.07, 1.0e-5);
    const auto small_bubble =
        Physics::henry_equilibrium_concentration(
            1.0e-5, 1.0e5, 0.07, 1.0e-6);
    EXPECT_GT(high_pressure, low_pressure);
    EXPECT_GT(small_bubble, low_pressure);
}

/** Material-specific correlations live in external libraries; test only the contract here. */
TEST(ReactingGasPropertiesTest, ExternalSurfaceAndNucleationProperties)
{
    SimpleFluid::ReactingGasOptions options;
    options.surface_tension_mode = SimpleFluid::SurfaceTensionMode::External;
    options.nucleation_radius_mode = SimpleFluid::NucleationRadiusMode::External;
    EXPECT_THROW(Physics::surface_tension(options, 300), std::invalid_argument);
    EXPECT_THROW(Physics::nucleation_radius(options, 300, 1e5), std::invalid_argument);
    const double scale = 2e-10;
    options.surface_tension_correlation = [](double t) { return .07 * t / 300; };
    options.nucleation_radius_correlation = [scale](double t, double p) { return scale * t * 1e5 / p; };
    EXPECT_DOUBLE_EQ(Physics::surface_tension(options, 350), .07 * 350 / 300);
    EXPECT_DOUBLE_EQ(Physics::nucleation_radius(options, 300, 2e5), scale * 300 / 2);
    EXPECT_THROW(Physics::surface_tension(options, 0), std::invalid_argument);
    EXPECT_THROW(Physics::nucleation_radius(options, 300, 0), std::invalid_argument);
    options.surface_tension_correlation = [](double) { return std::numeric_limits<double>::quiet_NaN(); };
    options.nucleation_radius_correlation = [](double, double) { return -1.; };
    EXPECT_THROW(Physics::surface_tension(options, 300), std::invalid_argument);
    EXPECT_THROW(Physics::nucleation_radius(options, 300, 1e5), std::invalid_argument);
}

TEST(ReactingGasPropertiesTest, RemovedMaterialSelectorsAreRejected)
{
    for (const auto* key : {"surface_tension_model", "hydrogen_diffusivity_model"})
    {
        SimpleFluid::Database database;
        database.set(key, std::string{"sheng2024"});
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
    }
    for (const auto* key : {"uranium_concentration_mol_per_m3", "hydrogen_yield_molecules_per_100_ev", "atmospheric_pressure"})
    {
        SimpleFluid::Database database;
        database.set(key, SimpleFluid::real_t{1000});
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database), std::invalid_argument);
    }
}

/** @brief Verifies Hughmark-correlation branches and validity constraints. */
TEST(ReactingGasPropertiesTest, HughmarkBranchesAndValidity)
{
    const auto low =
        Physics::hughmark_sherwood(100.0, 100.0);
    const auto expected_low =
        2.0 + 0.6 * std::sqrt(100.0) * std::cbrt(100.0);
    EXPECT_NEAR(low, expected_low, 1.0e-13);

    const auto high =
        Physics::hughmark_sherwood(1000.0, 100.0);
    const auto expected_high =
        2.0 + 0.27 * std::pow(1000.0, 0.63) * std::cbrt(100.0);
    EXPECT_NEAR(high, expected_high, 1.0e-13);
    EXPECT_THROW(
        Physics::hughmark_sherwood(1.0, 250.0),
        std::invalid_argument);
}

/** @brief Verifies that the Celata bubble-rise velocity balances drag. */
TEST(ReactingGasPropertiesTest, CelataRiseVelocityBalancesDrag)
{
    constexpr double radius = 1.0e-3;
    constexpr double liquid_density = 998.0;
    constexpr double gas_density = 1.2;
    constexpr double viscosity = 1.0e-3;
    constexpr double surface_tension = 0.072;
    constexpr double gravity = 9.80665;
    const auto result = Physics::celata2007_bubble_rise_velocity(
        radius,
        liquid_density,
        gas_density,
        viscosity,
        surface_tension,
        gravity);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.velocity, 0.0);
    EXPECT_GT(result.reynolds, 0.0);
    EXPECT_GT(result.eotvos, 0.0);
    EXPECT_GT(result.drag_coefficient, 0.0);
    EXPECT_TRUE(result.within_experimental_range);
    const auto velocity_scale = 8.0 * radius * gravity / 3.0;
    EXPECT_LE(
        std::abs(result.residual), 1.0e-9 * velocity_scale);
}

/** @brief Verifies that the computed bubble radius satisfies the equation of state. */
TEST(ReactingGasPropertiesTest, BubbleRadiusSolvesEos)
{
    constexpr double pressure = 101325.0;
    constexpr double surface_tension = 0.07;
    constexpr double gas_constant = 8.314462618;
    constexpr double temperature = 300.0;
    constexpr double radius = 2.0e-5;
    const auto moles =
        4.0 * std::numbers::pi / 3.0
        * (pressure * radius * radius * radius
           + 2.0 * surface_tension * radius * radius)
        / (gas_constant * temperature);

    const auto result = Physics::solve_bubble_radius(
        moles, pressure, surface_tension, gas_constant, temperature,
        1.0e-8, 1.0e-2, 100, 1.0e-11);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.radius, radius, radius * 1.0e-9);

    const auto larger = Physics::solve_bubble_radius(
        2.0 * moles, pressure, surface_tension, gas_constant,
        temperature, 1.0e-8, 1.0e-2);
    EXPECT_TRUE(larger.converged);
    EXPECT_GT(larger.radius, result.radius);
}

/** @brief Verifies expected bubble-radius trends with thermodynamic state. */
TEST(ReactingGasPropertiesTest, BubbleRadiusTrendsWithState)
{
    constexpr double pressure = 101325.0;
    constexpr double surface_tension = 0.07;
    constexpr double gas_constant = 8.314462618;
    constexpr double temperature = 300.0;
    constexpr double moles = 1.0e-12;

    const auto base = Physics::solve_bubble_radius(
        moles, pressure, surface_tension, gas_constant, temperature,
        1.0e-8, 1.0e-2);
    const auto hot = Physics::solve_bubble_radius(
        moles, pressure, surface_tension, gas_constant, 2.0 * temperature,
        1.0e-8, 1.0e-2);
    const auto compressed = Physics::solve_bubble_radius(
        moles, 2.0 * pressure, surface_tension, gas_constant, temperature,
        1.0e-8, 1.0e-2);
    const auto more_bubbles = Physics::solve_bubble_radius(
        0.5 * moles, pressure, surface_tension, gas_constant, temperature,
        1.0e-8, 1.0e-2);

    ASSERT_TRUE(base.converged);
    ASSERT_TRUE(hot.converged);
    ASSERT_TRUE(compressed.converged);
    ASSERT_TRUE(more_bubbles.converged);
    EXPECT_GT(hot.radius, base.radius);
    EXPECT_LT(compressed.radius, base.radius);
    EXPECT_LT(more_bubbles.radius, base.radius);
}

/** @brief Verifies reconstruction of void fraction and characteristic radius. */
TEST(ReactingGasPropertiesTest, VoidAndCharacteristicRadiusReconstruct)
{
    const auto micro_void =
        Physics::bubble_void_fraction(2.0e9, 1.0e-6);
    const auto large_void =
        Physics::bubble_void_fraction(5.0e6, 1.0e-4);
    EXPECT_GT(micro_void, 0.0);
    EXPECT_GT(large_void, micro_void);

    const auto characteristic =
        Physics::characteristic_radius(
            2.0e9, 1.0e-6, 5.0e6, 1.0e-4);
    EXPECT_GT(characteristic, 1.0e-6);
    EXPECT_LT(characteristic, 1.0e-4);
    EXPECT_DOUBLE_EQ(
        Physics::characteristic_radius(0.0, 0.0, 0.0, 0.0),
        0.0);
}

/** @brief Verifies parsing of flat runtime correlation selectors. */
TEST(ReactingGasPropertiesTest, ParsesFlatRuntimeSelectors)
{
    SimpleFluid::Database database;
    database.set("enable_reaction_source", true);
    database.set(
        "hydrogen_yield_mol_per_j", SimpleFluid::real_t{2.0e-7});
    database.set(
        "max_source_alpha_rate", SimpleFluid::real_t{0.2});
    database.set(
        "reacting_pressure_mode", std::string{"reconstructed"});

    const auto options =
        SimpleFluid::reacting_gas_options_from_database(database);
    EXPECT_EQ(
        options.mode, SimpleFluid::ReactingGasMode::IdealGasSource);
    EXPECT_EQ(
        options.pressure_mode,
        SimpleFluid::ReactingPressureMode::Reconstructed);
    EXPECT_DOUBLE_EQ(options.reference_pressure, 101325.0);
    EXPECT_DOUBLE_EQ(options.alpha_max, 0.95);
}

/** @brief Verifies contextual diagnostics for incorrectly typed gas options. */
TEST(ReactingGasPropertiesTest, ReportsWrongTypedOptionContext)
{
    SimpleFluid::Database database;
    database.set("enable_reaction_source", std::string{"yes"});

    try
    {
        SimpleFluid::reacting_gas_options_from_database(database);
        FAIL() << "Expected a typed reacting gas option failure.";
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message(error.what());
        EXPECT_NE(message.find("Reacting gas model"), std::string::npos);
        EXPECT_NE(message.find("enable_reaction_source"), std::string::npos);
        EXPECT_NE(message.find("wrong type"), std::string::npos);
    }
}

/** @brief Verifies that enabled reaction_source requires yield and rate-limit inputs. */
TEST(ReactingGasPropertiesTest, EnabledModeRequiresYieldAndRateLimit)
{
    SimpleFluid::Database missing;
    missing.set("enable_reaction_source", true);
    EXPECT_THROW(
        SimpleFluid::reacting_gas_options_from_database(missing),
        std::invalid_argument);

    SimpleFluid::ReactingGasOptions options;
    options.mode = SimpleFluid::ReactingGasMode::IdealGasSource;
    options.hydrogen_yield_mol_per_j = 1.0e-7;
    EXPECT_THROW(
        SimpleFluid::validate_reacting_gas_options(options),
        std::invalid_argument);
}

TEST(ReactingGasPropertiesTest, TransportToleranceIsOptionalValidatedAndParsed)
{
    EXPECT_DOUBLE_EQ(SimpleFluid::ReactingGasOptions{}.transport_solver_tolerance,1e-10);
    SimpleFluid::Database database;
    database.set("enable_reaction_source",true);
    database.set("hydrogen_yield_mol_per_j",SimpleFluid::real_t{2e-7});
    database.set("max_source_alpha_rate",SimpleFluid::real_t{1});
    database.set("reacting_transport_solver_tolerance",SimpleFluid::real_t{1e-14});
    EXPECT_DOUBLE_EQ(SimpleFluid::reacting_gas_options_from_database(database).transport_solver_tolerance,1e-14);
    for(double value:{0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
    {
        database.set("reacting_transport_solver_tolerance",value);
        EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(database),std::invalid_argument);
    }
}

} // namespace
