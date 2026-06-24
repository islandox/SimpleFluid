#include <gtest/gtest.h>

#include "equations/RadiolyticGasProperties.hh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

namespace Physics = SimpleFluid::RadiolyticGasPhysics;

TEST(RadiolyticGasPropertiesTest, IdealGasSourceHasExpectedScaling)
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

TEST(RadiolyticGasPropertiesTest, RejectsInvalidThermodynamicInputs)
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

TEST(RadiolyticGasPropertiesTest, HenryAndLaplaceTermsAreMonotone)
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

TEST(RadiolyticGasPropertiesTest, PublishedPressureCorrectionRegression)
{
    EXPECT_NEAR(
        Physics::pressure_nucleation_correction(101325.0, 101325.0),
        0.99936965,
        1.0e-12);
    EXPECT_LT(
        Physics::pressure_nucleation_correction(2.0 * 101325.0, 101325.0),
        Physics::pressure_nucleation_correction(101325.0, 101325.0));
}

TEST(RadiolyticGasPropertiesTest, WinterLetUsesMolPerCubicMetre)
{
    constexpr double temperature = 298.15;
    constexpr double concentration = 1000.0;
    const auto expected =
        (-1.3387e-6 * temperature - 3.4319e-5) * concentration
      - 6.6431e-3 * temperature + 8.8142;

    EXPECT_NEAR(
        Physics::mean_fission_fragment_let(
            temperature, concentration),
        expected,
        1.0e-13);
    EXPECT_LT(
        Physics::mean_fission_fragment_let(
            temperature, concentration),
        Physics::mean_fission_fragment_let(temperature, 1.0));
}

TEST(RadiolyticGasPropertiesTest, ShengSurfaceTensionUsesMolPerCubicMetre)
{
    constexpr double temperature_celsius = 25.0;
    constexpr double concentration = 1000.0;
    const auto expected =
        1.7160e-7 * temperature_celsius * temperature_celsius
      - 1.4427e-4 * temperature_celsius
      + 2.0163e-6 * concentration + 7.5725e-2;

    EXPECT_NEAR(
        Physics::sheng2024_surface_tension(
            temperature_celsius, concentration),
        expected,
        1.0e-15);
}

TEST(RadiolyticGasPropertiesTest, WinterYieldCorrectionChecksRange)
{
    EXPECT_THROW(
        Physics::atmospheric_nucleation_radius(1.0e-9, 0.5),
        std::invalid_argument);
    EXPECT_NO_THROW(
        Physics::atmospheric_nucleation_radius(1.0e-9, 1.8));
    EXPECT_THROW(
        Physics::atmospheric_nucleation_radius(1.0e-9, 4.5),
        std::invalid_argument);
}

TEST(RadiolyticGasPropertiesTest, HughmarkBranchesAndValidity)
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

TEST(RadiolyticGasPropertiesTest, CelataRiseVelocityBalancesDrag)
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

TEST(RadiolyticGasPropertiesTest, BubbleRadiusSolvesEos)
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

TEST(RadiolyticGasPropertiesTest, BubbleRadiusTrendsWithState)
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

TEST(RadiolyticGasPropertiesTest, VoidAndCharacteristicRadiusReconstruct)
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

TEST(RadiolyticGasPropertiesTest, ParsesFlatRuntimeSelectors)
{
    SimpleFluid::Database database;
    database.set("enable_radiolysis", true);
    database.set(
        "hydrogen_yield_mol_per_j", SimpleFluid::real_t{2.0e-7});
    database.set(
        "max_source_alpha_rate", SimpleFluid::real_t{0.2});
    database.set(
        "radiolytic_pressure_mode", std::string{"reconstructed"});

    const auto options =
        SimpleFluid::radiolytic_gas_options_from_database(database);
    EXPECT_EQ(
        options.mode, SimpleFluid::RadiolyticGasMode::IdealGasSource);
    EXPECT_EQ(
        options.pressure_mode,
        SimpleFluid::RadiolyticPressureMode::Reconstructed);
    EXPECT_DOUBLE_EQ(options.reference_pressure, 101325.0);
    EXPECT_DOUBLE_EQ(options.alpha_max, 0.95);
}

TEST(RadiolyticGasPropertiesTest, EnabledModeRequiresYieldAndRateLimit)
{
    SimpleFluid::Database missing;
    missing.set("enable_radiolysis", true);
    EXPECT_THROW(
        SimpleFluid::radiolytic_gas_options_from_database(missing),
        std::invalid_argument);

    SimpleFluid::RadiolyticGasOptions options;
    options.mode = SimpleFluid::RadiolyticGasMode::IdealGasSource;
    options.hydrogen_yield_mol_per_j = 1.0e-7;
    EXPECT_THROW(
        SimpleFluid::validate_radiolytic_gas_options(options),
        std::invalid_argument);
}

} // namespace
