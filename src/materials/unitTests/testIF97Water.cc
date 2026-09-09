#include "materials/IF97Water.hh"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
namespace Water = SimpleFluid::IF97Water;

// Independent published computer verification points, IAPWS R7-97(2012),
// Tables 5, 15 and 33. https://iapws.org/technical-guidance/release/IF97-Rev
TEST(IF97WaterTest, MatchesPublishedThermodynamicTablesInSI)
{
    struct Point
    {
        double t, p, volume, h, s, cp, tolerance;
    };
    const Point points[]{{300, 3e6, 0.00100215168, 115331.273, 392.294792, 4173.01218, 2e-8},
        {500, 3e6, 0.00120241800, 975542.239, 2580.41912, 4655.80682, 2e-8},
        {700, 3500, 92.3015898, 3335683.75, 10174.9996, 2081.41274, 2e-8},
        {700, 30e6, 0.00542946619, 2631494.74, 5175.40298, 10350.5092, 2e-8},
        // Region 3 uses upstream's direct backward density approximation.
        {650, 25.5837018e6, 0.002, 1863430.19, 4054.27273, 13893.5717, 5e-5}};
    for (const auto& point : points)
    {
        SCOPED_TRACE(point.t);
        SCOPED_TRACE(point.p);
        const auto state = Water::evaluate(point.t, point.p);
        EXPECT_NEAR(1.0 / state.density, point.volume, point.tolerance * point.volume);
        EXPECT_NEAR(state.specific_enthalpy, point.h, point.tolerance * point.h);
        EXPECT_NEAR(state.specific_entropy, point.s, point.tolerance * point.s);
        EXPECT_NEAR(state.specific_heat_capacity, point.cp, point.tolerance * point.cp);
    }
}

TEST(IF97WaterTest, RoomTemperatureTransportAndDerivedCoefficients)
{
    const auto water = Water::liquid(298.15, 101325.0);
    EXPECT_NEAR(water.density, 997.05, 0.03);
    EXPECT_NEAR(water.specific_heat_capacity, 4181.9, 1.0);
    EXPECT_NEAR(water.dynamic_viscosity, 0.0008900, 1.0e-6);
    EXPECT_NEAR(water.thermal_conductivity, 0.6065, 0.001);
    EXPECT_NEAR(water.kinematic_viscosity(), 8.927e-7, 2.0e-9);
    EXPECT_NEAR(water.thermal_diffusivity(), 1.4545e-7, 3.0e-10);
    EXPECT_GT(Water::liquid(350.0, 101325.0).thermal_diffusivity(), water.thermal_diffusivity());
    EXPECT_LT(Water::liquid(350.0, 101325.0).dynamic_viscosity, water.dynamic_viscosity);
}

TEST(IF97WaterTest, MatchesPublishedSaturationTables)
{
    // IAPWS R7-97(2012), Tables 35 and 36: pressure MPa -> Pa.
    EXPECT_NEAR(Water::saturation_pressure(300), 0.00353658941e6, 1e-5);
    EXPECT_NEAR(Water::saturation_pressure(500), 2.63889776e6, 0.01);
    EXPECT_NEAR(Water::saturation_pressure(600), 12.3443146e6, 0.1);
    EXPECT_NEAR(Water::saturation_temperature(1e5), 372.755919, 1e-6);
    EXPECT_NEAR(Water::saturation_temperature(1e6), 453.035632, 1e-6);
    EXPECT_NEAR(Water::saturation_temperature(1e7), 584.149488, 1e-6);
}

TEST(IF97WaterTest, KeepsSaturatedLiquidAndSteamSeparate)
{
    const auto sat = Water::saturation_at_pressure(101325.0);
    EXPECT_NEAR(sat.liquid.temperature, 373.1243, 1e-4);
    EXPECT_DOUBLE_EQ(sat.liquid.temperature, sat.vapor.temperature);
    EXPECT_DOUBLE_EQ(sat.liquid.absolute_pressure, 101325.0);
    EXPECT_NEAR(sat.liquid.density, 958.37, 0.1);
    EXPECT_NEAR(sat.vapor.density, 0.5976, 0.001);
    EXPECT_NEAR(sat.latent_heat(), 2.2565e6, 300.0);
    EXPECT_NEAR(sat.surface_tension, 0.05892, 0.0001);
    EXPECT_THROW(Water::evaluate(sat.liquid.temperature, 101325.0), std::out_of_range);
    EXPECT_DOUBLE_EQ(Water::liquid(sat.liquid.temperature, 101325.0).density, sat.liquid.density);
    EXPECT_THROW(Water::liquid(sat.liquid.temperature + 0.001, 101325.0), std::out_of_range);
    const auto dense = Water::saturation_at_pressure(20e6);
    EXPECT_GT(dense.liquid.density, dense.vapor.density);
    EXPECT_GT(dense.latent_heat(), 0.0);
}

TEST(IF97WaterTest, ThermalExpansionRespectsLiquidBranchAndDensityAnomaly)
{
    EXPECT_NEAR(Water::liquid_thermal_expansion(300, 3e6), 2.7735e-4, 1e-8);
    EXPECT_LT(Water::liquid_thermal_expansion(273.15, 101325.0), 0.0);
    const double ts = Water::saturation_temperature(101325.0);
    EXPECT_NEAR(Water::liquid_thermal_expansion(ts, 101325.0), 7.508e-4, 2e-6);
    EXPECT_NEAR(
        Water::liquid_thermal_expansion(ts, 101325.0), Water::liquid_thermal_expansion(ts - 0.01, 101325.0), 2e-7);
    // Differencing across the separate region 1/3 density fits would give
    // a negative, nonphysical expansion coefficient at this boundary.
    for (const double t : {623.1499, 623.15, 623.1501})
        EXPECT_NEAR(Water::liquid_thermal_expansion(t, 20e6), 0.006995, 2e-5);
}

TEST(IF97WaterTest, RejectsInvalidInputsAndUnsupportedPhases)
{
    for (const double value :
        {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_THROW(Water::evaluate(value, 101325.0), std::invalid_argument);
        EXPECT_THROW(Water::evaluate(300.0, value), std::invalid_argument);
        EXPECT_THROW(Water::saturation_temperature(value), std::invalid_argument);
        EXPECT_THROW(Water::saturation_pressure(value), std::invalid_argument);
    }
    EXPECT_THROW(Water::liquid(400.0, 101325.0), std::out_of_range);
    EXPECT_THROW(Water::liquid(650.0, 25e6), std::out_of_range);
    EXPECT_THROW(Water::evaluate(273.14, 101325.0), std::out_of_range);
    EXPECT_THROW(Water::evaluate(1100.0, 101325.0), std::out_of_range);
    EXPECT_THROW(Water::evaluate(300.0, 100.0), std::out_of_range);
    EXPECT_THROW(Water::evaluate(300.0, 101e6), std::out_of_range);
    EXPECT_THROW(Water::saturation_temperature(25e6), std::out_of_range);
    EXPECT_THROW(Water::saturation_at_pressure(22.064e6), std::out_of_range);
    EXPECT_THROW(Water::surface_tension(650.0), std::out_of_range);
}

TEST(IF97WaterTest, SaturationEndpointsStayWithinThePublishedRange)
{
    const auto pmin = Water::saturation_pressure(273.15);
    EXPECT_NEAR(Water::saturation_temperature(pmin), 273.15, 1e-5);
    EXPECT_NO_THROW(Water::saturation_at_pressure(pmin));
    EXPECT_NEAR(Water::saturation_temperature(Water::saturation_pressure(273.150001)), 273.150001, 1e-5);
    EXPECT_NEAR(Water::saturation_temperature(Water::saturation_pressure(647.096)), 647.096, 1e-7);
    EXPECT_DOUBLE_EQ(Water::surface_tension(647.096), 0.0);
    EXPECT_NO_THROW(Water::evaluate(1073.15, 1e8));
    EXPECT_FALSE(Water::backend_version().empty());
}

TEST(IF97WaterTest, ConcurrentSaturationQueriesMatchSerialResults)
{
    const std::array<double, 4> pressures{1e5, 1e6, 1e7, 20e6};
    std::vector<std::future<bool>> workers;
    for (const auto pressure : pressures)
    {
        const auto expected = Water::saturation_at_pressure(pressure);
        workers.push_back(std::async(std::launch::async,
            [pressure, expected]
            {
                for (int i = 0; i < 100; ++i)
                {
                    const auto actual = Water::saturation_at_pressure(pressure);
                    if (actual.liquid.density != expected.liquid.density ||
                        actual.vapor.density != expected.vapor.density)
                        return false;
                }
                return true;
            }));
    }
    for (auto& worker : workers)
        EXPECT_TRUE(worker.get());
}
} // namespace
