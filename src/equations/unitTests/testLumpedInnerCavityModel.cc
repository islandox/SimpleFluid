#include "equations/LumpedInnerCavityModel.hh"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace SimpleFluid;

namespace
{
// Synthetic saturation law for isolated balance tests, not a water property fit.
LumpedInnerCavityOptions options()
{
    LumpedInnerCavityOptions o;
    o.saturation_pressure = [](real_t t) { return 3000.0 * std::exp(0.05 * (t - 300.0)); };
    return o;
}

InnerCavitySurface surface(const char* name, real_t temperature, real_t h = 0.0)
{
    InnerCavitySurface s;
    s.name = name;
    s.temperature = temperature;
    s.convection.heat_transfer_coefficient = h;
    return s;
}

InnerCavitySurface liquid(real_t temperature)
{
    auto s = surface("liquid", temperature, 3.0);
    s.moisture_exchange = CavityMoistureExchange::LiquidSurface;
    s.mass_transfer_coefficient = 0.002;
    return s;
}
} // namespace

TEST(LumpedInnerCavityModelTest, IdealGasHumidityAndEnergyDiagnostics)
{
    const auto o = options();
    LumpedInnerCavityModel model(o, {300.0, 0.01});
    EXPECT_DOUBLE_EQ(model.vapor_mass(), 0.01);
    EXPECT_NEAR(model.vapor_pressure(), 1384.5, 1.0e-12);
    EXPECT_NEAR(model.pressure(), 87499.5, 1.0e-10);
    EXPECT_NEAR(model.relative_humidity(), 1384.5 / 3000.0, 1.0e-15);
    const auto expected = (o.dry_air_cp - o.dry_air_gas_constant) * (300.0 - o.reference_temperature) +
                          0.01 * (o.reference_latent_heat + o.vapor_cp * (300.0 - o.reference_temperature) -
                                     o.vapor_gas_constant * 300.0);
    EXPECT_NEAR(model.internal_energy(), expected, 1.0e-10);
}

TEST(LumpedInnerCavityModelTest, EmptyCavityPreservesState)
{
    LumpedInnerCavityModel model(options(), {300.0, 0.01});
    const auto step = model.advance(1.0e6, {});
    EXPECT_DOUBLE_EQ(step.state.temperature, 300.0);
    EXPECT_DOUBLE_EQ(step.state.humidity_ratio, 0.01);
    EXPECT_DOUBLE_EQ(step.mass_balance_residual, 0.0);
    EXPECT_DOUBLE_EQ(step.energy_balance_residual, 0.0);
}

TEST(LumpedInnerCavityModelTest, DryHeatingMatchesBackwardEulerAndConvergesToExponential)
{
    const auto o = options();
    const auto cv = o.dry_air_cp - o.dry_air_gas_constant;
    const std::array surfaces{surface("wall", 330.0, 5.0)};
    LumpedInnerCavityModel model(o, {290.0, 0.0});
    const auto step = model.advance(100.0, surfaces);
    EXPECT_NEAR(step.state.temperature, (cv * 290.0 + 500.0 * 330.0) / (cv + 500.0), 1.0e-12);
    EXPECT_DOUBLE_EQ(step.state.humidity_ratio, 0.0);
    const auto exact = 330.0 - 40.0 * std::exp(-5.0 * 100.0 / cv);
    const auto error = [&](int count)
    {
        LumpedInnerCavityModel refined(o, {290.0, 0.0});
        for (int i = 0; i < count; ++i)
            refined.advance(100.0 / count, surfaces);
        return std::abs(refined.state().temperature - exact);
    };
    EXPECT_LT(error(20), 0.55 * error(10));
}

TEST(LumpedInnerCavityModelTest, MoistSensibleCoolingUsesBothGasHeatCapacities)
{
    const auto o = options();
    LumpedInnerCavityModel model(o, {320.0, 0.02});
    const std::array surfaces{surface("wall", 280.0, 10.0)};
    const auto capacity = o.dry_air_cp - o.dry_air_gas_constant + 0.02 * (o.vapor_cp - o.vapor_gas_constant);
    const auto step = model.advance(10.0, surfaces);
    EXPECT_NEAR(step.state.temperature, (capacity * 320.0 + 100.0 * 280.0) / (capacity + 100.0), 1.0e-12);
    EXPECT_DOUBLE_EQ(step.state.humidity_ratio, 0.02);
}

TEST(LumpedInnerCavityModelTest, LiquidEvaporationConservesMassAndTotalEnergy)
{
    const auto o = options();
    LumpedInnerCavityModel model(o, {295.0, 0.005});
    const std::array surfaces{liquid(310.0), surface("wall", 290.0, 4.0)};
    const std::array factors{0.0, 1.0, 1.0, 0.0};
    const auto old_mass = model.vapor_mass(), old_energy = model.internal_energy();
    const auto step = model.advance(5.0, surfaces, factors);
    EXPECT_GT(step.exchange.surfaces[0].vapor_mass_rate, 0.0);
    EXPECT_GT(step.exchange.surfaces[0].latent_heat_rate, 0.0);
    EXPECT_GT(model.state().humidity_ratio, 0.005);
    EXPECT_NEAR(model.vapor_mass() - old_mass, 5.0 * step.exchange.vapor_mass_rate, 1.0e-14);
    real_t surface_energy = 0.0;
    for (const auto& s : step.exchange.surfaces)
        surface_energy += s.energy_rate;
    EXPECT_NEAR(model.internal_energy() - old_energy, 5.0 * surface_energy, 1.0e-8);
    EXPECT_NEAR(step.exchange.air_energy_rate, surface_energy, 1.0e-10);
    EXPECT_NEAR(step.mass_balance_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(step.energy_balance_residual, 0.0, 1.0e-8);
    const auto& transfer = step.exchange.surfaces[0];
    EXPECT_NEAR(transfer.energy_rate - transfer.heat_rate,
        transfer.vapor_mass_rate * o.liquid_cp * (310.0 - o.reference_temperature), 1.0e-12);
}

TEST(LumpedInnerCavityModelTest, GasFilmTransferMatchesPressureDifferenceAndImplicitMassSolution)
{
    auto o = options();
    o.volume = 0.5;
    o.dry_air_mass = 0.6;
    LumpedInnerCavityModel model(o, {300.0, 0.01});
    auto pool = liquid(310.0);
    pool.area = 0.2;
    const std::array surfaces{pool};
    const auto conductance = pool.area * pool.mass_transfer_coefficient / (o.vapor_gas_constant * 305.0);
    const auto psat = o.saturation_pressure(310.0);
    const auto rate = conductance * (psat - model.vapor_pressure());
    const auto evaluated = model.evaluate(surfaces);
    EXPECT_NEAR(evaluated.vapor_mass_rate, rate, 1.0e-18);
    EXPECT_NEAR(evaluated.surfaces[0].latent_heat_rate,
        rate * (o.reference_latent_heat + (o.vapor_cp - o.liquid_cp) * (310.0 - o.reference_temperature)), 1.0e-12);
    const auto old_mass = model.vapor_mass();
    const auto dt = 1000.0;
    const auto step = model.advance(dt, surfaces);
    const auto expected_mass = (old_mass + dt * conductance * psat) /
                               (1.0 + dt * conductance * o.vapor_gas_constant * step.state.temperature / o.volume);
    EXPECT_NEAR(model.vapor_mass(), expected_mass, 1.0e-14);
}

TEST(LumpedInnerCavityModelTest, LongStepApproachesSaturationWithoutNegativeMassOrTemperatureOvershoot)
{
    const auto o = options();
    LumpedInnerCavityModel model(o, {290.0, 0.0});
    const std::array surfaces{liquid(300.0)};
    const auto step = model.advance(1.0e6, surfaces);
    EXPECT_GT(step.state.temperature, 290.0);
    EXPECT_LE(step.state.temperature, 300.0);
    EXPECT_NEAR(step.state.temperature, 300.0, 0.01);
    EXPECT_NEAR(model.relative_humidity(), 1.0, 0.002);
    EXPECT_GE(step.state.humidity_ratio, 0.0);
    EXPECT_NEAR(step.mass_balance_residual, 0.0, 1.0e-12);
    EXPECT_NEAR(step.energy_balance_residual, 0.0, 1.0e-6);
}

TEST(LumpedInnerCavityModelTest, SaturatedIsothermalEnclosureIsAtEquilibrium)
{
    const auto o = options();
    const auto humidity = o.saturation_pressure(300.0) * o.volume / (o.vapor_gas_constant * 300.0 * o.dry_air_mass);
    LumpedInnerCavityModel model(o, {300.0, humidity});
    auto wall = surface("wall", 300.0, 5.0);
    wall.moisture_exchange = CavityMoistureExchange::CondensationOnly;
    wall.mass_transfer_coefficient = 0.01;
    wall.emissivity = 0.4;
    const std::array surfaces{liquid(300.0), wall};
    const std::array factors{0.0, 1.0, 1.0, 0.0};
    const auto step = model.advance(100.0, surfaces, factors);
    EXPECT_NEAR(step.state.temperature, 300.0, 1.0e-12);
    EXPECT_NEAR(step.state.humidity_ratio, humidity, 1.0e-14);
    for (const auto& s : step.exchange.surfaces)
    {
        EXPECT_NEAR(s.vapor_mass_rate, 0.0, 1.0e-15);
        EXPECT_NEAR(s.heat_rate, 0.0, 1.0e-9);
    }
}

TEST(LumpedInnerCavityModelTest, ColdWallCondensesButCannotEvaporate)
{
    LumpedInnerCavityModel model(options(), {310.0, 0.03});
    auto wall = surface("wall", 285.0, 5.0);
    wall.moisture_exchange = CavityMoistureExchange::CondensationOnly;
    wall.mass_transfer_coefficient = 0.02;
    const std::array surfaces{wall};
    const auto initial_mass = model.vapor_mass();
    const auto step = model.advance(500.0, surfaces);
    EXPECT_LT(model.vapor_mass(), initial_mass);
    EXPECT_GT(model.vapor_mass(), 0.0);
    EXPECT_LT(step.exchange.surfaces[0].vapor_mass_rate, 0.0);
    EXPECT_LT(step.exchange.surfaces[0].latent_heat_rate, 0.0);
    EXPECT_NEAR(step.mass_balance_residual, 0.0, 1.0e-12);
    model.set_state({300.0, 0.0});
    EXPECT_DOUBLE_EQ(model.evaluate(surfaces).vapor_mass_rate, 0.0);
    EXPECT_DOUBLE_EQ(model.advance(10.0, surfaces).state.humidity_ratio, 0.0);
}

TEST(LumpedInnerCavityModelTest, SimultaneousEvaporationAndWallCondensationConserveInventory)
{
    LumpedInnerCavityModel model(options(), {305.0, 0.025});
    auto wall = surface("wall", 280.0, 3.0);
    wall.moisture_exchange = CavityMoistureExchange::CondensationOnly;
    wall.mass_transfer_coefficient = 0.004;
    const std::array surfaces{liquid(320.0), wall};
    const auto step = model.advance(10.0, surfaces);
    EXPECT_GT(step.exchange.surfaces[0].vapor_mass_rate, 0.0);
    EXPECT_LT(step.exchange.surfaces[1].vapor_mass_rate, 0.0);
    EXPECT_NEAR(step.mass_balance_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(step.energy_balance_residual, 0.0, 1.0e-8);
}

TEST(LumpedInnerCavityModelTest, TwoGraySurfacesMatchRadiationResistanceAndDoNotHeatAir)
{
    LumpedInnerCavityModel model(options(), {300.0, 0.01});
    auto hot = surface("hot", 330.0), cold = surface("cold", 280.0);
    hot.area = cold.area = 2.0;
    hot.emissivity = 0.8;
    cold.emissivity = 0.6;
    const std::array surfaces{hot, cold};
    const std::array factors{0.0, 1.0, 1.0, 0.0};
    const auto exchange = model.evaluate(surfaces, factors);
    const auto expected =
        2.0 * 5.670374419e-8 * (std::pow(330.0, 4) - std::pow(280.0, 4)) / (1.0 / 0.8 + 1.0 / 0.6 - 1.0);
    EXPECT_NEAR(exchange.surfaces[0].radiation_heat_rate, expected, 1.0e-10);
    EXPECT_DOUBLE_EQ(exchange.surfaces[1].radiation_heat_rate, -exchange.surfaces[0].radiation_heat_rate);
    EXPECT_DOUBLE_EQ(exchange.air_energy_rate, 0.0);
    const auto step = model.advance(100.0, surfaces, factors);
    EXPECT_DOUBLE_EQ(step.state.temperature, 300.0);
    EXPECT_DOUBLE_EQ(step.state.humidity_ratio, 0.01);
}

TEST(LumpedInnerCavityModelTest, UnequalAreasAndConcaveWallMatchEnclosureResistance)
{
    LumpedInnerCavityModel model(options(), {300.0, 0.0});
    auto pool = surface("pool", 330.0), wall = surface("wall", 290.0);
    pool.emissivity = 0.7;
    wall.emissivity = 0.4;
    wall.area = 4.0;
    const std::array surfaces{pool, wall};
    const std::array factors{0.0, 1.0, 0.25, 0.75};
    const auto result = model.evaluate(surfaces, factors);
    const auto resistance = (1.0 - 0.7) / 0.7 + 1.0 + (1.0 - 0.4) / (4.0 * 0.4);
    EXPECT_NEAR(result.surfaces[0].radiation_heat_rate,
        5.670374419e-8 * (std::pow(330.0, 4) - std::pow(290.0, 4)) / resistance, 1.0e-10);
    EXPECT_DOUBLE_EQ(result.surfaces[0].radiation_heat_rate + result.surfaces[1].radiation_heat_rate, 0.0);
}

TEST(LumpedInnerCavityModelTest, SingularAndNearlySingularRadiosityPreserveState)
{
    LumpedInnerCavityModel model(options(), {300.0, 0.01});
    std::array surfaces{surface("pool", 330.0), surface("wall", 290.0)};
    const std::array factors{0.0, 1.0, 1.0, 0.0};
    // At 1e-20 the matrix rounds to exact singularity; 1e-15 exercises the
    // retained near-singularity guard after a successful LU factorization.
    for (const auto emissivity : {1.0e-20, 1.0e-15})
    {
        SCOPED_TRACE(emissivity);
        surfaces[0].emissivity = surfaces[1].emissivity = emissivity;
        EXPECT_THROW(model.advance(1.0, surfaces, factors), std::domain_error);
        EXPECT_DOUBLE_EQ(model.state().temperature, 300.0);
        EXPECT_DOUBLE_EQ(model.state().humidity_ratio, 0.01);
    }
}

TEST(LumpedInnerCavityModelTest, MultipleBlackSurfacesHaveConservativePairExchange)
{
    LumpedInnerCavityModel model(options(), {300.0, 0.0});
    const std::array surfaces{surface("pool", 340.0), surface("side", 300.0), surface("lid", 280.0)};
    const std::array factors{0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0};
    const auto result = model.evaluate(surfaces, factors);
    real_t total = 0.0;
    for (std::size_t i = 0; i < surfaces.size(); ++i)
    {
        real_t expected = 0.0;
        for (std::size_t j = 0; j < surfaces.size(); ++j)
            expected += factors[3 * i + j] * 5.670374419e-8 *
                        (std::pow(surfaces[i].temperature, 4) - std::pow(surfaces[j].temperature, 4));
        EXPECT_NEAR(result.surfaces[i].radiation_heat_rate, expected, 1.0e-10);
        total += result.surfaces[i].radiation_heat_rate;
    }
    EXPECT_NEAR(total, 0.0, 1.0e-12);
}

TEST(LumpedInnerCavityModelTest, NaturalConvectionUsesCavityTemperatureAndLagsCoefficient)
{
    LumpedInnerCavityModel model(options(), {295.0, 0.01});
    auto wall = surface("vertical-wall", 320.0);
    auto& b = wall.convection;
    b.correlation = ConvectionCorrelation::ChurchillChuVerticalPlate;
    b.ambient_temperature = 1000.0; // Must be replaced by the lumped air temperature.
    b.characteristic_length = 0.5;
    b.fluid_thermal_conductivity = 0.026;
    b.kinematic_viscosity = 1.6e-5;
    b.thermal_diffusivity = 2.3e-5;
    b.thermal_expansion = 1.0 / 307.5;
    auto reference = b;
    reference.ambient_temperature = 295.0;
    const auto expected = evaluate_convection(reference, wall.temperature).heat_transfer_coefficient;
    const std::array surfaces{wall};
    const auto step = model.advance(10.0, surfaces);
    EXPECT_DOUBLE_EQ(step.exchange.surfaces[0].heat_transfer_coefficient, expected);
    EXPECT_GT(step.state.temperature, 295.0);
    EXPECT_NEAR(step.exchange.surfaces[0].sensible_heat_rate, expected * (320.0 - step.state.temperature), 1.0e-12);
}

TEST(LumpedInnerCavityModelTest, GeometrySpecificConvectionIsAnExtensionPoint)
{
    LumpedInnerCavityModel model(options(), {300.0, 0.0});
    auto s = liquid(310.0);
    int calls = 0;
    s.convection_coefficient = [&](real_t ts, real_t ta)
    {
        ++calls;
        EXPECT_DOUBLE_EQ(ts, 310.0);
        EXPECT_DOUBLE_EQ(ta, 300.0);
        return 2.0;
    };
    const std::array surfaces{s};
    const auto step = model.preview(10.0, surfaces);
    EXPECT_EQ(calls, 1);
    EXPECT_DOUBLE_EQ(step.exchange.surfaces[0].heat_transfer_coefficient, 2.0);
    EXPECT_DOUBLE_EQ(model.state().temperature, 300.0);
}

TEST(LumpedInnerCavityModelTest, PreviewAndFailedAdvancePreserveState)
{
    LumpedInnerCavityModel model(options(), {300.0, 0.01});
    const std::array surfaces{liquid(310.0)};
    const auto preview = model.preview(1.0, surfaces);
    EXPECT_DOUBLE_EQ(model.state().temperature, 300.0);
    EXPECT_DOUBLE_EQ(model.state().humidity_ratio, 0.01);
    const auto accepted = model.advance(1.0, surfaces);
    EXPECT_DOUBLE_EQ(accepted.state.temperature, preview.state.temperature);
    EXPECT_DOUBLE_EQ(accepted.state.humidity_ratio, preview.state.humidity_ratio);
    const std::array excessive_heat{surface("hot", 1000.0, 10000.0)};
    EXPECT_THROW(model.advance(10000.0, excessive_heat), std::out_of_range);
    EXPECT_DOUBLE_EQ(model.state().temperature, accepted.state.temperature);
    EXPECT_DOUBLE_EQ(model.state().humidity_ratio, accepted.state.humidity_ratio);
    EXPECT_THROW(model.set_state({300.0, -0.1}), std::invalid_argument);
    EXPECT_DOUBLE_EQ(model.state().temperature, accepted.state.temperature);
}

TEST(LumpedInnerCavityModelTest, InvalidPropertiesAndStatesAreRejected)
{
    EXPECT_THROW((LumpedInnerCavityModel({}, {})), std::invalid_argument);
    auto o = options();
    o.volume = -1.0;
    EXPECT_THROW((LumpedInnerCavityModel(o, {})), std::invalid_argument);
    o = options();
    o.vapor_cp = o.vapor_gas_constant;
    EXPECT_THROW((LumpedInnerCavityModel(o, {})), std::invalid_argument);
    o = options();
    o.saturation_pressure = [](real_t) { return std::numeric_limits<real_t>::quiet_NaN(); };
    EXPECT_THROW((LumpedInnerCavityModel(o, {})), std::invalid_argument);
    EXPECT_THROW((LumpedInnerCavityModel(options(), {200.0, 0.01})), std::out_of_range);
    EXPECT_THROW((LumpedInnerCavityModel(options(), {300.0, -0.01})), std::invalid_argument);
}

TEST(LumpedInnerCavityModelTest, InvalidSurfacesAndRadiationAreRejectedTransactionally)
{
    LumpedInnerCavityModel model(options(), {300.0, 0.01});
    std::array surfaces{surface("liquid", 310.0), surface("wall", 290.0)};
    const auto reject = [&](std::span<const real_t> factors)
    {
        EXPECT_THROW(model.advance(1.0, surfaces, factors), std::invalid_argument);
        EXPECT_DOUBLE_EQ(model.state().temperature, 300.0);
        EXPECT_DOUBLE_EQ(model.state().humidity_ratio, 0.01);
    };
    reject(std::array{0.0, 1.0, 1.0});
    reject(std::array{0.0, 0.5, 0.5, 0.0}); // Open enclosure.
    reject(std::array{0.0, 1.0, 0.5, 0.5}); // Nonreciprocal.
    reject(std::array{0.0, -1.0, -1.0, 0.0});
    surfaces[1].emissivity = 0.0;
    reject(std::array{0.0, 1.0, 1.0, 0.0});
    surfaces[1].emissivity = 1.0;
    surfaces[1].name = "liquid";
    reject({});
    surfaces[1].name = "wall";
    surfaces[1].area = 0.0;
    reject({});
    surfaces[1].area = 1.0;
    surfaces[1].convection_coefficient = [](real_t, real_t) { return -1.0; };
    reject({});
    EXPECT_THROW(model.advance(0.0, {}), std::invalid_argument);
    EXPECT_THROW(model.advance(std::numeric_limits<real_t>::quiet_NaN(), {}), std::invalid_argument);
}

TEST(LumpedInnerCavityModelTest, MaterialCallbackFailureDoesNotCommitState)
{
    auto o = options();
    o.saturation_pressure = [](real_t t)
    {
        if (t != 300.0)
            throw std::domain_error("Material domain exceeded");
        return 3000.0;
    };
    LumpedInnerCavityModel model(o, {300.0, 0.01});
    const std::array surfaces{surface("wall", 310.0, 5.0)};
    EXPECT_THROW(model.advance(1.0, surfaces), std::domain_error);
    EXPECT_DOUBLE_EQ(model.state().temperature, 300.0);
    EXPECT_DOUBLE_EQ(model.state().humidity_ratio, 0.01);
}

TEST(LumpedInnerCavityModelTest, SupersaturationIsReportedWithoutDiscardingWater)
{
    LumpedInnerCavityModel model(options(), {300.0, 0.1});
    EXPECT_GT(model.relative_humidity(), 1.0);
    const auto step = model.advance(10.0, {});
    EXPECT_DOUBLE_EQ(step.state.humidity_ratio, 0.1);
}
