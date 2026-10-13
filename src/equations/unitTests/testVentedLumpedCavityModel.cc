#include "equations/VentedLumpedCavityModel.hh"
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace SimpleFluid;
namespace
{
InnerCavitySurface wall(real_t t, real_t h)
{
    InnerCavitySurface s;
    s.temperature = t;
    s.convection.heat_transfer_coefficient = h;
    return s;
}
}
TEST(VentedLumpedCavityModelTest, HeatingMatchesAnalyticImplicitBalance)
{
    VentedLumpedCavityOptions o;
    VentedLumpedCavityModel m(o, {300.0, 1.0});
    const auto capacity = m.dry_air_mass() * o.dry_air_cp;
    const std::array surfaces{wall(350.0, 10.0)};
    const auto step = m.preview(20.0, 1.0, surfaces);
    EXPECT_NEAR(step.state.temperature, (capacity * 300.0 + 200.0 * 350.0) / (capacity + 200.0), 1e-10);
    EXPECT_LT(step.vent_mass_transfer, 0.0);
    EXPECT_GT(step.exchange.surfaces[0].heat_rate, 0.0);
    EXPECT_NEAR(step.energy_balance_residual, 0.0, 1e-8);
    EXPECT_DOUBLE_EQ(m.state().temperature, 300.0);
}
TEST(VentedLumpedCavityModelTest, CoolingMatchesAnalyticInflowQuadratic)
{
    VentedLumpedCavityOptions o;
    o.ambient_temperature = 290.0;
    VentedLumpedCavityModel m(o, {330.0, 1.0});
    const auto c = o.dry_air_cp * o.ambient_pressure / o.dry_air_gas_constant;
    const std::array surfaces{wall(280.0, 10.0)};
    // 100 T^2 + (c Ta/Told - 100 Ts) T - c Ta = 0.
    const auto b = c * o.ambient_temperature / 330.0 - 100.0 * 280.0;
    const auto exact = (-b + std::sqrt(b * b + 400.0 * c * o.ambient_temperature)) / 200.0;
    const auto step = m.advance(10.0, 1.0, surfaces);
    EXPECT_NEAR(step.state.temperature, exact, 1e-10);
    EXPECT_GT(step.vent_mass_transfer, 0.0);
    EXPECT_LT(step.exchange.surfaces[0].heat_rate, 0.0);
    EXPECT_NEAR(step.vent_enthalpy_transfer, step.vent_mass_transfer * o.dry_air_cp * 290.0, 1e-10);
    EXPECT_NEAR(step.energy_balance_residual, 0.0, 1e-8);
}
TEST(VentedLumpedCavityModelTest, ExpandingVolumeMixesAmbientInflow)
{
    VentedLumpedCavityOptions o;
    o.ambient_temperature = 280.0;
    VentedLumpedCavityModel m(o, {320.0, 1.0});
    const auto old_u = m.internal_energy();
    const auto step = m.preview(1.0, 2.0, {});
    const auto exact = 2.0 / (1.0 / 320.0 + 1.0 / 280.0);
    EXPECT_NEAR(step.state.temperature, exact, 1e-10);
    EXPECT_GT(step.vent_mass_transfer, 0.0);
    VentedLumpedCavityModel final(o, step.state);
    EXPECT_NEAR(final.internal_energy() - old_u - step.vent_enthalpy_transfer + step.volume_work, 0.0, 1e-8);
    EXPECT_NEAR(step.mass_balance_residual, 0.0, 1e-14);
}
TEST(VentedLumpedCavityModelTest, ContractingVolumeVentsAtCavityTemperature)
{
    VentedLumpedCavityModel m({}, {320.0, 2.0});
    const auto step = m.preview(1.0, 1.0, {});
    EXPECT_NEAR(step.state.temperature, 320.0, 1e-10);
    EXPECT_LT(step.vent_mass_transfer, 0.0);
    // The computed final temperature differs from the analytic constant value
    // by floating-point root cancellation. Scale this product check by its ULP.
    const auto expected_vent = step.vent_mass_transfer * 1005.0 * 320.0;
    EXPECT_NEAR(step.vent_enthalpy_transfer, expected_vent,
        8.0 * std::numeric_limits<real_t>::epsilon() * std::abs(expected_vent));
    EXPECT_NEAR(step.energy_balance_residual, 0.0, 1e-8);
}
TEST(VentedLumpedCavityModelTest, FailedPreviewAndAdvancePreserveState)
{
    VentedLumpedCavityOptions o;
    o.maximum_temperature = 310.0;
    VentedLumpedCavityModel m(o, {300.0, 1.0});
    const std::array hot{wall(500.0, 1e6)};
    EXPECT_THROW(m.preview(100.0, 1.0, hot), std::domain_error);
    EXPECT_THROW(m.advance(100.0, 1.0, hot), std::domain_error);
    EXPECT_THROW(m.preview(-1.0, 1.0, {}), std::invalid_argument);
    EXPECT_THROW(m.preview(1.0, 0.0, {}), std::invalid_argument);
    auto wet = wall(300.0, 1.0);
    wet.moisture_exchange = CavityMoistureExchange::LiquidSurface;
    EXPECT_THROW(m.preview(1.0, 1.0, std::array{wet}), std::invalid_argument);
    EXPECT_THROW(m.preview(1.0, std::numeric_limits<real_t>::max(), {}), std::domain_error);
    EXPECT_DOUBLE_EQ(m.state().temperature, 300.0);
    EXPECT_DOUBLE_EQ(m.state().volume, 1.0);
}
TEST(VentedLumpedCavityModelTest, EmptyFixedVolumePreservesStateExactly)
{
    VentedLumpedCavityModel m({}, {317.3, 0.37});
    const auto step = m.advance(1e6, 0.37, {});
    EXPECT_DOUBLE_EQ(step.state.temperature, 317.3);
    EXPECT_DOUBLE_EQ(step.vent_mass_transfer, 0.0);
    EXPECT_DOUBLE_EQ(step.energy_balance_residual, 0.0);
}

TEST(VentedLumpedCavityModelTest, UnderflowInventoryIsRejectedTransactionally)
{
    VentedLumpedCavityOptions o;
    o.ambient_pressure = 1e-300;
    VentedLumpedCavityModel model(o, {300.0, 1.0});
    EXPECT_THROW(model.set_state({300.0, 1e-300}), std::invalid_argument);
    EXPECT_THROW(model.advance(1.0, 1e-300, {}), std::domain_error);
    EXPECT_DOUBLE_EQ(model.state().temperature, 300.0);
    EXPECT_DOUBLE_EQ(model.state().volume, 1.0);
}
