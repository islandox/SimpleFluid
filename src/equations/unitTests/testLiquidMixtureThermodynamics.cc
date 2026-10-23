/** @file testLiquidMixtureThermodynamics.cc
 * @brief Independent caloric/EOS identities and declared-domain rejection through the shared mixture API.
 */
#include "equations/ReactingThermodynamics.hh"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace
{
namespace V = SimpleFluid::Reacting;
const V::ThermodynamicBounds bounds{250.0, 400.0, 10000.0, 1000000.0};
const std::array<double, 1> composition{1};
std::shared_ptr<const V::SpeciesCatalog> catalog()
{
    static const auto species = std::make_shared<const V::SpeciesCatalog>(
        std::vector<V::SpeciesDefinition>{{14, "synthetic-liquid", .018, {{"X", 1}}, {V::SpeciesPhase::Liquid}}});
    return species;
}
V::LinearCompressibleMixtureLiquid parameters()
{
    // A synthetic local EOS fixture, not a material-specific correlation.
    V::LinearCompressibleMixtureLiquid p;
    p.reference_temperature = 300.0;
    p.reference_pressure = 100000.0;
    p.reference_density = 1000.0;
    p.isobaric_heat_capacity = 4000.0;
    p.thermal_expansion = 2e-4;
    p.compressibility = 5e-9;
    p.reference_internal_energy = 100000.0;
    p.dynamic_viscosity = .001;
    p.thermal_conductivity = .6;
    return p;
}
std::shared_ptr<const V::MixtureThermodynamics> make_liquid(V::LinearCompressibleMixtureLiquid p)
{
    return V::make_linear_compressible_mixture_liquid(catalog(), bounds, [p](std::span<const double>) { return p; });
}

TEST(LiquidMixtureThermodynamicsTest, ReferenceStateAndEnthalpyIdentityUseTheSuppliedEnergyReference)
{
    const auto p = parameters();
    const auto eos = make_liquid(p);
    const auto reference = eos->evaluate(p.reference_temperature, p.reference_pressure, composition);
    EXPECT_DOUBLE_EQ(reference.density, p.reference_density);
    EXPECT_DOUBLE_EQ(reference.internal_energy, p.reference_internal_energy);
    EXPECT_DOUBLE_EQ(reference.enthalpy, p.reference_internal_energy + p.reference_pressure / p.reference_density);
    EXPECT_DOUBLE_EQ(reference.dynamic_viscosity, p.dynamic_viscosity);
    EXPECT_DOUBLE_EQ(reference.thermal_conductivity, p.thermal_conductivity);
    for (const double temperature : {250.0, 311.0, 400.0})
        for (const double pressure : {10000.0, 257000.0, 1000000.0})
        {
            const auto point = eos->evaluate(temperature, pressure, composition);
            EXPECT_NEAR(point.enthalpy - point.internal_energy, pressure / point.density, 2e-10);
            EXPECT_GT(point.density, 0.0);
            EXPECT_GT(point.internal_energy_temperature, 0.0);
        }
}

TEST(LiquidMixtureThermodynamicsTest, IndependentDerivativesSatisfyTheGibbsIdentities)
{
    const auto p = parameters();
    const auto eos = make_liquid(p);
    constexpr double temperature = 323.0, pressure = 240000.0, dT = .01, dP = 100.0;
    const auto center = eos->evaluate(temperature, pressure, composition);
    const auto hot = eos->evaluate(temperature + dT, pressure, composition);
    const auto cold = eos->evaluate(temperature - dT, pressure, composition);
    const auto high = eos->evaluate(temperature, pressure + dP, composition);
    const auto low = eos->evaluate(temperature, pressure - dP, composition);
    const double dv_dT = (1.0 / hot.density - 1.0 / cold.density) / (2.0 * dT);
    const double dv_dP = (1.0 / high.density - 1.0 / low.density) / (2.0 * dP);
    const double dh_dP = (high.enthalpy - low.enthalpy) / (2.0 * dP);
    const double de_dP = (high.internal_energy - low.internal_energy) / (2.0 * dP);
    EXPECT_NEAR(dv_dT, p.thermal_expansion / p.reference_density, 2e-17);
    EXPECT_NEAR(dv_dP, -p.compressibility / p.reference_density, 2e-21);
    EXPECT_NEAR((hot.enthalpy - cold.enthalpy) / (2.0 * dT), p.isobaric_heat_capacity, 5e-8);
    EXPECT_NEAR((hot.internal_energy - cold.internal_energy) / (2.0 * dT), center.internal_energy_temperature, 5e-8);
    EXPECT_NEAR(dh_dP, 1.0 / center.density - temperature * dv_dT, 1e-12);
    EXPECT_NEAR(de_dP, -temperature * dv_dT - pressure * dv_dP, 1e-12);
}

TEST(LiquidMixtureThermodynamicsTest, ClosedVolumeEnergyChangeMatchesIntegratedIsochoricHeatCapacity)
{
    const auto p = parameters();
    const auto eos = make_liquid(p);
    const double temperature = p.reference_temperature + 7.0;
    const double pressure = p.reference_pressure + p.thermal_expansion / p.compressibility * 7.0;
    const auto initial = eos->evaluate(p.reference_temperature, p.reference_pressure, composition);
    const auto final = eos->evaluate(temperature, pressure, composition);
    const double integrated_cv = p.isobaric_heat_capacity * 7.0 -
                                 p.thermal_expansion * p.thermal_expansion /
                                     (2.0 * p.reference_density * p.compressibility) *
                                     (temperature * temperature - p.reference_temperature * p.reference_temperature);
    EXPECT_NEAR(final.density, initial.density, 2e-12);
    EXPECT_NEAR(final.internal_energy - initial.internal_energy, integrated_cv, 5e-10);
}

TEST(LiquidMixtureThermodynamicsTest, MissingDomainsInvalidCaloricDataAndInconsistentEnthalpyReject)
{
    const auto p = parameters();
    const auto good = make_liquid(p);
    EXPECT_THROW((void) V::MixtureThermodynamics(catalog(), {}, V::EOSKind::Liquid, {}), std::invalid_argument);
    EXPECT_THROW((void) V::MixtureThermodynamics(catalog(), bounds, V::EOSKind::Liquid, {}), std::invalid_argument);
    EXPECT_THROW((void) good->evaluate(249.0, p.reference_pressure, composition), std::invalid_argument);
    EXPECT_THROW((void) good->evaluate(300.0, 1000001.0, composition), std::invalid_argument);
    EXPECT_THROW((void) good->evaluate(std::numeric_limits<double>::quiet_NaN(), p.reference_pressure, composition),
        std::invalid_argument);
    const V::MixtureThermodynamics bad_enthalpy(catalog(), bounds, V::EOSKind::Liquid,
        [good](double t, double pressure, std::span<const double> x)
        {
            auto point = good->evaluate(t, pressure, x);
            point.enthalpy += 1;
            return point;
        });
    EXPECT_THROW((void) bad_enthalpy.evaluate(300, p.reference_pressure, composition), std::invalid_argument);
    const V::MixtureThermodynamics bad_capacity(catalog(), bounds, V::EOSKind::Liquid,
        [good](double t, double pressure, std::span<const double> x)
        {
            auto point = good->evaluate(t, pressure, x);
            point.internal_energy_temperature = -1;
            return point;
        });
    EXPECT_THROW((void) bad_capacity.evaluate(300, p.reference_pressure, composition), std::invalid_argument);
    auto unstable = p;
    unstable.compressibility = 1e-16;
    EXPECT_THROW((void) make_liquid(unstable)->evaluate(300, p.reference_pressure, composition), std::invalid_argument);
    auto negative_volume = p;
    negative_volume.compressibility = 1e-4;
    EXPECT_THROW((void) make_liquid(negative_volume)->evaluate(300, bounds.maximum_pressure, composition),
        std::invalid_argument);
}
} // namespace
