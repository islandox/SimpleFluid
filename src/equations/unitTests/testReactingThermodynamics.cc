#include "equations/ReactingThermodynamics.hh"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace
{
namespace R = SimpleFluid::Reacting;
constexpr double gas_constant = 8.31446261815324;
const R::ThermodynamicBounds bounds{250, 600, 10000, 2000000};

std::shared_ptr<const R::SpeciesCatalog> catalog()
{
    const std::vector phases{
        R::SpeciesPhase::Liquid, R::SpeciesPhase::Dissolved, R::SpeciesPhase::Dispersed, R::SpeciesPhase::Resolved};
    return std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{11, "solvent-vapor", 0.018, {{"A", 1}}, phases},
            {27, "noncondensable", 0.028, {{"B", 1}}, phases}, {91, "third-gas", 0.040, {{"C", 1}}, phases}});
}

std::shared_ptr<const R::MixtureThermodynamics> ideal(std::shared_ptr<const R::SpeciesCatalog> species)
{
    return R::make_ideal_gas_mixture(std::move(species), {{25, 300, 60000}, {20, 300, 1200}, {30, 300, 2000}}, bounds);
}

std::shared_ptr<const R::MixtureThermodynamics> liquid(std::shared_ptr<const R::SpeciesCatalog> species)
{
    return R::make_linear_compressible_mixture_liquid(std::move(species), bounds,
        [](std::span<const double> x)
        {
            // Entirely synthetic composition dependence, not a material correlation.
            return R::LinearCompressibleMixtureLiquid{
                300, 100000, 1000 + 100 * x[1], 4200, 2e-4, 5e-9, 10000, 0.001, 0.6};
        });
}

R::PhaseThermalState phase(std::shared_ptr<const R::MixtureThermodynamics> eos, std::vector<double> moles,
    double temperature, double pressure = 100000)
{
    const double n = std::accumulate(moles.begin(), moles.end(), 0.0);
    std::vector<double> x(moles.size());
    double mass = 0;
    for (std::size_t i = 0; i < moles.size(); ++i)
    {
        x[i] = n > 0 ? moles[i] / n : 0;
        mass += moles[i] * eos->catalog().species()[i].molarMass;
    }
    const double energy = n > 0 ? mass * eos->evaluate(temperature, pressure, x).internal_energy : 0;
    return {std::move(eos), std::move(moles), pressure, energy, n > 0 ? temperature : 0};
}

std::shared_ptr<const R::MixtureThermodynamics> covolume(std::shared_ptr<const R::SpeciesCatalog> species)
{
    // Independent nonideal fixture: p(V-n*b)=nRT, e=cv*(T-300)/M; ln(phi)=bp/(RT).
    return std::make_shared<const R::MixtureThermodynamics>(species, bounds, R::EOSKind::Gas,
        [species](double t, double p, std::span<const double> x)
        {
            constexpr double b = 1e-3, cv = 24;
            double mw = 0;
            for (std::size_t i = 0; i < x.size(); ++i)
                mw += x[i] * species->species()[i].molarMass;
            const double v = (gas_constant * t / p + b) / mw;
            const double rho = 1 / v;
            R::MixtureThermodynamicPoint result{rho, cv * (t - 300) / mw, cv * (t - 300) / mw + p * v, cv / mw, 0,
                -gas_constant / (mw * p * v * v), gas_constant * t / (mw * p * p * v * v), {}, {}};
            for (double fraction : x)
                result.fugacities.push_back(fraction * p * std::exp(b * p / (gas_constant * t)));
            return result;
        });
}

TEST(ReactingThermodynamicsTest, ArbitraryIdealMixtureMatchesIndependentMolarEOSAndCaloricReference)
{
    const auto species = catalog();
    const auto gas = ideal(species);
    const std::vector<double> x{0.2, 0.3, 0.5};
    constexpr double t = 410, p = 350000;
    const auto point = gas->evaluate(t, p, x);
    const double mw = 0.2 * 0.018 + 0.3 * 0.028 + 0.5 * 0.040;
    const double cv = 0.2 * 25 + 0.3 * 20 + 0.5 * 30;
    const double reference = 0.2 * 60000 + 0.3 * 1200 + 0.5 * 2000;
    EXPECT_NEAR(point.density, p * mw / (gas_constant * t), 1e-13);
    EXPECT_NEAR(point.internal_energy, (reference + cv * 110) / mw, 1e-9);
    EXPECT_NEAR(point.enthalpy - point.internal_energy, gas_constant * t / mw, 1e-9);
    EXPECT_NEAR(point.internal_energy_temperature, cv / mw, 1e-11);
    EXPECT_DOUBLE_EQ(point.fugacities[1], 0.3 * p);
    auto parcels = std::vector{phase(gas, {2, 3, 5}, t, p)};
    const auto recovered = R::recover_phase_temperatures(parcels, parcels[0].internal_energy);
    EXPECT_NEAR(recovered.phases[0].temperature, t, 1e-7);
    EXPECT_NEAR(recovered.total_volume, 10 * gas_constant * t / p, 1e-11);
    EXPECT_NEAR(recovered.volume_pressure_susceptibility, -10 * gas_constant * t / (p * p), 1e-14);
}

TEST(ReactingThermodynamicsTest, CompositionAwareLiquidDerivativesAndTransportComeFromSuppliedProperties)
{
    const auto eos = liquid(catalog());
    const std::vector<double> x{0.4, 0.6, 0};
    constexpr double t = 350, p = 230000, dt = 0.01, dp = 10;
    const auto center = eos->evaluate(t, p, x);
    const auto hot = eos->evaluate(t + dt, p, x), cold = eos->evaluate(t - dt, p, x);
    const auto high = eos->evaluate(t, p + dp, x), low = eos->evaluate(t, p - dp, x);
    EXPECT_NEAR((hot.internal_energy - cold.internal_energy) / (2 * dt), center.internal_energy_temperature, 2e-7);
    EXPECT_NEAR((high.internal_energy - low.internal_energy) / (2 * dp), center.internal_energy_pressure, 2e-11);
    EXPECT_NEAR((hot.density - cold.density) / (2 * dt), center.density_temperature, 1e-10);
    EXPECT_NEAR((high.density - low.density) / (2 * dp), center.density_pressure, 1e-12);
    EXPECT_DOUBLE_EQ(center.dynamic_viscosity, 0.001);
    EXPECT_DOUBLE_EQ(center.thermal_conductivity, 0.6);
    EXPECT_DOUBLE_EQ(eos->evaluate(300, 100000, x).density, 1060);
}

TEST(ReactingThermodynamicsTest, IndependentNonidealFixtureHasCovolumeFugacityAndConservedEnergyPressureInversion)
{
    const auto species = catalog();
    const auto gas = covolume(species);
    const std::vector<double> x{0.25, 0.75, 0};
    const auto point = gas->evaluate(400, 300000, x);
    const double mw = 0.25 * 0.018 + 0.75 * 0.028;
    EXPECT_NEAR(mw / point.density, gas_constant * 400 / 300000 + 1e-3, 1e-15);
    EXPECT_GT(point.fugacities[1], 0.75 * 300000);
    std::vector parcels{phase(gas, {1, 3, 0}, 400, 100000)};
    const double volume = 4 * (gas_constant * 400 / 300000 + 1e-3);
    const auto solved = R::solve_phase_pressure(parcels, parcels[0].internal_energy, volume, 20000, 1000000);
    EXPECT_NEAR(solved.reference_pressure, 300000, 1e-4);
    EXPECT_NEAR(solved.thermal.phases[0].temperature, 400, 1e-7);
    EXPECT_NEAR(solved.thermal.total_internal_energy, parcels[0].internal_energy, 1e-7);
    EXPECT_NEAR(solved.thermal.total_volume, volume, volume * 1e-10);
}

TEST(ReactingThermodynamicsTest, SeparatePhaseTemperaturesAndClosedAdiabaticRelaxationReachCommonEquilibrium)
{
    const auto gas = ideal(catalog());
    std::vector parcels{phase(gas, {1, 1, 0}, 450), phase(gas, {0, 1, 1}, 300)};
    const double energy = parcels[0].internal_energy + parcels[1].internal_energy;
    const auto separate =
        R::recover_phase_temperatures(parcels, energy, R::PhaseThermalMode::SeparatePhaseTemperatures);
    EXPECT_NEAR(separate.phases[0].temperature, 450, 1e-7);
    EXPECT_NEAR(separate.phases[1].temperature, 300, 1e-7);
    const double equilibrium_t = (45.0 * 450 + 50.0 * 300) / 95;
    const double q = 45.0 * (450 - equilibrium_t);
    const auto relaxed = R::preview_interphase_exchange(
        parcels, 0, 1, [q](const auto&, const auto&, const auto&, const auto&)
        { return R::InterphaseExchange{{0, 0, 0}, {0, 0, 0}, q}; }, R::PhaseThermalMode::SeparatePhaseTemperatures);
    const auto equilibrium = R::recover_phase_temperatures(parcels, energy);
    EXPECT_NEAR(relaxed.thermal.total_internal_energy, energy, 1e-7);
    for (std::size_t i = 0; i < parcels.size(); ++i)
    {
        EXPECT_NEAR(relaxed.thermal.phases[i].temperature, equilibrium_t, 1e-7);
        EXPECT_NEAR(relaxed.thermal.phases[i].temperature, equilibrium.phases[i].temperature, 1e-7);
        EXPECT_NEAR(relaxed.thermal.phases[i].internal_energy, equilibrium.phases[i].internal_energy, 1e-6);
    }
}

TEST(ReactingThermodynamicsTest, CommonEnergyRecoveryAndFixedEnergyPressureSusceptibilityAgreeWithIndependentDifference)
{
    const auto species = catalog();
    std::vector parcels{phase(liquid(species), {10, 0, 0}, 380), phase(ideal(species), {0.2, 1, 0.1}, 320)};
    const double energy = parcels[0].internal_energy + parcels[1].internal_energy;
    const auto recovered = R::recover_phase_temperatures(parcels, energy);
    EXPECT_NEAR(recovered.phases[0].temperature, recovered.phases[1].temperature, 1e-10);
    auto low = parcels, high = parcels;
    for (auto& parcel : low)
        parcel.absolute_pressure -= 100;
    for (auto& parcel : high)
        parcel.absolute_pressure += 100;
    const double numerical = (R::recover_phase_temperatures(high, energy).total_volume -
                                 R::recover_phase_temperatures(low, energy).total_volume) /
                             200;
    EXPECT_NEAR(
        numerical, recovered.volume_pressure_susceptibility, std::abs(recovered.volume_pressure_susceptibility) * 2e-6);
}

TEST(ReactingThermodynamicsTest, GenericBubbleRadiusUsesNonidealEOSAndLaplacePressureWithSuppliedBounds)
{
    const auto gas = covolume(catalog());
    constexpr double radius = 0.002, number = 20, sigma = 0.05, p = 100000, t = 400;
    const double pressure = p + 2 * sigma / radius;
    const double geometric_volume = number * 4 * std::numbers::pi * radius * radius * radius / 3;
    const double amount = geometric_volume / (gas_constant * t / pressure + 1e-3);
    const std::vector<double> moles{amount * 0.25, amount * 0.75, 0};
    const auto result = R::solve_bubble_radius(*gas, moles, t, p, sigma, number, 1e-5, 0.1);
    EXPECT_NEAR(result.radius, radius, 2e-14);
    EXPECT_NEAR(result.volume, geometric_volume, geometric_volume * 1e-10);
    EXPECT_NEAR(result.absolute_pressure, pressure, 1e-6);
    EXPECT_THROW((void) R::solve_bubble_radius(*gas, moles, t, p, sigma, number, 0.01, 0.1), std::invalid_argument);
}

TEST(ReactingThermodynamicsTest, BoilingAndCondensationReverseWithPairedEnergyAndRetainedNoncondensable)
{
    const auto species = catalog();
    std::vector parcels{phase(liquid(species), {10, 0, 0}, 360), phase(ideal(species), {0.2, 1, 0}, 330)};
    const double energy = parcels[0].internal_energy + parcels[1].internal_energy;
    // Supplied synthetic interface enthalpy includes the vapor caloric/formation reference once.
    const double h = 60000 + 25 * 30 + gas_constant * 330;
    const auto boiling = R::preview_interphase_exchange(
        parcels, 0, 1, [h](const auto&, const auto&, const auto&, const auto&)
        { return R::InterphaseExchange{{0.1, 0, 0}, {h, 0, 0}, 0}; }, R::PhaseThermalMode::SeparatePhaseTemperatures);
    EXPECT_NEAR(boiling.mass_first_to_second, 0.1 * 0.018, 1e-16);
    EXPECT_NEAR(boiling.enthalpy_first_to_second, 0.1 * h, 1e-10);
    EXPECT_NEAR(boiling.thermal.total_internal_energy, energy, 1e-6);
    EXPECT_DOUBLE_EQ(boiling.thermal.phases[1].moles[1], 1);
    const auto condensation = R::preview_interphase_exchange(
        boiling.thermal.phases, 0, 1, [h](const auto&, const auto&, const auto&, const auto&)
        { return R::InterphaseExchange{{-0.1, 0, 0}, {h, 0, 0}, 0}; }, R::PhaseThermalMode::SeparatePhaseTemperatures);
    EXPECT_NEAR(condensation.thermal.total_internal_energy, energy, 1e-6);
    for (std::size_t i = 0; i < parcels.size(); ++i)
    {
        EXPECT_NEAR(condensation.thermal.phases[i].temperature, parcels[i].temperature, 1e-7);
        EXPECT_NEAR(condensation.thermal.phases[i].internal_energy, parcels[i].internal_energy, 1e-6);
        for (std::size_t s = 0; s < 3; ++s)
            EXPECT_NEAR(condensation.thermal.phases[i].moles[s], parcels[i].moles[s], 1e-14);
    }
}

TEST(ReactingThermodynamicsTest, NonidealEquilibriumClosureReceivesFugacitiesInsteadOfAssumedPartialPressure)
{
    const auto species = catalog();
    std::vector parcels{phase(liquid(species), {10, 0, 0}, 350), phase(covolume(species), {1, 1, 0}, 350)};
    bool called = false;
    const auto result = R::preview_interphase_exchange(parcels, 0, 1,
        [&called](const auto&, const auto&, const auto&, const auto& gas)
        {
            called = true;
            EXPECT_GT(gas.fugacities[0], 50000);
            return R::InterphaseExchange{{0, 0, 0}, {0, 0, 0}, 0};
        });
    EXPECT_TRUE(called);
    EXPECT_DOUBLE_EQ(result.mass_first_to_second, 0);
}

TEST(ReactingThermodynamicsTest, InvalidDomainDerivativeEquilibriumDataAndCaloricReferencesReject)
{
    const auto species = catalog();
    const auto good = ideal(species);
    const std::vector<double> x{1, 0, 0};
    EXPECT_THROW((void) good->evaluate(249, 100000, x), std::invalid_argument);
    EXPECT_THROW((void) good->evaluate(300, 100000, std::vector<double>{0.5, 0, 0}), std::invalid_argument);
    for (int fault = 0; fault < 5; ++fault)
    {
        auto bad = R::MixtureThermodynamics(species, bounds, R::EOSKind::Gas,
            [good, fault](double t, double p, std::span<const double> composition)
            {
                auto result = good->evaluate(t, p, composition);
                if (fault == 0)
                    result.density_pressure = 0;
                if (fault == 1)
                    result.internal_energy_pressure = 1;
                if (fault == 2)
                    result.internal_energy_temperature = -1;
                if (fault == 3)
                    result.enthalpy += 100;
                if (fault == 4)
                    result.fugacities.clear();
                return result;
            });
        EXPECT_THROW((void) bad.evaluate(300, 100000, x), std::invalid_argument);
    }
    const auto chemical_hook = R::MixtureThermodynamics(
        species, bounds, R::EOSKind::Gas,
        [good](double t, double p, std::span<const double> composition)
        {
            auto result = good->evaluate(t, p, composition);
            result.fugacities.clear();
            return result;
        },
        R::EnergyConvention::IncludesFormationEnergy,
        [](double, double, std::span<const double>) { return std::vector<double>{1, 2, 3}; });
    EXPECT_EQ(chemical_hook.evaluate(300, 100000, x).chemical_potentials.size(), 3);
}

TEST(ReactingThermodynamicsTest, EmptyPhasesAndJointDonorEnergyRejectionAreAtomic)
{
    const auto gas = ideal(catalog());
    std::vector parcels{phase(gas, {1, 1, 0}, 350), phase(gas, {0, 0, 0}, 0)};
    const auto original = parcels;
    const double energy = parcels[0].internal_energy;
    const auto recovered = R::recover_phase_temperatures(parcels, energy);
    EXPECT_DOUBLE_EQ(recovered.phases[1].temperature, 0);
    EXPECT_DOUBLE_EQ(recovered.phases[1].internal_energy, 0);
    auto invalid = parcels;
    invalid[1].internal_energy = 1;
    EXPECT_THROW((void) R::recover_phase_temperatures(invalid, energy + 1), std::invalid_argument);
    invalid = parcels;
    invalid[0].internal_energy = 1e20;
    EXPECT_THROW((void) R::recover_phase_temperatures(invalid, 1e20), std::invalid_argument);
    EXPECT_THROW(
        (void) R::preview_interphase_exchange(parcels, 0, 1, [](const auto&, const auto&, const auto&, const auto&)
            { return R::InterphaseExchange{{2, 0, 0}, {60000, 0, 0}, 0}; }),
        std::invalid_argument);
    EXPECT_THROW(
        (void) R::preview_interphase_exchange(parcels, 0, 1, [](const auto&, const auto&, const auto&, const auto&)
            { return R::InterphaseExchange{{0.1, 0, 0}, {1e12, 0, 0}, 0}; }),
        std::invalid_argument);
    EXPECT_EQ(parcels[0].moles, original[0].moles);
    EXPECT_DOUBLE_EQ(parcels[0].internal_energy, original[0].internal_energy);
    EXPECT_DOUBLE_EQ(parcels[1].temperature, 0);
}
TEST(ReactingThermodynamicsTest, InvalidEnumsAndMixedFormationConventionsReject)
{
    const auto species = catalog();
    const auto gas = ideal(species);
    auto sensible = R::make_ideal_gas_mixture(
        species, {{25, 300, 60000}, {20, 300, 1200}, {30, 300, 2000}}, bounds, R::EnergyConvention::SensibleEnergy);
    std::vector parcels{phase(gas, {1, 0, 0}, 350), phase(sensible, {0, 1, 0}, 350)};
    EXPECT_THROW((void) R::recover_phase_temperatures(parcels, parcels[0].internal_energy + parcels[1].internal_energy),
        std::invalid_argument);
    parcels.pop_back();
    EXPECT_THROW((void) R::recover_phase_temperatures(
                     parcels, parcels[0].internal_energy, static_cast<R::PhaseThermalMode>(999)),
        std::invalid_argument);
    EXPECT_THROW((void) R::MixtureThermodynamics(species, bounds, static_cast<R::EOSKind>(99),
                     [](double, double, std::span<const double>) { return R::MixtureThermodynamicPoint{}; }),
        std::invalid_argument);
    EXPECT_THROW((void) R::MixtureThermodynamics(
                     species, bounds, R::EOSKind::Gas, [](double, double, std::span<const double>)
                     { return R::MixtureThermodynamicPoint{}; }, static_cast<R::EnergyConvention>(99)),
        std::invalid_argument);
}

TEST(ReactingThermodynamicsTest, FixedEnergyPressureRootSurvivesCaloricallyInfeasibleBracketEndpoints)
{
    const auto species = catalog();
    const R::ThermodynamicBounds narrow{300, 300.01, 10000, 2000000};
    const auto eos = R::make_linear_compressible_mixture_liquid(species, narrow, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4200, 2e-4, 5e-9, 10000}; });
    std::vector parcels{phase(eos, {10, 0, 0}, 300.005, 100000)};
    const auto actual = R::recover_phase_temperatures(parcels, parcels[0].internal_energy);
    const auto solved =
        R::solve_phase_pressure(parcels, parcels[0].internal_energy, actual.total_volume, 10000, 2000000);
    EXPECT_NEAR(solved.reference_pressure, 100000, 1e-5);
    EXPECT_NEAR(solved.thermal.phases[0].temperature, 300.005, 1e-8);
    // Also recover an interior root different from the pressure used to locate the feasible branch.
    auto guessed = parcels;
    guessed[0].absolute_pressure = 100100;
    const auto shifted =
        R::solve_phase_pressure(guessed, parcels[0].internal_energy, actual.total_volume, 10000, 2000000);
    EXPECT_NEAR(shifted.reference_pressure, 100000, 0.01);
}

TEST(ReactingThermodynamicsTest, CompleteCondensationAndVaporCreationOwnPhasePressureWorkWithoutLatentHeatDuplication)
{
    const auto species = catalog();
    std::vector parcels{phase(liquid(species), {10, 0, 0}, 360), phase(ideal(species), {0.2, 0, 0}, 330)};
    const double energy = parcels[0].internal_energy + parcels[1].internal_energy;
    const auto initial = R::recover_phase_temperatures(parcels, energy, R::PhaseThermalMode::SeparatePhaseTemperatures);
    const double gas_volume = initial.volumes[1];
    const double gas_h = 60000 + 25 * 30 + gas_constant * 330;
    const double transferred_h = 0.2 * gas_h;
    const double final_mass = 10.2 * 0.018;
    const double final_liquid_h =
        (parcels[0].internal_energy + 100000 * initial.volumes[0] + transferred_h) / final_mass;
    const double final_temperature = 300 + (final_liquid_h - 10100) / 4200;
    const double final_liquid_volume = final_mass / 1000 * (1 + 2e-4 * (final_temperature - 300));
    const double liquid_volume_change = final_liquid_volume - initial.volumes[0];
    const auto condensation = R::preview_interphase_exchange(
        parcels, 0, 1,
        [=](const auto&, const auto&, const auto&, const auto&)
        {
            return R::InterphaseExchange{
                {-0.2, 0, 0}, {gas_h, 0, 0}, 0, R::IsobaricPhaseVolumeWork{liquid_volume_change, -gas_volume}};
        },
        R::PhaseThermalMode::SeparatePhaseTemperatures);
    EXPECT_DOUBLE_EQ(condensation.thermal.phases[1].internal_energy, 0);
    EXPECT_DOUBLE_EQ(condensation.thermal.phases[1].temperature, 0);
    EXPECT_DOUBLE_EQ(condensation.thermal.phases[1].moles[0], 0);
    EXPECT_NEAR(condensation.thermal.phases[0].temperature, final_temperature, 1e-7);
    EXPECT_NEAR(condensation.external_mechanical_work, -100000 * (liquid_volume_change - gas_volume), 1e-10);
    EXPECT_NEAR(condensation.thermal.total_internal_energy,
        energy + condensation.external_mechanical_work + condensation.empty_phase_energy_roundoff, 1e-7);
    EXPECT_DOUBLE_EQ(condensation.heat_first_to_second, 0);
    const auto evaporation = R::preview_interphase_exchange(
        condensation.thermal.phases, 0, 1,
        [=](const auto&, const auto&, const auto&, const auto&)
        {
            return R::InterphaseExchange{
                {0.2, 0, 0}, {gas_h, 0, 0}, 0, R::IsobaricPhaseVolumeWork{-liquid_volume_change, gas_volume}};
        },
        R::PhaseThermalMode::SeparatePhaseTemperatures);
    EXPECT_NEAR(evaporation.thermal.phases[0].temperature, 360, 1e-7);
    EXPECT_NEAR(evaporation.thermal.phases[1].temperature, 330, 1e-7);
    EXPECT_NEAR(evaporation.thermal.total_internal_energy, energy, 1e-7);
    EXPECT_THROW((void) R::preview_interphase_exchange(
                     parcels, 0, 1,
                     [=](const auto&, const auto&, const auto&, const auto&)
                     {
                         return R::InterphaseExchange{{-0.2, 0, 0}, {gas_h, 0, 0}, 0,
                             R::IsobaricPhaseVolumeWork{liquid_volume_change * 2, -gas_volume}};
                     },
                     R::PhaseThermalMode::SeparatePhaseTemperatures),
        std::invalid_argument);
}
TEST(ReactingThermodynamicsTest, TinyOccupiedBinsRecoverTemperatureWithoutAnAbsoluteEnergyFloor)
{
    const auto gas = ideal(catalog());
    std::vector parcels{phase(gas, {0, 1e-30, 0}, 350)};
    const auto result = R::recover_phase_temperatures(
        parcels, parcels[0].internal_energy, R::PhaseThermalMode::SeparatePhaseTemperatures);
    EXPECT_NEAR(result.phases[0].temperature, 350, 1e-8);
    EXPECT_NEAR(result.total_internal_energy / parcels[0].internal_energy, 1, 1e-12);
    EXPECT_GT(result.total_volume, 0);
    EXPECT_GT(result.phases[0].moles[1], 0);
}
TEST(ReactingThermodynamicsTest, ExactIncompressibleLiquidRetainsConstantVolumeAndPositiveCaloricRecovery)
{
    const auto species = catalog();
    const auto eos = R::make_linear_compressible_mixture_liquid(species, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4200, 0, 0, 10000}; });
    const std::vector<double> x{1, 0, 0};
    for (double pressure : {10000., 100000., 2000000.})
    {
        const auto point = eos->evaluate(330, pressure, x);
        EXPECT_DOUBLE_EQ(point.density, 1000);
        EXPECT_DOUBLE_EQ(point.density_pressure, 0);
        EXPECT_DOUBLE_EQ(point.density_temperature, 0);
        EXPECT_DOUBLE_EQ(point.internal_energy_pressure, 0);
        EXPECT_DOUBLE_EQ(point.internal_energy_temperature, 4200);
        EXPECT_NEAR(point.internal_energy, 10000 + 4200 * 30, 1e-9);
    }
    std::vector phases{phase(eos, {10, 0, 0}, 330)};
    const auto recovered = R::recover_phase_temperatures(phases, phases[0].internal_energy);
    EXPECT_NEAR(recovered.phases[0].temperature, 330, 1e-8);
    EXPECT_DOUBLE_EQ(recovered.volume_pressure_susceptibility, 0);
    const auto closure =
        R::solve_phase_pressure(phases, phases[0].internal_energy, recovered.total_volume, 20000, 1000000);
    EXPECT_DOUBLE_EQ(closure.reference_pressure, 100000);
    EXPECT_THROW((void) R::solve_phase_pressure(
                     phases, phases[0].internal_energy, 1.01 * recovered.total_volume, 20000, 1000000),
        std::invalid_argument);
}

TEST(ReactingThermodynamicsTest, ZeroLiquidCompressibilityRejectsThermalExpansionAndGasStillRequiresCompressibility)
{
    const auto species = catalog();
    const auto inconsistent = R::make_linear_compressible_mixture_liquid(species, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4200, 1e-4, 0, 10000}; });
    EXPECT_THROW((void) inconsistent->evaluate(300, 100000, std::vector<double>{1, 0, 0}), std::invalid_argument);
    const auto good = ideal(species);
    R::MixtureThermodynamics bad_gas(species, bounds, R::EOSKind::Gas,
        [good](double t, double p, std::span<const double> x)
        {
            auto point = good->evaluate(t, p, x);
            point.density_pressure = 0;
            return point;
        });
    EXPECT_THROW((void) bad_gas.evaluate(300, 100000, std::vector<double>{1, 0, 0}), std::invalid_argument);
}
TEST(ReactingThermodynamicsTest, BubbleRadiusRestrictsBothEOSInvalidUserEndpointsToTheValidInterior)
{
    const auto gas = covolume(catalog());
    constexpr double radius = 1e-5, liquid_pressure = 5000, sigma = .05, t = 400, number = 3;
    const double pressure = liquid_pressure + 2 * sigma / radius;
    const double volume = number * 4 * std::numbers::pi * radius * radius * radius / 3;
    const double amount = volume / (gas_constant * t / pressure + 1e-3);
    const std::vector<double> moles{.25 * amount, .75 * amount, 0};
    ASSERT_GT(liquid_pressure + 2 * sigma / 1e-12, gas->bounds().maximum_pressure);
    ASSERT_LT(liquid_pressure + 2 * sigma / .1, gas->bounds().minimum_pressure);
    const auto root = R::solve_bubble_radius(*gas, moles, t, liquid_pressure, sigma, number, 1e-12, .1);
    EXPECT_NEAR(root.radius, radius, radius * 1e-10);
    EXPECT_NEAR(root.absolute_pressure, pressure, pressure * 1e-10);
    EXPECT_NEAR(root.volume, volume, volume * 1e-10);
    EXPECT_THROW(
        (void) R::solve_bubble_radius(*gas, moles, t, liquid_pressure, sigma, number, .01, .1), std::invalid_argument);
}
TEST(ReactingThermodynamicsTest, BubbleRadiusRoundsPressureIntersectionInwardWithoutWideningEOSDomain)
{
    const auto species = catalog();
    constexpr double sigma = .07, t = 330, number = 3;
    for (const double lower_pressure : {1., 1e6})
        for (const double liquid_pressure : {99000., 101000.})
        {
            SCOPED_TRACE(lower_pressure);
            SCOPED_TRACE(liquid_pressure);
            const R::ThermodynamicBounds domain{250, 600, lower_pressure, 3e7};
            const auto base =
                R::make_ideal_gas_mixture(species, {{25, 300, 60000}, {20, 300, 1200}, {30, 300, 2000}}, domain);
            std::vector<double> evaluated_pressures;
            const R::MixtureThermodynamics gas(species, domain, R::EOSKind::Gas,
                [base, &evaluated_pressures](double temperature, double pressure, std::span<const double> fractions)
                {
                    evaluated_pressures.push_back(pressure);
                    return base->evaluate(temperature, pressure, fractions);
                });
            const double radius = lower_pressure == 1 ? 1e-6 : 1e-7;
            const double pressure = liquid_pressure + 2 * sigma / radius;
            const double volume = number * 4 * std::numbers::pi * radius * radius * radius / 3;
            const double moles = pressure * volume / (gas_constant * t);
            const std::vector<double> amounts{.25 * moles, .75 * moles, 0};
            const auto result = R::solve_bubble_radius(gas, amounts, t, liquid_pressure, sigma, number, 1e-12, 1e-3);
            EXPECT_NEAR(result.radius, radius, 32 * std::numeric_limits<double>::epsilon() * radius);
            EXPECT_GE(result.absolute_pressure, domain.minimum_pressure);
            EXPECT_LE(result.absolute_pressure, domain.maximum_pressure);
            ASSERT_FALSE(evaluated_pressures.empty());
            for (const double evaluated : evaluated_pressures)
            {
                EXPECT_GE(evaluated, domain.minimum_pressure);
                EXPECT_LE(evaluated, domain.maximum_pressure);
            }
            // A physically excluded radius still fails the signed volume bracket.
            EXPECT_THROW(
                (void) R::solve_bubble_radius(gas, amounts, t, liquid_pressure, sigma, number, 1.1 * radius, 1e-3),
                std::invalid_argument);
        }
}

} // namespace
