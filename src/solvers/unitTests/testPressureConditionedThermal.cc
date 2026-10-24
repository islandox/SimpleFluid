/** Independent local interphase energy, pressure and donor oracles. */
#include "ConditionedThermalTestSupport.hh"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>
#include <numeric>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using namespace SimpleFluid::Testing::ConditionedTransfer;
TEST(PressureConditionedThermalTest, CompleteDonorAndInitiallyAbsentVaporCloseAtNewPressureWithRetainedOtherGas)
{
    Fixture f;
    const auto plan = f.plan();
    const auto accepted = f.before;
    const double expected_p = 140000;
    const auto result = f.evaluate(plan, expected_p);
    EXPECT_NEAR(result.total_volume, 1, 2e-10);
    EXPECT_NEAR(f.root(plan), expected_p, 2e-9 * expected_p);
    const double expected_t = expected_p / (200 + .4 * gas_constant);
    EXPECT_NEAR(result.phases[1].temperature, expected_t, 2e-9 * expected_t);
    EXPECT_NEAR(result.phases[2].temperature, expected_t, 2e-9 * expected_t);
    EXPECT_EQ(result.phases[0].moles, std::vector<double>({0, 0}));
    EXPECT_DOUBLE_EQ(result.phases[0].internal_energy, 0);
    EXPECT_DOUBLE_EQ(result.phases[0].temperature, 0);
    EXPECT_DOUBLE_EQ(result.phases[1].moles[0], f.nl);
    EXPECT_DOUBLE_EQ(result.phases[2].moles[1], f.ng);
    EXPECT_NEAR(result.volume_pressure_derivative, -cv / (cp * expected_p), 2e-10 * cv / (cp * expected_p));
    EXPECT_NEAR(result.pressure_work, 0, 2e-8);
    EXPECT_NE(f.root(plan), p_old);
    // Frozen old carried H would force donor emptiness at p_old, incompatible with closed V=1.
    // Along the actual enthalpy/work path the off-root volume is the independent expression below.
    EXPECT_NEAR(
        f.evaluate(plan, p_old).total_volume, (gas_constant / cp) * f.physical_U0 / p_old + gas_constant / cp, 2e-10);
    EXPECT_EQ(f.before[0].moles, accepted[0].moles);
    EXPECT_DOUBLE_EQ(f.before[0].internal_energy, accepted[0].internal_energy);
}
TEST(PressureConditionedThermalTest, StrongIndependentHeatWithEmptyReceiverUsesSameBoundsAndPressureTangent)
{
    Fixture f;
    const auto plan = f.plan(.5 * f.physical_U0);
    const auto result = f.evaluate(plan, 210000);
    EXPECT_NEAR(result.total_volume, 1, 2e-10);
    EXPECT_NEAR(f.root(plan), 210000, 2e-9 * 210000);
    const double expected_t = 210000 / (200 + .4 * gas_constant);
    EXPECT_NEAR(result.phases[1].temperature, expected_t, 2e-9 * expected_t);
    EXPECT_LT(expected_t, 1200);
    EXPECT_DOUBLE_EQ(result.phases[0].temperature, 0);
    EXPECT_DOUBLE_EQ(result.phases[1].moles[0], f.nl);
    const double energy = std::accumulate(result.phases.begin(), result.phases.end(), 0.,
        [](double u, const auto& phase) { return u + phase.internal_energy; });
    EXPECT_NEAR(energy, 1.5 * f.physical_U0, 2e-10 * 1.5 * f.physical_U0);
}
TEST(PressureConditionedThermalTest, SeparateTemperatureReceiverMayBeInfeasibleWithoutExplicitHeatAllocation)
{
    Fixture f;
    const auto plan = f.plan();
    EXPECT_THROW(
        (void) f.evaluate(plan, 140000, R::PhaseThermalMode::SeparatePhaseTemperatures), R::ThermalCaloricRangeError);
}
TEST(PressureConditionedThermalTest, PerSpeciesCaloricReferenceShiftsPreservePhysicalPressureTemperatureAndTangent)
{
    Fixture base, shifted(1e6, -3e5);
    const auto first = base.plan(), second = shifted.plan();
    const auto a = base.evaluate(first, 140000), b = shifted.evaluate(second, 140000);
    EXPECT_NEAR(shifted.root(second), base.root(first), 2e-9 * 140000);
    for (size_t p = 0; p < a.phases.size(); ++p)
    {
        EXPECT_NEAR(a.phases[p].temperature, b.phases[p].temperature,
            2e-9 * std::max(a.phases[p].temperature, b.phases[p].temperature));
        EXPECT_EQ(a.phases[p].moles, b.phases[p].moles);
    }
    EXPECT_NEAR(
        a.volume_pressure_derivative, b.volume_pressure_derivative, 2e-10 * std::abs(a.volume_pressure_derivative));
}
TEST(PressureConditionedThermalTest, RawLargeStaticCancellationRetainsIndependentFiniteHeatBeforeDynamicNarrowing)
{
    Fixture f;
    const auto plan = R::compile_interphase_plan(f.configuration, f.before, {{0}}, {{1e20L, -1e20L, 10}, {}, {}}, {10});
    const auto result = f.evaluate(plan, p_old);
    const double applied = std::accumulate(result.applied_phase_energy.begin(), result.applied_phase_energy.end(), 0.);
    EXPECT_NEAR(applied, 10, 2e-8);
    const double u = std::accumulate(result.phases.begin(), result.phases.end(), 0.,
        [](double v, const auto& phase) { return v + phase.internal_energy; });
    const double old = std::accumulate(f.Uadv.begin(), f.Uadv.end(), 0.);
    EXPECT_NEAR(u - old - result.pressure_work, 10, 2e-8);
}
TEST(PressureConditionedThermalTest, JointOriginalEventDonorsCannotBorrowIncomingMaterial)
{
    Fixture f;
    auto config = *f.configuration;
    config.stages[0].material.push_back({44, 0, 1});
    auto compiled = std::make_shared<const R::InterphaseConfiguration>(config);
    EXPECT_THROW((void) R::compile_interphase_plan(compiled, f.before, {{.3, .2}}, {{}, {}, {}}, {}), std::exception);
    EXPECT_DOUBLE_EQ(f.before[0].moles[0], .4);
}
TEST(PressureConditionedThermalTest, HeatAfterDisappearanceCannotReadInventedDonorTemperature)
{
    Fixture f;
    auto config = *f.configuration;
    config.stages.push_back({"after-empty-heat", R::InterphaseStageKind::Heat, {}, {{0, 2}}});
    auto shared = std::make_shared<const R::InterphaseConfiguration>(config);
    const auto plan = R::compile_interphase_plan(shared, f.before, {{.4}, {1}}, {{}, {}, {}}, {});
    EXPECT_THROW((void) f.evaluate(plan, 140000), std::invalid_argument);
}
TEST(PressureConditionedThermalTest, GenuineProviderAndEulerDerivativeFailuresAreHardNotCaloricSearchSignals)
{
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        Fixture f;
        auto config = *f.configuration;
        const auto original = config.phase_enthalpy[0];
        config.phase_enthalpy[0] = std::make_shared<const R::PartialMolarEnthalpyLaw>(R::PartialMolarEnthalpyLaw{
            original->metadata, original->bounds, [original, scenario](double t, double p, std::span<const double> x)
            {
                if (scenario == 0)
                    throw std::logic_error("Real supplied partial-h failure.");
                auto h = original->evaluate(t, p, x);
                if (scenario == 1)
                    h.temperature[0] *= 2;
                else
                    h.pressure[0] *= 2;
                return h;
            }});
        auto shared = std::make_shared<const R::InterphaseConfiguration>(config);
        const auto plan = R::compile_interphase_plan(shared, f.before, {{.4}}, {{}, {}, {}}, {});
        if (scenario == 0)
            EXPECT_THROW((void) f.evaluate(plan, 140000), std::logic_error);
        else
            EXPECT_THROW((void) f.evaluate(plan, 140000), std::invalid_argument);
    }
}
TEST(PressureConditionedThermalTest, TrulyImpossibleHeatThrowsTypedRangeAndDoesNotAlterFrozenAmounts)
{
    Fixture f;
    const auto plan = f.plan(4 * f.physical_U0);
    const auto original = plan.amounts;
    for (double p : {50000., 100000., 1000000.})
        EXPECT_THROW((void) f.evaluate(plan, p), R::ThermalCaloricRangeError);
    EXPECT_EQ(plan.amounts, original);
    EXPECT_DOUBLE_EQ(f.before[0].moles[0], .4);
}
TEST(PressureConditionedThermalTest, PressureTrialsEvaluateConstitutiveLawWithoutAnyKineticCallbackOrAmountChange)
{
    Fixture f;
    int kinetic_calls = 0;
    auto integrate = [&]
    {
        ++kinetic_calls;
        return f.plan();
    };
    const auto plan = integrate();
    const auto original = plan.amounts;
    for (double p : {90000., 100000., 140000., 150000.})
        (void) f.evaluate(plan, p);
    EXPECT_EQ(kinetic_calls, 1);
    EXPECT_GE(f.constitutive_calls, 4);
    EXPECT_EQ(plan.amounts, original);
}
TEST(PressureConditionedThermalTest, PartialEvaporationAndReversedCondensationRetainNoncondensableAndCloseEnergy)
{
    Fixture f;
    const double amount = .1 * f.nl;
    const auto evaporation = f.plan(0, amount);
    const double liquid_mass = .4 - amount;
    const double e_ref = f.before[0].internal_energy / .4;
    const double expected_t =
        (f.physical_U0 - liquid_mass * e_ref + liquid_mass * 1000 * 300) / (liquid_mass * 1000 + cv * (f.ng + amount));
    const double expected_p = (f.ng + amount) * gas_constant * expected_t / (1 - liquid_mass);
    const auto first = f.evaluate(evaporation, expected_p);
    EXPECT_NEAR(first.total_volume, 1, 2e-10);
    EXPECT_NEAR(first.phases[1].temperature, expected_t, 2e-9 * expected_t);
    EXPECT_DOUBLE_EQ(first.phases[2].moles[1], f.ng);
    auto configuration = *f.configuration;
    configuration.stages = {{"explicit-reverse-condensation", R::InterphaseStageKind::Material, {{44, 1, 0}}, {}}};
    configuration.phase_enthalpy[1] = std::make_shared<const R::PartialMolarEnthalpyLaw>(
        R::PartialMolarEnthalpyLaw{{"analytic-ideal-gas-partial-h", 1, f.catalog->fingerprint()},
            {250, 1200, 50000, 1000000}, [](double t, double, std::span<const double>)
            { return R::PartialMolarEnthalpyPoint{{cp * t, cp * t}, {cp, cp}, {0, 0}}; }});
    auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    const auto condensation = R::compile_interphase_plan(shared, first.phases, {{amount}}, {{}, {}, {}}, {});
    auto templates = first.phases;
    std::vector<double> U;
    for (size_t a = 0; a < templates.size(); ++a)
    {
        U.push_back(templates[a].internal_energy);
        templates[a].moles = condensation.final_moles[a];
    }
    const auto second = R::evaluate_pressure_conditioned_thermal(
        condensation, templates, U, first.volumes, p_old, R::PhaseThermalMode::Equilibrium);
    EXPECT_NEAR(second.total_volume, 1, 2e-10);
    EXPECT_NEAR(second.phases[0].temperature, 300, 2e-9 * 300);
    EXPECT_EQ(second.phases[1].moles, std::vector<double>({0, 0}));
    EXPECT_DOUBLE_EQ(second.phases[1].internal_energy, 0);
    EXPECT_DOUBLE_EQ(second.phases[2].moles[1], f.ng);
    for (size_t a = 0; a < f.before.size(); ++a)
        EXPECT_NEAR(second.phases[a].internal_energy, f.before[a].internal_energy,
            2e-10 * std::max(std::abs(second.phases[a].internal_energy), std::abs(f.before[a].internal_energy)));
}
TEST(PressureConditionedThermalTest, SeparatePhaseHeatRelaxationHasIndependentClosedAdiabaticPressureWorkOracle)
{
    Fixture f;
    auto before = f.before;
    before[0].temperature = 350;
    before[0].internal_energy += .4 * 1000 * 50;
    auto configuration = *f.configuration;
    configuration.stages = {{"explicit-supplied-heat", R::InterphaseStageKind::Heat, {}, {{0, 2}}}};
    auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    const auto plan = R::compile_interphase_plan(shared, before, {{100}}, {{}, {}, {}}, {});
    std::vector<double> U;
    for (const auto& phase : before)
        U.push_back(phase.internal_energy);
    const double liquid_t = 350 - 100 / (.4 * 1000), gas_t = 300 + 100 / (f.ng * cv);
    const double pressure = p_old * gas_t / 300;
    const auto result = R::evaluate_pressure_conditioned_thermal(
        plan, before, U, f.Vadv, pressure, R::PhaseThermalMode::SeparatePhaseTemperatures);
    EXPECT_NEAR(result.phases[0].temperature, liquid_t, 2e-9 * liquid_t);
    EXPECT_NEAR(result.phases[2].temperature, gas_t, 2e-9 * gas_t);
    EXPECT_NEAR(result.total_volume, 1, 2e-10);
    EXPECT_NEAR(result.pressure_work, 0, 2e-8);
    EXPECT_DOUBLE_EQ(result.raw_conductive_heat[0], -100);
    EXPECT_DOUBLE_EQ(result.raw_conductive_heat[2], 100);
    double energy = 0, original = 0;
    for (size_t a = 0; a < before.size(); ++a)
    {
        energy += result.phases[a].internal_energy;
        original += U[a];
    }
    EXPECT_NEAR(energy, original, 2e-10 * std::abs(original));
}
TEST(PressureConditionedThermalTest, EqualTemperatureZeroHeatIsTheEquilibriumLimitWithoutCreatedPhaseEnergy)
{
    Fixture f;
    auto configuration = *f.configuration;
    configuration.stages = {{"zero-heat-limit", R::InterphaseStageKind::Heat, {}, {{0, 2}}}};
    auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    const auto plan = R::compile_interphase_plan(shared, f.before, {{0}}, {{}, {}, {}}, {});
    const auto result = R::evaluate_pressure_conditioned_thermal(
        plan, f.before, f.Uadv, f.Vadv, p_old, R::PhaseThermalMode::Equilibrium);
    EXPECT_NEAR(result.total_volume, 1, 2e-10);
    EXPECT_DOUBLE_EQ(result.phases[1].temperature, 0);
    EXPECT_DOUBLE_EQ(result.phases[1].internal_energy, 0);
    EXPECT_NEAR(result.phases[0].temperature, 300, 2e-9 * 300);
    EXPECT_NEAR(result.phases[2].temperature, 300, 2e-9 * 300);
    EXPECT_DOUBLE_EQ(result.raw_conductive_heat[0], 0);
    EXPECT_DOUBLE_EQ(result.raw_conductive_heat[2], 0);
}
TEST(PressureConditionedThermalTest, IndependentExcludedVolumeMixtureUsesActualNonidealEnthalpyPressureTangent)
{
    Fixture f;
    constexpr std::array<double, 2> b{.001, .002};
    f.gas = std::make_shared<const R::MixtureThermodynamics>(f.catalog,
        R::ThermodynamicBounds{250, 1200, 50000, 1000000}, R::EOSKind::Gas,
        [catalog = f.catalog](double t, double p, std::span<const double> x)
        {
            double mass = 0, covolume = 0;
            for (size_t i = 0; i < x.size(); ++i)
            {
                mass += x[i] * catalog->species()[i].molarMass;
                covolume += x[i] * (i == 0 ? .001 : .002);
            }
            const double volume = gas_constant * t / p + covolume;
            R::MixtureThermodynamicPoint point;
            point.density = mass / volume;
            point.internal_energy = cv * t / mass;
            point.enthalpy = (cp * t + p * covolume) / mass;
            point.internal_energy_temperature = cv / mass;
            point.internal_energy_pressure = 0;
            point.density_temperature = -mass * gas_constant / (p * volume * volume);
            point.density_pressure = mass * gas_constant * t / (p * p * volume * volume);
            point.fugacities.resize(x.size());
            for (size_t i = 0; i < x.size(); ++i)
                point.fugacities[i] = x[i] * p * std::exp((i == 0 ? .001 : .002) * p / (gas_constant * t));
            return point;
        });
    f.ng = .6 / (gas_constant * 300 / p_old + b[1]);
    f.before[1].thermodynamics = f.gas;
    f.before[2].thermodynamics = f.gas;
    f.before[2].moles = {0, f.ng};
    f.before[2].internal_energy = cv * f.ng * 300;
    f.Uadv[2] = f.before[2].internal_energy;
    f.physical_U0 = f.Uadv[0] + f.Uadv[2];
    const auto plan = f.plan();
    const double excluded = f.nl * b[0] + f.ng * b[1];
    const double pressure = (gas_constant / cv) * f.physical_U0 / (1 - excluded);
    const double temperature = f.physical_U0 / (cv * (f.nl + f.ng));
    const auto result = f.evaluate(plan, pressure);
    EXPECT_NEAR(result.total_volume, 1, 2e-10);
    EXPECT_NEAR(result.phases[1].temperature, temperature, 2e-9 * temperature);
    EXPECT_NEAR(result.volume_pressure_derivative, -(cv / cp) * (1 - excluded) / pressure,
        2e-10 * (cv / cp) * (1 - excluded) / pressure);
    EXPECT_DOUBLE_EQ(result.phases[2].moles[1], f.ng);
    EXPECT_DOUBLE_EQ(result.phases[0].internal_energy, 0);
    EXPECT_GT(pressure, p_old);
}
TEST(PressureConditionedThermalTest, JointHeatDonorsCannotBorrowIncomingEnergyOrHideBehindCommonTemperature)
{
    Fixture f;
    auto configuration = *f.configuration;
    configuration.stages = {{"joint-finite-heat", R::InterphaseStageKind::Heat, {}, {{2, 0}, {0, 2}}}};
    auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    // Liquid has only .4*cp*(300-250)=20000 J sensible capacity at this pressure.
    // Incoming 20000 J does not finance a simultaneous outgoing 30000 J.
    const auto plan = R::compile_interphase_plan(shared, f.before, {{20000, 30000}}, {{}, {}, {}}, {});
    EXPECT_THROW((void) f.evaluate(plan, p_old), R::ThermalCaloricRangeError);
}
TEST(PressureConditionedThermalTest, GrossMaterialReservationRetainsAnOverdrawSmallerThanThePositiveStockUlp)
{
    Fixture f;
    auto configuration = *f.configuration;
    configuration.stages[0].material.push_back({44, 0, 1});
    auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    EXPECT_THROW((void) R::compile_interphase_plan(shared, f.before, {{.4, 1e-30}}, {{}, {}, {}}, {}), std::exception);
    EXPECT_DOUBLE_EQ(f.before[0].moles[0], .4);
}
TEST(PressureConditionedThermalTest, OrderedMaterialStagesMayUseOnlyActualPrecedingStageCreditAndReplayExactly)
{
    Fixture f;
    auto configuration = *f.configuration;
    auto gas_partial = std::make_shared<const R::PartialMolarEnthalpyLaw>(
        R::PartialMolarEnthalpyLaw{{"ordered-ideal-gas-partials", 1, f.catalog->fingerprint()}, f.gas->bounds(),
            [](double t, double, std::span<const double>)
            { return R::PartialMolarEnthalpyPoint{{cp * t, cp * t}, {cp, cp}, {0, 0}}; }});
    configuration.phase_enthalpy = {gas_partial, gas_partial, nullptr};
    configuration.stages = {{"first-admission", R::InterphaseStageKind::Material, {{44, 0, 1}}, {}},
        {"funded-next-stage", R::InterphaseStageKind::Material, {{44, 1, 2}}, {}}};
    const auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    const std::vector<R::PhaseThermalState> before{
        {f.gas, {2, 0}, p_old, 2 * cv * T0, T0}, {f.gas, {0, 0}, p_old, 0, 0}, {f.gas, {0, 0}, p_old, 0, 0}};
    const std::vector<double> energy{2 * cv * T0, 0, 0}, volumes{2 * gas_constant * T0 / p_old, 0, 0};
    const auto plan = R::compile_interphase_plan(shared, before, {{1}, {1}}, {{}, {}, {}}, {});
    EXPECT_EQ(plan.final_moles, std::vector<std::vector<double>>({{1, 0}, {0, 0}, {1, 0}}));
    auto templates = before;
    for (size_t a = 0; a < templates.size(); ++a)
        templates[a].moles = plan.final_moles[a];
    const auto evaluate = [&]
    {
        return R::evaluate_pressure_conditioned_thermal(
            plan, templates, energy, volumes, p_old, R::PhaseThermalMode::Equilibrium);
    };
    const auto result = evaluate(), replay = evaluate();
    EXPECT_NEAR(result.total_volume, volumes[0], 2e-10 * volumes[0]);
    EXPECT_NEAR(result.pressure_work, 0, 2e-8);
    EXPECT_EQ(plan.amounts, std::vector<std::vector<double>>({{1}, {1}}));
    for (size_t a = 0; a < templates.size(); ++a)
    {
        EXPECT_EQ(result.phases[a].moles, plan.final_moles[a]);
        if (a == 1)
            EXPECT_DOUBLE_EQ(result.phases[a].temperature, 0);
        else
            EXPECT_NEAR(result.phases[a].temperature, T0, 2e-9 * T0);
        EXPECT_EQ(result.phases[a].moles, replay.phases[a].moles);
        EXPECT_DOUBLE_EQ(result.phases[a].internal_energy, replay.phases[a].internal_energy);
        EXPECT_DOUBLE_EQ(result.phases[a].temperature, replay.phases[a].temperature);
    }
    // The same two transfers in one stage cannot borrow the simultaneously incoming molecule.
    configuration.stages = {{"simultaneous-unfunded", R::InterphaseStageKind::Material, {{44, 0, 1}, {44, 1, 2}}, {}}};
    EXPECT_THROW((void) R::compile_interphase_plan(std::make_shared<const R::InterphaseConfiguration>(configuration),
                     before, {{1, 1}}, {{}, {}, {}}, {}),
        std::invalid_argument);
    EXPECT_DOUBLE_EQ(before[0].moles[0], 2);
    EXPECT_DOUBLE_EQ(before[1].moles[0], 0);
}
TEST(PressureConditionedThermalTest, OrderedSubUlpDebitCannotResetTheNextStageDonorMarginOrPressureReplay)
{
    Fixture f;
    auto configuration = *f.configuration;
    auto gas_partial = std::make_shared<const R::PartialMolarEnthalpyLaw>(
        R::PartialMolarEnthalpyLaw{{"huge-stock-ideal-gas-partials", 1, f.catalog->fingerprint()}, f.gas->bounds(),
            [](double t, double, std::span<const double>)
            { return R::PartialMolarEnthalpyPoint{{cp * t, cp * t}, {cp, cp}, {0, 0}}; }});
    configuration.phase_enthalpy = {gas_partial, nullptr, nullptr};
    configuration.stages = {{"small-first-debit", R::InterphaseStageKind::Material, {{44, 0, 1}}, {}},
        {"unfunded-complete-debit", R::InterphaseStageKind::Material, {{44, 0, 2}}, {}}};
    const auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    const std::vector<R::PhaseThermalState> before{
        {f.gas, {1e20, 0}, p_old, 1e20 * cv * T0, T0}, {f.gas, {0, 0}, p_old, 0, 0}, {f.gas, {0, 0}, p_old, 0, 0}};
    EXPECT_THROW(
        (void) R::compile_interphase_plan(shared, before, {{1}, {1e20}}, {{}, {}, {}}, {}), std::invalid_argument);

    // Public frozen-plan data are independently validated on pressure-domain and caloric replay.
    // Forge the previously accepted rounded projection to exercise both replay consumers independently.
    auto forged = R::compile_interphase_plan(shared, before, {{0}, {0}}, {{}, {}, {}}, {});
    forged.amounts = {{1}, {1e20}};
    forged.final_moles = {{0, 0}, {1, 0}, {1e20, 0}};
    auto templates = before;
    for (size_t a = 0; a < templates.size(); ++a)
        templates[a].moles = forged.final_moles[a];
    const std::vector<double> energy{1e20 * cv * T0, 0, 0}, volumes{1e20 * gas_constant * T0 / p_old, 0, 0};
    EXPECT_THROW((void) R::interphase_pressure_bounds(forged, templates), std::invalid_argument);
    EXPECT_THROW((void) R::evaluate_pressure_conditioned_thermal(
                     forged, templates, energy, volumes, p_old, R::PhaseThermalMode::Equilibrium),
        std::invalid_argument);
    EXPECT_EQ(forged.amounts, std::vector<std::vector<double>>({{1}, {1e20}}));
    EXPECT_DOUBLE_EQ(before[0].moles[0], 1e20);
    EXPECT_DOUBLE_EQ(before[1].moles[0], 0);
}
TEST(PressureConditionedThermalTest, UpstreamStaticDepletionIsExplicitlyUnsupportedEvenWhenOldPressureCancelsItsEnergy)
{
    Fixture f;
    auto predecessor = f.before;
    predecessor[0].moles = {0, 0};
    predecessor[1].moles = {f.nl, 0};
    const double old_h = f.Uadv[0] + p_old * f.Vadv[0];
    const auto plan = R::compile_interphase_plan(f.configuration, predecessor, {{0}}, {{-old_h}, {old_h}, {}}, {});
    EXPECT_THROW((void) R::evaluate_pressure_conditioned_thermal(
                     plan, predecessor, f.Uadv, f.Vadv, p_old, R::PhaseThermalMode::Equilibrium),
        std::invalid_argument);
    EXPECT_EQ(f.before[0].moles, std::vector<double>({.4, 0}));
}

TEST(PressureConditionedThermalTest, RawPredecessorDebitSurvivesCompilationAndPressureReplay)
{
    Fixture f;
    auto configuration = *f.configuration;
    auto partial = std::make_shared<const R::PartialMolarEnthalpyLaw>(
        R::PartialMolarEnthalpyLaw{{"raw ideal gas partials", 1, f.catalog->fingerprint()}, f.gas->bounds(),
            [](double t, double, std::span<const double>)
            { return R::PartialMolarEnthalpyPoint{{cp * t, cp * t}, {cp, cp}, {0, 0}}; }});
    configuration.phase_enthalpy = {partial, nullptr, nullptr};
    const auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    const std::vector<R::PhaseThermalState> before{
        {f.gas, {1e20, 0}, p_old, 1e20 * cv * T0, T0}, {f.gas, {0, 0}, p_old, 0, 0},
        {f.gas, {0, 0}, p_old, 0, 0}};
    std::vector<std::vector<std::array<long double, 2>>> raw(3, std::vector<std::array<long double, 2>>(2));
    raw[0][0] = {1e20, -1};
    EXPECT_THROW(
        (void) R::compile_interphase_plan(shared, before, {{1e20}}, {{}, {}, {}}, {}, raw), std::invalid_argument);
    auto forged = R::compile_interphase_plan(shared, before, {{0}}, {{}, {}, {}}, {}, raw);
    EXPECT_EQ(forged.raw_pre_moles, raw);
    forged.amounts = {{1e20}};
    forged.final_moles = {{0, 0}, {1e20, 0}, {0, 0}};
    auto templates = before;
    for (std::size_t a = 0; a < templates.size(); ++a)
        templates[a].moles = forged.final_moles[a];
    const std::vector<double> energy{1e20 * cv * T0, 0, 0},
        volume{1e20 * gas_constant * T0 / p_old, 0, 0};
    EXPECT_THROW((void) R::interphase_pressure_bounds(forged, templates), std::invalid_argument);
    EXPECT_THROW((void) R::evaluate_pressure_conditioned_thermal(
                     forged, templates, energy, volume, p_old, R::PhaseThermalMode::Equilibrium),
        std::invalid_argument);
    EXPECT_DOUBLE_EQ(before[0].moles[0], 1e20);
}
TEST(PressureConditionedThermalTest, RawPredecessorCreditFundsHighAndCorrectionOutgoingMaterialEvents)
{
    Fixture f;
    auto configuration = *f.configuration;
    auto partial = std::make_shared<const R::PartialMolarEnthalpyLaw>(
        R::PartialMolarEnthalpyLaw{{"funded raw ideal gas partials", 1, f.catalog->fingerprint()}, f.gas->bounds(),
            [](double t, double, std::span<const double>)
            { return R::PartialMolarEnthalpyPoint{{cp * t, cp * t}, {cp, cp}, {0, 0}}; }});
    configuration.phase_enthalpy = {partial, nullptr, nullptr};
    configuration.stages = {{"joint predecessor funding", R::InterphaseStageKind::Material,
        {{44, 0, 1}, {44, 0, 2}}, {}}};
    const auto shared = std::make_shared<const R::InterphaseConfiguration>(configuration);
    const std::vector<R::PhaseThermalState> before{
        {f.gas, {1e20, 0}, p_old, 1e20 * cv * T0, T0}, {f.gas, {0, 0}, p_old, 0, 0},
        {f.gas, {0, 0}, p_old, 0, 0}};
    std::vector<std::vector<std::array<long double, 2>>> raw(3, std::vector<std::array<long double, 2>>(2));
    raw[0][0] = {1e20, 1};
    EXPECT_THROW(
        (void) R::compile_interphase_plan(shared, before, {{1e20, 1}}, {{}, {}, {}}, {}), std::invalid_argument);
    const auto plan = R::compile_interphase_plan(shared, before, {{1e20, 1}}, {{}, {}, {}}, {}, raw);
    EXPECT_EQ(plan.final_moles, std::vector<std::vector<double>>({{0, 0}, {1e20, 0}, {1, 0}}));
    auto templates = before;
    for (std::size_t a = 0; a < templates.size(); ++a)
        templates[a].moles = plan.final_moles[a];
    const std::vector<double> energy{1e20 * cv * T0, 0, 0},
        volume{1e20 * gas_constant * T0 / p_old, 0, 0};
    const auto evaluated = R::evaluate_pressure_conditioned_thermal(
        plan, templates, energy, volume, p_old, R::PhaseThermalMode::Equilibrium);
    EXPECT_DOUBLE_EQ(evaluated.phases[2].moles[0], 1);
    EXPECT_EQ(evaluated.material_events.size(), 2u);
    EXPECT_EQ(plan.raw_pre_moles, raw);
    const auto bounds = R::interphase_pressure_bounds(plan, templates);
    EXPECT_LE(bounds.lower, p_old);
    EXPECT_GE(bounds.upper, p_old);
    auto malformed = raw;
    malformed[0][0] = {2e20, 0};
    EXPECT_THROW((void) R::compile_interphase_plan(shared, before, {{0, 0}}, {{}, {}, {}}, {}, malformed),
        std::invalid_argument);
}

} // namespace
