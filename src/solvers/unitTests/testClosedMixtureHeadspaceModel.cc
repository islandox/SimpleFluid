/** @file testClosedMixtureHeadspaceModel.cc
 * @brief Independent pressure/inventory oracles for the prescribed-temperature mixture adapter.
 */
#include "solvers/ClosedMixtureHeadspaceModel.hh"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
namespace S = SimpleFluid;
namespace R = S::Reacting;
constexpr double gas_constant = 8.31446261815324, covolume = 2e-4;
std::shared_ptr<const R::MixtureThermodynamics> provider()
{
    using P = R::SpeciesPhase;
    auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {11, "G1", .020, {{"A", 1}}, {P::Resolved, P::Dispersed}},
        {73, "G2", .040, {{"A", 2}}, {P::Resolved, P::Dispersed}}});
    return std::make_shared<const R::MixtureThermodynamics>(catalog,
        R::ThermodynamicBounds{200., 600., 100., 1e6}, R::EOSKind::Gas,
        [](double temperature, double pressure, std::span<const double> x)
        {
            const double M = .020 * x[0] + .040 * x[1];
            const double b = covolume * (x[0] + 2. * x[1]);
            const double denominator = gas_constant * temperature + b * pressure;
            R::MixtureThermodynamicPoint result;
            result.density = M * pressure / denominator;
            result.density_temperature = -M * pressure * gas_constant / (denominator * denominator);
            result.density_pressure = M * gas_constant * temperature / (denominator * denominator);
            result.internal_energy = 25. * temperature / M;
            result.internal_energy_temperature = 25. / M;
            result.enthalpy = result.internal_energy + pressure / result.density;
            result.fugacities = {x[0] * pressure * std::exp(covolume * pressure / (gas_constant * temperature)),
                x[1] * pressure * std::exp(2. * covolume * pressure / (gas_constant * temperature))};
            return result;
        });
}
S::HeadspaceOptions options()
{
    S::HeadspaceOptions result;
    result.mode = S::HeadspaceMode::Closed;
    result.infer_initial_moles = false;
    result.initial_moles = {{"G1", .5}, {"G2", 1.5}};
    result.initial_temperature = 300.;
    result.total_internal_volume = 1.;
    result.initial_pressure = 2. * gas_constant * 300. / (.5 - 3.5 * covolume);
    return result;
}
TEST(ClosedMixtureHeadspaceModelTest, NonidealMixturePressureUsesExplicitSpeciesAndExactEscape)
{
    const auto selected = options();
    S::ClosedMixtureHeadspaceModel model(selected, provider(), "manufactured-excluded-volume", 1);
    model.initialize(.5, 300.);
    EXPECT_NEAR(model.state().pressure, selected.initial_pressure, 1e-7);
    const S::GasMolesBySpecies escaped{{"G1", .25}, {"G2", .5}};
    const auto trial = model.trialState(.6, 300., escaped);
    const double expected = 2.75 * gas_constant * 300. / (.4 - 4.75 * covolume);
    EXPECT_NEAR(trial.pressure, expected, 1e-7);
    EXPECT_GT(std::abs(trial.pressure - 2.75 * gas_constant * 300. / .4), 1.);
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("G1"), .5);
    model.commit(trial, escaped);
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("G1"), .75);
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("G2"), 2.);
    EXPECT_TRUE(model.ventedMoles().empty());
    EXPECT_EQ(model.accumulatedEscapedMoles(), escaped);
    EXPECT_NEAR(model.state().pressure, expected, 1e-7);
}
TEST(ClosedMixtureHeadspaceModelTest, InvalidInventoryEOSAndInitialPressureLeaveAcceptedState)
{
    auto selected = options();
    selected.infer_initial_moles = true;
    EXPECT_ANY_THROW(S::ClosedMixtureHeadspaceModel(selected, provider(), "manufactured-excluded-volume", 1));
    selected = options();
    selected.initial_pressure *= 2.;
    S::ClosedMixtureHeadspaceModel invalid(selected, provider(), "manufactured-excluded-volume", 1);
    EXPECT_ANY_THROW(invalid.initialize(.5, 300.));
    EXPECT_TRUE(invalid.headspaceMoles().empty());
    S::ClosedMixtureHeadspaceModel model(options(), provider(), "manufactured-excluded-volume", 1);
    model.initialize(.5, 300.);
    const auto before = model.headspaceMoles();
    const auto accepted = model.state();
    EXPECT_ANY_THROW((void) model.trialState(.5, 300., {{"foreign", 1.}}));
    EXPECT_ANY_THROW((void) model.trialState(.5, 300., {{"G1", -1.}}));
    EXPECT_ANY_THROW((void) model.trialState(.999999, 300., {{"G1", 1.}}));
    auto wrong = model.trialState(.5, 300., {{"G1", .1}});
    wrong.pressure *= 1.1;
    EXPECT_ANY_THROW(model.commit(wrong, {{"G1", .1}}));
    EXPECT_EQ(model.headspaceMoles(), before);
    EXPECT_TRUE(model.accumulatedEscapedMoles().empty());
    EXPECT_DOUBLE_EQ(model.state().pressure, accepted.pressure);
    EXPECT_DOUBLE_EQ(model.state().volume, accepted.volume);
}
TEST(ClosedMixtureHeadspaceModelTest, CloningAndRepeatedPreviewsDoNotConsumeSpecies)
{
    S::ClosedMixtureHeadspaceModel model(options(), provider(), "manufactured-excluded-volume", 1);
    model.initialize(.5, 300.);
    auto copy = model.clone();
    const S::GasMolesBySpecies escaped{{"G1", .125}};
    const auto trial = model.trialState(.5, 300., escaped);
    EXPECT_DOUBLE_EQ(model.trialState(.5, 300., escaped).pressure, trial.pressure);
    model.commit(trial, escaped);
    EXPECT_DOUBLE_EQ(copy->headspaceMoles().at("G1"), .5);
    copy->commit(copy->trialState(.5, 300., escaped), escaped);
    EXPECT_EQ(copy->headspaceMoles(), model.headspaceMoles());
}
TEST(ClosedMixtureHeadspaceModelTest, SubUlpEscapesAccumulateAcrossAcceptedStepsAndClones)
{
    S::ClosedMixtureHeadspaceModel model(options(), provider(), "manufactured-excluded-volume", 1);
    model.initialize(.5, 300.);
    const double amount = std::numeric_limits<double>::epsilon() / 32.;
    const S::GasMolesBySpecies escaped{{"G1", amount}};
    for (int step = 0; step < 256; ++step)
        model.commit(model.trialState(.5, 300., escaped), escaped);
    EXPECT_DOUBLE_EQ(model.accumulatedEscapedMoles().at("G1"), 256. * amount);
    const double expected = static_cast<double>(.5L + 256.L * amount);
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("G1"), expected);
    auto clone = model.clone();
    for (int step = 0; step < 256; ++step)
        clone->commit(clone->trialState(.5, 300., escaped), escaped);
    EXPECT_DOUBLE_EQ(clone->headspaceMoles().at("G1"), static_cast<double>(.5L + 512.L * amount));
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("G1"), expected);
}

TEST(ClosedMixtureHeadspaceModelTest, RejectsUninitializedAndNonfiniteStatesWithoutPublishing)
{
    S::ClosedMixtureHeadspaceModel model(options(), provider(), "manufactured-excluded-volume", 1);
    EXPECT_THROW((void) model.trialState(.5, 300., {}), std::logic_error);
    EXPECT_THROW(model.commit({1., .5, 300., 1.}, {}), std::logic_error);
    model.initialize(.5, 300.);
    const auto accepted = model.state();
    const auto trial = model.trialState(.5, 300., {{"G1", .1}});
    for (const auto invalid : {std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN(), -1.})
    {
        auto corrupted = trial;
        corrupted.pressure = invalid;
        EXPECT_ANY_THROW(model.commit(corrupted, {{"G1", .1}}));
        corrupted = trial;
        corrupted.total_moles = invalid;
        EXPECT_ANY_THROW(model.commit(corrupted, {{"G1", .1}}));
        EXPECT_ANY_THROW((void) model.trialState(.5, 300., {{"G2", invalid}}));
    }
    auto oversized = trial;
    oversized.volume = 1.1;
    EXPECT_ANY_THROW(model.commit(oversized, {{"G1", .1}}));
    EXPECT_DOUBLE_EQ(model.state().pressure, accepted.pressure);
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("G1"), .5);
    EXPECT_TRUE(model.accumulatedEscapedMoles().empty());
}

TEST(ClosedMixtureHeadspaceModelTest, FailedTrialsAndCommitsDoNotConsumeSubUlpEvents)
{
    S::ClosedMixtureHeadspaceModel model(options(), provider(), "manufactured-excluded-volume", 1);
    model.initialize(.5, 300.);
    const double amount = std::numeric_limits<double>::epsilon() / 32.;
    const S::GasMolesBySpecies escaped{{"G1", amount}};
    for (int step = 0; step < 256; ++step)
    {
        const auto trial = model.trialState(.5, 300., escaped);
        auto invalid = trial;
        invalid.pressure *= 2.;
        EXPECT_ANY_THROW(model.commit(invalid, escaped));
        EXPECT_ANY_THROW((void) model.trialState(1., 300., escaped));
        // Repeated corrector previews and failed attempts must not advance either accumulator.
        EXPECT_DOUBLE_EQ(model.trialState(.5, 300., escaped).pressure, trial.pressure);
        model.commit(trial, escaped);
    }
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("G1"), static_cast<double>(.5L + 256.L * amount));
    EXPECT_DOUBLE_EQ(model.accumulatedEscapedMoles().at("G1"), 256. * amount);
    const auto before = model.headspaceMoles();
    EXPECT_ANY_THROW(model.initialize(1., 300.));
    EXPECT_EQ(model.headspaceMoles(), before);
    EXPECT_DOUBLE_EQ(model.accumulatedEscapedMoles().at("G1"), 256. * amount);
}

TEST(ClosedMixtureHeadspaceModelTest, PrescribedTemperatureUsesSuppliedEOSWithoutAnEnergyBalance)
{
    auto selected = options();
    selected.temperature_mode = S::HeadspaceTemperatureMode::Prescribed;
    selected.prescribed_temperature_times = {0., 1.};
    selected.prescribed_temperature_values = {300., 450.};
    // Legacy ideal-gas knobs do not determine this adapter's pressure.
    selected.gas_constant = 2.;
    selected.compressibility_factor = 5.;
    S::ClosedMixtureHeadspaceModel model(selected, provider(), "manufactured-excluded-volume", 1);
    model.initialize(.5, 300.);
    const double requested = S::prescribed_headspace_temperature(selected, .5);
    const auto trial = model.trialState(.5, requested, {});
    const double expected = 2. * gas_constant * 375. / (.5 - 3.5 * covolume);
    EXPECT_NEAR(trial.pressure, expected, 1e-7);
    EXPECT_DOUBLE_EQ(trial.temperature, 375.);
    EXPECT_ANY_THROW((void) model.trialState(.5, 700., {}));
    EXPECT_ANY_THROW((void) model.trialState(.5, std::numeric_limits<double>::quiet_NaN(), {}));
    EXPECT_DOUBLE_EQ(model.state().temperature, 300.);
    model.commit(trial, {});
    EXPECT_DOUBLE_EQ(model.state().temperature, 375.);
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("G1"), .5);
    EXPECT_TRUE(model.accumulatedEscapedMoles().empty());
    // Temperature is an imposed bath condition: no adiabatic/carried-energy oracle is asserted.
}

TEST(ClosedMixtureHeadspaceModelTest, ArbitraryCatalogOrderAndNewSpeciesUseActualAmounts)
{
    using P = R::SpeciesPhase;
    auto catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {9007, "vapor", .018, {{"X", 1}}, {P::Liquid, P::Resolved}},
        {4, "spectator", .080, {{"Y", 1}}, {P::Dispersed}},
        {203, "solvent", .050, {{"Z", 1}}, {P::Liquid}},
        {89, "product", .030, {{"W", 1}}, {P::Resolved, P::Dispersed}}});
    const std::vector<R::IdealGasCaloricSpecies> caloric(4, {25., 300., 0.});
    const auto eos = R::make_ideal_gas_mixture(catalog, caloric, {200., 600., 100., 1e6});
    auto selected = options();
    selected.initial_moles = {{"spectator", .25}, {"vapor", .75}};
    selected.initial_pressure = gas_constant * 300. / .5;
    S::ClosedMixtureHeadspaceModel model(selected, eos, "manufactured-mixture", 1);
    model.initialize(.5, 300.);
    const S::GasMolesBySpecies escaped{{"product", .5}, {"vapor", .125}};
    const auto trial = model.trialState(.6, 300., escaped);
    EXPECT_NEAR(trial.pressure, 1.625 * gas_constant * 300. / .4, 1e-7);
    model.commit(trial, escaped);
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("product"), .5);
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("vapor"), .875);
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("spectator"), .25);
    EXPECT_EQ(model.accumulatedEscapedMoles(), escaped);
    const auto before = model.headspaceMoles();
    EXPECT_ANY_THROW((void) model.trialState(.5, 300., {{"solvent", .01}}));
    EXPECT_ANY_THROW((void) model.trialState(.5, 300., {{"unnamedAir", 0.}}));
    EXPECT_EQ(model.headspaceMoles(), before);
}

TEST(ClosedMixtureHeadspaceModelTest, SuppliedPressureBoundsAndCapacityAreHardLimits)
{
    // Choose an exactly representable volume using the same supplied EOS relation;
    // this exercises the root endpoint rather than weakening its bracket.
    auto eos = provider();
    const auto initial = options();
    const long double mass = .5L * .020 + 1.5L * .040;
    const std::vector<double> x{.25, .75};
    const double upper = eos->bounds().maximum_pressure;
    const double at_upper = static_cast<double>(mass / eos->evaluate(300., upper, x).density);
    auto selected = initial;
    selected.total_internal_volume = at_upper;
    selected.initial_pressure = upper;
    S::ClosedMixtureHeadspaceModel model(selected, eos, "manufactured-mixture", 1);
    model.initialize(0., 300.);
    EXPECT_DOUBLE_EQ(model.state().pressure, upper);
    EXPECT_ANY_THROW((void) model.trialState(at_upper / 2., 300., {}));
    EXPECT_ANY_THROW((void) model.trialState(0., 300., {{"G1", 1.}}));
    EXPECT_DOUBLE_EQ(model.state().pressure, upper);
    EXPECT_TRUE(model.accumulatedEscapedMoles().empty());
    auto invalid = selected;
    invalid.initial_moles = {{"G1", std::numeric_limits<double>::max()},
        {"G2", std::numeric_limits<double>::max()}};
    EXPECT_ANY_THROW(S::ClosedMixtureHeadspaceModel(invalid, eos, "manufactured-mixture", 1));
}

TEST(ClosedMixtureHeadspaceModelTest, RejectsUnrepresentableTraceCompositionInsteadOfDroppingSpecies)
{
    S::ClosedMixtureHeadspaceModel model(options(), provider(), "manufactured-excluded-volume", 1);
    model.initialize(.5, 300.);
    // A trace added to an existing component cannot disappear from the event ledger.
    const double tiny = std::numeric_limits<double>::denorm_min();
    const S::GasMolesBySpecies escaped{{"G1", tiny}};
    model.commit(model.trialState(.5, 300., escaped), escaped);
    EXPECT_DOUBLE_EQ(model.accumulatedEscapedMoles().at("G1"), tiny);
    auto pure = options();
    pure.initial_moles = {{"G1", 2.}};
    pure.initial_pressure = 2. * gas_constant * 300. / (.5 - 2. * covolume);
    S::ClosedMixtureHeadspaceModel pure_model(pure, provider(), "manufactured-excluded-volume", 1);
    pure_model.initialize(.5, 300.);
    EXPECT_ANY_THROW((void) pure_model.trialState(.5, 300., {{"G2", tiny}}));
    EXPECT_FALSE(pure_model.headspaceMoles().contains("G2"));
    EXPECT_TRUE(pure_model.accumulatedEscapedMoles().empty());
    const auto before = model.headspaceMoles();
    EXPECT_ANY_THROW((void) model.trialState(.5, 300., {{"G1", std::numeric_limits<double>::max()}}));
    EXPECT_EQ(model.headspaceMoles(), before);
    EXPECT_DOUBLE_EQ(model.accumulatedEscapedMoles().at("G1"), tiny);
}


TEST(ClosedMixtureHeadspaceModelTest, CloneRetainsCompensationBelowPublishedInventoryResolution)
{
    S::ClosedMixtureHeadspaceModel model(options(), provider(), "manufactured-excluded-volume", 1);
    model.initialize(.5, 300.);
    const double amount = std::numeric_limits<double>::epsilon() / 32.;
    const S::GasMolesBySpecies escaped{{"G1", amount}};
    for (int step = 0; step < 7; ++step)
        model.commit(model.trialState(.5, 300., escaped), escaped);
    EXPECT_DOUBLE_EQ(model.headspaceMoles().at("G1"), .5);
    auto clone = model.clone();
    for (int step = 0; step < 2; ++step)
        clone->commit(clone->trialState(.5, 300., escaped), escaped);
    EXPECT_DOUBLE_EQ(clone->headspaceMoles().at("G1"), static_cast<double>(.5L + 9.L * amount));
    EXPECT_GT(clone->headspaceMoles().at("G1"), model.headspaceMoles().at("G1"));
    EXPECT_DOUBLE_EQ(model.accumulatedEscapedMoles().at("G1"), 7. * amount);
}

TEST(ClosedMixtureHeadspaceModelTest, InvalidTemperaturePolicyOrMissingEOSIsRejected)
{
    auto selected = options();
    selected.temperature_mode = static_cast<S::HeadspaceTemperatureMode>(-1);
    EXPECT_ANY_THROW(S::ClosedMixtureHeadspaceModel(selected, provider(), "manufactured-excluded-volume", 1));
    EXPECT_ANY_THROW(S::ClosedMixtureHeadspaceModel(options(), nullptr, "missing-eos", 1));
}

} // namespace
