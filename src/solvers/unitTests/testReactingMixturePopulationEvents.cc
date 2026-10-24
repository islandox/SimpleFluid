/** Supplied population events, raw count budgets and coupled receipt/replay regressions. */
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/details/LowMachPopulationSource.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>

namespace
{
using namespace SimpleFluid;
namespace R = Reacting;
namespace P = R::population_source_detail;
using Flow = R::LowMachMixtureSolver;
using Coupler = R::ReactingMixtureCoupler;
using Event = BubbleGasPhysics::PopulationNumberEvent;
using LO = Flow::Pack::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
const std::vector<Flow::PhaseRoleDescriptor> roles{{Flow::PhaseRole::BulkLiquid, "liquid"},
    {Flow::PhaseRole::DispersedGas, "micro"}, {Flow::PhaseRole::DispersedGas, "large"}};

TEST(ReactingMixturePopulationEventsTest, PureFixedMaterialCoalescenceBreakupAndGrossChannels)
{
    const std::vector<std::optional<double>> before{std::nullopt, 100, 40};
    const std::vector<int> occupied{1, 1, 1};
    const auto result = P::apply(roles, before, occupied, {{"micro", 0, 25}, {"large", 20, 0}, {"micro", 2, 3}});
    EXPECT_EQ(result.events.size(), 3);
    EXPECT_EQ(result.changes, std::vector<double>({0, -26, 20}));
    EXPECT_EQ(result.births, std::vector<double>({0, 2, 20}));
    EXPECT_EQ(result.deaths, std::vector<double>({0, 28, 0}));
    EXPECT_DOUBLE_EQ(*result.final_numbers[1], 74);
    EXPECT_DOUBLE_EQ(*result.final_numbers[2], 60);
    EXPECT_EQ(result.transported_numbers, before);
}
TEST(ReactingMixturePopulationEventsTest, ExactGasAbsenceRequiresExplicitDeathAndCreationRequiresBirth)
{
    const std::vector<std::optional<double>> before{std::nullopt, 10, 0};
    EXPECT_THROW((void) P::apply(roles, before, std::vector<int>{1, 0, 0}, {}), std::exception);
    const auto empty = P::apply(roles, before, std::vector<int>{1, 0, 0}, {{"micro", 0, 10}});
    EXPECT_DOUBLE_EQ(*empty.final_numbers[1], 0);
    EXPECT_THROW((void) P::apply(roles, before, std::vector<int>{1, 1, 0}, {{"micro", 0, 10}}), std::exception);
    EXPECT_THROW((void) P::apply(roles, before, std::vector<int>{1, 1, 1}, {}), std::exception);
    const auto born = P::apply(roles, before, std::vector<int>{1, 1, 1}, {{"large", 3, 0}});
    EXPECT_DOUBLE_EQ(*born.final_numbers[2], 3);
}
TEST(ReactingMixturePopulationEventsTest, DeathReservationsCannotBorrowBirthsAndNoFiniteResidualMayDisappear)
{
    const std::vector<std::optional<double>> before{std::nullopt, 1, 0};
    EXPECT_THROW((void) P::apply(roles, before, std::vector<int>{1, 1, 0}, {{"micro", 5, 2}}), std::exception);
    const std::vector<std::optional<double>> huge{std::nullopt, 1e20, 0};
    EXPECT_THROW((void) P::apply(roles, huge, std::vector<int>{1, 1, 0}, {{"micro", 1, 1e20}}), std::exception);
    const auto exact = P::apply(roles, before, std::vector<int>{1, 1, 0}, {{"micro", .5, .25}});
    EXPECT_DOUBLE_EQ(*exact.final_numbers[1], 1.25);
}
TEST(ReactingMixturePopulationEventsTest, InvalidOrUnknownNamedEventsRejectWithoutMutatingInputs)
{
    const std::vector<std::optional<double>> before{std::nullopt, 10, 0};
    for (const auto event : std::vector<Event>{{"unbound", 1, 0}, {"liquid", 1, 0}, {"micro", -1, 0},
             {"micro", 0, std::numeric_limits<double>::infinity()},
             {"micro", std::numeric_limits<double>::quiet_NaN(), 0}})
        EXPECT_THROW((void) P::apply(roles, before, std::vector<int>{1, 1, 0}, {event}), std::exception);
    EXPECT_DOUBLE_EQ(*before[1], 10);
}
TEST(ReactingMixturePopulationEventsTest, OpposingHugeRawEventsRetainNetOneAtUnchangedLargeEndpoint)
{
    const std::vector<std::optional<double>> before{std::nullopt, 1e20, 0};
    const auto result = P::apply(roles, before, std::vector<int>{1, 1, 0}, {{"micro", 1e20, 1e20}, {"micro", 1, 0}});
    EXPECT_DOUBLE_EQ(result.changes[1], 1);
    EXPECT_DOUBLE_EQ(*result.final_numbers[1], 1e20);
    EXPECT_EQ(result.events.size(), 2);
}
TEST(ReactingMixturePopulationEventsTest, RawDeathMarginRejectsOneBeyondHugeDonorEvenWithIncomingBirth)
{
    const std::vector<std::optional<double>> before{std::nullopt, 1e20, 0};
    EXPECT_THROW((void) P::apply(roles, before, std::vector<int>{1, 1, 0}, {{"micro", 1e20, 1e20}, {"micro", 0, 1}}),
        std::exception);
    EXPECT_THROW((void) P::apply(roles, before, std::vector<int>{1, 1, 0}, {{"micro", 0, 1}, {"micro", 1e20, 1e20}}),
        std::exception);
}
struct Fixture
{
    SP<Flow::Mesh> mesh;
    std::shared_ptr<const R::SpeciesCatalog> catalog;
    std::shared_ptr<const R::MixtureThermodynamics> liquid, gas;
    std::shared_ptr<const R::ReactionMechanism> mechanism;
    std::shared_ptr<Flow> flow;
    explicit Fixture(bool seeded = true)
    {
        Flow::Mesh::DistributionOptions distribution;
        distribution.allow_empty_partitions = true;
        mesh = std::make_shared<Flow::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                                Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, .5, 1}}),
            distribution);
        catalog = std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
            {5, "A", .10, {{"X", 4}}, {R::SpeciesPhase::Liquid}}, {31, "B", .04, {{"X", 2}}, {R::SpeciesPhase::Liquid}},
            {101, "G1", .02, {{"X", 1}}, {R::SpeciesPhase::Dissolved, R::SpeciesPhase::Dispersed}},
            {709, "G2", .04, {{"X", 1}}, {R::SpeciesPhase::Dissolved, R::SpeciesPhase::Dispersed}}});
        const auto convention = R::EnergyConvention::IncludesFormationEnergy;
        liquid = R::make_linear_compressible_mixture_liquid(
            catalog, {250, 600, 10000, 2000000}, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1e6, 0, 0}; }, convention);
        gas = R::make_ideal_gas_mixture(catalog, {{20, 300, 6000}, {20, 300, 6000}, {20, 300, 6000}, {20, 300, 6000}},
            {250, 600, 10000, 2000000}, convention);
        mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
            std::vector<R::ReactionDefinition>{{42, "declared-gas-production",
                {{5, R::SpeciesPhase::Liquid, -1, {}}, {31, R::SpeciesPhase::Liquid, 1, {}},
                    {101, R::SpeciesPhase::Dispersed, 1, "micro"}, {709, R::SpeciesPhase::Dispersed, 1, "large"}},
                0}},
            convention);
        Flow::Options options;
        options.phase_roles = roles;
        options.minimum_pressure = 20000;
        options.maximum_pressure = 1000000;
        options.linear.tolerance = 1e-12;
        options.linear.preconditioner = LinearPreconditioner::None;
        flow = std::make_shared<Flow>(mesh, std::vector{liquid, gas, gas}, options);
        Flow::State initial;
        initial.thermodynamic_pressure = 100000;
        initial.cells.resize(mesh->num_owned_cells());
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            const double volume = mesh->cell_volume(static_cast<LO>(c));
            const double mass = (seeded ? .8 : 1.) * 1000 * volume;
            const double gas_moles = seeded ? 100000 * .1 * volume / (8.31446261815324 * 300) : 0;
            initial.cells[c].phases = {{liquid, {mass / .1, 0, 0, 0}, 100000, mass * 1e6, 300},
                {gas, {0, 0, gas_moles, 0}, 100000, 6000 * gas_moles, seeded ? 300. : 0.},
                {gas, {0, 0, 0, gas_moles}, 100000, 6000 * gas_moles, seeded ? 300. : 0.}};
            initial.cells[c].phase_numbers = {std::nullopt, seeded ? 100. : 0., seeded ? 40. : 0.};
        }
        flow->initialize(std::move(initial));
    }
    R::ReactingMixtureOptions options() const
    {
        R::ReactingMixtureOptions options;
        options.mechanism = mechanism;
        options.source_identity = "source-only-count-test";
        options.phases = {{R::SpeciesPhase::Liquid, {},
                              {R::SpeciesPhase::Liquid, R::SpeciesPhase::Liquid, R::SpeciesPhase::Dissolved,
                                  R::SpeciesPhase::Dissolved}},
            {R::SpeciesPhase::Dispersed, "micro", {}}, {R::SpeciesPhase::Dispersed, "large", {}}};
        return options;
    }
    template<class Law>
    auto provider(Law law, const std::string& identity = "explicit-number-test", std::uint64_t revision = 1,
        std::uint64_t schema = 0) const
    {
        return std::make_shared<const R::MixturePopulationEventProvider>(R::MixturePopulationEventProvider{
            identity, revision, schema ? schema : catalog->fingerprint(), std::move(law)});
    }
};
void same(const Flow::State& a, const Flow::State& b)
{
    EXPECT_EQ(a.revision, b.revision);
    EXPECT_DOUBLE_EQ(a.time, b.time);
    EXPECT_DOUBLE_EQ(a.thermodynamic_pressure, b.thermodynamic_pressure);
    ASSERT_EQ(a.cells.size(), b.cells.size());
    for (size_t c = 0; c < a.cells.size(); ++c)
    {
        EXPECT_EQ(a.cells[c].phase_numbers, b.cells[c].phase_numbers);
        EXPECT_DOUBLE_EQ(a.cells[c].mechanical_pressure, b.cells[c].mechanical_pressure);
        ASSERT_EQ(a.cells[c].drift_velocity.size(), b.cells[c].drift_velocity.size());
        for (size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_DOUBLE_EQ(a.cells[c].velocity.component(axis), b.cells[c].velocity.component(axis));
            for (size_t p = 0; p < a.cells[c].drift_velocity.size(); ++p)
                EXPECT_DOUBLE_EQ(
                    a.cells[c].drift_velocity[p].component(axis), b.cells[c].drift_velocity[p].component(axis));
        }
        ASSERT_EQ(a.cells[c].phases.size(), b.cells[c].phases.size());
        for (size_t p = 0; p < a.cells[c].phases.size(); ++p)
        {
            const auto& x = a.cells[c].phases[p];
            const auto& y = b.cells[c].phases[p];
            EXPECT_EQ(x.thermodynamics.get(), y.thermodynamics.get());
            EXPECT_EQ(x.moles, y.moles);
            EXPECT_DOUBLE_EQ(x.internal_energy, y.internal_energy);
            EXPECT_DOUBLE_EQ(x.temperature, y.temperature);
            EXPECT_DOUBLE_EQ(x.absolute_pressure, y.absolute_pressure);
        }
    }
}
TEST(ReactingMixturePopulationEventsTest, NewChemicalGasNeedsExplicitBirthAndLawSeesActualPostmaterialMoles)
{
    Fixture missing(false);
    auto no_birth = missing.options();
    no_birth.source = [](const auto&, double)
    {
        R::ReactionSourceInput input;
        input.integratedExtents = {{42, .001}};
        return input;
    };
    const auto accepted = missing.flow->state();
    {
        Coupler controller(missing.flow, no_birth);
        EXPECT_THROW((void) controller.preview(.01), std::exception);
        same(missing.flow->state(), accepted);
    }
    Fixture f(false);
    auto options = f.options();
    options.source = no_birth.source;
    options.population_events = f.provider(
        [](const R::MixturePopulationEventContext& context)
        {
            if (*context.transported.phases[1].population_number != 0 ||
                context.post_material.amount(context.catalog, 101, R::SpeciesPhase::Dispersed, "micro") != .001 ||
                context.thermodynamic_guess.phases[1].moles[2] != .001 ||
                *context.thermodynamic_guess.phase_numbers[1] != 0)
                throw std::runtime_error("Number law did not receive actual postmaterial moles and Nadv.");
            return std::vector<Event>{{"micro", 1, 0}, {"large", 2, 0}};
        });
    Coupler controller(f.flow, options);
    const auto trial = controller.preview(.01);
    for (size_t c = 0; c < trial.state().cells.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(*trial.state().cells[c].phase_numbers[1], 1);
        EXPECT_DOUBLE_EQ(*trial.state().cells[c].phase_numbers[2], 2);
        EXPECT_DOUBLE_EQ(*trial.transport_snapshot().cells[c].phases[1].population_number, 0);
        EXPECT_EQ(trial.local_population_events()[c].provider_identity, "explicit-number-test");
        EXPECT_EQ(trial.local_population_events()[c].source_stage, trial.transport_snapshot().stage);
    }
}
TEST(ReactingMixturePopulationEventsTest, FixedMoleNumberLawRetainsEventsAndCheckpointReplayWithoutCallbacksOnCommit)
{
    Fixture f;
    auto options = f.options();
    int callbacks = 0;
    options.population_events = f.provider(
        [&](const auto& context)
        {
            ++callbacks;
            return std::vector<Event>{{"micro", 0, .25 * *context.transported.phases[1].population_number},
                {"large", .5 * *context.transported.phases[2].population_number, 0}};
        });
    Coupler controller(f.flow, options);
    const auto checkpoint = controller.checkpoint();
    const auto accepted = f.flow->state();
    const auto first = controller.preview(.01), retained = controller.preview(.01);
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(*first.state().cells[c].phase_numbers[1], 75);
        EXPECT_DOUBLE_EQ(*first.state().cells[c].phase_numbers[2], 60);
        EXPECT_EQ(first.state().cells[c].phases[1].moles, accepted.cells[c].phases[1].moles);
        EXPECT_EQ(first.local_population_events()[c].births, std::vector<double>({0, 0, 20}));
        EXPECT_EQ(first.local_population_events()[c].deaths, std::vector<double>({0, 25, 0}));
    }
    EXPECT_DOUBLE_EQ(first.diagnostics().population_births[2], 40);
    EXPECT_DOUBLE_EQ(first.diagnostics().population_deaths[1], 50);
    same(f.flow->state(), accepted);
    const int calls = callbacks;
    controller.commit(first);
    EXPECT_EQ(callbacks, calls);
    EXPECT_THROW(controller.commit(retained), std::exception);
    const auto expected = f.flow->state();
    controller.restore(checkpoint);
    EXPECT_EQ(callbacks, calls);
    const auto replay = controller.preview(.01);
    same(replay.state(), expected);
    EXPECT_EQ(first.diagnostics().population_number_changes, replay.diagnostics().population_number_changes);
}
TEST(ReactingMixturePopulationEventsTest, RankLocalLawFailurePreservesEarlierTrialAndAcceptedNumbers)
{
    Fixture f;
    auto options = f.options();
    const auto comm = f.mesh->owned_cell_map()->getComm();
    options.population_events = f.provider(
        [comm](const auto& context)
        {
            if (comm->getRank() == 0 && context.time_step > .015)
                throw std::runtime_error("Injected pure number-law failure.");
            return std::vector<Event>{{"micro", 1, 0}};
        });
    Coupler controller(f.flow, options);
    const auto accepted = f.flow->state();
    const auto valid = controller.preview(.01);
    EXPECT_THROW((void) controller.preview(.02), std::exception);
    same(f.flow->state(), accepted);
    EXPECT_NO_THROW(controller.commit(valid));
}
TEST(ReactingMixturePopulationEventsTest, NumberOnlyDepletionCannotSilentlyDestroyExistingGas)
{
    Fixture f;
    auto options = f.options();
    options.population_events = f.provider([](const auto& context)
        { return std::vector<Event>{{"micro", 0, *context.transported.phases[1].population_number}}; });
    const auto accepted = f.flow->state();
    Coupler controller(f.flow, options);
    EXPECT_THROW((void) controller.preview(.01), std::exception);
    same(f.flow->state(), accepted);
}
TEST(ReactingMixturePopulationEventsTest, ProviderMetadataAndNamesRejectDivergenceBeforeCallsIncludingEmptyOwners)
{
    Fixture f;
    const auto comm = f.mesh->owned_cell_map()->getComm();
    int calls = 0;
    int empty = f.mesh->num_owned_cells() == 0, total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total);
    if (comm->getSize() == 4)
        EXPECT_EQ(total, 2);
    for (int scenario = 0; scenario < 5; ++scenario)
    {
        auto options = f.options();
        const bool change = comm->getRank() == comm->getSize() - 1;
        auto law = [&](const auto&)
        {
            ++calls;
            return std::vector<Event>{};
        };
        options.population_events = f.provider(law,
            change && scenario == 1 ? "other-number-test" : "explicit-number-test", change && scenario == 2 ? 2 : 1,
            change && scenario == 3 ? f.catalog->fingerprint() + 1 : f.catalog->fingerprint());
        if (change && scenario == 0)
            options.population_events.reset();
        if (change && scenario == 4)
            options.phases[1].population = "other";
        if (comm->getSize() == 1 && scenario < 3)
            options.population_events = f.provider(law, "", 1);
        EXPECT_THROW((void) Coupler(f.flow, options), std::exception);
    }
    EXPECT_EQ(calls, 0);
}
TEST(ReactingMixturePopulationEventsTest, GrossBirthDeathConvergenceCannotHideBehindUnchangedNetNumber)
{
    Fixture f;
    auto options = f.options();
    options.population_events = f.provider(
        [](const auto& context)
        {
            const double turnover = context.stage.corrector == 0 ? 1 : 2;
            return std::vector<Event>{{"micro", turnover, turnover}};
        });
    Coupler controller(f.flow, options);
    const auto trial = controller.preview(.01);
    EXPECT_GE(trial.diagnostics().correctors, 3);
    for (const auto& receipt : trial.local_population_events())
    {
        EXPECT_DOUBLE_EQ(receipt.births[1], 2);
        EXPECT_DOUBLE_EQ(receipt.deaths[1], 2);
        EXPECT_DOUBLE_EQ(receipt.changes[1], 0);
    }
}
} // namespace
