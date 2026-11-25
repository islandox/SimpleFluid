#include "equations/SpeciesBubbleProcesses.hh"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <numbers>

namespace
{
namespace B = SimpleFluid::BubbleGasPhysics;
namespace R = SimpleFluid::Reacting;
using P = R::SpeciesPhase;

struct Fixture
{
    // Deliberately noncontiguous IDs, reversed gas order and a nongas solvent.
    R::SpeciesCatalog catalog{{{91, "gas B", .040, {{"B", 1}}, {P::Dissolved, P::Dispersed}},
        {2, "solvent", .018, {{"H", 2}, {"O", 1}}, {P::Liquid}},
        {17, "gas A", .020, {{"A", 1}}, {P::Dissolved, P::Dispersed}}}};
    R::CellSpeciesState state{catalog};
    R::ReactionEnvironment environment{0, 300, 100000, .5, 2};
    std::vector<B::PopulationCondition> conditions{{"small", 100, .25, {}}, {"upper", 0, 0, {}}};

    Fixture()
    {
        state.dispersedPopulations = {{"small", {3, 0, 1}}, {"upper", {0, 0, 0}}};
        state.set_amount(catalog, 17, P::Dissolved, 2);
        state.set_amount(catalog, 91, P::Dissolved, 6);
        state.set_amount(catalog, 2, P::Liquid, 10);
    }
    B::MultispeciesPopulationPacket packet(const B::PopulationProcess& process, double dt = 1) const
    {
        const R::ReactionContext chemistry{catalog, state, environment};
        const B::MultispeciesPopulationContext context{chemistry, conditions};
        return process(context, dt);
    }
    B::MultispeciesPopulationResult advance(const B::PopulationProcess& process, double dt = 1) const
    {
        return B::advance_multispecies_population_exchange(catalog, state,
            {{"small", conditions[0].number}, {"upper", conditions[1].number}}, environment, packet(process, dt));
    }
};

B::PopulationRate constant(double rate)
{
    return [rate](const B::MultispeciesPopulationContext&) { return rate; };
}

double total(const Fixture& f, const R::CellSpeciesState& state, R::SpeciesId id)
{
    return state.amount(f.catalog, id, P::Dissolved) + state.phase_amount(f.catalog, id, P::Dispersed);
}

TEST(SpeciesBubbleProcessesTest, MixedNucleationUsesActualBulkVolumeAndSuppliedNumberLaw)
{
    Fixture f;
    const auto result = f.advance(
        B::make_nucleation_process("upper", {{17, constant(.5)}, {91, constant(1.5)}}, constant(50)), .25);
    EXPECT_DOUBLE_EQ(result.state.amount(f.catalog, 17, P::Dispersed, "upper"), .25);
    EXPECT_DOUBLE_EQ(result.state.amount(f.catalog, 91, P::Dispersed, "upper"), .75);
    EXPECT_DOUBLE_EQ(result.numbers[1].number, 25);
    EXPECT_DOUBLE_EQ(total(f, result.state, 17), 3);
    EXPECT_DOUBLE_EQ(total(f, result.state, 91), 9);
    EXPECT_DOUBLE_EQ(result.state.amount(f.catalog, 2, P::Liquid), 10);
    EXPECT_DOUBLE_EQ(result.ledger.thermal_energy(), 0);
}

TEST(SpeciesBubbleProcessesTest, FrozenDecayAndConversionPreserveEverySpeciesAndOwnCounts)
{
    Fixture f;
    const auto process = B::compose_population_processes({
        B::make_population_dissolution_process("small", constant(std::log(4. / 3))),
        B::make_population_conversion_process("small", "upper", constant(std::log(4. / 3)))});
    const auto result = f.advance(process);
    // Each law independently consumes exactly one quarter of the ORIGINAL mixed population.
    EXPECT_NEAR(result.state.amount(f.catalog, 17, P::Dispersed, "small"), .5, 1e-15);
    EXPECT_NEAR(result.state.amount(f.catalog, 91, P::Dispersed, "small"), 1.5, 1e-15);
    EXPECT_NEAR(result.state.amount(f.catalog, 17, P::Dispersed, "upper"), .25, 1e-15);
    EXPECT_NEAR(result.state.amount(f.catalog, 91, P::Dispersed, "upper"), .75, 1e-15);
    EXPECT_NEAR(result.state.amount(f.catalog, 17, P::Dissolved), 2.25, 1e-15);
    EXPECT_NEAR(result.numbers[0].number, 50, 1e-13);
    EXPECT_NEAR(result.numbers[1].number, 25, 1e-13);
    EXPECT_DOUBLE_EQ(total(f, result.state, 17), 3);
    EXPECT_DOUBLE_EQ(total(f, result.state, 91), 9);
    EXPECT_DOUBLE_EQ(result.ledger.thermal_energy(), 0);
}

TEST(SpeciesBubbleProcessesTest, FilmUsesSpeciesEquilibriumActualLiquidVolumeAndPhysicalArea)
{
    Fixture f;
    const double coefficient = 1 / (4 * std::numbers::pi * .25 * .25 * 100);
    const auto law = [coefficient](const B::MultispeciesPopulationContext&, const B::PopulationCondition&)
    { return coefficient; };
    const auto eqA = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition&) { return 1.; };
    const auto eqB = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition&) { return 16.; };
    const auto result = f.advance(B::make_film_transfer_process("small", {{17, law, eqA}, {91, law, eqB}}), .1);
    // area*kL=1 m3/s: A gains .1*(2/.5-1)=.3 mol; B loses .1*(16-6/.5)=.4 mol.
    EXPECT_NEAR(result.state.amount(f.catalog, 17, P::Dispersed, "small"), 1.3, 1e-15);
    EXPECT_NEAR(result.state.amount(f.catalog, 91, P::Dispersed, "small"), 2.6, 1e-15);
    EXPECT_DOUBLE_EQ(result.numbers[0].number, 100);
    EXPECT_DOUBLE_EQ(total(f, result.state, 17), 3);
    EXPECT_DOUBLE_EQ(total(f, result.state, 91), 9);
    EXPECT_DOUBLE_EQ(result.ledger.thermal_energy(), 0);
}

TEST(SpeciesBubbleProcessesTest, ImplicitFilmResolvesStiffAbsorptionAndDesorption)
{
    Fixture f;
    f.conditions[0].properties.fugacities = {13e5, 0, 1e5};
    const double coefficient = 10 / (4 * std::numbers::pi * .25 * .25 * 100);
    const auto rate = [coefficient](const B::MultispeciesPopulationContext&, const B::PopulationCondition&)
    { return coefficient; };
    const auto equilibrium = [](R::SpeciesId id)
    {
        return [id](const B::MultispeciesPopulationContext& c, const B::PopulationCondition& p)
        { return 1e-5 * p.properties.fugacities.at(c.chemistry.catalog.index(id)); };
    };
    const std::vector<B::SpeciesFilmLaw> laws{{17, rate, equilibrium(17)}, {91, rate, equilibrium(91)}};
    EXPECT_THROW((void) f.advance(B::make_film_transfer_process("small", laws)), std::runtime_error);
    const auto result = f.advance(B::make_implicit_film_transfer_process({{"small", laws}}));
    // dt*kL*A/V_liquid = 20. The dissolved backward Euler relations are 21*D'_A = 12 and 21*D'_B = 136.
    EXPECT_NEAR(result.state.amount(f.catalog, 17, P::Dissolved), 12. / 21, 2e-15);
    EXPECT_NEAR(result.state.amount(f.catalog, 91, P::Dissolved), 136. / 21, 2e-15);
    EXPECT_NEAR(result.state.amount(f.catalog, 17, P::Dispersed, "small"), 51. / 21, 2e-15);
    EXPECT_NEAR(result.state.amount(f.catalog, 91, P::Dispersed, "small"), 53. / 21, 2e-15);
    EXPECT_DOUBLE_EQ(total(f, result.state, 17), 3);
    EXPECT_DOUBLE_EQ(total(f, result.state, 91), 9);
    EXPECT_DOUBLE_EQ(result.numbers[0].number, 100);
    EXPECT_DOUBLE_EQ(result.ledger.thermal_energy(), 0);
}

TEST(SpeciesBubbleProcessesTest, ImplicitFilmAvoidsTheMeasuredCell20161Overshoot)
{
    Fixture f;
    // R196 reproducer, step 18, corrector 11, at the source-stage input.
    constexpr double dt = 1e-4, alpha = 1.47306279, dissolved = 1.8771802469915432e-5;
    constexpr double dispersed = 1.6840765868664801e-5, number = 421069416530.17004;
    constexpr double radius = 3.8285542718019145e-7, fugacity = 471771.98251515924;
    f.environment = {0., 333.49002917934712, 101325., 1.6496524760906874e-6, 1.8860912397731989e-6};
    f.conditions[0] = {"small", number, radius, {}};
    f.state.set_amount(f.catalog, 17, P::Dissolved, dissolved);
    f.state.set_amount(f.catalog, 17, P::Dispersed, dispersed, "small");
    const B::SpeciesFilmLaw law{17, [](const auto& context, const auto&)
        {
            return alpha * context.chemistry.environment.liquidVolume /
                   (dt * 4 * std::numbers::pi * radius * radius * number);
        }, [](const auto&, const auto&) { return 1e-5 * fugacity; }};
    const double equilibrium = 1e-5 * fugacity * f.environment.liquidVolume;
    const auto explicitResult = f.advance(B::make_film_transfer_process("small", {law}), dt);
    EXPECT_LT(explicitResult.state.amount(f.catalog, 17, P::Dissolved), equilibrium);
    const auto implicitResult = f.advance(B::make_implicit_film_transfer_process({{"small", {law}}}), dt);
    const auto endpoint = equilibrium + (dissolved - equilibrium) / (1. + alpha);
    EXPECT_GT(implicitResult.state.amount(f.catalog, 17, P::Dissolved), equilibrium);
    EXPECT_NEAR(implicitResult.state.amount(f.catalog, 17, P::Dissolved), endpoint, 1e-20);
    EXPECT_NEAR(implicitResult.state.amount(f.catalog, 17, P::Dispersed, "small") - dispersed,
        dissolved - endpoint, 1e-20);
    EXPECT_DOUBLE_EQ(implicitResult.numbers[0].number, number);
}

TEST(SpeciesBubbleProcessesTest, ImplicitFilmUsesOneEndpointForCompetingPopulations)
{
    Fixture f;
    f.conditions[1] = {"upper", 50, .125, {}};
    f.state.set_amount(f.catalog, 17, P::Dispersed, .75, "upper");
    const auto rate = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition& p)
    { return 5 / (4 * std::numbers::pi * p.radius * p.radius * p.number); };
    const auto equilibrium = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition& p)
    { return p.id == "small" ? 1. : 1.25; };
    const B::PopulationFilmLaws small{"small", {{17, rate, equilibrium}}};
    const B::PopulationFilmLaws upper{"upper", {{17, rate, equilibrium}}};
    EXPECT_THROW((void) f.advance(B::compose_population_processes({B::make_implicit_film_transfer_process({small}),
                     B::make_implicit_film_transfer_process({upper})})), std::runtime_error);

    // Both films have alpha = 10 and request from the same D' = (2 + 5 + 6.25)/21.
    for (const auto& groups : {std::vector{small, upper}, std::vector{upper, small}})
    {
        const auto result = f.advance(B::make_implicit_film_transfer_process(groups));
        EXPECT_NEAR(result.state.amount(f.catalog, 17, P::Dissolved), 13.25 / 21, 2e-15);
        EXPECT_NEAR(result.state.amount(f.catalog, 17, P::Dispersed, "small"), 1 + 27.5 / 21, 2e-15);
        EXPECT_NEAR(result.state.amount(f.catalog, 17, P::Dispersed, "upper"), .75 + 1.25 / 21, 2e-15);
        EXPECT_DOUBLE_EQ(total(f, result.state, 17), 3.75);
        EXPECT_DOUBLE_EQ(result.numbers[0].number, 100);
        EXPECT_DOUBLE_EQ(result.numbers[1].number, 50);
    }
}

TEST(SpeciesBubbleProcessesTest, ImplicitFilmPreservesDrivingDifferenceAtVeryLargeWeight)
{
    Fixture f;
    const auto coefficient = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition&)
    { return 1e30; };
    const auto equilibrium = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition&) { return 1.; };
    const auto film = B::make_implicit_film_transfer_process({{"small", {{17, coefficient, equilibrium}}}});
    const auto result = f.advance(film);
    EXPECT_DOUBLE_EQ(result.state.amount(f.catalog, 17, P::Dissolved), .5);
    EXPECT_DOUBLE_EQ(result.state.amount(f.catalog, 17, P::Dispersed, "small"), 2.5);
}

TEST(SpeciesBubbleProcessesTest, ImplicitFilmInflowsCannotFinanceGrossOutgoingDonors)
{
    Fixture f;
    f.conditions[1] = {"upper", 50, .125, {}};
    f.state.set_amount(f.catalog, 17, P::Dispersed, 2, "upper");
    f.state.set_amount(f.catalog, 17, P::Dissolved, .25);
    const auto coefficient = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition& p)
    { return 1 / (4 * std::numbers::pi * p.radius * p.radius * p.number); };
    const auto equilibrium = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition& p)
    { return p.id == "small" ? 0. : 2.; };
    const auto film = B::make_implicit_film_transfer_process(
        {{"small", {{17, coefficient, equilibrium}}}, {"upper", {{17, coefficient, equilibrium}}}});
    // The joint endpoint is .45 mol but its .9 mol outgoing transfer exceeds .25 mol of original dissolved donor.
    EXPECT_THROW((void) f.advance(film), std::runtime_error);
    EXPECT_DOUBLE_EQ(f.state.amount(f.catalog, 17, P::Dissolved), .25);
    EXPECT_DOUBLE_EQ(f.state.amount(f.catalog, 17, P::Dispersed, "upper"), 2);
}

TEST(SpeciesBubbleProcessesTest, ExactImplicitFilmDepletionRequiresExplicitNumberDeathOwner)
{
    Fixture f;
    f.state.set_amount(f.catalog, 91, P::Dispersed, 0, "small");
    f.state.set_amount(f.catalog, 17, P::Dissolved, 0);
    const auto coefficient = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition& p)
    { return 1 / (4 * std::numbers::pi * p.radius * p.radius * p.number); };
    const auto equilibrium = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition&) { return 1.; };
    const auto film = B::make_implicit_film_transfer_process({{"small", {{17, coefficient, equilibrium}}}});
    const auto packet = f.packet(film);
    ASSERT_EQ(packet.transfers.size(), 1);
    EXPECT_TRUE(packet.numberEvents.empty());
    const auto moles = packet.transfers[0].moles;
    EXPECT_NEAR(moles, 1. / 3, 1e-15);
    f.state.set_amount(f.catalog, 17, P::Dispersed, moles, "small");
    EXPECT_THROW((void) f.advance(film), std::runtime_error);
    const auto result = f.advance(B::compose_population_processes(
        {film, B::make_population_number_process("small", {}, constant(50))}));
    EXPECT_DOUBLE_EQ(result.state.amount(f.catalog, 17, P::Dissolved), moles);
    EXPECT_DOUBLE_EQ(result.numbers[0].number, 0);
    EXPECT_DOUBLE_EQ(total(f, result.state, 17), moles);
}

TEST(SpeciesBubbleProcessesTest, IncomingDissolutionCannotFinanceComposedNucleation)
{
    Fixture f;
    f.state.set_amount(f.catalog, 17, P::Dissolved, 0);
    const auto process = B::compose_population_processes({
        B::make_population_dissolution_process("small", constant(std::log(2.))),
        B::make_nucleation_process("upper", {{17, constant(.125)}}, constant(5))});
    EXPECT_THROW((void) f.advance(process), std::runtime_error);
    EXPECT_DOUBLE_EQ(f.state.amount(f.catalog, 17, P::Dispersed, "small"), 1);
    EXPECT_DOUBLE_EQ(f.state.amount(f.catalog, 17, P::Dissolved), 0);
}

TEST(SpeciesBubbleProcessesTest, CompetingDissolutionAndConversionCannotOverdrawOriginalCount)
{
    Fixture f;
    const auto process = B::compose_population_processes({
        B::make_population_dissolution_process("small", constant(std::log(4.))),
        B::make_population_conversion_process("small", "upper", constant(std::log(4.)))});
    EXPECT_THROW((void) f.advance(process), std::runtime_error);
    EXPECT_DOUBLE_EQ(f.state.amount(f.catalog, 91, P::Dispersed, "small"), 3);
}

TEST(SpeciesBubbleProcessesTest, ExactFilmDepletionRequiresExplicitNumberDeathOwner)
{
    Fixture f;
    f.state.set_amount(f.catalog, 91, P::Dispersed, 0, "small");
    f.state.set_amount(f.catalog, 17, P::Dissolved, 0);
    const auto coefficient = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition& p)
    { return 1 / (4 * std::numbers::pi * p.radius * p.radius * p.number); };
    const auto equilibrium = [](const B::MultispeciesPopulationContext&, const B::PopulationCondition&) { return 1.; };
    const auto film = B::make_film_transfer_process("small", {{17, coefficient, equilibrium}});
    EXPECT_THROW((void) f.advance(film), std::runtime_error);
    const auto result = f.advance(B::compose_population_processes(
        {film, B::make_population_number_process("small", {}, constant(50))}));
    EXPECT_DOUBLE_EQ(result.state.amount(f.catalog, 17, P::Dissolved), 1);
    EXPECT_DOUBLE_EQ(result.numbers[0].number, 0);
    EXPECT_DOUBLE_EQ(total(f, result.state, 17), 1);
}

TEST(SpeciesBubbleProcessesTest, InvalidSchemaPopulationRatesAndUnpairedBirthsReject)
{
    Fixture f;
    EXPECT_THROW((void) f.advance(B::make_nucleation_process("upper", {{999, constant(1)}}, constant(1))),
        std::invalid_argument);
    EXPECT_THROW((void) f.advance(B::make_nucleation_process("upper", {{2, constant(1)}}, constant(1))),
        std::invalid_argument);
    EXPECT_THROW((void) f.advance(B::make_population_dissolution_process("missing", constant(1))),
        std::invalid_argument);
    EXPECT_THROW((void) f.advance(B::make_nucleation_process("upper", {{17, constant(1)}}, constant(0))),
        std::invalid_argument);
    EXPECT_THROW((void) f.advance(B::make_population_conversion_process("small", "upper", constant(-1))),
        std::invalid_argument);
    EXPECT_THROW((void) f.advance(B::make_population_number_process(
                     "small", constant(std::numeric_limits<double>::quiet_NaN()))), std::invalid_argument);
    EXPECT_THROW((void) B::make_nucleation_process("small", {{17, constant(1)}, {17, constant(1)}}, constant(1)),
        std::invalid_argument);
    EXPECT_THROW((void) B::compose_population_processes({B::PopulationProcess{}}), std::invalid_argument);
}
} // namespace
