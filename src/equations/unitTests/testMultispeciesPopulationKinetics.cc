#include "equations/BubblePopulationKinetics.hh"
#include <gtest/gtest.h>
#include <limits>

namespace
{
namespace R = SimpleFluid::Reacting;
namespace B = SimpleFluid::BubbleGasPhysics;
using P = R::SpeciesPhase;
std::shared_ptr<const R::SpeciesCatalog> catalog()
{
    return std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{17, "gas A", .020, {{"A", 1}}, {P::Dissolved, P::Dispersed}},
            {91, "gas B", .040, {{"B", 1}}, {P::Dissolved, P::Dispersed}}});
}
R::CellSpeciesState initial(const R::SpeciesCatalog& schema)
{
    R::CellSpeciesState state(schema);
    state.dispersedPopulations = {{"micro", {1., 3.}}, {"large", {0., 0.}}};
    return state;
}
R::ReactionEnvironment environment()
{
    return {0., 300., 100000., .5, 1.};
}

TEST(MultispeciesPopulationKineticsTest, FullMixedGasDissolutionAndRebirthPreserveBothSpeciesAndCount)
{
    const auto schema = catalog();
    const auto state = initial(*schema);
    B::MultispeciesPopulationPacket dissolution;
    dissolution.transfers = {
        {17, P::Dispersed, P::Dissolved, "micro", {}, 1.}, {91, P::Dispersed, P::Dissolved, "micro", {}, 3.}};
    const auto dissolved = B::advance_multispecies_population_exchange(
        *schema, state, {{"micro", 100.}, {"large", 0.}}, environment(), dissolution);
    EXPECT_DOUBLE_EQ(dissolved.numbers[0].number, 0.);
    EXPECT_DOUBLE_EQ(dissolved.numberChanges[0], -100.);
    EXPECT_DOUBLE_EQ(dissolved.state.amount(*schema, 17, P::Dissolved), 1.);
    EXPECT_DOUBLE_EQ(dissolved.state.amount(*schema, 91, P::Dissolved), 3.);
    B::MultispeciesPopulationPacket birth;
    birth.transfers = {
        {17, P::Dissolved, P::Dispersed, {}, "large", 1.}, {91, P::Dissolved, P::Dispersed, {}, "large", 3.}};
    birth.numberEvents = {{"large", 50., 0.}};
    const auto reborn =
        B::advance_multispecies_population_exchange(*schema, dissolved.state, dissolved.numbers, environment(), birth);
    EXPECT_DOUBLE_EQ(reborn.numbers[1].number, 50.);
    EXPECT_DOUBLE_EQ(reborn.state.amount(*schema, 17, P::Dispersed, "large"), 1.);
    EXPECT_DOUBLE_EQ(reborn.state.amount(*schema, 91, P::Dispersed, "large"), 3.);
    EXPECT_DOUBLE_EQ(reborn.ledger.thermal_energy(), 0.);
}

TEST(MultispeciesPopulationKineticsTest, ConversionPreservesEverySpeciesWithoutAReservedGasSlot)
{
    const auto schema = catalog();
    B::MultispeciesPopulationPacket conversion;
    conversion.transfers = {{17, P::Dispersed, P::Dispersed, "micro", "large", .25},
        {91, P::Dispersed, P::Dispersed, "micro", "large", .75}};
    conversion.numberEvents = {{"micro", 0., 25.}, {"large", 25., 0.}};
    const auto result = B::advance_multispecies_population_exchange(
        *schema, initial(*schema), {{"micro", 100.}, {"large", 0.}}, environment(), conversion);
    EXPECT_DOUBLE_EQ(result.numbers[0].number, 75.);
    EXPECT_DOUBLE_EQ(result.numbers[1].number, 25.);
    EXPECT_DOUBLE_EQ(result.state.phase_amount(*schema, 17, P::Dispersed), 1.);
    EXPECT_DOUBLE_EQ(result.state.phase_amount(*schema, 91, P::Dispersed), 3.);
    EXPECT_DOUBLE_EQ(result.state.amount(*schema, 91, P::Dispersed, "large"), .75);
}

TEST(MultispeciesPopulationKineticsTest, CompetingBubblesJointlyReserveDissolvedAndNumberDonors)
{
    const auto schema = catalog();
    auto state = initial(*schema);
    state.set_amount(*schema, 17, P::Dissolved, 1.);
    B::MultispeciesPopulationPacket packet;
    packet.transfers = {
        {17, P::Dissolved, P::Dispersed, {}, "micro", .6}, {17, P::Dissolved, P::Dispersed, {}, "large", .6}};
    packet.numberEvents = {{"large", 10., 0.}};
    EXPECT_THROW((void) B::advance_multispecies_population_exchange(
                     *schema, state, {{"micro", 100.}, {"large", 0.}}, environment(), packet),
        std::runtime_error);
    packet.transfers.clear();
    packet.numberEvents = {{"micro", 0., 60.}, {"micro", 0., 60.}};
    EXPECT_THROW((void) B::advance_multispecies_population_exchange(
                     *schema, state, {{"micro", 100.}, {"large", 0.}}, environment(), packet),
        std::runtime_error);
    EXPECT_DOUBLE_EQ(state.amount(*schema, 17, P::Dissolved), 1.);
}

TEST(MultispeciesPopulationKineticsTest, NumberBirthWithoutGasAndNonfiniteTransferRejectAtomically)
{
    const auto schema = catalog();
    const auto state = initial(*schema);
    B::MultispeciesPopulationPacket packet;
    packet.numberEvents = {{"large", 1., 0.}};
    EXPECT_THROW((void) B::advance_multispecies_population_exchange(
                     *schema, state, {{"micro", 100.}, {"large", 0.}}, environment(), packet),
        std::runtime_error);
    packet.numberEvents.clear();
    packet.transfers = {{17, P::Dispersed, P::Dissolved, "micro", {}, std::numeric_limits<double>::quiet_NaN()}};
    EXPECT_THROW((void) B::advance_multispecies_population_exchange(
                     *schema, state, {{"micro", 100.}, {"large", 0.}}, environment(), packet),
        std::invalid_argument);
}
} // namespace
