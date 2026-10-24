#include "equations/BubblePopulationKinetics.hh"
#include "utils/CompensatedSum.hh"
#include <gtest/gtest.h>
#include <limits>

namespace
{
namespace R = SimpleFluid::Reacting;
namespace B = SimpleFluid::BubbleGasPhysics;
constexpr R::SpeciesId gas = 101;
using P = R::SpeciesPhase;
struct PureFixture
{
    R::SpeciesCatalog catalog{{{gas, "ordinary gas", .002, {{"X", 1}}, {P::Dissolved, P::Dispersed}}}};
    R::CellSpeciesState state{catalog};
    R::ReactionEnvironment environment{0, 300, 100000, 1, 1};
    std::vector<B::PopulationNumberState> counts;
    explicit PureFixture(double count = 1e20, double moles = 1) : counts{{"bin", count}}
    {
        state.dispersedPopulations = {{"bin", std::vector<double>(1, moles)}};
        state.set_amount(catalog, gas, P::Dissolved, 1e-4);
    }
    auto apply(const B::MultispeciesPopulationPacket& packet) const
    {
        return B::advance_multispecies_population_exchange(catalog, state, counts, environment, packet);
    }
};
long double replay(double stock, const B::PopulationNumberChangeReceipt& raw)
{
    SimpleFluid::detail::CompensatedSum<long double> sum;
    sum += stock;
    sum += raw.high;
    sum += raw.correction;
    return sum.value();
}
TEST(PopulationRawCountsTest, OriginalOpposingHugeEventsRetainFiniteSignedChange)
{
    PureFixture f;
    const auto result = f.apply({{}, {{"bin", 1e20, 0}, {"bin", 0, 1e20}, {"bin", 1, 0}}});
    EXPECT_DOUBLE_EQ(result.numberChanges.front(), 1);
    ASSERT_EQ(result.localNumberChanges.size(), 1u);
    EXPECT_EQ(result.localNumberChanges.front().high + result.localNumberChanges.front().correction, 1);
    EXPECT_DOUBLE_EQ(result.numbers.front().number, 1e20);
    EXPECT_EQ(f.counts.front().number, 1e20);
}
TEST(PopulationRawCountsTest, GrossDeathMarginCannotBorrowIncomingBirthOrHideOne)
{
    PureFixture f;
    for (const bool small_first : {false, true})
    {
        B::MultispeciesPopulationPacket packet;
        packet.numberEvents = {{"bin", 1e20, 0}};
        packet.numberEvents.push_back({"bin", 0, small_first ? 1 : 1e20});
        packet.numberEvents.push_back({"bin", 0, small_first ? 1e20 : 1});
        EXPECT_THROW((void) f.apply(packet), std::runtime_error);
        EXPECT_DOUBLE_EQ(f.counts.front().number, 1e20);
        EXPECT_DOUBLE_EQ(f.state.amount(f.catalog, gas, P::Dispersed, "bin"), 1);
    }
}
TEST(PopulationRawCountsTest, OriginalStockCancelsBeforeRawSmallEndpointNarrows)
{
    PureFixture f;
    const auto result = f.apply({{}, {{"bin", 1, 1e20}}});
    EXPECT_DOUBLE_EQ(result.numbers.front().number, 1);
    EXPECT_DOUBLE_EQ(result.numberChanges.front(), -1e20); // Reporting cannot be replayed.
    EXPECT_EQ(replay(1e20, result.localNumberChanges.front()), 1);
}
TEST(PopulationRawCountsTest, CompleteMaterialReplacementRetainsRepresentableTinyGasAndCount)
{
    PureFixture f(2e8, 3e-6);
    // Independent continuous production/decay oracle at tau1e-300, q1e5,
    // external coefficient2e-7: retained gas = coefficient*q*tau.
    constexpr double tiny_gas = 2e-302;
    constexpr double mol_per_bubble = 2e-18;
    const double tiny_count = tiny_gas / mol_per_bubble;
    B::MultispeciesPopulationPacket packet;
    packet.transfers = {
        {gas, P::Dispersed, P::Dissolved, "bin", {}, 3e-6}, {gas, P::Dissolved, P::Dispersed, {}, "bin", tiny_gas}};
    packet.numberEvents = {{"bin", tiny_count, 2e8}};
    const auto result = f.apply(packet);
    EXPECT_DOUBLE_EQ(result.state.amount(f.catalog, gas, P::Dispersed, "bin"), tiny_gas);
    EXPECT_DOUBLE_EQ(result.numbers.front().number, tiny_count);
    EXPECT_EQ(replay(2e8, result.localNumberChanges.front()), static_cast<long double>(tiny_count));
    EXPECT_GT(result.numbers.front().number, 0);
}
TEST(PopulationRawCountsTest, UnrepresentablePositiveCountCannotEraseAnOccupiedGasPhase)
{
    PureFixture f(1, 1e-30);
    const long double desired = static_cast<long double>(std::numeric_limits<double>::denorm_min()) / 2;
    ASSERT_GT(desired, 0);
    ASSERT_EQ(static_cast<double>(desired), 0);
    EXPECT_THROW((void) f.apply({{}, {{"bin", static_cast<double>(desired), 1}}}), std::runtime_error);
    EXPECT_DOUBLE_EQ(f.counts.front().number, 1);
    EXPECT_DOUBLE_EQ(f.state.amount(f.catalog, gas, P::Dispersed, "bin"), 1e-30);
}
TEST(PopulationRawCountsTest, FiniteEventInputsStillRejectOverflowingEndpointAtomically)
{
    PureFixture f;
    const double largest = std::numeric_limits<double>::max();
    EXPECT_THROW((void) f.apply({{}, {{"bin", largest, 0}, {"bin", largest, 0}}}), std::runtime_error);
    EXPECT_DOUBLE_EQ(f.counts.front().number, 1e20);
}
TEST(PopulationRawCountsTest, NonfiniteOrNegativeOriginalEventsRejectBeforePublication)
{
    PureFixture f;
    for (const double invalid :
        {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_THROW((void) f.apply({{}, {{"bin", invalid, 0}}}), std::invalid_argument);
        EXPECT_THROW((void) f.apply({{}, {{"bin", 0, invalid}}}), std::invalid_argument);
    }
    EXPECT_DOUBLE_EQ(f.counts.front().number, 1e20);
}
TEST(PopulationRawCountsTest, CompleteDissolutionNeedsItsExplicitCountDeath)
{
    PureFixture f(7, 1);
    B::MultispeciesPopulationPacket packet;
    packet.transfers = {{gas, P::Dispersed, P::Dissolved, "bin", {}, 1}};
    EXPECT_THROW((void) f.apply(packet), std::runtime_error);
    EXPECT_DOUBLE_EQ(f.counts.front().number, 7);
    EXPECT_DOUBLE_EQ(f.state.amount(f.catalog, gas, P::Dispersed, "bin"), 1);
    packet.numberEvents = {{"bin", 0, 7}};
    const auto result = f.apply(packet);
    EXPECT_DOUBLE_EQ(result.state.amount(f.catalog, gas, P::Dispersed, "bin"), 0);
    EXPECT_DOUBLE_EQ(result.state.amount(f.catalog, gas, P::Dissolved), 1.0001);
    EXPECT_DOUBLE_EQ(result.numbers.front().number, 0);
    EXPECT_EQ(replay(7, result.localNumberChanges.front()), 0);
}
TEST(PopulationRawCountsTest, EarlierChemicalDisappearanceNeverManufacturesADeathEvent)
{
    PureFixture f(3, 0);
    EXPECT_THROW((void) f.apply({}), std::runtime_error);
    EXPECT_DOUBLE_EQ(f.counts.front().number, 3);
    // A separately supplied count owner can explicitly close that prior
    // material event. Absence alone never decides what happened to the count.
    const auto result = f.apply({{}, {{"bin", 0, 3}}});
    EXPECT_DOUBLE_EQ(result.numbers.front().number, 0);
    EXPECT_EQ(replay(3, result.localNumberChanges.front()), 0);
}
} // namespace
