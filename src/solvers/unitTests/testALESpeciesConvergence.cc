#include "solvers/details/ALESpeciesConvergence.hh"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

namespace
{
namespace A = SimpleFluid::detail::ale_species;
using Event = SimpleFluid::Reacting::SpeciesCompartmentEvents;

Event cancelling_event(double scale = 1)
{
    Event event;
    event.chemical_production = 1e-5 * scale;
    event.interphase_outflow = 1e-5 * scale;
    return event;
}

A::ConvergenceEntry entry(double raw, const Event& event)
{
    return {raw, A::source_cancellation_allowance(event)};
}

TEST(ALESpeciesConvergenceTest, OpposingGrossOperandsBoundObservedNetCancellationWithoutEditingReceipt)
{
    const auto event = cancelling_event();
    const double before = 1.08526096366957231e-21, after = -2.27309232329572005e-21;
    const auto old_entry = entry(before, event), new_entry = entry(after, event);
    const auto expected = static_cast<long double>(std::numeric_limits<double>::epsilon()) *
                          (static_cast<long double>(1e-5) + 1e-5L);
    EXPECT_NEAR(static_cast<double>(old_entry.cancellation_allowance / expected), 1, 1e-15);
    EXPECT_DOUBLE_EQ(A::relative_change(old_entry, new_entry), 0);
    EXPECT_DOUBLE_EQ(old_entry.value, before);
    EXPECT_DOUBLE_EQ(new_entry.value, after);
}

TEST(ALESpeciesConvergenceTest, ResolvedNetVariationStillFailsTheRelativeGate)
{
    const auto event = cancelling_event();
    const auto before = entry(0, event);
    const auto after = entry(1e-14, event);
    const auto residual = A::relative_change(before, after);
    EXPECT_GT(residual, .999999);
    EXPECT_GT(residual, 1e-10);
    const double twice_combined_bounds = static_cast<double>(4 * before.cancellation_allowance);
    EXPECT_NEAR(A::relative_change(before, entry(twice_combined_bounds, event)), .5, 1e-15);
}

TEST(ALESpeciesConvergenceTest, UncancelledTinySourcesWithdrawalsAndStocksRetainPureRelativeComparison)
{
    Event production;
    production.chemical_production = 1e-21;
    Event withdrawal;
    withdrawal.external_withdrawal = 1e-21;
    for (const auto& event : {production, withdrawal, Event{}})
    {
        EXPECT_EQ(A::source_cancellation_allowance(event), 0);
        EXPECT_DOUBLE_EQ(A::relative_change(entry(1e-21, event), entry(2e-21, event)), .5);
    }
    // All material, number, heat, gross-event and subset-net entries use this zero-allowance constructor.
    const A::ConvergenceEntry inventory(1e-21), changed_inventory(2e-21);
    EXPECT_EQ(inventory.cancellation_allowance, 0);
    EXPECT_DOUBLE_EQ(A::relative_change(inventory, changed_inventory), .5);
    EXPECT_DOUBLE_EQ(A::relative_change(A::ConvergenceEntry(0), A::ConvergenceEntry(1e-300)), 1);
}

TEST(ALESpeciesConvergenceTest, OnlyMatchingSourceSignsContributeAndEscapeHasNoAllowance)
{
    Event event;
    event.chemical_production = 1;
    event.external_supply = 2;
    event.interphase_inflow = 4;
    event.chemical_consumption = 8;
    event.external_withdrawal = 16;
    event.interphase_outflow = 32;
    const auto expected = 63.L * std::numeric_limits<double>::epsilon();
    EXPECT_EQ(A::source_cancellation_allowance(event), expected);
    event.physical_outflow = 1e100;
    EXPECT_EQ(A::source_cancellation_allowance(event), expected);
    Event escape;
    escape.physical_outflow = 1e100;
    EXPECT_EQ(A::source_cancellation_allowance(escape), 0);
}

TEST(ALESpeciesConvergenceTest, DecisionsAreInvariantUnderSmallAndLargeUnitScaling)
{
    for (const auto scale : {1e-200, 1., 1e200})
    {
        const auto event = cancelling_event(scale);
        EXPECT_DOUBLE_EQ(A::relative_change(entry(1.08526096366957231e-21 * scale, event),
                             entry(-2.27309232329572005e-21 * scale, event)), 0);
        EXPECT_GT(A::relative_change(entry(0, event), entry(1e-14 * scale, event)), .999999);
        EXPECT_DOUBLE_EQ(A::relative_change(A::ConvergenceEntry(1e-21 * scale),
                             A::ConvergenceEntry(2e-21 * scale)), .5);
    }
}

TEST(ALESpeciesConvergenceTest, SoleSolutionPhaseInheritsOnlyItsExactSpeciesCompartmentBound)
{
    using Phase = SimpleFluid::Reacting::SpeciesPhase;
    const SimpleFluid::Reacting::SpeciesCatalog catalog{{
        {17, "dissolved-only gas", .020, {{"A", 1}}, {Phase::Dissolved, Phase::Dispersed}},
        {91, "both solution phases", .040, {{"B", 1}}, {Phase::Liquid, Phase::Dissolved, Phase::Dispersed}},
        {4, "liquid-only reactant", .018, {{"C", 1}}, {Phase::Liquid}},
        {5, "liquid-only cancelling species", .010, {{"D", 1}}, {Phase::Liquid}}}};
    auto dissolved = cancelling_event();
    dissolved.species = 17;
    dissolved.phase = Phase::Dissolved;
    auto unrelated = cancelling_event(1e100);
    unrelated.species = 91;
    unrelated.phase = Phase::Dissolved;
    auto internal_liquid = unrelated;
    internal_liquid.phase = Phase::Liquid;
    Event reactant;
    reactant.species = 4;
    reactant.phase = Phase::Liquid;
    reactant.chemical_consumption = 1e-4;
    auto liquid_only = cancelling_event();
    liquid_only.species = 5;
    liquid_only.phase = Phase::Liquid;
    auto gas_population = cancelling_event(1e100);
    gas_population.species = 17;
    gas_population.phase = Phase::Dispersed;
    gas_population.population = "upper";
    const std::vector<Event> events{unrelated, internal_liquid, reactant, dissolved, liquid_only, gas_population};
    EXPECT_EQ(A::solution_subset_cancellation_allowance(catalog, 17, events),
        A::source_cancellation_allowance(dissolved));
    EXPECT_EQ(A::solution_subset_cancellation_allowance(catalog, 91, events), 0);
    EXPECT_EQ(A::solution_subset_cancellation_allowance(catalog, 4, events), 0);
    EXPECT_EQ(A::solution_subset_cancellation_allowance(catalog, 5, events),
        A::source_cancellation_allowance(liquid_only));
    auto duplicates = events;
    duplicates.push_back(dissolved);
    EXPECT_THROW((void) A::solution_subset_cancellation_allowance(catalog, 17, duplicates), std::logic_error);
    EXPECT_EQ(A::solution_subset_cancellation_allowance(catalog, 17, std::span<const Event>{}), 0);
    // A two-phase solution subset must still reject its unrelated tiny net variation.
    EXPECT_DOUBLE_EQ(A::relative_change(
                         A::ConvergenceEntry(1e-21, A::solution_subset_cancellation_allowance(catalog, 91, events)),
                         A::ConvergenceEntry(2e-21, A::solution_subset_cancellation_allowance(catalog, 91, events))), .5);
}

TEST(ALESpeciesConvergenceTest, ZeroAllowancePreservesTheOriginalBinary64GateAtItsAcceptanceBoundary)
{
    constexpr double tolerance = 1e-10;
    for (const auto before : {1e-200, .125, 1., 3.141592653589793, 1e200})
        for (const auto candidate : {before * (1 - tolerance), before * (1 + tolerance)})
            for (const auto after : {std::nextafter(candidate, 0.), candidate,
                     std::nextafter(candidate, std::numeric_limits<double>::infinity())})
            {
                const double original = std::abs(after - before) / std::max(std::abs(before), std::abs(after));
                const auto observed = A::relative_change(A::ConvergenceEntry(before), A::ConvergenceEntry(after));
                EXPECT_DOUBLE_EQ(observed, original);
                EXPECT_EQ(observed <= tolerance, original <= tolerance);
            }
}

TEST(ALESpeciesConvergenceTest, InvalidOperandsAndEntriesRejectWithoutAnInfiniteAllowance)
{
    for (const auto value : {-1., std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()})
    {
        auto event = cancelling_event();
        event.external_supply = value;
        EXPECT_THROW((void) A::source_cancellation_allowance(event), std::runtime_error);
        EXPECT_THROW((void) A::relative_change(A::ConvergenceEntry(0), A::ConvergenceEntry(0, value)),
            std::runtime_error);
    }
    EXPECT_THROW((void) A::relative_change(A::ConvergenceEntry(0),
                     A::ConvergenceEntry(std::numeric_limits<double>::infinity())), std::runtime_error);
    auto large = cancelling_event();
    large.chemical_production = large.interphase_outflow = std::numeric_limits<double>::max();
    const auto allowance = A::source_cancellation_allowance(large);
    EXPECT_TRUE(std::isfinite(allowance));
    EXPECT_GT(A::relative_change(entry(std::numeric_limits<double>::max(), large),
                  entry(-std::numeric_limits<double>::max(), large)), 1.99);
}
} // namespace
