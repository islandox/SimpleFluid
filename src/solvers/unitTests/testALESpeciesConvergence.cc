#include "solvers/details/ALESpeciesConvergence.hh"
#include "solvers/details/ALESpeciesSnapshot.hh"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <type_traits>

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

using DenseState = std::vector<A::ConvergenceEntry>;

auto visit_fixture(const DenseState& entries)
{
    return [&entries](auto& sink) {
        for (std::size_t index = 0; index < entries.size(); ++index)
            if (index == 109 || index == 117 || index == 119)
                sink.signed_net(entries[index].value, entries[index].cancellation_allowance);
            else
                sink.ordinary(entries[index].value);
    };
}

A::SnapshotMaximum dense_oracle(const DenseState& previous, const DenseState& current)
{
    A::SnapshotMaximum maximum;
    for (std::size_t index = 0; index < previous.size(); ++index)
    {
        const auto& before = previous[index];
        const auto& after = current[index];
        const double scale = std::max(std::abs(before.value), std::abs(after.value));
        double residual = 0;
        if (scale != 0 && !before.cancellation_allowance && !after.cancellation_allowance)
            residual = std::abs(after.value - before.value) / scale;
        else if (scale != 0)
        {
            const long double delta = std::abs(static_cast<long double>(after.value) - before.value);
            residual = static_cast<double>(
                std::max(0.L, delta - before.cancellation_allowance - after.cancellation_allowance) / scale);
        }
        if (residual > maximum.residual)
            maximum = {residual, index, before, after};
    }
    return maximum;
}

void check_maximum(const A::SnapshotMaximum& actual, const A::SnapshotMaximum& expected)
{
    EXPECT_EQ(actual.residual, expected.residual);
    EXPECT_EQ(actual.index, expected.index);
    EXPECT_EQ(actual.before.value, expected.before.value);
    EXPECT_EQ(actual.after.value, expected.after.value);
    EXPECT_EQ(actual.before.cancellation_allowance, expected.before.cancellation_allowance);
    EXPECT_EQ(actual.after.cancellation_allowance, expected.after.cancellation_allowance);
    EXPECT_EQ(actual.residual <= 1e-10, expected.residual <= 1e-10);
}

TEST(ALESpeciesConvergenceTest, StreamingMatchesDenseMaximaAndFirstTieAcrossCorrectors)
{
    const auto layout = A::SnapshotLayout::typed(1, 1, 0, 2, 1);
    ASSERT_EQ(layout.cell_values, 102U);
    ASSERT_EQ(layout.values, 120U);
    ASSERT_EQ(layout.allowances, 3U);
    DenseState previous(120);
    previous[4] = 1.0;
    previous[109] = {2e-12, 2e-13L};
    previous[119] = {1e-24, 1e-22L};
    auto reference = A::SnapshotReference::capture(layout, visit_fixture(previous));
    auto current = previous;
    const auto compare = [&] {
        A::SnapshotMaximum maximum;
        reference = A::SnapshotReference::stage(std::move(reference), layout, visit_fixture(current), maximum);
        check_maximum(maximum, dense_oracle(previous, current));
        previous = current;
        return maximum;
    };
    EXPECT_EQ(compare().residual, 0);
    current[5] = 1e-300;
    current[6] = -1e-300;
    EXPECT_EQ(compare().index, 5U);
    current[5] = current[6] = 0.0;
    EXPECT_EQ(compare().index, 5U);
    current[109] = {4e-12, 7e-13L};
    current[119] = {-1e-24, 1e-22L};
    EXPECT_EQ(compare().index, 109U);
    for (const auto after : {1.0 - 1e-10, std::nextafter(1.0 - 1e-10, 1.0)})
    {
        current[4] = after;
        compare();
        current[4] = 1.0;
        compare();
    }
    for (const auto equal : {std::size_t{4}, std::size_t{119}})
    {
        current[equal] = 1e10;
        compare();
        current[equal] = 1e10 - 1;
        EXPECT_EQ(compare().residual, 1e-10);
        current[equal] = 1e10;
        compare();
        current[equal] = std::nextafter(1e10 - 1, 0.);
        EXPECT_GT(compare().residual, 1e-10);
    }
}

TEST(ALESpeciesConvergenceTest, IncompleteOrInvalidStreamCannotPublishACandidate)
{
    const auto layout = A::SnapshotLayout::typed(1, 1, 0, 2, 1);
    const DenseState accepted(120, 1.0);
    A::SnapshotMaximum maximum{.residual = .25, .index = 31};
    const auto capture = [&] { return A::SnapshotReference::capture(layout, visit_fixture(accepted)); };
    auto reference = capture();
    const auto interrupted = [](auto& sink) { sink.ordinary(17.); throw std::runtime_error("interrupted"); };
    EXPECT_THROW(reference = A::SnapshotReference::stage(std::move(reference), layout, interrupted, maximum),
        std::runtime_error);
    EXPECT_TRUE(reference.empty());
    EXPECT_EQ(maximum.index, 31U);
    EXPECT_EQ(maximum.residual, .25);
    EXPECT_THROW(A::SnapshotReference::stage(capture(), layout, [](auto& sink) { sink.ordinary(4.); }, maximum),
        std::logic_error);
    EXPECT_THROW(A::SnapshotReference::stage(capture(), layout, [](auto& sink) { sink.signed_net(1., 0.L); }, maximum),
        std::logic_error);
    auto retry = capture();
    auto trial = accepted;
    trial.back() = 0.;
    retry = A::SnapshotReference::stage(std::move(retry), layout, visit_fixture(trial), maximum);
    check_maximum(maximum, dense_oracle(accepted, trial));
    EXPECT_THROW(A::SnapshotLayout::typed(std::numeric_limits<std::size_t>::max(), 1, 0, 0, 0), std::length_error);
}

TEST(ALESpeciesConvergenceTest, SnapshotSizesCountRankEventsOnceEvenWithoutOwnedCells)
{
    static_assert(!std::is_copy_constructible_v<A::SnapshotReference>);
    const auto populated = A::SnapshotLayout::typed(153458, 7, 2, 42, 7);
    EXPECT_EQ(populated.cell_values, 153458U * 402U);
    EXPECT_EQ(populated.values, populated.cell_values + 350);
    EXPECT_EQ(populated.allowances, 49U);

    const auto empty_rank = A::SnapshotLayout::typed(0, 7, 2, 42, 7);
    ASSERT_EQ(empty_rank.cell_values, 0U);
    ASSERT_EQ(empty_rank.values, 350U);
    ASSERT_EQ(empty_rank.allowances, 49U);
    const auto emit = [](auto& sink, double last_net) {
        for (size_t event = 0; event < 42; ++event)
        {
            for (size_t field = 0; field < 7; ++field) sink.ordinary(0.);
            sink.signed_net(0., 0.L);
        }
        for (size_t subset = 0; subset < 7; ++subset)
        {
            sink.ordinary(0.);
            sink.signed_net(subset == 6 ? last_net : 0., 0.L);
        }
    };
    auto ref = A::SnapshotReference::capture(empty_rank, [&](auto& sink) { emit(sink, 0.); });
    A::SnapshotMaximum maximum;
    ref = A::SnapshotReference::stage(std::move(ref), empty_rank, [&](auto& sink) { emit(sink, 1e-290); }, maximum);
    EXPECT_EQ(maximum.index, 349U);
    EXPECT_EQ(maximum.residual, 1.);
}

TEST(ALESpeciesConvergenceTest, LiquidOnlyOmissionsReduceStorageButKeepDenseEventOrdinals)
{
    const auto sized = A::SnapshotLayout::typed(153458, 7, 2, 42, 7, {6, 30, 6});
    EXPECT_EQ(sized.cell_values, 153458U * 192U);
    EXPECT_EQ(sized.values, sized.cell_values + 104);
    EXPECT_EQ(sized.allowances, 19U);
    EXPECT_EQ(sized.logical_cell_values, 153458U * 402U);
    EXPECT_THROW(A::SnapshotLayout::typed(1, 2, 1, 10, 2, {3, 4, 1}), std::logic_error);

    const auto layout = A::SnapshotLayout::typed(0, 2, 1, 10, 2, {1, 4, 1});
    ASSERT_EQ(layout.values, 51U);
    ASSERT_EQ(layout.allowances, 8U);
    const auto emit = [](auto& sink, double solution_net) {
        for (std::size_t event = 0; event < 10; ++event)
        {
            if (event >= 2 && event % 2 == 0)
                sink.skip(8);
            else
            {
                for (std::size_t field = 0; field < 7; ++field) sink.ordinary(0.);
                sink.signed_net(0., 0.L);
            }
        }
        sink.skip(1);
        sink.signed_net(solution_net, .125L);
        sink.ordinary(0.);
        sink.signed_net(solution_net, .125L);
    };
    auto ref = A::SnapshotReference::capture(layout, [&](auto& sink) { emit(sink, 0.); });
    A::SnapshotMaximum maximum;
    ref = A::SnapshotReference::stage(std::move(ref), layout, [&](auto& sink) { emit(sink, 1.); }, maximum);
    EXPECT_EQ(maximum.index, 81U);
    EXPECT_EQ(maximum.residual, .75);
    EXPECT_EQ(maximum.before.cancellation_allowance, .125L);
    EXPECT_EQ(maximum.after.cancellation_allowance, .125L);
}
} // namespace
