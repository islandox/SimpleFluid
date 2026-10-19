/** @file testRadiolyticBubbleKinetics.cc
 * @brief Analytic moments, legacy micro splitting and rejection for pure population reactions.
 */
#include "equations/RadiolyticBubbleKinetics.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
namespace P = SimpleFluid::RadiolyticGasPhysics;

P::MicrobubbleSubstep micro_step(double dt)
{
    P::MicrobubbleSubstep step;
    step.duration = dt;
    step.bulk_production_rate = 0.3;
    step.liquid_fraction = 0.4;
    step.nucleation_moles = 0.25;
    step.critical_concentration = 1e6;
    step.micro_decay_fraction = 1.0 - std::exp(-dt);
    step.micro_retention = 1.0 - step.micro_decay_fraction;
    return step;
}

P::BubblePopulationKernels manufactured_kernels()
{
    P::BubblePopulationKernels kernels;
    kernels.model_name = "analytical verification fixture";
    kernels.validity_description = "Prescribed toy rates on a positive aqueous fraction; no physical calibration.";
    kernels.validate = [](const P::BubblePopulationContext& context)
    {
        if (context.environment.liquid_fraction <= 0.0)
            throw std::domain_error("This verification fixture requires aqueous liquid.");
    };
    return kernels;
}

void expect_closed(const P::BubblePopulationStepResult& result, double total)
{
    EXPECT_NEAR(result.hydrogen_after, total, 2e-13 * std::max(total, 1e-300));
    EXPECT_NEAR(result.hydrogen_error, 0.0, 2e-13 * std::max(total, 1e-300));
    EXPECT_NEAR(result.number_error, 0.0, 2e-13);
    for (const auto number : result.state.number_density)
        EXPECT_GE(number, 0.0);
    EXPECT_GE(result.state.dissolved_inventory, 0.0);
}

TEST(RadiolyticBubbleKineticsTest, MicroSourceBeforeDecayMatchesTheLegacyGeometricRecurrence)
{
    constexpr double dt = 0.2;
    constexpr int count = 31;
    auto step = micro_step(dt);
    const P::MicrobubbleState initial{0.1, 4.0, 1.0};
    auto state = initial;
    for (int i = 0; i < count; ++i)
    {
        const auto result = P::advance_microbubble_substep(state, step);
        EXPECT_DOUBLE_EQ(result.converted.number_density, 0.0);
        EXPECT_DOUBLE_EQ(result.converted.molar_inventory, 0.0);
        EXPECT_NEAR(result.number_error, 0.0, 2e-14);
        state = result.state;
    }
    const double retention = std::exp(-dt), retained_initial = std::pow(retention, count);
    const double discrete_source =
        step.bulk_production_rate * dt * retention * (1.0 - retained_initial) / (1.0 - retention);
    EXPECT_NEAR(state.molar_inventory, initial.molar_inventory * retained_initial + discrete_source, 2e-14);
    EXPECT_NEAR(state.number_density,
        initial.number_density * retained_initial + discrete_source / step.nucleation_moles, 8e-14);
    EXPECT_NEAR(state.dissolved_inventory + state.molar_inventory,
        initial.dissolved_inventory + initial.molar_inventory + count * step.bulk_production_rate * dt, 2e-14);
    // The unchanged legacy split differs from the continuous-source solution.
    const double continuous =
        initial.molar_inventory * retained_initial + step.bulk_production_rate * (1.0 - retained_initial);
    EXPECT_GT(std::abs(state.molar_inventory - continuous), 0.02);
}

TEST(RadiolyticBubbleKineticsTest, MicroProductionIsBulkBasedAndDecayReturnsAllLostMoles)
{
    const P::MicrobubbleState initial{0.25, 2.0, 0.5};
    auto step = micro_step(0.125);
    step.liquid_fraction = 0.2;
    const auto lower_liquid = P::advance_microbubble_substep(initial, step);
    step.liquid_fraction = 0.8;
    const auto higher_liquid = P::advance_microbubble_substep(initial, step);
    EXPECT_DOUBLE_EQ(lower_liquid.produced_moles, 0.3 * 0.125);
    EXPECT_DOUBLE_EQ(lower_liquid.state.molar_inventory, higher_liquid.state.molar_inventory);
    EXPECT_DOUBLE_EQ(lower_liquid.state.number_density, higher_liquid.state.number_density);
    EXPECT_DOUBLE_EQ(lower_liquid.state.dissolved_inventory, higher_liquid.state.dissolved_inventory);
    const double before_decay = initial.molar_inventory + step.bulk_production_rate * step.duration;
    EXPECT_DOUBLE_EQ(lower_liquid.dissolved_moles, before_decay * step.micro_decay_fraction);
    EXPECT_NEAR(lower_liquid.state.dissolved_inventory + lower_liquid.state.molar_inventory,
        initial.dissolved_inventory + initial.molar_inventory + lower_liquid.produced_moles, 2e-15);
}

TEST(RadiolyticBubbleKineticsTest, MicroThresholdAndConversionPreserveTheLegacyNumberAndMolarAmounts)
{
    auto step = micro_step(0.25);
    step.bulk_production_rate = 0.0;
    step.micro_decay_fraction = 0.0;
    step.micro_retention = 1.0;
    step.liquid_fraction = 0.5;
    step.critical_concentration = 4.0;
    step.nucleation_moles = 0.1;
    step.conversion_pressure_factor = 2.0;
    const auto critical = P::advance_microbubble_substep({2.0, 10.0, 2.0}, step);
    EXPECT_DOUBLE_EQ(critical.converted.number_density, 0.0);
    const auto active = P::advance_microbubble_substep({3.0, 10.0, 2.0}, step);
    const double fraction = 1.0 - std::exp(-0.25);
    EXPECT_DOUBLE_EQ(active.converted.number_density, 10.0 * fraction);
    EXPECT_DOUBLE_EQ(active.converted.molar_inventory, 0.1 * (10.0 * fraction));
    EXPECT_DOUBLE_EQ(active.state.dissolved_inventory, 3.0);
    EXPECT_NEAR(active.state.number_density + active.converted.number_density, 10.0, 1e-14);
    EXPECT_NEAR(active.number_error, 0.0, 1e-14);
    EXPECT_NEAR(active.state.molar_inventory + active.converted.molar_inventory, 2.0, 1e-14);
    // Preserve the historical min(micro moles, nucleation moles*converted N).
    const auto limited = P::advance_microbubble_substep({3.0, 10.0, 0.01}, step);
    EXPECT_DOUBLE_EQ(limited.converted.molar_inventory, 0.01);
    EXPECT_DOUBLE_EQ(limited.state.molar_inventory, 0.0);
    EXPECT_DOUBLE_EQ(limited.converted.number_density, active.converted.number_density);
    step.smooth_heaviside_width = -1.0; // Unused in exact-threshold legacy mode.
    EXPECT_NO_THROW((void) P::advance_microbubble_substep({3.0, 10.0, 2.0}, step));
}

TEST(RadiolyticBubbleKineticsTest, MicroInputsRejectInvalidPhysicalCoefficients)
{
    const P::MicrobubbleState state{0.1, 2.0, 0.5};
    auto step = micro_step(0.1);
    step.liquid_fraction = 0.0;
    EXPECT_THROW((void) P::advance_microbubble_substep(state, step), std::invalid_argument);
    step = micro_step(0.1);
    step.micro_retention = 0.3;
    EXPECT_THROW((void) P::advance_microbubble_substep(state, step), std::invalid_argument);
    step = micro_step(0.1);
    step.bulk_production_rate = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) P::advance_microbubble_substep(state, step), std::invalid_argument);
}

TEST(RadiolyticBubbleKineticsTest, FixedPivotBirthsPreserveNumberAndMolesOnANonuniformGrid)
{
    const P::MolarPivotGrid grid{{1.0, 3.0, 8.0}};
    const P::UpperBubblePopulation initial{5.0, {0.0, 0.0, 0.0}};
    const auto result = P::advance_upper_bubble_population(grid, initial, {}, 1.0, {}, {{2.0, 4.0}});
    EXPECT_EQ(result.state.number_density, (std::vector<double>{1.0, 1.0, 0.0}));
    EXPECT_EQ(P::population_bin_moles(grid, result.state), (std::vector<double>{1.0, 3.0, 0.0}));
    const auto moments = P::population_moments(grid, result.state);
    EXPECT_DOUBLE_EQ(moments.number_density, 2.0);
    EXPECT_DOUBLE_EQ(moments.molar_inventory, 4.0);
    EXPECT_DOUBLE_EQ(result.hydrogen_added_by_births, 4.0);
    expect_closed(result, 9.0);
}

TEST(RadiolyticBubbleKineticsTest, EmptyKernelsAreIdentityAndTinyMolarBirthsRemainPositive)
{
    const P::MolarPivotGrid grid{{1e-18, 2e-18, 4e-18}};
    const P::UpperBubblePopulation initial{1e-18, {0.0, 0.0, 0.0}};
    const auto unchanged = P::advance_upper_bubble_population(grid, initial, {}, 1.0);
    EXPECT_EQ(unchanged.state.number_density, initial.number_density);
    EXPECT_DOUBLE_EQ(unchanged.state.dissolved_inventory, initial.dissolved_inventory);
    const auto born = P::advance_upper_bubble_population(grid, initial, {}, 1.0, {}, {{2.0, 3e-18}});
    EXPECT_NEAR(P::population_moments(grid, born.state).molar_inventory, 3e-18, 1e-32);
    EXPECT_GT(P::population_moments(grid, born.state).molar_inventory, 0.0);
    expect_closed(born, 4e-18);
    const P::UpperBubblePopulation populated{1e-18, {1.3, 0.9, 3.7}};
    auto kernels = manufactured_kernels();
    kernels.growth_rate = [](std::size_t, const auto&) { return 0.0; };
    const auto equilibrium = P::advance_upper_bubble_population(grid, populated, {}, 1.0, kernels);
    EXPECT_EQ(equilibrium.state.number_density, populated.number_density);
    EXPECT_DOUBLE_EQ(equilibrium.state.dissolved_inventory, populated.dissolved_inventory);
}

TEST(RadiolyticBubbleKineticsTest, GrowthAndPartialDissolutionMoveSizeWithoutChangingNumber)
{
    const P::MolarPivotGrid grid{{1.0, 2.0, 4.0, 8.0}};
    const P::UpperBubblePopulation initial{10.0, {2.0, 0.0, 0.0, 0.0}};
    auto kernels = manufactured_kernels();
    kernels.growth_rate = [](std::size_t, const auto&) { return 1.0; };
    const auto grown = P::advance_upper_bubble_population(grid, initial, {}, 1.0, kernels);
    EXPECT_EQ(grown.state.number_density, (std::vector<double>{0.0, 2.0, 0.0, 0.0}));
    EXPECT_DOUBLE_EQ(grown.state.dissolved_inventory, 8.0);
    EXPECT_DOUBLE_EQ(grown.dissolved_to_gas_moles, 2.0);
    expect_closed(grown, 12.0);
    kernels.growth_rate = [](std::size_t, const auto&) { return -0.5; };
    const auto dissolved = P::advance_upper_bubble_population(grid, grown.state, {}, 1.0, kernels);
    EXPECT_EQ(dissolved.state.number_density, (std::vector<double>{1.0, 1.0, 0.0, 0.0}));
    EXPECT_DOUBLE_EQ(dissolved.state.dissolved_inventory, 9.0);
    EXPECT_DOUBLE_EQ(dissolved.gas_to_dissolved_moles, 1.0);
    expect_closed(dissolved, 12.0);
}

TEST(RadiolyticBubbleKineticsTest, CompleteDissolutionReturnsMolesAndRemovesThePhysicalBubbleCount)
{
    const P::MolarPivotGrid grid{{1.0, 2.0, 4.0}};
    auto kernels = manufactured_kernels();
    kernels.growth_rate = [](std::size_t, const auto&) { return -2.0; };
    const auto result = P::advance_upper_bubble_population(grid, {0.0, {0.0, 3.0, 0.0}}, {}, 1.0, kernels);
    EXPECT_EQ(result.state.number_density, (std::vector<double>{0.0, 0.0, 0.0}));
    EXPECT_DOUBLE_EQ(result.state.dissolved_inventory, 6.0);
    EXPECT_DOUBLE_EQ(result.dissolved_bubble_number, 3.0);
    expect_closed(result, 6.0);
}

TEST(RadiolyticBubbleKineticsTest, SimultaneousGrowthAndDissolutionShareOnlyTheAvailableMoles)
{
    const P::MolarPivotGrid grid{{1.0, 2.0, 4.0}};
    const P::UpperBubblePopulation initial{0.0, {2.0, 0.0, 1.0}};
    auto kernels = manufactured_kernels();
    kernels.growth_rate = [](std::size_t i, const auto&) { return i == 0 ? 1.0 : -2.0; };
    const auto result = P::advance_upper_bubble_population(grid, initial, {}, 1.0, kernels);
    EXPECT_EQ(result.state.number_density, (std::vector<double>{0.0, 3.0, 0.0}));
    EXPECT_DOUBLE_EQ(result.state.dissolved_inventory, 0.0);
    EXPECT_DOUBLE_EQ(result.dissolved_to_gas_moles, 2.0);
    EXPECT_DOUBLE_EQ(result.gas_to_dissolved_moles, 2.0);
    expect_closed(result, 6.0);
}

TEST(RadiolyticBubbleKineticsTest, CoalescenceCountsIdenticalPairsOnceAndConservesMoles)
{
    const P::MolarPivotGrid grid{{1.0, 2.0, 3.0, 4.0}};
    auto kernels = manufactured_kernels();
    kernels.coalescence_kernel = [](std::size_t i, std::size_t j, const auto&)
    { return i == 0 && j == 1 ? 0.25 : 0.0; };
    const auto cross = P::advance_upper_bubble_population(grid, {0.0, {2.0, 1.0, 0.0, 0.0}}, {}, 1.0, kernels);
    EXPECT_EQ(cross.state.number_density, (std::vector<double>{1.5, 0.5, 0.5, 0.0}));
    EXPECT_DOUBLE_EQ(cross.coalescence_events, 0.5);
    EXPECT_DOUBLE_EQ(P::population_moments(grid, cross.state).number_density, 2.5);
    expect_closed(cross, 4.0);
    P::BubblePopulationEnvironment half_carrier;
    half_carrier.liquid_fraction = half_carrier.carrier_fraction = 0.5;
    const auto same_host_density =
        P::advance_upper_bubble_population(grid, {0.0, {1.0, 0.5, 0.0, 0.0}}, half_carrier, 1.0, kernels);
    EXPECT_DOUBLE_EQ(same_host_density.coalescence_events, 0.5 * cross.coalescence_events);
    expect_closed(same_host_density, 2.0);
    const auto same_bulk_density =
        P::advance_upper_bubble_population(grid, {0.0, {2.0, 1.0, 0.0, 0.0}}, half_carrier, 1.0, kernels);
    EXPECT_DOUBLE_EQ(same_bulk_density.coalescence_events, 2.0 * cross.coalescence_events);
    expect_closed(same_bulk_density, 4.0);
    kernels.coalescence_kernel = [](std::size_t i, std::size_t j, const auto&) { return i == 0 && j == 0 ? 0.5 : 0.0; };
    const auto identical = P::advance_upper_bubble_population(grid, {0.0, {2.0, 0.0, 0.0, 0.0}}, {}, 0.25, kernels);
    EXPECT_EQ(identical.state.number_density, (std::vector<double>{1.5, 0.25, 0.0, 0.0}));
    EXPECT_DOUBLE_EQ(identical.coalescence_events, 0.25);
    expect_closed(identical, 2.0);
}

TEST(RadiolyticBubbleKineticsTest, BreakupRebinsDaughtersAndChecksBothPhysicalMoments)
{
    const P::MolarPivotGrid grid{{1.0, 2.0, 4.0, 8.0}};
    auto kernels = manufactured_kernels();
    kernels.breakup_frequency = [](std::size_t i, const auto&) { return i == 2 ? 0.5 : 0.0; };
    kernels.breakup_daughters = [](std::size_t, const auto&)
    { return std::vector<P::BubbleBreakupDaughter>{{1.0, 1.0}, {3.0, 1.0}}; };
    const auto result = P::advance_upper_bubble_population(grid, {0.0, {0.0, 0.0, 1.0, 0.0}}, {}, 1.0, kernels);
    EXPECT_EQ(result.state.number_density, (std::vector<double>{0.5, 0.25, 0.75, 0.0}));
    EXPECT_DOUBLE_EQ(result.breakup_events, 0.5);
    EXPECT_DOUBLE_EQ(result.breakup_created_number, 0.5);
    expect_closed(result, 4.0);
    kernels.breakup_daughters = [](std::size_t, const auto&)
    { return std::vector<P::BubbleBreakupDaughter>{{3.0, 2.0}}; };
    EXPECT_THROW((void) P::advance_upper_bubble_population(grid, {0.0, {0.0, 0.0, 1.0, 0.0}}, {}, 1.0, kernels),
        std::runtime_error);
}

TEST(RadiolyticBubbleKineticsTest, BelowGridAmountsRequireAnExplicitConservativeMicroReturn)
{
    const P::MolarPivotGrid grid{{1.0, 2.0, 4.0}};
    const P::UpperBubblePopulation initial{0.0, {2.0, 0.0, 0.0}};
    auto kernels = manufactured_kernels();
    kernels.growth_rate = [](std::size_t, const auto&) { return -0.25; };
    EXPECT_THROW((void) P::advance_upper_bubble_population(grid, initial, {}, 1.0, kernels), std::out_of_range);
    const P::BubblePopulationStepOptions options{P::BubbleUnderflowPolicy::ReturnToMicro};
    const auto result = P::advance_upper_bubble_population(grid, initial, {}, 1.0, kernels, {}, options);
    EXPECT_DOUBLE_EQ(result.returned_to_micro.number_density, 2.0);
    EXPECT_DOUBLE_EQ(result.returned_to_micro.molar_inventory, 1.5);
    EXPECT_DOUBLE_EQ(result.state.dissolved_inventory, 0.5);
    EXPECT_EQ(result.state.number_density, (std::vector<double>{0.0, 0.0, 0.0}));
    expect_closed(result, 2.0);
    kernels = manufactured_kernels();
    kernels.breakup_frequency = [](std::size_t, const auto&) { return 0.5; };
    kernels.breakup_daughters = [](std::size_t i, const auto& context)
    { return std::vector<P::BubbleBreakupDaughter>{{context.grid.mol_per_bubble[i] / 2.0, 2.0}}; };
    const auto broken = P::advance_upper_bubble_population(grid, initial, {}, 1.0, kernels, {}, options);
    EXPECT_DOUBLE_EQ(broken.returned_to_micro.number_density, 2.0);
    EXPECT_DOUBLE_EQ(broken.returned_to_micro.molar_inventory, 1.0);
    EXPECT_DOUBLE_EQ(broken.breakup_created_number, 1.0);
    expect_closed(broken, 2.0);
}

TEST(RadiolyticBubbleKineticsTest, ExhaustedDonorsAndAboveGridEventsRejectWithoutClippingInput)
{
    const P::MolarPivotGrid grid{{1.0, 2.0, 4.0}};
    const P::UpperBubblePopulation initial{0.0, {2.0, 0.0, 0.0}};
    auto kernels = manufactured_kernels();
    kernels.growth_rate = [](std::size_t, const auto&) { return 1.0; };
    EXPECT_THROW((void) P::advance_upper_bubble_population(grid, initial, {}, 1.0, kernels), std::runtime_error);
    kernels.growth_rate = [](std::size_t, const auto&) { return -2.0; };
    EXPECT_THROW((void) P::advance_upper_bubble_population(grid, initial, {}, 1.0, kernels), std::runtime_error);
    kernels.growth_rate = [](std::size_t, const auto&) { return 1.0; };
    EXPECT_THROW(
        (void) P::advance_upper_bubble_population(grid, {10.0, {0.0, 0.0, 1.0}}, {}, 1.0, kernels), std::out_of_range);
    kernels = manufactured_kernels();
    kernels.coalescence_kernel = [](std::size_t, std::size_t, const auto&) { return 2.0; };
    EXPECT_THROW((void) P::advance_upper_bubble_population(grid, initial, {}, 1.0, kernels), std::runtime_error);
    kernels = manufactured_kernels();
    kernels.breakup_frequency = [](std::size_t, const auto&) { return 2.0; };
    kernels.breakup_daughters = [](std::size_t, const auto&)
    { return std::vector<P::BubbleBreakupDaughter>{{0.5, 2.0}}; };
    EXPECT_THROW((void) P::advance_upper_bubble_population(grid, initial, {}, 1.0, kernels), std::runtime_error);
    EXPECT_THROW(
        (void) P::advance_upper_bubble_population(grid, initial, {}, 1.0, {}, {{1.0, 5.0}}), std::out_of_range);
    kernels = manufactured_kernels();
    kernels.coalescence_kernel = [](std::size_t, std::size_t, const auto&) { return 0.1; };
    EXPECT_THROW(
        (void) P::advance_upper_bubble_population(grid, {0.0, {0.0, 0.0, 1.0}}, {}, 1.0, kernels), std::out_of_range);
    EXPECT_EQ(initial.number_density, (std::vector<double>{2.0, 0.0, 0.0}));
    EXPECT_DOUBLE_EQ(initial.dissolved_inventory, 0.0);
}

TEST(RadiolyticBubbleKineticsTest, AqueousConcentrationAndSuppliedValidityControlTheGrowthLaw)
{
    const P::MolarPivotGrid grid{{1.0, 2.0, 4.0}};
    auto kernels = manufactured_kernels();
    kernels.growth_rate = [](std::size_t, const auto& context)
    { return 0.25 * (context.dissolved_concentration - 1.0); };
    P::BubblePopulationEnvironment environment;
    environment.liquid_fraction = 0.5;
    const auto first = P::advance_upper_bubble_population(grid, {1.0, {2.0, 0.0, 0.0}}, environment, 1.0, kernels);
    EXPECT_EQ(first.state.number_density, (std::vector<double>{1.5, 0.5, 0.0}));
    EXPECT_DOUBLE_EQ(first.state.dissolved_inventory, 0.5);
    expect_closed(first, 3.0);
    const auto equilibrium = P::advance_upper_bubble_population(grid, first.state, environment, 1.0, kernels);
    EXPECT_EQ(equilibrium.state.number_density, first.state.number_density);
    EXPECT_DOUBLE_EQ(equilibrium.state.dissolved_inventory, first.state.dissolved_inventory);
    environment.liquid_fraction = 0.0;
    EXPECT_THROW((void) P::advance_upper_bubble_population(grid, {0.0, {2.0, 0.0, 0.0}}, environment, 1.0, kernels),
        std::domain_error);
}

TEST(RadiolyticBubbleKineticsTest, InvalidGridsStatesAndUndeclaredKernelsAreRejected)
{
    const P::UpperBubblePopulation initial{0.0, {1.0, 0.0}};
    EXPECT_THROW((void) P::population_moments({{1.0, 1.0}}, initial), std::invalid_argument);
    EXPECT_THROW((void) P::population_moments({{2.0, 1.0}}, initial), std::invalid_argument);
    EXPECT_THROW((void) P::population_moments({{0.0, 1.0}}, initial), std::invalid_argument);
    EXPECT_THROW((void) P::population_moments({{1.0, 2.0}}, {0.0, {-1.0, 0.0}}), std::invalid_argument);
    EXPECT_THROW((void) P::advance_upper_bubble_population({{1.0, 2.0}}, initial, {}, 1.0, {}, {{0.0, 1.0}}),
        std::invalid_argument);
    P::BubblePopulationKernels undeclared;
    undeclared.growth_rate = [](std::size_t, const auto&) { return 0.0; };
    EXPECT_THROW(
        (void) P::advance_upper_bubble_population({{1.0, 2.0}}, initial, {}, 1.0, undeclared), std::invalid_argument);
    auto invalid = manufactured_kernels();
    invalid.growth_rate = [](std::size_t, const auto&) { return std::numeric_limits<double>::quiet_NaN(); };
    EXPECT_THROW(
        (void) P::advance_upper_bubble_population({{1.0, 2.0}}, initial, {}, 1.0, invalid), std::invalid_argument);
}

TEST(RadiolyticBubbleKineticsTest, TinyMicroEventsRetainTheirSignedIncrementWhenTheLargeStateRounds)
{
    const P::MicrobubbleState initial{1e20, 1e20, 1e20};
    auto step = micro_step(1.0);
    step.nucleation_moles = 1.0;
    step.critical_concentration = 1e30;
    step.bulk_production_rate = 1.0;
    step.micro_decay_fraction = 0.0;
    step.micro_retention = 1.0;
    const auto produced = P::advance_microbubble_substep(initial, step);
    EXPECT_DOUBLE_EQ(produced.state.number_density, initial.number_density);
    EXPECT_DOUBLE_EQ(produced.state.molar_inventory, initial.molar_inventory);
    EXPECT_DOUBLE_EQ(produced.number_change, 1.0);
    EXPECT_DOUBLE_EQ(produced.molar_change, 1.0);
    step.bulk_production_rate = 0.0;
    step.micro_decay_fraction = 1e-20;
    step.micro_retention = 1.0 - step.micro_decay_fraction;
    const auto decayed = P::advance_microbubble_substep(initial, step);
    EXPECT_DOUBLE_EQ(decayed.state.molar_inventory, initial.molar_inventory);
    EXPECT_NEAR(decayed.number_change, -1.0, 2e-15);
    EXPECT_NEAR(decayed.molar_change, -1.0, 2e-15);
    EXPECT_DOUBLE_EQ(decayed.molar_change + decayed.dissolved_moles, 0.0);
}

TEST(RadiolyticBubbleKineticsTest, TinyBirthAndGrowthLedgersRemainConservativeBelowEndpointPrecision)
{
    const P::MolarPivotGrid grid{{1.0, 2.0, 4.0}};
    const P::UpperBubblePopulation initial{1e20, {1e20, 0.0, 0.0}};
    const auto identity = P::advance_upper_bubble_population(grid, initial, {}, 1.0);
    EXPECT_EQ(identity.number_change, (std::vector<double>{0.0, 0.0, 0.0}));
    EXPECT_EQ(identity.hydrogen_change, (std::vector<double>{0.0, 0.0, 0.0}));
    const auto birth = P::advance_upper_bubble_population(grid, initial, {}, 1.0, {}, {{1.0, 1.0}});
    EXPECT_DOUBLE_EQ(birth.state.number_density[0], initial.number_density[0]);
    EXPECT_EQ(birth.number_change, (std::vector<double>{1.0, 0.0, 0.0}));
    EXPECT_EQ(birth.hydrogen_change, (std::vector<double>{1.0, 0.0, 0.0}));
    auto kernels = manufactured_kernels();
    kernels.growth_rate = [](std::size_t i, const auto&) { return i == 0 ? 1e-20 : 0.0; };
    const auto growth = P::advance_upper_bubble_population(grid, initial, {}, 1.0, kernels);
    EXPECT_DOUBLE_EQ(growth.state.number_density[0], initial.number_density[0]);
    EXPECT_NEAR(growth.number_change[0], -1.0, 2e-15);
    EXPECT_NEAR(growth.number_change[1], 1.0, 2e-15);
    EXPECT_NEAR(growth.hydrogen_change[0] + growth.hydrogen_change[1], growth.dissolved_to_gas_moles, 2e-15);
    EXPECT_NEAR(growth.number_change[0] + growth.number_change[1], 0.0, 2e-15);
}

TEST(RadiolyticBubbleKineticsTest, TinyCollisionAndBreakupLedgersDebitTheirUnchangedLargeDonor)
{
    const P::MolarPivotGrid grid{{1.0, 2.0, 4.0}};
    auto kernels = manufactured_kernels();
    kernels.coalescence_kernel = [](std::size_t i, std::size_t j, const auto&)
    { return i == 0 && j == 0 ? 2e-40 : 0.0; };
    const auto coalesced = P::advance_upper_bubble_population(grid, {0.0, {1e20, 0.0, 0.0}}, {}, 1.0, kernels);
    EXPECT_DOUBLE_EQ(coalesced.state.number_density[0], 1e20);
    EXPECT_NEAR(coalesced.coalescence_events, 1.0, 2e-15);
    EXPECT_NEAR(coalesced.number_change[0], -2.0, 4e-15);
    EXPECT_NEAR(coalesced.number_change[1], 1.0, 2e-15);
    EXPECT_DOUBLE_EQ(coalesced.hydrogen_change[0] + coalesced.hydrogen_change[1], 0.0);
    kernels = manufactured_kernels();
    kernels.breakup_frequency = [](std::size_t, const auto&) { return 1e-20; };
    kernels.breakup_daughters = [](std::size_t, const auto&)
    { return std::vector<P::BubbleBreakupDaughter>{{1.0, 2.0}}; };
    const auto broken = P::advance_upper_bubble_population(grid, {0.0, {0.0, 1e20, 0.0}}, {}, 1.0, kernels);
    EXPECT_DOUBLE_EQ(broken.state.number_density[1], 1e20);
    EXPECT_NEAR(broken.number_change[0], 2.0, 4e-15);
    EXPECT_NEAR(broken.number_change[1], -1.0, 2e-15);
    EXPECT_DOUBLE_EQ(broken.hydrogen_change[0] + broken.hydrogen_change[1], 0.0);
}

} // namespace
