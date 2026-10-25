/** @file testReactingVoFCoupler.cc
 * @brief Independent planar transfer, capture, reacting_gas and vent conservation oracles.
 */
#include "solvers/ReactingVoFCoupler.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Solver = AdaptiveVoFSolver;
using State = VoF::LowMachState;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr double temperature = 300.0, pressure = 100000.0, area = 0.25;
constexpr double film_thickness = 0.02, diffusivity = 1e-5, henry = 1e-5;

double sum(double local)
{
    double global = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

double total(const std::vector<double>& values)
{
    return sum(std::accumulate(values.begin(), values.end(), 0.0));
}

double free_hydrogen(const State& s)
{
    double value = total(s.resolved_hydrogen_moles) + total(s.dissolved_hydrogen_moles);
    for (const auto& population : s.dispersed)
        value += total(population.hydrogen_moles);
    return value;
}

double mass(const State& s, const VoF::LowMachOptions& options)
{
    double value = total(s.liquid_mass) + options.hydrogen.molar_mass * total(s.resolved_hydrogen_moles) +
                   options.inert.molar_mass * total(s.resolved_inert_moles);
    for (const auto& population : s.dispersed)
        value += options.hydrogen.molar_mass * total(population.hydrogen_moles);
    return value;
}

VoF::LowMachOptions thermodynamics()
{
    VoF::LinearCompressibleLiquid liquid;
    liquid.bounds = {250.0, 400.0, 10000.0, 1000000.0};
    liquid.reference_temperature = temperature;
    liquid.reference_pressure = pressure;
    liquid.reference_density = 1000.0;
    liquid.isobaric_heat_capacity = 4000.0;
    liquid.thermal_expansion = 2e-4;
    liquid.compressibility = 5e-9;
    liquid.reference_internal_energy = 100000.0;
    VoF::LowMachOptions options;
    options.liquid = VoF::make_linear_compressible_liquid(liquid);
    options.hydrogen = {0.002, 20.0, temperature, 5000.0, 0.0, 0.0};
    options.inert = {0.028, 25.0, temperature, 7500.0, 0.0, 0.0};
    options.volume_relative_tolerance = 2e-11;
    return options;
}

SP<Solver> make_flow(bool refine = false, VoF::Vector gravity = {})
{
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {1.0, 0.5, 0.5};
    grid.periodic = {false, true, true};
    grid.max_level = 1;
    std::shared_ptr<const SimpleFluid::Meshes::AMR::CartesianAMR> hierarchy =
        std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid);
    if (refine)
        hierarchy = hierarchy->adapt({0, 1});
    VoFMaterial material;
    material.liquid_density = 1000.0;
    material.gas_density = 0.1;
    material.liquid_viscosity = material.gas_viscosity = material.surface_tension = 0.0;
    VoFSolverOptions flow_options;
    flow_options.gravity = gravity;
    return std::make_shared<Solver>(hierarchy, material, flow_options);
}

ReactingVoFOptions disabled_options()
{
    ReactingVoFOptions options;
    options.reacting_gas.mode = ReactingGasMode::Disabled;
    options.reacting_gas.surface_tension = 0.0;
    options.reacting_gas.henry_coefficient = henry;
    options.reacting_gas.hydrogen_diffusivity = diffusivity;
    options.reacting_gas.nucleation_radius = 1e-5;
    options.reacting_gas.micro_to_large_conversion_coefficient = 0.0;
    options.dissolved_diffusion = false;
    return options;
}

void enable_film(ReactingVoFOptions& options)
{
    options.resolved_transfer.enabled = true;
    options.resolved_transfer.model_name = "prescribed steady planar verification film";
    options.resolved_transfer.validity_description =
        "Planar x-normal interface, constant synthetic diffusivity and supplied thickness.";
    options.resolved_transfer.thickness = [](const auto&, double, double) { return film_thickness; };
    options.resolved_transfer.validate = [](const auto& patch, double delta, double T, double P)
    {
        if (std::abs(patch.normal[0]) < 0.999 || delta != film_thickness || T < 250 || T > 400 || P <= 0)
            throw std::domain_error("The planar verification film is outside its declared domain.");
    };
}

VoF::LowMachInitialState planar_state(const Solver& flow, double concentration, double hydrogen_fraction = 1.0,
    double number_density = 0.0, VoF::Vector slip = {}, double transverse_velocity = 0.0,
    bool prescribed_bulk_film = false)
{
    const auto thermo = thermodynamics();
    VoF::LowMachInitialState initial;
    initial.thermodynamic_pressure = pressure;
    const auto n = flow.mesh()->num_owned_cells();
    initial.temperature.assign(n, temperature);
    initial.velocity.assign(n, VoF::Vector{0.0, transverse_velocity, 0.0});
    VoF::DispersedPopulation population;
    population.name = "incident";
    constexpr double radius = 1e-4;
    const double bubble_volume = 4.0 * std::numbers::pi / 3.0 * radius * radius * radius;
    const double bubble_moles = pressure * bubble_volume / (thermo.gas_constant * temperature);
    for (std::size_t c = 0; c < n; ++c)
    {
        const auto box = flow.hierarchy()->cell_box(flow.mesh()->cell_global_id(static_cast<LO>(c)));
        const bool gas = 0.5 * (box.lower[0] + box.upper[0]) < 0.5;
        const double volume = flow.mesh()->cell_volume(static_cast<LO>(c));
        const double count = gas ? 0.0 : number_density * volume;
        initial.resolved_fraction.push_back(gas ? 1.0 : 0.0);
        initial.resolved_hydrogen_mole_fraction.push_back(hydrogen_fraction);
        double C = concentration;
        if (prescribed_bulk_film && box.lower[0] == 0.5)
        {
            const double factor = 1.0 - film_thickness / (2.0 * (box.upper[0] - box.lower[0]));
            const double equilibrium = henry * hydrogen_fraction * pressure;
            C = equilibrium + (concentration - equilibrium) * factor;
        }
        initial.dissolved_hydrogen_moles.push_back(gas ? 0.0 : C * (volume - count * bubble_volume));
        population.number.push_back(count);
        population.hydrogen_moles.push_back(count * bubble_moles);
        population.drift_velocity.push_back(slip);
    }
    if (number_density > 0)
        initial.dispersed.push_back(std::move(population));
    return initial;
}

void expect_primary_equal(const State& actual, const State& expected)
{
    EXPECT_DOUBLE_EQ(actual.thermodynamic_pressure, expected.thermodynamic_pressure);
    EXPECT_EQ(actual.liquid_mass, expected.liquid_mass);
    EXPECT_EQ(actual.resolved_hydrogen_moles, expected.resolved_hydrogen_moles);
    EXPECT_EQ(actual.resolved_inert_moles, expected.resolved_inert_moles);
    EXPECT_EQ(actual.dissolved_hydrogen_moles, expected.dissolved_hydrogen_moles);
    EXPECT_EQ(actual.donor_hydrogen_moles, expected.donor_hydrogen_moles);
    EXPECT_EQ(actual.internal_energy, expected.internal_energy);
    EXPECT_EQ(actual.momentum, expected.momentum);
    EXPECT_EQ(actual.temperature, expected.temperature);
    EXPECT_EQ(actual.resolved_fraction, expected.resolved_fraction);
    EXPECT_EQ(actual.dispersed_fraction, expected.dispersed_fraction);
    EXPECT_EQ(actual.liquid_fraction, expected.liquid_fraction);
    ASSERT_EQ(actual.dispersed.size(), expected.dispersed.size());
    for (std::size_t k = 0; k < actual.dispersed.size(); ++k)
    {
        EXPECT_EQ(actual.dispersed[k].number, expected.dispersed[k].number);
        EXPECT_EQ(actual.dispersed[k].hydrogen_moles, expected.dispersed[k].hydrogen_moles);
        EXPECT_EQ(actual.dispersed[k].drift_velocity, expected.dispersed[k].drift_velocity);
        EXPECT_EQ(actual.dispersed[k].volume, expected.dispersed[k].volume);
        EXPECT_EQ(actual.dispersed[k].radius, expected.dispersed[k].radius);
    }
}

void expect_budgets(const State& before, const ReactingVoFCoupler& coupler, double escaped_mass = 0.0)
{
    const auto& flow = *coupler.solver();
    const auto& after = flow.low_mach_state();
    const auto& stats = coupler.statistics();
    const auto& fluid = flow.low_mach_diagnostics();
    const double hydrogen_scale = std::max({1e-20, free_hydrogen(before), stats.hydrogen_produced});
    EXPECT_NEAR(free_hydrogen(after) - free_hydrogen(before), stats.hydrogen_produced - stats.hydrogen_escaped,
        2e-9 * hydrogen_scale);
    EXPECT_NEAR(total(after.donor_hydrogen_moles) - total(before.donor_hydrogen_moles), -stats.donor_consumed,
        2e-9 * std::max(1e-20, total(before.donor_hydrogen_moles)));
    EXPECT_NEAR(mass(after, flow.low_mach_options()) - mass(before, flow.low_mach_options()), -escaped_mass,
        5e-10 * mass(before, flow.low_mach_options()));
    EXPECT_NEAR(total(after.internal_energy) - total(before.internal_energy),
        stats.thermal_input - stats.escaped_enthalpy + fluid.pressure_work,
        5e-10 * std::max(std::abs(total(before.internal_energy)), 1e-20));
    EXPECT_NEAR(stats.hydrogen_error, 0.0, 2e-9 * hydrogen_scale);
    EXPECT_LE(fluid.maximum_occupancy_error, 2e-8);
    EXPECT_LE(fluid.maximum_continuity_residual, 5e-8);
}

TEST(ReactingVoFCouplerTest, DisabledProcessesPreserveTheAcceptedStateAndProduceNoEscape)
{
    auto flow = make_flow();
    flow->initialize_low_mach(thermodynamics(), planar_state(*flow, 1.0));
    const auto before = flow->low_mach_state();
    ReactingVoFCoupler coupler(flow, disabled_options());
    const auto trial = coupler.preview(0.001, {});
    expect_primary_equal(flow->low_mach_state(), before);
    EXPECT_DOUBLE_EQ(flow->time(), 0.0);
    EXPECT_DOUBLE_EQ(coupler.statistics().cumulative_escaped, 0.0);
    coupler.commit(trial);
    EXPECT_DOUBLE_EQ(coupler.statistics().hydrogen_produced, 0.0);
    EXPECT_DOUBLE_EQ(coupler.statistics().hydrogen_captured, 0.0);
    EXPECT_DOUBLE_EQ(coupler.statistics().hydrogen_escaped, 0.0);
    expect_budgets(before, coupler);
}

TEST(ReactingVoFCouplerTest, BulkReactingGasDebitsFiniteDonorOnceAndReplayDoesNotDoubleProduce)
{
    auto flow = make_flow();
    auto initial = planar_state(*flow, 0.0);
    initial.resolved_fraction.assign(initial.temperature.size(), 0.2);
    VoF::DispersedPopulation micro;
    micro.name = "micro";
    micro.number.assign(initial.temperature.size(), 0.0);
    micro.hydrogen_moles = micro.number;
    initial.dispersed.push_back(micro);
    ReactingVoFInput input;
    for (std::size_t c = 0; c < initial.temperature.size(); ++c)
    {
        const double volume = flow->mesh()->cell_volume(static_cast<LO>(c));
        initial.donor_hydrogen_moles.push_back(0.01 * volume);
        input.reaction_driver_energy.push_back(100.0 * volume);
    }
    flow->initialize_low_mach(thermodynamics(), initial);
    const auto before = flow->low_mach_state();
    auto options = disabled_options();
    options.reacting_gas.mode = ReactingGasMode::TwoPopulation;
    options.reacting_gas.hydrogen_yield_mol_per_j = 2e-7;
    options.reacting_gas.gas_release_efficiency = 0.5;
    options.reacting_gas.microbubble_lifetime = 1e100;
    ReactingVoFCoupler coupler(flow, options);
    const double expected = 1e-7 * total(input.reaction_driver_energy);
    const auto checkpoint = coupler.checkpoint();
    const auto trial = coupler.preview(1e-4, input);
    expect_primary_equal(flow->low_mach_state(), before);
    EXPECT_DOUBLE_EQ(coupler.statistics().cumulative_produced, 0.0);
    EXPECT_NEAR(trial.statistics().hydrogen_produced, expected, 2e-13 * expected);
    EXPECT_NEAR(total(trial.state().dispersed[0].hydrogen_moles), expected, 2e-9 * expected);
    coupler.commit(trial);
    EXPECT_NEAR(coupler.statistics().donor_consumed, expected, 2e-13 * expected);
    EXPECT_DOUBLE_EQ(coupler.statistics().thermal_input, 0.0);
    expect_budgets(before, coupler);
    const auto accepted = flow->low_mach_state();
    EXPECT_THROW(coupler.commit(trial), std::exception);
    expect_primary_equal(flow->low_mach_state(), accepted);
    coupler.restore(checkpoint);
    expect_primary_equal(flow->low_mach_state(), before);
    EXPECT_DOUBLE_EQ(coupler.statistics().cumulative_produced, 0.0);
    const auto replay = coupler.preview(1e-4, input);
    expect_primary_equal(replay.state(), trial.state());
    coupler.commit(replay);
    coupler.release_checkpoint(checkpoint);
    EXPECT_NEAR(coupler.statistics().cumulative_produced, expected, 2e-13 * expected);
    auto exhausted = input;
    for (double& energy : exhausted.reaction_driver_energy)
        energy *= 1e5;
    const auto retained = flow->low_mach_state();
    EXPECT_THROW((void) coupler.preview(1e-4, exhausted), std::exception);
    expect_primary_equal(flow->low_mach_state(), retained);
}

TEST(ReactingVoFCouplerTest, PlanarHenryTransferHasEquilibriumReversalAndUsesHydrogenPartialPressure)
{
    constexpr double dt = 0.001, liquid_cell_width = 0.25;
    const double length = film_thickness * (1.0 - film_thickness / (2.0 * liquid_cell_width));
    for (const auto [concentration, hydrogen_fraction] :
        {std::pair{1.0, 1.0}, std::pair{2.0, 1.0}, std::pair{0.0, 1.0}, std::pair{1.0, 0.0}})
    {
        SCOPED_TRACE(concentration);
        SCOPED_TRACE(hydrogen_fraction);
        auto flow = make_flow();
        flow->initialize_low_mach(thermodynamics(), planar_state(*flow, concentration, hydrogen_fraction));
        const auto before = flow->low_mach_state();
        auto options = disabled_options();
        enable_film(options);
        ReactingVoFCoupler coupler(flow, options);
        const auto stats = coupler.step(dt, {});
        const double expected =
            dt * area * diffusivity / length * (concentration - henry * pressure * hydrogen_fraction);
        const double measured = stats.dissolved_to_resolved - stats.resolved_to_dissolved;
        if (expected == 0.0)
            EXPECT_NEAR(measured, 0.0, 1e-12);
        else
            EXPECT_NEAR(measured, expected, 1e-4 * std::abs(expected));
        EXPECT_NEAR(total(flow->low_mach_state().resolved_hydrogen_moles) - total(before.resolved_hydrogen_moles),
            measured, 2e-10);
        EXPECT_NEAR(total(flow->low_mach_state().dissolved_hydrogen_moles) - total(before.dissolved_hydrogen_moles),
            -measured, 2e-10);
        EXPECT_DOUBLE_EQ(stats.hydrogen_escaped, 0.0);
        expect_budgets(before, coupler);
    }
}

TEST(ReactingVoFCouplerTest, CellAverageLinearFilmMatchesTheSamePhysicalFluxOnTwoMeshes)
{
    constexpr double dt = 0.001, bulk_concentration = 2.0;
    const double expected = dt * area * diffusivity / film_thickness * (bulk_concentration - henry * pressure);
    for (bool refined : {false, true})
    {
        SCOPED_TRACE(refined);
        auto flow = make_flow(refined);
        flow->initialize_low_mach(thermodynamics(), planar_state(*flow, bulk_concentration, 1.0, 0.0, {}, 0.0, true));
        const auto before = flow->low_mach_state();
        auto options = disabled_options();
        enable_film(options);
        ReactingVoFCoupler coupler(flow, options);
        const auto stats = coupler.step(dt, {});
        EXPECT_NEAR(stats.dissolved_to_resolved - stats.resolved_to_dissolved, expected, 1e-4 * expected);
        expect_budgets(before, coupler);
    }
}

TEST(ReactingVoFCouplerTest, CaptureEfficiencyAndCommonTranslationPreserveTheIncidentPopulationLedger)
{
    constexpr double density = 1e5, speed = 0.02, dt = 0.001;
    const double expected_number = dt * area * speed * density;
    double stationary_capture = 0.0;
    for (const auto [efficiency, translation] : {std::pair{0.0, 0.0}, std::pair{1.0, 0.0}, std::pair{1.0, 0.3}})
    {
        SCOPED_TRACE(efficiency);
        SCOPED_TRACE(translation);
        // A weak, explicitly defined gravity direction selects the prescribed
        // ConstantSlip direction without appreciable hydrostatic variation.
        auto flow = make_flow(false, {1e-8, 0.0, 0.0});
        flow->initialize_low_mach(
            thermodynamics(), planar_state(*flow, 0.0, 1.0, density, {-speed, 0.0, 0.0}, translation));
        const auto before = flow->low_mach_state();
        auto options = disabled_options();
        options.reacting_gas.rise_velocity_mode = BubbleRiseVelocityMode::ConstantSlip;
        options.reacting_gas.constant_slip_velocity = speed;
        options.capture.enabled = true;
        options.capture.model_name = "prescribed capture efficiency verification";
        options.capture.validity_description =
            "Dilute point bubbles incident on a planar surface with prescribed liquid-relative slip.";
        options.capture.efficiency = [efficiency](std::size_t, double, double, double) { return efficiency; };
        options.capture.validate = [](std::size_t, double radius, double approach, double T)
        {
            if (radius <= 0 || radius > 0.01 || approach < 0 || T < 250 || T > 400)
                throw std::domain_error("Capture verification domain violated.");
        };
        ReactingVoFCoupler coupler(flow, options);
        const auto stats = coupler.step(dt, {});
        if (efficiency == 0.0)
            EXPECT_DOUBLE_EQ(stats.captured_number, 0.0);
        else
            EXPECT_NEAR(stats.captured_number, expected_number, 1e-4 * expected_number);
        if (translation == 0.0 && efficiency == 1.0)
            stationary_capture = stats.captured_number;
        if (translation != 0.0)
            EXPECT_NEAR(stats.captured_number, stationary_capture, 1e-8 * expected_number);
        EXPECT_NEAR(total(flow->low_mach_state().dispersed[0].number) - total(before.dispersed[0].number),
            -stats.captured_number, 1e-8);
        EXPECT_NEAR(
            total(flow->low_mach_state().dispersed[0].hydrogen_moles) - total(before.dispersed[0].hydrogen_moles),
            -stats.hydrogen_captured, 1e-13);
        EXPECT_DOUBLE_EQ(stats.hydrogen_escaped, 0.0);
        expect_budgets(before, coupler);
        if (efficiency == 1.0 && translation == 0.0)
        {
            const auto physical_momentum = [](const State& state)
            {
                VoF::Vector result{};
                for (const auto& value : state.momentum)
                    for (std::size_t a = 0; a < 3; ++a)
                        result[a] += value[a];
                for (double& value : result)
                    value = sum(value);
                return result;
            };
            const auto first_state = flow->low_mach_state();
            const auto first_momentum = physical_momentum(first_state);
            VoF::RefinementPlan plan;
            plan.source_hierarchy = flow->hierarchy();
            plan.source_observation = flow->hierarchy()->observation();
            plan.refine = {0}; // Exercise gas-side refinement before the next physical step.
            const auto generation = flow->topology_generation();
            coupler.regrid(plan);
            EXPECT_GT(flow->topology_generation(), generation);
            const auto remapped = flow->low_mach_state();
            EXPECT_NEAR(total(remapped.liquid_mass), total(first_state.liquid_mass), 1e-9);
            EXPECT_NEAR(total(remapped.resolved_hydrogen_moles), total(first_state.resolved_hydrogen_moles), 2e-11);
            EXPECT_NEAR(total(remapped.dispersed[0].number), total(first_state.dispersed[0].number), 1e-8);
            EXPECT_NEAR(
                total(remapped.dispersed[0].hydrogen_moles), total(first_state.dispersed[0].hydrogen_moles), 1e-13);
            EXPECT_NEAR(total(remapped.internal_energy), total(first_state.internal_energy), 1e-6);
            const auto remapped_momentum = physical_momentum(remapped);
            for (std::size_t a = 0; a < 3; ++a)
                if (flow->hierarchy()->config().periodic[a])
                {
                    // Normal momentum can exchange with the closed x walls
                    // when the remapped velocity is projected onto its target.
                    SCOPED_TRACE(a);
                    EXPECT_NEAR(remapped_momentum[a], first_momentum[a], 1e-10);
                }
            const auto checkpoint = coupler.checkpoint();
            const auto second = coupler.preview(dt, {});
            expect_primary_equal(flow->low_mach_state(), remapped);
            coupler.commit(second);
            expect_budgets(remapped, coupler);
            EXPECT_NEAR(total(flow->low_mach_state().dispersed[0].number) - total(remapped.dispersed[0].number),
                -second.statistics().captured_number, 1e-8);
            coupler.restore(checkpoint);
            expect_primary_equal(flow->low_mach_state(), remapped);
            EXPECT_NEAR(coupler.statistics().cumulative_captured, stats.hydrogen_captured, 1e-13);
            const auto replay = coupler.preview(dt, {});
            expect_primary_equal(replay.state(), second.state());
            coupler.commit(replay);
            coupler.release_checkpoint(checkpoint);
            EXPECT_NEAR(coupler.statistics().cumulative_captured,
                stats.hydrogen_captured + second.statistics().hydrogen_captured, 1e-13);
        }
    }
}

TEST(ReactingVoFCouplerTest, ThermalEOSMotionAloneDoesNotCaptureZeroSlipBubbles)
{
    auto flow = make_flow();
    flow->initialize_low_mach(thermodynamics(), planar_state(*flow, 0.0, 1.0, 1e5));
    const auto before = flow->low_mach_state();
    auto options = disabled_options();
    options.capture.enabled = true;
    options.capture.model_name = "unit efficiency zero-slip verification";
    options.capture.validity_description =
        "No interfacial Henry mass transfer; thermal EOS motion is not a collision flux.";
    options.capture.efficiency = [](std::size_t, double, double, double) { return 1.0; };
    options.capture.validate = [](std::size_t, double, double, double) {};
    ReactingVoFInput input;
    for (std::size_t c = 0; c < before.temperature.size(); ++c)
        input.net_thermal_energy.push_back(flow->mesh()->cell_volume(static_cast<LO>(c)));
    ReactingVoFCoupler coupler(flow, options);
    const auto stats = coupler.step(0.001, input);
    EXPECT_DOUBLE_EQ(stats.captured_number, 0.0);
    EXPECT_DOUBLE_EQ(stats.hydrogen_captured, 0.0);
    EXPECT_GT(total(flow->low_mach_state().temperature) - total(before.temperature), 0.0);
    EXPECT_NEAR(total(flow->low_mach_state().dispersed[0].number), total(before.dispersed[0].number), 1e-8);
    expect_budgets(before, coupler);
}

TEST(ReactingVoFCouplerTest, OnlyDesignatedHeadspaceWithdrawalCountsAsEscapeAndCarriesEnthalpy)
{
    auto flow = make_flow();
    auto initial = planar_state(*flow, 0.0, 0.25);
    initial.resolved_fraction.assign(initial.temperature.size(), 1.0);
    flow->initialize_low_mach(thermodynamics(), initial);
    const auto before = flow->low_mach_state();
    auto options = disabled_options();
    options.headspace.enabled = true;
    options.headspace.regions = {{{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}}};
    options.headspace.model_name = "prescribed headspace molar outflow verification";
    options.headspace.validity_description = "Well-mixed H2/inert ideal gas in explicitly designated pure-gas cells.";
    options.headspace.molar_rate = [](double, double, double) { return 0.01; };
    options.headspace.validate = [](double P, double T, double y)
    {
        if (P <= 0 || T < 250 || T > 400 || y < 0 || y > 1)
            throw std::domain_error("Invalid headspace state.");
    };
    ReactingVoFCoupler coupler(flow, options);
    const auto stats = coupler.step(0.001, {});
    constexpr double removed = 1e-5;
    EXPECT_NEAR(stats.hydrogen_escaped, 0.25 * removed, 2e-12);
    EXPECT_NEAR(flow->low_mach_diagnostics().escaped_inert, 0.75 * removed, 2e-12);
    EXPECT_NEAR(total(flow->low_mach_state().resolved_hydrogen_moles) - total(before.resolved_hydrogen_moles),
        -0.25 * removed, 2e-10);
    const auto& thermo = flow->low_mach_options();
    const double expected_enthalpy =
        removed * (0.25 * thermo.hydrogen.reference_internal_energy + 0.75 * thermo.inert.reference_internal_energy +
                      thermo.gas_constant * temperature);
    EXPECT_NEAR(stats.escaped_enthalpy, expected_enthalpy, 1e-4 * expected_enthalpy);
    EXPECT_NEAR(stats.cumulative_escaped, stats.hydrogen_escaped, 2e-12);
    const double removed_mass = removed * (0.25 * thermo.hydrogen.molar_mass + 0.75 * thermo.inert.molar_mass);
    expect_budgets(before, coupler, removed_mass);
}

TEST(ReactingVoFCouplerTest, RankLocalFilmFailureRetainsAllAcceptedPhysicalInventories)
{
    auto flow = make_flow();
    flow->initialize_low_mach(thermodynamics(), planar_state(*flow, 2.0));
    const auto before = flow->low_mach_state();
    const auto fraction = flow->gas_fraction(), faces = flow->face_velocity();
    auto options = disabled_options();
    enable_film(options);
    options.resolved_transfer.validate = [](const auto&, double, double, double)
    {
        if (Tpetra::getDefaultComm()->getRank() == 0)
            throw std::domain_error("Manufactured rank-local film failure.");
    };
    ReactingVoFCoupler coupler(flow, options);
    EXPECT_THROW((void) coupler.preview(0.001, {}), std::exception);
    expect_primary_equal(flow->low_mach_state(), before);
    EXPECT_EQ(flow->gas_fraction(), fraction);
    EXPECT_EQ(flow->face_velocity(), faces);
    EXPECT_DOUBLE_EQ(flow->time(), 0.0);
    EXPECT_DOUBLE_EQ(coupler.statistics().cumulative_produced, 0.0);
    EXPECT_DOUBLE_EQ(coupler.statistics().cumulative_captured, 0.0);
    EXPECT_DOUBLE_EQ(coupler.statistics().cumulative_escaped, 0.0);
}

TEST(ReactingVoFCouplerTest, CoarseFineDiffusionUsesDistanceWeightedAqueousMaterialResistance)
{
    auto base = make_flow();
    auto flow = std::make_shared<Solver>(base->hierarchy()->adapt({1}), base->material(), base->options());
    ASSERT_TRUE(std::any_of(flow->hierarchy()->blocks().begin(), flow->hierarchy()->blocks().end(),
        [](const auto& block) { return block.level == 0; }));
    ASSERT_TRUE(std::any_of(flow->hierarchy()->blocks().begin(), flow->hierarchy()->blocks().end(),
        [](const auto& block) { return block.level == 1; }));
    auto thermo = thermodynamics();
    VoF::LinearCompressibleLiquid liquid;
    liquid.bounds = {250.0, 400.0, 10000.0, 1000000.0};
    liquid.reference_temperature = temperature;
    liquid.reference_pressure = pressure;
    liquid.reference_density = 1000.0;
    liquid.isobaric_heat_capacity = 4000.0;
    liquid.compressibility = 5e-9;
    liquid.reference_internal_energy = 100000.0;
    // No thermal expansion or conductivity: the prescribed T/D discontinuity
    // remains stationary while the dilute dissolved species counterdiffuses.
    thermo.liquid = VoF::make_linear_compressible_liquid(liquid);
    auto initial = planar_state(*flow, 0.0);
    initial.resolved_fraction.assign(initial.temperature.size(), 0.0);
    initial.dissolved_hydrogen_moles.clear();
    VoF::DispersedPopulation population;
    population.name = "prescribed_void";
    for (std::size_t c = 0; c < initial.temperature.size(); ++c)
    {
        const bool left = flow->mesh()->cell_centroid(static_cast<LO>(c)).x < 0.5;
        const double T = left ? 290.0 : 310.0, alpha = left ? 0.02 : 0.08;
        const double volume = flow->mesh()->cell_volume(static_cast<LO>(c));
        initial.temperature[c] = T;
        initial.dissolved_hydrogen_moles.push_back((left ? 2.0 : 1.0) * (1.0 - alpha) * volume);
        const double moles = alpha * volume * pressure / (thermo.gas_constant * T);
        population.hydrogen_moles.push_back(moles);
        population.number.push_back(moles / 1e-8);
    }
    initial.dispersed.push_back(std::move(population));
    flow->initialize_low_mach(thermo, initial);
    const auto before = flow->low_mach_state();
    auto options = disabled_options();
    options.dissolved_diffusion = true;
    options.reacting_gas.diffusivity_mode = HydrogenDiffusivityMode::External;
    options.reacting_gas.hydrogen_diffusivity_correlation = [](double T) { return T < 300.0 ? 1e-5 : 4e-5; };
    ReactingVoFCoupler coupler(flow, options);
    constexpr double dt = 0.001;
    (void) coupler.step(dt, {});
    const double expected = dt * area * (2.0 - 1.0) / (0.125 / (0.98 * 1e-5) + 0.0625 / (0.92 * 4e-5));
    double removed_left = 0;
    for (std::size_t c = 0; c < initial.temperature.size(); ++c)
    {
        if (flow->mesh()->cell_centroid(static_cast<LO>(c)).x < 0.5)
            removed_left += before.dissolved_hydrogen_moles[c] - flow->low_mach_state().dissolved_hydrogen_moles[c];
        EXPECT_NEAR(flow->low_mach_state().temperature[c], before.temperature[c], 5e-7);
    }
    EXPECT_NEAR(sum(removed_left), expected, 1e-5 * expected);
    EXPECT_NEAR(total(flow->low_mach_state().dissolved_hydrogen_moles), total(before.dissolved_hydrogen_moles), 2e-11);
    expect_budgets(before, coupler);
}

TEST(ReactingVoFCouplerTest, NamedUpperBinsRouteGrowthAndExplicitUnderflowToTheCorrectPopulations)
{
    constexpr double pivot = 1e-9, density = 1e6, dt = 0.001;
    for (bool underflow : {false, true})
    {
        SCOPED_TRACE(underflow);
        auto flow = make_flow();
        auto initial = planar_state(*flow, 0.0);
        initial.resolved_fraction.assign(initial.temperature.size(), 0.0);
        for (std::size_t c = 0; c < initial.temperature.size(); ++c)
            initial.dissolved_hydrogen_moles[c] = 0.01 * flow->mesh()->cell_volume(static_cast<LO>(c));
        // Transport storage deliberately differs from fixed-pivot order.
        for (const auto& name : {"upper_four", "micro", "upper_two", "upper_one"})
        {
            VoF::DispersedPopulation population;
            population.name = name;
            for (std::size_t c = 0; c < initial.temperature.size(); ++c)
            {
                const double number =
                    population.name == "upper_one" ? density * flow->mesh()->cell_volume(static_cast<LO>(c)) : 0.0;
                population.number.push_back(number);
                population.hydrogen_moles.push_back(pivot * number);
            }
            initial.dispersed.push_back(std::move(population));
        }
        flow->initialize_low_mach(thermodynamics(), initial);
        const auto before = flow->low_mach_state();
        auto options = disabled_options();
        options.upper_pivots.mol_per_bubble = {pivot, 2.0 * pivot, 4.0 * pivot};
        options.upper_populations = {"upper_one", "upper_two", "upper_four"};
        options.population_step.underflow = underflow ? BubbleGasPhysics::BubbleUnderflowPolicy::ReturnToMicro
                                                      : BubbleGasPhysics::BubbleUnderflowPolicy::Reject;
        options.upper_kernels.model_name = "prescribed fixed-pivot routing verification";
        options.upper_kernels.validity_description =
            "Positive aqueous volume, dilute prescribed populations, bounded exact size increment.";
        options.upper_kernels.validate = [](const auto& context)
        {
            if (!(context.environment.liquid_fraction > 0))
                throw std::domain_error("No aqueous kinetic carrier.");
            if (context.environment.hydrogen_diffusivity != diffusivity)
                throw std::domain_error("Upper growth did not receive its supplied molecular diffusivity.");
        };
        options.upper_kernels.growth_rate = [underflow](std::size_t i, const auto&)
        { return i == 0 ? (underflow ? -0.5 : 1.0) * pivot / dt : 0.0; };
        ReactingVoFCoupler coupler(flow, options);
        (void) coupler.step(dt, {});
        const auto& after = flow->low_mach_state();
        const double old_number = total(before.dispersed[3].number);
        EXPECT_NEAR(total(after.dispersed[3].number), 0.0, 1e-10 * old_number);
        EXPECT_NEAR(total(after.dispersed[0].number), 0.0, 1e-10 * old_number);
        const std::size_t destination = underflow ? 1 : 2;
        EXPECT_NEAR(total(after.dispersed[destination].number), old_number, 2e-10 * old_number);
        const double target_size = underflow ? 0.5 * pivot : 2.0 * pivot;
        EXPECT_NEAR(total(after.dispersed[destination].hydrogen_moles), target_size * old_number,
            2e-10 * target_size * old_number);
        const double dissolved_change = underflow ? 0.5 * pivot * old_number : -pivot * old_number;
        EXPECT_NEAR(total(after.dissolved_hydrogen_moles) - total(before.dissolved_hydrogen_moles), dissolved_change,
            2e-10 * std::abs(dissolved_change));
        EXPECT_DOUBLE_EQ(coupler.statistics().hydrogen_produced, 0.0);
        expect_budgets(before, coupler);
    }
}

TEST(ReactingVoFCouplerTest, NormalGalileanTranslationLeavesTheIncidentCaptureRateUnchanged)
{
    // The common normal velocity exceeds the slip: bubbles move in +x in the
    // laboratory frame, yet the faster +x interface still overtakes them.
    // dt*u/h <= 2.4e-6 bounds the first-order frozen-geometry capacity error.
    constexpr double density = 1e5, speed = 0.02, dt = 1e-6;
    const double expected = dt * area * speed * density;
    double stationary = 0;
    for (double translation : {0.0, 0.3})
    {
        SCOPED_TRACE(translation);
        SimpleFluid::Meshes::AMR::AMRGrid grid;
        grid.root_blocks = {4, 1, 1};
        grid.block_cells = {2, 2, 2};
        grid.extent = {1.0, 0.5, 0.5};
        grid.periodic = {true, true, true};
        grid.max_level = 0;
        const auto prototype = make_flow(false, {1e-8, 0.0, 0.0});
        auto flow = std::make_shared<Solver>(std::make_shared<const SimpleFluid::Meshes::AMR::CartesianAMR>(grid),
            prototype->material(), prototype->options());
        auto init = planar_state(*flow, 0.0);
        const auto thermo = thermodynamics();
        const double bubble_moles =
            pressure * (4.0 * std::numbers::pi / 3.0) * 1e-12 / (thermo.gas_constant * temperature);
        VoF::DispersedPopulation population;
        population.name = "incident";
        for (std::size_t c = 0; c < init.temperature.size(); ++c)
        {
            const double x = flow->mesh()->cell_centroid(static_cast<LO>(c)).x;
            const bool gas = x > 0.25 && x < 0.75;
            init.resolved_fraction[c] = gas ? 1.0 : 0.0;
            init.velocity[c] = {translation, 0.0, 0.0};
            const double count = gas ? 0.0 : density * flow->mesh()->cell_volume(static_cast<LO>(c));
            population.number.push_back(count);
            population.hydrogen_moles.push_back(count * bubble_moles);
            population.drift_velocity.push_back({-speed, 0.0, 0.0});
        }
        init.dispersed.push_back(std::move(population));
        flow->initialize_low_mach(thermo, init);
        const auto before = flow->low_mach_state();
        auto options = disabled_options();
        options.reacting_gas.rise_velocity_mode = BubbleRiseVelocityMode::ConstantSlip;
        options.reacting_gas.constant_slip_velocity = speed;
        options.capture.enabled = true;
        options.capture.model_name = "normal Galilean capture verification";
        options.capture.validity_description =
            "Dilute prescribed-slip bubbles at two planar interfaces, one incident and one receding.";
        options.capture.efficiency = [](std::size_t, double, double, double) { return 1.0; };
        options.capture.validate = [](std::size_t, double radius, double approach, double T)
        {
            if (!(radius > 0) || approach < 0 || T < 250 || T > 400)
                throw std::domain_error("Normal-translation capture fixture is outside its domain.");
        };
        ReactingVoFCoupler coupler(flow, options);
        const auto stats = coupler.step(dt, {});
        EXPECT_NEAR(stats.captured_number, expected, 1e-5 * expected);
        if (translation == 0.0)
            stationary = stats.captured_number;
        else
            EXPECT_NEAR(stats.captured_number, stationary, 1e-5 * expected);
        expect_budgets(before, coupler);
    }
}
TEST(ReactingVoFCouplerTest, ReinitializationRebindsRequiredPopulationNames)
{
    auto flow = make_flow();
    auto initial = planar_state(*flow, 0);
    initial.resolved_fraction.assign(initial.temperature.size(), 0.2);
    VoF::DispersedPopulation micro, spectator;
    micro.name = "micro";
    spectator.name = "spectator";
    micro.number.assign(initial.temperature.size(), 0);
    micro.hydrogen_moles = micro.number;
    spectator.number = spectator.hydrogen_moles = micro.number;
    initial.dispersed = {micro, spectator};
    ReactingVoFInput input;
    for (std::size_t c = 0; c < initial.temperature.size(); ++c)
    {
        const double volume = flow->mesh()->cell_volume(static_cast<LO>(c));
        initial.donor_hydrogen_moles.push_back(0.01 * volume);
        input.reaction_driver_energy.push_back(100 * volume);
    }
    flow->initialize_low_mach(thermodynamics(), initial);
    auto options = disabled_options();
    options.reacting_gas.mode = ReactingGasMode::TwoPopulation;
    options.reacting_gas.hydrogen_yield_mol_per_j = 2e-7;
    options.reacting_gas.gas_release_efficiency = 0.5;
    options.reacting_gas.microbubble_lifetime = 1e100;
    ReactingVoFCoupler coupler(flow, options);
    const auto old_trial = coupler.preview(1e-4, input);
    initial.dispersed = {spectator, micro};
    flow->initialize_low_mach(thermodynamics(), initial);
    EXPECT_THROW(coupler.commit(old_trial), std::exception);
    const auto before = flow->low_mach_state();
    const auto trial = coupler.preview(1e-4, input);
    const double expected = 1e-7 * total(input.reaction_driver_energy);
    EXPECT_EQ(trial.state().dispersed[0].name, "spectator");
    EXPECT_DOUBLE_EQ(total(trial.state().dispersed[0].hydrogen_moles), 0);
    EXPECT_NEAR(total(trial.state().dispersed[1].hydrogen_moles), expected, 2e-9 * expected);
    RecordProperty("micro_source_moles", total(trial.state().dispersed[1].hydrogen_moles));
    expect_primary_equal(flow->low_mach_state(), before);
    coupler.commit(trial);
    EXPECT_DOUBLE_EQ(total(flow->low_mach_state().dispersed[0].hydrogen_moles), 0);
    EXPECT_NEAR(total(flow->low_mach_state().dispersed[1].hydrogen_moles), expected, 2e-9 * expected);
    initial.dispersed = {spectator};
    flow->initialize_low_mach(thermodynamics(), initial);
    const auto missing = flow->low_mach_state();
    EXPECT_THROW((void) coupler.preview(1e-4, input), std::exception);
    expect_primary_equal(flow->low_mach_state(), missing);
}

TEST(ReactingVoFCouplerTest, ConstantSlipHasNoClosureOutsideTheLiquidCarrier)
{
    auto flow = make_flow(false, {-1e-12, 0, 0});
    auto initial = planar_state(*flow, 0);
    VoF::DispersedPopulation population;
    population.name = "empty";
    population.number.assign(initial.temperature.size(), 0);
    population.hydrogen_moles = population.number;
    initial.dispersed = {population};
    flow->initialize_low_mach(thermodynamics(), initial);
    auto options = disabled_options();
    options.reacting_gas.rise_velocity_mode = BubbleRiseVelocityMode::ConstantSlip;
    options.reacting_gas.constant_slip_velocity = 0.02;
    ReactingVoFCoupler coupler(flow, options);
    const auto trial = coupler.preview(0.01, {});
    for (std::size_t c = 0; c < initial.temperature.size(); ++c)
    {
        const double speed = initial.resolved_fraction[c] == 1 ? 0 : 0.02;
        EXPECT_DOUBLE_EQ(trial.state().dispersed[0].drift_velocity[c][0], speed);
        EXPECT_DOUBLE_EQ(trial.state().dispersed[0].radius[c], 0);
        EXPECT_DOUBLE_EQ(trial.state().dispersed[0].volume[c], 0);
    }
}

TEST(ReactingVoFCouplerTest, ParticleSlipUsesThePopulatedSideAtAnEmptyFront)
{
    // A radius-based closure has no radius in an empty cell. Supply its
    // evaluated drift only where moments exist and test the transport seam.
    auto flow = make_flow();
    auto initial = planar_state(*flow, 0);
    initial.resolved_fraction.assign(initial.temperature.size(), 0);
    constexpr double speed = 0.02, dt = 0.01;
    const auto thermo = thermodynamics();
    const double per_bubble = pressure * (4 * std::numbers::pi / 3) * 1e-12 / (thermo.gas_constant * temperature);
    VoF::DispersedPopulation population;
    population.name = "radius_closure";
    for (std::size_t c = 0; c < initial.temperature.size(); ++c)
    {
        const double x = flow->mesh()->cell_centroid(static_cast<LO>(c)).x;
        const double count = x > 0.25 && x < 0.5 ? 10 : 0;
        population.number.push_back(count);
        population.hydrogen_moles.push_back(count * per_bubble);
        population.drift_velocity.push_back({count > 0 ? speed : 0, 0, 0});
    }
    initial.dispersed = {population};
    flow->initialize_low_mach(thermo, initial);
    for (std::size_t c = 0; c < initial.temperature.size(); ++c)
        if (population.number[c] == 0)
            EXPECT_DOUBLE_EQ(flow->low_mach_state().dispersed[0].radius[c], 0);
    const auto trial = flow->preview_low_mach_step(dt, {});
    VoF::AdaptiveVoF geometry(flow->hierarchy());
    const auto moles = geometry.exchange_cells(trial.state().dispersed[0].hydrogen_moles);
    const auto T = geometry.exchange_cells(trial.state().temperature);
    const auto P = geometry.exchange_cells(trial.state().liquid_absolute_pressure);
    const auto& mesh = *flow->mesh();
    double front_faces = 0;
    for (std::size_t id = 0; id < mesh.num_owned_faces(); ++id)
    {
        const auto face = static_cast<LO>(id);
        const auto normal = mesh.face_normal(face);
        if (mesh.face_centroid(face).x != 0.5 || std::abs(normal.x) < 0.5)
            continue;
        const auto donor = normal.x > 0 ? mesh.owner_cell(face) : mesh.neighbor_cell(face);
        const double alpha = moles[donor] / mesh.cell_volume(donor) * thermo.gas_constant * T[donor] / P[donor];
        EXPECT_NEAR(trial.transport().dispersed_slip_volume[0][id] / (dt * mesh.face_area(face) * normal.x * alpha),
            speed, 2e-12);
        front_faces += 1;
    }
    EXPECT_GT(sum(front_faces), 0);
    EXPECT_NEAR(total(trial.state().dispersed[0].number), total(population.number), 2e-10);
    EXPECT_NEAR(total(trial.state().dispersed[0].hydrogen_moles), total(population.hydrogen_moles),
        2e-10 * total(population.hydrogen_moles));
}

TEST(ReactingVoFCouplerTest, ConstantSlipExtendsThroughEmptyLiquidAndReplay)
{
    // The x=0.5 front is also an MPI ownership boundary with two ranks.
    auto flow = make_flow(false, {-1e-12, 0, 0});
    auto initial = planar_state(*flow, 0);
    initial.resolved_fraction.assign(initial.temperature.size(), 0);
    VoF::DispersedPopulation population;
    population.name = "front";
    const auto thermo = thermodynamics();
    const double radius = 1e-4;
    const double per_bubble =
        pressure * (4 * std::numbers::pi / 3) * std::pow(radius, 3) / (thermo.gas_constant * temperature);
    for (std::size_t c = 0; c < initial.temperature.size(); ++c)
    {
        const auto box = flow->hierarchy()->cell_box(flow->mesh()->cell_global_id(static_cast<LO>(c)));
        const double count = box.lower[0] == 0.25 ? 10.0 : 0;
        population.number.push_back(count);
        population.hydrogen_moles.push_back(count * per_bubble);
    }
    initial.dispersed = {population};
    flow->initialize_low_mach(thermo, initial);
    auto options = disabled_options();
    options.reacting_gas.rise_velocity_mode = BubbleRiseVelocityMode::ConstantSlip;
    options.reacting_gas.constant_slip_velocity = 0.02;
    ReactingVoFCoupler coupler(flow, options);
    const auto checkpoint = coupler.checkpoint();
    const auto before = flow->low_mach_state();
    const auto trial = coupler.preview(0.01, {});
    // Independently reconstruct the implicit upwind profile using the published
    // carrier flux plus the configured relative slip. This separates closure
    // correctness from the carrier's (small, physical) response to bubble motion.
    VoF::AdaptiveVoF geometry(flow->hierarchy());
    const auto moles = geometry.exchange_cells(trial.state().dispersed[0].hydrogen_moles);
    const auto T = geometry.exchange_cells(trial.state().temperature);
    const auto P = geometry.exchange_cells(trial.state().liquid_absolute_pressure);
    std::array<double, 3> plane_flux{}, plane_faces{};
    double front_faces = 0;
    const auto& mesh = *flow->mesh();
    constexpr double dt = 0.01, speed = 0.02, cell_volume = 0.25 * 0.25 * 0.25;
    for (std::size_t id = 0; id < mesh.num_owned_faces(); ++id)
    {
        const auto face = static_cast<LO>(id);
        if (mesh.is_geometry_exterior_face(face))
            continue;
        const auto normal = mesh.face_normal(face);
        if (std::abs(normal.x) < 0.5)
            continue;
        const auto x = mesh.face_centroid(face).x;
        const auto plane = static_cast<std::size_t>(std::lround(4 * x)) - 1;
        ASSERT_LT(plane, plane_flux.size());
        const double relative = dt * mesh.face_area(face) * speed * normal.x;
        const double combined = trial.transport().carrier_volume[id] + relative;
        plane_flux[plane] += combined * normal.x;
        plane_faces[plane] += 1;
        if (x == 0.5)
        {
            const auto donor = combined >= 0 ? mesh.owner_cell(face) : mesh.neighbor_cell(face);
            const double alpha = moles[donor] / cell_volume * thermo.gas_constant * T[donor] / P[donor];
            ASSERT_GT(alpha, 0);
            const double measured_speed =
                trial.transport().dispersed_slip_volume[0][id] / (dt * mesh.face_area(face) * alpha * normal.x);
            EXPECT_NEAR(measured_speed, speed, 2e-12);
            RecordProperty("front_relative_slip", measured_speed);
            front_faces += 1;
        }
    }
    EXPECT_GT(sum(front_faces), 0);
    std::array<std::array<double, 5>, 4> system{};
    for (std::size_t i = 0; i < 4; ++i)
    {
        system[i][i] = 1;
        system[i][4] = i == 1 ? 10 : 0;
    }
    for (std::size_t plane = 0; plane < 3; ++plane)
    {
        const double flux = sum(plane_flux[plane]) / sum(plane_faces[plane]) / cell_volume;
        const auto donor = flux >= 0 ? plane : plane + 1;
        system[plane][donor] += flux;
        system[plane + 1][donor] -= flux;
    }
    for (std::size_t pivot = 0; pivot < 4; ++pivot)
    {
        const double diagonal = system[pivot][pivot];
        for (std::size_t j = pivot; j < 5; ++j)
            system[pivot][j] /= diagonal;
        for (std::size_t i = 0; i < 4; ++i)
            if (i != pivot)
            {
                const double factor = system[i][pivot];
                for (std::size_t j = pivot; j < 5; ++j)
                    system[i][j] -= factor * system[pivot][j];
            }
    }
    double profile_error = 0;
    for (std::size_t c = 0; c < initial.temperature.size(); ++c)
    {
        EXPECT_NEAR(trial.state().dispersed[0].drift_velocity[c][0], 0.02, 2e-15);
        const auto i = static_cast<std::size_t>(4 * mesh.cell_centroid(static_cast<LO>(c)).x);
        EXPECT_NEAR(trial.state().dispersed[0].number[c], system[i][4], 2e-10);
        profile_error += std::abs(trial.state().dispersed[0].number[c] - system[i][4]);
        if (i == 2)
            EXPECT_GT(trial.state().dispersed[0].number[c], 0);
        EXPECT_GE(trial.state().dispersed[0].number[c], 0);
        EXPECT_GE(trial.state().dispersed[0].hydrogen_moles[c], 0);
    }
    EXPECT_NEAR(total(trial.state().dispersed[0].number), total(before.dispersed[0].number), 2e-10);
    EXPECT_NEAR(total(trial.state().dispersed[0].hydrogen_moles), total(before.dispersed[0].hydrogen_moles),
        2e-10 * total(before.dispersed[0].hydrogen_moles));
    RecordProperty("number_profile_l1_error", sum(profile_error));
    RecordProperty(
        "number_conservation_error", total(trial.state().dispersed[0].number) - total(before.dispersed[0].number));
    RecordProperty("mole_conservation_error",
        total(trial.state().dispersed[0].hydrogen_moles) - total(before.dispersed[0].hydrogen_moles));
    coupler.commit(trial);
    const auto accepted = flow->low_mach_state();
    coupler.restore(checkpoint);
    const auto replay = coupler.preview(0.01, {});
    expect_primary_equal(replay.state(), accepted);
    EXPECT_EQ(replay.transport().dispersed_slip_volume, trial.transport().dispersed_slip_volume);
    coupler.release_checkpoint(checkpoint);
}

} // namespace
