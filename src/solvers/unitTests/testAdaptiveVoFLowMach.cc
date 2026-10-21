/** @file testAdaptiveVoFLowMach.cc
 * @brief Closed-volume thermodynamic solutions, conservative sources and transaction guards.
 */
#include "solvers/AdaptiveVoFSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <utility>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Solver = AdaptiveVoFSolver;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr double initial_temperature = 300.0, initial_pressure = 100000.0;

double global_sum(double local)
{
    double global = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

double total(const std::vector<double>& values)
{
    return global_sum(std::accumulate(values.begin(), values.end(), 0.0));
}

VoF::LinearCompressibleLiquid liquid_parameters()
{
    VoF::LinearCompressibleLiquid p;
    p.bounds = {250.0, 400.0, 10000.0, 1000000.0};
    p.reference_temperature = initial_temperature;
    p.reference_pressure = initial_pressure;
    p.reference_density = 1000.0;
    p.isobaric_heat_capacity = 4000.0;
    p.thermal_expansion = 2e-4;
    p.compressibility = 5e-9;
    p.reference_internal_energy = 100000.0;
    return p;
}

VoF::LowMachOptions thermodynamics()
{
    VoF::LowMachOptions options;
    options.liquid = VoF::make_linear_compressible_liquid(liquid_parameters());
    options.hydrogen = {0.002, 20.0, initial_temperature, 5000.0, 0.0, 0.0};
    options.inert = {0.028, 25.0, initial_temperature, 7500.0, 0.0, 0.0};
    // The liquid pressure oracle resolves volume errors below beta*0.02 Pa.
    options.volume_relative_tolerance = 2e-11;
    return options;
}

std::unique_ptr<Solver> solver(bool refine = false, bool periodic = false, VoF::Vector gravity = {})
{
    Meshes::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {1.0, 0.5, 0.5};
    if (periodic)
    {
        // Keep the periodic partners coarse while refining an interior root.
        grid.root_blocks = {4, 1, 1};
        grid.periodic = {true, true, true};
    }
    grid.max_level = 1;
    std::shared_ptr<const Meshes::CartesianAMR> hierarchy = std::make_shared<const Meshes::CartesianAMR>(grid);
    if (refine)
        hierarchy = hierarchy->adapt({1});
    VoFMaterial material;
    material.liquid_density = 1000.0;
    material.gas_density = 0.1;
    material.liquid_viscosity = material.gas_viscosity = material.surface_tension = 0.0;
    VoFSolverOptions flow_options;
    flow_options.gravity = gravity;
    return std::make_unique<Solver>(hierarchy, material, flow_options);
}

VoF::LowMachInitialState initial(const Solver& solver, double gas_fraction, double hydrogen_fraction = 1.0)
{
    VoF::LowMachInitialState state;
    const auto n = solver.mesh()->num_owned_cells();
    state.thermodynamic_pressure = initial_pressure;
    state.temperature.assign(n, initial_temperature);
    state.resolved_fraction.assign(n, gas_fraction);
    state.resolved_hydrogen_mole_fraction.assign(n, hydrogen_fraction);
    return state;
}

void expect_same_state(const VoF::LowMachState& actual, const VoF::LowMachState& expected)
{
    EXPECT_DOUBLE_EQ(actual.thermodynamic_pressure, expected.thermodynamic_pressure);
    const std::array<std::pair<const std::vector<double>*, const std::vector<double>*>, 15> fields{
        {{&actual.liquid_mass, &expected.liquid_mass},
            {&actual.resolved_hydrogen_moles, &expected.resolved_hydrogen_moles},
            {&actual.resolved_inert_moles, &expected.resolved_inert_moles},
            {&actual.dissolved_hydrogen_moles, &expected.dissolved_hydrogen_moles},
            {&actual.donor_hydrogen_moles, &expected.donor_hydrogen_moles},
            {&actual.internal_energy, &expected.internal_energy}, {&actual.temperature, &expected.temperature},
            {&actual.resolved_fraction, &expected.resolved_fraction},
            {&actual.dispersed_fraction, &expected.dispersed_fraction},
            {&actual.liquid_fraction, &expected.liquid_fraction}, {&actual.liquid_density, &expected.liquid_density},
            {&actual.resolved_density, &expected.resolved_density},
            {&actual.liquid_absolute_pressure, &expected.liquid_absolute_pressure},
            {&actual.resolved_absolute_pressure, &expected.resolved_absolute_pressure},
            {&actual.resolved_hydrogen_partial_pressure, &expected.resolved_hydrogen_partial_pressure}}};
    for (std::size_t f = 0; f < fields.size(); ++f)
    {
        SCOPED_TRACE(f);
        EXPECT_EQ(*fields[f].first, *fields[f].second);
    }
    EXPECT_EQ(actual.momentum, expected.momentum);
    ASSERT_EQ(actual.dispersed.size(), expected.dispersed.size());
    for (std::size_t q = 0; q < actual.dispersed.size(); ++q)
    {
        EXPECT_EQ(actual.dispersed[q].name, expected.dispersed[q].name);
        EXPECT_EQ(actual.dispersed[q].number, expected.dispersed[q].number);
        EXPECT_EQ(actual.dispersed[q].hydrogen_moles, expected.dispersed[q].hydrogen_moles);
        EXPECT_EQ(actual.dispersed[q].drift_velocity, expected.dispersed[q].drift_velocity);
        EXPECT_EQ(actual.dispersed[q].volume, expected.dispersed[q].volume);
        EXPECT_EQ(actual.dispersed[q].radius, expected.dispersed[q].radius);
    }
}

struct Snapshot
{
    double time;
    VoF::LowMachState state;
    std::vector<double> fraction, pressure, face_velocity;
    std::vector<VoF::Vector> velocity;
    explicit Snapshot(const Solver& s)
        : time(s.time()), state(s.low_mach_state()), fraction(s.gas_fraction()), pressure(s.pressure()),
          face_velocity(s.face_velocity()), velocity(s.velocity())
    {
    }
    void expect_equal(const Solver& s) const
    {
        EXPECT_DOUBLE_EQ(s.time(), time);
        expect_same_state(s.low_mach_state(), state);
        EXPECT_EQ(s.gas_fraction(), fraction);
        EXPECT_EQ(s.pressure(), pressure);
        EXPECT_EQ(s.face_velocity(), face_velocity);
        EXPECT_EQ(s.velocity(), velocity);
        for (std::size_t c = 0; c < fraction.size(); ++c)
        {
            const auto lid = static_cast<LO>(c);
            EXPECT_DOUBLE_EQ(s.fraction_field().value(lid), fraction[c]);
            EXPECT_DOUBLE_EQ(s.pressure_field().value(lid), pressure[c]);
            const auto u = s.velocity_field().value(lid);
            EXPECT_DOUBLE_EQ(u.x, velocity[c][0]);
            EXPECT_DOUBLE_EQ(u.y, velocity[c][1]);
            EXPECT_DOUBLE_EQ(u.z, velocity[c][2]);
        }
    }
};

void expect_conservation(const VoF::LowMachDiagnostics& diagnostics)
{
    EXPECT_NEAR(diagnostics.mass_error, 0.0, 5e-10 * std::max(diagnostics.total_mass, 1e-30));
    EXPECT_NEAR(diagnostics.hydrogen_error, 0.0, 5e-10 * std::max(diagnostics.hydrogen_inventory, 1e-30));
    EXPECT_NEAR(diagnostics.energy_error, 0.0, 5e-10 * std::max(std::abs(diagnostics.internal_energy), 1e-30));
    EXPECT_LE(diagnostics.maximum_occupancy_error, 2e-8);
    EXPECT_LE(diagnostics.maximum_continuity_residual, 5e-8);
}

TEST(AdaptiveVoFLowMachTest, ClosedIdealGasHeatingMatchesTheExactMixtureAndDoesNotPublishTrials)
{
    auto s = solver(true);
    const auto options = thermodynamics();
    s->initialize_low_mach(options, initial(*s, 1.0, 0.25));
    const Snapshot accepted(*s);
    const auto& old = accepted.state;
    VoF::LowMachSources sources;
    constexpr double rise = 15.0, dt = 0.001;
    for (std::size_t c = 0; c < old.temperature.size(); ++c)
    {
        const double heat = rise * (old.resolved_hydrogen_moles[c] * options.hydrogen.molar_heat_capacity_v +
                                       old.resolved_inert_moles[c] * options.inert.molar_heat_capacity_v);
        sources.net_thermal_energy.push_back(heat);
        sources.radiolysis_energy.push_back(100.0 * heat); // Chemistry diagnostic, not another thermal source.
    }
    const auto checkpoint = s->checkpoint();
    const auto trial = s->preview_low_mach_step(dt, sources);
    accepted.expect_equal(*s);
    const double expected_pressure = initial_pressure * (initial_temperature + rise) / initial_temperature;
    EXPECT_NEAR(trial.state().thermodynamic_pressure, expected_pressure, 5e-4);
    for (std::size_t c = 0; c < old.temperature.size(); ++c)
    {
        EXPECT_NEAR(trial.state().temperature[c], initial_temperature + rise, 5e-7);
        EXPECT_NEAR(trial.state().resolved_fraction[c], 1.0, 2e-9);
        EXPECT_NEAR(trial.state().resolved_hydrogen_partial_pressure[c], 0.25 * expected_pressure, 2e-4);
        EXPECT_NEAR(trial.state().resolved_hydrogen_moles[c], old.resolved_hydrogen_moles[c], 2e-11);
        EXPECT_NEAR(trial.state().resolved_inert_moles[c], old.resolved_inert_moles[c], 2e-11);
    }
    EXPECT_NEAR(
        total(trial.state().internal_energy) - total(old.internal_energy), total(sources.net_thermal_energy), 1e-7);
    expect_conservation(trial.diagnostics());
    s->commit_low_mach_trial(trial);
    EXPECT_DOUBLE_EQ(s->time(), dt);
    const Snapshot committed(*s);
    EXPECT_THROW(s->commit_low_mach_trial(trial), std::exception);
    committed.expect_equal(*s);
    s->restore(checkpoint);
    accepted.expect_equal(*s);
    EXPECT_THROW(s->commit_low_mach_trial(trial), std::exception);
    accepted.expect_equal(*s);
    const auto replay = s->preview_low_mach_step(dt, sources);
    expect_same_state(replay.state(), trial.state());
    s->commit_low_mach_trial(replay);
    s->release_checkpoint(checkpoint);
}

TEST(AdaptiveVoFLowMachTest, RetainedTrialSurvivesRejectedPreviewButNotReinitialization)
{
    auto s = solver(true);
    const auto options = thermodynamics();
    const auto init = initial(*s, 1.0, 0.25);
    s->initialize_low_mach(options, init);
    const auto observation = s->hierarchy()->observation();
    const auto revision = s->state_revision();
    const Snapshot accepted(*s);
    const auto retained = s->preview_low_mach_step(0.001, {});
    const auto retained_state = retained.state();
    const auto retained_flux = retained.transport().resolved_volume;
    EXPECT_THROW((void) s->preview_low_mach_step(-0.001, {}), std::exception);
    VoF::LowMachSources impossible;
    impossible.net_thermal_energy.assign(accepted.state.temperature.size(), -1e20);
    EXPECT_THROW((void) s->preview_low_mach_step(0.001, impossible), std::exception);
    accepted.expect_equal(*s);
    EXPECT_EQ(s->state_revision(), revision);
    EXPECT_EQ(s->hierarchy()->observation(), observation);
    expect_same_state(retained.state(), retained_state);
    EXPECT_EQ(retained.transport().resolved_volume, retained_flux);
    EXPECT_NO_THROW(s->commit_low_mach_trial(retained));
    EXPECT_GT(s->state_revision(), revision);
    EXPECT_EQ(s->hierarchy()->observation(), observation);

    const auto stale = s->preview_low_mach_step(0.001, {});
    s->initialize_low_mach(options, init);
    const Snapshot reset(*s);
    EXPECT_EQ(s->hierarchy()->observation(), observation);
    EXPECT_THROW(s->commit_low_mach_trial(stale), std::exception);
    reset.expect_equal(*s);
    expect_same_state(retained.state(), retained_state);
    EXPECT_EQ(retained.transport().resolved_volume, retained_flux);
}

TEST(AdaptiveVoFLowMachTest, ClosedLiquidHeatingMatchesTheIntegratedIsochoricSolution)
{
    auto s = solver(true);
    const auto parameters = liquid_parameters();
    s->initialize_low_mach(thermodynamics(), initial(*s, 0.0));
    const auto old = s->low_mach_state();
    constexpr double rise = 2.0;
    const double expected_temperature = initial_temperature + rise;
    const double specific_heat =
        parameters.isobaric_heat_capacity * rise -
        parameters.thermal_expansion * parameters.thermal_expansion /
            (2.0 * parameters.reference_density * parameters.compressibility) *
            (expected_temperature * expected_temperature - initial_temperature * initial_temperature);
    VoF::LowMachSources sources;
    for (double mass : old.liquid_mass)
        sources.net_thermal_energy.push_back(mass * specific_heat);
    const auto trial = s->preview_low_mach_step(0.001, sources);
    const double expected_pressure =
        initial_pressure + parameters.thermal_expansion / parameters.compressibility * rise;
    EXPECT_NEAR(trial.state().thermodynamic_pressure, expected_pressure, 0.02);
    for (std::size_t c = 0; c < old.temperature.size(); ++c)
    {
        EXPECT_NEAR(trial.state().temperature[c], expected_temperature, 5e-7);
        EXPECT_NEAR(trial.state().liquid_density[c], parameters.reference_density, 2e-6);
        EXPECT_NEAR(trial.state().liquid_fraction[c], 1.0, 2e-9);
        EXPECT_DOUBLE_EQ(trial.state().resolved_fraction[c], 0.0);
    }
    EXPECT_NEAR(total(trial.state().liquid_mass), total(old.liquid_mass), 5e-11);
    EXPECT_NEAR(
        total(trial.state().internal_energy) - total(old.internal_energy), total(sources.net_thermal_energy), 1e-5);
    expect_conservation(trial.diagnostics());
    s->commit_low_mach_trial(trial);
}

TEST(AdaptiveVoFLowMachTest, PeriodicConductionMatchesImplicitFourierDecayIndependentlyOfTheThermodynamicHint)
{
    constexpr std::size_t nx = 8;
    constexpr double spacing = 0.001, amplitude = 10.0, dt = 0.2;
    Meshes::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {nx, 2, 2};
    grid.extent = {nx * spacing, 2 * spacing, 2 * spacing};
    grid.periodic = {true, true, true};
    grid.max_level = 0;
    const auto hierarchy = std::make_shared<const Meshes::CartesianAMR>(grid);
    VoFMaterial material;
    material.liquid_density = 1000.0;
    material.gas_density = 0.1;
    material.liquid_viscosity = material.gas_viscosity = material.surface_tension = 0.0;
    Solver flow(hierarchy, material);
    auto parameters = liquid_parameters();
    parameters.thermal_expansion = 0.0;
    parameters.thermal_conductivity = 0.5;
    auto options = thermodynamics();
    options.liquid = VoF::make_linear_compressible_liquid(parameters);
    auto init = initial(flow, 0.0);
    std::vector<double> shape(init.temperature.size());
    for (std::size_t c = 0; c < shape.size(); ++c)
    {
        const double x = flow.mesh()->cell_centroid(static_cast<LO>(c)).x;
        shape[c] = std::sin(2.0 * std::numbers::pi * x / grid.extent[0]);
        init.temperature[c] += amplitude * shape[c];
    }
    flow.initialize_low_mach(options, init);
    ASSERT_LE(dt, flow.stable_timestep());
    const Snapshot accepted(flow);
    auto hint = [&](double trial_amplitude)
    {
        auto state = accepted.state;
        for (std::size_t c = 0; c < shape.size(); ++c)
        {
            state.temperature[c] = initial_temperature + trial_amplitude * shape[c];
            const auto liquid = options.liquid->evaluate(state.temperature[c], state.liquid_absolute_pressure[c]);
            state.liquid_density[c] = liquid.density;
            state.internal_energy[c] = state.liquid_mass[c] * liquid.internal_energy;
            state.resolved_density[c] = options.hydrogen.molar_mass * state.resolved_absolute_pressure[c] /
                                        (options.gas_constant * state.temperature[c]);
        }
        return state;
    };
    // Both guesses obey the supplied EOS and have the same total energy as
    // the accepted state, but neither has the final local thermal profile.
    const auto low_hint = hint(0.5 * amplitude), high_hint = hint(1.5 * amplitude);
    const auto low_trial = flow.preview_low_mach_step(dt, {}, &low_hint);
    accepted.expect_equal(flow);
    const auto high_trial = flow.preview_low_mach_step(dt, {}, &high_hint);
    accepted.expect_equal(flow);
    const double wave = 2.0 * std::sin(std::numbers::pi / nx) / spacing;
    const double decay = dt * parameters.thermal_conductivity * wave * wave /
                         (parameters.reference_density * parameters.isobaric_heat_capacity);
    const double expected_amplitude = amplitude / (1.0 + decay);
    // The discrete implicit and explicit amplitudes differ by about .0021 K;
    // this gate resolves the time-level error without a continuum truncation fit.
    ASSERT_GT(std::abs(expected_amplitude - amplitude * (1.0 - decay)), 0.001);
    double local_norm = 0.0;
    for (std::size_t c = 0; c < shape.size(); ++c)
        local_norm += shape[c] * shape[c] * flow.mesh()->cell_volume(static_cast<LO>(c));
    const double norm = global_sum(local_norm);
    for (const auto* trial : {&low_trial, &high_trial})
    {
        double local_amplitude = 0.0;
        for (std::size_t c = 0; c < shape.size(); ++c)
        {
            const auto& state = trial->state();
            const double volume = flow.mesh()->cell_volume(static_cast<LO>(c));
            EXPECT_NEAR(state.temperature[c], initial_temperature + expected_amplitude * shape[c], 5e-7);
            EXPECT_NEAR(state.liquid_mass[c], accepted.state.liquid_mass[c], 2e-14);
            EXPECT_NEAR(state.liquid_fraction[c], 1.0, 2e-11);
            EXPECT_DOUBLE_EQ(state.resolved_fraction[c], 0.0);
            for (double component : trial->velocity()[c])
                EXPECT_NEAR(component, 0.0, 1e-10);
            local_amplitude += (state.temperature[c] - initial_temperature) * shape[c] * volume;
        }
        EXPECT_NEAR(global_sum(local_amplitude) / norm, expected_amplitude, 5e-7);
        EXPECT_NEAR(total(trial->state().liquid_mass), total(accepted.state.liquid_mass), 2e-14);
        EXPECT_NEAR(total(trial->state().internal_energy), total(accepted.state.internal_energy), 1e-10);
        EXPECT_NEAR(trial->state().thermodynamic_pressure, initial_pressure, 0.002);
        EXPECT_NEAR(trial->diagnostics().pressure_work, 0.0, 1e-10);
        expect_conservation(trial->diagnostics());
    }
    for (std::size_t c = 0; c < shape.size(); ++c)
        EXPECT_NEAR(low_trial.state().temperature[c], high_trial.state().temperature[c], 5e-7);
    flow.commit_low_mach_trial(low_trial);
    EXPECT_DOUBLE_EQ(flow.time(), dt);
}

TEST(AdaptiveVoFLowMachTest, IsothermalChemicalTransferBalancesLiquidMassDonorHydrogenAndClosedVolume)
{
    auto s = solver();
    const auto options = thermodynamics();
    auto init = initial(*s, 0.2);
    for (std::size_t c = 0; c < init.temperature.size(); ++c)
        init.donor_hydrogen_moles.push_back(2.0 * s->mesh()->cell_volume(static_cast<LO>(c)));
    s->initialize_low_mach(options, init);
    const auto old = s->low_mach_state();
    const auto& p = liquid_parameters();
    const double density_moles = 0.2 * initial_pressure / (options.gas_constant * initial_temperature) + 1.0;
    const double density_liquid_mass = 0.8 * p.reference_density - options.hydrogen.molar_mass;
    auto occupied = [&](double pressure)
    {
        const double specific_volume = (1.0 - p.compressibility * (pressure - initial_pressure)) / p.reference_density;
        return density_liquid_mass * specific_volume +
               density_moles * options.gas_constant * initial_temperature / pressure;
    };
    double lower = initial_pressure, upper = p.bounds.maximum_pressure;
    for (int i = 0; i < 80; ++i)
    {
        const double midpoint = 0.5 * (lower + upper);
        if (occupied(midpoint) > 1.0)
            lower = midpoint;
        else
            upper = midpoint;
    }
    const double expected_pressure = 0.5 * (lower + upper);
    const auto liquid = options.liquid->evaluate(initial_temperature, expected_pressure);
    VoF::LowMachSources sources;
    for (std::size_t c = 0; c < old.temperature.size(); ++c)
    {
        const double added = s->mesh()->cell_volume(static_cast<LO>(c));
        sources.resolved_hydrogen_change.push_back(added);
        sources.donor_hydrogen_change.push_back(-added);
        sources.liquid_mass_change.push_back(-options.hydrogen.molar_mass * added);
        sources.produced_hydrogen.push_back(added);
        const double target_energy =
            (old.liquid_mass[c] - options.hydrogen.molar_mass * added) * liquid.internal_energy +
            (old.resolved_hydrogen_moles[c] + added) * options.hydrogen.reference_internal_energy;
        sources.net_thermal_energy.push_back(target_energy - old.internal_energy[c]);
    }
    const auto trial = s->preview_low_mach_step(0.001, sources);
    EXPECT_NEAR(trial.state().thermodynamic_pressure, expected_pressure, 0.002);
    for (std::size_t c = 0; c < old.temperature.size(); ++c)
    {
        EXPECT_NEAR(trial.state().temperature[c], initial_temperature, 5e-7);
        EXPECT_NEAR(trial.state().liquid_fraction[c] + trial.state().resolved_fraction[c], 1.0, 2e-9);
    }
    EXPECT_NEAR(total(trial.state().resolved_hydrogen_moles) - total(old.resolved_hydrogen_moles),
        total(sources.resolved_hydrogen_change), 2e-10);
    EXPECT_NEAR(total(trial.state().donor_hydrogen_moles) - total(old.donor_hydrogen_moles),
        total(sources.donor_hydrogen_change), 2e-10);
    expect_conservation(trial.diagnostics());
    s->commit_low_mach_trial(trial);
}

TEST(AdaptiveVoFLowMachTest, RankLocalInvalidOrUnbalancedSourcesRejectWithoutPublishing)
{
    auto s = solver();
    s->initialize_low_mach(thermodynamics(), initial(*s, 1.0));
    const Snapshot accepted(*s);
    const auto n = s->mesh()->num_owned_cells();
    const bool first_rank = Tpetra::getDefaultComm()->getRank() == 0;
    EXPECT_THROW(s->register_inventory(
                     {"legacy_mixture_mass", VoF::InventoryCarrier::MixtureMass, std::vector<double>(n, 0.0), true}),
        std::exception);
    EXPECT_TRUE(s->inventories().empty());
    accepted.expect_equal(*s);
    VoF::LowMachSources sources;
    sources.net_thermal_energy.assign(n, 0.0);
    if (first_rank && n)
        sources.net_thermal_energy[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) s->preview_low_mach_step(0.001, sources), std::exception);
    accepted.expect_equal(*s);
    sources = {};
    sources.net_thermal_energy.assign(n, 0.0);
    if (first_rank)
        sources.net_thermal_energy.push_back(0.0);
    EXPECT_THROW((void) s->preview_low_mach_step(0.001, sources), std::exception);
    accepted.expect_equal(*s);
    sources = {};
    sources.resolved_hydrogen_change.assign(n, 0.0);
    if (first_rank && n)
        sources.resolved_hydrogen_change[0] = -2.0 * accepted.state.resolved_hydrogen_moles[0];
    EXPECT_THROW((void) s->preview_low_mach_step(0.001, sources), std::exception);
    accepted.expect_equal(*s);
    sources = {};
    sources.enthalpy_transfer.assign(n, 0.0);
    if (first_rank && n)
        sources.enthalpy_transfer[0] = 1.0;
    EXPECT_THROW((void) s->preview_low_mach_step(0.001, sources), std::exception);
    accepted.expect_equal(*s);
    sources = {};
    sources.surface_tension.assign(n, 1.0);
    EXPECT_THROW((void) s->preview_low_mach_step(0.001, sources), std::exception);
    accepted.expect_equal(*s);
}

TEST(AdaptiveVoFLowMachTest, FailedCaloricIterationRetainsTheAcceptedState)
{
    auto s = solver();
    auto options = thermodynamics();
    options.maximum_thermodynamic_iterations = 1;
    s->initialize_low_mach(options, initial(*s, 1.0));
    const Snapshot accepted(*s);
    VoF::LowMachSources sources;
    for (double moles : accepted.state.resolved_hydrogen_moles)
        sources.net_thermal_energy.push_back(moles * options.hydrogen.molar_heat_capacity_v * 15.0);
    EXPECT_THROW((void) s->preview_low_mach_step(0.001, sources), std::exception);
    accepted.expect_equal(*s);
}

TEST(AdaptiveVoFLowMachTest, RegriddingPreservesThermalAndSpeciesInventoriesAndRejectsStaleTrials)
{
    auto s = solver(false, true);
    const auto options = thermodynamics();
    auto init = initial(*s, 0.0, 0.25);
    VoF::DispersedPopulation population;
    population.name = "carrier_bubbles";
    init.velocity.assign(init.temperature.size(), VoF::Vector{0.0, 0.125, 0.0});
    for (std::size_t c = 0; c < init.temperature.size(); ++c)
    {
        const auto box = s->hierarchy()->cell_box(s->mesh()->cell_global_id(static_cast<LO>(c)));
        const double fraction = std::clamp((0.4 - box.lower[0]) / (box.upper[0] - box.lower[0]), 0.0, 1.0);
        init.resolved_fraction[c] = fraction;
        const double carrier = (1.0 - fraction) * s->mesh()->cell_volume(static_cast<LO>(c));
        population.number.push_back(1e5 * carrier);
        population.hydrogen_moles.push_back(1e-3 * carrier);
    }
    init.dispersed.push_back(std::move(population));
    s->initialize_low_mach(options, init);
    const auto old = s->low_mach_state();
    const auto momentum = [](const VoF::LowMachState& state)
    {
        VoF::Vector result{};
        for (const auto& value : state.momentum)
            for (std::size_t a = 0; a < 3; ++a)
                result[a] += value[a];
        for (double& value : result)
            value = global_sum(value);
        return result;
    };
    const auto old_momentum = momentum(old);
    const auto trial = s->preview_low_mach_step(0.001, {});
    const auto generation = s->topology_generation();
    s->regrid({1});
    EXPECT_GT(s->topology_generation(), generation);
    const auto& mapped = s->low_mach_state();
    EXPECT_NEAR(total(mapped.liquid_mass), total(old.liquid_mass), 1e-9);
    EXPECT_NEAR(total(mapped.resolved_hydrogen_moles), total(old.resolved_hydrogen_moles), 2e-11);
    EXPECT_NEAR(total(mapped.resolved_inert_moles), total(old.resolved_inert_moles), 2e-11);
    EXPECT_NEAR(total(mapped.dispersed[0].number), total(old.dispersed[0].number), 1e-7);
    EXPECT_NEAR(total(mapped.dispersed[0].hydrogen_moles), total(old.dispersed[0].hydrogen_moles), 2e-14);
    EXPECT_NEAR(total(mapped.internal_energy), total(old.internal_energy), 1e-6);
    EXPECT_NEAR(mapped.thermodynamic_pressure, initial_pressure, 5e-4);
    const auto mapped_momentum = momentum(mapped);
    for (std::size_t a = 0; a < 3; ++a)
        EXPECT_NEAR(mapped_momentum[a], old_momentum[a], 1e-9);
    for (std::size_t c = 0; c < mapped.temperature.size(); ++c)
    {
        const double volume = s->mesh()->cell_volume(static_cast<LO>(c));
        const double carrier = (1.0 - s->gas_fraction()[c]) * volume;
        EXPECT_NEAR(mapped.temperature[c], initial_temperature, 5e-7);
        EXPECT_NEAR(s->velocity()[c][0], 0.0, 1e-9);
        EXPECT_NEAR(s->velocity()[c][1], 0.125, 1e-9);
        EXPECT_NEAR(s->velocity()[c][2], 0.0, 1e-9);
        EXPECT_NEAR(mapped.dispersed[0].number[c], 1e5 * carrier, 2e-7);
        EXPECT_NEAR(mapped.liquid_mass[c], 1000.0 * (carrier - mapped.dispersed[0].volume[c]), 2e-8);
        EXPECT_NEAR(mapped.resolved_fraction[c] + mapped.dispersed_fraction[c] + mapped.liquid_fraction[c], 1.0, 2e-9);
    }
    const Snapshot remapped(*s);
    EXPECT_THROW(s->commit_low_mach_trial(trial), std::exception);
    remapped.expect_equal(*s);
}

VoF::LowMachInitialState translating_population(const Solver& flow, bool nonuniform, bool counterflow)
{
    const auto options = thermodynamics();
    auto state = initial(flow, 0.0);
    VoF::DispersedPopulation population;
    population.name = "fixed_molar_pivot";
    constexpr double pivot = 1e-8, speed = 0.125;
    for (std::size_t c = 0; c < state.temperature.size(); ++c)
    {
        const auto box = flow.hierarchy()->cell_box(flow.mesh()->cell_global_id(static_cast<LO>(c)));
        const double k = 2.0 * std::numbers::pi;
        const double average_sine =
            (std::cos(k * box.lower[0]) - std::cos(k * box.upper[0])) / (k * (box.upper[0] - box.lower[0]));
        const double alpha = 0.05 * (1.0 + (nonuniform ? 0.2 * average_sine : 0.0));
        const double moles = alpha * flow.mesh()->cell_volume(static_cast<LO>(c)) * initial_pressure /
                             (options.gas_constant * initial_temperature);
        population.number.push_back(moles / pivot);
        population.hydrogen_moles.push_back(moles);
        population.drift_velocity.push_back({counterflow ? -speed : 0.0, 0.0, 0.0});
        state.velocity.push_back({speed, 0.0, 0.0});
    }
    state.dispersed.push_back(std::move(population));
    return state;
}

VoF::Vector physical_momentum(const VoF::LowMachState& state)
{
    VoF::Vector result{};
    for (const auto& value : state.momentum)
        for (std::size_t a = 0; a < 3; ++a)
            result[a] += value[a];
    for (double& value : result)
        value = global_sum(value);
    return result;
}

TEST(AdaptiveVoFLowMachTest, CounterflowRetainsSlipVolumeWhenTheNetPopulationFluxCancels)
{
    auto s = solver(true, true);
    ASSERT_TRUE(std::any_of(s->hierarchy()->blocks().begin(), s->hierarchy()->blocks().end(),
        [](const auto& block) { return block.level == 0; }));
    ASSERT_TRUE(std::any_of(s->hierarchy()->blocks().begin(), s->hierarchy()->blocks().end(),
        [](const auto& block) { return block.level == 1; }));
    const auto options = thermodynamics();
    s->initialize_low_mach(options, translating_population(*s, false, true));
    const auto old = s->low_mach_state();
    const Snapshot accepted(*s);
    constexpr double dt = 1.0 / 128.0, speed = 0.125, alpha = 0.05;
    const auto trial = s->preview_low_mach_step(dt, {});
    accepted.expect_equal(*s);
    auto hint = old;
    // An internally consistent alternative guess changes the molar-volume
    // coefficient without changing this source-free accepted-old problem.
    for (std::size_t c = 0; c < hint.temperature.size(); ++c)
    {
        hint.temperature[c] = 315.0;
        const auto liquid = options.liquid->evaluate(hint.temperature[c], hint.liquid_absolute_pressure[c]);
        hint.liquid_density[c] = liquid.density;
        hint.liquid_mass[c] = hint.liquid_fraction[c] * s->mesh()->cell_volume(static_cast<LO>(c)) * liquid.density;
        hint.resolved_density[c] = options.hydrogen.molar_mass * hint.resolved_absolute_pressure[c] /
                                   (options.gas_constant * hint.temperature[c]);
        auto& population = hint.dispersed[0];
        population.hydrogen_moles[c] =
            population.volume[c] * hint.liquid_absolute_pressure[c] / (options.gas_constant * hint.temperature[c]);
        population.number[c] = population.hydrogen_moles[c] / 1e-8;
        population.radius[c] = std::cbrt(3.0 * population.volume[c] / (4.0 * std::numbers::pi * population.number[c]));
        const double molar_energy =
            options.hydrogen.reference_internal_energy +
            options.hydrogen.molar_heat_capacity_v * (hint.temperature[c] - options.hydrogen.reference_temperature);
        hint.internal_energy[c] =
            hint.liquid_mass[c] * liquid.internal_energy + population.hydrogen_moles[c] * molar_energy;
        hint.momentum[c] = {hint.liquid_mass[c] * speed, 0.0, 0.0}; // Dispersed laboratory velocity is zero.
    }
    const auto alternate_trial = s->preview_low_mach_step(dt, {}, &hint);
    accepted.expect_equal(*s);
    ASSERT_EQ(trial.transport().dispersed_slip_volume.size(), 1u);
    ASSERT_EQ(trial.transport().dispersed_hydrogen.size(), 1u);
    ASSERT_EQ(alternate_trial.transport().dispersed_slip_volume.size(), 1u);
    double tested_faces = 0.0;
    const auto& mesh = *s->mesh();
    for (std::size_t id = 0; id < mesh.num_owned_faces(); ++id)
    {
        const auto face = static_cast<LO>(id);
        if (mesh.is_geometry_exterior_face(face))
            continue;
        const auto normal = mesh.face_normal(face);
        if (std::abs(normal.x) < 0.5)
            continue;
        const double advected_volume = dt * mesh.face_area(face) * speed * normal.x;
        EXPECT_NEAR(trial.transport().carrier_volume[id], advected_volume, 2e-10 * std::abs(advected_volume));
        EXPECT_NEAR(trial.transport().dispersed_hydrogen[0][id], 0.0, 1e-13);
        EXPECT_NEAR(trial.transport().dispersed_slip_volume[0][id], -alpha * advected_volume,
            2e-10 * alpha * std::abs(advected_volume));
        EXPECT_NEAR(alternate_trial.transport().dispersed_slip_volume[0][id], -alpha * advected_volume,
            2e-10 * alpha * std::abs(advected_volume));
        EXPECT_NEAR(alternate_trial.transport().dispersed_slip_volume[0][id],
            trial.transport().dispersed_slip_volume[0][id], 2e-10 * alpha * std::abs(advected_volume));
        tested_faces += 1.0;
    }
    EXPECT_GT(global_sum(tested_faces), 0.0);
    for (std::size_t c = 0; c < old.temperature.size(); ++c)
    {
        EXPECT_NEAR(trial.state().dispersed[0].hydrogen_moles[c], old.dispersed[0].hydrogen_moles[c], 1e-12);
        EXPECT_NEAR(trial.velocity()[c][0], speed, 1e-9);
        EXPECT_NEAR(trial.state().temperature[c], initial_temperature, 5e-7);
        EXPECT_NEAR(alternate_trial.state().temperature[c], initial_temperature, 5e-7);
        EXPECT_NEAR(alternate_trial.state().dispersed[0].hydrogen_moles[c], old.dispersed[0].hydrogen_moles[c], 1e-12);
    }
    const auto before_momentum = physical_momentum(old), after_momentum = physical_momentum(trial.state());
    for (std::size_t a = 0; a < 3; ++a)
        EXPECT_NEAR(after_momentum[a], before_momentum[a], 1e-9);
    EXPECT_NEAR(total(trial.state().internal_energy), total(old.internal_energy), 1e-6);
    EXPECT_NEAR(trial.diagnostics().pressure_work, 0.0, 1e-6);
    expect_conservation(trial.diagnostics());
    const auto alternate_momentum = physical_momentum(alternate_trial.state());
    for (std::size_t a = 0; a < 3; ++a)
        EXPECT_NEAR(alternate_momentum[a], before_momentum[a], 1e-9);
    EXPECT_NEAR(total(alternate_trial.state().internal_energy), total(old.internal_energy), 1e-6);
    EXPECT_NEAR(alternate_trial.diagnostics().pressure_work, 0.0, 1e-6);
    expect_conservation(alternate_trial.diagnostics());
}

TEST(AdaptiveVoFLowMachTest, PeriodicMixtureAdvectionPreservesTheFixedPivotAndPhysicalMomentum)
{
    auto s = solver(true, true);
    ASSERT_TRUE(std::any_of(s->hierarchy()->blocks().begin(), s->hierarchy()->blocks().end(),
        [](const auto& block) { return block.level == 0; }));
    ASSERT_TRUE(std::any_of(s->hierarchy()->blocks().begin(), s->hierarchy()->blocks().end(),
        [](const auto& block) { return block.level == 1; }));
    s->initialize_low_mach(thermodynamics(), translating_population(*s, true, false));
    const auto old = s->low_mach_state();
    constexpr double dt = 1.0 / 128.0, pivot = 1e-8;
    const auto trial = s->preview_low_mach_step(dt, {});
    double local_change = 0.0;
    for (std::size_t c = 0; c < old.temperature.size(); ++c)
    {
        const double moles = trial.state().dispersed[0].hydrogen_moles[c];
        const double from_number = pivot * trial.state().dispersed[0].number[c];
        EXPECT_NEAR(moles, from_number, 2e-9 * moles);
        local_change += std::abs(moles - old.dispersed[0].hydrogen_moles[c]);
        EXPECT_NEAR(trial.velocity()[c][0], 0.125, 2e-8);
        EXPECT_NEAR(trial.state().temperature[c], initial_temperature, 5e-7);
    }
    const double old_moles = total(old.dispersed[0].hydrogen_moles);
    EXPECT_GT(global_sum(local_change), 1e-5 * old_moles);
    EXPECT_NEAR(total(trial.state().dispersed[0].hydrogen_moles), old_moles, 2e-10 * old_moles);
    EXPECT_NEAR(total(trial.state().dispersed[0].number), total(old.dispersed[0].number),
        2e-10 * total(old.dispersed[0].number));
    EXPECT_NEAR(total(trial.state().liquid_mass), total(old.liquid_mass), 1e-9);
    const auto before_momentum = physical_momentum(old), after_momentum = physical_momentum(trial.state());
    for (std::size_t a = 0; a < 3; ++a)
        EXPECT_NEAR(after_momentum[a], before_momentum[a], 1e-9);
    EXPECT_NEAR(total(trial.state().internal_energy), total(old.internal_energy), 1e-6);
    EXPECT_NEAR(trial.diagnostics().pressure_work, 0.0, 1e-6);
    expect_conservation(trial.diagnostics());
    s->commit_low_mach_trial(trial);
}

TEST(AdaptiveVoFLowMachTest, SourceFreeConstantDensityLimitPreservesPassiveTransverseMomentumBounds)
{
    Meshes::AMRGrid grid;
    grid.root_blocks = {2, 2, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4.0, 4.0, 200.0};
    grid.periodic = {true, true, true};
    VoFMaterial material;
    material.liquid_density = 1000.0;
    material.gas_density = 1.0;
    material.liquid_viscosity = material.gas_viscosity = material.surface_tension = 0.0;
    Solver flow(std::make_shared<const Meshes::CartesianAMR>(grid), material);
    auto options = thermodynamics();
    options.liquid =
        std::make_shared<const VoF::LiquidThermodynamics>(VoF::ThermodynamicBounds{250.0, 400.0, 10000.0, 1000000.0},
            [](double temperature, double pressure)
            {
                constexpr double rho = 1000.0, cv = 4000.0;
                const double energy = 100000.0 + cv * (temperature - initial_temperature);
                return VoF::LiquidThermodynamicPoint{rho, energy, energy + pressure / rho, cv, 0.0, 0.0};
            });
    // Equal gas species masses give rho_g=1 at the fixed isothermal state.
    options.hydrogen.molar_mass = options.inert.molar_mass =
        options.gas_constant * initial_temperature / initial_pressure;
    auto state = initial(flow, 1.0);
    state.velocity.assign(state.temperature.size(), {1.0, 1.0, 0.0});
    for (std::size_t c = 0; c < state.temperature.size(); ++c)
    {
        const auto x = flow.mesh()->cell_centroid(static_cast<LO>(c));
        state.resolved_fraction[c] = x.x < 1.0 ? 0.0 : 1.0;
        state.velocity[c][2] = x.x >= 1.0 && x.x < 2.0 && x.y >= 1.0 && x.y < 2.0 ? 1.0 : 0.0;
    }
    flow.initialize_low_mach(options, state);
    const auto old = flow.low_mach_state();
    const auto before = physical_momentum(old);
    const double before_mass =
        total(old.liquid_mass) + options.hydrogen.molar_mass * total(old.resolved_hydrogen_moles);
    const auto trial = flow.preview_low_mach_step(0.1, {});
    double minimum = 1.0, maximum = 0.0;
    for (std::size_t c = 0; c < state.temperature.size(); ++c)
    {
        minimum = std::min(minimum, trial.velocity()[c][2]);
        maximum = std::max(maximum, trial.velocity()[c][2]);
        // Inspect the transport ledger independently of pressure projection.
        double transported_momentum = old.momentum[c][2];
        flow.mesh()->visit_cell_faces(static_cast<LO>(c),
            [&](LO face)
            {
                transported_momentum -= (flow.mesh()->owner_cell(face) == static_cast<LO>(c) ? 1.0 : -1.0) *
                                        trial.transport().momentum[face][2];
            });
        const double transported_mass =
            trial.state().liquid_mass[c] + options.hydrogen.molar_mass * trial.state().resolved_hydrogen_moles[c];
        EXPECT_GE(transported_momentum / transported_mass, -2e-10);
        EXPECT_LE(transported_momentum / transported_mass, 1.0 + 2e-10);
        EXPECT_NEAR(trial.state().temperature[c], initial_temperature, 2e-7);
        EXPECT_NEAR(trial.state().liquid_density[c], 1000.0, 1e-10);
        EXPECT_NEAR(trial.state().resolved_density[c], 1.0, 2e-9);
    }
    double global_minimum = 0.0, global_maximum = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &minimum, &global_minimum);
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &maximum, &global_maximum);
    EXPECT_GE(global_minimum, -2e-10);
    EXPECT_LE(global_maximum, 1.0 + 2e-10);
    const auto after = physical_momentum(trial.state());
    for (std::size_t a = 0; a < 3; ++a)
        EXPECT_NEAR(after[a], before[a], 2e-10 * std::max(1.0, std::abs(before[a])));
    const double mass_error = total(trial.state().liquid_mass) +
                              options.hydrogen.molar_mass * total(trial.state().resolved_hydrogen_moles) - before_mass;
    EXPECT_NEAR(mass_error, 0.0, 2e-10 * before_mass);
    expect_conservation(trial.diagnostics());
    flow.commit_low_mach_trial(trial);
    EXPECT_EQ(flow.velocity(), trial.velocity());
    EXPECT_DOUBLE_EQ(flow.time(), 0.1);
    for (std::size_t c = 0; c < flow.velocity().size(); ++c)
        EXPECT_DOUBLE_EQ(flow.velocity_field().value(static_cast<LO>(c)).z, trial.velocity()[c][2]);
    if (Tpetra::getDefaultComm()->getRank() == 0)
        std::cout << std::setprecision(17) << "low_mach_passive_momentum,minimum=" << global_minimum
                  << ",maximum=" << global_maximum << ",mass_error=" << mass_error
                  << ",transverse_momentum_error=" << after[2] - before[2] << '\n';
}

TEST(AdaptiveVoFLowMachTest, DepletedPopulationUsesPostSourceSupportForSlipAndTimestep)
{
    // Each equal-width z pair begins with numbers [10,10]. The source removes
    // the upper population completely and supplies evaluated drifts [0.02,0].
    // Periodicity gives both incident faces the same populated/empty support;
    // no wall-face drift can conceal the source timestep bound.
    Meshes::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {0.5, 0.5, 0.5};
    grid.periodic = {true, true, true};
    const auto hierarchy = std::make_shared<const Meshes::CartesianAMR>(grid);
    VoFMaterial material;
    material.liquid_density = 1000;
    material.gas_density = 0.1;
    material.liquid_viscosity = material.gas_viscosity = material.surface_tension = 0;
    Solver flow(hierarchy, material);
    auto init = initial(flow, 0);
    const auto options = thermodynamics();
    constexpr double speed = 0.02, dt = 0.01;
    const double per_bubble =
        initial_pressure * (4 * std::numbers::pi / 3) * 1e-12 / (options.gas_constant * initial_temperature);
    VoF::DispersedPopulation population;
    population.name = "depleted";
    VoF::PopulationSource particles;
    particles.name = population.name;
    VoF::LowMachSources source;
    const auto& mesh = *flow.mesh();
    for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const bool depleted = mesh.cell_centroid(static_cast<LO>(c)).z > 0.25;
        population.number.push_back(10);
        population.hydrogen_moles.push_back(10 * per_bubble);
        // The accepted drift is zero, isolating the supplied-closure guard.
        population.drift_velocity.push_back({});
        particles.number_change.push_back(depleted ? -10 : 0);
        particles.hydrogen_change.push_back(depleted ? -10 * per_bubble : 0);
        particles.drift_velocity.push_back({0, 0, depleted ? 0 : speed});
        source.dissolved_hydrogen_change.push_back(depleted ? 10 * per_bubble : 0);
        source.liquid_mass_change.push_back(depleted ? options.hydrogen.molar_mass * 10 * per_bubble : 0);
    }
    init.dispersed = {population};
    source.dispersed = {particles};
    flow.initialize_low_mach(options, init);
    const Snapshot accepted(flow);
    const auto trial = flow.preview_low_mach_step(dt, source);
    accepted.expect_equal(flow);
    VoF::AdaptiveVoF geometry(hierarchy);
    const auto moles = geometry.exchange_cells(trial.state().dispersed[0].hydrogen_moles);
    const auto temperature = geometry.exchange_cells(trial.state().temperature);
    const auto pressure = geometry.exchange_cells(trial.state().liquid_absolute_pressure);
    double measured_faces = 0, partition_faces = 0;
    for (std::size_t id = 0; id < mesh.num_owned_faces(); ++id)
    {
        const auto face = static_cast<LO>(id);
        const auto normal = mesh.face_normal(face);
        if (std::abs(normal.z) < 0.5 || mesh.face_centroid(face).z != 0.25)
            continue;
        const double relative = dt * mesh.face_area(face) * speed * normal.z;
        const double combined = trial.transport().carrier_volume[id] + relative;
        const auto donor = combined >= 0 ? mesh.owner_cell(face) : mesh.neighbor_cell(face);
        const double alpha =
            moles[donor] / mesh.cell_volume(donor) * options.gas_constant * temperature[donor] / pressure[donor];
        ASSERT_GT(alpha, 0);
        const double measured =
            trial.transport().dispersed_slip_volume[0][id] / (dt * mesh.face_area(face) * normal.z * alpha);
        EXPECT_NEAR(measured, speed, 2e-12);
        RecordProperty("depleted_front_relative_slip", measured);
        measured_faces += 1;
        if (static_cast<std::size_t>(mesh.owner_cell(face)) >= mesh.num_owned_cells() ||
            static_cast<std::size_t>(mesh.neighbor_cell(face)) >= mesh.num_owned_cells())
            partition_faces += 1;
    }
    EXPECT_GT(global_sum(measured_faces), 0);
    if (Tpetra::getDefaultComm()->getSize() == 2)
        EXPECT_GT(global_sum(partition_faces), 0);
    EXPECT_NEAR(
        total(trial.state().dispersed[0].number), total(population.number) + total(particles.number_change), 2e-10);
    EXPECT_NEAR(total(trial.state().dispersed[0].hydrogen_moles) + total(trial.state().dissolved_hydrogen_moles),
        total(population.hydrogen_moles), 2e-10 * total(population.hydrogen_moles));
    for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        EXPECT_GE(trial.state().dispersed[0].number[c], 0);
        EXPECT_GE(trial.state().dispersed[0].hydrogen_moles[c], 0);
    }
    expect_conservation(trial.diagnostics());
    // The correct source-drift limit is max_courant*h/0.02. Half-speed
    // interpolation incorrectly admits this timestep; test the guard's own
    // error so an unrelated downstream failure cannot satisfy the regression.
    const double limit = flow.options().max_courant * 0.25 / speed;
    EXPECT_NO_THROW((void) flow.preview_low_mach_step(limit, source));
    bool source_drift_rejected = false;
    try
    {
        (void) flow.preview_low_mach_step(1.5 * limit, source);
    }
    catch (const std::exception& error)
    {
        source_drift_rejected = std::string(error.what()).find("supplied population drift") != std::string::npos;
    }
    EXPECT_TRUE(source_drift_rejected);
    accepted.expect_equal(flow);
    // A retained successful candidate is unaffected by the rejected attempt.
    flow.commit_low_mach_trial(trial);
    EXPECT_EQ(flow.low_mach_state().dispersed[0].number, trial.state().dispersed[0].number);
}

TEST(AdaptiveVoFLowMachTest, EmptyOwnedRankParticipatesInInitializationPopulationSourcesAndHintGuards)
{
    Meshes::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {0.5, 0.5, 0.5};
    grid.periodic = {true, true, true};
    grid.max_level = 0;
    const auto hierarchy = std::make_shared<const Meshes::CartesianAMR>(grid);
    VoFMaterial material;
    material.liquid_density = 1000.0;
    material.gas_density = 0.1;
    material.liquid_viscosity = material.gas_viscosity = material.surface_tension = 0.0;
    Solver flow(hierarchy, material);
    const auto n = flow.mesh()->num_owned_cells();
    const auto comm = Tpetra::getDefaultComm();
    const auto ranks = static_cast<std::size_t>(comm->getSize());
    const auto rank = static_cast<std::size_t>(comm->getRank());
    EXPECT_EQ(n, 8u / ranks + (rank < 8u % ranks ? 1u : 0u));
    const double empty_ranks = global_sum(n == 0 ? 1.0 : 0.0);
    EXPECT_EQ(empty_ranks, ranks > 8 ? static_cast<double>(ranks - 8) : 0.0);
    // The dedicated nine-rank CTest is necessary: even one minimal AMR block
    // has eight cells, and native ownership partitions cells rather than blocks.
    if (ranks > 8)
        ASSERT_GT(empty_ranks, 0.0);
    auto init = initial(flow, 0.2, 0.25);
    VoF::DispersedPopulation population;
    population.name = "fixed_pivot";
    VoF::LowMachSources source;
    VoF::PopulationSource population_source;
    population_source.name = population.name;
    const auto options = thermodynamics();
    for (std::size_t c = 0; c < n; ++c)
    {
        const double volume = flow.mesh()->cell_volume(static_cast<LO>(c));
        population.number.push_back(100.0 * volume);
        population.hydrogen_moles.push_back(1e-6 * volume);
        population.drift_velocity.push_back({0.0, 0.0, 0.0});
        init.donor_hydrogen_moles.push_back(1e-4 * volume);
        population_source.number_change.push_back(volume);
        population_source.hydrogen_change.push_back(1e-8 * volume);
        population_source.drift_velocity.push_back({0.0, 0.0, 0.0});
        source.donor_hydrogen_change.push_back(-1e-8 * volume);
        source.liquid_mass_change.push_back(-options.hydrogen.molar_mass * 1e-8 * volume);
        source.produced_hydrogen.push_back(1e-8 * volume);
    }
    init.dispersed.push_back(std::move(population));
    source.dispersed.push_back(std::move(population_source));
    flow.initialize_low_mach(options, init);
    const Snapshot accepted(flow);
    const auto trial = flow.preview_low_mach_step(0.001, source);
    accepted.expect_equal(flow);
    EXPECT_NEAR(total(trial.state().dispersed[0].hydrogen_moles) - total(accepted.state.dispersed[0].hydrogen_moles),
        total(source.dispersed[0].hydrogen_change), 2e-12);
    expect_conservation(trial.diagnostics());
    flow.commit_low_mach_trial(trial);
    {
        const Snapshot committed(flow);
        VoF::LowMachSources fast_drift;
        fast_drift.dispersed.push_back({"fixed_pivot", {}, {}, std::vector<VoF::Vector>(n, {1e6, 0.0, 0.0})});
        EXPECT_THROW((void) flow.preview_low_mach_step(0.001, fast_drift), std::exception);
        committed.expect_equal(flow);
    }
    if (comm->getSize() > 1)
    {
        const Snapshot committed(flow);
        const auto* hint = comm->getRank() == 0 ? &flow.low_mach_state() : nullptr;
        EXPECT_THROW((void) flow.preview_low_mach_step(0.001, {}, hint), std::exception);
        committed.expect_equal(flow);
    }
}

TEST(AdaptiveVoFLowMachTest, CurvedInterfaceUsesOneSidedAbsolutePressuresWithoutADoubleLaplaceJump)
{
    Meshes::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {8, 8, 8};
    grid.extent = {1.0, 1.0, 1.0};
    grid.max_level = 0;
    const auto hierarchy = std::make_shared<const Meshes::CartesianAMR>(grid);
    VoFMaterial material;
    material.liquid_density = 1000.0;
    material.gas_density = 0.1;
    material.liquid_viscosity = material.gas_viscosity = 0.0;
    material.surface_tension = 0.07;
    Solver flow(hierarchy, material);
    auto init = initial(flow, 0.0, 0.35);
    // This supplied curved fraction field exercises the pressure contract.
    // It is not a sphere quadrature or a new curvature-convergence benchmark.
    for (std::size_t c = 0; c < init.temperature.size(); ++c)
    {
        const auto center = flow.mesh()->cell_centroid(static_cast<LO>(c));
        const double radius = std::sqrt((center.x - 0.5) * (center.x - 0.5) + (center.y - 0.5) * (center.y - 0.5) +
                                        (center.z - 0.5) * (center.z - 0.5));
        init.resolved_fraction[c] = std::clamp(0.5 + (0.3 - radius) / 0.125, 0.0, 1.0);
    }
    const auto options = thermodynamics();
    flow.initialize_low_mach(options, init);
    const auto& state = flow.low_mach_state();
    double mixed = 0.0, pure_gas = 0.0, curved = 0.0;
    for (std::size_t c = 0; c < init.temperature.size(); ++c)
    {
        const double fraction = init.resolved_fraction[c];
        const double base_pressure = state.thermodynamic_pressure + flow.pressure()[c];
        const double jump = material.surface_tension * flow.curvature()[c];
        const double gas_pressure = base_pressure + (1.0 - fraction) * jump;
        const double liquid_pressure = base_pressure - fraction * jump;
        EXPECT_NEAR(state.resolved_absolute_pressure[c], gas_pressure, 1e-8);
        EXPECT_NEAR(state.liquid_absolute_pressure[c], liquid_pressure, 1e-8);
        if (fraction > 0.0)
        {
            EXPECT_NEAR(state.resolved_hydrogen_partial_pressure[c], 0.35 * gas_pressure, 1e-8);
            const double expected_hydrogen = 0.35 * fraction * flow.mesh()->cell_volume(static_cast<LO>(c)) *
                                             gas_pressure / (options.gas_constant * initial_temperature);
            EXPECT_NEAR(state.resolved_hydrogen_moles[c], expected_hydrogen, 2e-11 * expected_hydrogen);
        }
        else
            EXPECT_DOUBLE_EQ(state.resolved_hydrogen_partial_pressure[c], 0.0);
        if (fraction == 1.0)
        {
            EXPECT_NEAR(state.resolved_absolute_pressure[c], base_pressure, 1e-8);
            pure_gas += 1.0;
        }
        if (fraction > 0.0 && fraction < 1.0)
        {
            mixed += 1.0;
            if (std::abs(jump) > 1e-3)
                curved += 1.0;
            EXPECT_NEAR(state.resolved_absolute_pressure[c] - state.liquid_absolute_pressure[c], jump, 2e-8);
        }
    }
    EXPECT_GT(global_sum(mixed), 0.0);
    EXPECT_GT(global_sum(pure_gas), 0.0);
    EXPECT_GT(global_sum(curved), 0.0);
}
TEST(AdaptiveVoFLowMachTest, ThermallyContractedCarrierSurvivesPureFractionRefinementAndReplay)
{
    constexpr VoF::Vector acceleration{0.0, 1.0, 0.0};
    auto flow = solver(false, true, acceleration);
    auto options = thermodynamics();
    auto parameters = liquid_parameters();
    parameters.bounds.minimum_temperature = 270.0;
    parameters.thermal_expansion = 0.025;
    parameters.isobaric_heat_capacity = 1e8;
    parameters.reference_internal_energy = 1e10;
    options.liquid = VoF::make_linear_compressible_liquid(parameters);
    // v/v0=1+alpha*(270-300)=.25 before the small pressure correction;
    // Cp-T*v0*alpha^2/beta remains positive throughout the declared range.
    auto init = initial(*flow, std::nextafter(1.0, 0.0));
    VoF::DispersedPopulation population;
    population.name = "retained_bubbles";
    constexpr double pivot = 1e-12;
    const double initial_carrier_fraction = 1.0 - std::nextafter(1.0, 0.0);
    for (std::size_t c = 0; c < init.temperature.size(); ++c)
    {
        const double dispersed_volume = 0.05 * initial_carrier_fraction * flow->mesh()->cell_volume(static_cast<LO>(c));
        const double moles = dispersed_volume * initial_pressure / (options.gas_constant * initial_temperature);
        population.number.push_back(moles / pivot);
        population.hydrogen_moles.push_back(moles);
    }
    init.dispersed.push_back(std::move(population));
    flow->initialize_low_mach(options, init);
    const auto before = flow->low_mach_state();
    VoF::LowMachSources cooling;
    for (std::size_t c = 0; c < before.temperature.size(); ++c)
    {
        constexpr double target_temperature = 270.0;
        const auto liquid = options.liquid->evaluate(target_temperature, initial_pressure);
        const double gas_moles = before.resolved_hydrogen_moles[c] + before.dispersed[0].hydrogen_moles[c];
        const double target_energy =
            before.liquid_mass[c] * liquid.internal_energy +
            gas_moles * (options.hydrogen.reference_internal_energy +
                            options.hydrogen.molar_heat_capacity_v * (target_temperature - initial_temperature));
        cooling.net_thermal_energy.push_back(target_energy - before.internal_energy[c]);
    }
    // Uniform periodic body acceleration forces genuine velocity correctors,
    // with identical phase/scalar rows and balanced incoming/outgoing carrier
    // flux in every cell. Cooling alone can converge before native f changes.
    constexpr double dt = 1e-3;
    const auto thermal = flow->preview_low_mach_step(dt, cooling);
    flow->commit_low_mach_trial(thermal);
    const auto contracted = flow->low_mach_state();
    const auto absent_carrier = [](const Solver& s)
    {
        std::array<double, 3> amounts{};
        for (std::size_t c = 0; c < s.gas_fraction().size(); ++c)
            if (std::max(0.0, 1.0 - s.gas_fraction()[c]) == 0.0)
            {
                amounts[0] += s.low_mach_state().liquid_mass[c];
                amounts[1] += s.low_mach_state().dispersed[0].number[c];
                amounts[2] += s.low_mach_state().dispersed[0].hydrogen_moles[c];
            }
        for (double& amount : amounts)
            amount = global_sum(amount);
        return amounts;
    };
    const auto retained = absent_carrier(*flow);
    ASSERT_GT(retained[0], 0.0);
    ASSERT_GT(retained[1], 0.0);
    ASSERT_GT(retained[2], 0.0);
    EXPECT_NEAR(total(contracted.liquid_mass), total(before.liquid_mass), 2e-9 * total(before.liquid_mass));
    EXPECT_NEAR(total(contracted.dispersed[0].number), total(before.dispersed[0].number),
        2e-9 * total(before.dispersed[0].number));
    EXPECT_NEAR(total(contracted.internal_energy) - total(before.internal_energy),
        total(cooling.net_thermal_energy) + thermal.diagnostics().pressure_work, 1e-7);
    const auto old_momentum = physical_momentum(contracted);
    const auto initial_momentum = physical_momentum(before);
    for (std::size_t a = 0; a < 3; ++a)
    {
        const double impulse = dt * thermal.diagnostics().total_mass * acceleration[a];
        EXPECT_NEAR(old_momentum[a] - initial_momentum[a], impulse, 5e-10 * std::max(std::abs(impulse), 1e-20));
    }
    flow->regrid({1});
    const auto mapped = flow->low_mach_state();
    const auto retained_after = absent_carrier(*flow);
    for (std::size_t i = 0; i < retained.size(); ++i)
        EXPECT_GT(retained_after[i], 0.0);
    // Native absence is a geometric classification, not a persistent material
    // tag. Global primary inventories are the conserved remap quantities.
    EXPECT_NEAR(total(mapped.liquid_mass), total(contracted.liquid_mass), 2e-9 * total(contracted.liquid_mass));
    EXPECT_NEAR(total(mapped.resolved_hydrogen_moles), total(contracted.resolved_hydrogen_moles),
        2e-11 * total(contracted.resolved_hydrogen_moles));
    EXPECT_NEAR(total(mapped.dispersed[0].number), total(contracted.dispersed[0].number),
        2e-9 * total(contracted.dispersed[0].number));
    EXPECT_NEAR(total(mapped.dispersed[0].hydrogen_moles), total(contracted.dispersed[0].hydrogen_moles),
        2e-9 * total(contracted.dispersed[0].hydrogen_moles));
    EXPECT_NEAR(total(mapped.internal_energy), total(contracted.internal_energy), 1e-7);
    const auto new_momentum = physical_momentum(mapped);
    for (std::size_t a = 0; a < 3; ++a)
        EXPECT_NEAR(new_momentum[a], old_momentum[a], 1e-12);
    const auto checkpoint = flow->checkpoint();
    const auto next = flow->preview_low_mach_step(dt, {});
    flow->commit_low_mach_trial(next);
    EXPECT_NEAR(total(flow->low_mach_state().liquid_mass), total(mapped.liquid_mass), 2e-9 * total(mapped.liquid_mass));
    const auto advanced_momentum = physical_momentum(flow->low_mach_state());
    for (std::size_t a = 0; a < 3; ++a)
    {
        const double impulse = dt * next.diagnostics().total_mass * acceleration[a];
        EXPECT_NEAR(advanced_momentum[a] - new_momentum[a], impulse, 5e-10 * std::max(std::abs(impulse), 1e-20));
    }
    EXPECT_GT(absent_carrier(*flow)[0], 0.0);
    flow->restore(checkpoint);
    expect_same_state(flow->low_mach_state(), mapped);
    const auto replay = flow->preview_low_mach_step(dt, {});
    expect_same_state(replay.state(), next.state());
    flow->commit_low_mach_trial(replay);
    flow->release_checkpoint(checkpoint);
}

} // namespace
