#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>

namespace
{
namespace R = SimpleFluid::Reacting;
using namespace SimpleFluid;
using Solver = R::LowMachMixtureSolver;
using Coupler = R::ReactingMixtureCoupler;
using LO = Solver::Pack::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr double gas_constant = 8.31446261815324;
constexpr R::SpeciesId donor = 1, liquid_product = 2, first_gas = 17, second_gas = 81;

SP<Solver::Mesh> mesh()
{
    Solver::Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    return std::make_shared<Solver::Mesh>(std::make_shared<Meshes::OrthogonalCartesian3D>(
                                              Vec3D<ArrReal>{ArrReal{0, 1}, ArrReal{0, 1}, ArrReal{0, 0.5, 1}}),
        distribution);
}
std::shared_ptr<const R::SpeciesCatalog> catalog()
{
    const std::vector liquid{R::SpeciesPhase::Liquid};
    const std::vector gas{R::SpeciesPhase::Dissolved, R::SpeciesPhase::Resolved, R::SpeciesPhase::Dispersed};
    return std::make_shared<const R::SpeciesCatalog>(std::vector<R::SpeciesDefinition>{
        {donor, "A", .10, {{"X", 4}}, liquid}, {liquid_product, "B", .04, {{"X", 2}}, liquid},
        {first_gas, "G1", .02, {{"X", 1}}, gas}, {second_gas, "G2", .04, {{"X", 1}}, gas}});
}
struct Fixture
{
    SP<Solver::Mesh> grid = mesh();
    std::shared_ptr<const R::SpeciesCatalog> species = catalog();
    std::shared_ptr<const R::MixtureThermodynamics> liquid, gas;
    std::shared_ptr<const R::ReactionMechanism> mechanism;
    std::shared_ptr<Solver> flow;
    bool bins;
    explicit Fixture(bool separate = false, bool named_bins = false, bool sensible = false, double heat_per_extent = 0.)
        : bins(named_bins)
    {
        const auto convention =
            sensible ? R::EnergyConvention::SensibleEnergy : R::EnergyConvention::IncludesFormationEnergy;
        const R::ThermodynamicBounds bounds{250, 600, 10000, 2000000};
        // Numerical caloric/solution fixtures only; no material/rate correlation is inferred.
        liquid = R::make_linear_compressible_mixture_liquid(
            species, bounds, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 0, 5e-9, 1e6, 0, 0}; }, convention);
        gas = R::make_ideal_gas_mixture(
            species, {{20, 300, 6000}, {20, 300, 6000}, {20, 300, 6000}, {20, 300, 6000}}, bounds, convention);
        const auto product_phase = bins ? R::SpeciesPhase::Dispersed : R::SpeciesPhase::Resolved;
        mechanism = std::make_shared<const R::ReactionMechanism>(species,
            std::vector<R::ReactionDefinition>{{42, "A-to-B-and-two-gases",
                {{donor, R::SpeciesPhase::Liquid, -1, {}}, {liquid_product, R::SpeciesPhase::Liquid, 1, {}},
                    {first_gas, product_phase, 1, bins ? "micro" : ""},
                    {second_gas, product_phase, 1, bins ? "large" : ""}},
                heat_per_extent}},
            convention);
        Solver::Options controls;
        controls.minimum_pressure = 20000;
        controls.maximum_pressure = 1000000;
        controls.linear.tolerance = 1e-12;
        controls.linear.preconditioner = LinearPreconditioner::None;
        controls.thermal_mode =
            separate ? R::PhaseThermalMode::SeparatePhaseTemperatures : R::PhaseThermalMode::Equilibrium;
        std::vector<std::shared_ptr<const R::MixtureThermodynamics>> phases{liquid, gas};
        if (bins)
            phases.push_back(gas);
        flow = std::make_shared<Solver>(grid, phases, controls);
        Solver::State initial;
        initial.thermodynamic_pressure = 100000;
        initial.cells.resize(grid->num_owned_cells());
        for (size_t c = 0; c < initial.cells.size(); ++c)
        {
            const double volume = grid->cell_volume(static_cast<LO>(c));
            const double liquid_mass = 0.8 * volume * 1000;
            const double liquid_t = separate ? 350 : 300;
            const double liquid_e = 1e6 + 4000 * (liquid_t - 300);
            initial.cells[c].phases.push_back(
                {liquid, {liquid_mass / .1, 0, 0, 0}, 100000, liquid_mass * liquid_e, liquid_t});
            const double gas_volume = (bins ? 0.1 : 0.2) * volume;
            const double gas_amount = 100000 * gas_volume / (gas_constant * 300);
            initial.cells[c].phases.push_back({gas, {0, 0, 0, gas_amount}, 100000, gas_amount * 6000, 300});
            if (bins)
                initial.cells[c].phases.push_back({gas, {0, 0, 0, gas_amount}, 100000, gas_amount * 6000, 300});
        }
        flow->initialize(std::move(initial));
    }
    R::ReactingMixtureOptions options() const
    {
        R::ReactingMixtureOptions result;
        result.mechanism = mechanism;
        result.source_identity = "synthetic-first-order-integrated-budget-v1";
        result.phases = {{R::SpeciesPhase::Liquid, {},
                             {R::SpeciesPhase::Liquid, R::SpeciesPhase::Liquid, R::SpeciesPhase::Dissolved,
                                 R::SpeciesPhase::Dissolved}},
            {bins ? R::SpeciesPhase::Dispersed : R::SpeciesPhase::Resolved, bins ? "micro" : "", {}}};
        if (bins)
            result.phases.push_back({R::SpeciesPhase::Dispersed, "large", {}});
        result.source = [](const R::CoupledReactionContext& context, double dt)
        {
            R::ReactionSourceInput input;
            const auto& reaction = context.reaction;
            const double amount = reaction.state.amount(reaction.catalog, donor, R::SpeciesPhase::Liquid);
            // An explicit integrated extent, always recomputed from transported inventory, never a fresh volumetric
            // rate.
            input.integratedExtents = {{42, 1e-4 * amount * dt}};
            return input;
        };
        return result;
    }
};

double total_energy(const Solver::State& state)
{
    double value = 0;
    for (const auto& cell : state.cells)
        for (const auto& phase : cell.phases)
            value += phase.internal_energy;
    double global = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &value, &global);
    return global;
}
void expect_state_equal(const Solver::State& a, const Solver::State& b)
{
    EXPECT_EQ(a.revision, b.revision);
    EXPECT_DOUBLE_EQ(a.time, b.time);
    EXPECT_DOUBLE_EQ(a.thermodynamic_pressure, b.thermodynamic_pressure);
    ASSERT_EQ(a.cells.size(), b.cells.size());
    for (size_t c = 0; c < a.cells.size(); ++c)
    {
        EXPECT_DOUBLE_EQ(a.cells[c].mechanical_pressure, b.cells[c].mechanical_pressure);
        for (size_t axis = 0; axis < 3; ++axis)
            EXPECT_DOUBLE_EQ(a.cells[c].velocity.component(axis), b.cells[c].velocity.component(axis));
        ASSERT_EQ(a.cells[c].drift_velocity.size(), b.cells[c].drift_velocity.size());
        for (size_t p = 0; p < a.cells[c].drift_velocity.size(); ++p)
            for (size_t axis = 0; axis < 3; ++axis)
                EXPECT_DOUBLE_EQ(
                    a.cells[c].drift_velocity[p].component(axis), b.cells[c].drift_velocity[p].component(axis));
        ASSERT_EQ(a.cells[c].phases.size(), b.cells[c].phases.size());
        for (size_t p = 0; p < a.cells[c].phases.size(); ++p)
        {
            EXPECT_EQ(a.cells[c].phases[p].moles, b.cells[c].phases[p].moles);
            EXPECT_DOUBLE_EQ(a.cells[c].phases[p].internal_energy, b.cells[c].phases[p].internal_energy);
            EXPECT_DOUBLE_EQ(a.cells[c].phases[p].temperature, b.cells[c].phases[p].temperature);
            EXPECT_DOUBLE_EQ(a.cells[c].phases[p].absolute_pressure, b.cells[c].phases[p].absolute_pressure);
        }
    }
}

TEST(ReactingMixtureCouplerTest, SolutionReactionProducesBothGasesWithoutExternalHeatAndConservesPhysicalBudgets)
{
    Fixture f;
    const auto accepted = f.flow->state();
    Coupler controller(f.flow, f.options());
    const auto trial = controller.preview(0.01);
    expect_state_equal(f.flow->state(), accepted);
    EXPECT_GE(trial.diagnostics().correctors, 2);
    EXPECT_DOUBLE_EQ(trial.diagnostics().chemical_heat, 0);
    EXPECT_DOUBLE_EQ(trial.diagnostics().external_heat, 0);
    EXPECT_DOUBLE_EQ(trial.diagnostics().external_mass, 0);
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        const double extent = 1e-6 * accepted.cells[c].phases[0].moles[0];
        EXPECT_NEAR(trial.state().cells[c].phases[0].moles[0], accepted.cells[c].phases[0].moles[0] - extent, 1e-10);
        EXPECT_NEAR(trial.state().cells[c].phases[0].moles[1], extent, 1e-12);
        EXPECT_NEAR(trial.state().cells[c].phases[1].moles[2], extent, 1e-12);
        EXPECT_NEAR(trial.state().cells[c].phases[1].moles[3], accepted.cells[c].phases[1].moles[3] + extent, 1e-12);
        EXPECT_GT(trial.state().cells[c].phases[0].temperature, accepted.cells[c].phases[0].temperature);
        EXPECT_NEAR(trial.local_sources()[c].reactionExtents[0], extent, 1e-12);
    }
    EXPECT_NEAR(trial.diagnostics().flow.mass_error, 0, 1e-10);
    EXPECT_NEAR(total_energy(trial.state()) - total_energy(accepted) - trial.diagnostics().flow.pressure_work, 0, 1e-3);
    controller.commit(trial);
    EXPECT_EQ(f.flow->state().revision, accepted.revision + 1);
    EXPECT_THROW(controller.commit(trial), std::exception);
}

TEST(ReactingMixtureCouplerTest, CheckpointReplayRecordsEachReactionExtentExactlyOnce)
{
    Fixture f;
    Coupler controller(f.flow, f.options());
    const auto saved = controller.checkpoint();
    const auto accepted = f.flow->state();
    const auto first = controller.advance(0.01);
    const auto state = f.flow->state();
    controller.restore(saved);
    expect_state_equal(f.flow->state(), accepted);
    const auto repeated = controller.advance(0.01);
    EXPECT_EQ(repeated.reaction_extents, first.reaction_extents);
    EXPECT_DOUBLE_EQ(repeated.chemical_heat, first.chemical_heat);
    EXPECT_DOUBLE_EQ(repeated.external_heat, first.external_heat);
    EXPECT_NEAR(f.flow->state().thermodynamic_pressure, state.thermodynamic_pressure, 1e-8);
    for (size_t c = 0; c < state.cells.size(); ++c)
        EXPECT_EQ(f.flow->state().cells[c].phases[0].moles, state.cells[c].phases[0].moles);
}

TEST(ReactingMixtureCouplerTest, ExternalCoolingAloneDoesNotCreateReactionProducts)
{
    Fixture f;
    auto options = f.options();
    options.source_identity = "external-cooling-only-v1";
    options.source = [](const auto&, double)
    {
        R::ReactionSourceInput input;
        input.heat.integratedEnergy = -100;
        return input;
    };
    const auto accepted = f.flow->state();
    Coupler controller(f.flow, std::move(options));
    const auto result = controller.advance(0.01);
    EXPECT_DOUBLE_EQ(result.reaction_extents[0], 0);
    EXPECT_DOUBLE_EQ(result.chemical_heat, 0);
    EXPECT_DOUBLE_EQ(result.external_heat, -200);
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        EXPECT_EQ(f.flow->state().cells[c].phases[0].moles, accepted.cells[c].phases[0].moles);
        EXPECT_EQ(f.flow->state().cells[c].phases[1].moles, accepted.cells[c].phases[1].moles);
        EXPECT_LT(f.flow->state().cells[c].phases[0].temperature, accepted.cells[c].phases[0].temperature);
    }
}
TEST(ReactingMixtureCouplerTest, DeterministicSourceFailurePreservesAcceptedStateAndRetainedValidTrial)
{
    Fixture f;
    auto options = f.options();
    const auto production = options.source;
    options.source = [production](const auto& context, double dt)
    {
        if (dt > .015)
            throw std::runtime_error("Injected source-domain failure.");
        return production(context, dt);
    };
    Coupler controller(f.flow, std::move(options));
    const auto accepted = f.flow->state();
    const auto valid = controller.preview(.01);
    EXPECT_THROW((void) controller.preview(.02), std::exception);
    expect_state_equal(f.flow->state(), accepted);
    EXPECT_NO_THROW(controller.commit(valid));
}

TEST(ReactingMixtureCouplerTest, LaterThermalCorrectorFailurePreservesAnEarlierValidImmutableTrial)
{
    Fixture f;
    auto options = f.options();
    options.source_identity = "thermal-domain-failure-v1";
    options.source = [](const R::CoupledReactionContext& context, double dt)
    {
        if (context.thermodynamic_trial.phases[0].temperature > 300.0001)
            throw std::runtime_error("Injected trial-temperature validity failure.");
        R::ReactionSourceInput input;
        input.heat.integratedEnergy = 100000 * dt;
        return input;
    };
    Coupler controller(f.flow, std::move(options));
    const auto accepted = f.flow->state();
    const auto valid = controller.preview(.0001);
    EXPECT_THROW((void) controller.preview(.01), std::exception);
    expect_state_equal(f.flow->state(), accepted);
    EXPECT_NO_THROW(controller.commit(valid));
}

TEST(ReactingMixtureCouplerTest, ExternalControllerRetainsSharedSolverOwnershipAndExclusiveAdvancement)
{
    Fixture f;
    auto flow = f.flow;
    auto options = f.options();
    std::weak_ptr<Solver> weak = flow;
    f.flow.reset();
    {
        Coupler controller(flow, std::move(options));
        EXPECT_THROW((void) flow->advance(.01), std::exception);
        flow.reset();
        EXPECT_FALSE(weak.expired());
        EXPECT_NO_THROW((void) controller.advance(.01));
    }
    EXPECT_TRUE(weak.expired());
}

TEST(ReactingMixtureCouplerTest, SeparatePhaseCrossReactionRequiresAndUsesExplicitConservativeHeatPartition)
{
    Fixture f(true);
    auto options = f.options();
    EXPECT_THROW((void) Coupler(f.flow, options), std::invalid_argument);
    options.partition_heat = [species = f.species](
                                 double heat, const R::SpeciesSourceLedger& ledger, const Solver::CellState& cell)
    {
        const double first = ledger.chemicalChange.amount(*species, first_gas, R::SpeciesPhase::Resolved);
        const double second = ledger.chemicalChange.amount(*species, second_gas, R::SpeciesPhase::Resolved);
        const double enthalpy = (first + second) * (20 + gas_constant) * cell.phases[1].temperature;
        return std::vector<double>{heat - enthalpy, enthalpy};
    };
    Coupler controller(f.flow, std::move(options));
    const auto accepted = f.flow->state();
    const auto trial = controller.preview(.01);
    EXPECT_DOUBLE_EQ(trial.diagnostics().chemical_heat, 0);
    EXPECT_DOUBLE_EQ(trial.diagnostics().external_heat, 0);
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        EXPECT_GT(trial.state().cells[c].phases[0].temperature, accepted.cells[c].phases[0].temperature);
        const auto& gas_old = accepted.cells[c].phases[1];
        const double old_amount = std::accumulate(gas_old.moles.begin(), gas_old.moles.end(), 0.0);
        const double old_volume = old_amount * gas_constant * gas_old.temperature / gas_old.absolute_pressure;
        // The transfer carries h(T_g,new); its n*cp*T terms cancel, leaving compression of accepted gas.
        const double expected_gas_temperature =
            (gas_old.internal_energy + trial.state().cells[c].phases[1].absolute_pressure * old_volume) /
            (old_amount * (20 + gas_constant));
        EXPECT_NEAR(trial.state().cells[c].phases[1].temperature, expected_gas_temperature, 1e-6);
        EXPECT_NE(trial.state().cells[c].phases[0].temperature, trial.state().cells[c].phases[1].temperature);
    }
    controller.commit(trial);
}

TEST(ReactingMixtureCouplerTest, InvalidHeatPartitionRejectsWithoutPublishingOrTakingPermanentOwnership)
{
    Fixture f(true);
    auto options = f.options();
    options.partition_heat = [](double heat, const auto&, const auto&) { return std::vector<double>{heat + 1, 0}; };
    const auto accepted = f.flow->state();
    {
        Coupler controller(f.flow, std::move(options));
        EXPECT_THROW((void) controller.preview(.01), std::exception);
        expect_state_equal(f.flow->state(), accepted);
    }
    EXPECT_NO_THROW((void) f.flow->advance(.01));
}

TEST(ReactingMixtureCouplerTest, NamedGasBinsReceiveOnlyTheirDeclaredStoichiometricProducts)
{
    Fixture f(false, true);
    const auto accepted = f.flow->state();
    Coupler controller(f.flow, f.options());
    const auto trial = controller.preview(.01);
    for (size_t c = 0; c < accepted.cells.size(); ++c)
    {
        const double extent = 1e-6 * accepted.cells[c].phases[0].moles[0];
        EXPECT_NEAR(trial.state().cells[c].phases[1].moles[2], extent, 1e-12);
        EXPECT_DOUBLE_EQ(trial.state().cells[c].phases[2].moles[2], 0);
        EXPECT_DOUBLE_EQ(trial.state().cells[c].phases[1].moles[3], accepted.cells[c].phases[1].moles[3]);
        EXPECT_NEAR(trial.state().cells[c].phases[2].moles[3], accepted.cells[c].phases[2].moles[3] + extent, 1e-12);
        EXPECT_EQ(trial.local_sources()[c].chemicalChange.dispersedPopulations.size(), 2);
        EXPECT_NEAR(
            trial.local_sources()[c].chemicalChange.amount(*f.species, first_gas, R::SpeciesPhase::Dispersed, "micro"),
            extent, 1e-12);
        EXPECT_NEAR(
            trial.local_sources()[c].chemicalChange.amount(*f.species, second_gas, R::SpeciesPhase::Dispersed, "large"),
            extent, 1e-12);
    }
    controller.commit(trial);
}

TEST(ReactingMixtureCouplerTest, MPIFixtureExercisesActualPartitionAndEmptyOwners)
{
    Fixture f;
    const auto comm = f.grid->owned_cell_map()->getComm();
    const int local_present = f.grid->num_owned_cells() > 0;
    int present = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_present, &present);
    EXPECT_EQ(present, std::min(2, comm->getSize()));
    if (comm->getSize() == 4)
        EXPECT_EQ(comm->getSize() - present, 2);
}
TEST(ReactingMixtureCouplerTest, DissolvedInventoryCannotOwnASeparatePhysicalEnergyOrGasVolume)
{
    Fixture f;
    auto options = f.options();
    options.mechanism = std::make_shared<const R::ReactionMechanism>(
        f.species, std::vector<R::ReactionDefinition>{}, R::EnergyConvention::IncludesFormationEnergy);
    options.source = {};
    options.phases[0].species_phases.assign(4, R::SpeciesPhase::Liquid);
    options.phases[1] = {R::SpeciesPhase::Dissolved, {}, {}};
    EXPECT_THROW((void) Coupler(f.flow, std::move(options)), std::invalid_argument);
    EXPECT_NO_THROW((void) f.flow->advance(.01));
}
} // namespace

namespace
{
TEST(ReactingMixtureCouplerTest, CompensatedReservoirEnergyMatchesIndependentOneJouleHeatBudget)
{
    Fixture packets;
    Fixture heat;
    auto packet_options = packets.options();
    packet_options.source_identity = "signed-reservoir-energy-cancellation-v1";
    packet_options.source = [](const auto&, double)
    {
        R::ReactionSourceInput source;
        source.externalTransfers = {{"large-energy-supply", donor, R::SpeciesPhase::Liquid, 0., 0., 1e18, {}, {}},
            {"small-energy-supply", donor, R::SpeciesPhase::Liquid, 0., 0., 1., {}, {}},
            {"large-energy-withdrawal", donor, R::SpeciesPhase::Liquid, 0., 0., -1e18, {}, {}}};
        return source;
    };
    auto heat_options = heat.options();
    heat_options.source_identity = "independent-one-joule-heat-v1";
    heat_options.source = [](const auto&, double)
    {
        R::ReactionSourceInput source;
        source.heat.integratedEnergy = 1.;
        return source;
    };
    const auto packet_accepted = packets.flow->state();
    const auto heat_accepted = heat.flow->state();
    const double old_energy = total_energy(packet_accepted);
    Coupler packet_controller(packets.flow, std::move(packet_options));
    Coupler heat_controller(heat.flow, std::move(heat_options));
    const auto packet_trial = packet_controller.preview(.01);
    const auto heat_trial = heat_controller.preview(.01);
    expect_state_equal(packets.flow->state(), packet_accepted);
    expect_state_equal(heat.flow->state(), heat_accepted);
    expect_state_equal(packet_trial.state(), heat_trial.state());
    EXPECT_EQ(packet_trial.local_sources().size(), packet_accepted.cells.size());
    for (std::size_t cell = 0; cell < packet_accepted.cells.size(); ++cell)
    {
        const auto& ledger = packet_trial.local_sources()[cell];
        EXPECT_DOUBLE_EQ(ledger.transferEnergy, 1.);
        EXPECT_DOUBLE_EQ(ledger.externalHeat, 0.);
        EXPECT_DOUBLE_EQ(ledger.reactionHeat, 0.);
        EXPECT_DOUBLE_EQ(ledger.externalMass, 0.);
        EXPECT_EQ(ledger.externalTransfers.size(), 3U);
        EXPECT_DOUBLE_EQ(heat_trial.local_sources()[cell].externalHeat, 1.);
        EXPECT_DOUBLE_EQ(heat_trial.local_sources()[cell].transferEnergy, 0.);
        for (std::size_t phase = 0; phase < packet_accepted.cells[cell].phases.size(); ++phase)
        {
            EXPECT_EQ(
                packet_trial.state().cells[cell].phases[phase].moles, packet_accepted.cells[cell].phases[phase].moles);
            EXPECT_DOUBLE_EQ(packet_trial.state().cells[cell].phases[phase].absolute_pressure,
                heat_trial.state().cells[cell].phases[phase].absolute_pressure);
        }
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_DOUBLE_EQ(packets.flow->velocity().value(static_cast<LO>(cell)).component(axis),
                packet_accepted.cells[cell].velocity.component(axis));
            EXPECT_DOUBLE_EQ(heat.flow->velocity().value(static_cast<LO>(cell)).component(axis),
                heat_accepted.cells[cell].velocity.component(axis));
            EXPECT_DOUBLE_EQ(packet_trial.state().cells[cell].velocity.component(axis),
                heat_trial.state().cells[cell].velocity.component(axis));
        }
    }
    const auto comm = packets.grid->owned_cell_map()->getComm();
    const double local_budget = static_cast<double>(packet_accepted.cells.size());
    double global_budget = 0.;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_budget, &global_budget);
    EXPECT_DOUBLE_EQ(global_budget, 2.);
    EXPECT_DOUBLE_EQ(packet_trial.diagnostics().transfer_energy, global_budget);
    EXPECT_DOUBLE_EQ(packet_trial.diagnostics().external_heat, 0.);
    EXPECT_DOUBLE_EQ(heat_trial.diagnostics().external_heat, global_budget);
    EXPECT_DOUBLE_EQ(heat_trial.diagnostics().transfer_energy, 0.);
    EXPECT_NEAR(total_energy(packet_trial.state()) - old_energy - packet_trial.diagnostics().flow.pressure_work,
        global_budget, 1e-3);
    packet_controller.commit(packet_trial);
    heat_controller.commit(heat_trial);
    expect_state_equal(packets.flow->state(), heat.flow->state());
    EXPECT_NEAR(total_energy(packets.flow->state()) - old_energy - packet_controller.diagnostics().flow.pressure_work,
        global_budget, 1e-3);
}
} // namespace

namespace
{
TEST(ReactingMixtureCouplerTest, ChemicalHeatSurvivesCancellationAcrossIndependentReservoirChannels)
{
    Fixture packets(false, false, true, 1000.);
    Fixture chemical(false, false, true, 1000.);
    auto packet_options = packets.options();
    auto chemical_options = chemical.options();
    packet_options.source_identity = "chemical-one-joule-plus-cancelling-reservoirs-v1";
    chemical_options.source_identity = "same-chemical-one-joule-alone-v1";
    packet_options.source = [](const auto&, double)
    {
        R::ReactionSourceInput source;
        source.integratedExtents = {{42, .001}};
        source.heat.integratedEnergy = 1e18;
        source.externalTransfers = {
            {"independent-energy-withdrawal", donor, R::SpeciesPhase::Liquid, 0., 0., -1e18, {}, {}}};
        return source;
    };
    chemical_options.source = [](const auto&, double)
    {
        R::ReactionSourceInput source;
        source.integratedExtents = {{42, .001}};
        return source;
    };
    // Both sources execute the same mass/element-balanced reaction with 1000 J/mol
    // sensible heat, so the independent net budget is exactly .001*1000 = 1 J/cell.
    const auto packet_accepted = packets.flow->state();
    const auto chemical_accepted = chemical.flow->state();
    const double old_energy = total_energy(packet_accepted);
    Coupler packet_controller(packets.flow, std::move(packet_options));
    Coupler chemical_controller(chemical.flow, std::move(chemical_options));
    const auto packet_trial = packet_controller.preview(.01);
    const auto chemical_trial = chemical_controller.preview(.01);
    expect_state_equal(packets.flow->state(), packet_accepted);
    expect_state_equal(chemical.flow->state(), chemical_accepted);
    expect_state_equal(packet_trial.state(), chemical_trial.state());
    for (std::size_t cell = 0; cell < packet_accepted.cells.size(); ++cell)
    {
        const auto& ledger = packet_trial.local_sources()[cell];
        EXPECT_DOUBLE_EQ(ledger.reactionExtents[0], .001);
        EXPECT_DOUBLE_EQ(ledger.reactionHeat, 1.);
        EXPECT_DOUBLE_EQ(ledger.externalHeat, 1e18);
        EXPECT_DOUBLE_EQ(ledger.transferEnergy, -1e18);
        EXPECT_DOUBLE_EQ(ledger.thermal_energy(), 1.);
        EXPECT_DOUBLE_EQ(chemical_trial.local_sources()[cell].reactionHeat, 1.);
        EXPECT_DOUBLE_EQ(chemical_trial.local_sources()[cell].externalHeat, 0.);
        EXPECT_DOUBLE_EQ(chemical_trial.local_sources()[cell].transferEnergy, 0.);
        EXPECT_EQ(packet_trial.state().cells[cell].phases[0].moles, chemical_trial.state().cells[cell].phases[0].moles);
        EXPECT_GT(
            packet_trial.state().cells[cell].phases[0].temperature, packet_accepted.cells[cell].phases[0].temperature);
    }
    const auto comm = packets.grid->owned_cell_map()->getComm();
    const double local_budget = static_cast<double>(packet_accepted.cells.size());
    double global_budget = 0.;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_budget, &global_budget);
    EXPECT_DOUBLE_EQ(global_budget, 2.);
    EXPECT_DOUBLE_EQ(packet_trial.diagnostics().chemical_heat, global_budget);
    EXPECT_DOUBLE_EQ(packet_trial.diagnostics().net_thermal_energy, global_budget);
    EXPECT_DOUBLE_EQ(chemical_trial.diagnostics().net_thermal_energy, global_budget);
    EXPECT_NEAR(total_energy(packet_trial.state()) - old_energy - packet_trial.diagnostics().flow.pressure_work,
        global_budget, 1e-3);
    packet_controller.commit(packet_trial);
    chemical_controller.commit(chemical_trial);
    expect_state_equal(packets.flow->state(), chemical.flow->state());
    EXPECT_NEAR(total_energy(packets.flow->state()) - old_energy - packet_controller.diagnostics().flow.pressure_work,
        global_budget, 1e-3);
}

TEST(ReactingMixtureCouplerTest, FullConversionUsesActualTransportedDonorsAndCommitsTheExactReceiptOnce)
{
    const auto grid = mesh();
    constexpr R::SpeciesId a = 103, b = 709;
    const auto species = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{a, "finite_A", .02, {{"X", 1}}, {R::SpeciesPhase::Liquid}},
            {b, "finite_B", .02, {{"X", 1}}, {R::SpeciesPhase::Liquid}}});
    const auto liquid =
        R::make_linear_compressible_mixture_liquid(species, {250, 600, 10000, 2000000}, [](std::span<const double>)
            { return R::LinearCompressibleMixtureLiquid{300, 100000, 1000, 4000, 1e-3, 5e-9, 1e6, 0, 0}; });
    const auto mechanism = std::make_shared<const R::ReactionMechanism>(species,
        std::vector<R::ReactionDefinition>{
            {64, "isomerization", {{a, R::SpeciesPhase::Liquid, -1, {}}, {b, R::SpeciesPhase::Liquid, 1, {}}}, 0}},
        R::EnergyConvention::IncludesFormationEnergy);
    Solver::Options flow_options;
    flow_options.minimum_pressure = 20000;
    flow_options.maximum_pressure = 1000000;
    flow_options.linear.preconditioner = LinearPreconditioner::None;
    flow_options.linear.tolerance = 1e-12;
    auto flow = std::make_shared<Solver>(grid, std::vector{liquid}, flow_options);
    Solver::State initial;
    initial.thermodynamic_pressure = 100000;
    initial.cells.resize(grid->num_owned_cells());
    for (size_t c = 0; c < initial.cells.size(); ++c)
    {
        const double mass = grid->cell_volume(static_cast<LO>(c)) * 1000;
        initial.cells[c].phases = {{liquid, {.8 * mass / .02, .2 * mass / .02}, 100000, mass * 1e6, 300}};
    }
    flow->initialize(initial);
    initial = flow->state();
    R::ReactingMixtureOptions options;
    options.mechanism = mechanism;
    options.phases = {{R::SpeciesPhase::Liquid, {}, {}}};
    options.source_identity = "complete-isomerization-after-transport";
    options.source_revision = 3;
    auto calls = std::make_shared<int>(0);
    options.source = [grid, calls](const R::CoupledReactionContext& context, double)
    {
        ++*calls; // Instrumentation only; the source is independent of invocation count.
        const auto& transported = context.transported.phases.at(0);
        const double donor_amount = context.reaction.state.amount(context.reaction.catalog, a, R::SpeciesPhase::Liquid);
        EXPECT_DOUBLE_EQ(donor_amount, transported.moles.at(0));
        EXPECT_EQ(context.thermodynamic_trial.phases[0].moles, transported.moles);
        R::ReactionSourceInput input;
        input.integratedExtents = {{64, donor_amount}};
        input.heat.integratedEnergy = grid->cell_centroid(static_cast<LO>(context.owned_cell)).z < .5 ? 1000. : 0.;
        return input;
    };
    Coupler controller(flow, std::move(options));
    const auto checkpoint = controller.checkpoint();
    const auto trial = controller.preview(.01);
    expect_state_equal(flow->state(), initial);
    const auto& snapshot = trial.transport_snapshot();
    EXPECT_EQ(snapshot.accepted_revision, initial.revision);
    EXPECT_EQ(snapshot.stage.provider_revision, 3);
    EXPECT_EQ(snapshot.stage.provider_identity, "ReactingMixtureCoupler:complete-isomerization-after-transport");
    EXPECT_EQ(snapshot.stage.corrector + 1, size_t(trial.diagnostics().correctors));
    EXPECT_GE(trial.diagnostics().correctors, 2);
    double local_transport_change = 0, global_transport_change = 0;
    for (size_t c = 0; c < initial.cells.size(); ++c)
    {
        const double donor_amount = snapshot.cells[c].phases[0].moles[0];
        local_transport_change =
            std::max(local_transport_change, std::abs(donor_amount - initial.cells[c].phases[0].moles[0]));
        EXPECT_DOUBLE_EQ(trial.state().cells[c].phases[0].moles[0], 0.);
        EXPECT_DOUBLE_EQ(trial.local_sources()[c].reactionExtents[0], donor_amount);
        EXPECT_DOUBLE_EQ(trial.local_sources()[c].forwardReactionExtents[0], donor_amount);
        EXPECT_DOUBLE_EQ(trial.local_sources()[c].reverseReactionExtents[0], 0.);
        EXPECT_NEAR(
            trial.state().cells[c].phases[0].moles[1], snapshot.cells[c].phases[0].moles[1] + donor_amount, 1e-10);
    }
    Teuchos::reduceAll(
        *grid->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_transport_change, &global_transport_change);
    EXPECT_GT(global_transport_change, 1e-8);
    const int count = *calls;
    controller.commit(trial);
    EXPECT_EQ(*calls, count);
    EXPECT_EQ(controller.diagnostics().forward_reaction_extents, controller.diagnostics().reaction_extents);
    EXPECT_EQ(controller.diagnostics().reverse_reaction_extents, std::vector<double>{0.});
    controller.restore(checkpoint);
    EXPECT_EQ(*calls, count);
    expect_state_equal(flow->state(), initial);
    const auto replay = controller.preview(.01);
    EXPECT_EQ(replay.diagnostics().reaction_extents, trial.diagnostics().reaction_extents);
    expect_state_equal(replay.state(), trial.state());
    int local_empty = grid->num_owned_cells() == 0 ? 1 : 0, empty = 0;
    Teuchos::reduceAll(*grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    if (grid->owned_cell_map()->getComm()->getSize() == 4)
        EXPECT_EQ(empty, 2);
}

TEST(ReactingMixtureCouplerTest, ForwardAndReverseAcceptedStagesRetainGrossTurnoverWithZeroNetReaction)
{
    Fixture f;
    auto options = f.options();
    // The forward SSPRK2 stage supplies the finite donors used by the reverse stage.
    options.source_identity = "forward-reverse-stage-turnover";
    options.source = [schema = f.species](const auto&, double)
    {
        R::ReactionRateProvider rate;
        rate.identity = "manufactured-time-reversal";
        rate.schemaRevision = schema->revision();
        rate.schemaFingerprint = schema->fingerprint();
        rate.validityDescription = "finite donor regression";
        rate.parameterUnits = "mol/(m3 s)";
        rate.validate = [](const auto&) {};
        rate.evaluate = [](const R::ReactionContext& context)
        {
            return std::vector<R::ReactionRate>{
                {42, context.environment.time == 0 ? .1 : -.1, R::SourceVolumeBasis::BulkVolume}};
        };
        R::ReactionSourceInput result;
        result.rates = std::move(rate);
        return result;
    };
    Coupler controller(f.flow, std::move(options));
    const auto accepted = f.flow->state();
    const auto trial = controller.preview(.01);
    EXPECT_NEAR(trial.diagnostics().reaction_extents[0], 0., 1e-16);
    EXPECT_NEAR(trial.diagnostics().forward_reaction_extents[0], .0005, 1e-14);
    EXPECT_NEAR(trial.diagnostics().reverse_reaction_extents[0], .0005, 1e-14);
    EXPECT_DOUBLE_EQ(trial.diagnostics().chemical_heat, 0.);
    for (const auto& ledger : trial.local_sources())
    {
        EXPECT_DOUBLE_EQ(ledger.forwardReactionExtents[0], ledger.reverseReactionExtents[0]);
        EXPECT_GT(ledger.forwardReactionExtents[0], 0.);
    }
    expect_state_equal(f.flow->state(), accepted);
    controller.commit(trial);
}

TEST(ReactingMixtureCouplerTest, ArrivingIndependentPhaseUsesTransportedEnergyForItsMissingThermalGuess)
{
    Fixture f(true);
    auto state = f.flow->state();
    for (size_t c = 0; c < state.cells.size(); ++c)
    {
        auto& cell = state.cells[c];
        cell.drift_velocity[1] = Solver::Vector{0, 0, .1};
        if (f.grid->cell_centroid(static_cast<LO>(c)).z > .5)
        {
            const double mass = f.grid->cell_volume(static_cast<LO>(c)) * 1000;
            cell.phases[0].moles = {mass / .1, 0, 0, 0};
            cell.phases[0].internal_energy = mass * (1e6 + 4000 * 50);
            cell.phases[1].moles.assign(4, 0.);
            cell.phases[1].internal_energy = 0;
            cell.phases[1].temperature = 0;
        }
    }
    f.flow->initialize(state);
    state = f.flow->state();
    auto options = f.options();
    options.mechanism = std::make_shared<const R::ReactionMechanism>(
        f.species, std::vector<R::ReactionDefinition>{}, R::EnergyConvention::IncludesFormationEnergy);
    options.source_identity = "arriving-gas-independent-caloric-state";
    auto checked = std::make_shared<std::vector<bool>>(state.cells.size(), false);
    options.source = [state, checked](const R::CoupledReactionContext& context, double)
    {
        const auto c = context.owned_cell;
        const auto& packet = context.transported.phases[1];
        const double amount = std::accumulate(packet.moles.begin(), packet.moles.end(), 0.);
        if (!(*checked)[c] && state.cells[c].phases[1].temperature == 0 && amount > 0)
        {
            auto arriving = context.thermodynamic_trial.phases[1];
            arriving.internal_energy = packet.internal_energy;
            const double expected = R::recover_phase_temperatures(
                std::vector{arriving}, packet.internal_energy, R::PhaseThermalMode::SeparatePhaseTemperatures)
                                        .phases[0]
                                        .temperature;
            EXPECT_NEAR(context.thermodynamic_trial.phases[1].temperature, expected, 1e-10);
            EXPECT_NEAR(expected, 300., 1e-9);
            EXPECT_GT(std::abs(context.thermodynamic_trial.phases[0].temperature - expected), 40.);
            (*checked)[c] = true;
        }
        return R::ReactionSourceInput{};
    };
    Coupler controller(f.flow, std::move(options));
    constexpr double dt = .001;
    const auto trial = controller.preview(dt);
    int local_checked = std::count(checked->begin(), checked->end(), true), global_checked = 0;
    Teuchos::reduceAll(*f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_checked, &global_checked);
    EXPECT_EQ(global_checked, 1);
    expect_state_equal(f.flow->state(), state);
    EXPECT_NEAR(trial.diagnostics().flow.mass_error, 0., 1e-10);
    controller.commit(trial);
    // Independent Rhie--Chow identity on this two-cell closed-Neumann line.
    // The LS z normal coefficient is h^2+(h/2)^2, so grad(p)=.8*Delta p/h.
    // This checks synchronized cell gradients, the old-pressure interpolation,
    // and the bare pressure correction rather than reusing assembly helpers.
    std::array<double, 8> local{}, global{};
    for (size_t c = 0; c < f.flow->state().cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const size_t side = f.grid->cell_centroid(cell).z < .5 ? 0 : 1;
        const auto& accepted_cell = f.flow->state().cells[c];
        double mass = 0;
        for (const auto& phase : accepted_cell.phases)
            for (size_t species = 0; species < phase.moles.size(); ++species)
                mass += phase.moles[species] * f.species->species()[species].molarMass;
        local[side] = mass / f.grid->cell_volume(cell);
        local[2 + side] = accepted_cell.velocity.z;
        local[4 + side] = accepted_cell.mechanical_pressure;
        local[6 + side] = 1;
    }
    Teuchos::reduceAll(
        *f.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    ASSERT_DOUBLE_EQ(global[6], 1.);
    ASSERT_DOUBLE_EQ(global[7], 1.);
    constexpr double center_distance = .5;
    const double pressure_jump = global[5] - global[4];
    const double gradient = .8 * pressure_jump / center_distance;
    const double expected_carrier = .5 * (global[2] + global[3]) +
                                    .5 * dt * gradient * (1. / global[0] + 1. / global[1]) -
                                    dt * 2. / (global[0] + global[1]) * pressure_jump / center_distance;
    for (const auto face : f.flow->volume_flux().owned_face_ids())
        if (!f.grid->is_geometry_exterior_face(face))
        {
            const double sign = f.grid->cell_centroid(f.grid->owner_cell(face)).z < .5 ? 1. : -1.;
            EXPECT_NEAR(f.flow->volume_flux().local_value(face), sign * expected_carrier, 1e-11);
        }
}
} // namespace
