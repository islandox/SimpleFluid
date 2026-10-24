#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
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
            // An explicit integrated extent, always recomputed from accepted inventory, never a fresh volumetric rate.
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
        ASSERT_EQ(a.cells[c].phases.size(), b.cells[c].phases.size());
        for (size_t p = 0; p < a.cells[c].phases.size(); ++p)
        {
            EXPECT_EQ(a.cells[c].phases[p].moles, b.cells[c].phases[p].moles);
            EXPECT_DOUBLE_EQ(a.cells[c].phases[p].internal_energy, b.cells[c].phases[p].internal_energy);
            EXPECT_DOUBLE_EQ(a.cells[c].phases[p].temperature, b.cells[c].phases[p].temperature);
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
} // namespace
