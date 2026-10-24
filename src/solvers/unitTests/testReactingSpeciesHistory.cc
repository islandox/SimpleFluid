#include "equations/BubblePopulationKinetics.hh"
#include "equations/ReactingGasModel.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace
{
namespace S = SimpleFluid;
namespace R = S::Reacting;
namespace B = S::BubbleGasPhysics;
using Pack = S::DefaultTpetraTypes;
using Mesh = S::MeshHandle<Pack>;
using Model = S::ReactingGasModel<Pack, Mesh>;
using Field = Model::field_type;
using Velocity = Model::velocity_field_type;
using Flux = Model::face_flux_field_type;
constexpr R::SpeciesId donor = 11, liquidProduct = 22, gasOne = 51, gasTwo = 73;
constexpr double gasConstant = 8.31446261815324;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<const R::SpeciesCatalog> schema()
{
    using P = R::SpeciesPhase;
    return std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{donor, "manufactured donor", .010, {{"X", 5}}, {P::Liquid}},
            {liquidProduct, "manufactured liquid product", .004, {{"X", 2}}, {P::Liquid}},
            {gasOne, "ordinary gas one", .002, {{"X", 1}}, {P::Dissolved, P::Dispersed, P::Resolved}},
            {gasTwo, "ordinary gas two", .004, {{"X", 2}}, {P::Dissolved, P::Dispersed, P::Resolved}}},
        5);
}

S::ReactingGasOptions options()
{
    using P = R::SpeciesPhase;
    auto catalog = schema();
    S::ReactingGasOptions result;
    result.mode = S::ReactingGasMode::TwoPopulation;
    result.reference_pressure = 100000.;
    result.surface_tension = 0.;
    result.min_radius = 1e-9;
    result.max_radius = .1;
    result.dissolved_transport = S::ReactingTransportMode::Advective;
    result.mass_transfer_mode = S::BubbleMassTransferMode::SphericalDiffusion;
    result.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(
        catalog, std::vector<R::ReactionDefinition>{{8, "A to B plus G1 plus G2",
                     {{donor, P::Liquid, -1}, {liquidProduct, P::Liquid, 1}, {gasOne, P::Dissolved, 1},
                         {gasTwo, P::Dissolved, 1}},
                     100.}});
    const R::ThermodynamicBounds bounds{250., 600., 10000., 2000000.};
    result.gas_thermodynamics =
        R::make_ideal_gas_mixture(catalog, {{20., 300., 0.}, {20., 300., 0.}, {20., 300., 0.}, {25., 300., 0.}}, bounds,
            R::EnergyConvention::SensibleEnergy);
    result.liquid_thermodynamics = R::make_linear_compressible_mixture_liquid(
        catalog, bounds, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300., 100000., 1000., 4200., 0., 1e-9, 0., .001, .6}; },
        R::EnergyConvention::SensibleEnergy);
    for (const auto id : {gasOne, gasTwo})
    {
        S::ReactingSpeciesProperties property;
        property.species = id;
        property.carried_molar_enthalpy = [id](const auto&, auto, auto) { return id == gasOne ? 100. : 200.; };
        result.species_properties.push_back(std::move(property));
    }
    return result;
}

std::shared_ptr<Mesh> mesh(std::size_t cells = 4, double length = 1.)
{
    std::vector<double> edges(cells + 1);
    for (std::size_t i = 0; i <= cells; ++i)
        edges[i] = length * static_cast<double>(i) / cells;
    auto geometry = std::make_shared<Mesh::Cartesian>(S::Vec3D<S::ArrReal>{{{0., 1.}, {0., 1.}, edges}});
    Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    return std::make_shared<Mesh>(std::move(geometry), distribution);
}

S::BoussinesqModelOptions material_options()
{
    S::BoussinesqModelOptions result;
    result.reference_density = result.density = 1000.;
    result.specific_heat_capacity = 4200.;
    result.dynamic_viscosity = .001;
    result.thermal_conductivity = .6;
    return result;
}

struct Fixture
{
    std::shared_ptr<Mesh> grid;
    S::ReactingGasOptions settings;
    Model model;
    Field temperature, pressure;
    Velocity velocity;
    Flux flux;
    S::MaterialPropertyFields<Pack, Mesh> material;
    explicit Fixture(S::ReactingGasOptions selected = options(), std::size_t cells = 4, double length = 1.)
        : grid(mesh(cells, length)), settings(std::move(selected)), model(grid, settings), temperature(grid, 300., "T"),
          pressure(grid, 0., "p"), velocity(grid, S::vec3<double>{}, "U"), flux(grid, 0., "phi"),
          material(grid, material_options(), S::TimeStepperOptions{})
    {
    }
    std::vector<R::CellSpeciesState> initial(
        double g1 = 0., double g2 = 0., bool gas = false, double occupied_fraction = 1.) const
    {
        const auto& catalog = settings.reaction_mechanism->catalog();
        std::vector<R::CellSpeciesState> result;
        for (std::size_t owned = 0; owned < grid->num_owned_cells(); ++owned)
        {
            const double volume = grid->cell_volume(static_cast<Pack::local_ordinal_type>(owned));
            R::CellSpeciesState state(catalog);
            state.dispersedPopulations = {{"micro", std::vector<double>(4)}, {"large", std::vector<double>(4)}};
            const double gasVolume = gas ? (g1 + g2) * volume * gasConstant * 300. / 100000. : 0.;
            const double dissolvedMass = gas ? 0. : (g1 * .002 + g2 * .004) * volume;
            state.set_amount(
                catalog, donor, R::SpeciesPhase::Liquid,
                (1000. * (occupied_fraction * volume - gasVolume) - dissolvedMass) / .010);
            state.set_amount(catalog, gasOne, gas ? R::SpeciesPhase::Dispersed : R::SpeciesPhase::Dissolved,
                g1 * volume, gas ? "micro" : "");
            state.set_amount(catalog, gasTwo, gas ? R::SpeciesPhase::Dispersed : R::SpeciesPhase::Dissolved,
                g2 * volume, gas ? "micro" : "");
            result.push_back(std::move(state));
        }
        return result;
    }
    void initialize(const std::vector<R::CellSpeciesState>& states, bool gas = false)
    {
        std::vector<std::vector<double>> numbers;
        for (std::size_t owned = 0; owned < states.size(); ++owned)
            numbers.push_back({gas ? 1e6 * grid->cell_volume(static_cast<Pack::local_ordinal_type>(owned)) : 0., 0.});
        model.initialize_species(states, numbers);
        model.initialize_state(0., temperature, pressure, velocity, material);
    }
    void advance(double dt = .01) { model.advance(dt, dt, temperature, pressure, velocity, flux, material, nullptr); }
    double sum(double local) const
    {
        double result = 0.;
        Teuchos::reduceAll(*grid->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &result);
        return result;
    }
};

const R::SpeciesInventoryHistory& history(const Model& model, R::SpeciesId species)
{
    const auto& reports = model.species_histories();
    const auto found = std::ranges::find(reports, species, &R::SpeciesInventoryHistory::species);
    if (found == reports.end())
        throw std::logic_error("Missing species history.");
    return *found;
}
const R::SpeciesCompartmentEvents& events(
    const Model& model, R::SpeciesId species, R::SpeciesPhase phase, std::string_view population = {})
{
    const auto& reports = model.species_step_events();
    const auto found = std::ranges::find_if(reports, [&](const auto& event)
        { return event.species == species && event.phase == phase && event.population == population; });
    if (found == reports.end())
        throw std::logic_error("Missing compartment event.");
    return *found;
}

void expect_liquid_headroom(const Fixture& fixture)
{
    for (std::size_t owned = 0; owned < fixture.grid->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_LT(fixture.model.liquid_volume(cell), fixture.grid->cell_volume(cell));
    }
}

TEST(ReactingSpeciesHistoryTest, GrossSourceAndInterphaseEventsRemainDistinctAndReplayExactly)
{
    auto selected = options();
    selected.species_source = [](const R::ReactionContext& context, double dt)
    {
        if (dt > .02)
            throw std::runtime_error("Deterministic source-domain failure.");
        const double volume = context.environment.bulkVolume;
        R::ReactionSourceInput source;
        source.integratedExtents = {{8, .02 * volume}};
        source.externalTransfers = {
            {"arbitrary-fed-species", gasTwo, R::SpeciesPhase::Dissolved, .003 * volume, .000012 * volume, 0., {}, {}},
            {"arbitrary-withdrawal", donor, R::SpeciesPhase::Liquid, -.001 * volume, -.000010 * volume, 0., {}, {}}};
        return source;
    };
    selected.population_process = [](const B::MultispeciesPopulationContext& context, double)
    {
        B::MultispeciesPopulationPacket packet;
        const double volume = context.chemistry.environment.bulkVolume;
        for (const auto id : {gasOne, gasTwo})
            packet.transfers.push_back(
                {id, R::SpeciesPhase::Dissolved, R::SpeciesPhase::Dispersed, {}, "micro", .01 * volume});
        packet.numberEvents = {{"micro", 1e6 * volume, 0.}};
        return packet;
    };
    Fixture fixture(selected, 2);
    // Accounting requires finite geometric room for supplied mass and subsequent gas birth.
    fixture.initialize(fixture.initial(0., 0., false, .99));
    expect_liquid_headroom(fixture);
    const auto initial_reports = fixture.model.species_histories();
    const auto accepted = fixture.model.snapshot();
    fixture.advance();
    EXPECT_NEAR(history(fixture.model, donor).initial_moles, 99000., 1e-9);
    EXPECT_NEAR(history(fixture.model, donor).chemical_consumption, .02, 1e-15);
    EXPECT_NEAR(history(fixture.model, donor).external_withdrawal, .001, 1e-15);
    EXPECT_NEAR(history(fixture.model, gasTwo).chemical_production, .02, 1e-15);
    EXPECT_NEAR(history(fixture.model, gasTwo).external_supply, .003, 1e-15);
    EXPECT_NEAR(history(fixture.model, gasTwo).current_moles, .023, 1e-14);
    const auto& liquid_event = events(fixture.model, gasTwo, R::SpeciesPhase::Dissolved);
    const auto& gas_event = events(fixture.model, gasTwo, R::SpeciesPhase::Dispersed, "micro");
    EXPECT_NEAR(liquid_event.chemical_production, .02, 1e-15);
    EXPECT_NEAR(liquid_event.external_supply, .003, 1e-15);
    EXPECT_NEAR(liquid_event.interphase_outflow, .01, 1e-15);
    EXPECT_NEAR(gas_event.interphase_inflow, .01, 1e-15);
    EXPECT_DOUBLE_EQ(gas_event.chemical_production, 0.);
    EXPECT_TRUE(fixture.model.physical_escape_events().empty());
    const auto reports = fixture.model.species_histories();
    const auto step = fixture.model.species_step_events();
    fixture.model.refresh_species_thermodynamics(
        .01, fixture.temperature, fixture.pressure, fixture.velocity, fixture.material);
    EXPECT_EQ(fixture.model.species_histories(), reports);
    EXPECT_EQ(fixture.model.species_step_events(), step);
    EXPECT_ANY_THROW(fixture.advance(.1));
    EXPECT_EQ(fixture.model.species_histories(), reports);
    EXPECT_EQ(fixture.model.species_step_events(), step);
    fixture.model.restore(accepted);
    EXPECT_EQ(fixture.model.species_histories(), initial_reports);
    fixture.advance();
    EXPECT_EQ(fixture.model.species_histories(), reports);
    EXPECT_EQ(fixture.model.species_step_events(), step);
    const auto comm = fixture.grid->owned_cell_map()->getComm();
    EXPECT_EQ(fixture.sum(fixture.grid->num_owned_cells() == 0 ? 1. : 0.),
        static_cast<double>(std::max(0, comm->getSize() - 2)));
}

TEST(ReactingSpeciesHistoryTest, ExternalWithdrawalNeverMasqueradesAsPhysicalEscape)
{
    auto selected = options();
    selected.species_source = [](const R::ReactionContext& context, double)
    {
        const double amount = .01 * context.environment.bulkVolume;
        R::ReactionSourceInput source;
        // Even a misleading user-chosen reservoir string cannot reclassify a source packet.
        source.externalTransfers = {
            {"physical boundary", gasOne, R::SpeciesPhase::Dissolved, -amount, -.002 * amount, -100. * amount, {}, {}}};
        return source;
    };
    Fixture fixture(selected, 2);
    fixture.initialize(fixture.initial(1.));
    fixture.advance();
    EXPECT_NEAR(history(fixture.model, gasOne).external_withdrawal, .01, 1e-15);
    EXPECT_DOUBLE_EQ(history(fixture.model, gasOne).physical_outflow, 0.);
    EXPECT_TRUE(fixture.model.physical_escape_events().empty());
    EXPECT_DOUBLE_EQ(events(fixture.model, gasOne, R::SpeciesPhase::Dissolved).physical_outflow, 0.);
    EXPECT_NEAR(fixture.model.global_species_moles(gasOne), .99, 1e-13);
}

TEST(ReactingSpeciesHistoryTest, ActualEscapeHasExactStepCompositionAndExplicitEnergyTraceLimitation)
{
    auto selected = options();
    selected.rise_velocity_mode = S::BubbleRiseVelocityMode::ConstantSlip;
    selected.constant_slip_velocity = .3;
    selected.free_surface_patches = {"zmax"};
    Fixture fixture(selected, 2);
    fixture.initialize(fixture.initial(.01, .02, true), true);
    const auto checkpoint = fixture.model.snapshot();
    fixture.advance(.1);
    double one = 0., two = 0.;
    for (const auto& packet : fixture.model.physical_escape_events())
    {
        EXPECT_EQ(packet.phase, R::SpeciesPhase::Dispersed);
        EXPECT_EQ(packet.population, "micro");
        EXPECT_EQ(packet.energy_contract, R::PhysicalEscapeEnergyContract::SuppliedPartialMolarEnthalpyTrace);
        EXPECT_FALSE(packet.phase_energy_debited);
        EXPECT_FALSE(packet.phase_momentum_debited);
        EXPECT_EQ(packet.energy_convention, R::EnergyConvention::SensibleEnergy);
        if (packet.species == gasOne)
        {
            one += packet.moles;
            EXPECT_NEAR(packet.mass, .002 * packet.moles, 1e-16);
            EXPECT_NEAR(packet.carried_enthalpy_trace, 100. * packet.moles, 1e-12);
        }
        else if (packet.species == gasTwo)
        {
            two += packet.moles;
            EXPECT_NEAR(packet.mass, .004 * packet.moles, 1e-16);
            EXPECT_NEAR(packet.carried_enthalpy_trace, 200. * packet.moles, 1e-12);
        }
        else
            ADD_FAILURE() << "Unexpected escaping species.";
    }
    EXPECT_GT(one, 0.);
    EXPECT_NEAR(two, 2 * one, 1e-13);
    EXPECT_NEAR(history(fixture.model, gasOne).physical_outflow, one, 1e-14);
    EXPECT_DOUBLE_EQ(history(fixture.model, gasOne).external_withdrawal, 0.);
    EXPECT_NEAR(events(fixture.model, gasOne, R::SpeciesPhase::Dispersed, "micro").physical_outflow, one, 1e-14);
    const auto escape = fixture.model.physical_escape_events();
    const auto histories = fixture.model.species_histories();
    fixture.model.restore(checkpoint);
    fixture.advance(.1);
    EXPECT_EQ(fixture.model.physical_escape_events(), escape);
    EXPECT_EQ(fixture.model.species_histories(), histories);
}

TEST(ReactingSpeciesPressureTest, CandidateLiquidVolumeIsPureAndLiquidEOSFailureIsAtomic)
{
    auto selected = options();
    selected.liquid_thermodynamics = R::make_linear_compressible_mixture_liquid(
        schema(), {250., 600., 10000., 150000.}, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300., 100000., 1000., 4200., 0., 1e-9, 0., .001, .6}; },
        R::EnergyConvention::SensibleEnergy);
    Fixture fixture(selected, 2);
    fixture.initialize(fixture.initial());
    const auto initial_history = fixture.model.species_histories();
    EXPECT_NEAR(fixture.model.evaluate_liquid_volume(120000.), 1. - 1e-9 * 20000., 1e-13);
    EXPECT_DOUBLE_EQ(fixture.model.absolute_pressure_offset(), 100000.);
    EXPECT_EQ(fixture.model.species_histories(), initial_history);
    EXPECT_ANY_THROW(fixture.model.set_absolute_pressure_offset(180000.));
    EXPECT_DOUBLE_EQ(fixture.model.absolute_pressure_offset(), 100000.);
    EXPECT_EQ(fixture.model.species_histories(), initial_history);
    for (std::size_t c = 0; c < fixture.grid->num_owned_cells(); ++c)
    {
        EXPECT_DOUBLE_EQ(fixture.model.absolute_pressure().value(static_cast<Pack::local_ordinal_type>(c)), 100000.);
        EXPECT_NEAR(fixture.model.pure_liquid_density().value(static_cast<Pack::local_ordinal_type>(c)), 1000., 1e-12);
    }
    fixture.model.set_absolute_pressure_offset(120000.);
    EXPECT_NEAR(fixture.sum(
                    [&]
                    {
                        double value = 0.;
                        for (std::size_t c = 0; c < fixture.grid->num_owned_cells(); ++c)
                            value += fixture.model.liquid_volume(static_cast<Pack::local_ordinal_type>(c));
                        return value;
                    }()),
        1. - 1e-9 * 20000., 1e-13);
    EXPECT_EQ(fixture.model.species_histories(), initial_history);
}

TEST(ReactingSpeciesPressureTest, GasEOSAndCapacityFailuresPublishNothingAfterValidLiquidQuery)
{
    auto selected = options();
    selected.gas_thermodynamics =
        R::make_ideal_gas_mixture(schema(), {{20., 300., 0.}, {20., 300., 0.}, {20., 300., 0.}, {25., 300., 0.}},
            {250., 600., 10000., 110000.}, R::EnergyConvention::SensibleEnergy);
    Fixture fixture(selected, 2);
    fixture.initialize(fixture.initial(20., 0., true), true);
    const auto histories = fixture.model.species_histories();
    const double bubble = fixture.model.evaluate_submerged_bubble_volume(100000.);
    EXPECT_GT(fixture.model.evaluate_liquid_volume(120000.), 0.);
    EXPECT_ANY_THROW(fixture.model.set_absolute_pressure_offset(120000.)); // gas EOS supplied domain
    EXPECT_ANY_THROW(fixture.model.set_absolute_pressure_offset(20000.));  // valid EOS, excessive gas volume
    EXPECT_DOUBLE_EQ(fixture.model.absolute_pressure_offset(), 100000.);
    EXPECT_EQ(fixture.model.species_histories(), histories);
    EXPECT_DOUBLE_EQ(fixture.model.evaluate_submerged_bubble_volume(100000.), bubble);
    if (fixture.grid->owned_cell_map()->getComm()->getSize() > 1)
    {
        const auto comm = fixture.grid->owned_cell_map()->getComm();
        const double candidate = comm->getRank() == comm->getSize() - 1 ? 100001. : 100000.;
        EXPECT_ANY_THROW((void) fixture.model.evaluate_liquid_volume(candidate));
        EXPECT_ANY_THROW(fixture.model.set_absolute_pressure_offset(candidate));
        EXPECT_DOUBLE_EQ(fixture.model.absolute_pressure_offset(), 100000.);
    }
}

TEST(ReactingSpeciesPressureTest, ExactlyEmptyLiquidSkipsItsCaloricDomainAndKeepsZeroVolume)
{
    auto selected = options();
    selected.liquid_thermodynamics = std::make_shared<const R::MixtureThermodynamics>(
        schema(), R::ThermodynamicBounds{250., 600., 10000., 2000000.}, R::EOSKind::Liquid,
        [](double, double, std::span<const double>) -> R::MixtureThermodynamicPoint
        { throw std::logic_error("Empty liquid must never call its EOS."); }, R::EnergyConvention::SensibleEnergy);
    Fixture fixture(selected, 2);
    auto states = fixture.initial();
    const auto& catalog = selected.reaction_mechanism->catalog();
    for (auto& state : states)
        state.set_amount(catalog, donor, R::SpeciesPhase::Liquid, 0.);
    fixture.initialize(states);
    EXPECT_DOUBLE_EQ(fixture.model.evaluate_liquid_volume(120000.), 0.);
    fixture.model.set_absolute_pressure_offset(120000.);
    EXPECT_DOUBLE_EQ(fixture.model.evaluate_liquid_volume(120000.), 0.);
    for (const auto& report : fixture.model.species_histories())
    {
        EXPECT_DOUBLE_EQ(report.initial_moles, 0.);
        EXPECT_DOUBLE_EQ(report.current_moles, 0.);
    }
}

TEST(ReactingSpeciesHistoryTest, PhaseOnlyStoichiometryDoesNotInventMolecularProduction)
{
    auto selected = options();
    selected.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(
        schema(), std::vector<R::ReactionDefinition>{{19, "same molecule changes compartment",
                      {{gasOne, R::SpeciesPhase::Dissolved, -1, {}}, {gasOne, R::SpeciesPhase::Resolved, 1, {}}}, 0.}});
    selected.species_source = [](const R::ReactionContext& context, double)
    {
        R::ReactionSourceInput source;
        source.integratedExtents = {{19, .005 * context.environment.bulkVolume}};
        return source;
    };
    Fixture fixture(selected, 2);
    fixture.initialize(fixture.initial(.01, 0., false, .99));
    expect_liquid_headroom(fixture);
    fixture.advance();
    EXPECT_DOUBLE_EQ(history(fixture.model, gasOne).chemical_production, 0.);
    EXPECT_DOUBLE_EQ(history(fixture.model, gasOne).chemical_consumption, 0.);
    EXPECT_NEAR(history(fixture.model, gasOne).current_moles, .01, 1e-14);
    EXPECT_NEAR(events(fixture.model, gasOne, R::SpeciesPhase::Dissolved).chemical_consumption, .005, 1e-15);
    EXPECT_NEAR(events(fixture.model, gasOne, R::SpeciesPhase::Resolved).chemical_production, .005, 1e-15);
}

TEST(ReactingSpeciesHistoryTest, AcceptedForwardAndReverseRateStagesRetainGrossTurnoverAtZeroNetExtent)
{
    auto selected = options();
    selected.reaction_advance.maximumSubstep = .01;
    selected.species_source = [](const R::ReactionContext& context, double)
    {
        R::ReactionRateProvider provider;
        provider.identity = "manufactured-sign-changing-bulk-rate";
        provider.schemaRevision = context.catalog.revision();
        provider.schemaFingerprint = context.catalog.fingerprint();
        provider.validityDescription = "Linear time rate over [0,0.01] with initially finite reverse donors.";
        provider.parameterUnits = "mol/(m3 bulk s)";
        provider.validate = [](const auto&) {};
        provider.evaluate = [](const auto& stage)
        {
            return std::vector<R::ReactionRate>{
                {8, 2. * stage.environment.time - .01, R::SourceVolumeBasis::BulkVolume}};
        };
        R::ReactionSourceInput source;
        source.rates = std::move(provider);
        return source;
    };
    Fixture fixture(selected, 2);
    auto initial = fixture.initial(.1, .1, false, .99);
    const auto& catalog = selected.reaction_mechanism->catalog();
    for (std::size_t c = 0; c < initial.size(); ++c)
    {
        const double volume = fixture.grid->cell_volume(static_cast<Pack::local_ordinal_type>(c));
        const auto donor_amount = initial[c].amount(catalog, donor, R::SpeciesPhase::Liquid);
        initial[c].set_amount(catalog, donor, R::SpeciesPhase::Liquid, donor_amount - .04 * volume);
        initial[c].set_amount(catalog, liquidProduct, R::SpeciesPhase::Liquid, .1 * volume);
    }
    fixture.initialize(initial);
    expect_liquid_headroom(fixture);
    fixture.advance(.01);
    for (const auto species : {donor, liquidProduct, gasOne, gasTwo})
    {
        const auto& report = history(fixture.model, species);
        EXPECT_NEAR(report.chemical_production, .00005, 1e-16);
        EXPECT_NEAR(report.chemical_consumption, .00005, 1e-16);
        EXPECT_NEAR(report.current_moles, report.initial_moles, 2e-10);
    }
    for (std::size_t c = 0; c < initial.size(); ++c)
    {
        const double volume = fixture.grid->cell_volume(static_cast<Pack::local_ordinal_type>(c));
        const auto& ledger = fixture.model.source_ledgers()[c];
        EXPECT_NEAR(ledger.forwardReactionExtents[0], .00005 * volume, 1e-17);
        EXPECT_NEAR(ledger.reverseReactionExtents[0], .00005 * volume, 1e-17);
        EXPECT_NEAR(ledger.reactionExtents[0], 0., 1e-17);
        EXPECT_NEAR(ledger.reactionHeat, 0., 1e-15);
    }
}

TEST(ReactingSpeciesPressureTest, RankLocalGasEOSFailureRejectsCollectivelyAndRestoresPressureReplay)
{
    auto selected = options();
    selected.pressure_mode = S::ReactingPressureMode::Reconstructed;
    selected.gas_thermodynamics =
        R::make_ideal_gas_mixture(schema(), {{20., 300., 0.}, {20., 300., 0.}, {20., 300., 0.}, {25., 300., 0.}},
            {250., 600., 10000., 125000.}, R::EnergyConvention::SensibleEnergy);
    Fixture fixture(selected, 2);
    for (std::size_t owned = 0; owned < fixture.grid->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        fixture.pressure.set_owned_value(cell, fixture.grid->cell_global_id(cell) == 0 ? 20000. : 0.);
    }
    fixture.pressure.sync_ghosts();
    fixture.initialize(fixture.initial(.01, .02, true), true);
    const auto accepted = fixture.model.snapshot();
    const auto histories = fixture.model.species_histories();
    const double volume = fixture.model.evaluate_submerged_bubble_volume(100000.);
    // At candidate offset 120000, only global cell zero reaches 130000 Pa.
    // Reconstruction centers gauge [20000,0] to [10000,-10000].
    // Other owners have a valid 110000 Pa gas state, and MPI4 has two empty ranks.
    EXPECT_GT(fixture.model.evaluate_liquid_volume(120000.), 0.);
    EXPECT_ANY_THROW(fixture.model.set_absolute_pressure_offset(120000.));
    EXPECT_DOUBLE_EQ(fixture.model.absolute_pressure_offset(), 100000.);
    EXPECT_EQ(fixture.model.species_histories(), histories);
    for (std::size_t owned = 0; owned < fixture.grid->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(
            fixture.model.absolute_pressure().value(cell), fixture.grid->cell_global_id(cell) == 0 ? 110000. : 90000.);
    }
    EXPECT_DOUBLE_EQ(fixture.model.evaluate_submerged_bubble_volume(100000.), volume);
    EXPECT_NO_THROW(fixture.model.set_absolute_pressure_offset(105000.));
    fixture.model.restore(accepted);
    EXPECT_DOUBLE_EQ(fixture.model.absolute_pressure_offset(), 100000.);
    EXPECT_DOUBLE_EQ(fixture.model.evaluate_submerged_bubble_volume(100000.), volume);
    EXPECT_EQ(fixture.model.species_histories(), histories);
}

TEST(ReactingSpeciesHistoryTest, NetEventReportSurvivesRoundedOpposingGrossReservoirTotals)
{
    auto selected = options();
    selected.species_source = [](const R::ReactionContext& context, double)
    {
        const double stock = context.state.amount(context.catalog, donor, R::SpeciesPhase::Liquid);
        R::ReactionSourceInput input;
        input.externalTransfers = {{"large supply", donor, R::SpeciesPhase::Liquid, stock, .010 * stock, 0., {}, {}},
            {"small independent supply", donor, R::SpeciesPhase::Liquid, 1e-12, 1e-14, 0., {}, {}},
            {"large withdrawal", donor, R::SpeciesPhase::Liquid, -stock, -.010 * stock, 0., {}, {}}};
        return input;
    };
    Fixture fixture(selected, 2);
    fixture.initialize(fixture.initial());
    fixture.advance();
    auto first = history(fixture.model, donor);
    EXPECT_DOUBLE_EQ(first.external_supply, first.external_withdrawal);
    EXPECT_DOUBLE_EQ(first.current_moles, first.initial_moles);
    // Two real 1e-12 mol supplies are below each cell's inventory ULP. They must
    // remain visible in the physical residual, even though gross reports cancel.
    EXPECT_NEAR(first.inventory_error, 2e-12, 1e-25);
    const auto saved = fixture.model.snapshot();
    fixture.advance();
    const auto second = history(fixture.model, donor);
    EXPECT_NEAR(second.inventory_error, 4e-12, 1e-25);
    fixture.model.restore(saved);
    EXPECT_EQ(history(fixture.model, donor), first);
    fixture.advance();
    EXPECT_EQ(history(fixture.model, donor), second);
}
TEST(ReactingSpeciesHistoryTest, FieldThermalSourceUsesAcceptedHeatDespiteEqualRoundedReactionExtents)
{
    using P = R::SpeciesPhase;
    auto selected = options();
    const auto catalog = schema();
    const double large = std::ldexp(1., 60), quantum = std::ldexp(1., -60);
    selected.reaction_mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
        std::vector<R::ReactionDefinition>{
            {41, "forward finite isomerization",
                {{donor, P::Liquid, -.4 * quantum}, {liquidProduct, P::Liquid, quantum}}, 1.},
            {83, "opposing finite isomerization",
                {{liquidProduct, P::Liquid, -quantum}, {donor, P::Liquid, .4 * quantum}}, -1.}});
    selected.reaction_advance.maximumSubstep = .5;
    selected.species_source = [large](const R::ReactionContext& context, double)
    {
        R::ReactionRateProvider provider;
        provider.identity = "finite normalized field cancellation";
        provider.schemaRevision = context.catalog.revision();
        provider.schemaFingerprint = context.catalog.fingerprint();
        provider.validityDescription = "manufactured t in [0,1], finite opposing donors";
        provider.parameterUnits = "mol/(m3 bulk s)";
        provider.validate = [](const auto&) {};
        provider.evaluate = [large](const R::ReactionContext& stage)
        {
            const double time = stage.environment.time, volume = stage.environment.bulkVolume;
            return std::vector<R::ReactionRate>{
                {41, (time == 0 ? 4 * large : (time == 1 ? 4. : 0.)) / volume,
                    R::SourceVolumeBasis::BulkVolume},
                {83, (time == 0 ? 4 * large : 0.) / volume, R::SourceVolumeBasis::BulkVolume}};
        };
        R::ReactionSourceInput source;
        source.rates = std::move(provider);
        return source;
    };
    Fixture fixture(selected, 2);
    auto initial = fixture.initial(0., 0., false, .99);
    for (auto& cell : initial)
        cell.set_amount(*catalog, liquidProduct, P::Liquid, 2.);
    fixture.initialize(initial);
    expect_liquid_headroom(fixture);
    const auto saved = fixture.model.snapshot();
    const auto check_source = [&]
    {
        double energy = 0.;
        for (std::size_t c = 0; c < initial.size(); ++c)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(c);
            const double volume = fixture.grid->cell_volume(cell);
            const auto& ledger = fixture.model.source_ledgers()[c];
            EXPECT_EQ(ledger.reactionExtents.size(), 2U);
            if (ledger.reactionExtents.size() == 2)
            {
                EXPECT_DOUBLE_EQ(ledger.reactionExtents[0], large);
                EXPECT_DOUBLE_EQ(ledger.reactionExtents[1], large);
            }
            EXPECT_DOUBLE_EQ(ledger.reactionHeat, 1.);
            EXPECT_DOUBLE_EQ(fixture.model.thermal_source().value(cell), 1. / volume); // dt=1, exactly one J/cell.
            energy += fixture.model.thermal_source().value(cell) * volume;
        }
        EXPECT_DOUBLE_EQ(fixture.sum(energy), 2.);
    };
    fixture.advance(1.);
    check_source();
    fixture.model.restore(saved);
    fixture.advance(1.);
    check_source();
    const auto ranks = fixture.grid->owned_cell_map()->getComm()->getSize();
    EXPECT_EQ(fixture.sum(fixture.grid->num_owned_cells() == 0 ? 1. : 0.),
        static_cast<double>(std::max(0, ranks - 2)));
}

TEST(ReactingSpeciesHistoryTest, RawVolumetricHeatAndCarriedEnergyCancelBeforeFieldSourceNarrowingAndReplay)
{
    const double factor = std::ldexp(1., 30), large = std::ldexp(1., 60);
    auto selected = options();
    selected.species_source = [factor, large](const R::ReactionContext&, double)
    {
        R::ReactionSourceInput source;
        source.heat.rate = R::VolumetricHeatRate{factor + 1., R::SourceVolumeBasis::BulkVolume};
        // A real finite species withdrawal carries this declared external energy packet.
        source.externalTransfers = {{"manufactured carried-energy withdrawal", donor, R::SpeciesPhase::Liquid,
            -1., -.010, -large, {}, {}}};
        return source;
    };
    Fixture fixture(selected, 1, factor - 1.);
    fixture.initialize(fixture.initial(0., 0., false, .99));
    expect_liquid_headroom(fixture);
    const auto saved = fixture.model.snapshot();
    const auto check_source = [&]
    {
        double energy = 0.;
        for (std::size_t c = 0; c < fixture.grid->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(c);
            const double volume = fixture.grid->cell_volume(cell);
            EXPECT_DOUBLE_EQ(volume, factor - 1.);
            const auto& ledger = fixture.model.source_ledgers()[c];
            EXPECT_DOUBLE_EQ(ledger.externalHeat, large); // Reports round the actual 2^60-1 J.
            EXPECT_DOUBLE_EQ(ledger.transferEnergy, -large);
            EXPECT_DOUBLE_EQ(ledger.thermal_energy(), 0.); // This reporting route loses the finite cooling packet.
            EXPECT_DOUBLE_EQ(fixture.model.thermal_source().value(cell), -1. / volume);
            energy += fixture.model.thermal_source().value(cell) * volume;
        }
        EXPECT_NEAR(fixture.sum(energy), -1., 3e-16);
    };
    fixture.advance(1.);
    check_source();
    fixture.model.restore(saved);
    fixture.advance(1.);
    check_source();
    const auto ranks = fixture.grid->owned_cell_map()->getComm()->getSize();
    EXPECT_EQ(fixture.sum(fixture.grid->num_owned_cells() == 0 ? 1. : 0.),
        static_cast<double>(ranks - 1));
}

} // namespace
