#include "equations/BubblePopulationKinetics.hh"
#include "equations/ReactingGasModel.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

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

std::shared_ptr<Mesh> mesh(std::size_t cells = 4)
{
    std::vector<double> edges(cells + 1);
    for (std::size_t i = 0; i <= cells; ++i)
        edges[i] = static_cast<double>(i) / cells;
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
    explicit Fixture(S::ReactingGasOptions selected = options(), std::size_t cells = 4)
        : grid(mesh(cells)), settings(std::move(selected)), model(grid, settings), temperature(grid, 300., "T"),
          pressure(grid, 0., "p"), velocity(grid, S::vec3<double>{}, "U"), flux(grid, 0., "phi"),
          material(grid, material_options(), S::TimeStepperOptions{})
    {
    }
    std::vector<R::CellSpeciesState> initial(double g1 = 0., double g2 = 0., bool gas = false) const
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
                catalog, donor, R::SpeciesPhase::Liquid, (1000. * (volume - gasVolume) - dissolvedMass) / .010);
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

TEST(ReactingSpeciesModelTest, FiniteDonorMultigasSourcesAreIndependentOfPrescribedHeatAndReplay)
{
    auto selected = options();
    selected.species_source = [](const R::ReactionContext& context, double)
    {
        R::ReactionSourceInput source;
        source.integratedExtents = {{8, .25 * context.environment.bulkVolume}};
        source.heat.integratedEnergy = -2. * context.environment.bulkVolume;
        return source;
    };
    Fixture fixture(selected);
    fixture.initialize(fixture.initial());
    const auto checkpoint = fixture.model.snapshot();
    fixture.advance(2.);
    EXPECT_NEAR(fixture.model.global_species_moles(gasOne), .25, 1e-13);
    EXPECT_NEAR(fixture.model.global_species_moles(gasTwo), .25, 1e-13);
    EXPECT_NEAR(fixture.model.global_species_moles(liquidProduct), .25, 1e-13);
    EXPECT_NEAR(fixture.model.global_species_moles(donor), 99999.75, 1e-9);
    std::vector<std::vector<double>> first;
    for (std::size_t owned = 0; owned < fixture.grid->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_NEAR(fixture.model.thermal_source().value(cell), 11.5, 1e-12);
        first.push_back(fixture.model.cell_species_state(cell).amounts);
    }
    fixture.model.restore(checkpoint);
    fixture.advance(2.);
    for (std::size_t owned = 0; owned < first.size(); ++owned)
        EXPECT_EQ(fixture.model.cell_species_state(static_cast<Pack::local_ordinal_type>(owned)).amounts, first[owned]);
}

TEST(ReactingSpeciesModelTest, MixedGasNucleationPairsEachSpeciesWithOnePhysicalNumberSource)
{
    auto selected = options();
    selected.species_source = [](const R::ReactionContext& context, double)
    {
        R::ReactionSourceInput source;
        source.integratedExtents = {{8, .02 * context.environment.bulkVolume}};
        return source;
    };
    selected.population_process = [](const B::MultispeciesPopulationContext& context, double)
    {
        B::MultispeciesPopulationPacket result;
        const auto volume = context.chemistry.environment.bulkVolume;
        for (const auto id : {gasOne, gasTwo})
            result.transfers.push_back(
                {id, R::SpeciesPhase::Dissolved, R::SpeciesPhase::Dispersed, {}, "micro", .01 * volume});
        result.numberEvents = {{"micro", 1e6 * volume, 0.}};
        return result;
    };
    Fixture fixture(selected);
    fixture.initialize(fixture.initial());
    fixture.advance();
    double volume = 0.;
    for (std::size_t owned = 0; owned < fixture.grid->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto bulk = fixture.grid->cell_volume(cell);
        EXPECT_NEAR(fixture.model.species_field(gasOne, R::SpeciesPhase::Dispersed, "micro").value(cell), .01, 1e-14);
        EXPECT_NEAR(fixture.model.species_field(gasTwo, R::SpeciesPhase::Dissolved).value(cell), .01, 1e-14);
        EXPECT_NEAR(fixture.model.population_number_field("micro").value(cell), 1e6, 1e-7);
        volume += fixture.model.raw_bubble_volume_fraction().value(cell) * bulk;
    }
    EXPECT_NEAR(fixture.sum(volume), .02 * gasConstant * 300. / 100000., 1e-12);
}

TEST(ReactingSpeciesModelTest, SpeciesSpecificFugacityTransferGrowsOneGasAndDissolvesAnother)
{
    auto selected = options();
    selected.species_properties[0].diffusivity = 1e-8;
    selected.species_properties[0].henry_coefficient = 1e-6;
    selected.species_properties[1].diffusivity = 2e-8;
    selected.species_properties[1].henry_coefficient = 2e-6;
    Fixture fixture(selected);
    auto initial = fixture.initial(.01, .03, true);
    const auto& catalog = selected.reaction_mechanism->catalog();
    for (std::size_t owned = 0; owned < initial.size(); ++owned)
        initial[owned].set_amount(catalog, gasOne, R::SpeciesPhase::Dissolved,
            .1 * fixture.grid->cell_volume(static_cast<Pack::local_ordinal_type>(owned)));
    fixture.initialize(initial, true);
    fixture.advance(.01);
    EXPECT_NEAR(fixture.model.global_species_moles(gasOne), .11, 1e-12);
    EXPECT_NEAR(fixture.model.global_species_moles(gasTwo), .03, 1e-12);
    for (std::size_t owned = 0; owned < initial.size(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_GT(fixture.model.species_field(gasOne, R::SpeciesPhase::Dispersed, "micro").value(cell), .01);
        EXPECT_LT(fixture.model.species_field(gasTwo, R::SpeciesPhase::Dispersed, "micro").value(cell), .03);
        EXPECT_GT(fixture.model.species_field(gasTwo, R::SpeciesPhase::Dissolved).value(cell), 0.);
    }
}

TEST(ReactingSpeciesModelTest, ActualMixedSpeciesSlipEscapeCarriesIndependentMolarAndEnergyBudgets)
{
    auto selected = options();
    selected.rise_velocity_mode = S::BubbleRiseVelocityMode::ConstantSlip;
    selected.constant_slip_velocity = .3;
    selected.free_surface_patches = {"zmax"};
    Fixture fixture(selected);
    fixture.initialize(fixture.initial(.01, .02, true), true);
    fixture.advance(.1);
    double one = 0., two = 0., energy = 0.;
    for (const auto& ledger : fixture.model.escape_ledgers())
    {
        one -= ledger.externalChange.phase_amount(
            selected.reaction_mechanism->catalog(), gasOne, R::SpeciesPhase::Dispersed);
        two -= ledger.externalChange.phase_amount(
            selected.reaction_mechanism->catalog(), gasTwo, R::SpeciesPhase::Dispersed);
        energy += ledger.transferEnergy;
    }
    one = fixture.sum(one);
    two = fixture.sum(two);
    energy = fixture.sum(energy);
    EXPECT_GT(one, 0.);
    EXPECT_GT(two, 0.);
    EXPECT_NEAR(two, 2 * one, 1e-13);
    EXPECT_NEAR(fixture.model.global_species_moles(gasOne) + one, .01, 1e-12);
    EXPECT_NEAR(fixture.model.global_species_moles(gasTwo) + two, .02, 1e-12);
    EXPECT_NEAR(energy, -100. * one - 200. * two, 1e-11);
}

TEST(ReactingSpeciesModelTest, JointDonorFailureRestoresEveryDynamicFieldAndSourceBudget)
{
    auto selected = options();
    selected.species_source = [](const R::ReactionContext& context, double)
    {
        R::ReactionSourceInput source;
        source.integratedExtents = {{8, 1e9 * context.environment.bulkVolume}};
        return source;
    };
    Fixture fixture(selected);
    const auto initial = fixture.initial();
    fixture.initialize(initial);
    EXPECT_ANY_THROW(fixture.advance());
    EXPECT_NEAR(fixture.model.global_species_moles(donor), 100000., 1e-9);
    EXPECT_DOUBLE_EQ(fixture.model.global_species_moles(gasOne), 0.);
    EXPECT_TRUE(fixture.model.source_ledgers().empty());
    for (std::size_t owned = 0; owned < initial.size(); ++owned)
        EXPECT_EQ(fixture.model.cell_species_state(static_cast<Pack::local_ordinal_type>(owned)).amounts,
            initial[owned].amounts);
}

TEST(ReactingSpeciesModelTest, CollectiveInitializationRejectsOneRankSchemaFailureAndSupportsEmptyOwners)
{
    Fixture fixture(options(), 1);
    EXPECT_EQ(fixture.sum(fixture.grid->num_owned_cells() == 0 ? 1. : 0.),
        static_cast<double>(fixture.grid->owned_cell_map()->getComm()->getSize() - 1));
    auto initial = fixture.initial();
    if (!initial.empty())
        initial.front().schemaFingerprint ^= 1;
    EXPECT_ANY_THROW(fixture.model.initialize_species(initial));
    initial = fixture.initial();
    fixture.initialize(initial);
    fixture.advance();
    EXPECT_NEAR(fixture.model.global_species_moles(donor), 100000., 1e-9);
}

TEST(ReactingSpeciesModelTest, ALEUsesAcceptedOldAndTrialNewVolumesAndRestoresAcrossEpochChanges)
{
    auto selected = options();
    selected.species_source = [](const R::ReactionContext& context, double)
    {
        R::ReactionSourceInput source;
        R::ReactionRateProvider provider;
        provider.identity = "manufactured liquid-volume rate";
        provider.schemaRevision = context.catalog.revision();
        provider.schemaFingerprint = context.catalog.fingerprint();
        provider.validityDescription = "Manufactured constant rate in physical liquid.";
        provider.parameterUnits = "mol/(m3 liquid s)";
        provider.validate = [](const auto&) {};
        provider.evaluate = [](const auto&)
        { return std::vector<R::ReactionRate>{{8, 1., R::SourceVolumeBasis::LiquidVolume}}; };
        source.rates = std::move(provider);
        return source;
    };
    Fixture fixture(selected);
    fixture.initialize(fixture.initial());
    const auto checkpoint = fixture.model.snapshot();
    S::PlanarALEMeshMotion<Pack> motion(fixture.grid);
    for (int replay = 0; replay < 2; ++replay)
    {
        motion.begin_trial(1.1, .1);
        const auto ale = S::FVM::make_ale_control_volume_state(*fixture.grid, motion);
        fixture.model.refresh_geometry();
        fixture.model.advance(.1, .1, fixture.temperature, fixture.pressure, fixture.velocity, fixture.flux,
            fixture.material, nullptr, &ale);
        EXPECT_NEAR(fixture.model.global_species_moles(gasOne), .1, 2e-12);
        EXPECT_NEAR(fixture.model.global_species_moles(gasTwo), .1, 2e-12);
        for (std::size_t owned = 0; owned < fixture.grid->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(owned);
            EXPECT_NEAR(fixture.model.liquid_mass_density().value(cell), 1000. / 1.1, 2e-9);
            EXPECT_NEAR(fixture.model.pure_liquid_density().value(cell), 1000., 1e-10);
        }
        motion.rollback_trial();
        fixture.model.restore(checkpoint); // Rollback changed the epoch; accepted inventories still restore.
        EXPECT_NEAR(fixture.model.global_species_moles(donor), 100000., 1e-9);
        EXPECT_DOUBLE_EQ(fixture.model.global_species_moles(gasOne), 0.);
        EXPECT_TRUE(fixture.model.source_ledgers().empty());
    }
}

TEST(ReactingSpeciesModelTest, LegacyMeshExecutesTheSameSchemaIndexedSourcePath)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Legacy mesh fixture is serial; native cases cover MPI.";
    auto selected = options();
    selected.species_source = [](const R::ReactionContext& context, double)
    {
        R::ReactionSourceInput source;
        source.integratedExtents = {{8, .3 * context.environment.bulkVolume}};
        return source;
    };
    auto grid = S::test::build_mesh<Pack>(S::test::make_single_hex_database());
    S::ReactingGasModel<Pack> model(grid, selected);
    S::CellField<Pack> temperature(grid, "T"), pressure(grid, "p");
    S::VectorCellField<Pack> velocity(grid, "U");
    S::FaceField<Pack> flux(grid, "phi");
    temperature.put_scalar(300.);
    pressure.put_scalar(0.);
    velocity.put_scalar(S::vec3<double>{});
    flux.put_scalar(0.);
    S::MaterialPropertyFields<Pack> material(grid, material_options(), S::TimeStepperOptions{});
    std::vector<R::CellSpeciesState> states;
    for (std::size_t owned = 0; owned < grid->num_owned_cells(); ++owned)
    {
        R::CellSpeciesState state(selected.reaction_mechanism->catalog());
        state.dispersedPopulations = {{"micro", std::vector<double>(4)}, {"large", std::vector<double>(4)}};
        state.set_amount(selected.reaction_mechanism->catalog(), donor, R::SpeciesPhase::Liquid,
            100000. * grid->cell_volume(static_cast<Pack::local_ordinal_type>(owned)));
        states.push_back(std::move(state));
    }
    model.initialize_species(states);
    model.initialize_state(0., temperature, pressure, velocity, material);
    model.advance(.1, .1, temperature, pressure, velocity, flux, material, nullptr);
    EXPECT_NEAR(model.global_species_moles(gasOne), .3, 1e-12);
    EXPECT_NEAR(model.global_species_moles(gasTwo), .3, 1e-12);
    EXPECT_NEAR(model.global_species_moles(donor), 99999.7, 1e-9);
}
TEST(ReactingSpeciesModelTest, RankAsymmetricSchemaPresenceAndTransportMetadataRejectBeforeAllocation)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "This rejection regression requires at least two MPI ranks.";
    auto selected = options();
    if (comm->getRank() != 0)
    {
        selected.mode = S::ReactingGasMode::Disabled;
        selected.reaction_mechanism.reset();
    }
    auto grid = mesh();
    EXPECT_ANY_THROW(Model rejected(grid, selected));
    selected = options();
    selected.species_properties[0].diffusivity = comm->getRank() == 0 ? 1e-8 : 0.;
    EXPECT_ANY_THROW(Model rejected(grid, selected));
    selected = options();
    if (comm->getRank() == 0)
        selected.species_source = [](const auto&, double) { return R::ReactionSourceInput{}; };
    EXPECT_ANY_THROW(Model rejected(grid, selected));
    selected = options();
    if (comm->getRank() == 0)
        selected.species_properties[0].diffusivity_provider = [](const auto&) { return 1e-8; };
    EXPECT_ANY_THROW(Model rejected(grid, selected));
    Fixture valid(options());
    valid.initialize(valid.initial());
    valid.advance();
    EXPECT_NEAR(valid.model.global_species_moles(donor), 100000., 1e-9);
}
TEST(ReactingSpeciesModelTest, StationaryResolvedCompartmentRetainsExtensiveMolesUnderALE)
{
    Fixture fixture;
    auto initial = fixture.initial();
    const auto& catalog = fixture.settings.reaction_mechanism->catalog();
    for (std::size_t owned = 0; owned < initial.size(); ++owned)
        initial[owned].set_amount(catalog, gasOne, R::SpeciesPhase::Resolved,
            .01 * fixture.grid->cell_volume(static_cast<Pack::local_ordinal_type>(owned)));
    fixture.initialize(initial);
    S::PlanarALEMeshMotion<Pack> motion(fixture.grid);
    motion.begin_trial(1.1, .1);
    const auto ale = S::FVM::make_ale_control_volume_state(*fixture.grid, motion);
    fixture.model.refresh_geometry();
    fixture.model.advance(
        .1, .1, fixture.temperature, fixture.pressure, fixture.velocity, fixture.flux, fixture.material, nullptr, &ale);
    EXPECT_NEAR(fixture.model.global_species_moles(gasOne), .01, 1e-12);
    EXPECT_NEAR(fixture.model.global_species_moles(donor), 100000., 1e-9);
    motion.accept_trial();
}
TEST(ReactingSpeciesModelTest, UnclosedBulkMixtureDiffusionIsAnExplicitRejectedSelection)
{
    auto selected = options();
    EXPECT_EQ(selected.bulk_diffusion, S::ReactingBulkDiffusionMode::Disabled);
    selected.species_properties[0].diffusivity = 1e-8; // Valid independent film-transfer data.
    EXPECT_NO_THROW(Fixture filmOnly(selected));
    selected.bulk_diffusion = S::ReactingBulkDiffusionMode::ConservativeMixture;
    EXPECT_ANY_THROW(Fixture unclosedBulkDiffusion(selected));
}
} // namespace
