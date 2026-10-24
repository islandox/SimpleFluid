/** Source-only location draft: native field/controller and trial-new geometry oracles. */
#include "FVM/ALEControlVolumeState.hh"
#include "equations/ReactingGasModel.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/ReactingMixtureCoupler.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
namespace S = SimpleFluid;
namespace R = S::Reacting;
using Pack = S::DefaultTpetraTypes;
using Grid = S::MeshHandle<Pack>;
using LO = Pack::local_ordinal_type;
using Model = S::ReactingGasModel<Pack, Grid>;
using Flow = R::LowMachMixtureSolver;
constexpr R::SpeciesId left = 37, right = 903;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
S::SP<Grid> two_cells()
{
    Grid::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    return std::make_shared<Grid>(std::make_shared<Grid::Cartesian>(
        S::Vec3D<S::ArrReal>{{{0., 1.}, {0., 1.}, {0., .5, 1.}}}), distribution);
}
S::SP<Grid> hybrid_cells()
{
    using Semi = S::Meshes::SemiStructuredXY_Z;
    S::Arr<Semi::Vec3> vertices;
    for (unsigned x = 0; x <= 4; ++x)
    {
        vertices.push_back({double(x), 0., 0.});
        vertices.push_back({double(x), 1., 0.});
    }
    S::Arr<S::Arr<unsigned>> cells;
    for (const unsigned x : {3U, 0U, 2U, 1U})
        cells.push_back({2 * x, 2 * x + 2, 2 * x + 3, 2 * x + 1});
    Grid::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    const auto ranks = Tpetra::getDefaultComm()->getSize();
    if (ranks == 2 || ranks == 4)
    {
        distribution.xy_partitions = 2;
        distribution.z_partitions = ranks / 2;
    }
    return std::make_shared<Grid>(
        std::make_shared<Semi>(std::move(vertices), std::move(cells), S::ArrReal{0., .4, 1.}), distribution);
}
double source_shape(const R::ReactionLocation& location)
{
    return 1. + location.centroid[0] + 2. * location.centroid[2];
}
void validate_location(const R::ReactionEnvironment& environment, const Grid& grid)
{
    require(environment.location && environment.location->ownedCell, "A mesh source requires its owned location.");
    const auto& location = *environment.location;
    require(*location.ownedCell < grid.num_owned_cells(), "Source owned row is outside this frame.");
    const auto cell = static_cast<LO>(*location.ownedCell);
    const auto center = grid.cell_centroid(cell);
    require(location.geometryCellId == static_cast<std::uint64_t>(grid.cell_geometry_global_id(cell)),
        "Source canonical identity was confused with an algebraic GID or local row.");
    require(location.geometryEpoch == grid.geometry_epoch(), "Source location retained stale geometry.");
    require(location.centroid == std::array<double, 3>{center.x, center.y, center.z},
        "Source location does not describe the actual donor centroid.");
}
struct Chemistry
{
    std::shared_ptr<const R::SpeciesCatalog> catalog = std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{left, "left", 1., {{"X", 1}}, {R::SpeciesPhase::Liquid}},
            {right, "right", 1., {{"X", 1}}, {R::SpeciesPhase::Liquid}}});
    std::shared_ptr<const R::ReactionMechanism> mechanism = std::make_shared<const R::ReactionMechanism>(catalog,
        std::vector<R::ReactionDefinition>{{41, "isomerization",
            {{left, R::SpeciesPhase::Liquid, -1}, {right, R::SpeciesPhase::Liquid, 1}}, 0.}});
    std::shared_ptr<const R::MixtureThermodynamics> liquid = R::make_linear_compressible_mixture_liquid(
        catalog, {250., 500., 10000., 2000000.}, [](std::span<const double>)
        { return R::LinearCompressibleMixtureLiquid{300., 100000., 1., 8., 0., 1e-9, 0., 0., 0.}; },
        R::EnergyConvention::SensibleEnergy);
    std::shared_ptr<const R::MixtureThermodynamics> gas = R::make_ideal_gas_mixture(
        catalog, {{20., 300., 0.}, {20., 300., 0.}}, {250., 500., 10000., 2000000.},
        R::EnergyConvention::SensibleEnergy);
};
R::ReactionSourceInput spatial_source(const R::ReactionContext& context, double dt, const Grid& grid)
{
    validate_location(context.environment, grid);
    const double shape = source_shape(*context.environment.location);
    R::ReactionSourceInput input;
    input.integratedExtents = {{41, .01 * shape * context.environment.bulkVolume * dt}};
    input.heat.rate = R::VolumetricHeatRate{4. * shape, R::SourceVolumeBasis::BulkVolume};
    return input;
}
S::ReactingGasOptions field_options(const Chemistry& chemistry, const S::SP<Grid>& grid)
{
    S::ReactingGasOptions options;
    options.mode = S::ReactingGasMode::TwoPopulation;
    options.reference_pressure = 100000.;
    options.reaction_mechanism = chemistry.mechanism;
    options.liquid_thermodynamics = chemistry.liquid;
    options.gas_thermodynamics = chemistry.gas;
    options.mass_transfer_mode = S::BubbleMassTransferMode::SphericalDiffusion;
    options.dissolved_transport = S::ReactingTransportMode::Advective;
    options.initial_species_state = [schema = chemistry.catalog, grid](const auto& environment, std::int64_t)
    {
        validate_location(environment, *grid);
        require(environment.liquidVolume == 0., "Initial callback liquid-volume contract changed.");
        require(environment.volumeClosure == R::ReactionVolumeClosure::ReferenceControlVolume,
            "Field initialization lost its explicit weak control-volume contract.");
        R::CellSpeciesState state(*schema);
        state.dispersedPopulations = {{"micro", std::vector<double>(2)}, {"large", std::vector<double>(2)}};
        const double z = environment.location->centroid[2];
        state.set_amount(*schema, left, R::SpeciesPhase::Liquid, (.8 + .1 * z) * environment.bulkVolume);
        return state; // Explicit physical headroom; no occupancy tolerance or fake volume basis.
    };
    options.initial_population_numbers = [](const auto&) { return std::vector<double>{0., 0.}; };
    options.species_source = [grid](const auto& context, double dt)
    {
        require(context.environment.volumeClosure == R::ReactionVolumeClosure::ReferenceControlVolume,
            "Field source lost its explicit weak control-volume contract.");
        return spatial_source(context, dt, *grid);
    };
    return options;
}
S::BoussinesqModelOptions material_options()
{
    S::BoussinesqModelOptions material;
    material.reference_density = material.density = 1.;
    material.specific_heat_capacity = 8.;
    material.dynamic_viscosity = material.thermal_conductivity = 0.;
    return material;
}
struct FieldFixture
{
    S::SP<Grid> grid;
    Chemistry chemistry;
    Model model;
    Model::field_type temperature, pressure;
    Model::velocity_field_type velocity;
    Model::face_flux_field_type relative_flux;
    S::MaterialPropertyFields<Pack, Grid> material;
    explicit FieldFixture(S::SP<Grid> selected)
        : grid(std::move(selected)), model(grid, field_options(chemistry, grid)),
          temperature(grid, 300., "T"), pressure(grid, 0., "p"), velocity(grid, S::vec3<double>{}, "U"),
          relative_flux(grid, 0., "relative_phi"), material(grid, material_options(), S::TimeStepperOptions{})
    {
        model.initialize_state(0., temperature, pressure, velocity, material);
    }
    void advance(double dt, const S::FVM::ALEControlVolumeState* ale = nullptr)
    {
        model.advance(dt, dt, temperature, pressure, velocity, relative_flux, material, nullptr, ale);
    }
    void expect_source(double dt) const
    {
        for (std::size_t c = 0; c < grid->num_owned_cells(); ++c)
        {
            const auto cell = static_cast<LO>(c);
            const auto center = grid->cell_centroid(cell);
            const double shape = 1. + center.x + 2. * center.z, volume = grid->cell_volume(cell);
            const auto& ledger = model.source_ledgers()[c];
            EXPECT_EQ(ledger.reactionExtents.size(), 1U);
            if (!ledger.reactionExtents.empty())
                EXPECT_NEAR(ledger.reactionExtents[0], .01 * shape * volume * dt, 1e-15);
            EXPECT_NEAR(ledger.externalHeat, 4. * shape * volume * dt, 1e-14);
            EXPECT_NEAR(model.thermal_source().value(cell), 4. * shape, 1e-13);
            EXPECT_NEAR(model.cell_species_state(cell).amount(*chemistry.catalog, right, R::SpeciesPhase::Liquid),
                .01 * shape * volume * dt, 1e-14);
        }
    }
};

TEST(ReactingSourceLocationTest, NativeFieldPublishesSpatialSourcesAndReplaysOnActualEmptyOwners)
{
    FieldFixture fixture(two_cells());
    const auto saved = fixture.model.snapshot();
    fixture.advance(.1);
    fixture.expect_source(.1);
    const auto ledger = fixture.model.source_ledgers();
    fixture.model.restore(saved);
    fixture.advance(.1);
    fixture.expect_source(.1);
    for (std::size_t c = 0; c < ledger.size(); ++c)
    {
        EXPECT_EQ(fixture.model.source_ledgers()[c].reactionExtents, ledger[c].reactionExtents);
        EXPECT_DOUBLE_EQ(fixture.model.source_ledgers()[c].externalHeat, ledger[c].externalHeat);
    }
    const auto comm = fixture.grid->owned_cell_map()->getComm();
    const int empty = fixture.grid->num_owned_cells() == 0, expected = std::max(0, comm->getSize() - 2);
    int empty_owners = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &empty_owners);
    EXPECT_EQ(empty_owners, expected);
}

TEST(ReactingSourceLocationTest, HybridSourceUsesCanonicalGeometryIdentityRatherThanContiguousAlgebraicGID)
{
    const auto ranks = Tpetra::getDefaultComm()->getSize();
    if (ranks != 1 && ranks != 2 && ranks != 4)
        GTEST_SKIP() << "This hybrid decomposition fixture targets serial, two and four ranks.";
    FieldFixture fixture(hybrid_cells());
    int differs = 0, global_differs = 0;
    for (std::size_t c = 0; c < fixture.grid->num_owned_cells(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        differs |= fixture.grid->cell_global_id(cell) != fixture.grid->cell_geometry_global_id(cell);
    }
    Teuchos::reduceAll(*fixture.grid->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &differs, &global_differs);
    if (ranks > 1)
        EXPECT_EQ(global_differs, 1);
    fixture.advance(.1);
    fixture.expect_source(.1);
}

TEST(ReactingSourceLocationTest, KinematicALEUsesTrialNewCentroidVolumeEpochAndValidatedRollbackFrame)
{
    FieldFixture fixture(two_cells());
    const auto saved = fixture.model.snapshot();
    S::PlanarALEMeshMotion<Pack> motion(fixture.grid);
    std::vector<R::SpeciesSourceLedger> first;
    for (int replay = 0; replay < 2; ++replay)
    {
        motion.begin_trial(1.1, .1);
        const auto ale = S::FVM::make_ale_control_volume_state(*fixture.grid, motion);
        fixture.model.refresh_geometry();
        // Zero relative flux is a co-moving carrier test. It is not a coupled momentum/GCL qualification.
        fixture.advance(.1, &ale);
        fixture.expect_source(.1);
        if (replay == 0)
            first = fixture.model.source_ledgers();
        else
            for (std::size_t c = 0; c < first.size(); ++c)
            {
                EXPECT_EQ(fixture.model.source_ledgers()[c].reactionExtents, first[c].reactionExtents);
                EXPECT_DOUBLE_EQ(fixture.model.source_ledgers()[c].externalHeat, first[c].externalHeat);
            }
        motion.rollback_trial();
        fixture.model.restore(saved);
        EXPECT_TRUE(fixture.model.source_ledgers().empty());
    }
}

TEST(ReactingSourceLocationTest, LowMachControllerUsesTheActualTransportFrameAndKeepsAcceptedStateImmutable)
{
    Chemistry chemistry;
    const auto grid = two_cells();
    Flow::Options settings;
    settings.minimum_pressure = 20000.;
    settings.maximum_pressure = 1000000.;
    settings.linear.preconditioner = S::LinearPreconditioner::None;
    settings.linear.tolerance = 1e-12;
    auto flow = std::make_shared<Flow>(grid, std::vector{chemistry.liquid}, settings);
    Flow::State initial;
    initial.thermodynamic_pressure = 100000.;
    initial.cells.resize(grid->num_owned_cells());
    for (std::size_t c = 0; c < initial.cells.size(); ++c)
    {
        const double volume = grid->cell_volume(static_cast<LO>(c));
        initial.cells[c].phases = {{chemistry.liquid, {volume, 0.}, 100000., 0., 300.}};
    }
    flow->initialize(initial);
    R::ReactingMixtureOptions options;
    options.mechanism = chemistry.mechanism;
    options.phases = {{R::SpeciesPhase::Liquid, {}, {}}};
    options.source_identity = "explicit-spatial-controller-source";
    options.source = [grid](const R::CoupledReactionContext& context, double dt)
    {
        validate_location(context.reaction.environment, *grid);
        require(context.reaction.environment.volumeClosure == R::ReactionVolumeClosure::ThermodynamicPredictor,
            "Native flow source lost its thermodynamic predictor ownership.");
        require(*context.reaction.environment.location->ownedCell == context.owned_cell,
            "Source location row and coupled source row disagree.");
        return spatial_source(context.reaction, dt, *grid);
    };
    R::ReactingMixtureCoupler controller(flow, options);
    const auto saved = controller.checkpoint();
    const auto trial = controller.preview(.1);
    for (std::size_t c = 0; c < initial.cells.size(); ++c)
    {
        const auto cell = static_cast<LO>(c);
        const auto center = grid->cell_centroid(cell);
        const double shape = 1. + center.x + 2. * center.z, volume = grid->cell_volume(cell);
        EXPECT_EQ(flow->state().cells[c].phases[0].moles, initial.cells[c].phases[0].moles);
        EXPECT_DOUBLE_EQ(flow->state().cells[c].phases[0].temperature, 300.);
        EXPECT_NEAR(trial.state().cells[c].phases[0].moles[1], .01 * shape * volume * .1, 1e-14);
        EXPECT_NEAR(trial.state().cells[c].phases[0].temperature, 300. + .05 * shape, 1e-10);
    }
    controller.commit(trial);
    controller.restore(saved);
    const auto replay = controller.preview(.1);
    for (std::size_t c = 0; c < initial.cells.size(); ++c)
        EXPECT_EQ(replay.state().cells[c].phases[0].moles, trial.state().cells[c].phases[0].moles);
    EXPECT_DOUBLE_EQ(flow->state().time, 0.);
}
} // namespace
