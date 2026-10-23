#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "solvers/PlanarFreeSurfaceModel.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace
{
namespace S = SimpleFluid;
using Pack = S::DefaultTpetraTypes;
using Mesh = S::MeshHandle<Pack>;
using Inventory = S::LiquidMassInventory<Pack, Mesh>;
using Field = Inventory::field_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<Mesh> mesh(std::size_t cells = 4)
{
    std::vector<double> edges(cells + 1);
    for (std::size_t i = 0; i <= cells; ++i)
        edges[i] = static_cast<double>(i) / cells;
    Mesh::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    return std::make_shared<Mesh>(
        std::make_shared<Mesh::Cartesian>(S::Vec3D<S::ArrReal>{{{0., 1.}, {0., 1.}, edges}}), distribution);
}
S::LiquidMassInventoryOptions options()
{
    S::LiquidMassInventoryOptions result;
    result.mode = S::LiquidVolumeMode::CellMassInventory;
    return result;
}
double sum(const Mesh& grid, double local)
{
    double global = 0.;
    Teuchos::reduceAll(*grid.owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

TEST(SpeciesLiquidMassInventoryTest, InitialActualMassAndEOSVolumeAreCopiedWithoutSmearing)
{
    auto grid = mesh();
    Field mass(grid, 0., "actualSpeciesLiquidMass"), density(grid, 0., "actualLiquidEOSDensity");
    double localMass = 0., localVolume = 0.;
    for (std::size_t owned = 0; owned < grid->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const double z = grid->cell_centroid(cell).z;
        const double m = 100. + 20. * z, rho = 800. + 100. * z;
        mass.owned_write_view()(owned, 0) = m;
        density.owned_write_view()(owned, 0) = rho;
        localMass += m * grid->cell_volume(cell);
        localVolume += m / rho * grid->cell_volume(cell);
    }
    Inventory inventory(grid, options());
    inventory.initializeFromSpecies(mass, density);
    EXPECT_TRUE(inventory.speciesBound());
    EXPECT_NEAR(inventory.totalMass(), sum(*grid, localMass), 1e-12);
    EXPECT_NEAR(inventory.liquidVolume(), sum(*grid, localVolume), 1e-14);
    for (std::size_t owned = 0; owned < grid->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(inventory.cellMassInventory().value(cell), mass.value(cell));
        EXPECT_DOUBLE_EQ(inventory.pureLiquidDensity().value(cell), density.value(cell));
    }
}

TEST(SpeciesLiquidMassInventoryTest, ConfiguredMassAndUnsupportedOwnershipModesRejectBeforeBinding)
{
    auto grid = mesh();
    Field mass(grid, 100., "actualMass"), density(grid, 1000., "actualDensity");
    auto selected = options();
    selected.initial_liquid_mass = 99.;
    Inventory mismatch(grid, selected);
    EXPECT_ANY_THROW(mismatch.initializeFromSpecies(mass, density));
    EXPECT_FALSE(mismatch.initialized());
    selected = options();
    selected.mode = S::LiquidVolumeMode::GlobalConstantMass;
    Inventory global(grid, selected);
    EXPECT_ANY_THROW(global.initializeFromSpecies(mass, density));
    selected = options();
    selected.depletion_policy = S::FreeSurfaceRangePolicy::ClampAndReport;
    Inventory clamp(grid, selected);
    EXPECT_ANY_THROW(clamp.initializeFromSpecies(mass, density));
}

TEST(SpeciesLiquidMassInventoryTest, AlreadyTransportedMassAndEOSDensityRemainTransactionalAndHaveSeparateBudgets)
{
    auto grid = mesh();
    Field mass(grid, 100., "actualMass"), density(grid, 1000., "actualDensity");
    Inventory inventory(grid, options());
    inventory.initializeFromSpecies(mass, density);
    mass.put_scalar(92.);
    density.put_scalar(800.);
    const auto preview = inventory.previewSpeciesMass(mass, density, -5., 3.);
    EXPECT_FALSE(preview.transportStatistics().has_value());
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 100.);
    EXPECT_NEAR(preview.diagnostics().total_mass, 92., 1e-12);
    EXPECT_NEAR(preview.diagnostics().liquid_volume, 92. / 800., 1e-14);
    const auto& trialMass = inventory.trialCellMassInventory(preview);
    for (std::size_t owned = 0; owned < grid->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(inventory.pureLiquidDensity().value(cell), 1000.);
        EXPECT_DOUBLE_EQ(trialMass.value(cell), 92.);
    }
    inventory.commitPhaseChange(preview);
    const auto diagnostics = inventory.diagnostics();
    EXPECT_DOUBLE_EQ(diagnostics.cumulative_species_mass_change, -5.);
    EXPECT_DOUBLE_EQ(diagnostics.cumulative_species_outflow, 3.);
    EXPECT_DOUBLE_EQ(diagnostics.species_step_mass_change, -5.);
    EXPECT_DOUBLE_EQ(diagnostics.species_step_outflow, 3.);
    EXPECT_DOUBLE_EQ(diagnostics.cumulative_evaporated_mass, 0.);
    EXPECT_DOUBLE_EQ(diagnostics.cumulative_condensed_mass, 0.);
    EXPECT_NEAR(diagnostics.mass_balance_residual, 0., 1e-12);
    Inventory::face_flux_field_type flux(grid, .3, "unusedAlreadyAppliedMassFlux");
    EXPECT_ANY_THROW((void) inventory.previewCellwiseAdvance(.1, flux));
    EXPECT_ANY_THROW(inventory.updatePureLiquidDensity([](auto) { return 700.; }));
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 92.);
    EXPECT_ANY_THROW(inventory.commitPhaseChange(preview));
}

TEST(SpeciesLiquidMassInventoryTest, BadBudgetsAndFieldsPreserveAcceptedAndPreviousTrialState)
{
    auto grid = mesh();
    Field mass(grid, 100., "actualMass"), density(grid, 1000., "actualDensity");
    Inventory inventory(grid, options());
    inventory.initializeFromSpecies(mass, density);
    mass.put_scalar(96.);
    const auto valid = inventory.previewSpeciesMass(mass, density, -4.);
    mass.put_scalar(90.);
    density.put_scalar(800.);
    EXPECT_ANY_THROW((void) inventory.previewSpeciesMass(mass, density, 0.));
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 100.);
    EXPECT_NEAR(inventory.liquidVolume(), .1, 1e-15);
    // An invalid preview never changed shared trial storage, so the prior token stays valid.
    const auto& trial = inventory.trialCellMassInventory(valid);
    for (std::size_t owned = 0; owned < grid->num_owned_cells(); ++owned)
        EXPECT_DOUBLE_EQ(trial.value(static_cast<Pack::local_ordinal_type>(owned)), 96.);
    inventory.commitPhaseChange(valid);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 96.);
    EXPECT_DOUBLE_EQ(inventory.diagnostics().cumulative_species_mass_change, -4.);
}

TEST(SpeciesLiquidMassInventoryTest, EOSRefreshStaysPrivateAndDoesNotRepeatSpeciesSource)
{
    auto grid = mesh();
    Field mass(grid, 100., "actualMass"), density(grid, 1000., "actualDensity");
    Inventory inventory(grid, options());
    inventory.initializeFromSpecies(mass, density);
    const auto accepted = inventory.snapshot();
    mass.put_scalar(99.);
    density.put_scalar(800.);
    const auto first = inventory.previewSpeciesMass(mass, density, -1.);
    const auto refreshed = inventory.refreshCellwiseDensity(first, [](auto) { return 900.; });
    EXPECT_NEAR(refreshed.diagnostics().liquid_volume, 99. / 900., 1e-14);
    EXPECT_ANY_THROW(inventory.commitPhaseChange(first));
    inventory.commitPhaseChange(refreshed);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 99.);
    EXPECT_DOUBLE_EQ(inventory.diagnostics().cumulative_species_mass_change, -1.);
    for (std::size_t owned = 0; owned < grid->num_owned_cells(); ++owned)
        EXPECT_DOUBLE_EQ(inventory.pureLiquidDensity().value(static_cast<Pack::local_ordinal_type>(owned)), 900.);
    inventory.restore(accepted);
    EXPECT_TRUE(inventory.speciesBound());
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 100.);
    EXPECT_DOUBLE_EQ(inventory.diagnostics().cumulative_species_mass_change, 0.);
    EXPECT_ANY_THROW(inventory.commitPhaseChange(refreshed));
}

TEST(SpeciesLiquidMassInventoryTest, EmptyLiquidHasExactlyZeroDensityAndSkipsCaloricPropertyEvaluation)
{
    auto grid = mesh(1);
    EXPECT_EQ(sum(*grid, grid->num_owned_cells() == 0 ? 1. : 0.),
        static_cast<double>(grid->owned_cell_map()->getComm()->getSize() - 1));
    Field mass(grid, 0., "emptyMass"), density(grid, 0., "emptyDensity");
    Inventory inventory(grid, options());
    inventory.initializeFromSpecies(mass, density);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 0.);
    EXPECT_DOUBLE_EQ(inventory.liquidVolume(), 0.);
    const auto preview = inventory.previewSpeciesMass(mass, density, 0.);
    const auto refreshed = inventory.refreshCellwiseDensity(preview,
        [](auto) -> double { throw std::logic_error("Empty phase property callback must not be evaluated."); });
    inventory.commitPhaseChange(refreshed);
    EXPECT_DOUBLE_EQ(inventory.liquidVolume(), 0.);
    mass.put_scalar(10.);
    density.put_scalar(500.);
    const auto born = inventory.previewSpeciesMass(mass, density, 10.);
    inventory.commitPhaseChange(born);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 10.);
    EXPECT_NEAR(inventory.liquidVolume(), .02, 1e-15);
}

TEST(SpeciesLiquidMassInventoryTest, ALEConsumesAcceptedOldAndTrialNewVolumesOnceAndRollsBackEpochs)
{
    auto grid = mesh();
    Field mass(grid, 100., "speciesMass"), density(grid, 1000., "EOSDensity");
    Inventory inventory(grid, options());
    inventory.initializeFromSpecies(mass, density);
    const auto accepted = inventory.snapshot();
    S::PlanarALEMeshMotion<Pack> motion(grid);
    for (int replay = 0; replay < 2; ++replay)
    {
        motion.begin_trial(1.2, .1);
        const auto ale = S::FVM::make_ale_control_volume_state(*grid, motion);
        mass.put_scalar(100. / 1.2); // Already transported actual species density, not another mass advance.
        const auto preview = inventory.previewSpeciesMass(mass, density, 0., 0., &ale);
        EXPECT_NEAR(preview.diagnostics().total_mass, 100., 1e-12);
        EXPECT_NEAR(preview.diagnostics().liquid_volume, .1, 1e-14);
        inventory.commitPhaseChange(preview);
        motion.rollback_trial();
        inventory.restore(accepted);
        EXPECT_DOUBLE_EQ(inventory.totalMass(), 100.);
        EXPECT_ANY_THROW(inventory.commitPhaseChange(preview));
    }
    motion.begin_trial(1.2, .1);
    const auto ale = S::FVM::make_ale_control_volume_state(*grid, motion);
    const auto final = inventory.previewSpeciesMass(mass, density, 0., 0., &ale);
    inventory.commitPhaseChange(final);
    motion.accept_trial();
    const auto next = inventory.previewSpeciesMass(mass, density, 0.);
    EXPECT_NEAR(next.diagnostics().total_mass, 100., 1e-12);
    EXPECT_NEAR(next.diagnostics().liquid_volume, .1, 1e-14);
}

TEST(SpeciesLiquidMassInventoryTest, GeometryChangesInvalidateUncommittedFollowerTokens)
{
    auto grid = mesh();
    Field mass(grid, 100., "speciesMass"), density(grid, 1000., "EOSDensity");
    Inventory inventory(grid, options());
    inventory.initializeFromSpecies(mass, density);
    const auto stale = inventory.previewSpeciesMass(mass, density, 0.);
    S::PlanarALEMeshMotion<Pack> motion(grid);
    motion.begin_trial(1.1, .1);
    EXPECT_ANY_THROW(inventory.commitPhaseChange(stale));
    EXPECT_ANY_THROW((void) inventory.trialCellMassInventory(stale));
    motion.rollback_trial();
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 100.);
}

TEST(SpeciesLiquidMassInventoryTest, TinyActualMassUsesRelativeClosureRatherThanAOneKilogramFloor)
{
    auto grid = mesh();
    Field mass(grid, 1e-20, "tinySpeciesMass"), density(grid, 1000., "EOSDensity");
    auto selected = options();
    selected.initial_liquid_mass = 0.;
    Inventory mismatch(grid, selected);
    EXPECT_ANY_THROW(mismatch.initializeFromSpecies(mass, density));
    Inventory inventory(grid, options());
    inventory.initializeFromSpecies(mass, density);
    mass.put_scalar(.5e-20);
    EXPECT_ANY_THROW((void) inventory.previewSpeciesMass(mass, density, 0.));
    EXPECT_NEAR(inventory.totalMass(), 1e-20, 1e-35);
}
TEST(SpeciesLiquidMassInventoryTest, SnapshotRestorePreservesUnboundAndBoundOwnershipStates)
{
    auto grid = mesh();
    Field mass(grid, 100., "speciesMass"), density(grid, 1000., "EOSDensity");
    Inventory inventory(grid, options());
    const auto unbound = inventory.snapshot();
    inventory.initializeFromSpecies(mass, density);
    EXPECT_TRUE(inventory.speciesBound());
    inventory.restore(unbound);
    EXPECT_FALSE(inventory.speciesBound());
    EXPECT_FALSE(inventory.initialized());
    inventory.initializeFromSpecies(mass, density);
    const auto bound = inventory.snapshot();
    mass.put_scalar(98.);
    const auto preview = inventory.previewSpeciesMass(mass, density, -2.);
    inventory.commitPhaseChange(preview);
    inventory.restore(bound);
    EXPECT_TRUE(inventory.speciesBound());
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 100.);
    EXPECT_DOUBLE_EQ(inventory.diagnostics().cumulative_species_mass_change, 0.);
}
TEST(SpeciesLiquidMassInventoryTest, IndividuallySmallBudgetDefectsCannotAccumulatePastGlobalRelativeClosure)
{
    auto grid = mesh();
    Field mass(grid, 1e-20, "tinySpeciesMass"), density(grid, 1000., "EOSDensity");
    Inventory inventory(grid, options());
    inventory.initializeFromSpecies(mass, density);
    int accepted = 0;
    bool rejected = false;
    for (int step = 1; step <= 200; ++step)
    {
        mass.put_scalar(1e-20 * (1. - step * 1e-12));
        try
        {
            const auto preview = inventory.previewSpeciesMass(mass, density, 0.);
            inventory.commitPhaseChange(preview);
            ++accepted;
        }
        catch (const std::logic_error&)
        {
            rejected = true;
            break;
        }
    }
    EXPECT_TRUE(rejected);
    EXPECT_GT(accepted, 50);
    EXPECT_LT(accepted, 150);
    EXPECT_DOUBLE_EQ(inventory.diagnostics().cumulative_species_mass_change, 0.);
}
} // namespace
