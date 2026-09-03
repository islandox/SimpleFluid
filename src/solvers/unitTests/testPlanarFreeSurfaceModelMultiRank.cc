/**
 * @file testPlanarFreeSurfaceModelMultiRank.cc
 * @brief Distributed invariance tests for the planar free-surface core.
 */

#include <gtest/gtest.h>

#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/PlanarFreeSurfaceModel.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <cmath>
#include <memory>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

double global_sum(const Teuchos::Comm<int>& communicator, double local)
{
    double global = 0.0;
    Teuchos::reduceAll(communicator, Teuchos::REDUCE_SUM, 1, &local, &global);
    return global;
}

void expect_replicated(const Teuchos::Comm<int>& communicator, double value)
{
    double minimum = 0.0;
    double maximum = 0.0;
    Teuchos::reduceAll(communicator, Teuchos::REDUCE_MIN, 1, &value, &minimum);
    Teuchos::reduceAll(communicator, Teuchos::REDUCE_MAX, 1, &value, &maximum);
    EXPECT_DOUBLE_EQ(minimum, maximum);
}

} // namespace

TEST(LiquidMassInventoryMultiRankTest, NonuniformDensityIsPartitionIndependent)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    ASSERT_EQ(communicator->getSize(), 2);

    auto initial_density = [&mesh](Pack::local_ordinal_type cell)
    { return 950.0 + 100.0 * mesh->cell_centroid(cell).x; };
    auto updated_density = [&mesh](Pack::local_ordinal_type cell)
    { return 775.0 + 50.0 * mesh->cell_centroid(cell).x; };

    SimpleFluid::LiquidMassInventory<Pack> inventory(mesh);
    inventory.initialize(4.0, initial_density);
    inventory.updatePureLiquidDensity(updated_density);

    double local_expected_volume = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        local_expected_volume += initial_density(cell) / updated_density(cell) * mesh->cell_volume(cell);
    }
    const auto expected_volume = global_sum(*communicator, local_expected_volume);
    EXPECT_NEAR(inventory.liquidVolume(), expected_volume, 1.0e-13);
    expect_replicated(*communicator, inventory.totalMass());
    expect_replicated(*communicator, inventory.liquidVolume());
    expect_replicated(*communicator, inventory.diagnostics().mass_balance_residual);
}

TEST(LiquidMassInventoryMultiRankTest, CollectivelyValidatesInitializationAndPreviewCommit)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    ASSERT_EQ(communicator->getSize(), 2);

    {
        SimpleFluid::LiquidMassInventory<Pack> divergent(mesh);
        const auto initial_volume = communicator->getRank() == 0 ? 4.0 : 3.0;
        EXPECT_THROW(divergent.initialize(initial_volume, [](Pack::local_ordinal_type) { return 1000.0; }),
            std::invalid_argument);
    }

    SimpleFluid::LiquidMassInventory<Pack> inventory(mesh);
    inventory.initialize(4.0, [](Pack::local_ordinal_type) { return 1000.0; });
    const auto preview = inventory.previewPhaseChange(100.0, 25.0);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 4000.0);
    EXPECT_DOUBLE_EQ(preview.diagnostics().total_mass, 3925.0);
    inventory.commitPhaseChange(preview);
    expect_replicated(*communicator, inventory.totalMass());
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 3925.0);
    EXPECT_THROW(inventory.commitPhaseChange(preview), std::logic_error);
}

TEST(PlanarFreeSurfaceModelMultiRankTest, ReducedInputsProduceReplicatedDiagnostics)
{
    const auto communicator = Tpetra::getDefaultComm();
    ASSERT_EQ(communicator->getSize(), 2);
    const auto rank = communicator->getRank();

    const auto initial_liquid = global_sum(*communicator, 0.5 + 0.25 * static_cast<double>(rank));
    const auto initial_bubble = global_sum(*communicator, 0.05 + 0.025 * static_cast<double>(rank));
    const auto initial_submerged = global_sum(*communicator, 0.5);

    auto map = std::make_shared<SimpleFluid::ConstantAreaVesselVolumeMap>(0.0, 10.0, 1.0);
    SimpleFluid::HeadspaceOptions headspace_options;
    headspace_options.mode = SimpleFluid::HeadspaceMode::Vented;
    headspace_options.total_internal_volume = 10.0;
    auto headspace = std::make_unique<SimpleFluid::VentedHeadspaceModel>(headspace_options);
    SimpleFluid::PlanarFreeSurfaceModel model(map, std::move(headspace));

    SimpleFluid::FreeSurfaceUpdate initial;
    initial.liquid_volume_at_pressure = [initial_liquid](double) { return initial_liquid; };
    initial.bubble_volume_at_pressure = [initial_bubble](double) { return initial_bubble; };
    initial.gas.submerged_moles = {{"H2", initial_submerged}};
    model.initialize(initial);

    const auto escaped = global_sum(*communicator, 0.1);
    SimpleFluid::FreeSurfaceUpdate next;
    next.time = 1.0;
    next.time_step = 1.0;
    next.liquid_volume_at_pressure = [initial_liquid](double) { return initial_liquid + 0.2; };
    next.bubble_volume_at_pressure = [initial_bubble](double) { return initial_bubble - 0.05; };
    next.gas.submerged_moles = {{"H2", initial_submerged - escaped}};
    next.gas.escaped_moles_this_step = {{"H2", escaped}};
    model.update(next);

    const auto diagnostics = model.diagnostics();
    expect_replicated(*communicator, diagnostics.liquid_volume);
    expect_replicated(*communicator, diagnostics.pool_volume);
    expect_replicated(*communicator, diagnostics.pool_level);
    expect_replicated(*communicator, diagnostics.pool_level_rate);
    expect_replicated(*communicator, diagnostics.gas_closure_residual);
    EXPECT_NEAR(diagnostics.gas_closure_residual, 0.0, 1.0e-14);
    EXPECT_DOUBLE_EQ(diagnostics.vented_gas_moles.at("H2"), escaped);
}
