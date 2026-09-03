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

TEST(LiquidMassInventoryMultiRankTest, CellwiseTransportAndPhaseSourcesArePartitionIndependent)
{
    using NativeMesh = SimpleFluid::MeshHandle<Pack>;
    using Inventory = SimpleFluid::LiquidMassInventory<Pack, NativeMesh>;
    auto cartesian = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0, 4.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<NativeMesh>(std::move(cartesian));
    const auto communicator = mesh->owned_cell_map()->getComm();
    ASSERT_EQ(communicator->getSize(), 2);

    SimpleFluid::LiquidMassInventoryOptions options;
    options.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    Inventory inventory(mesh, options);
    inventory.initialize(
        4.0, [&mesh](Pack::local_ordinal_type cell) { return 100.0 + 25.0 * mesh->cell_centroid(cell).x; });
    inventory.updatePureLiquidDensity([](Pack::local_ordinal_type) { return 100.0; });
    const auto mass_before = inventory.totalMass();

    Inventory::face_flux_field_type flux(mesh, 0.0, "liquidMassFlux");
    int local_partition_fluxes = 0;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        if (!mesh->is_interior_face(face_lid) || !flux.is_owned_face(face_lid))
        {
            continue;
        }
        const auto owner = mesh->owner_cell(face_lid);
        const auto neighbor = mesh->neighbor_cell(face_lid);
        if (!mesh->is_owned_cell(owner) || !mesh->is_owned_cell(neighbor))
        {
            flux.set_value(face_lid, 0.05);
            ++local_partition_fluxes;
        }
    }
    flux.sync_ghosts();
    const auto partition_fluxes = global_sum(*communicator, static_cast<double>(local_partition_fluxes));
    ASSERT_GT(partition_fluxes, 0.0);

    const auto transport_preview = inventory.previewCellwiseAdvance(0.1, flux);
    ASSERT_TRUE(transport_preview.transportStatistics().has_value());
    EXPECT_TRUE(transport_preview.transportStatistics()->converged);
    expect_replicated(*communicator, transport_preview.diagnostics().total_mass);
    EXPECT_NEAR(transport_preview.diagnostics().total_mass, mass_before, mass_before * 1.0e-9);
    EXPECT_NEAR(transport_preview.diagnostics().step_mass_balance_residual, 0.0, mass_before * 1.0e-9);
    inventory.commitPhaseChange(transport_preview);

    Inventory::field_type evaporation(mesh, 0.0, "evaporationMassRate");
    Inventory::field_type condensation(mesh, 0.0, "condensationMassRate");
    double local_evaporation = 0.0;
    double local_condensation = 0.0;
    constexpr double time_step = 0.2;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto x = mesh->cell_centroid(cell).x;
        const auto evaporation_rate = x < 2.0 ? 1.0 : 0.0;
        const auto condensation_rate = x >= 2.0 ? 0.25 : 0.0;
        evaporation.set_owned_value(cell, evaporation_rate);
        condensation.set_owned_value(cell, condensation_rate);
        local_evaporation += evaporation_rate * mesh->cell_volume(cell) * time_step;
        local_condensation += condensation_rate * mesh->cell_volume(cell) * time_step;
    }
    evaporation.sync_ghosts();
    condensation.sync_ghosts();
    flux.put_scalar(0.0);
    flux.sync_ghosts();
    const auto expected_evaporation = global_sum(*communicator, local_evaporation);
    const auto expected_condensation = global_sum(*communicator, local_condensation);
    const auto phase_preview = inventory.previewCellwiseAdvance(time_step, flux, &evaporation, &condensation);
    EXPECT_NEAR(phase_preview.diagnostics().cumulative_evaporated_mass, expected_evaporation, 1.0e-11);
    EXPECT_NEAR(phase_preview.diagnostics().cumulative_condensed_mass, expected_condensation, 1.0e-11);
    EXPECT_NEAR(phase_preview.diagnostics().total_mass, mass_before - expected_evaporation + expected_condensation,
        mass_before * 1.0e-9);
    expect_replicated(*communicator, phase_preview.diagnostics().liquid_volume);
    inventory.commitPhaseChange(phase_preview);
}

TEST(LiquidMassInventoryMultiRankTest, CellwiseAdvanceCollectivelyRejectsDivergentSourceSelection)
{
    using NativeMesh = SimpleFluid::MeshHandle<Pack>;
    using Inventory = SimpleFluid::LiquidMassInventory<Pack, NativeMesh>;
    auto cartesian = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0, 4.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<NativeMesh>(std::move(cartesian));
    const auto communicator = mesh->owned_cell_map()->getComm();
    ASSERT_EQ(communicator->getSize(), 2);

    SimpleFluid::LiquidMassInventoryOptions options;
    options.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    Inventory inventory(mesh, options);
    inventory.initialize(4.0, [](Pack::local_ordinal_type) { return 1000.0; });
    Inventory::face_flux_field_type flux(mesh, 0.0, "liquidMassFlux");
    Inventory::field_type evaporation(mesh, 0.0, "evaporationMassRate");
    const auto* selected_evaporation = communicator->getRank() == 0 ? &evaporation : nullptr;

    EXPECT_THROW(
        static_cast<void>(inventory.previewCellwiseAdvance(0.1, flux, selected_evaporation)), std::invalid_argument);
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 4000.0);
}

TEST(LiquidMassInventoryMultiRankTest, DensityChangeCollectivelyInvalidatesCellwisePreview)
{
    using NativeMesh = SimpleFluid::MeshHandle<Pack>;
    using Inventory = SimpleFluid::LiquidMassInventory<Pack, NativeMesh>;
    auto cartesian = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0, 4.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<NativeMesh>(std::move(cartesian));
    const auto communicator = mesh->owned_cell_map()->getComm();
    ASSERT_EQ(communicator->getSize(), 2);

    SimpleFluid::LiquidMassInventoryOptions options;
    options.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    Inventory inventory(mesh, options);
    inventory.initialize(4.0, [](Pack::local_ordinal_type) { return 1000.0; });
    Inventory::face_flux_field_type flux(mesh, 0.0, "liquidMassFlux");

    const auto stale = inventory.previewCellwiseAdvance(0.1, flux);
    inventory.updatePureLiquidDensity(
        [&mesh](Pack::local_ordinal_type cell) { return mesh->cell_centroid(cell).x < 1.0 ? 500.0 : 1000.0; });

    EXPECT_THROW(inventory.commitPhaseChange(stale), std::logic_error);
    expect_replicated(*communicator, inventory.totalMass());
    expect_replicated(*communicator, inventory.liquidVolume());
    EXPECT_DOUBLE_EQ(inventory.totalMass(), 4000.0);
    EXPECT_NEAR(inventory.liquidVolume(), 5.0, 1.0e-12);

    const auto current = inventory.previewCellwiseAdvance(0.1, flux);
    EXPECT_NO_THROW(inventory.commitPhaseChange(current));
    EXPECT_NEAR(inventory.liquidVolume(), 5.0, 1.0e-10);
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
