/** @file testAdaptiveVoF.cc
 * @brief Independent geometric, carrier-inventory and MPI contracts for adaptive VoF.
 */
#include "FVM/AdaptiveVoF.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
using Hierarchy = Meshes::CartesianAMR;
using Grid = Meshes::AMRGrid;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

double global_sum(double local)
{
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}

double global_max(double local)
{
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local, &result);
    return result;
}

Grid column(std::size_t roots, double length)
{
    Grid grid;
    grid.root_blocks = {roots, 1, 1};
    grid.block_cells = {4, 4, 4};
    grid.extent = {length, length / roots, length / roots};
    grid.max_level = 1;
    return grid;
}

std::vector<double> slab(const AdaptiveVoF& geometry, double lower, double upper)
{
    std::vector<double> fraction(geometry.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const auto box = geometry.hierarchy()->cell_box(geometry.mesh()->cell_global_id(static_cast<LO>(c)));
        fraction[c] = std::max(0.0, std::min(box.upper[0], upper) - std::max(box.lower[0], lower)) /
                      (box.upper[0] - box.lower[0]);
    }
    return fraction;
}

std::vector<double> owned_faces(const AdaptiveVoF& geometry, Vector velocity = {})
{
    const auto& mesh = *geometry.mesh();
    // Non-owned entries must be replaced by the canonical owner's values.
    std::vector<double> result(mesh.num_faces(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t f = 0; f < mesh.num_owned_faces(); ++f)
    {
        const auto face = static_cast<LO>(f);
        const auto normal = mesh.face_normal(face);
        result[f] = mesh.is_geometry_exterior_face(face)
                        ? 0.0
                        : velocity[0] * normal.x + velocity[1] * normal.y + velocity[2] * normal.z;
    }
    return result;
}

double gas_volume(const AdaptiveVoF& geometry, const std::vector<double>& fraction)
{
    double local = 0.0;
    for (std::size_t c = 0; c < fraction.size(); ++c)
        local += fraction[c] * geometry.mesh()->cell_volume(static_cast<LO>(c));
    return global_sum(local);
}

std::vector<ConservedQuantity> carrier_inventories(const AdaptiveVoF& geometry, const std::vector<double>& fraction)
{
    std::vector<ConservedQuantity> quantities{{"bulk_amount", InventoryCarrier::BulkVolume, {}, true},
        {"gas_moles", InventoryCarrier::GasVolume, {}, true},
        {"dissolved_moles", InventoryCarrier::LiquidVolume, {}, true},
        {"momentum_x", InventoryCarrier::MixtureMass, {}, false},
        {"momentum_y", InventoryCarrier::MixtureMass, {}, false},
        {"thermal_energy", InventoryCarrier::MixtureMass, {}, true}};
    for (auto& quantity : quantities)
        quantity.amount.resize(fraction.size());
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const double volume = geometry.mesh()->cell_volume(static_cast<LO>(c));
        const double gas = volume * fraction[c], liquid = volume - gas;
        const double mass = 1000.0 * liquid + 0.1 * gas;
        const std::array<double, 6> amounts{
            3.0 * volume, 7.0 * gas, 2.0 * liquid, -0.3 * mass, 0.2 * mass, 17.0 * mass};
        for (std::size_t q = 0; q < quantities.size(); ++q)
            quantities[q].amount[c] = amounts[q];
    }
    return quantities;
}

void expect_inventory_conservation(
    const std::vector<ConservedQuantity>& before, const std::vector<ConservedQuantity>& after)
{
    ASSERT_EQ(before.size(), after.size());
    for (std::size_t q = 0; q < before.size(); ++q)
    {
        const double old_sum = global_sum(std::accumulate(before[q].amount.begin(), before[q].amount.end(), 0.0));
        const double new_sum = global_sum(std::accumulate(after[q].amount.begin(), after[q].amount.end(), 0.0));
        EXPECT_EQ(before[q].name, after[q].name);
        EXPECT_EQ(before[q].carrier, after[q].carrier);
        EXPECT_EQ(before[q].nonnegative, after[q].nonnegative);
        EXPECT_NEAR(new_sum, old_sum, 2e-12 * std::max(std::abs(old_sum), 1e-30));
    }
}

bool contains(const std::vector<Meshes::AMRBlock>& keys, Meshes::AMRBlock key)
{
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}
} // namespace

TEST(AdaptiveVoFTest, RefinementAndRestrictionConservePhaseCarriersAndSignedExtensives)
{
    const auto coarse_tree = std::make_shared<Hierarchy>(column(2, 1.0));
    AdaptiveVoF coarse(coarse_tree);
    AdaptiveVoF fine(coarse_tree->adapt({0}));
    const auto fraction = slab(coarse, 0.0, 0.31);
    const auto quantities = carrier_inventories(coarse, fraction);
    const auto refined = coarse.remap(fine, fraction, quantities, 1000.0, 0.1);
    const auto exact = slab(fine, 0.0, 0.31);
    const auto expected_quantities = carrier_inventories(fine, exact);
    ASSERT_EQ(refined.fraction.size(), exact.size());
    for (std::size_t c = 0; c < exact.size(); ++c)
    {
        EXPECT_NEAR(refined.fraction[c], exact[c], 2e-12);
        for (std::size_t q = 0; q < quantities.size(); ++q)
            EXPECT_NEAR(refined.inventories[q].amount[c], expected_quantities[q].amount[c], 2e-11);
    }
    EXPECT_NEAR(gas_volume(fine, refined.fraction), gas_volume(coarse, fraction), 2e-13);
    expect_inventory_conservation(quantities, refined.inventories);
    // Direct remap restriction tests the transfer kernel. Automatic coarsening
    // policy is stricter and will keep this mixed-phase parent protected.
    const auto restricted = fine.remap(coarse, refined.fraction, refined.inventories, 1000.0, 0.1);
    ASSERT_EQ(restricted.fraction.size(), fraction.size());
    for (std::size_t c = 0; c < fraction.size(); ++c)
        EXPECT_NEAR(restricted.fraction[c], fraction[c], 2e-12);
    expect_inventory_conservation(quantities, restricted.inventories);
    EXPECT_EQ(coarse.mesh()->connectivity_storage_bytes(), 0u);
    EXPECT_EQ(fine.mesh()->connectivity_storage_bytes(), 0u);
    const double global_cells = global_sum(static_cast<double>(fine.mesh()->num_owned_cells()));
    EXPECT_EQ(global_cells, fine.hierarchy()->blocks().size() * 64.0);
    if (Tpetra::getDefaultComm()->getSize() > 1)
        EXPECT_LT(fine.mesh()->num_owned_cells(), global_cells);
}

TEST(AdaptiveVoFTest, TinyPositiveGasAndMolarInventorySurviveGeometricRemap)
{
    const auto tree = std::make_shared<Hierarchy>(column(2, 1.0));
    AdaptiveVoF source(tree), target(tree->adapt({0}));
    const std::vector<double> fraction(source.mesh()->num_owned_cells(), 1e-18);
    ConservedQuantity moles{"gas_moles", InventoryCarrier::GasVolume, {}, true};
    for (std::size_t c = 0; c < fraction.size(); ++c)
        moles.amount.push_back(7.0 * fraction[c] * source.mesh()->cell_volume(static_cast<LO>(c)));
    const auto remapped = source.remap(target, fraction, {moles});
    const double initial = gas_volume(source, fraction), final = gas_volume(target, remapped.fraction);
    EXPECT_GT(initial, 0.0);
    EXPECT_GT(final, 0.0);
    EXPECT_NEAR(final, initial, 2e-12 * initial);
    expect_inventory_conservation({moles}, remapped.inventories);
    for (const double f : remapped.fraction)
        EXPECT_GE(f, 0.0);
    RefinementOptions options;
    options.curvature_resolution = 0.0;
    const auto plan = target.refinement_plan(remapped.fraction, owned_faces(target), options);
    EXPECT_TRUE(plan.resolution_limited);
    EXPECT_TRUE(plan.coarsen.empty());
}

TEST(AdaptiveVoFTest, CoarseFineTranslationMatchesCanonicalPhaseLedgerAndAnalyticalSlab)
{
    auto config = column(4, 1.0);
    config.periodic[0] = true;
    const auto tree = std::make_shared<Hierarchy>(config);
    AdaptiveVoF geometry(tree->adapt({1}));
    const auto& mesh = *geometry.mesh();
    auto fraction = slab(geometry, 0.10, 0.20);
    const double initial_volume = gas_volume(geometry, fraction);
    const auto velocity = owned_faces(geometry, {0.2, 0.0, 0.0});
    const auto exchanged_velocity = geometry.exchange_faces(velocity);
    double coarse_fine_faces = 0.0, partition_coarse_fine_faces = 0.0;
    for (std::size_t f = 0; f < mesh.num_owned_faces(); ++f)
    {
        const auto face = static_cast<LO>(f);
        if (!mesh.is_interior_face(face))
            continue;
        const auto owner = mesh.owner_cell(face), neighbor = mesh.neighbor_cell(face);
        const auto ob = geometry.hierarchy()->cell_block(mesh.cell_global_id(owner));
        const auto nb = geometry.hierarchy()->cell_block(mesh.cell_global_id(neighbor));
        if (geometry.hierarchy()->blocks()[ob].level == geometry.hierarchy()->blocks()[nb].level)
            continue;
        ++coarse_fine_faces;
        if (static_cast<std::size_t>(owner) >= mesh.num_owned_cells() ||
            static_cast<std::size_t>(neighbor) >= mesh.num_owned_cells())
            ++partition_coarse_fine_faces;
    }
    EXPECT_GT(global_sum(coarse_fine_faces), 0.0);
    if (Tpetra::getDefaultComm()->getSize() > 1)
        EXPECT_GT(global_sum(partition_coarse_fine_faces), 0.0);
    constexpr double dt = 0.03125;
    constexpr int steps = 48;
    for (int step = 0; step < steps; ++step)
    {
        const auto result = geometry.advect(fraction, velocity, dt, step % 3);
        for (std::size_t c = 0; c < fraction.size(); ++c)
        {
            double ledger = fraction[c] * mesh.cell_volume(static_cast<LO>(c));
            mesh.visit_cell_faces(static_cast<LO>(c),
                [&](LO face)
                {
                    const double sign = mesh.owner_cell(face) == static_cast<LO>(c) ? 1.0 : -1.0;
                    ledger -= sign * result.gas_volume[face];
                });
            EXPECT_NEAR(result.fraction[c] * mesh.cell_volume(static_cast<LO>(c)), ledger, 3e-15);
            EXPECT_GE(result.fraction[c], -2e-12);
            EXPECT_LE(result.fraction[c], 1.0 + 2e-12);
        }
        for (std::size_t f = 0; f < mesh.num_faces(); ++f)
            EXPECT_NEAR(result.gas_volume[f] + result.liquid_volume[f],
                dt * exchanged_velocity[f] * mesh.face_area(static_cast<LO>(f)), 3e-16);
        fraction = result.fraction;
        EXPECT_NEAR(gas_volume(geometry, fraction), initial_volume, 3e-13);
    }
    const auto exact = slab(geometry, 0.10 + steps * dt * 0.2, 0.20 + steps * dt * 0.2);
    double maximum_error = 0.0;
    for (std::size_t c = 0; c < fraction.size(); ++c)
        maximum_error = std::max(maximum_error, std::abs(fraction[c] - exact[c]));
    EXPECT_LT(global_max(maximum_error), 2e-10);
}

TEST(AdaptiveVoFTest, ImportedOwnedCellsAndFacesReplaceGhostPlaceholders)
{
    auto config = column(4, 1.0);
    config.periodic[0] = true;
    const auto tree = std::make_shared<Hierarchy>(config);
    AdaptiveVoF geometry(tree->adapt({1}));
    const auto& mesh = *geometry.mesh();
    std::vector<double> owned(mesh.num_owned_cells());
    for (std::size_t c = 0; c < owned.size(); ++c)
        owned[c] = 1.0 + mesh.cell_global_id(static_cast<LO>(c));
    const auto halo = geometry.exchange_cells(owned);
    ASSERT_EQ(halo.size(), mesh.num_local_cells());
    for (std::size_t c = 0; c < halo.size(); ++c)
        EXPECT_DOUBLE_EQ(halo[c], 1.0 + mesh.cell_global_id(static_cast<LO>(c)));
    const auto faces = geometry.exchange_faces(owned_faces(geometry, {0.2, 0.0, 0.0}));
    for (std::size_t f = 0; f < faces.size(); ++f)
        EXPECT_DOUBLE_EQ(faces[f],
            mesh.is_geometry_exterior_face(static_cast<LO>(f)) ? 0.0 : 0.2 * mesh.face_normal(static_cast<LO>(f)).x);
    if (Tpetra::getDefaultComm()->getSize() > 1)
        EXPECT_GT(global_sum(mesh.num_local_cells() - mesh.num_owned_cells()), 0.0);
}

TEST(AdaptiveVoFTest, TinySweptWidthsRetainThePureGasSubfaceFlux)
{
    auto config = column(4, 1.0);
    config.periodic[0] = true;
    const auto tree = std::make_shared<Hierarchy>(config);
    AdaptiveVoF geometry(tree->adapt({1}));
    const std::vector<double> fraction(geometry.mesh()->num_owned_cells(), 1.0);
    const auto velocity = owned_faces(geometry, {0.2, 0.0, 0.0});
    const auto exchanged_velocity = geometry.exchange_faces(velocity);
    constexpr double dt = 1e-20;
    const auto result = geometry.advect(fraction, velocity, dt);
    double nonzero_fluxes = 0.0;
    for (std::size_t f = 0; f < geometry.mesh()->num_faces(); ++f)
    {
        const double total = dt * exchanged_velocity[f] * geometry.mesh()->face_area(static_cast<LO>(f));
        EXPECT_DOUBLE_EQ(result.gas_volume[f], total);
        EXPECT_DOUBLE_EQ(result.liquid_volume[f], 0.0);
        if (f < geometry.mesh()->num_owned_faces() && total != 0.0)
        {
            EXPECT_NE(result.gas_volume[f], 0.0);
            ++nonzero_fluxes;
        }
    }
    EXPECT_GT(global_sum(nonzero_fluxes), 0.0);
    EXPECT_EQ(result.fraction, fraction);
}

TEST(AdaptiveVoFTest, InterfaceAndMotionBandsRefineAheadWithoutCoarseningDistinctGasRegions)
{
    const auto tree = std::make_shared<Hierarchy>(column(8, 8.0));
    AdaptiveVoF geometry(tree);
    const auto fraction = slab(geometry, 0.0, 3.37);
    const auto velocity = owned_faces(geometry, {1.0, 0.0, 0.0});
    RefinementOptions options;
    options.curvature_resolution = 0.0;
    const auto stationary = geometry.refinement_plan(fraction, velocity, options);
    options.forecast_time = 1.1;
    const auto moving = geometry.refinement_plan(fraction, velocity, options);
    EXPECT_GT(moving.refine.size(), stationary.refine.size());
    for (const auto block : stationary.refine)
        EXPECT_NE(std::find(moving.refine.begin(), moving.refine.end(), block), moving.refine.end());
    EXPECT_NE(std::find(stationary.refine.begin(), stationary.refine.end(), 3), stationary.refine.end());

    AdaptiveVoF separated(tree->adapt({2, 6}));
    auto left = slab(separated, 2.125, 2.25), right = slab(separated, 2.75, 2.875);
    for (std::size_t c = 0; c < left.size(); ++c)
        left[c] += right[c];
    options.minimum_resolved_diameter = 100.0;
    options.forecast_time = 0.0;
    const auto protected_plan = separated.refinement_plan(left, owned_faces(separated), options);
    EXPECT_FALSE(contains(protected_plan.coarsen, {0, {2, 0, 0}}));
    EXPECT_TRUE(contains(protected_plan.coarsen, {0, {6, 0, 0}}));
}

TEST(AdaptiveVoFTest, WakeAndPromotionHintsProtectOtherwiseCoarsenablePureSiblings)
{
    const auto tree = std::make_shared<Hierarchy>(column(8, 8.0));
    AdaptiveVoF geometry(tree->adapt({2, 6}));
    const std::vector<double> fraction(geometry.mesh()->num_owned_cells(), 0.0);
    RefinementOptions options;
    options.minimum_resolved_diameter = 100.0;
    options.curvature_resolution = 0.0;
    const auto baseline = geometry.refinement_plan(fraction, owned_faces(geometry), options);
    EXPECT_TRUE(contains(baseline.coarsen, {0, {2, 0, 0}}));
    EXPECT_TRUE(contains(baseline.coarsen, {0, {6, 0, 0}}));
    RefinementHints hints;
    hints.promotion_regions.push_back({{2.4, 0.4, 0.4}, {2.6, 0.6, 0.6}});
    hints.wake_indicator.assign(fraction.size(), 0.0);
    for (std::size_t c = 0; c < fraction.size(); ++c)
        if (geometry.mesh()->cell_centroid(static_cast<LO>(c)).x > 6.4 &&
            geometry.mesh()->cell_centroid(static_cast<LO>(c)).x < 6.6)
            hints.wake_indicator[c] = 2.0;
    const auto protected_plan = geometry.refinement_plan(fraction, owned_faces(geometry), options, hints);
    EXPECT_FALSE(contains(protected_plan.coarsen, {0, {2, 0, 0}}));
    EXPECT_FALSE(contains(protected_plan.coarsen, {0, {6, 0, 0}}));
}

TEST(AdaptiveVoFTest, InvalidRankLocalFractionsFacesAndPhaseAmountsRejectCollectively)
{
    const auto tree = std::make_shared<Hierarchy>(column(2, 1.0));
    AdaptiveVoF source(tree), target(tree->adapt({0}));
    const auto rank = Tpetra::getDefaultComm()->getRank();
    std::vector<double> fraction(source.mesh()->num_owned_cells(), 0.0);
    auto bad_fraction = fraction;
    if (rank == 0)
        bad_fraction.front() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) source.reconstruct(bad_fraction), std::invalid_argument);
    auto bad_faces = owned_faces(source);
    if (rank == 0)
        bad_faces.front() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) source.advect(fraction, bad_faces, 0.01), std::invalid_argument);
    ConservedQuantity impossible{
        "gas_moles", InventoryCarrier::GasVolume, std::vector<double>(fraction.size(), 0.0), true};
    if (rank == 0)
        impossible.amount.front() = 1.0;
    EXPECT_THROW((void) source.remap(target, fraction, {impossible}), std::invalid_argument);
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        EXPECT_THROW(
            (void) source.advect(fraction, owned_faces(source), rank == 0 ? 0.01 : 0.02), std::invalid_argument);
        auto quantity = impossible;
        std::fill(quantity.amount.begin(), quantity.amount.end(), 0.0);
        quantity.name = rank == 0 ? "gas_a" : "gas_b";
        EXPECT_THROW((void) source.remap(target, fraction, {quantity}), std::invalid_argument);
    }
    EXPECT_DOUBLE_EQ(gas_volume(source, fraction), 0.0);
}
