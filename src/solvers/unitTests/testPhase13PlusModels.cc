/**
 * @file testPhase13PlusModels.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit and integration tests for Phase 13 and later multiphysics models.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/BoilingSourceModel.hh"
#include "equations/DelayedNeutronPrecursorModel.hh"
#include "equations/FeedbackMap.hh"
#include "equations/MaterialFeedbackModel.hh"
#include "equations/ScalarVoidFractionModel.hh"
#include "dataclass/Database.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VelocityFieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** @brief Build the shared single-cell fixture mesh. @return Assembled mesh. */
SimpleFluid::SP<MeshType> make_single_cell_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
}

/** @brief Build a native one-cell Cartesian handle without a legacy mesh. */
SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>>
make_native_single_cell_mesh()
{
    auto cartesian =
        std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
                {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    return std::make_shared<SimpleFluid::MeshHandle<Pack>>(
        std::move(cartesian));
}

/** @brief Build an eight-cell native line that partitions across MPI ranks. */
SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>>
make_native_distributed_line_mesh()
{
    auto cartesian =
        std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
                {0.0, 0.125, 0.25, 0.375, 0.5,
                 0.625, 0.75, 0.875, 1.0},
                {0.0, 1.0},
                {0.0, 1.0}}});
    return std::make_shared<SimpleFluid::MeshHandle<Pack>>(
        std::move(cartesian));
}

/**
 * @brief Construct water-like material fields on a mesh.
 *
 * @param mesh Mesh owning the material fields.
 * @return Initialized water-property fields.
 */
SimpleFluid::MaterialPropertyFields<Pack> make_water_properties(
    const SimpleFluid::SP<MeshType>& mesh)
{
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = 1000.0;
    options.density = 1000.0;
    options.specific_heat_capacity = 4200.0;
    options.dynamic_viscosity = 1.0e-3;
    options.thermal_conductivity = 0.6;
    return {mesh, options, time_options};
}

/** @brief Build deterministic material options for energy tests. @return Model options. */
SimpleFluid::BoussinesqModelOptions make_energy_test_model_options()
{
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = 2.0;
    options.density = 2.0;
    options.specific_heat_capacity = 5.0;
    options.dynamic_viscosity = 0.0;
    options.thermal_conductivity = 0.0;
    return options;
}

/**
 * @brief Build transport-free time options for energy tests.
 *
 * @param time_step Requested physical time step.
 * @return Configured time options.
 */
SimpleFluid::TimeStepperOptions make_energy_test_time_options(
    double time_step)
{
    SimpleFluid::TimeStepperOptions options;
    options.time_step = time_step;
    options.kinematic_viscosity = 0.0;
    options.thermal_diffusivity = 0.0;
    options.thermal_expansion = 0.0;
    options.gravity_x = 0.0;
    options.gravity_y = 0.0;
    options.gravity_z = 0.0;
    return options;
}

/** @brief Build stable Sheng two-population test options. @return Radiolysis options. */
SimpleFluid::RadiolyticGasOptions make_sheng_test_options()
{
    SimpleFluid::RadiolyticGasOptions options;
    options.mode =
        SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    options.hydrogen_yield_mol_per_j = 2.0e-7;
    options.max_source_alpha_rate = 1.0;
    options.reference_pressure = 1.0e5;
    options.henry_coefficient = 1.0e-5;
    options.surface_tension = 0.07;
    options.hydrogen_diffusivity = 1.0e-8;
    options.uranium_concentration_mol_per_m3 = 1000.0;
    options.hydrogen_yield_molecules_per_100_ev = 1.8;
    options.min_radius = 1.0e-12;
    options.max_radius = 1.0e-3;
    options.min_population = 1.0e-40;
    options.max_population = 1.0e40;
    options.micro_to_large_conversion_coefficient = 0.0;
    return options;
}

} // namespace

/** @brief Verify bulk boiling activation and latent heat remain consistent. */
TEST(BoilingSourceModelTest, BulkThresholdAndLatentHeatAreConsistent)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 360.0, "temperature");
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(mesh);

    SimpleFluid::BoilingSourceOptions options;
    options.enable_bulk_boiling = true;
    options.saturation_temperature = 373.0;
    options.boiling_activation_delta_t = 2.0;
    options.boiling_time_scale = 4.0;
    options.latent_heat = 10.0;
    options.gas_density = 2.0;
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);

    model.update(1.0e-6, temperature, material, void_model);
    EXPECT_DOUBLE_EQ(model.source_alpha_boil().value(0), 0.0);
    EXPECT_DOUBLE_EQ(model.latent_heat_sink().value(0), 0.0);
    EXPECT_DOUBLE_EQ(model.phase_change_mass_rate().value(0), 0.0);
    EXPECT_DOUBLE_EQ(model.rejected_vapor_mass_rate().value(0), 0.0);
    void_model.update_explicit(1.0e-6, nullptr, &model.source_alpha_boil());
    model.complete_void_fraction_update(1.0e-6, void_model, true);

    temperature.put_scalar(383.0);
    model.update(1.0e-6, temperature, material, void_model);
    const auto expected_energy =
        1000.0 * 4200.0 * (383.0 - 373.0) / 4.0;
    EXPECT_NEAR(
        model.latent_heat_sink().value(0), expected_energy, 1.0e-9);
    EXPECT_NEAR(
        model.source_alpha_boil().value(0),
        expected_energy / options.latent_heat / options.gas_density,
        1.0e-9);
    EXPECT_NEAR(
        -model.temperature_source(0),
        model.latent_heat_sink().value(0),
        1.0e-12);
    EXPECT_DOUBLE_EQ(model.phase_change_mass_rate().value(0), model.latent_heat_sink().value(0) / options.latent_heat);
    const auto expected_accepted_mass = expected_energy / options.latent_heat * mesh->cell_volume(0) * 1.0e-6;
    EXPECT_NEAR(model.accepted_evaporation_mass_this_step(), expected_accepted_mass, 1.0e-12);
    EXPECT_DOUBLE_EQ(model.rejected_vapor_mass_this_step(), 0.0);

    void_model.update_explicit(1.0e-6, nullptr, &model.source_alpha_boil());
    model.complete_void_fraction_update(1.0e-6, void_model, true);
    EXPECT_NEAR(model.global_submerged_steam_mass(), expected_accepted_mass, 1.0e-12);
    EXPECT_NEAR(model.global_submerged_steam_volume(), expected_accepted_mass / options.gas_density, 1.0e-12);
    EXPECT_NEAR(model.last_phase_change_diagnostics().mass_balance_residual, 0.0, 1.0e-14);
}

/** @brief Verify wall boiling power is distributed to the owning cell. */
TEST(BoilingSourceModelTest, WallSourceDistributesToOwnerCell)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 300.0, "temperature");
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(mesh);

    SimpleFluid::BoilingSourceOptions options;
    options.enable_wall_boiling = true;
    options.latent_heat = 20.0;
    options.gas_density = 4.0;
    options.wall_evaporation_fraction = 0.5;
    options.wall_heat_flux = 80.0;
    options.wall_boiling_patches = {"zmax"};
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);
    model.update(1.0, temperature, material, void_model);

    Pack::local_ordinal_type zmax_face = 0;
    bool found = false;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) == "zmax"
            && !batch.face_lids.empty())
        {
            zmax_face = batch.face_lids.front();
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found);
    const auto expected_mass_rate =
        options.wall_evaporation_fraction
      * options.wall_heat_flux
      / options.latent_heat
      * mesh->face_area(zmax_face)
      / mesh->cell_volume(0);
    EXPECT_NEAR(
        model.latent_heat_sink().value(0),
        expected_mass_rate * options.latent_heat,
        1.0e-12);
    EXPECT_NEAR(
        model.source_alpha_boil().value(0),
        expected_mass_rate / options.gas_density,
        1.0e-12);
    EXPECT_DOUBLE_EQ(model.phase_change_mass_rate().value(0), model.latent_heat_sink().value(0) / options.latent_heat);
    EXPECT_DOUBLE_EQ(model.rejected_vapor_mass_rate().value(0), 0.0);
}

/** @brief Verify active boiling modes reject nonphysical parameters. */
TEST(BoilingSourceModelTest, RejectsInvalidActiveParameters)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::BoilingSourceOptions options;
    options.enable_bulk_boiling = true;
    options.latent_heat = 0.0;
    EXPECT_THROW(
        SimpleFluid::BoilingSourceModel<Pack>(mesh, options),
        std::invalid_argument);

    options.latent_heat = 1.0;
    options.gas_density = 0.0;
    EXPECT_THROW(
        SimpleFluid::BoilingSourceModel<Pack>(mesh, options),
        std::invalid_argument);

    options.enable_bulk_boiling = false;
    options.enable_wall_boiling = true;
    options.gas_density = 1.0;
    options.wall_heat_flux = -1.0;
    EXPECT_THROW(
        SimpleFluid::BoilingSourceModel<Pack>(mesh, options),
        std::invalid_argument);
}

/** @brief Verify boiling updates reject a non-positive time step. */
TEST(BoilingSourceModelTest, RejectsInvalidTimeStep)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 383.0, "temperature");
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(mesh);
    SimpleFluid::BoilingSourceOptions options;
    options.enable_bulk_boiling = true;
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);

    EXPECT_THROW(
        model.update(0.0, temperature, material, void_model),
        std::invalid_argument);
}

/** @brief Verify boiling reserves shared void capacity for radiolysis. */
TEST(BoilingSourceModelTest,
     ReservesCanonicalVoidCapacityForRadiolysis)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 383.0, "temperature");
    FieldType radiolysis(mesh, 0.08, "S_alpha_rad");

    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.initial_alpha = 0.1;
    void_options.alpha_max = 0.2;
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(
        mesh, void_options);

    SimpleFluid::BoilingSourceOptions boiling_options;
    boiling_options.enable_bulk_boiling = true;
    boiling_options.saturation_temperature = 373.0;
    boiling_options.boiling_time_scale = 1.0;
    boiling_options.latent_heat = 10.0;
    boiling_options.gas_density = 1.0;
    SimpleFluid::BoilingSourceModel<Pack> boiling_model(
        mesh, boiling_options);

    boiling_model.update(
        1.0,
        temperature,
        material,
        void_model,
        &radiolysis);
    void_model.update_explicit(
        1.0, &radiolysis, &boiling_model.source_alpha_boil());

    EXPECT_NEAR(
        boiling_model.source_alpha_boil().value(0), 0.02, 1.0e-14);
    EXPECT_NEAR(
        boiling_model.latent_heat_sink().value(0), 0.2, 1.0e-14);
    EXPECT_DOUBLE_EQ(boiling_model.phase_change_mass_rate().value(0),
        boiling_model.latent_heat_sink().value(0) / boiling_options.latent_heat);
    EXPECT_NEAR(void_model.alpha_g().value(0), 0.2, 1.0e-14);
    EXPECT_NEAR(
        void_model.source_alpha_total().value(0), 0.1, 1.0e-14);
    const auto requested_mass = 1000.0 * 4200.0 * (383.0 - 373.0) / boiling_options.boiling_time_scale /
                                boiling_options.latent_heat * mesh->cell_volume(0);
    const auto accepted_mass = 0.02 * boiling_options.gas_density * mesh->cell_volume(0);
    EXPECT_NEAR(boiling_model.accepted_evaporation_mass_this_step(), accepted_mass, 1.0e-14);
    EXPECT_NEAR(boiling_model.rejected_vapor_mass_this_step(), requested_mass - accepted_mass, 1.0e-9);
    EXPECT_NEAR(
        boiling_model.last_phase_change_diagnostics().rejected_void_volume, requested_mass - accepted_mass, 1.0e-9);
    boiling_model.complete_void_fraction_update(1.0, void_model, true);
    EXPECT_NEAR(boiling_model.global_submerged_steam_mass(), accepted_mass, 1.0e-14);
    EXPECT_NEAR(boiling_model.last_phase_change_diagnostics().void_balance_residual, 0.0, 1.0e-14);

    radiolysis.put_scalar(0.0);
    boiling_model.update(
        1.0,
        temperature,
        material,
        void_model,
        &radiolysis);
    EXPECT_DOUBLE_EQ(boiling_model.source_alpha_boil().value(0), 0.0);
    EXPECT_DOUBLE_EQ(boiling_model.latent_heat_sink().value(0), 0.0);
    EXPECT_DOUBLE_EQ(boiling_model.phase_change_mass_rate().value(0), 0.0);
    EXPECT_NEAR(boiling_model.rejected_vapor_mass_this_step(), requested_mass, 1.0e-9);
    void_model.update_explicit(1.0, nullptr, &boiling_model.source_alpha_boil());
    boiling_model.complete_void_fraction_update(1.0, void_model, true);
    EXPECT_NEAR(
        boiling_model.last_phase_change_diagnostics().cumulative_accepted_evaporation_mass, accepted_mass, 1.0e-14);
}

/** @brief Verify collapse-created void capacity is bounded by the time step. */
TEST(BoilingSourceModelTest, CollapseCapacityIsTimestepBounded)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 383.0, "temperature");

    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.2;
    void_options.initial_alpha = void_options.alpha_max;
    void_options.alpha_collapse_time = 0.1;
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(
        mesh, void_options);

    SimpleFluid::BoilingSourceOptions boiling_options;
    boiling_options.enable_bulk_boiling = true;
    boiling_options.saturation_temperature = 373.0;
    boiling_options.boiling_time_scale = 1.0;
    boiling_options.latent_heat = 10.0;
    boiling_options.gas_density = 1.0;
    SimpleFluid::BoilingSourceModel<Pack> boiling_model(
        mesh, boiling_options);

    boiling_model.update(
        2.0,
        temperature,
        material,
        void_model);
    void_model.update_explicit(
        2.0, nullptr, &boiling_model.source_alpha_boil());

    EXPECT_NEAR(
        boiling_model.source_alpha_boil().value(0), 0.1, 1.0e-14);
    EXPECT_NEAR(
        boiling_model.latent_heat_sink().value(0), 1.0, 1.0e-14);
    EXPECT_NEAR(void_model.alpha_g().value(0), 0.2, 1.0e-14);
    EXPECT_NEAR(
        void_model.source_alpha_total().value(0), 0.0, 1.0e-14);
    boiling_model.complete_void_fraction_update(2.0, void_model, true);
    EXPECT_NEAR(boiling_model.accepted_evaporation_mass_this_step(), 0.2, 1.0e-14);
    EXPECT_NEAR(boiling_model.condensed_liquid_mass_this_step(), 0.0, 1.0e-14);
    EXPECT_NEAR(boiling_model.global_submerged_steam_mass(), 0.2, 1.0e-14);
    EXPECT_NEAR(boiling_model.last_phase_change_diagnostics().nonsteam_collapse_volume, 0.2, 1.0e-14);
    EXPECT_DOUBLE_EQ(boiling_model.condensation_latent_heat_release().value(0), 0.0);
    EXPECT_NEAR(boiling_model.temperature_source(0), -1.0, 1.0e-14);
    EXPECT_NEAR(boiling_model.last_phase_change_diagnostics().mass_balance_residual, 0.0, 1.0e-14);
}

/** @brief Legacy boiling does not add condensate heat without free-surface ownership. */
TEST(BoilingSourceModelTest, PhaseInventoryCouplingIsExplicitAndDefaultOff)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 383.0, "temperature");
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.2;
    void_options.initial_alpha = 0.1;
    void_options.alpha_collapse_time = 1.0;
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(mesh, void_options);
    SimpleFluid::BoilingSourceOptions options;
    options.enable_bulk_boiling = true;
    options.saturation_temperature = 373.0;
    options.boiling_time_scale = 1.0;
    options.latent_heat = 10.0;
    options.gas_density = 1.0;
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);

    model.update(0.1, temperature, material, void_model);
    void_model.update_explicit(0.1, nullptr, &model.source_alpha_boil());
    model.complete_void_fraction_update(0.1, void_model);

    EXPECT_DOUBLE_EQ(model.condensed_liquid_mass_this_step(), 0.0);
    EXPECT_DOUBLE_EQ(model.global_submerged_steam_mass(), 0.0);
    EXPECT_DOUBLE_EQ(model.condensation_latent_heat_release().value(0), 0.0);
    EXPECT_DOUBLE_EQ(model.temperature_source(0), -model.latent_heat_sink().value(0));
}

/** @brief Verify accepted mass is the sole liquid-inventory decrement input. */
TEST(BoilingSourceModelTest, PublishesAcceptedEvaporationMassWithoutSilentCapLoss)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 300.0, "temperature");
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.1;
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(mesh, void_options);

    SimpleFluid::BoilingSourceOptions options;
    options.enable_wall_boiling = true;
    options.latent_heat = 20.0;
    options.gas_density = 4.0;
    options.wall_evaporation_fraction = 0.5;
    options.wall_heat_flux = 80.0;
    options.wall_boiling_patches = {"zmax"};
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);

    constexpr double time_step = 0.5;
    model.update(time_step, temperature, material, void_model);

    // Requested wall evaporation is 2 kg/(m^3 s), but the void cap admits
    // only 0.8 kg/(m^3 s): a liquid inventory must remove the accepted 0.4 kg.
    EXPECT_NEAR(model.phase_change_mass_rate().value(0), 0.8, 1.0e-14);
    EXPECT_NEAR(model.rejected_vapor_mass_rate().value(0), 1.2, 1.0e-14);
    EXPECT_NEAR(model.accepted_evaporation_mass_this_step(), 0.4, 1.0e-14);
    EXPECT_NEAR(model.rejected_vapor_mass_this_step(), 0.6, 1.0e-14);
    EXPECT_DOUBLE_EQ(model.phase_change_mass_rate().value(0), model.latent_heat_sink().value(0) / options.latent_heat);

    void_model.update_explicit(time_step, nullptr, &model.source_alpha_boil());
    model.complete_void_fraction_update(time_step, void_model, true);
    EXPECT_NEAR(model.global_submerged_steam_mass(), 0.4, 1.0e-14);
    EXPECT_NEAR(model.global_submerged_steam_volume(), 0.1, 1.0e-14);
    EXPECT_NEAR(model.last_phase_change_diagnostics().mass_balance_residual, 0.0, 1.0e-14);
}

/** @brief Verify scalar collapse returns tracked steam mass conservatively. */
TEST(BoilingSourceModelTest, CompletionBalancesEvaporationCollapseAndCondensateReturn)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 383.0, "temperature");
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.2;
    void_options.alpha_collapse_time = 0.5;
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(mesh, void_options);

    SimpleFluid::BoilingSourceOptions options;
    options.enable_bulk_boiling = true;
    options.saturation_temperature = 373.0;
    options.boiling_time_scale = 1.0;
    options.latent_heat = 10.0;
    options.gas_density = 2.0;
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);

    constexpr double time_step = 0.1;
    model.update(time_step, temperature, material, void_model);
    void_model.update_explicit(time_step, nullptr, &model.source_alpha_boil());
    model.complete_void_fraction_update(time_step, void_model, true);
    EXPECT_NEAR(model.accepted_evaporation_mass_this_step(), 0.4, 1.0e-14);
    EXPECT_NEAR(model.global_submerged_steam_mass(), 0.4, 1.0e-14);
    EXPECT_NEAR(model.global_submerged_steam_volume(), 0.2, 1.0e-14);

    temperature.put_scalar(300.0);
    model.update(time_step, temperature, material, void_model);
    void_model.update_explicit(time_step, nullptr, &model.source_alpha_boil());
    model.complete_void_fraction_update(time_step, void_model, true);

    const auto& diagnostics = model.last_phase_change_diagnostics();
    EXPECT_DOUBLE_EQ(diagnostics.accepted_evaporation_mass, 0.0);
    EXPECT_NEAR(diagnostics.scalar_void_collapse_volume, 0.04, 1.0e-14);
    EXPECT_DOUBLE_EQ(diagnostics.nonsteam_collapse_volume, 0.0);
    EXPECT_NEAR(diagnostics.condensed_liquid_mass, 0.08, 1.0e-14);
    EXPECT_NEAR(diagnostics.condensation_latent_energy_release, diagnostics.condensed_liquid_mass * options.latent_heat,
        1.0e-14);
    EXPECT_NEAR(diagnostics.submerged_steam_mass, 0.32, 1.0e-14);
    EXPECT_NEAR(diagnostics.submerged_steam_volume, 0.16, 1.0e-14);
    EXPECT_NEAR(diagnostics.cumulative_accepted_evaporation_mass, 0.4, 1.0e-14);
    EXPECT_NEAR(diagnostics.cumulative_condensed_liquid_mass, 0.08, 1.0e-14);
    EXPECT_NEAR(diagnostics.mass_balance_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(diagnostics.void_balance_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(diagnostics.latent_energy_balance_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(model.condensation_latent_heat_release().value(0),
        diagnostics.condensed_liquid_mass * options.latent_heat / (mesh->cell_volume(0) * time_step), 1.0e-14);
    EXPECT_NEAR(model.temperature_source(0), model.condensation_latent_heat_release().value(0), 1.0e-14);
}

/** @brief Verify source acceptance and inventory completion cannot overlap. */
TEST(BoilingSourceModelTest, EnforcesPhaseChangeCompletionOrder)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 383.0, "temperature");
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(mesh);
    SimpleFluid::BoilingSourceOptions options;
    options.enable_bulk_boiling = true;
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);

    model.update(0.1, temperature, material, void_model);
    EXPECT_TRUE(model.phase_change_completion_pending());
    EXPECT_THROW(model.update(0.1, temperature, material, void_model), std::logic_error);

    void_model.update_explicit(0.1, nullptr, &model.source_alpha_boil());
    model.complete_void_fraction_update(0.1, void_model, true);
    EXPECT_FALSE(model.phase_change_completion_pending());
    EXPECT_THROW(model.complete_void_fraction_update(0.1, void_model, true), std::logic_error);

    SimpleFluid::BoilingSourceModel<Pack> disabled(mesh);
    EXPECT_NO_THROW(disabled.complete_void_fraction_update(0.1, void_model));
}

/** @brief A failed void closure leaves steam and cumulative state uncommitted. */
TEST(BoilingSourceModelTest, FailedCompletionIsTransactionalAndCanBeRetried)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 383.0, "temperature");
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(mesh);
    SimpleFluid::BoilingSourceOptions options;
    options.enable_bulk_boiling = true;
    options.saturation_temperature = 373.0;
    options.boiling_time_scale = 1.0;
    options.latent_heat = 10.0;
    options.gas_density = 1.0;
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);

    constexpr double time_step = 0.1;
    model.update(time_step, temperature, material, void_model);
    const auto accepted = model.accepted_evaporation_mass_this_step();
    ASSERT_GT(accepted, 0.0);

    // Deliberately omit the accepted boiling source from the scalar update.
    void_model.update_explicit(time_step, nullptr, nullptr);
    EXPECT_THROW(model.complete_void_fraction_update(time_step, void_model, true), std::runtime_error);
    EXPECT_TRUE(model.phase_change_completion_pending());
    EXPECT_DOUBLE_EQ(model.global_submerged_steam_mass(), 0.0);
    EXPECT_DOUBLE_EQ(model.last_phase_change_diagnostics().cumulative_accepted_evaporation_mass, 0.0);
    EXPECT_DOUBLE_EQ(model.condensation_latent_heat_release().value(0), 0.0);

    void_model.update_explicit(time_step, nullptr, &model.source_alpha_boil());
    EXPECT_NO_THROW(model.complete_void_fraction_update(time_step, void_model, true));
    EXPECT_FALSE(model.phase_change_completion_pending());
    EXPECT_NEAR(model.global_submerged_steam_mass(), accepted, 1.0e-14);
}

/** @brief Rank-local invalid inputs are rejected collectively before reductions. */
TEST(BoilingSourceModelTest, CollectivelyRejectsRankAsymmetricInvalidCellState)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(2, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 383.0, "temperature");
    FieldType reserved(mesh, 0.0, "reserved_alpha_source");
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(mesh);
    SimpleFluid::BoilingSourceOptions options;
    options.enable_bulk_boiling = true;
    options.saturation_temperature = 373.0;
    options.boiling_time_scale = 1.0;
    options.latent_heat = 10.0;
    options.gas_density = 1.0;
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);

    int local_injected = 0;
    if (communicator->getRank() == 1 && mesh->num_owned_cells() > 0)
    {
        reserved.set_owned_value(0, std::numeric_limits<Pack::scalar_type>::quiet_NaN());
        local_injected = 1;
    }
    int injected = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, 1, &local_injected, &injected);
    ASSERT_EQ(injected, 1);
    reserved.sync_ghosts();
    EXPECT_THROW(model.update(0.1, temperature, material, void_model, &reserved), std::invalid_argument);
    EXPECT_FALSE(model.phase_change_completion_pending());

    reserved.put_scalar(0.0);
    if (communicator->getRank() == 1 && mesh->num_owned_cells() > 0)
    {
        temperature.set_owned_value(0, std::numeric_limits<Pack::scalar_type>::quiet_NaN());
    }
    temperature.sync_ghosts();
    EXPECT_THROW(model.update(0.1, temperature, material, void_model, &reserved), std::runtime_error);
    EXPECT_FALSE(model.phase_change_completion_pending());
}

/** @brief Verify accepted evaporation is reduced over owned cells only. */
TEST(BoilingSourceModelTest, GloballyReducesAcceptedEvaporationMass)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 374.0, "temperature");
    SimpleFluid::ScalarVoidFractionModel<Pack> void_model(mesh);
    SimpleFluid::BoilingSourceOptions options;
    options.enable_bulk_boiling = true;
    options.saturation_temperature = 373.0;
    options.boiling_time_scale = 1.0;
    options.latent_heat = 1.0e8;
    options.gas_density = 1.0;
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);

    double local_volume = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        local_volume += mesh->cell_volume(static_cast<MeshType::local_ordinal_type>(owned));
    }
    double global_volume = 0.0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_volume, &global_volume);

    constexpr double time_step = 0.1;
    constexpr double accepted_mass_rate = 0.042;
    model.update(time_step, temperature, material, void_model);
    EXPECT_NEAR(model.accepted_evaporation_mass_this_step(), accepted_mass_rate * global_volume * time_step, 1.0e-14);
    void_model.update_explicit(time_step, nullptr, &model.source_alpha_boil());
    model.complete_void_fraction_update(time_step, void_model, true);
    EXPECT_NEAR(model.global_submerged_steam_mass(), accepted_mass_rate * global_volume * time_step, 1.0e-14);
}

/** @brief Verify radiolysis and boiling sources aggregate within void bounds. */
TEST(ScalarVoidFractionModelTest, AggregatesAndBoundsSources)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::ScalarVoidFractionOptions options;
    options.alpha_max = 0.2;
    SimpleFluid::ScalarVoidFractionModel<Pack> model(mesh, options);
    FieldType radiolysis(mesh, 0.4, "S_alpha_rad");
    FieldType boiling(mesh, 0.1, "S_alpha_boil");

    model.update_explicit(0.1, &radiolysis, &boiling);
    EXPECT_NEAR(model.alpha_g().value(0), 0.05, 1.0e-14);
    EXPECT_NEAR(model.alpha_l().value(0), 0.95, 1.0e-14);
    EXPECT_NEAR(
        model.source_alpha_total().value(0), 0.5, 1.0e-14);

    radiolysis.put_scalar(100.0);
    boiling.put_scalar(0.0);
    model.update_explicit(0.1, &radiolysis, &boiling);
    EXPECT_DOUBLE_EQ(model.alpha_g().value(0), options.alpha_max);
    EXPECT_NEAR(
        model.source_alpha_total().value(0),
        (options.alpha_max - 0.05) / 0.1,
        1.0e-14);
}

/** @brief Verify scalar slip is rejected directly but ignored by flat parsing. */
TEST(ScalarVoidFractionModelTest, RejectsNonzeroUnimplementedSlipVelocity)
{
    SimpleFluid::ScalarVoidFractionOptions options;
    options.constant_slip_velocity = 0.1;
    EXPECT_THROW(
        SimpleFluid::validate_scalar_void_fraction_options(options),
        std::invalid_argument);

    SimpleFluid::Database database;
    database.set("constant_slip_velocity", SimpleFluid::real_t{0.1});
    database.set(
        "bubble_rise_velocity_model", std::string{"constantSlip"});
    const auto parsed =
        SimpleFluid::scalar_void_fraction_options_from_database(database);
    EXPECT_DOUBLE_EQ(parsed.constant_slip_velocity, 0.0);

    const auto radiolytic =
        SimpleFluid::radiolytic_gas_options_from_database(database);
    EXPECT_EQ(
        radiolytic.rise_velocity_mode,
        SimpleFluid::BubbleRiseVelocityMode::ConstantSlip);
    EXPECT_DOUBLE_EQ(radiolytic.constant_slip_velocity, 0.1);
}

/** @brief Verify zero diffusivity preserves a nonuniform void field. */
TEST(ScalarVoidFractionModelTest,
     ZeroDiffusivityLeavesNonuniformAlphaUnchanged)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    SimpleFluid::ScalarVoidFractionOptions options;
    SimpleFluid::ScalarVoidFractionModel<Pack> model(mesh, options);
    FieldType initial_alpha(mesh, "initial_alpha_g");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        initial_alpha.set_owned_value(
            cell_lid,
            mesh->cell_centroid(cell_lid).x < 1.0 ? 0.12 : 0.34);
    }
    initial_alpha.sync_ghosts();
    model.initialize_from(initial_alpha);

    model.update_explicit(10.0, nullptr, nullptr);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto expected =
            mesh->cell_centroid(cell_lid).x < 1.0 ? 0.12 : 0.34;
        EXPECT_DOUBLE_EQ(model.alpha_g().value(cell_lid), expected);
        EXPECT_DOUBLE_EQ(model.alpha_l().value(cell_lid), 1.0 - expected);
        EXPECT_DOUBLE_EQ(
            model.source_alpha_total().value(cell_lid), 0.0);
    }
}

/** @brief Verify void diffusion smooths locally while conserving global inventory. */
TEST(ScalarVoidFractionModelTest,
     DiffusionSmoothsAndConservesGlobalVoid)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 1, 1, 0.125));
    const auto communicator = mesh->owned_cell_map()->getComm();
    SimpleFluid::ScalarVoidFractionOptions options;
    options.alpha_max = 0.9;
    options.alpha_diffusivity = 0.1;
    SimpleFluid::ScalarVoidFractionModel<Pack> model(mesh, options);
    FieldType initial_alpha(mesh, "initial_alpha_g");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        initial_alpha.set_owned_value(
            cell_lid,
            mesh->cell_centroid(cell_lid).x < 0.5 ? 0.8 : 0.1);
    }
    initial_alpha.sync_ghosts();
    model.initialize_from(initial_alpha);

    auto global_integral = [&](const FieldType& field)
    {
        double local_integral = 0.0;
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<MeshType::local_ordinal_type>(owned);
            local_integral +=
                field.value(cell_lid) * mesh->cell_volume(cell_lid);
        }
        double integral = 0.0;
        Teuchos::reduceAll(
            *communicator,
            Teuchos::REDUCE_SUM,
            1,
            &local_integral,
            &integral);
        return integral;
    };
    const auto inventory_before = global_integral(model.alpha_g());

    model.update_explicit(0.2, nullptr, nullptr);

    const auto inventory_after = global_integral(model.alpha_g());
    double local_minimum = std::numeric_limits<double>::max();
    double local_maximum = std::numeric_limits<double>::lowest();
    int local_partition_faces = 0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto alpha = model.alpha_g().value(cell_lid);
        local_minimum = std::min(local_minimum, alpha);
        local_maximum = std::max(local_maximum, alpha);
        EXPECT_GE(alpha, options.alpha_min);
        EXPECT_LE(alpha, options.alpha_max);
        EXPECT_NEAR(
            model.alpha_l().value(cell_lid), 1.0 - alpha, 1.0e-14);
        EXPECT_DOUBLE_EQ(
            model.source_alpha_total().value(cell_lid), 0.0);
        for (const auto face_lid : mesh->faces(cell_lid))
        {
            if (!mesh->is_interior_face(face_lid))
            {
                continue;
            }
            const auto other =
                mesh->opposite_or_periodic_neighbor_cell(
                    face_lid, cell_lid);
            if (!mesh->is_owned_cell(other))
            {
                ++local_partition_faces;
            }
        }
    }
    double global_minimum = 0.0;
    double global_maximum = 0.0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        1,
        &local_minimum,
        &global_minimum);
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_maximum,
        &global_maximum);
    EXPECT_GT(global_minimum, 0.1);
    EXPECT_LT(global_maximum, 0.8);
    EXPECT_NEAR(
        inventory_after,
        inventory_before,
        std::max(1.0e-12, std::abs(inventory_before) * 1.0e-10));

    if (communicator->getSize() > 1)
    {
        int global_partition_faces = 0;
        Teuchos::reduceAll(
            *communicator,
            Teuchos::REDUCE_SUM,
            1,
            &local_partition_faces,
            &global_partition_faces);
        EXPECT_GT(global_partition_faces, 0);
    }
}

/** @brief Verify mirroring derives the realized rate and liquid complement. */
TEST(ScalarVoidFractionModelTest, MirrorDerivesRealizedRateAndComplement)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::ScalarVoidFractionOptions options;
    options.initial_alpha = 0.1;
    SimpleFluid::ScalarVoidFractionModel<Pack> model(mesh, options);
    FieldType advanced_alpha(mesh, 0.02, "advanced_alpha_g");

    model.mirror(advanced_alpha, 0.2);

    EXPECT_DOUBLE_EQ(model.alpha_g().value(0), 0.02);
    EXPECT_DOUBLE_EQ(model.alpha_l().value(0), 0.98);
    EXPECT_NEAR(model.source_alpha_total().value(0), -0.4, 1.0e-14);
}

/** @brief Verify initial Sheng state publication is not reported as evolution. */
TEST(Phase13PlusCouplingTest,
     AdvancedMirrorDoesNotReportInitialPublicationAsAStateChange)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh,
        {},
        make_energy_test_time_options(0.1),
        {},
        make_energy_test_model_options());
    solver.initialize_heated_box(300.0, 300.0);

    SimpleFluid::FissionPowerSourceOptions fission;
    fission.profile = SimpleFluid::FissionPowerProfile::Constant;
    fission.power_density = 0.0;
    solver.configure_fission_power_source(fission);

    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.initial_alpha = 0.1;
    auto& void_model =
        solver.configure_scalar_void_fraction(void_options);
    auto& radiolysis =
        solver.configure_radiolytic_gas(make_sheng_test_options());

    solver.step();

    EXPECT_DOUBLE_EQ(radiolysis.alpha_g().value(0), 0.0);
    EXPECT_DOUBLE_EQ(radiolysis.source_alpha_rad().value(0), 0.0);
    EXPECT_DOUBLE_EQ(void_model.alpha_g().value(0), 0.0);
    EXPECT_DOUBLE_EQ(void_model.alpha_l().value(0), 1.0);
    EXPECT_DOUBLE_EQ(void_model.source_alpha_total().value(0), 0.0);
}

/** @brief Verify ideal radiolysis seeds only implicit pre-step void models. */
TEST(Phase13PlusCouplingTest,
     IdealRadiolysisSeedsOnlyImplicitPreStepVoidConfiguration)
{
    SimpleFluid::RadiolyticGasOptions radiolysis;
    radiolysis.mode = SimpleFluid::RadiolyticGasMode::IdealGasSource;
    radiolysis.hydrogen_yield_mol_per_j = 1.0e-7;
    radiolysis.max_source_alpha_rate = 1.0;
    radiolysis.alpha_min = 0.05;
    radiolysis.alpha_max = 0.2;

    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh,
            {},
            make_energy_test_time_options(0.1),
            {},
            make_energy_test_model_options());
        solver.configure_boiling_source(
            SimpleFluid::BoilingSourceOptions{});
        solver.configure_radiolytic_gas(radiolysis);

        const auto* void_model =
            solver.find_scalar_void_fraction_model();
        ASSERT_NE(void_model, nullptr);
        EXPECT_DOUBLE_EQ(void_model->options().alpha_min, 0.05);
        EXPECT_DOUBLE_EQ(void_model->options().alpha_max, 0.2);
        EXPECT_DOUBLE_EQ(void_model->alpha_g().value(0), 0.05);
    }

    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh,
            {},
            make_energy_test_time_options(0.1),
            {},
            make_energy_test_model_options());
        SimpleFluid::ScalarVoidFractionOptions void_options;
        void_options.alpha_max = 0.3;
        void_options.initial_alpha = 0.1;
        auto& void_model =
            solver.configure_scalar_void_fraction(void_options);
        solver.configure_radiolytic_gas(radiolysis);

        EXPECT_DOUBLE_EQ(void_model.options().alpha_min, 0.0);
        EXPECT_DOUBLE_EQ(void_model.options().alpha_max, 0.3);
        EXPECT_DOUBLE_EQ(void_model.alpha_g().value(0), 0.1);
    }
}

/** @brief Verify late radiolysis configuration preserves evolved implicit void. */
TEST(Phase13PlusCouplingTest,
     RadiolysisConfigurationPreservesEvolvedImplicitVoid)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh,
        {},
        make_energy_test_time_options(1.0),
        {},
        make_energy_test_model_options());
    solver.initialize_heated_box(383.0, 383.0);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 373.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    solver.configure_boiling_source(boiling);
    solver.step();

    auto* void_model = solver.find_scalar_void_fraction_model();
    ASSERT_NE(void_model, nullptr);
    const auto evolved_alpha = void_model->alpha_g().value(0);
    ASSERT_GT(evolved_alpha, 0.0);

    SimpleFluid::RadiolyticGasOptions radiolysis;
    radiolysis.mode = SimpleFluid::RadiolyticGasMode::IdealGasSource;
    radiolysis.hydrogen_yield_mol_per_j = 1.0e-7;
    radiolysis.max_source_alpha_rate = 1.0;
    radiolysis.alpha_max = 0.2;
    solver.configure_radiolytic_gas(radiolysis);

    EXPECT_DOUBLE_EQ(void_model->options().alpha_max, 0.95);
    EXPECT_DOUBLE_EQ(void_model->alpha_g().value(0), evolved_alpha);
}

/** @brief Verify ideal radiolysis honors the authoritative scalar void limit. */
TEST(Phase13PlusCouplingTest,
     IdealRadiolysisUsesAuthoritativeScalarVoidLimit)
{
    auto mesh = make_single_cell_mesh();
    auto time_options = make_energy_test_time_options(0.1);
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh,
        {},
        time_options,
        {},
        make_energy_test_model_options());
    solver.initialize_heated_box(300.0, 300.0);

    SimpleFluid::FissionPowerSourceOptions fission;
    fission.profile = SimpleFluid::FissionPowerProfile::Constant;
    fission.power_density = 100.0;
    solver.configure_fission_power_source(fission);

    SimpleFluid::RadiolyticGasOptions radiolysis;
    radiolysis.mode = SimpleFluid::RadiolyticGasMode::IdealGasSource;
    radiolysis.hydrogen_yield_mol_per_j = 1.0;
    radiolysis.reference_pressure = 1.0e5;
    radiolysis.max_source_alpha_rate = 1.0;
    auto& radiolytic_model =
        solver.configure_radiolytic_gas(radiolysis);

    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.2;
    void_options.initial_alpha = 0.19;
    auto& void_model =
        solver.configure_scalar_void_fraction(void_options);

    solver.step();

    const auto capacity_rate =
        (void_options.alpha_max - void_options.initial_alpha) / 0.1;
    EXPECT_NEAR(
        radiolytic_model.source_alpha_rad().value(0),
        capacity_rate,
        1.0e-13);
    EXPECT_NEAR(
        void_model.source_alpha_total().value(0),
        capacity_rate,
        1.0e-13);
    EXPECT_NEAR(
        void_model.alpha_g().value(0), void_options.alpha_max, 1.0e-13);
    EXPECT_NEAR(void_model.alpha_l().value(0), 0.8, 1.0e-13);
    EXPECT_NEAR(
        radiolytic_model.alpha_g().value(0),
        void_model.alpha_g().value(0),
        1.0e-13);
    EXPECT_NEAR(
        radiolytic_model.alpha_l().value(0),
        void_model.alpha_l().value(0),
        1.0e-13);

    solver.step();
    EXPECT_DOUBLE_EQ(radiolytic_model.source_alpha_rad().value(0), 0.0);
    EXPECT_DOUBLE_EQ(void_model.source_alpha_total().value(0), 0.0);
}

/** @brief Verify boiling energy matches void admitted at the scalar limit. */
TEST(Phase13PlusCouplingTest,
     BoilingEnergyMatchesVoidAdmittedAtScalarLimit)
{
    auto mesh = make_single_cell_mesh();
    auto time_options = make_energy_test_time_options(1.0);
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh,
        {},
        time_options,
        {},
        make_energy_test_model_options());
    solver.initialize_heated_box(383.0, 383.0);

    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.2;
    void_options.initial_alpha = 0.19;
    auto& void_model =
        solver.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling_options;
    boiling_options.enable_bulk_boiling = true;
    boiling_options.saturation_temperature = 373.0;
    boiling_options.boiling_time_scale = 1.0;
    boiling_options.latent_heat = 10.0;
    boiling_options.gas_density = 1.0;
    auto& boiling_model =
        solver.configure_boiling_source(boiling_options);

    solver.step();

    constexpr double accepted_source = 0.01;
    constexpr double accepted_sink = 0.1;
    EXPECT_NEAR(
        boiling_model.source_alpha_boil().value(0),
        accepted_source,
        1.0e-13);
    EXPECT_NEAR(
        void_model.source_alpha_total().value(0),
        accepted_source,
        1.0e-13);
    EXPECT_NEAR(void_model.alpha_g().value(0), 0.2, 1.0e-13);
    EXPECT_NEAR(
        boiling_model.latent_heat_sink().value(0),
        accepted_sink,
        1.0e-13);
    EXPECT_NEAR(
        boiling_model.latent_heat_sink().value(0),
        boiling_model.source_alpha_boil().value(0)
          * boiling_options.gas_density
          * boiling_options.latent_heat,
        1.0e-13);
    EXPECT_NEAR(solver.temperature().value(0), 382.99, 1.0e-10);
}

/** @brief Verify bulk boiling cannot consume more than step sensible heat. */
TEST(Phase13PlusCouplingTest,
     BulkBoilingCannotConsumeMoreThanStepSensibleSuperheat)
{
    auto mesh = make_single_cell_mesh();
    auto time_options = make_energy_test_time_options(2.0);
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh,
        {},
        time_options,
        {},
        make_energy_test_model_options());
    solver.initialize_heated_box(383.0, 383.0);

    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.95;
    auto& void_model =
        solver.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling_options;
    boiling_options.enable_bulk_boiling = true;
    boiling_options.saturation_temperature = 373.0;
    boiling_options.boiling_time_scale = 1.0;
    boiling_options.latent_heat = 1000.0;
    boiling_options.gas_density = 1.0;
    auto& boiling_model =
        solver.configure_boiling_source(boiling_options);

    solver.step();

    EXPECT_NEAR(
        boiling_model.latent_heat_sink().value(0), 50.0, 1.0e-10);
    EXPECT_NEAR(
        boiling_model.source_alpha_boil().value(0), 0.05, 1.0e-13);
    EXPECT_NEAR(
        void_model.source_alpha_total().value(0), 0.05, 1.0e-13);
    EXPECT_NEAR(void_model.alpha_g().value(0), 0.1, 1.0e-13);
    EXPECT_NEAR(solver.temperature().value(0), 373.0, 1.0e-10);
}

/** @brief Verify Sheng radiolysis rejects unconserved boiling combinations. */
TEST(Phase13PlusCouplingTest,
     RejectsUnconservedShengBoilingCombinationInEitherOrder)
{
    auto time_options = make_energy_test_time_options(0.1);
    SimpleFluid::BoilingSourceOptions boiling_options;
    boiling_options.enable_bulk_boiling = true;

    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh,
            {},
            time_options,
            {},
            make_energy_test_model_options());
        solver.configure_radiolytic_gas(make_sheng_test_options());
        EXPECT_THROW(
            solver.configure_boiling_source(boiling_options),
            std::invalid_argument);
    }

    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh,
            {},
            time_options,
            {},
            make_energy_test_model_options());
        solver.configure_boiling_source(boiling_options);
        EXPECT_THROW(
            solver.configure_radiolytic_gas(make_sheng_test_options()),
            std::invalid_argument);
    }

    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh,
            {},
            time_options,
            {},
            make_energy_test_model_options());
        SimpleFluid::RadiolyticGasOptions ideal;
        ideal.mode = SimpleFluid::RadiolyticGasMode::IdealGasSource;
        ideal.hydrogen_yield_mol_per_j = 1.0e-7;
        ideal.max_source_alpha_rate = 1.0;
        auto& radiolysis = solver.configure_radiolytic_gas(ideal);
        solver.configure_boiling_source(boiling_options);

        radiolysis.configure(make_sheng_test_options());
        EXPECT_THROW(solver.step(), std::logic_error);
        EXPECT_DOUBLE_EQ(solver.time(), 0.0);
        EXPECT_EQ(solver.step_index(), 0);
    }
}

/** @brief Verify Sheng radiolysis rejects unconserved void collapse. */
TEST(Phase13PlusCouplingTest,
     RejectsUnconservedShengCollapseCombinationInEitherOrder)
{
    const auto time_options = make_energy_test_time_options(0.1);
    SimpleFluid::ScalarVoidFractionOptions collapse_options;
    collapse_options.alpha_collapse_time = 1.0;

    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh,
            {},
            time_options,
            {},
            make_energy_test_model_options());
        solver.configure_scalar_void_fraction(collapse_options);
        EXPECT_THROW(
            solver.configure_radiolytic_gas(make_sheng_test_options()),
            std::invalid_argument);
    }

    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh,
            {},
            time_options,
            {},
            make_energy_test_model_options());
        solver.configure_radiolytic_gas(make_sheng_test_options());
        EXPECT_THROW(
            solver.configure_scalar_void_fraction(collapse_options),
            std::invalid_argument);
    }

    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh,
            {},
            time_options,
            {},
            make_energy_test_model_options());
        solver.configure_radiolytic_gas(make_sheng_test_options());
        auto* void_model = solver.find_scalar_void_fraction_model();
        ASSERT_NE(void_model, nullptr);
        void_model->configure(collapse_options);

        EXPECT_THROW(solver.step(), std::logic_error);
        EXPECT_DOUBLE_EQ(solver.time(), 0.0);
        EXPECT_EQ(solver.step_index(), 0);
    }
}

/** @brief Verify Boussinesq-void density feedback and material floors. */
TEST(MaterialFeedbackModelTest, BoussinesqVoidDensityAndFloors)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 310.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FieldType alpha(mesh, 0.25, "alpha_g");

    SimpleFluid::BoussinesqUpdateContext<Pack> context{
        0.0, 0, *mesh, temperature, pressure, velocity};
    SimpleFluid::MaterialFeedbackOptions options;
    options.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqVoid;
    options.reference_density = 1000.0;
    options.gas_density = 2.0;
    options.reference_temperature = 300.0;
    options.thermal_expansion = 0.01;
    options.reference_dynamic_viscosity = 1.0e-3;
    options.min_density = 10.0;
    options.min_viscosity = 2.0e-3;
    SimpleFluid::MaterialFeedbackModel<Pack> model(mesh, options);
    model.apply(context, &alpha, material);

    const auto expected_liquid_density = 1000.0 * (1.0 - 0.1);
    const auto expected_density =
        expected_liquid_density * 0.75 + 2.0 * 0.25;
    EXPECT_NEAR(material.density.value(0), expected_density, 1.0e-12);
    EXPECT_DOUBLE_EQ(material.dynamic_viscosity.value(0), 2.0e-3);
    EXPECT_DOUBLE_EQ(model.density_feedback().value(0), expected_density);
    EXPECT_DOUBLE_EQ(model.pure_liquid_density(temperature.value(0)), expected_liquid_density);

    alpha.put_scalar(0.75);
    model.apply(context, &alpha, material);
    EXPECT_NE(material.density.value(0), expected_density);
    EXPECT_DOUBLE_EQ(model.pure_liquid_density(temperature.value(0)), expected_liquid_density);
}

/** @brief Compare precursor source and decay integration with analytic values. */
TEST(DelayedNeutronPrecursorModelTest, SourceAndDecayAreAnalytic)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    SimpleFluid::DelayedNeutronPrecursorModel<Pack> zero_model(
        mesh, options);
    FieldType alpha_l(mesh, 1.0, "alpha_l");

    zero_model.advance(0.5, alpha_l, nullptr);
    EXPECT_DOUBLE_EQ(zero_model.concentration(0).value(0), 0.0);

    options.decay_constants = {2.0};
    options.initial_concentrations = {10.0};
    SimpleFluid::DelayedNeutronPrecursorModel<Pack> model(mesh, options);

    model.advance(0.5, alpha_l, nullptr);
    EXPECT_NEAR(
        model.concentration(0).value(0),
        10.0 * std::exp(-1.0),
        1.0e-12);

    options.initial_concentrations = {0.0};
    options.source_terms = {4.0};
    model.configure(options);
    model.advance(0.5, alpha_l, nullptr);
    const auto expected_source_solution =
        4.0 / 2.0 * (1.0 - std::exp(-1.0));
    EXPECT_NEAR(
        model.concentration(0).value(0),
        expected_source_solution,
        1.0e-12);
    const auto& diagnostics = model.last_inventory_diagnostics(0);
    const auto volume = mesh->cell_volume(0);
    EXPECT_NEAR(diagnostics.inventory_before, 0.0, 1.0e-14);
    EXPECT_NEAR(diagnostics.source_added, 2.0 * volume, 1.0e-12);
    EXPECT_NEAR(
        diagnostics.decay_removed,
        (2.0 - expected_source_solution) * volume,
        1.0e-12);
    EXPECT_NEAR(
        diagnostics.inventory_after,
        expected_source_solution * volume,
        1.0e-12);
    EXPECT_NEAR(diagnostics.balance_error, 0.0, 1.0e-12);

    options.decay_constants = {1.0e-18};
    model.configure(options);
    model.advance(0.5, alpha_l, nullptr);
    EXPECT_NEAR(model.concentration(0).value(0), 2.0, 1.0e-12);
}

/** @brief Keep the source response finite as positive decay approaches zero. */
TEST(DelayedNeutronPrecursorModelTest, TinyDecayConstantPreservesFiniteSourceLimit)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    options.decay_constants = {1.0e-320};
    options.source_terms = {4.0};
    SimpleFluid::DelayedNeutronPrecursorModel<Pack> model(mesh, options);
    FieldType alpha_l(mesh, 1.0, "alpha_l");

    model.advance(0.5, alpha_l, nullptr);

    const auto concentration = model.concentration(0).value(0);
    EXPECT_TRUE(std::isfinite(concentration));
    EXPECT_NEAR(concentration, 2.0, 1.0e-12);
    const auto& diagnostics = model.last_inventory_diagnostics(0);
    const auto volume = mesh->cell_volume(0);
    EXPECT_NEAR(diagnostics.source_added, 2.0 * volume, 1.0e-12);
    EXPECT_NEAR(diagnostics.decay_removed, 0.0, 1.0e-14);
    EXPECT_TRUE(std::isfinite(diagnostics.balance_error));
    EXPECT_NEAR(diagnostics.balance_error, 0.0, 1.0e-12);
}

/** @brief Avoid cancellation when decay is much faster than the time step. */
TEST(DelayedNeutronPrecursorModelTest, LargeDecayStepUsesDirectExponentialResponse)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    options.decay_constants = {1.0};
    options.source_terms = {1.0};
    SimpleFluid::DelayedNeutronPrecursorModel<Pack> model(mesh, options);
    FieldType alpha_l(mesh, 1.0, "alpha_l");

    model.advance(1.0e16, alpha_l, nullptr);

    const auto concentration = model.concentration(0).value(0);
    EXPECT_TRUE(std::isfinite(concentration));
    EXPECT_NEAR(concentration, 1.0, 1.0e-14);
    const auto& diagnostics = model.last_inventory_diagnostics(0);
    const auto volume = mesh->cell_volume(0);
    EXPECT_NEAR(diagnostics.source_added, 1.0e16 * volume, 4.0);
    EXPECT_NEAR(diagnostics.decay_removed, 1.0e16 * volume, 4.0);
    EXPECT_NEAR(
        diagnostics.inventory_after,
        concentration * volume,
        1.0e-14);
    EXPECT_NEAR(diagnostics.balance_error, 0.0, 1.0e-14);
}

/** @brief Keep balance closure when a large inventory decays to a survivor. */
TEST(DelayedNeutronPrecursorModelTest,
     LargeInitialInventoryKeepsStableBalanceDiagnostic)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    options.decay_constants = {1.0};
    options.initial_concentrations = {1.0e16};
    options.source_terms = {1.0};
    SimpleFluid::DelayedNeutronPrecursorModel<Pack> model(mesh, options);
    FieldType alpha_l(mesh, 1.0, "alpha_l");

    model.advance(1.0e16, alpha_l, nullptr);

    const auto concentration = model.concentration(0).value(0);
    EXPECT_NEAR(concentration, 1.0, 1.0e-14);
    const auto& diagnostics = model.last_inventory_diagnostics(0);
    const auto volume = mesh->cell_volume(0);
    EXPECT_NEAR(diagnostics.source_added, 1.0e16 * volume, 4.0);
    EXPECT_NEAR(diagnostics.decay_removed, 2.0e16 * volume, 8.0);
    EXPECT_NEAR(
        diagnostics.inventory_after,
        concentration * volume,
        1.0e-14);
    EXPECT_NEAR(diagnostics.balance_error, 0.0, 1.0e-14);
}

/** @brief Verify precursor inventory survives a changing liquid fraction. */
TEST(DelayedNeutronPrecursorModelTest,
     ChangingLiquidFractionPreservesInventory)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    options.initial_concentrations = {4.0};
    SimpleFluid::DelayedNeutronPrecursorModel<Pack> model(mesh, options);
    FieldType alpha_l(mesh, 0.8, "alpha_l");

    model.advance(0.25, alpha_l, nullptr);
    const auto volume = mesh->cell_volume(0);
    const auto initial_inventory =
        model.liquid_inventory(0).value(0) * volume;
    EXPECT_NEAR(model.concentration(0).value(0), 4.0, 1.0e-13);
    EXPECT_NEAR(initial_inventory, 0.8 * 4.0 * volume, 1.0e-13);

    alpha_l.put_scalar(0.2);
    model.advance(0.25, alpha_l, nullptr);

    EXPECT_NEAR(model.concentration(0).value(0), 16.0, 1.0e-12);
    EXPECT_NEAR(
        model.liquid_inventory(0).value(0) * volume,
        initial_inventory,
        1.0e-13);
    EXPECT_NEAR(
        alpha_l.value(0) * model.concentration(0).value(0) * volume,
        initial_inventory,
        1.0e-13);
}

/** @brief Verify effective precursor diffusion conserves liquid inventory. */
TEST(DelayedNeutronPrecursorModelTest,
     EffectiveDiffusivityConservesLiquidInventory)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    options.power_yields = {1.0};
    options.effective_diffusivity = 0.25;
    SimpleFluid::DelayedNeutronPrecursorModel<Pack> model(mesh, options);
    FieldType alpha_l(mesh, 0.5, "alpha_l");
    alpha_l.set_value(1, 0.25);
    FieldType fission_power(mesh, 0.0, "qdot_fission");
    fission_power.set_value(0, 5.0);

    model.advance(0.1, alpha_l, &fission_power);
    const auto old_0 = model.concentration(0).value(0);
    const auto old_1 = model.concentration(0).value(1);
    ASSERT_GT(old_0, old_1);

    const auto inventory_before =
        model.liquid_inventory(0).value(0) * mesh->cell_volume(0)
      + model.liquid_inventory(0).value(1) * mesh->cell_volume(1);
    constexpr double time_step = 0.2;
    model.advance(time_step, alpha_l, nullptr);

    MeshType::local_ordinal_type interior_face = -1;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<MeshType::local_ordinal_type>(face);
        if (mesh->is_interior_face(face_lid))
        {
            interior_face = face_lid;
            break;
        }
    }
    ASSERT_GE(interior_face, 0);
    const auto owner = mesh->owner_cell(interior_face);
    const auto neighbor = mesh->neighbor_cell(interior_face);
    const auto owner_diffusivity =
        alpha_l.value(owner) * options.effective_diffusivity;
    const auto neighbor_diffusivity =
        alpha_l.value(neighbor) * options.effective_diffusivity;
    const auto owner_distance =
        mesh->cell_to_face_distance(interior_face, owner);
    const auto neighbor_distance =
        mesh->cell_to_face_distance(interior_face, neighbor);
    const auto face_diffusivity =
        (owner_distance + neighbor_distance)
      / (owner_distance / owner_diffusivity
         + neighbor_distance / neighbor_diffusivity);
    const auto face_coefficient =
        face_diffusivity * mesh->face_area(interior_face)
      / mesh->face_cell_center_distance(interior_face);
    const auto weight_0 = alpha_l.value(0) * mesh->cell_volume(0);
    const auto weight_1 = alpha_l.value(1) * mesh->cell_volume(1);
    const auto difference =
        (old_0 - old_1)
      / (1.0
         + time_step * face_coefficient
           * (1.0 / weight_0 + 1.0 / weight_1));
    const auto weighted_total = weight_0 * old_0 + weight_1 * old_1;
    const auto expected_0 =
        (weighted_total + weight_1 * difference)
      / (weight_0 + weight_1);
    const auto expected_1 = expected_0 - difference;
    const auto new_0 = model.concentration(0).value(0);
    const auto new_1 = model.concentration(0).value(1);

    EXPECT_NEAR(new_0, expected_0, 1.0e-10);
    EXPECT_NEAR(new_1, expected_1, 1.0e-10);
    EXPECT_LT(new_0, old_0);
    EXPECT_GT(new_1, old_1);
    const auto inventory_after =
        model.liquid_inventory(0).value(0) * mesh->cell_volume(0)
      + model.liquid_inventory(0).value(1) * mesh->cell_volume(1);
    EXPECT_NEAR(inventory_after, inventory_before, 1.0e-11);
    for (MeshType::local_ordinal_type cell_lid = 0; cell_lid < 2; ++cell_lid)
    {
        EXPECT_NEAR(
            model.liquid_inventory(0).value(cell_lid),
            alpha_l.value(cell_lid)
              * model.concentration(0).value(cell_lid),
            1.0e-12);
    }
}

/** @brief Verify liquid flux advects precursors and reports a closed balance. */
TEST(DelayedNeutronPrecursorModelTest,
     LiquidAdvectionConservesDistributedInventory)
{
    using NativeMesh = SimpleFluid::MeshHandle<Pack>;
    using NativeModel =
        SimpleFluid::DelayedNeutronPrecursorModel<Pack, NativeMesh>;
    using NativeField = typename NativeModel::field_type;
    using NativeFaceField = typename NativeModel::face_flux_field_type;

    auto mesh = make_native_distributed_line_mesh();
    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    options.power_yields = {1.0};
    NativeModel model(mesh, options);
    NativeField alpha_l(mesh, 0.7, "alpha_l");
    NativeField fission_power(mesh, "qdot_fission");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<NativeMesh::local_ordinal_type>(owned);
        const auto x = mesh->cell_centroid(cell_lid).x;
        fission_power.set_owned_value(cell_lid, 2.0 - x);
    }
    fission_power.sync_ghosts();
    model.advance(0.1, alpha_l, &fission_power);

    const auto communicator = mesh->owned_cell_map()->getComm();
    NativeFaceField liquid_flux(mesh, 0.0, "liquid_face_flux");
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<NativeMesh::local_ordinal_type>(face);
        if (!mesh->is_owned_face(face_lid)
            || !mesh->is_interior_face(face_lid))
        {
            continue;
        }
        const auto owner = mesh->owner_cell(face_lid);
        const auto neighbor = mesh->neighbor_cell(face_lid);
        const auto crosses_partition =
            !mesh->is_owned_cell(owner) || !mesh->is_owned_cell(neighbor);
        if (communicator->getSize() > 1 && !crosses_partition)
        {
            continue;
        }
        const auto outward_area =
            mesh->face_area_vector_outward(face_lid, owner);
        liquid_flux.set_owned_value(face_lid, 0.05 * outward_area.x);
    }
    liquid_flux.sync_ghosts();

    std::vector<double> concentration_before(mesh->num_owned_cells());
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<NativeMesh::local_ordinal_type>(owned);
        concentration_before[owned] = model.concentration(0).value(cell_lid);
    }
    int local_nonzero_partition_fluxes = 0;
    for (const auto face_lid : liquid_flux.owned_face_ids())
    {
        if (!mesh->is_interior_face(face_lid))
        {
            continue;
        }
        const auto owner = mesh->owner_cell(face_lid);
        const auto neighbor = mesh->neighbor_cell(face_lid);
        if ((!mesh->is_owned_cell(owner) || !mesh->is_owned_cell(neighbor))
            && std::abs(liquid_flux.value(face_lid)) > 1.0e-14)
        {
            ++local_nonzero_partition_fluxes;
        }
    }
    if (communicator->getSize() > 1)
    {
        int global_nonzero_partition_fluxes = 0;
        Teuchos::reduceAll(
            *communicator,
            Teuchos::REDUCE_SUM,
            1,
            &local_nonzero_partition_fluxes,
            &global_nonzero_partition_fluxes);
        ASSERT_GT(global_nonzero_partition_fluxes, 0);
    }

    constexpr double time_step = 0.05;
    model.advance(time_step, alpha_l, nullptr, &liquid_flux);
    const auto& diagnostics = model.last_inventory_diagnostics(0);

    double local_profile_change = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<NativeMesh::local_ordinal_type>(owned);
        local_profile_change +=
            std::abs(model.concentration(0).value(cell_lid)
                     - concentration_before[owned])
          * mesh->cell_volume(cell_lid);
    }
    double global_profile_change = 0.0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_SUM,
        1,
        &local_profile_change,
        &global_profile_change);

    EXPECT_GT(diagnostics.inventory_before, 0.0);
    EXPECT_GT(global_profile_change, 1.0e-12);
    EXPECT_DOUBLE_EQ(diagnostics.source_added, 0.0);
    EXPECT_DOUBLE_EQ(diagnostics.decay_removed, 0.0);
    EXPECT_DOUBLE_EQ(diagnostics.boundary_outflow, 0.0);
    EXPECT_NEAR(
        diagnostics.inventory_after,
        diagnostics.inventory_before,
        std::max(1.0e-12,
                 std::abs(diagnostics.inventory_before) * 1.0e-10));
    EXPECT_NEAR(
        diagnostics.balance_error,
        0.0,
        std::max(1.0e-12,
                 std::abs(diagnostics.inventory_before) * 1.0e-10));
}

/** @brief Verify rank-local input differences fail collectively before transport. */
TEST(DelayedNeutronPrecursorModelTest,
     CollectivelyRejectsRankDivergentAdvanceSelection)
{
    using NativeMesh = SimpleFluid::MeshHandle<Pack>;
    using NativeModel =
        SimpleFluid::DelayedNeutronPrecursorModel<Pack, NativeMesh>;
    using NativeField = typename NativeModel::field_type;
    using NativeFaceField = typename NativeModel::face_flux_field_type;

    auto mesh = make_native_distributed_line_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "Rank-divergent validation requires at least two ranks.";
    }

    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    NativeModel model(mesh, options);
    NativeField alpha_l(mesh, 0.7, "alpha_l");
    NativeFaceField liquid_flux(mesh, 0.0, "liquid_face_flux");
    const auto* selected_flux =
        communicator->getRank() == 0 ? &liquid_flux : nullptr;
    EXPECT_THROW(
        model.advance(0.05, alpha_l, nullptr, selected_flux),
        std::invalid_argument);

    NativeModel invalid_field_model(mesh, options);
    if (communicator->getRank() == 0)
    {
        alpha_l.set_owned_value(0, -0.1);
    }
    EXPECT_THROW(
        invalid_field_model.advance(0.05, alpha_l, nullptr),
        std::invalid_argument);

    NativeModel inconsistent_timestep_model(mesh, options);
    alpha_l.put_scalar(0.7);
    const auto rank_dependent_time_step =
        communicator->getRank() == 0 ? 0.05 : 0.1;
    EXPECT_THROW(
        inconsistent_timestep_model.advance(
            rank_dependent_time_step, alpha_l, nullptr),
        std::invalid_argument);

}

/** @brief Reject rank-divergent configuration and invalid power before syncs. */
TEST(DelayedNeutronPrecursorModelTest,
     CollectivelyRejectsRankDivergentConfigurationAndPower)
{
    using NativeMesh = SimpleFluid::MeshHandle<Pack>;
    using NativeModel =
        SimpleFluid::DelayedNeutronPrecursorModel<Pack, NativeMesh>;
    using NativeField = typename NativeModel::field_type;

    auto mesh = make_native_distributed_line_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() != 2)
    {
        GTEST_SKIP() << "This regression requires exactly two MPI ranks.";
    }

    SimpleFluid::DelayedNeutronPrecursorOptions divergent_groups;
    divergent_groups.group_count =
        communicator->getRank() == 0 ? 0 : 1;
    EXPECT_THROW(
        (NativeModel(mesh, divergent_groups)),
        std::invalid_argument);

    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    options.initial_concentrations = {1.0};
    NativeModel model(mesh, options);

    EXPECT_THROW(model.configure(divergent_groups), std::invalid_argument);
    EXPECT_TRUE(model.enabled());
    EXPECT_EQ(model.group_count(), 1u);

    auto divergent_initial = options;
    divergent_initial.initial_concentrations = {
        communicator->getRank() == 0 ? 1.0 : 2.0};
    EXPECT_THROW(
        model.configure(divergent_initial),
        std::invalid_argument);

    auto divergent_options = options;
    divergent_options.source_terms = {
        communicator->getRank() == 0 ? 1.0 : 2.0};
    EXPECT_THROW(
        model.configure(divergent_options),
        std::invalid_argument);

    auto rank_local_invalid_options = options;
    rank_local_invalid_options.source_terms = {
        communicator->getRank() == 0 ? -1.0 : 1.0};
    EXPECT_THROW(
        model.configure(rank_local_invalid_options),
        std::invalid_argument);

    NativeField alpha_l(mesh, 0.7, "alpha_l");
    NativeField fission_power(mesh, 1.0, "qdot_fission");
    if (communicator->getRank() == 0)
    {
        EXPECT_GT(mesh->num_owned_cells(), 0u);
        if (mesh->num_owned_cells() > 0)
        {
            fission_power.set_owned_value(0, -1.0);
        }
    }
    fission_power.sync_ghosts();
    EXPECT_THROW(
        model.advance(0.05, alpha_l, &fission_power),
        std::invalid_argument);
}

/** @brief Reject asymmetric solver precursor state before entering a step. */
TEST(DelayedNeutronPrecursorModelTest,
     SolverRejectsRankDivergentPrecursorPresenceBeforeStep)
{
    auto mesh = make_native_distributed_line_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() != 2)
    {
        GTEST_SKIP() << "This regression requires exactly two MPI ranks.";
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.05;
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, {}, time_options, {});
    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    options.initial_concentrations = {1.0};
    solver.configure_precursors(options);
    if (communicator->getRank() == 0)
    {
        EXPECT_TRUE(solver.remove_precursor_model());
    }

    EXPECT_THROW(solver.step(), std::invalid_argument);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);
}

/** @brief Verify an implicit outlet loss is included in the diagnostics. */
TEST(DelayedNeutronPrecursorModelTest, BoundaryOutflowClosesInventoryBalance)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    options.initial_concentrations = {4.0};
    SimpleFluid::DelayedNeutronPrecursorModel<Pack> model(mesh, options);
    FieldType alpha_l(mesh, 0.75, "alpha_l");
    SimpleFluid::FaceField<Pack> liquid_flux(
        mesh, 0.0, "liquid_face_flux");

    MeshType::local_ordinal_type outlet = -1;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        static_cast<void>(batch_id);
        if (!batch.face_lids.empty())
        {
            outlet = batch.face_lids.front();
            break;
        }
    }
    ASSERT_GE(outlet, 0);
    liquid_flux.set_value(outlet, 0.25);

    constexpr double time_step = 0.2;
    model.advance(time_step, alpha_l, nullptr, &liquid_flux);
    const auto& diagnostics = model.last_inventory_diagnostics(0);
    const auto expected_concentration =
        4.0 / (1.0 + time_step * 0.25 / mesh->cell_volume(0));

    EXPECT_NEAR(model.concentration(0).value(0),
                expected_concentration, 1.0e-12);
    EXPECT_GT(diagnostics.boundary_outflow, 0.0);
    EXPECT_NEAR(
        diagnostics.inventory_after + diagnostics.boundary_outflow,
        diagnostics.inventory_before,
        1.0e-12);
    EXPECT_NEAR(diagnostics.balance_error, 0.0, 1.0e-12);
}

/** @brief Verify precursor diffusion includes explicit non-orthogonal correction. */
TEST(DelayedNeutronPrecursorModelTest,
     EffectiveDiffusivityIncludesExplicitNonOrthogonalCorrection)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    SimpleFluid::DelayedNeutronPrecursorOptions options;
    options.group_count = 1;
    options.power_yields = {1.0};
    options.effective_diffusivity = 0.2;
    SimpleFluid::DelayedNeutronPrecursorModel<Pack> model(mesh, options);
    FieldType alpha_l(mesh, 0.6, "alpha_l");
    FieldType fission_power(mesh, "qdot_fission");
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++cell_lid)
    {
        const auto center = mesh->cell_centroid(cell_lid);
        fission_power.set_owned_value(
            cell_lid,
            1.0 + center.x * center.x + 0.5 * center.y * center.z);
    }
    fission_power.sync_ghosts();

    model.advance(0.05, alpha_l, &fission_power);

    FieldType old_concentration(mesh, "precursor_old_concentration");
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++cell_lid)
    {
        old_concentration.set_owned_value(
            cell_lid, model.concentration(0).value(cell_lid));
    }
    old_concentration.sync_ghosts();

    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() > 1)
    {
        int local_partition_faces = 0;
        double local_partition_non_orthogonality = 0.0;
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<MeshType::local_ordinal_type>(owned);
            for (const auto face_lid : mesh->faces(cell_lid))
            {
                if (!mesh->is_interior_face(face_lid))
                {
                    continue;
                }
                const auto other =
                    mesh->opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                if (mesh->is_owned_cell(other))
                {
                    continue;
                }
                ++local_partition_faces;
                const auto tangential_area =
                    SimpleFluid::FVM::detail::non_orthogonal_area_vector(
                        mesh->face_area_vector_outward(face_lid, cell_lid),
                        mesh->cell_center_vector(face_lid, cell_lid));
                local_partition_non_orthogonality = std::max(
                    local_partition_non_orthogonality,
                    std::sqrt(tangential_area.dot(tangential_area)));
            }
        }
        int global_partition_faces = 0;
        double global_partition_non_orthogonality = 0.0;
        Teuchos::reduceAll(
            *communicator,
            Teuchos::REDUCE_SUM,
            1,
            &local_partition_faces,
            &global_partition_faces);
        Teuchos::reduceAll(
            *communicator,
            Teuchos::REDUCE_MAX,
            1,
            &local_partition_non_orthogonality,
            &global_partition_non_orthogonality);
        ASSERT_GT(global_partition_faces, 0);
        ASSERT_GT(global_partition_non_orthogonality, 1.0e-8);
    }

    auto global_inventory = [&]()
    {
        double local_inventory = 0.0;
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<MeshType::local_ordinal_type>(owned);
            local_inventory +=
                model.liquid_inventory(0).value(cell_lid)
              * mesh->cell_volume(cell_lid);
        }
        double inventory = 0.0;
        Teuchos::reduceAll(
            *communicator,
            Teuchos::REDUCE_SUM,
            1,
            &local_inventory,
            &inventory);
        return inventory;
    };
    const auto inventory_before = global_inventory();

    SimpleFluid::FaceField<Pack> zero_flux(
        mesh, 0.0, "precursor_zero_face_flux");
    FieldType diffusion_weight(
        mesh,
        0.6 * options.effective_diffusivity,
        "precursor_diffusion_weight");
    auto boundary_condition =
        [](int, size_t) -> SimpleFluid::BoundaryCondition
    {
        return {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    };
    auto boundary_value =
        [](int, size_t) -> Pack::scalar_type
    {
        return 0.0;
    };
    auto zero_source =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };
    constexpr double time_step = 0.05;
    const auto corrected_system =
        SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
            old_concentration,
            zero_flux,
            time_step,
            alpha_l,
            alpha_l,
            diffusion_weight,
            boundary_condition,
            boundary_value,
            zero_source,
            SimpleFluid::FVM::NonOrthogonalTreatment::Explicit,
            &old_concentration);
    const auto orthogonal_system =
        SimpleFluid::FVM::weighted_scalar_transport_system<Pack>(
            old_concentration,
            zero_flux,
            time_step,
            alpha_l,
            alpha_l,
            diffusion_weight,
            boundary_condition,
            boundary_value,
            zero_source,
            SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);
    FieldType corrected_solution(mesh, "corrected_solution");
    FieldType orthogonal_solution(mesh, "orthogonal_solution");
    SimpleFluid::BelosLinearSolver<Pack> transport_solver;
    const auto corrected_statistics =
        transport_solver.solve_with_statistics(
            corrected_system.matrix,
            *corrected_system.rhs,
            corrected_solution.owned_data(),
            SimpleFluid::LinearSolverOptions{});
    const auto orthogonal_statistics =
        transport_solver.solve_with_statistics(
            orthogonal_system.matrix,
            *orthogonal_system.rhs,
            orthogonal_solution.owned_data(),
            SimpleFluid::LinearSolverOptions{});
    ASSERT_TRUE(corrected_statistics.converged);
    ASSERT_TRUE(orthogonal_statistics.converged);

    model.advance(time_step, alpha_l, nullptr);
    const auto inventory_after = global_inventory();

    double max_non_orthogonal_effect = 0.0;
    double max_corrected_error = 0.0;
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
             mesh->num_owned_cells());
         ++cell_lid)
    {
        max_non_orthogonal_effect = std::max(
            max_non_orthogonal_effect,
            std::abs(corrected_solution.value(cell_lid)
                     - orthogonal_solution.value(cell_lid)));
        max_corrected_error = std::max(
            max_corrected_error,
            std::abs(model.concentration(0).value(cell_lid)
                     - corrected_solution.value(cell_lid)));
    }
    EXPECT_GT(max_non_orthogonal_effect, 1.0e-8);
    EXPECT_LT(max_corrected_error, 1.0e-11);
    EXPECT_NEAR(
        inventory_after,
        inventory_before,
        std::max(1.0e-12, std::abs(inventory_before) * 1.0e-11));
}

/** @brief Verify solver precursor configuration uses the initial liquid fraction. */
TEST(DelayedNeutronPrecursorModelTest,
     SolverConfigurationUsesInitialLiquidFraction)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    SimpleFluid::TimeStepperOptions time_options;
    auto model_options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(time_options);
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, boundary_conditions, time_options, {}, model_options);

    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.initial_alpha = 0.25;
    solver.configure_scalar_void_fraction(void_options);
    SimpleFluid::DelayedNeutronPrecursorOptions precursor_options;
    precursor_options.group_count = 1;
    precursor_options.initial_concentrations = {4.0};
    const auto& precursors =
        solver.configure_precursors(precursor_options);

    EXPECT_NEAR(precursors.concentration(0).value(0), 4.0, 1.0e-14);
    EXPECT_NEAR(
        precursors.liquid_inventory(0).value(0),
        0.75 * 4.0,
        1.0e-14);

    void_options.initial_alpha = 0.5;
    solver.configure_scalar_void_fraction(void_options);
    EXPECT_NEAR(precursors.concentration(0).value(0), 4.0, 1.0e-14);
    EXPECT_NEAR(
        precursors.liquid_inventory(0).value(0),
        0.5 * 4.0,
        1.0e-14);

    auto radiolytic_mesh = make_single_cell_mesh();
    SimpleFluid::BoussinesqSolver<Pack> radiolytic_solver(
        radiolytic_mesh,
        boundary_conditions,
        time_options,
        {},
        model_options);
    const auto& radiolytic_precursors =
        radiolytic_solver.configure_precursors(precursor_options);
    SimpleFluid::RadiolyticGasOptions radiolysis;
    radiolysis.mode =
        SimpleFluid::RadiolyticGasMode::IdealGasSource;
    radiolysis.alpha_min = 0.25;
    radiolysis.hydrogen_yield_mol_per_j = 1.0e-7;
    radiolysis.max_source_alpha_rate = 1.0;
    radiolytic_solver.configure_radiolytic_gas(radiolysis);
    EXPECT_NEAR(
        radiolytic_precursors.liquid_inventory(0).value(0),
        0.75 * 4.0,
        1.0e-14);
}

/** @brief Verify Phase 13 model options parse flat database keys and defaults. */
TEST(Phase13PlusDatabaseTest, ParsesFlatKeysAndDefaults)
{
    SimpleFluid::Database database;
    database.set("enable_bulk_boiling", true);
    database.set("saturation_temperature", SimpleFluid::real_t{371.0});
    database.set(
        "boiling_activation_delta_t", SimpleFluid::real_t{3.0});
    database.set("boiling_time_scale", SimpleFluid::real_t{5.0});
    database.set("latent_heat", SimpleFluid::real_t{100.0});
    database.set("gas_density", SimpleFluid::real_t{0.8});
    database.set("alpha_diffusivity", SimpleFluid::real_t{1.0e-5});
    database.set("alpha_collapse_time", SimpleFluid::real_t{12.0});
    database.set(
        "density_feedback_model", std::string{"boussinesqVoid"});
    database.set("min_density", SimpleFluid::real_t{5.0});
    database.set("min_viscosity", SimpleFluid::real_t{1.0e-5});
    database.set("precursor_group_count", 2);
    database.set(
        "precursor_decay_constants",
        SimpleFluid::ArrReal{0.1, 0.2});

    const auto boiling =
        SimpleFluid::boiling_source_options_from_database(database);
    EXPECT_TRUE(boiling.enable_bulk_boiling);
    EXPECT_DOUBLE_EQ(boiling.saturation_temperature, 371.0);
    EXPECT_DOUBLE_EQ(boiling.gas_density, 0.8);

    const auto alpha =
        SimpleFluid::scalar_void_fraction_options_from_database(
            database);
    EXPECT_DOUBLE_EQ(alpha.alpha_diffusivity, 1.0e-5);
    EXPECT_DOUBLE_EQ(alpha.alpha_collapse_time, 12.0);

    SimpleFluid::TimeStepperOptions time_options;
    time_options.reference_temperature = 300.0;
    time_options.thermal_expansion = 2.0e-4;
    auto model_options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    const auto feedback =
        SimpleFluid::material_feedback_options_from_database(
            database, model_options, time_options);
    EXPECT_EQ(
        feedback.density_mode,
        SimpleFluid::DensityFeedbackMode::BoussinesqVoid);
    EXPECT_DOUBLE_EQ(feedback.min_density, 5.0);
    EXPECT_DOUBLE_EQ(feedback.min_viscosity, 1.0e-5);

    const auto precursors =
        SimpleFluid::delayed_neutron_precursor_options_from_database(
            database);
    EXPECT_EQ(precursors.group_count, 2);
    EXPECT_EQ(
        precursors.decay_constants,
        (SimpleFluid::ArrReal{0.1, 0.2}));
}

/** @brief Verify each Phase 13 parser identifies an ill-typed option and owner. */
TEST(Phase13PlusDatabaseTest, ReportsWrongTypedOptionContext)
{
    auto expect_context =
        [](auto&& parse, const std::string& context, const std::string& key)
    {
        try
        {
            parse();
            FAIL() << "Expected a typed model option failure.";
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message(error.what());
            EXPECT_NE(message.find(context), std::string::npos);
            EXPECT_NE(message.find(key), std::string::npos);
            EXPECT_NE(message.find("wrong type"), std::string::npos);
        }
    };

    SimpleFluid::Database boiling;
    boiling.set("enable_bulk_boiling", std::string{"yes"});
    expect_context(
        [&] { SimpleFluid::boiling_source_options_from_database(boiling); },
        "Boiling source model",
        "enable_bulk_boiling");

    SimpleFluid::Database scalar_void;
    scalar_void.set("alpha_min", std::string{"zero"});
    expect_context(
        [&]
        {
            SimpleFluid::scalar_void_fraction_options_from_database(
                scalar_void);
        },
        "Scalar void-fraction model",
        "alpha_min");

    SimpleFluid::Database feedback;
    feedback.set("density_feedback_model", SimpleFluid::real_t{1.0});
    SimpleFluid::TimeStepperOptions time_options;
    const auto model_options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(time_options);
    expect_context(
        [&]
        {
            SimpleFluid::material_feedback_options_from_database(
                feedback, model_options, time_options);
        },
        "Material feedback model",
        "density_feedback_model");

    SimpleFluid::Database precursor;
    precursor.set("precursor_group_count", SimpleFluid::real_t{2.0});
    expect_context(
        [&]
        {
            SimpleFluid::delayed_neutron_precursor_options_from_database(
                precursor);
        },
        "Delayed-neutron precursor model",
        "precursor_group_count");
}

/** @brief Verify feedback mapping preserves a constant field average. */
TEST(FeedbackMapTest, PreservesConstantFieldAverage)
{
    auto mesh = make_single_cell_mesh();
    FieldType field(mesh, 42.0, "temperature");
    std::vector<SimpleFluid::FeedbackMap::FeedbackCell<Pack>> cells{
        {"tank", {0}}};
    const auto averages =
        SimpleFluid::FeedbackMap::volume_weighted_average<Pack>(
            field, cells);
    ASSERT_EQ(averages.size(), 1u);
    EXPECT_DOUBLE_EQ(averages[0], 42.0);

    SimpleFluid::FeedbackMap::import_power_density<Pack>(
        field, {12.0});
    EXPECT_DOUBLE_EQ(field.value(0), 12.0);
}

/** @brief Verify coarse feedback mapping preserves the source volume integral. */
TEST(FeedbackMapTest, PreservesVolumeIntegralForCoarsenedField)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    FieldType field(mesh, 0.0, "temperature");
    field.set_value(0, 2.0);
    field.set_value(1, 6.0);

    std::vector<SimpleFluid::FeedbackMap::FeedbackCell<Pack>> cells{
        {"tank", {0, 1}}};
    const auto averages =
        SimpleFluid::FeedbackMap::volume_weighted_average<Pack>(
            field, cells);
    const auto source_integral =
        field.value(0) * mesh->cell_volume(0)
      + field.value(1) * mesh->cell_volume(1);
    const auto mapped_volume =
        mesh->cell_volume(0) + mesh->cell_volume(1);

    ASSERT_EQ(averages.size(), 1u);
    EXPECT_NEAR(
        averages[0] * mapped_volume, source_integral, 1.0e-14);
}

/** @brief Advance the complete optional physical stack on a native handle. */
TEST(Phase13PlusCouplingTest, NativeMeshHandleAdvancesExtendedPhysicalStack)
{
    const auto mesh = make_native_single_cell_mesh();
    auto time_options = make_energy_test_time_options(0.1);
    auto model_options = make_energy_test_model_options();
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, {}, time_options, {}, model_options);
    solver.initialize_heated_box(383.0, 383.0);

    SimpleFluid::FissionPowerSourceOptions fission;
    fission.profile = SimpleFluid::FissionPowerProfile::Constant;
    fission.power_density = 100.0;
    solver.configure_fission_power_source(fission);

    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    void_options.initial_alpha = 0.1;
    auto& void_model =
        solver.configure_scalar_void_fraction(void_options);

    SimpleFluid::RadiolyticGasOptions radiolysis;
    radiolysis.mode = SimpleFluid::RadiolyticGasMode::IdealGasSource;
    radiolysis.hydrogen_yield_mol_per_j = 1.0e-7;
    radiolysis.max_source_alpha_rate = 1.0;
    auto& radiolytic_model =
        solver.configure_radiolytic_gas(radiolysis);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 373.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver.configure_boiling_source(boiling);

    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.density_mode = SimpleFluid::DensityFeedbackMode::Mixture;
    feedback.reference_density = 2.0;
    feedback.liquid_density = 2.0;
    feedback.gas_density = 1.0;
    feedback.reference_dynamic_viscosity = 0.0;
    feedback.min_viscosity = 0.0;
    auto& feedback_model = solver.configure_material_feedback(feedback);

    SimpleFluid::DelayedNeutronPrecursorOptions precursors;
    precursors.group_count = 1;
    precursors.decay_constants = {0.1};
    precursors.initial_concentrations = {1.0};
    precursors.power_yields = {1.0e-3};
    auto& precursor_model = solver.configure_precursors(precursors);

    solver.step();

    EXPECT_FALSE(mesh->legacy_mesh());
    EXPECT_EQ(solver.temperature().mesh_ptr(), mesh);
    EXPECT_EQ(solver.material_properties().density.mesh_ptr(), mesh);
    EXPECT_EQ(void_model.alpha_g().mesh_ptr(), mesh);
    EXPECT_EQ(radiolytic_model.source_alpha_rad().mesh_ptr(), mesh);
    EXPECT_EQ(boiling_model.source_alpha_boil().mesh_ptr(), mesh);
    EXPECT_EQ(feedback_model.density_feedback().mesh_ptr(), mesh);
    EXPECT_EQ(precursor_model.concentration(0).mesh_ptr(), mesh);
    EXPECT_TRUE(std::isfinite(solver.temperature().value(0)));
    EXPECT_TRUE(std::isfinite(void_model.alpha_g().value(0)));
    EXPECT_TRUE(std::isfinite(precursor_model.concentration(0).value(0)));
}

/** @brief Verify configured multiphysics fields are published to VTU output. */
TEST(Phase13PlusSolverOutputTest, PublishesConfiguredFields)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    auto model_options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, boundary_conditions, time_options, {}, model_options);

    SimpleFluid::RadiolyticGasOptions radiolysis;
    radiolysis.mode = SimpleFluid::RadiolyticGasMode::IdealGasSource;
    radiolysis.hydrogen_yield_mol_per_j = 1.0e-7;
    radiolysis.max_source_alpha_rate = 1.0;
    solver.configure_radiolytic_gas(radiolysis);
    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = false;
    solver.configure_boiling_source(boiling);
    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.density_mode = SimpleFluid::DensityFeedbackMode::Mixture;
    solver.configure_material_feedback(feedback);
    SimpleFluid::DelayedNeutronPrecursorOptions precursors;
    precursors.group_count = 1;
    precursors.decay_constants = {0.1};
    solver.configure_precursors(precursors);

    const auto unique_id =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output =
        std::filesystem::temp_directory_path()
      / ("SimpleFluid_phase13_plus_" + std::to_string(unique_id)
         + ".vtu");
    solver.write_solution_vtu(
        output.string(),
        SimpleFluid::SolutionOutputOptions{
            .include_sources = true,
            .include_material_properties = true,
            .include_radiolytic_gas_fields = true,
            .include_precursor_fields = true});

    std::ifstream input(output);
    const std::string contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    EXPECT_NE(contents.find("Name=\"alpha_g\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"S_alpha_rad\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"S_alpha_boil\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"S_alpha_total\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"latentHeatSink\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"condensationLatentHeatRelease\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"rhoFeedback\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"muFeedback\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"C_1\""), std::string::npos);
    std::filesystem::remove(output);
}
