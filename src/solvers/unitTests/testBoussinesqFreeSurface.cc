/**
 * @file testBoussinesqFreeSurface.cc
 * @brief Focused Boussinesq integration tests for planar volume accounting.
 */

#include <gtest/gtest.h>

#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using Solver = SimpleFluid::BoussinesqSolver<Pack>;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SimpleFluid::SP<MeshType> make_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
}

SimpleFluid::TimeStepperOptions time_options()
{
    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.1;
    options.kinematic_viscosity = 0.0;
    options.thermal_diffusivity = 0.0;
    options.thermal_expansion = 0.0;
    options.gravity_x = 0.0;
    options.gravity_y = 0.0;
    options.gravity_z = 0.0;
    return options;
}

SimpleFluid::BoussinesqModelOptions material_options()
{
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = 1000.0;
    options.density = 1000.0;
    options.specific_heat_capacity = 1.0;
    options.dynamic_viscosity = 1.0e-3;
    options.thermal_conductivity = 0.0;
    return options;
}

SimpleFluid::FreeSurfaceOptions free_surface_options(double initial_volume = 0.5)
{
    SimpleFluid::FreeSurfaceOptions options;
    options.enabled = true;
    options.mode = SimpleFluid::FreeSurfaceMode::PlanarVolumeBudget;
    options.initial_liquid_volume = initial_volume;
    options.vessel.bottom_elevation = 0.0;
    options.vessel.top_elevation = 1.0;
    options.vessel.cross_section_area = 1.0;
    options.vessel.total_internal_volume = 1.0;
    options.headspace.mode = SimpleFluid::HeadspaceMode::Vented;
    options.headspace.ambient_pressure = 101325.0;
    options.headspace.initial_pressure = 101325.0;
    options.headspace.initial_temperature = 300.0;
    return options;
}

SimpleFluid::FreeSurfaceOptions cell_mass_free_surface_options(double initial_volume = 0.5)
{
    auto options = free_surface_options(initial_volume);
    options.liquid_mass.mode = SimpleFluid::LiquidVolumeMode::CellMassInventory;
    return options;
}

SimpleFluid::RadiolyticGasOptions sheng_options()
{
    SimpleFluid::RadiolyticGasOptions options;
    options.mode = SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    options.pressure_mode = SimpleFluid::RadiolyticPressureMode::Constant;
    options.hydrogen_yield_mol_per_j = 2.0e-7;
    options.max_source_alpha_rate = 1.0;
    options.henry_coefficient = 1.0e-5;
    options.surface_tension = 0.07;
    options.hydrogen_diffusivity = 1.0e-8;
    options.uranium_concentration_mol_per_m3 = 1000.0;
    options.hydrogen_yield_molecules_per_100_ev = 1.8;
    options.min_radius = 1.0e-12;
    options.max_radius = 1.0e-3;
    options.min_population = 1.0e-40;
    options.max_population = 1.0e40;
    options.reference_pressure = 1.0e5;
    return options;
}

std::string read_file(const std::filesystem::path& filename)
{
    std::ifstream input(filename, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

TEST(BoussinesqFreeSurfaceTest, DisabledConfigurationPreservesBaseline)
{
    Solver baseline(make_mesh(), {}, time_options());
    Solver configured(make_mesh(), {}, time_options());
    SimpleFluid::FreeSurfaceOptions disabled;
    EXPECT_EQ(configured.configure_free_surface(disabled), nullptr);
    EXPECT_EQ(configured.find_free_surface_model(), nullptr);
    EXPECT_EQ(configured.find_liquid_mass_inventory(), nullptr);

    baseline.initialize_heated_box(300.0, 300.0);
    configured.initialize_heated_box(300.0, 300.0);
    baseline.step();
    configured.step();

    EXPECT_DOUBLE_EQ(configured.temperature().value(0), baseline.temperature().value(0));
    EXPECT_DOUBLE_EQ(configured.pressure().value(0), baseline.pressure().value(0));
    const auto baseline_velocity = baseline.velocity().value(0);
    const auto configured_velocity = configured.velocity().value(0);
    EXPECT_DOUBLE_EQ(configured_velocity.x, baseline_velocity.x);
    EXPECT_DOUBLE_EQ(configured_velocity.y, baseline_velocity.y);
    EXPECT_DOUBLE_EQ(configured_velocity.z, baseline_velocity.z);
}

TEST(BoussinesqFreeSurfaceTest, DisabledFreeSurfacePreservesLegacyBoilingEnergyPath)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    void_options.initial_alpha = 0.1;
    void_options.alpha_collapse_time = 1.0;
    solver.configure_scalar_void_fraction(void_options);
    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& model = solver.configure_boiling_source(boiling);
    solver.initialize_heated_box(301.0, 301.0);

    solver.step();

    EXPECT_EQ(solver.find_free_surface_model(), nullptr);
    EXPECT_DOUBLE_EQ(model.condensed_liquid_mass_this_step(), 0.0);
    EXPECT_DOUBLE_EQ(model.global_submerged_steam_mass(), 0.0);
    EXPECT_DOUBLE_EQ(model.condensation_latent_heat_release().value(0), 0.0);
}

TEST(BoussinesqFreeSurfaceTest, ThermalExpansionUsesPureTemperatureDependentDensity)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqTemperatureOnly;
    feedback.reference_density = 1000.0;
    feedback.liquid_density = 1000.0;
    feedback.gas_density = 1.0;
    feedback.reference_temperature = 300.0;
    feedback.thermal_expansion = 1.0e-3;
    feedback.reference_dynamic_viscosity = 1.0e-3;
    solver.configure_material_feedback(feedback);
    solver.configure_free_surface(free_surface_options());
    solver.initialize_heated_box(300.0, 300.0);

    EXPECT_NEAR(solver.free_surface_diagnostics().clear_level, 0.5, 1.0e-13);
    EXPECT_NEAR(solver.liquid_mass_inventory().totalMass(), 500.0, 1.0e-12);

    solver.temperature().put_scalar(310.0);
    solver.step();

    constexpr double expected_density = 990.0;
    constexpr double expected_volume = 500.0 / expected_density;
    const auto diagnostics = solver.free_surface_diagnostics();
    EXPECT_NEAR(solver.rho_liquid().value(0), expected_density, 1.0e-12);
    EXPECT_NEAR(diagnostics.liquid_volume, expected_volume, 1.0e-12);
    EXPECT_NEAR(diagnostics.clear_level, expected_volume, 1.0e-12);
    EXPECT_NEAR(diagnostics.pool_level, expected_volume, 1.0e-12);
    EXPECT_NEAR(diagnostics.time, 0.1, 1.0e-14);
    EXPECT_NEAR(diagnostics.time_step, 0.1, 1.0e-14);
    EXPECT_NEAR(diagnostics.clear_level_rate, (expected_volume - 0.5) / 0.1, 1.0e-12);
    EXPECT_NEAR(solver.liquid_mass_inventory().diagnostics().mass_balance_residual, 0.0, 1.0e-13);
}

TEST(BoussinesqFreeSurfaceTest, InitialMassUsesMaterialUpdaterAtInitializedTemperature)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    int updates = 0;
    solver.set_material_updater(
        [&updates](const auto& context, auto& material)
        {
            ++updates;
            EXPECT_DOUBLE_EQ(context.temperature.value(0), 300.0);
            material.density.put_scalar(800.0);
        });
    solver.configure_free_surface(free_surface_options());
    solver.initialize_heated_box(300.0, 300.0);

    EXPECT_DOUBLE_EQ(solver.rho_liquid().value(0), 800.0);
    EXPECT_DOUBLE_EQ(solver.liquid_mass_inventory().totalMass(), 400.0);
    EXPECT_DOUBLE_EQ(solver.free_surface_diagnostics().liquid_volume, 0.5);
    EXPECT_EQ(updates, 1);
    EXPECT_THROW(solver.set_material_updater([](const auto&, auto&) {}), std::logic_error);
}

TEST(BoussinesqFreeSurfaceTest, ConfigurationAfterFieldInitializationRefreshesMaterialUpdater)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    int updates = 0;
    solver.set_material_updater(
        [&updates](const auto& context, auto& material)
        {
            ++updates;
            EXPECT_DOUBLE_EQ(context.temperature.value(0), 300.0);
            material.density.put_scalar(800.0);
        });
    solver.initialize_heated_box(300.0, 300.0);

    solver.configure_free_surface(free_surface_options());

    EXPECT_DOUBLE_EQ(solver.rho_liquid().value(0), 800.0);
    EXPECT_DOUBLE_EQ(solver.liquid_mass_inventory().totalMass(), 400.0);
    EXPECT_DOUBLE_EQ(solver.free_surface_diagnostics().liquid_volume, 0.5);
    EXPECT_EQ(updates, 1);
}

TEST(BoussinesqFreeSurfaceTest, LazyFirstStepInitializesFromOneMaterialUpdate)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    int updates = 0;
    solver.set_material_updater(
        [&updates](const auto&, auto& material)
        {
            ++updates;
            material.density.put_scalar(800.0);
        });
    solver.configure_free_surface(free_surface_options());
    EXPECT_EQ(updates, 0);

    solver.step();

    EXPECT_EQ(updates, 1);
    EXPECT_DOUBLE_EQ(solver.rho_liquid().value(0), 800.0);
    EXPECT_DOUBLE_EQ(solver.liquid_mass_inventory().totalMass(), 400.0);
}

TEST(BoussinesqFreeSurfaceTest, ExplicitInitializationOrdersMaterialBeforeShengReconstruction)
{
    auto radiolysis = sheng_options();
    radiolysis.initial_micro_number_density = 1.0e10;
    radiolysis.initial_micro_moles = 1.0e-6;

    auto reference_options = material_options();
    reference_options.reference_density = 800.0;
    reference_options.density = 800.0;
    reference_options.dynamic_viscosity = 1.5e-3;
    Solver reference(make_mesh(), {}, time_options(), {}, reference_options);
    auto& reference_gas = reference.configure_radiolytic_gas(radiolysis);
    reference.configure_free_surface(free_surface_options());
    reference.initialize_heated_box(300.0, 300.0);

    Solver updated(make_mesh(), {}, time_options(), {}, material_options());
    int updates = 0;
    updated.set_material_updater(
        [&updates](const auto&, auto& material)
        {
            ++updates;
            material.density.put_scalar(800.0);
            material.dynamic_viscosity.put_scalar(1.5e-3);
        });
    auto& updated_gas = updated.configure_radiolytic_gas(radiolysis);
    updated.configure_free_surface(free_surface_options());
    updated.initialize_heated_box(300.0, 300.0);

    const auto reference_transfer = reference_gas.output_fields().at("K_L")->value(0);
    const auto updated_transfer = updated_gas.output_fields().at("K_L")->value(0);
    ASSERT_GT(reference_transfer, 0.0);
    EXPECT_NEAR(updated_transfer, reference_transfer, reference_transfer * 1.0e-13);
    EXPECT_EQ(updates, 1);
    EXPECT_DOUBLE_EQ(updated.liquid_mass_inventory().totalMass(), 400.0);
}

TEST(BoussinesqFreeSurfaceTest, BoilingCompletionRemovesAcceptedLiquidMassExactlyOnce)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_min = 0.0;
    void_options.alpha_max = 0.9;
    void_options.initial_alpha = 0.0;
    solver.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver.configure_boiling_source(boiling);
    solver.configure_free_surface(free_surface_options());
    solver.initialize_heated_box(301.0, 301.0);
    const auto initial_mass = solver.liquid_mass_inventory().totalMass();

    solver.step();

    const auto& phase = boiling_model.last_phase_change_diagnostics();
    ASSERT_GT(phase.accepted_evaporation_mass, 0.0);
    EXPECT_FALSE(boiling_model.phase_change_completion_pending());
    const auto liquid = solver.liquid_mass_inventory().diagnostics();
    EXPECT_NEAR(
        liquid.total_mass, initial_mass - phase.accepted_evaporation_mass + phase.condensed_liquid_mass, 1.0e-12);
    EXPECT_NEAR(liquid.cumulative_evaporated_mass, phase.accepted_evaporation_mass, 1.0e-14);
    EXPECT_NEAR(liquid.cumulative_condensed_mass, phase.condensed_liquid_mass, 1.0e-14);
    const auto surface = solver.free_surface_diagnostics();
    EXPECT_NEAR(surface.submerged_bubble_volume, phase.submerged_steam_volume, 1.0e-14);
    EXPECT_NEAR(surface.pool_volume, surface.liquid_volume + phase.submerged_steam_volume, 1.0e-14);
    ASSERT_EQ(solver.free_surface_history().size(), 2U);
    ASSERT_TRUE(solver.free_surface_history().back().boiling.has_value());
    EXPECT_NEAR(solver.free_surface_history().back().boiling->accepted_evaporation_mass,
        phase.accepted_evaporation_mass, 1.0e-14);
}

TEST(BoussinesqFreeSurfaceTest, CellMassInventoryZeroFlowStepPreservesLocalAndGlobalMass)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    solver.configure_free_surface(cell_mass_free_surface_options());
    solver.initialize_heated_box(300.0, 300.0);
    const auto initial_cell_mass = solver.liquid_mass_inventory().cellMassInventory().value(0);
    const auto initial_mass = solver.liquid_mass_inventory().totalMass();

    solver.step();

    const auto diagnostics = solver.liquid_mass_inventory().diagnostics();
    EXPECT_EQ(solver.liquid_mass_inventory().mode(), SimpleFluid::LiquidVolumeMode::CellMassInventory);
    EXPECT_NEAR(solver.liquid_mass_inventory().cellMassInventory().value(0), initial_cell_mass, 1.0e-10);
    EXPECT_NEAR(diagnostics.total_mass, initial_mass, 1.0e-10);
    EXPECT_NEAR(diagnostics.liquid_volume, 0.5, 1.0e-12);
    EXPECT_NEAR(diagnostics.step_mass_balance_residual, 0.0, 1.0e-10);
    EXPECT_NEAR(diagnostics.mass_balance_residual, 0.0, 1.0e-10);
}

TEST(BoussinesqFreeSurfaceTest, CellMassInventoryConsumesAcceptedLocalBoilingSourceOnce)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    solver.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver.configure_boiling_source(boiling);
    solver.configure_free_surface(cell_mass_free_surface_options());
    solver.initialize_heated_box(301.0, 301.0);
    const auto initial_cell_mass = solver.liquid_mass_inventory().cellMassInventory().value(0);
    const auto initial_mass = solver.liquid_mass_inventory().totalMass();

    solver.step();

    const auto& phase = boiling_model.last_phase_change_diagnostics();
    const auto liquid = solver.liquid_mass_inventory().diagnostics();
    ASSERT_GT(phase.accepted_evaporation_mass, 0.0);
    EXPECT_NEAR(
        liquid.total_mass, initial_mass - phase.accepted_evaporation_mass + phase.condensed_liquid_mass, 1.0e-9);
    EXPECT_NEAR(solver.liquid_mass_inventory().cellMassInventory().value(0),
        initial_cell_mass - phase.accepted_evaporation_mass + phase.condensed_liquid_mass, 1.0e-9);
    EXPECT_NEAR(liquid.cumulative_evaporated_mass, phase.accepted_evaporation_mass, 1.0e-14);
    EXPECT_NEAR(liquid.cumulative_condensed_mass, phase.condensed_liquid_mass, 1.0e-14);
    EXPECT_NEAR(liquid.step_mass_balance_residual, 0.0, 1.0e-9);
}

TEST(BoussinesqFreeSurfaceTest, CellMassInventoryReturnsPositiveCondensateExactlyOnce)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    void_options.alpha_collapse_time = 0.5;
    solver.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver.configure_boiling_source(boiling);
    solver.configure_free_surface(cell_mass_free_surface_options());
    solver.initialize_heated_box(301.0, 301.0);
    const auto initial_liquid_mass = solver.liquid_mass_inventory().totalMass();

    solver.step();
    const auto first_phase = boiling_model.last_phase_change_diagnostics();
    const auto first_liquid = solver.liquid_mass_inventory().diagnostics();
    const auto first_cell_mass = solver.liquid_mass_inventory().cellMassInventory().value(0);
    ASSERT_GT(first_phase.accepted_evaporation_mass, 0.0);
    EXPECT_DOUBLE_EQ(first_phase.condensed_liquid_mass, 0.0);
    EXPECT_NEAR(first_phase.submerged_steam_mass, first_phase.accepted_evaporation_mass, 1.0e-14);
    EXPECT_NEAR(first_liquid.total_mass, initial_liquid_mass - first_phase.accepted_evaporation_mass, 1.0e-9);

    solver.temperature().put_scalar(299.0);
    solver.step();

    const auto second_phase = boiling_model.last_phase_change_diagnostics();
    const auto second_liquid = solver.liquid_mass_inventory().diagnostics();
    const auto cell_volume = solver.temperature().mesh().cell_volume(0);
    const auto returned_mass = boiling_model.condensation_mass_rate().value(0) * cell_volume * time_options().time_step;
    ASSERT_GT(second_phase.condensed_liquid_mass, 0.0);
    EXPECT_DOUBLE_EQ(second_phase.accepted_evaporation_mass, 0.0);
    EXPECT_NEAR(returned_mass, second_phase.condensed_liquid_mass, 1.0e-14);
    EXPECT_NEAR(boiling_model.condensation_latent_heat_release().value(0),
        boiling_model.condensation_mass_rate().value(0) * boiling.latent_heat, 1.0e-14);
    EXPECT_NEAR(second_liquid.total_mass, first_liquid.total_mass + second_phase.condensed_liquid_mass, 1.0e-9);
    EXPECT_NEAR(solver.liquid_mass_inventory().cellMassInventory().value(0),
        first_cell_mass + second_phase.condensed_liquid_mass / cell_volume, 1.0e-9);
    EXPECT_NEAR(second_liquid.cumulative_evaporated_mass, first_phase.accepted_evaporation_mass, 1.0e-14);
    EXPECT_NEAR(second_liquid.cumulative_condensed_mass, second_phase.condensed_liquid_mass, 1.0e-14);
    EXPECT_NEAR(second_liquid.cumulative_evaporated_mass, second_phase.cumulative_accepted_evaporation_mass, 1.0e-14);
    EXPECT_NEAR(second_liquid.cumulative_condensed_mass, second_phase.cumulative_condensed_liquid_mass, 1.0e-14);
    EXPECT_NEAR(second_phase.submerged_steam_mass,
        first_phase.submerged_steam_mass - second_phase.condensed_liquid_mass, 1.0e-14);
    EXPECT_NEAR(second_liquid.total_mass + second_phase.submerged_steam_mass, initial_liquid_mass, 1.0e-9);
    EXPECT_NEAR(second_phase.condensation_latent_energy_release,
        second_phase.condensed_liquid_mass * boiling.latent_heat, 1.0e-14);
    EXPECT_NEAR(second_liquid.step_mass_balance_residual, 0.0, 1.0e-9);
}

TEST(BoussinesqFreeSurfaceTest, RejectsRemovalWhileBoilingOwnsSubmergedSteam)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    solver.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver.configure_boiling_source(boiling);
    solver.configure_free_surface(free_surface_options());
    solver.initialize_heated_box(301.0, 301.0);
    solver.step();

    const auto steam_mass = boiling_model.global_submerged_steam_mass();
    ASSERT_GT(steam_mass, 0.0);
    try
    {
        static_cast<void>(solver.remove_free_surface_model());
        FAIL() << "Expected nonzero submerged steam to block free-surface removal.";
    }
    catch (const std::logic_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("nonzero submerged steam"), std::string::npos);
    }
    EXPECT_NE(solver.find_free_surface_model(), nullptr);
    EXPECT_NE(solver.find_liquid_mass_inventory(), nullptr);
    EXPECT_DOUBLE_EQ(boiling_model.global_submerged_steam_mass(), steam_mass);
}

TEST(BoussinesqFreeSurfaceTest, AllowsRemovalWhileBoilingSteamInventoryIsZero)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    auto& boiling_model = solver.configure_boiling_source(boiling);
    solver.configure_free_surface(free_surface_options());
    solver.initialize_heated_box(300.0, 300.0);

    ASSERT_DOUBLE_EQ(boiling_model.global_submerged_steam_mass(), 0.0);
    EXPECT_TRUE(solver.remove_free_surface_model());
    EXPECT_EQ(solver.find_free_surface_model(), nullptr);
    EXPECT_EQ(solver.find_liquid_mass_inventory(), nullptr);
}

TEST(BoussinesqFreeSurfaceTest, TransfersOnlyExactSubmergedBubbleEscapeToVent)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqVoid;
    feedback.reference_density = 1000.0;
    feedback.liquid_density = 1000.0;
    feedback.gas_density = 1.0;
    feedback.reference_temperature = 300.0;
    feedback.thermal_expansion = 0.0;
    feedback.reference_dynamic_viscosity = 1.0e-3;
    solver.configure_material_feedback(feedback);
    auto radiolysis = sheng_options();
    radiolysis.initial_micro_number_density = 1.0e10;
    radiolysis.initial_micro_moles = 1.0e-6;
    radiolysis.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    radiolysis.constant_slip_velocity = 10.0;
    radiolysis.free_surface_patches = {"zmax"};
    radiolysis.microbubble_lifetime = 1.0e30;
    radiolysis.large_bubble_dissolution_time = 1.0e30;
    radiolysis.micro_to_large_conversion_coefficient = 0.0;
    auto& gas = solver.configure_radiolytic_gas(radiolysis);
    solver.add_fission_power_source().initialize_constant(0.0);
    solver.configure_free_surface(free_surface_options(0.4));
    solver.initialize_heated_box(300.0, 300.0);
    const auto initial = solver.free_surface_diagnostics();
    ASSERT_GT(initial.submerged_bubble_volume, 0.0);
    ASSERT_TRUE(initial.submerged_population_gas_moles.contains("microbubble"));
    EXPECT_GT(initial.submerged_population_gas_moles.at("microbubble").at("H2"), 0.0);
    EXPECT_DOUBLE_EQ(solver.free_surface_history().front().microbubble_hydrogen_moles,
        initial.submerged_population_gas_moles.at("microbubble").at("H2"));
    EXPECT_GT(initial.pool_level, initial.clear_level);
    EXPECT_DOUBLE_EQ(solver.rho_liquid().value(0), 1000.0);
    EXPECT_LT(solver.material_properties().density.value(0), 1000.0);

    solver.step();

    const auto escaped = gas.last_statistics().submerged_bubble_hydrogen_escaped;
    ASSERT_GT(escaped, 0.0);
    const auto diagnostics = solver.free_surface_diagnostics();
    ASSERT_TRUE(diagnostics.vented_gas_moles.contains("H2"));
    EXPECT_NEAR(diagnostics.vented_gas_moles.at("H2"), escaped, 1.0e-14);
    EXPECT_NEAR(diagnostics.gas_closure_by_species.at("H2"), 0.0, 1.0e-14);
    EXPECT_LT(diagnostics.submerged_bubble_volume, initial.submerged_bubble_volume);
    EXPECT_DOUBLE_EQ(gas.absolute_pressure_offset(), 101325.0);
}

TEST(BoussinesqFreeSurfaceTest, PublishesOptInFieldsAndOccupancyApproximationError)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    solver.initialize_heated_box(300.0, 300.0);

    SimpleFluid::Database database;
    database.set("free_surface_enabled", true);
    database.set("free_surface_model", std::string("planarVolumeBudget"));
    database.set("free_surface_liquid_volume_model", std::string("cellMassInventory"));
    database.set("free_surface_initial_liquid_volume", 0.75);
    database.set("free_surface_vessel_model", std::string("constantArea"));
    database.set("free_surface_bottom_elevation", 0.0);
    database.set("free_surface_top_elevation", 1.0);
    database.set("free_surface_cross_section_area", 1.0);
    database.set("free_surface_total_internal_volume", 1.0);
    ASSERT_NE(solver.configure_free_surface(database), nullptr);

    EXPECT_DOUBLE_EQ(solver.clear_level().value(0), 0.75);
    EXPECT_DOUBLE_EQ(solver.pool_level().value(0), 0.75);
    EXPECT_DOUBLE_EQ(solver.headspace_pressure().value(0), 101325.0);
    EXPECT_DOUBLE_EQ(solver.pool_occupancy().value(0), 1.0);
    EXPECT_DOUBLE_EQ(solver.pool_occupancy_volume_error(), 0.25);

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto prefix = std::filesystem::temp_directory_path() / ("SimpleFluid_free_surface_" + std::to_string(unique));
    const auto default_file = prefix.string() + "_default.vtu";
    const auto selected_file = prefix.string() + "_selected.vtu";
    solver.write_solution_vtu(default_file);
    solver.write_solution_vtu(selected_file, SimpleFluid::SolutionOutputOptions{.include_free_surface_fields = true});

    const auto default_contents = read_file(default_file);
    const auto selected_contents = read_file(selected_file);
    EXPECT_EQ(solver.liquid_mass_inventory().mode(), SimpleFluid::LiquidVolumeMode::CellMassInventory);
    for (const auto* name :
        {"rhoLiquid", "liquidMassInventory", "clearLevel", "poolLevel", "headspacePressure", "poolOccupancy"})
    {
        const auto marker = std::string("Name=\"") + name + "\"";
        EXPECT_EQ(default_contents.find(marker), std::string::npos);
        EXPECT_NE(selected_contents.find(marker), std::string::npos);
    }
    std::filesystem::remove(default_file);
    std::filesystem::remove(selected_file);

    EXPECT_TRUE(solver.remove_free_surface_model());
    EXPECT_EQ(solver.find_free_surface_model(), nullptr);
    EXPECT_EQ(solver.find_liquid_mass_inventory(), nullptr);
    EXPECT_FALSE(solver.remove_free_surface_model());
}

TEST(BoussinesqFreeSurfaceTest, RecordsInitializationAndAcceptedStepHistoryWithFixedCsvSchema)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    solver.configure_free_surface(free_surface_options());
    solver.initialize_heated_box(300.0, 300.0);
    ASSERT_EQ(solver.free_surface_history().size(), 1U);
    EXPECT_DOUBLE_EQ(solver.free_surface_history().front().free_surface.time, 0.0);

    solver.step();
    ASSERT_EQ(solver.free_surface_history().size(), 2U);
    const auto& accepted = solver.free_surface_history().back();
    EXPECT_DOUBLE_EQ(accepted.free_surface.time, 0.1);
    EXPECT_DOUBLE_EQ(accepted.free_surface.time_step, 0.1);
    EXPECT_DOUBLE_EQ(accepted.liquid_mass.total_mass, solver.liquid_mass_inventory().totalMass());
    EXPECT_DOUBLE_EQ(accepted.pool_occupancy_volume_error, solver.pool_occupancy_volume_error());
    EXPECT_FALSE(accepted.boiling.has_value());

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto filename = std::filesystem::temp_directory_path() /
                          ("SimpleFluid_free_surface_history_" + std::to_string(unique) + ".csv");
    solver.write_free_surface_history_csv(filename.string());
    const auto contents = read_file(filename);
    EXPECT_EQ(contents.rfind("time_s,time_step_s,liquid_mass_kg,", 0), 0U);
    EXPECT_NE(contents.find("h2_generated_mol"), std::string::npos);
    EXPECT_NE(contents.find("h2_microbubble_submerged_mol"), std::string::npos);
    EXPECT_NE(contents.find("h2_escaped_this_step_mol"), std::string::npos);
    EXPECT_NE(contents.find("h2_other_sink_mol"), std::string::npos);
    EXPECT_NE(contents.find("volume_closure_residual_normalized"), std::string::npos);
    EXPECT_NE(contents.find("gas_closure_normalized"), std::string::npos);
    EXPECT_NE(contents.find("dryout_mass_deficit_kg"), std::string::npos);
    EXPECT_NE(contents.find("liquid_step_mass_residual_kg"), std::string::npos);
    EXPECT_NE(contents.find("liquid_step_mass_residual_normalized"), std::string::npos);
    EXPECT_NE(contents.find("configured_level_underflow_m"), std::string::npos);
    EXPECT_NE(contents.find("configured_level_overflow_m"), std::string::npos);
    EXPECT_NE(contents.find("boiling_condensation_latent_energy_release_j"), std::string::npos);
    EXPECT_NE(contents.find("boiling_latent_energy_residual_j"), std::string::npos);
    EXPECT_EQ(contents.find("steam_generated_mol"), std::string::npos);
    EXPECT_EQ(std::count(contents.begin(), contents.end(), '\n'), 3);
    const auto header_end = contents.find('\n');
    ASSERT_NE(header_end, std::string::npos);
    const auto first_row_end = contents.find('\n', header_end + 1);
    ASSERT_NE(first_row_end, std::string::npos);
    const auto column_separators = std::count(contents.begin(), contents.begin() + header_end, ',');
    EXPECT_EQ(std::count(contents.begin() + header_end + 1, contents.begin() + first_row_end, ','), column_separators);
    EXPECT_EQ(std::count(contents.begin() + first_row_end + 1, contents.end(), ','), column_separators);
    std::filesystem::remove(filename);
}

TEST(BoussinesqFreeSurfaceTest, RejectsUnsupportedOwnershipAndPressureModes)
{
    {
        Solver solver(make_mesh(), {}, time_options());
        EXPECT_THROW(solver.configure_free_surface(free_surface_options()), std::invalid_argument);
    }
    {
        Solver solver(make_mesh(), {}, time_options(), {}, material_options());
        SimpleFluid::RadiolyticGasOptions ideal;
        ideal.mode = SimpleFluid::RadiolyticGasMode::IdealGasSource;
        ideal.hydrogen_yield_mol_per_j = 2.0e-7;
        ideal.max_source_alpha_rate = 1.0;
        solver.configure_radiolytic_gas(ideal);
        EXPECT_THROW(solver.configure_free_surface(free_surface_options()), std::invalid_argument);
    }
    {
        Solver solver(make_mesh(), {}, time_options(), {}, material_options());
        SimpleFluid::ScalarVoidFractionOptions scalar_void;
        scalar_void.alpha_max = 0.9;
        scalar_void.initial_alpha = 0.1;
        solver.configure_scalar_void_fraction(scalar_void);
        EXPECT_THROW(solver.configure_free_surface(free_surface_options()), std::invalid_argument);
    }
    {
        Solver solver(make_mesh(), {}, time_options(), {}, material_options());
        auto ale = free_surface_options();
        ale.mode = SimpleFluid::FreeSurfaceMode::PlanarALE;
        EXPECT_THROW(solver.configure_free_surface(ale), std::invalid_argument);
    }
    {
        Solver solver(make_mesh(), {}, time_options(), {}, material_options());
        SimpleFluid::BoilingSourceOptions boiling;
        boiling.enable_bulk_boiling = true;
        solver.configure_boiling_source(boiling);
        auto closed = free_surface_options();
        closed.headspace.mode = SimpleFluid::HeadspaceMode::Closed;
        EXPECT_THROW(solver.configure_free_surface(closed), std::invalid_argument);
    }
    for (const auto mode :
        {SimpleFluid::RadiolyticPressureMode::PrescribedHistory, SimpleFluid::RadiolyticPressureMode::Inertial})
    {
        Solver solver(make_mesh(), {}, time_options(), {}, material_options());
        auto radiolysis = sheng_options();
        radiolysis.pressure_mode = mode;
        if (mode == SimpleFluid::RadiolyticPressureMode::PrescribedHistory)
        {
            radiolysis.pressure_history_times = {0.0, 1.0};
            radiolysis.pressure_history_values = {1.0e5, 1.0e5};
        }
        solver.configure_radiolytic_gas(radiolysis);
        EXPECT_THROW(solver.configure_free_surface(free_surface_options()), std::invalid_argument);
    }
}

TEST(BoussinesqFreeSurfaceTest, InitializedLedgerRequiresFreeSurfaceToBeReconfiguredLast)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    solver.configure_free_surface(free_surface_options());
    solver.initialize_heated_box(300.0, 300.0);

    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.reference_density = 1000.0;
    feedback.liquid_density = 1000.0;
    feedback.gas_density = 1.0;
    feedback.reference_temperature = 300.0;
    feedback.reference_dynamic_viscosity = 0.0;
    EXPECT_THROW(solver.configure_material_feedback(feedback), std::logic_error);

    EXPECT_TRUE(solver.remove_free_surface_model());
    EXPECT_NO_THROW(solver.configure_material_feedback(feedback));
}

TEST(BoussinesqFreeSurfaceTest, RejectsInconsistentExplicitInitialMassAndFillVolume)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    auto options = free_surface_options(0.5);
    options.liquid_mass.initial_liquid_mass = 400.0;
    solver.configure_free_surface(options);
    EXPECT_THROW(solver.initialize_heated_box(300.0, 300.0), std::invalid_argument);
    EXPECT_EQ(solver.find_free_surface_model(), nullptr);
}

TEST(BoussinesqFreeSurfaceTest, FailedClosureDoesNotCommitLiquidPhaseChange)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_min = 0.0;
    void_options.alpha_max = 0.9;
    void_options.initial_alpha = 0.0;
    solver.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver.configure_boiling_source(boiling);
    solver.configure_free_surface(free_surface_options(0.95));
    solver.initialize_heated_box(301.0, 301.0);

    const auto initial_liquid = solver.liquid_mass_inventory().diagnostics();
    const auto initial_surface = solver.free_surface_diagnostics();
    EXPECT_THROW(solver.step(), std::out_of_range);
    ASSERT_GT(boiling_model.accepted_evaporation_mass_this_step(), 0.0);

    const auto liquid = solver.liquid_mass_inventory().diagnostics();
    EXPECT_DOUBLE_EQ(liquid.total_mass, initial_liquid.total_mass);
    EXPECT_DOUBLE_EQ(liquid.cumulative_evaporated_mass, initial_liquid.cumulative_evaporated_mass);
    EXPECT_DOUBLE_EQ(liquid.cumulative_condensed_mass, initial_liquid.cumulative_condensed_mass);
    const auto surface = solver.free_surface_diagnostics();
    EXPECT_DOUBLE_EQ(surface.time, initial_surface.time);
    EXPECT_DOUBLE_EQ(surface.liquid_volume, initial_surface.liquid_volume);
    EXPECT_DOUBLE_EQ(surface.pool_volume, initial_surface.pool_volume);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);
    EXPECT_THROW(solver.step(), std::logic_error);
}

TEST(BoussinesqFreeSurfaceTest, ClosedReconstructedPressureUsesCollectiveValidLowerBound)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    Solver solver(mesh, {}, time_options(), {}, material_options());
    auto radiolysis = sheng_options();
    radiolysis.pressure_mode = SimpleFluid::RadiolyticPressureMode::Reconstructed;
    radiolysis.minimum_absolute_pressure = 90000.0;
    radiolysis.initial_micro_number_density = 1.0e10;
    radiolysis.initial_micro_moles = 1.0e-6;
    auto& gas = solver.configure_radiolytic_gas(radiolysis);
    solver.add_fission_power_source().initialize_constant(0.0);
    solver.initialize_heated_box(300.0, 300.0);

    solver.pressure().set_owned_value(0, -1000.0);
    solver.pressure().set_owned_value(1, 1000.0);
    solver.pressure().sync_ghosts();
    gas.initialize_state(
        solver.time(), solver.temperature(), solver.pressure(), solver.velocity(), solver.material_properties(), true);
    EXPECT_NEAR(gas.minimum_valid_absolute_pressure_offset(), 91000.0, 1.0e-10);

    auto closed = free_surface_options(0.4);
    closed.headspace.mode = SimpleFluid::HeadspaceMode::Closed;
    closed.coupling.maximum_absolute_pressure = 2.0e5;
    ASSERT_NE(solver.configure_free_surface(closed), nullptr);
    const auto diagnostics = solver.free_surface_diagnostics();
    EXPECT_NEAR(diagnostics.headspace.pressure, closed.headspace.initial_pressure, 1.0e-5);
    EXPECT_GT(diagnostics.pool_level, diagnostics.clear_level);
}

TEST(BoussinesqFreeSurfaceTest, FailedStepRetainsEscapeButRejectsUnsafeFullStepRetry)
{
    Solver solver(make_mesh(), {}, time_options(), {}, material_options());
    auto radiolysis = sheng_options();
    radiolysis.initial_micro_number_density = 1.0e10;
    radiolysis.initial_micro_moles = 1.0e-6;
    radiolysis.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    radiolysis.constant_slip_velocity = 10.0;
    radiolysis.free_surface_patches = {"zmax"};
    radiolysis.microbubble_lifetime = 1.0e30;
    radiolysis.large_bubble_dissolution_time = 1.0e30;
    radiolysis.micro_to_large_conversion_coefficient = 0.0;
    auto& gas = solver.configure_radiolytic_gas(radiolysis);
    solver.add_fission_power_source().initialize_constant(0.0);

    auto surface_options = free_surface_options(0.4);
    surface_options.headspace.total_internal_volume = surface_options.vessel.total_internal_volume;
    surface_options.headspace.temperature_mode = SimpleFluid::HeadspaceTemperatureMode::Prescribed;
    surface_options.headspace.prescribed_temperature_times = {0.0, 0.05};
    surface_options.headspace.prescribed_temperature_values = {300.0, 300.0};
    solver.configure_free_surface(surface_options);
    solver.initialize_heated_box(300.0, 300.0);
    ASSERT_EQ(solver.free_surface_history().size(), 1U);

    EXPECT_THROW(solver.step(), std::out_of_range);
    const auto uncommitted_escape = gas.cumulative_submerged_bubble_hydrogen_escaped();
    ASSERT_GT(uncommitted_escape, 0.0);
    EXPECT_TRUE(solver.free_surface_diagnostics().vented_gas_moles.empty());
    EXPECT_EQ(solver.free_surface_history().size(), 1U);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);

    solver.set_time_step(0.05);
    EXPECT_THROW(solver.step(), std::logic_error);
    EXPECT_THROW(solver.configure_free_surface(surface_options), std::logic_error);
    EXPECT_EQ(solver.free_surface_history().size(), 1U);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);
}

TEST(BoussinesqFreeSurfaceTest, CellMassInventoryStepIsPartitionIndependent)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }
    Solver solver(mesh, {}, time_options(), {}, material_options());
    solver.configure_free_surface(cell_mass_free_surface_options());
    solver.initialize_heated_box(300.0, 300.0);

    solver.step();

    const auto diagnostics = solver.liquid_mass_inventory().diagnostics();
    EXPECT_NEAR(diagnostics.total_mass, 500.0, 1.0e-8);
    EXPECT_NEAR(diagnostics.liquid_volume, 0.5, 1.0e-10);
    EXPECT_NEAR(diagnostics.step_mass_balance_residual, 0.0, 1.0e-8);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        EXPECT_NEAR(
            solver.liquid_mass_inventory().cellMassInventory().value(static_cast<Pack::local_ordinal_type>(owned)),
            125.0, 1.0e-8);
    }
    double minimum_mass{};
    double maximum_mass{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &diagnostics.total_mass, &minimum_mass);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &diagnostics.total_mass, &maximum_mass);
    EXPECT_DOUBLE_EQ(minimum_mass, maximum_mass);
}

TEST(BoussinesqFreeSurfaceTest, CollectivelyRejectsRankDivergentOptions)
{
    // Give every rank owned work so an early rank-local empty-mesh failure
    // cannot mask the collective option-parity contract under test.
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(2, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }
    Solver solver(mesh, {}, time_options(), {}, material_options());
    auto options = free_surface_options();
    if (communicator->getRank() != 0)
    {
        options.vessel.cross_section_area = 2.0;
    }
    EXPECT_THROW(solver.configure_free_surface(options), std::invalid_argument);
    EXPECT_EQ(solver.find_free_surface_model(), nullptr);
}

TEST(BoussinesqFreeSurfaceTest, CollectivelyRejectsRemovalWhileBoilingOwnsSubmergedSteam)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(2, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }

    Solver solver(mesh, {}, time_options(), {}, material_options());
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    solver.configure_scalar_void_fraction(void_options);
    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver.configure_boiling_source(boiling);
    solver.configure_free_surface(free_surface_options());
    solver.initialize_heated_box(301.0, 301.0);
    solver.step();

    ASSERT_GT(boiling_model.global_submerged_steam_mass(), 0.0);
    try
    {
        static_cast<void>(solver.remove_free_surface_model());
        FAIL() << "Expected nonzero submerged steam to block free-surface removal.";
    }
    catch (const std::logic_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("nonzero submerged steam"), std::string::npos);
    }
    EXPECT_NE(solver.find_free_surface_model(), nullptr);
    EXPECT_NE(solver.find_liquid_mass_inventory(), nullptr);
}

TEST(BoussinesqFreeSurfaceTest, CollectivelyRejectsRankDivergentExistingStateDuringConfiguration)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(2, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }

    Solver solver(mesh, {}, time_options(), {}, material_options());
    ASSERT_NE(solver.configure_free_surface(free_surface_options()), nullptr);
    if (communicator->getRank() != 0)
    {
        EXPECT_TRUE(solver.remove_free_surface_model());
    }
    EXPECT_THROW(solver.configure_free_surface(free_surface_options()), std::invalid_argument);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);
}

TEST(BoussinesqFreeSurfaceTest, CollectivelyRejectsRankDivergentPresenceBeforeStep)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(2, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }

    Solver solver(mesh, {}, time_options(), {}, material_options());
    ASSERT_NE(solver.configure_free_surface(free_surface_options()), nullptr);
    if (communicator->getRank() != 0)
    {
        EXPECT_TRUE(solver.remove_free_surface_model());
    }

    EXPECT_THROW(solver.step(), std::invalid_argument);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);
}

TEST(BoussinesqFreeSurfaceTest, CollectivelyRejectsPositiveRankDivergentTimestepBeforeStep)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(2, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }

    Solver solver(mesh, {}, time_options(), {}, material_options());
    ASSERT_NE(solver.configure_free_surface(free_surface_options()), nullptr);
    solver.set_time_step(communicator->getRank() == 0 ? 0.25 : 0.5);

    EXPECT_THROW(solver.step(), std::invalid_argument);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);
}

} // namespace
