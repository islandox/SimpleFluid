/**
 * @file testBoussinesqFreeSurface.cc
 * @brief Focused Boussinesq integration tests for planar volume accounting.
 */

#include <gtest/gtest.h>
#include <memory>

#include "SurfaceGasFixtureSupport.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/ClosedMixtureHeadspaceModel.hh"
#include "solvers/MultiphysicsSolverCoupler.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <array>
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
namespace SurfaceGas = SimpleFluid::Testing::SurfaceGas;

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

SimpleFluid::ReactingGasOptions finite_gas_options(double liquid_fraction = .5, double density = 1000,
    double micro_count_density = 0, double micro_molar_density = 0, double minimum_pressure = 10000)
{
    SurfaceGas::Initial initial;
    initial.liquid_fraction = liquid_fraction;
    initial.liquid_density = density;
    initial.expansion = 0;
    initial.compressibility = 0;
    initial.dissolved_concentration = 0;
    initial.micro_number_density = micro_count_density;
    initial.micro_molar_density = micro_molar_density;
    initial.minimum_pressure = minimum_pressure;
    auto options = SurfaceGas::options(initial);
    options.species_source = {};
    options.population_process = {};
    options.reference_pressure = 100000;
    return options;
}

SimpleFluid::FreeSurfaceOptions typed_surface_options(double initial_volume = .5)
{
    auto options = cell_mass_free_surface_options(initial_volume);
    options.liquid_mass.depletion_policy = SimpleFluid::FreeSurfaceRangePolicy::Error;
    return options;
}

std::string read_file(const std::filesystem::path& filename)
{
    std::ifstream input(filename, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

TEST(BoussinesqFreeSurfaceTest, DisabledConfigurationPreservesBaseline)
{
    auto baseline_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options());
    auto& baseline = *baseline_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> baseline_coupling(baseline_shared);
    auto configured_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options());
    auto& configured = *configured_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> configured_coupling(configured_shared);
    SimpleFluid::FreeSurfaceOptions disabled;
    EXPECT_EQ(configured_coupling.configure_free_surface(disabled), nullptr);
    EXPECT_EQ(configured_coupling.find_free_surface_model(), nullptr);
    EXPECT_EQ(configured_coupling.find_liquid_mass_inventory(), nullptr);

    baseline_coupling.initialize_heated_box(300.0, 300.0);
    configured_coupling.initialize_heated_box(300.0, 300.0);
    baseline_coupling.step();
    configured_coupling.step();

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
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    void_options.initial_alpha = 0.1;
    void_options.alpha_collapse_time = 1.0;
    solver_coupling.configure_scalar_void_fraction(void_options);
    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& model = solver_coupling.configure_boiling_source(boiling);
    solver_coupling.initialize_heated_box(301.0, 301.0);

    solver_coupling.step();

    EXPECT_EQ(solver_coupling.find_free_surface_model(), nullptr);
    EXPECT_DOUBLE_EQ(model.condensed_liquid_mass_this_step(), 0.0);
    EXPECT_DOUBLE_EQ(model.global_submerged_steam_mass(), 0.0);
    EXPECT_DOUBLE_EQ(model.condensation_latent_heat_release().value(0), 0.0);
}

TEST(BoussinesqFreeSurfaceTest, ThermalExpansionUsesPureTemperatureDependentDensity)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqTemperatureOnly;
    feedback.reference_density = 1000.0;
    feedback.liquid_density = 1000.0;
    feedback.gas_density = 1.0;
    feedback.reference_temperature = 300.0;
    feedback.thermal_expansion = 1.0e-3;
    feedback.reference_dynamic_viscosity = 1.0e-3;
    solver_coupling.configure_material_feedback(feedback);
    solver_coupling.configure_free_surface(free_surface_options());
    solver_coupling.initialize_heated_box(300.0, 300.0);

    EXPECT_NEAR(solver_coupling.free_surface_diagnostics().clear_level, 0.5, 1.0e-13);
    EXPECT_NEAR(solver_coupling.liquid_mass_inventory().totalMass(), 500.0, 1.0e-12);

    solver.temperature().put_scalar(310.0);
    solver_coupling.step();

    constexpr double expected_density = 990.0;
    constexpr double expected_volume = 500.0 / expected_density;
    const auto diagnostics = solver_coupling.free_surface_diagnostics();
    EXPECT_NEAR(solver_coupling.rho_liquid().value(0), expected_density, 1.0e-12);
    EXPECT_NEAR(diagnostics.liquid_volume, expected_volume, 1.0e-12);
    EXPECT_NEAR(diagnostics.clear_level, expected_volume, 1.0e-12);
    EXPECT_NEAR(diagnostics.pool_level, expected_volume, 1.0e-12);
    EXPECT_NEAR(diagnostics.time, 0.1, 1.0e-14);
    EXPECT_NEAR(diagnostics.time_step, 0.1, 1.0e-14);
    EXPECT_NEAR(diagnostics.clear_level_rate, (expected_volume - 0.5) / 0.1, 1.0e-12);
    EXPECT_NEAR(solver_coupling.liquid_mass_inventory().diagnostics().mass_balance_residual, 0.0, 1.0e-13);
}

TEST(BoussinesqFreeSurfaceTest, InitialMassUsesMaterialUpdaterAtInitializedTemperature)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    int updates = 0;
    solver_coupling.set_material_updater(
        [&updates](const auto& context, auto& material)
        {
            ++updates;
            EXPECT_DOUBLE_EQ(context.temperature.value(0), 300.0);
            material.density.put_scalar(800.0);
        });
    solver_coupling.configure_free_surface(free_surface_options());
    solver_coupling.initialize_heated_box(300.0, 300.0);

    EXPECT_DOUBLE_EQ(solver_coupling.rho_liquid().value(0), 800.0);
    EXPECT_DOUBLE_EQ(solver_coupling.liquid_mass_inventory().totalMass(), 400.0);
    EXPECT_DOUBLE_EQ(solver_coupling.free_surface_diagnostics().liquid_volume, 0.5);
    EXPECT_EQ(updates, 1);
    EXPECT_THROW(solver_coupling.set_material_updater([](const auto&, auto&) {}), std::logic_error);
}

TEST(BoussinesqFreeSurfaceTest, ConfigurationAfterFieldInitializationRefreshesMaterialUpdater)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    int updates = 0;
    solver_coupling.set_material_updater(
        [&updates](const auto& context, auto& material)
        {
            ++updates;
            EXPECT_DOUBLE_EQ(context.temperature.value(0), 300.0);
            material.density.put_scalar(800.0);
        });
    solver_coupling.initialize_heated_box(300.0, 300.0);

    solver_coupling.configure_free_surface(free_surface_options());

    EXPECT_DOUBLE_EQ(solver_coupling.rho_liquid().value(0), 800.0);
    EXPECT_DOUBLE_EQ(solver_coupling.liquid_mass_inventory().totalMass(), 400.0);
    EXPECT_DOUBLE_EQ(solver_coupling.free_surface_diagnostics().liquid_volume, 0.5);
    EXPECT_EQ(updates, 1);
}

TEST(BoussinesqFreeSurfaceTest, LazyFirstStepInitializesFromOneMaterialUpdate)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    int updates = 0;
    solver_coupling.set_material_updater(
        [&updates](const auto&, auto& material)
        {
            ++updates;
            material.density.put_scalar(800.0);
        });
    solver_coupling.configure_free_surface(free_surface_options());
    EXPECT_EQ(updates, 0);

    solver_coupling.step();

    EXPECT_EQ(updates, 1);
    EXPECT_DOUBLE_EQ(solver_coupling.rho_liquid().value(0), 800.0);
    EXPECT_DOUBLE_EQ(solver_coupling.liquid_mass_inventory().totalMass(), 400.0);
}

TEST(BoussinesqFreeSurfaceTest, ExplicitInitializationOrdersMaterialBeforeTypedFilmTransfer)
{
    SurfaceGas::Initial initial;
    initial.liquid_fraction = .5;
    initial.liquid_density = 800;
    initial.expansion = 0;
    initial.compressibility = 0;
    initial.micro_number_density = 1e10;
    initial.micro_molar_density = 1e-6;
    auto reacting_gas = SurfaceGas::film_options(initial);

    auto reference_options = material_options();
    reference_options.reference_density = 800.0;
    reference_options.density = 800.0;
    reference_options.dynamic_viscosity = 1.5e-3;
    auto reference_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, reference_options);
    auto& reference = *reference_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> reference_coupling(reference_shared);
    auto& reference_gas = reference_coupling.configure_reacting_gas(reacting_gas);
    reference_coupling.configure_free_surface(typed_surface_options());
    reference_coupling.initialize_heated_box(300.0, 300.0);

    auto updated_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& updated = *updated_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> updated_coupling(updated_shared);
    int updates = 0;
    updated_coupling.set_material_updater(
        [&updates](const auto&, auto& material)
        {
            ++updates;
            material.density.put_scalar(800.0);
            material.dynamic_viscosity.put_scalar(1.5e-3);
        });
    auto& updated_gas = updated_coupling.configure_reacting_gas(reacting_gas);
    updated_coupling.configure_free_surface(typed_surface_options());
    updated_coupling.initialize_heated_box(300.0, 300.0);

    EXPECT_EQ(updates, 1);
    EXPECT_DOUBLE_EQ(updated_coupling.liquid_mass_inventory().totalMass(), 400);
    reference_coupling.step();
    updated_coupling.step();
    ASSERT_EQ(reference_gas.population_ledgers().size(), 1u);
    ASSERT_EQ(updated_gas.population_ledgers().size(), 1u);
    for (const auto [species, rate] : std::array<std::pair<SimpleFluid::Reacting::SpeciesId, double>, 2>{
             {{SurfaceGas::gas_id, 1e-7}, {SurfaceGas::M::gas_two, 2e-7}}})
    {
        const auto& catalog = reacting_gas.reaction_mechanism->catalog();
        const auto reference_transfer = -reference_gas.population_ledgers().front().chemicalChange.amount(
            catalog, species, SurfaceGas::Phase::Dispersed, SurfaceGas::micro);
        const auto updated_transfer = -updated_gas.population_ledgers().front().chemicalChange.amount(
            catalog, species, SurfaceGas::Phase::Dispersed, SurfaceGas::micro);
        ASSERT_GT(reference_transfer, 0.0);
        EXPECT_NEAR(reference_transfer, .1 * rate, .1 * rate * 1e-13);
        EXPECT_NEAR(updated_transfer, reference_transfer, reference_transfer * 1e-13);
    }
    EXPECT_NEAR(updated_coupling.liquid_mass_inventory().totalMass(), 400 + .1 * (.002 * 1e-7 + .004 * 2e-7), 1e-12);
}

TEST(BoussinesqFreeSurfaceTest, BoilingCompletionRemovesAcceptedLiquidMassExactlyOnce)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_min = 0.0;
    void_options.alpha_max = 0.9;
    void_options.initial_alpha = 0.0;
    solver_coupling.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver_coupling.configure_boiling_source(boiling);
    solver_coupling.configure_free_surface(free_surface_options());
    solver_coupling.initialize_heated_box(301.0, 301.0);
    const auto initial_mass = solver_coupling.liquid_mass_inventory().totalMass();

    solver_coupling.step();

    const auto& phase = boiling_model.last_phase_change_diagnostics();
    ASSERT_GT(phase.accepted_evaporation_mass, 0.0);
    EXPECT_FALSE(boiling_model.phase_change_completion_pending());
    const auto liquid = solver_coupling.liquid_mass_inventory().diagnostics();
    EXPECT_NEAR(
        liquid.total_mass, initial_mass - phase.accepted_evaporation_mass + phase.condensed_liquid_mass, 1.0e-12);
    EXPECT_NEAR(liquid.cumulative_evaporated_mass, phase.accepted_evaporation_mass, 1.0e-14);
    EXPECT_NEAR(liquid.cumulative_condensed_mass, phase.condensed_liquid_mass, 1.0e-14);
    const auto surface = solver_coupling.free_surface_diagnostics();
    EXPECT_NEAR(surface.submerged_bubble_volume, phase.submerged_steam_volume, 1.0e-14);
    EXPECT_NEAR(surface.pool_volume, surface.liquid_volume + phase.submerged_steam_volume, 1.0e-14);
    ASSERT_EQ(solver_coupling.free_surface_history().size(), 2U);
    ASSERT_TRUE(solver_coupling.free_surface_history().back().boiling.has_value());
    EXPECT_NEAR(solver_coupling.free_surface_history().back().boiling->accepted_evaporation_mass,
        phase.accepted_evaporation_mass, 1.0e-14);
}

TEST(BoussinesqFreeSurfaceTest, CellMassInventoryZeroFlowStepPreservesLocalAndGlobalMass)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    solver_coupling.configure_free_surface(cell_mass_free_surface_options());
    solver_coupling.initialize_heated_box(300.0, 300.0);
    const auto initial_cell_mass = solver_coupling.liquid_mass_inventory().cellMassInventory().value(0);
    const auto initial_mass = solver_coupling.liquid_mass_inventory().totalMass();

    solver_coupling.step();

    const auto diagnostics = solver_coupling.liquid_mass_inventory().diagnostics();
    EXPECT_EQ(solver_coupling.liquid_mass_inventory().mode(), SimpleFluid::LiquidVolumeMode::CellMassInventory);
    EXPECT_NEAR(solver_coupling.liquid_mass_inventory().cellMassInventory().value(0), initial_cell_mass, 1.0e-10);
    EXPECT_NEAR(diagnostics.total_mass, initial_mass, 1.0e-10);
    EXPECT_NEAR(diagnostics.liquid_volume, 0.5, 1.0e-12);
    EXPECT_NEAR(diagnostics.step_mass_balance_residual, 0.0, 1.0e-10);
    EXPECT_NEAR(diagnostics.mass_balance_residual, 0.0, 1.0e-10);
}

TEST(BoussinesqFreeSurfaceTest, CellMassInventoryConsumesAcceptedLocalBoilingSourceOnce)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    solver_coupling.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver_coupling.configure_boiling_source(boiling);
    solver_coupling.configure_free_surface(cell_mass_free_surface_options());
    solver_coupling.initialize_heated_box(301.0, 301.0);
    const auto initial_cell_mass = solver_coupling.liquid_mass_inventory().cellMassInventory().value(0);
    const auto initial_mass = solver_coupling.liquid_mass_inventory().totalMass();

    solver_coupling.step();

    const auto& phase = boiling_model.last_phase_change_diagnostics();
    const auto liquid = solver_coupling.liquid_mass_inventory().diagnostics();
    ASSERT_GT(phase.accepted_evaporation_mass, 0.0);
    EXPECT_NEAR(
        liquid.total_mass, initial_mass - phase.accepted_evaporation_mass + phase.condensed_liquid_mass, 1.0e-9);
    EXPECT_NEAR(solver_coupling.liquid_mass_inventory().cellMassInventory().value(0),
        initial_cell_mass - phase.accepted_evaporation_mass + phase.condensed_liquid_mass, 1.0e-9);
    EXPECT_NEAR(liquid.cumulative_evaporated_mass, phase.accepted_evaporation_mass, 1.0e-14);
    EXPECT_NEAR(liquid.cumulative_condensed_mass, phase.condensed_liquid_mass, 1.0e-14);
    EXPECT_NEAR(liquid.step_mass_balance_residual, 0.0, 1.0e-9);
}

TEST(BoussinesqFreeSurfaceTest, CellMassInventoryReturnsPositiveCondensateExactlyOnce)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    void_options.alpha_collapse_time = 0.5;
    solver_coupling.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver_coupling.configure_boiling_source(boiling);
    solver_coupling.configure_free_surface(cell_mass_free_surface_options());
    solver_coupling.initialize_heated_box(301.0, 301.0);
    const auto initial_liquid_mass = solver_coupling.liquid_mass_inventory().totalMass();

    solver_coupling.step();
    const auto first_phase = boiling_model.last_phase_change_diagnostics();
    const auto first_liquid = solver_coupling.liquid_mass_inventory().diagnostics();
    const auto first_cell_mass = solver_coupling.liquid_mass_inventory().cellMassInventory().value(0);
    ASSERT_GT(first_phase.accepted_evaporation_mass, 0.0);
    EXPECT_DOUBLE_EQ(first_phase.condensed_liquid_mass, 0.0);
    EXPECT_NEAR(first_phase.submerged_steam_mass, first_phase.accepted_evaporation_mass, 1.0e-14);
    EXPECT_NEAR(first_liquid.total_mass, initial_liquid_mass - first_phase.accepted_evaporation_mass, 1.0e-9);

    solver.temperature().put_scalar(299.0);
    solver_coupling.step();

    const auto second_phase = boiling_model.last_phase_change_diagnostics();
    const auto second_liquid = solver_coupling.liquid_mass_inventory().diagnostics();
    const auto cell_volume = solver.temperature().mesh().cell_volume(0);
    const auto returned_mass = boiling_model.condensation_mass_rate().value(0) * cell_volume * time_options().time_step;
    ASSERT_GT(second_phase.condensed_liquid_mass, 0.0);
    EXPECT_DOUBLE_EQ(second_phase.accepted_evaporation_mass, 0.0);
    EXPECT_NEAR(returned_mass, second_phase.condensed_liquid_mass, 1.0e-14);
    EXPECT_NEAR(boiling_model.condensation_latent_heat_release().value(0),
        boiling_model.condensation_mass_rate().value(0) * boiling.latent_heat, 1.0e-14);
    EXPECT_NEAR(second_liquid.total_mass, first_liquid.total_mass + second_phase.condensed_liquid_mass, 1.0e-9);
    EXPECT_NEAR(solver_coupling.liquid_mass_inventory().cellMassInventory().value(0),
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
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    solver_coupling.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver_coupling.configure_boiling_source(boiling);
    solver_coupling.configure_free_surface(free_surface_options());
    solver_coupling.initialize_heated_box(301.0, 301.0);
    solver_coupling.step();

    const auto steam_mass = boiling_model.global_submerged_steam_mass();
    ASSERT_GT(steam_mass, 0.0);
    try
    {
        static_cast<void>(solver_coupling.remove_free_surface_model());
        FAIL() << "Expected nonzero submerged steam to block free-surface removal.";
    }
    catch (const std::logic_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("nonzero submerged steam"), std::string::npos);
    }
    EXPECT_NE(solver_coupling.find_free_surface_model(), nullptr);
    EXPECT_NE(solver_coupling.find_liquid_mass_inventory(), nullptr);
    EXPECT_DOUBLE_EQ(boiling_model.global_submerged_steam_mass(), steam_mass);
}

TEST(BoussinesqFreeSurfaceTest, AllowsRemovalWhileBoilingSteamInventoryIsZero)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    auto& boiling_model = solver_coupling.configure_boiling_source(boiling);
    solver_coupling.configure_free_surface(free_surface_options());
    solver_coupling.initialize_heated_box(300.0, 300.0);

    ASSERT_DOUBLE_EQ(boiling_model.global_submerged_steam_mass(), 0.0);
    EXPECT_TRUE(solver_coupling.remove_free_surface_model());
    EXPECT_EQ(solver_coupling.find_free_surface_model(), nullptr);
    EXPECT_EQ(solver_coupling.find_liquid_mass_inventory(), nullptr);
}

TEST(BoussinesqFreeSurfaceTest, TransfersOnlyExactSubmergedBubbleEscapeToVent)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.density_mode = SimpleFluid::DensityFeedbackMode::BoussinesqVoid;
    feedback.reference_density = 1000.0;
    feedback.liquid_density = 1000.0;
    feedback.gas_density = 1.0;
    feedback.reference_temperature = 300.0;
    feedback.thermal_expansion = 0.0;
    feedback.reference_dynamic_viscosity = 1.0e-3;
    solver_coupling.configure_material_feedback(feedback);
    auto reacting_gas = finite_gas_options(.4, 1000, 1e10, 1e-6);
    reacting_gas.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    reacting_gas.constant_slip_velocity = 10.0;
    reacting_gas.free_surface_patches = {"zmax"};
    auto& gas = solver_coupling.configure_reacting_gas(reacting_gas);
    solver_coupling.add_prescribed_heat_source().initialize_constant(0.0);
    solver_coupling.configure_free_surface(typed_surface_options(0.4));
    solver_coupling.initialize_heated_box(300.0, 300.0);
    const auto initial = solver_coupling.free_surface_diagnostics();
    ASSERT_GT(initial.submerged_bubble_volume, 0.0);
    ASSERT_TRUE(initial.submerged_population_gas_moles.contains(SurfaceGas::micro));
    EXPECT_GT(initial.submerged_population_gas_moles.at(SurfaceGas::micro).at(SurfaceGas::gas_name), 0.0);
    EXPECT_DOUBLE_EQ(solver_coupling.free_surface_history()
                         .front()
                         .free_surface.submerged_population_gas_moles.at(SurfaceGas::micro)
                         .at(SurfaceGas::gas_name),
        initial.submerged_population_gas_moles.at(SurfaceGas::micro).at(SurfaceGas::gas_name));
    EXPECT_GT(initial.pool_level, initial.clear_level);
    EXPECT_DOUBLE_EQ(solver_coupling.rho_liquid().value(0), 1000.0);
    EXPECT_LT(solver.material_properties().density.value(0), 1000.0);

    solver_coupling.step();

    const auto escaped = SurfaceGas::step_escaped(gas);
    ASSERT_GT(escaped, 0.0);
    const auto diagnostics = solver_coupling.free_surface_diagnostics();
    ASSERT_TRUE(diagnostics.vented_gas_moles.contains(SurfaceGas::gas_name));
    EXPECT_NEAR(diagnostics.vented_gas_moles.at(SurfaceGas::gas_name), escaped, 1.0e-14);
    EXPECT_NEAR(diagnostics.gas_closure_by_species.at(SurfaceGas::gas_name), 0.0, 1.0e-14);
    EXPECT_LT(diagnostics.submerged_bubble_volume, initial.submerged_bubble_volume);
    EXPECT_DOUBLE_EQ(gas.absolute_pressure_offset(), 101325.0);
}

TEST(BoussinesqFreeSurfaceTest, PublishesOptInFieldsAndOccupancyApproximationError)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    solver_coupling.initialize_heated_box(300.0, 300.0);

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
    ASSERT_NE(solver_coupling.configure_free_surface(database), nullptr);

    EXPECT_DOUBLE_EQ(solver_coupling.clear_level().value(0), 0.75);
    EXPECT_DOUBLE_EQ(solver_coupling.pool_level().value(0), 0.75);
    EXPECT_DOUBLE_EQ(solver_coupling.headspace_pressure().value(0), 101325.0);
    EXPECT_DOUBLE_EQ(solver_coupling.pool_occupancy().value(0), 1.0);
    EXPECT_DOUBLE_EQ(solver_coupling.pool_occupancy_volume_error(), 0.25);

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto prefix = std::filesystem::temp_directory_path() / ("SimpleFluid_free_surface_" + std::to_string(unique));
    const auto default_file = prefix.string() + "_default.vtu";
    const auto selected_file = prefix.string() + "_selected.vtu";
    solver_coupling.write_solution_vtu(default_file);
    solver_coupling.write_solution_vtu(
        selected_file, SimpleFluid::SolutionOutputOptions{.include_free_surface_fields = true});

    const auto default_contents = read_file(default_file);
    const auto selected_contents = read_file(selected_file);
    EXPECT_EQ(solver_coupling.liquid_mass_inventory().mode(), SimpleFluid::LiquidVolumeMode::CellMassInventory);
    for (const auto* name :
        {"rhoLiquid", "liquidMassInventory", "clearLevel", "poolLevel", "headspacePressure", "poolOccupancy"})
    {
        const auto marker = std::string("Name=\"") + name + "\"";
        EXPECT_EQ(default_contents.find(marker), std::string::npos);
        EXPECT_NE(selected_contents.find(marker), std::string::npos);
    }
    std::filesystem::remove(default_file);
    std::filesystem::remove(selected_file);

    EXPECT_TRUE(solver_coupling.remove_free_surface_model());
    EXPECT_EQ(solver_coupling.find_free_surface_model(), nullptr);
    EXPECT_EQ(solver_coupling.find_liquid_mass_inventory(), nullptr);
    EXPECT_FALSE(solver_coupling.remove_free_surface_model());
}

TEST(BoussinesqFreeSurfaceTest, RecordsInitializationAndAcceptedStepHistoryWithFixedCsvSchema)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    solver_coupling.configure_free_surface(free_surface_options());
    solver_coupling.initialize_heated_box(300.0, 300.0);
    ASSERT_EQ(solver_coupling.free_surface_history().size(), 1U);
    EXPECT_DOUBLE_EQ(solver_coupling.free_surface_history().front().free_surface.time, 0.0);

    solver_coupling.step();
    ASSERT_EQ(solver_coupling.free_surface_history().size(), 2U);
    const auto& accepted = solver_coupling.free_surface_history().back();
    EXPECT_DOUBLE_EQ(accepted.free_surface.time, 0.1);
    EXPECT_DOUBLE_EQ(accepted.free_surface.time_step, 0.1);
    EXPECT_DOUBLE_EQ(accepted.liquid_mass.total_mass, solver_coupling.liquid_mass_inventory().totalMass());
    EXPECT_DOUBLE_EQ(accepted.pool_occupancy_volume_error, solver_coupling.pool_occupancy_volume_error());
    EXPECT_FALSE(accepted.boiling.has_value());

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto filename = std::filesystem::temp_directory_path() /
                          ("SimpleFluid_free_surface_history_" + std::to_string(unique) + ".csv");
    solver_coupling.write_free_surface_history_csv(filename.string());
    const auto contents = read_file(filename);
    EXPECT_EQ(contents.rfind("time_s,time_step_s,liquid_mass_kg,", 0), 0U);
    // This fixture has no reacting schema: its fixed CSV keeps physical volume,
    // headspace and boiling columns, without the retired single-species slots.
    for (const auto* name : {"h2_generated_mol", "h2_microbubble_submerged_mol",
             "h2_escaped_this_step_mol", "h2_other_sink_mol"})
        EXPECT_EQ(contents.find(name), std::string::npos);
    for (const auto* name : {"submerged_bubble_volume_m3", "headspace_total_moles", "gas_closure_mol",
             "cumulative_evaporated_mass_kg"})
        EXPECT_NE(contents.find(name), std::string::npos);
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
        auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options());
        auto& solver = *solver_shared;
        SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
        EXPECT_THROW(solver_coupling.configure_free_surface(typed_surface_options()), std::invalid_argument);
    }
    {
        auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
            SimpleFluid::LinearSolverOptions{}, material_options());
        auto& solver = *solver_shared;
        SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
        SimpleFluid::Database ideal;
        ideal.set("reacting_bubble_model", std::string("idealGasSource"));
        ideal.set("max_source_alpha_rate", 1.0);
        EXPECT_THROW(solver_coupling.configure_reacting_gas(ideal), std::invalid_argument);
        EXPECT_EQ(solver_coupling.find_reacting_gas_model(), nullptr);
    }
    {
        auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
            SimpleFluid::LinearSolverOptions{}, material_options());
        auto& solver = *solver_shared;
        SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
        SimpleFluid::ScalarVoidFractionOptions scalar_void;
        scalar_void.alpha_max = 0.9;
        scalar_void.initial_alpha = 0.1;
        solver_coupling.configure_scalar_void_fraction(scalar_void);
        EXPECT_THROW(solver_coupling.configure_free_surface(typed_surface_options()), std::invalid_argument);
    }
    {
        auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
            SimpleFluid::LinearSolverOptions{}, material_options());
        auto& solver = *solver_shared;
        SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
        auto ale = typed_surface_options();
        ale.mode = SimpleFluid::FreeSurfaceMode::PlanarALE;
        EXPECT_THROW(solver_coupling.configure_free_surface(ale), std::invalid_argument);
    }
    {
        auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
            SimpleFluid::LinearSolverOptions{}, material_options());
        auto& solver = *solver_shared;
        SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
        SimpleFluid::BoilingSourceOptions boiling;
        boiling.enable_bulk_boiling = true;
        solver_coupling.configure_boiling_source(boiling);
        auto closed = typed_surface_options();
        closed.headspace.mode = SimpleFluid::HeadspaceMode::Closed;
        EXPECT_THROW(solver_coupling.configure_free_surface(closed), std::invalid_argument);
    }
    for (const auto mode : {SimpleFluid::ReactingPressureMode::PrescribedHistory})
    {
        auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
            SimpleFluid::LinearSolverOptions{}, material_options());
        auto& solver = *solver_shared;
        SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
        auto reacting_gas = finite_gas_options();
        reacting_gas.pressure_mode = mode;
        if (mode == SimpleFluid::ReactingPressureMode::PrescribedHistory)
        {
            reacting_gas.pressure_history_times = {0.0, 1.0};
            reacting_gas.pressure_history_values = {1.0e5, 1.0e5};
        }
        solver_coupling.configure_reacting_gas(reacting_gas);
        EXPECT_THROW(solver_coupling.configure_free_surface(typed_surface_options()), std::invalid_argument);
    }
    // The retired inertial selector is rejected at the explicit configuration boundary.
    SimpleFluid::Database retired;
    retired.set("reacting_pressure_mode", std::string("inertial"));
    EXPECT_THROW(SimpleFluid::reacting_gas_options_from_database(retired), std::invalid_argument);
}

TEST(BoussinesqFreeSurfaceTest, InitializedLedgerRequiresFreeSurfaceToBeReconfiguredLast)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    solver_coupling.configure_free_surface(free_surface_options());
    solver_coupling.initialize_heated_box(300.0, 300.0);

    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.reference_density = 1000.0;
    feedback.liquid_density = 1000.0;
    feedback.gas_density = 1.0;
    feedback.reference_temperature = 300.0;
    feedback.reference_dynamic_viscosity = 0.0;
    EXPECT_THROW(solver_coupling.configure_material_feedback(feedback), std::logic_error);

    EXPECT_TRUE(solver_coupling.remove_free_surface_model());
    EXPECT_NO_THROW(solver_coupling.configure_material_feedback(feedback));
}

TEST(BoussinesqFreeSurfaceTest, RejectsInconsistentExplicitInitialMassAndFillVolume)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    auto options = free_surface_options(0.5);
    options.liquid_mass.initial_liquid_mass = 400.0;
    solver_coupling.configure_free_surface(options);
    EXPECT_THROW(solver_coupling.initialize_heated_box(300.0, 300.0), std::invalid_argument);
    EXPECT_EQ(solver_coupling.find_free_surface_model(), nullptr);
}

TEST(BoussinesqFreeSurfaceTest, FailedClosureDoesNotCommitLiquidPhaseChange)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_min = 0.0;
    void_options.alpha_max = 0.9;
    void_options.initial_alpha = 0.0;
    solver_coupling.configure_scalar_void_fraction(void_options);

    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver_coupling.configure_boiling_source(boiling);
    solver_coupling.configure_free_surface(free_surface_options(0.95));
    solver_coupling.initialize_heated_box(301.0, 301.0);

    const auto initial_liquid = solver_coupling.liquid_mass_inventory().diagnostics();
    const auto initial_surface = solver_coupling.free_surface_diagnostics();
    EXPECT_THROW(solver_coupling.step(), std::out_of_range);
    ASSERT_GT(boiling_model.accepted_evaporation_mass_this_step(), 0.0);

    const auto liquid = solver_coupling.liquid_mass_inventory().diagnostics();
    EXPECT_DOUBLE_EQ(liquid.total_mass, initial_liquid.total_mass);
    EXPECT_DOUBLE_EQ(liquid.cumulative_evaporated_mass, initial_liquid.cumulative_evaporated_mass);
    EXPECT_DOUBLE_EQ(liquid.cumulative_condensed_mass, initial_liquid.cumulative_condensed_mass);
    const auto surface = solver_coupling.free_surface_diagnostics();
    EXPECT_DOUBLE_EQ(surface.time, initial_surface.time);
    EXPECT_DOUBLE_EQ(surface.liquid_volume, initial_surface.liquid_volume);
    EXPECT_DOUBLE_EQ(surface.pool_volume, initial_surface.pool_volume);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);
    EXPECT_THROW(solver_coupling.step(), std::logic_error);
}

TEST(BoussinesqFreeSurfaceTest, ClosedReconstructedPressureUsesCollectiveValidLowerBound)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    auto solver_shared = std::make_shared<Solver>(mesh, SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    auto reacting_gas = finite_gas_options(.2, 1000, 1e10, 1e-6, 90000);
    reacting_gas.pressure_mode = SimpleFluid::ReactingPressureMode::Reconstructed;
    reacting_gas.minimum_absolute_pressure = 90000.0;
    auto& gas = solver_coupling.configure_reacting_gas(reacting_gas);
    solver_coupling.add_prescribed_heat_source().initialize_constant(0.0);
    solver_coupling.initialize_heated_box(300.0, 300.0);

    solver.pressure().set_owned_value(0, -1000.0);
    solver.pressure().set_owned_value(1, 1000.0);
    solver.pressure().sync_ghosts();
    gas.initialize_state(
        solver.time(), solver.temperature(), solver.pressure(), solver.velocity(), solver.material_properties(), true);
    EXPECT_NEAR(gas.minimum_valid_absolute_pressure_offset(), 91000.0, 1.0e-10);

    auto closed = typed_surface_options(0.4);
    closed.headspace.mode = SimpleFluid::HeadspaceMode::Closed;
    closed.headspace.total_internal_volume = closed.vessel.total_internal_volume;
    closed.coupling.maximum_absolute_pressure = 2.0e5;
    closed.headspace.infer_initial_moles = false;
    const double available = closed.vessel.total_internal_volume - gas.evaluate_liquid_volume(101325) -
                             gas.evaluate_submerged_bubble_volume(101325);
    closed.headspace.initial_moles = {{SurfaceGas::gas_name, 101325 * available / (SurfaceGas::M::gas_constant * 300)}};
    ASSERT_NE(solver_coupling.configure_free_surface(
                  closed, std::make_unique<SimpleFluid::ClosedMixtureHeadspaceModel>(
                              closed.headspace, reacting_gas.gas_thermodynamics, "test.explicit_ideal_headspace", 1)),
        nullptr);
    const auto diagnostics = solver_coupling.free_surface_diagnostics();
    EXPECT_NEAR(diagnostics.headspace.pressure, closed.headspace.initial_pressure, 1.0e-5);
    EXPECT_GT(diagnostics.pool_level, diagnostics.clear_level);
}

TEST(BoussinesqFreeSurfaceTest, FailedStepRestoresTypedEscapeAndAllowsSafeRetry)
{
    auto solver_shared = std::make_shared<Solver>(make_mesh(), SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    auto reacting_gas = finite_gas_options(.4, 1000, 1e10, 1e-6);
    reacting_gas.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    reacting_gas.constant_slip_velocity = 10.0;
    reacting_gas.free_surface_patches = {"zmax"};
    auto& gas = solver_coupling.configure_reacting_gas(reacting_gas);
    solver_coupling.add_prescribed_heat_source().initialize_constant(0.0);

    auto surface_options = typed_surface_options(0.4);
    surface_options.headspace.total_internal_volume = surface_options.vessel.total_internal_volume;
    surface_options.headspace.temperature_mode = SimpleFluid::HeadspaceTemperatureMode::Prescribed;
    surface_options.headspace.prescribed_temperature_times = {0.0, 0.05};
    surface_options.headspace.prescribed_temperature_values = {300.0, 300.0};
    solver_coupling.configure_free_surface(surface_options);
    solver_coupling.initialize_heated_box(300.0, 300.0);
    ASSERT_EQ(solver_coupling.free_surface_history().size(), 1U);

    const SurfaceGas::M::SavedReactingState accepted_gas(gas);
    const auto accepted_number_escape = gas.population_escape_receipt();
    const auto accepted_liquid_mass = solver_coupling.liquid_mass_inventory().totalMass();
    EXPECT_THROW(solver_coupling.step(), std::out_of_range);
    // The historical unsafe-retry case now uses typed atomic source ownership:
    // trial escape must not leak into accepted material, counts or receipts.
    accepted_gas.expect_restored(gas);
    EXPECT_EQ(gas.population_escape_receipt(), accepted_number_escape);
    EXPECT_DOUBLE_EQ(SurfaceGas::escaped(gas), 0.0);
    EXPECT_DOUBLE_EQ(solver_coupling.liquid_mass_inventory().totalMass(), accepted_liquid_mass);
    EXPECT_TRUE(solver_coupling.free_surface_diagnostics().vented_gas_moles.empty());
    EXPECT_EQ(solver_coupling.free_surface_history().size(), 1U);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);

    // The original full step still exceeds the supplied temperature table.
    EXPECT_THROW(solver_coupling.step(), std::out_of_range);
    accepted_gas.expect_restored(gas);
    EXPECT_EQ(gas.population_escape_receipt(), accepted_number_escape);
    EXPECT_EQ(solver_coupling.free_surface_history().size(), 1U);
    solver.set_time_step(0.05);
    ASSERT_NO_THROW(solver_coupling.step());
    EXPECT_GT(SurfaceGas::escaped(gas), 0.0);
    EXPECT_EQ(solver_coupling.free_surface_history().size(), 2U);
    EXPECT_DOUBLE_EQ(solver.time(), 0.05);
    EXPECT_EQ(solver.step_index(), 1);
}

TEST(BoussinesqFreeSurfaceTest, CellMassInventoryStepIsPartitionIndependent)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }
    auto solver_shared = std::make_shared<Solver>(mesh, SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    solver_coupling.configure_free_surface(cell_mass_free_surface_options());
    solver_coupling.initialize_heated_box(300.0, 300.0);

    solver_coupling.step();

    const auto diagnostics = solver_coupling.liquid_mass_inventory().diagnostics();
    EXPECT_NEAR(diagnostics.total_mass, 500.0, 1.0e-8);
    EXPECT_NEAR(diagnostics.liquid_volume, 0.5, 1.0e-10);
    EXPECT_NEAR(diagnostics.step_mass_balance_residual, 0.0, 1.0e-8);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        EXPECT_NEAR(solver_coupling.liquid_mass_inventory().cellMassInventory().value(
                        static_cast<Pack::local_ordinal_type>(owned)),
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
    auto solver_shared = std::make_shared<Solver>(mesh, SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    auto options = free_surface_options();
    if (communicator->getRank() != 0)
    {
        options.vessel.cross_section_area = 2.0;
    }
    EXPECT_THROW(solver_coupling.configure_free_surface(options), std::invalid_argument);
    EXPECT_EQ(solver_coupling.find_free_surface_model(), nullptr);
}

TEST(BoussinesqFreeSurfaceTest, CollectivelyRejectsRemovalWhileBoilingOwnsSubmergedSteam)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(2, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }

    auto solver_shared = std::make_shared<Solver>(mesh, SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    SimpleFluid::ScalarVoidFractionOptions void_options;
    void_options.alpha_max = 0.9;
    solver_coupling.configure_scalar_void_fraction(void_options);
    SimpleFluid::BoilingSourceOptions boiling;
    boiling.enable_bulk_boiling = true;
    boiling.saturation_temperature = 300.0;
    boiling.boiling_time_scale = 1.0;
    boiling.latent_heat = 1000.0;
    boiling.gas_density = 1.0;
    auto& boiling_model = solver_coupling.configure_boiling_source(boiling);
    solver_coupling.configure_free_surface(free_surface_options());
    solver_coupling.initialize_heated_box(301.0, 301.0);
    solver_coupling.step();

    ASSERT_GT(boiling_model.global_submerged_steam_mass(), 0.0);
    try
    {
        static_cast<void>(solver_coupling.remove_free_surface_model());
        FAIL() << "Expected nonzero submerged steam to block free-surface removal.";
    }
    catch (const std::logic_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("nonzero submerged steam"), std::string::npos);
    }
    EXPECT_NE(solver_coupling.find_free_surface_model(), nullptr);
    EXPECT_NE(solver_coupling.find_liquid_mass_inventory(), nullptr);
}

TEST(BoussinesqFreeSurfaceTest, CollectivelyRejectsRankDivergentExistingStateDuringConfiguration)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(2, 1, 1));
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }

    auto solver_shared = std::make_shared<Solver>(mesh, SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    ASSERT_NE(solver_coupling.configure_free_surface(free_surface_options()), nullptr);
    if (communicator->getRank() != 0)
    {
        EXPECT_TRUE(solver_coupling.remove_free_surface_model());
    }
    EXPECT_THROW(solver_coupling.configure_free_surface(free_surface_options()), std::invalid_argument);
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

    auto solver_shared = std::make_shared<Solver>(mesh, SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    ASSERT_NE(solver_coupling.configure_free_surface(free_surface_options()), nullptr);
    if (communicator->getRank() != 0)
    {
        EXPECT_TRUE(solver_coupling.remove_free_surface_model());
    }

    EXPECT_THROW(solver_coupling.step(), std::invalid_argument);
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

    auto solver_shared = std::make_shared<Solver>(mesh, SimpleFluid::BoundaryConditionSet{}, time_options(),
        SimpleFluid::LinearSolverOptions{}, material_options());
    auto& solver = *solver_shared;
    SimpleFluid::MultiphysicsSolverCoupler<Pack> solver_coupling(solver_shared);
    ASSERT_NE(solver_coupling.configure_free_surface(free_surface_options()), nullptr);
    solver.set_time_step(communicator->getRank() == 0 ? 0.25 : 0.5);

    EXPECT_THROW(solver_coupling.step(), std::invalid_argument);
    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    EXPECT_EQ(solver.step_index(), 0);
}

} // namespace
