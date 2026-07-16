#include <gtest/gtest.h>

#include "equations/BoilingSourceModel.hh"
#include "equations/DelayedNeutronPrecursorModel.hh"
#include "equations/FeedbackMap.hh"
#include "equations/MaterialFeedbackModel.hh"
#include "equations/ScalarVoidFractionModel.hh"
#include "dataclass/Database.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

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
using FieldType = SimpleFluid::CellField<Pack>;
using VelocityFieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_single_cell_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
}

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
}

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
}

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
    EXPECT_NEAR(void_model.alpha_g().value(0), 0.2, 1.0e-14);
    EXPECT_NEAR(
        void_model.source_alpha_total().value(0), 0.1, 1.0e-14);

    radiolysis.put_scalar(0.0);
    boiling_model.update(
        1.0,
        temperature,
        material,
        void_model,
        &radiolysis);
    EXPECT_DOUBLE_EQ(boiling_model.source_alpha_boil().value(0), 0.0);
    EXPECT_DOUBLE_EQ(boiling_model.latent_heat_sink().value(0), 0.0);
}

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
}

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

TEST(ScalarVoidFractionModelTest, DisabledSourcesLeaveAlphaUnchanged)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::ScalarVoidFractionOptions options;
    options.initial_alpha = 0.12;
    SimpleFluid::ScalarVoidFractionModel<Pack> model(mesh, options);

    model.update_explicit(10.0, nullptr, nullptr);
    EXPECT_DOUBLE_EQ(model.alpha_g().value(0), 0.12);
    EXPECT_DOUBLE_EQ(model.alpha_l().value(0), 0.88);
    EXPECT_DOUBLE_EQ(model.source_alpha_total().value(0), 0.0);
}

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

TEST(Phase13PlusCouplingTest,
     AdvancedMirrorAggregatesItsActualScalarStateChange)
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
    EXPECT_NEAR(void_model.source_alpha_total().value(0), -1.0, 1.0e-13);
}

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
    }
}

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
    }
}

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
}

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
    EXPECT_NEAR(
        model.concentration(0).value(0),
        4.0 / 2.0 * (1.0 - std::exp(-1.0)),
        1.0e-12);

    options.decay_constants = {1.0e-18};
    model.configure(options);
    model.advance(0.5, alpha_l, nullptr);
    EXPECT_NEAR(model.concentration(0).value(0), 2.0, 1.0e-12);
}

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
}

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
    EXPECT_NE(contents.find("Name=\"rhoFeedback\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"muFeedback\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"C_1\""), std::string::npos);
    std::filesystem::remove(output);
}
