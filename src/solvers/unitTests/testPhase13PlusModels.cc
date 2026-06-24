#include <gtest/gtest.h>

#include "equations/BoilingSourceModel.hh"
#include "equations/DelayedNeutronPrecursorModel.hh"
#include "equations/FeedbackMap.hh"
#include "equations/MaterialFeedbackModel.hh"
#include "equations/ScalarVoidFractionModel.hh"
#include "dataclass/Database.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
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

} // namespace

TEST(BoilingSourceModelTest, BulkThresholdAndLatentHeatAreConsistent)
{
    auto mesh = make_single_cell_mesh();
    auto material = make_water_properties(mesh);
    FieldType temperature(mesh, 360.0, "temperature");

    SimpleFluid::BoilingSourceOptions options;
    options.enable_bulk_boiling = true;
    options.saturation_temperature = 373.0;
    options.boiling_activation_delta_t = 2.0;
    options.boiling_time_scale = 4.0;
    options.latent_heat = 10.0;
    options.gas_density = 2.0;
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);

    model.update(temperature, material);
    EXPECT_DOUBLE_EQ(model.source_alpha_boil().value(0), 0.0);
    EXPECT_DOUBLE_EQ(model.latent_heat_sink().value(0), 0.0);

    temperature.put_scalar(383.0);
    model.update(temperature, material);
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

    SimpleFluid::BoilingSourceOptions options;
    options.enable_wall_boiling = true;
    options.latent_heat = 20.0;
    options.gas_density = 4.0;
    options.wall_evaporation_fraction = 0.5;
    options.wall_heat_flux = 80.0;
    options.wall_boiling_patches = {"zmax"};
    SimpleFluid::BoilingSourceModel<Pack> model(mesh, options);
    model.update(temperature, material);

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
