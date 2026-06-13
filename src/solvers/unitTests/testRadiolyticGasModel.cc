#include <gtest/gtest.h>

#include "equations/RadiolyticGasModel.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

#include <chrono>
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
using FaceFieldType = SimpleFluid::FaceField<Pack>;

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

SimpleFluid::RadiolyticGasOptions ideal_options()
{
    SimpleFluid::RadiolyticGasOptions options;
    options.mode = SimpleFluid::RadiolyticGasMode::IdealGasSource;
    options.hydrogen_yield_mol_per_j = 2.0e-7;
    options.max_source_alpha_rate = 1.0;
    options.reference_pressure = 1.0e5;
    return options;
}

SimpleFluid::RadiolyticGasOptions sheng_options()
{
    auto options = ideal_options();
    options.mode =
        SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    options.henry_coefficient = 1.0e-5;
    options.surface_tension = 0.07;
    options.hydrogen_diffusivity = 1.0e-8;
    options.uranium_concentration_mol_per_m3 = 1000.0;
    options.hydrogen_yield_molecules_per_100_ev = 1.8;
    options.min_radius = 1.0e-12;
    options.max_radius = 1.0e-3;
    options.min_population = 1.0e-40;
    options.max_population = 1.0e40;
    return options;
}

TEST(RadiolyticGasModelTest, IdealSourceDoesNotAdvanceVoidFraction)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::RadiolyticGasModel<Pack> model(
        mesh, ideal_options());
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 4.0e6, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    model.advance(
        0.1, 0.1, temperature, pressure, velocity, flux, material,
        &power);

    const auto expected =
        2.0e-7 * 4.0e6 * 8.31446261815324 * 300.0 / 1.0e5;
    EXPECT_NEAR(model.source_alpha_rad().value(0), expected, 1.0e-14);
    EXPECT_DOUBLE_EQ(model.alpha_g().value(0), 0.0);
    EXPECT_DOUBLE_EQ(model.alpha_l().value(0), 1.0);
}

TEST(RadiolyticGasModelTest, IdealSourceHonorsAlphaAndRateLimits)
{
    auto mesh = make_single_cell_mesh();
    auto options = ideal_options();
    options.alpha_max = 0.2;
    options.max_source_alpha_rate = 0.01;
    SimpleFluid::RadiolyticGasModel<Pack> model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 1.0e12, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    model.advance(
        0.1, 0.1, temperature, pressure, velocity, flux, material,
        &power);
    EXPECT_DOUBLE_EQ(model.source_alpha_rad().value(0), 0.01);

    model.alpha_g().put_scalar(options.alpha_max);
    model.advance(
        0.2, 0.1, temperature, pressure, velocity, flux, material,
        &power);
    EXPECT_DOUBLE_EQ(model.source_alpha_rad().value(0), 0.0);
}

TEST(RadiolyticGasModelTest, TwoPopulationStepConservesHydrogen)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    SimpleFluid::RadiolyticGasModel<Pack> model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 1.0e5, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    constexpr double time_step = 1.0e-6;
    model.advance(
        time_step,
        time_step,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    const auto& statistics = model.last_statistics();
    EXPECT_NEAR(
        statistics.hydrogen_produced,
        options.hydrogen_yield_mol_per_j * 1.0e5 * time_step,
        1.0e-15);
    EXPECT_NEAR(statistics.inventory_error, 0.0, 1.0e-14);
    EXPECT_GT(statistics.hydrogen_after, 0.0);
    EXPECT_GT(model.alpha_g().value(0), 0.0);
    EXPECT_GE(model.dissolved_hydrogen_inventory().value(0), 0.0);
    EXPECT_GE(model.micro_moles().value(0), 0.0);
}

TEST(RadiolyticGasModelTest, MicrobubbleDecayIsAnalyticAndConservative)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_micro_number_density = 1.0e10;
    options.initial_micro_moles = 1.0e-6;
    SimpleFluid::RadiolyticGasModel<Pack> model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    constexpr double time_step = 5.0e-6;
    model.advance(
        time_step,
        time_step,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    const auto retention =
        std::exp(-time_step / options.microbubble_lifetime);
    EXPECT_NEAR(
        model.micro_number_density().value(0),
        options.initial_micro_number_density * retention,
        options.initial_micro_number_density * 1.0e-12);
    EXPECT_NEAR(
        model.micro_moles().value(0),
        options.initial_micro_moles * retention,
        1.0e-16);
    EXPECT_NEAR(
        model.dissolved_hydrogen_inventory().value(0)
          + model.micro_moles().value(0),
        options.initial_micro_moles,
        1.0e-15);
}

TEST(RadiolyticGasModelTest, PrescribedPressureHistoryInterpolates)
{
    auto mesh = make_single_cell_mesh();
    auto options = ideal_options();
    options.pressure_mode =
        SimpleFluid::RadiolyticPressureMode::PrescribedHistory;
    options.pressure_history_times = {0.0, 1.0};
    options.pressure_history_values = {1.0e5, 2.0e5};
    SimpleFluid::RadiolyticGasModel<Pack> model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 8.0e4, "pressure");
    FieldType power(mesh, 1.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    model.advance(
        0.5, 0.1, temperature, pressure, velocity, flux, material,
        &power);
    EXPECT_DOUBLE_EQ(model.absolute_pressure().value(0), 1.5e5);
}

TEST(RadiolyticGasModelTest, FreeSurfaceEscapeAccumulatesGlobally)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_micro_number_density = 1.0e10;
    options.initial_micro_moles = 1.0e-6;
    options.rise_velocity_mode =
        SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    options.constant_slip_velocity = 0.1;
    options.free_surface_patches = {"zmax"};
    SimpleFluid::RadiolyticGasModel<Pack> model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    constexpr double time_step = 1.0e-6;
    model.advance(
        time_step,
        time_step,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);
    const auto first = model.last_statistics();
    EXPECT_GT(first.hydrogen_escaped, 0.0);
    EXPECT_GT(first.escaped_bubble_count, 0.0);
    EXPECT_DOUBLE_EQ(
        first.cumulative_hydrogen_escaped,
        first.hydrogen_escaped);

    model.advance(
        2.0 * time_step,
        time_step,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);
    EXPECT_GT(
        model.last_statistics().cumulative_hydrogen_escaped,
        first.cumulative_hydrogen_escaped);
    EXPECT_GT(
        model.last_statistics().cumulative_escaped_bubble_count,
        first.cumulative_escaped_bubble_count);
}

TEST(RadiolyticGasModelTest, CelataSelectorDrivesBubbleEscape)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_large_number_density = 100.0;
    options.initial_large_moles = 1.0e-6;
    options.rise_velocity_mode =
        SimpleFluid::BubbleRiseVelocityMode::Celata2007;
    options.bubble_gas_density = 1.2;
    options.free_surface_patches = {"zmax"};
    SimpleFluid::RadiolyticGasModel<Pack> model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    model.advance(
        1.0e-6,
        1.0e-6,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    EXPECT_GT(model.last_statistics().hydrogen_escaped, 0.0);
    EXPECT_GT(model.last_statistics().escaped_bubble_count, 0.0);
}

TEST(RadiolyticGasModelTest, AxialTransportDropsTransverseLiquidFlux)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_large_number_density = 100.0;
    options.initial_large_moles = 1.0e-6;
    options.free_surface_patches = {"xmax"};
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(
        mesh, MeshType::Vec3{0.1, 0.0, 0.0}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<Pack::local_ordinal_type>(face);
        if (flux.is_owned_face(face_lid))
        {
            flux.set_value(
                face_lid,
                0.1 * mesh->face_area_vector(face_lid).x);
        }
    }
    auto material = make_water_properties(mesh);

    options.bubble_transport =
        SimpleFluid::BubbleTransportMode::General;
    SimpleFluid::RadiolyticGasModel<Pack> general(mesh, options);
    general.advance(
        1.0e-6,
        1.0e-6,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);
    EXPECT_GT(general.last_statistics().hydrogen_escaped, 0.0);

    options.bubble_transport =
        SimpleFluid::BubbleTransportMode::Axial;
    SimpleFluid::RadiolyticGasModel<Pack> axial(mesh, options);
    axial.advance(
        1.0e-6,
        1.0e-6,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);
    EXPECT_DOUBLE_EQ(axial.last_statistics().hydrogen_escaped, 0.0);
}

TEST(RadiolyticGasModelTest, SolverOwnsAndPublishesOptionalModel)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::TimeStepperOptions time_options;
    auto model_options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, {}, time_options, {}, model_options);
    solver.configure_radiolytic_gas(ideal_options());
    ASSERT_NE(solver.find_radiolytic_gas_model(), nullptr);

    const auto unique_id =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output =
        std::filesystem::temp_directory_path()
        / ("SimpleFluid_radiolysis_" + std::to_string(unique_id)
           + ".vtu");
    solver.write_solution_vtu(
        output.string(),
        SimpleFluid::SolutionOutputOptions{
            .include_sources = true,
            .include_radiolytic_gas_fields = true});

    std::ifstream input(output);
    const std::string contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    EXPECT_NE(
        contents.find("Name=\"S_alpha_rad\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"alpha_g\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"p_abs\""), std::string::npos);

    EXPECT_TRUE(solver.remove_radiolytic_gas_model());
    EXPECT_EQ(solver.find_radiolytic_gas_model(), nullptr);
    std::filesystem::remove(output);
}

} // namespace
