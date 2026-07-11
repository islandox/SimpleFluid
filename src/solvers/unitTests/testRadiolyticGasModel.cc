/**
 * @file testRadiolyticGasModel.cc
 * @brief Unit tests for ideal and Sheng-style radiolytic gas models.
 */

#include <gtest/gtest.h>

#include "equations/RadiolyticGasModel.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/BoussinesqSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VelocityFieldType = SimpleFluid::VectorCellField<Pack>;
using FaceFieldType = SimpleFluid::FaceField<Pack>;
using RadiolyticModelType = SimpleFluid::RadiolyticGasModel<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/**
 * @brief Build the one-cell mesh used by local radiolysis checks.
 */
SimpleFluid::SP<MeshType> make_single_cell_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
}

/**
 * @brief Create water-like material fields for the radiolysis tests.
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

/**
 * @brief Baseline ideal-gas radiolysis options with bounded alpha source.
 */
SimpleFluid::RadiolyticGasOptions ideal_options()
{
    SimpleFluid::RadiolyticGasOptions options;
    options.mode = SimpleFluid::RadiolyticGasMode::IdealGasSource;
    options.hydrogen_yield_mol_per_j = 2.0e-7;
    options.max_source_alpha_rate = 1.0;
    options.reference_pressure = 1.0e5;
    return options;
}

/**
 * @brief Sheng two-population options with finite correlation bounds.
 */
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

/**
 * @brief Sum a scalar diagnostic across the mesh communicator.
 */
double global_sum(const MeshType& mesh, double local_value)
{
    double global_value = 0.0;
    Teuchos::reduceAll(
        *mesh.owned_cell_map()->getComm(),
        Teuchos::REDUCE_SUM,
        1,
        &local_value,
        &global_value);
    return global_value;
}

/**
 * @brief Compute the distributed cell-volume integral of a cell field.
 */
double global_integral(const FieldType& field)
{
    double local_integral = 0.0;
    const auto& mesh = field.mesh();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        local_integral +=
            field.value(cell_lid) * mesh.cell_volume(cell_lid);
    }
    return global_sum(mesh, local_integral);
}

/**
 * @brief Return a named radiolytic output field or fail the test helper.
 */
const FieldType& model_field(
    const RadiolyticModelType& model,
    const std::string& name)
{
    const auto fields = model.output_fields();
    const auto iterator = fields.find(name);
    if (iterator == fields.end() || iterator->second == nullptr)
    {
        throw std::runtime_error(
            "RadiolyticGasModel did not publish field '" + name + "'.");
    }
    return *iterator->second;
}

/**
 * @brief Expect every owned field value to be finite.
 */
void expect_field_finite(
    const FieldType& field,
    const std::string& name)
{
    const auto& mesh = field.mesh();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_TRUE(std::isfinite(field.value(cell_lid)))
            << name << " at owned cell " << owned;
    }
}

/**
 * @brief Expect every owned field value to be finite and non-negative.
 */
void expect_field_finite_non_negative(
    const FieldType& field,
    const std::string& name)
{
    const auto& mesh = field.mesh();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        const auto value = field.value(cell_lid);
        EXPECT_TRUE(std::isfinite(value))
            << name << " at owned cell " << owned;
        EXPECT_GE(value, 0.0) << name << " at owned cell " << owned;
    }
}

/**
 * @brief Check all published Sheng-model state and diagnostic bounds.
 */
void expect_state_fields_finite_bounded(
    const RadiolyticModelType& model)
{
    const auto& options = model.options();
    const std::array<std::string, 21> non_negative_fields{
        "C_H2",
        "I_H2",
        "I_H2_excluded",
        "N_micro",
        "M_micro",
        "N_large",
        "M_large",
        "r_nuc",
        "r_micro",
        "r_large",
        "C_H2_crit",
        "C_H2_eq",
        "K_L",
        "alpha_g_micro",
        "alpha_g_large",
        "alpha_g_raw",
        "alpha_g_excess",
        "r_characteristic",
        "H2_production_rate",
        "H2_escape_molar_rate",
        "bubble_escape_number_rate"};

    for (const auto& name : non_negative_fields)
    {
        expect_field_finite_non_negative(model_field(model, name), name);
    }

    const std::array<std::string, 4> finite_diagnostics{
        "micro_to_large_number_rate",
        "micro_to_large_molar_rate",
        "large_growth_rate",
        "H2_dissolution_rate"};
    for (const auto& name : finite_diagnostics)
    {
        expect_field_finite(model_field(model, name), name);
    }

    const auto& alpha_g = model.alpha_g();
    const auto& alpha_l = model.alpha_l();
    for (size_t owned = 0; owned < alpha_g.mesh().num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_GE(alpha_g.value(cell_lid), options.alpha_min);
        EXPECT_LE(alpha_g.value(cell_lid), options.alpha_max);
        EXPECT_GE(alpha_l.value(cell_lid), 1.0 - options.alpha_max);
        EXPECT_LE(alpha_l.value(cell_lid), 1.0 - options.alpha_min);
    }
}

/**
 * @brief Evaluate the pressure-dependent critical H2 concentration.
 */
double critical_concentration(
    const SimpleFluid::RadiolyticGasOptions& options,
    double pressure,
    double temperature)
{
    const auto radius =
        SimpleFluid::RadiolyticGasPhysics::sheng2024_nucleation_radius(
            temperature,
            options.uranium_concentration_mol_per_m3,
            options.hydrogen_yield_molecules_per_100_ev,
            pressure,
            options.atmospheric_pressure);
    return SimpleFluid::RadiolyticGasPhysics::
        henry_equilibrium_concentration(
            options.henry_coefficient,
            pressure,
            options.surface_tension,
            radius);
}

/**
 * @brief Advance the model and assert that the primary void field stays finite.
 */
void advance_model(
    RadiolyticModelType& model,
    const SimpleFluid::SP<MeshType>& mesh,
    double time,
    double time_step,
    FieldType& temperature,
    FieldType& pressure,
    FieldType& power,
    VelocityFieldType& velocity,
    FaceFieldType& flux,
    SimpleFluid::MaterialPropertyFields<Pack>& material)
{
    model.advance(
        time,
        time_step,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        ASSERT_TRUE(std::isfinite(model.alpha_g().value(cell_lid)));
    }
}

/**
 * @brief Ideal-gas source computes alpha production without changing alpha.
 */
TEST(RadiolyticGasModelTest,
     IdealSourceRequiresAndDoesNotAdvanceAuthoritativeVoid)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::RadiolyticGasModel<Pack> model(
        mesh, ideal_options());
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 4.0e6, "qdot_fission");
    FieldType authoritative_alpha(mesh, 0.0, "authoritative_alpha_g");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    EXPECT_THROW(
        model.advance(
            0.1,
            0.1,
            temperature,
            pressure,
            velocity,
            flux,
            material,
            &power),
        std::logic_error);
    model.advance(
        0.1,
        0.1,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power,
        authoritative_alpha,
        model.options().alpha_max);

    const auto expected =
        2.0e-7 * 4.0e6 * 8.31446261815324 * 300.0 / 1.0e5;
    EXPECT_NEAR(model.source_alpha_rad().value(0), expected, 1.0e-14);
    EXPECT_DOUBLE_EQ(model.alpha_g().value(0), 0.0);
    EXPECT_DOUBLE_EQ(model.alpha_l().value(0), 1.0);
}

/**
 * @brief Ideal-gas source respects configured rate and void-fraction limits.
 */
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
    FieldType authoritative_alpha(mesh, 0.0, "authoritative_alpha_g");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    model.advance(
        0.1, 0.1, temperature, pressure, velocity, flux, material,
        &power, authoritative_alpha, options.alpha_max);
    EXPECT_DOUBLE_EQ(model.source_alpha_rad().value(0), 0.01);

    authoritative_alpha.put_scalar(options.alpha_max);
    model.advance(
        0.2, 0.1, temperature, pressure, velocity, flux, material,
        &power, authoritative_alpha, options.alpha_max);
    EXPECT_DOUBLE_EQ(model.source_alpha_rad().value(0), 0.0);
    EXPECT_DOUBLE_EQ(model.alpha_g().value(0), options.alpha_max);
}

/**
 * @brief Sheng two-population update conserves the produced hydrogen inventory.
 */
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

/**
 * @brief Microbubble decay follows the analytic lifetime solution.
 */
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

/**
 * @brief Micro-to-large conversion transfers number and gas inventory.
 */
TEST(RadiolyticGasModelTest, MicroToLargeConversionConservesCategories)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_dissolved_hydrogen = 1.0e3;
    options.initial_micro_number_density = 1.0e8;
    options.initial_micro_moles = 1.0e-8;
    options.microbubble_lifetime = 1.0e9;
    options.large_bubble_dissolution_time = 1.0e9;
    options.micro_to_large_conversion_coefficient = 1.0e-8;
    SimpleFluid::RadiolyticGasModel<Pack> model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    const auto total_number_before =
        model.micro_number_density().value(0)
      + model.large_number_density().value(0);
    const auto total_moles_before =
        model.micro_moles().value(0)
      + model.large_moles().value(0);
    model.advance(
        1.0e-7,
        1.0e-7,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    EXPECT_LT(
        model.micro_number_density().value(0),
        options.initial_micro_number_density);
    EXPECT_GT(model.large_number_density().value(0), 0.0);
    EXPECT_NEAR(
        model.micro_number_density().value(0)
      + model.large_number_density().value(0),
        total_number_before,
        total_number_before * 1.0e-12);
    EXPECT_NEAR(
        model.micro_moles().value(0) + model.large_moles().value(0),
        total_moles_before,
        total_moles_before * 1.0e-10);
}

/**
 * @brief Large bubbles grow or dissolve according to local saturation.
 */
TEST(RadiolyticGasModelTest, LargeBubbleGrowthAndDissolutionFollowSaturation)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_dissolved_hydrogen = 1.0e3;
    options.initial_large_number_density = 1.0e6;
    options.initial_large_moles = 1.0e-7;
    options.large_bubble_dissolution_time = 1.0e9;
    SimpleFluid::RadiolyticGasModel<Pack> saturated(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    saturated.advance(
        1.0e-6,
        1.0e-6,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);
    EXPECT_GT(saturated.large_moles().value(0), 1.0e-7);
    EXPECT_GT(saturated.alpha_g().value(0), 0.0);
    EXPECT_NEAR(
        saturated.large_number_density().value(0),
        options.initial_large_number_density,
        options.initial_large_number_density * 1.0e-12);

    options.initial_dissolved_hydrogen = 0.0;
    options.large_bubble_dissolution_time = 1.0e-5;
    SimpleFluid::RadiolyticGasModel<Pack> undersaturated(mesh, options);
    undersaturated.advance(
        1.0e-6,
        1.0e-6,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);
    EXPECT_LT(undersaturated.large_moles().value(0), 1.0e-7);
    EXPECT_LT(
        undersaturated.large_number_density().value(0),
        options.initial_large_number_density);
}

/**
 * @brief Prescribed pressure mode interpolates the supplied pressure history.
 */
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
    FieldType authoritative_alpha(mesh, 0.0, "authoritative_alpha_g");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    model.advance(
        0.5, 0.1, temperature, pressure, velocity, flux, material,
        &power, authoritative_alpha, options.alpha_max);
    EXPECT_DOUBLE_EQ(model.absolute_pressure().value(0), 1.5e5);
}

/**
 * @brief Free-surface bubble escape updates step and cumulative statistics.
 */
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

/**
 * @brief Celata rise-velocity selection produces free-surface escape.
 */
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

/**
 * @brief Axial bubble transport ignores transverse liquid fluxes.
 */
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

/**
 * @brief Population transport conserves global number and H2 without escape.
 */
TEST(RadiolyticGasModelTest, PopulationTransportConservesInventoryWithoutEscape)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    auto options = sheng_options();
    options.initial_large_number_density = 1.0e6;
    options.initial_large_moles = 1.0e-7;
    options.microbubble_lifetime = 1.0e9;
    options.large_bubble_dissolution_time = 1.0e9;
    options.micro_to_large_conversion_coefficient = 0.0;
    const auto large_radius =
        SimpleFluid::RadiolyticGasPhysics::solve_bubble_radius(
            options.initial_large_moles
                / options.initial_large_number_density,
            options.reference_pressure,
            options.surface_tension,
            options.gas_constant,
            300.0,
            options.min_radius,
            options.max_radius);
    ASSERT_TRUE(large_radius.converged);
    options.initial_dissolved_hydrogen =
        SimpleFluid::RadiolyticGasPhysics::
            henry_equilibrium_concentration(
                options.henry_coefficient,
                options.reference_pressure,
                options.surface_tension,
                large_radius.radius);

    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{0.1, 0.0, 0.0}, "velocity");
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

    const auto number_before =
        global_integral(model.large_number_density());
    const auto moles_before =
        global_integral(model.large_moles());
    const auto hydrogen_before =
        global_integral(model.dissolved_hydrogen_inventory())
      + global_integral(model.micro_moles())
      + global_integral(model.large_moles());

    model.advance(
        1.0e-4,
        1.0e-4,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    const auto number_after =
        global_integral(model.large_number_density());
    const auto moles_after =
        global_integral(model.large_moles());
    const auto hydrogen_after =
        global_integral(model.dissolved_hydrogen_inventory())
      + global_integral(model.micro_moles())
      + global_integral(model.large_moles());

    EXPECT_NEAR(number_after, number_before, number_before * 1.0e-9);
    EXPECT_NEAR(moles_after, moles_before, moles_before * 1.0e-9);
    EXPECT_NEAR(hydrogen_after, hydrogen_before, hydrogen_before * 1.0e-9);
    EXPECT_DOUBLE_EQ(model.last_statistics().hydrogen_escaped, 0.0);
    EXPECT_DOUBLE_EQ(model.last_statistics().escaped_bubble_count, 0.0);

    ASSERT_GE(mesh->num_owned_cells(), 2U);
    const auto first = model.large_number_density().value(0);
    const auto second = model.large_number_density().value(1);
    EXPECT_GT(
        std::max(
            std::abs(first - options.initial_large_number_density),
            std::abs(second - options.initial_large_number_density)),
        options.initial_large_number_density * 1.0e-8);
}

/**
 * @brief Higher pressure raises the critical concentration and delays conversion.
 */
TEST(RadiolyticGasModelTest, HigherPressureRaisesCriticalConcentrationAndDelaysConversion)
{
    constexpr double temperature_value = 300.0;
    constexpr double low_pressure = 1.0e5;
    constexpr double high_pressure = 5.0e5;
    auto base_options = sheng_options();
    base_options.initial_micro_number_density = 1.0e8;
    base_options.initial_micro_moles = 1.0e-8;
    base_options.microbubble_lifetime = 1.0e9;
    base_options.large_bubble_dissolution_time = 1.0e9;
    base_options.micro_to_large_conversion_coefficient = 1.0e-4;
    const auto low_critical = critical_concentration(
        base_options, low_pressure, temperature_value);
    const auto high_critical = critical_concentration(
        base_options, high_pressure, temperature_value);
    ASSERT_GT(high_critical, low_critical);
    base_options.initial_dissolved_hydrogen =
        0.5 * (low_critical + high_critical);

    auto run_case = [&](double reference_pressure)
    {
        auto options = base_options;
        options.reference_pressure = reference_pressure;
        auto mesh = make_single_cell_mesh();
        RadiolyticModelType model(mesh, options);
        FieldType temperature(mesh, temperature_value, "temperature");
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
        return std::tuple{
            model_field(model, "C_H2_crit").value(0),
            model.large_number_density().value(0),
            model_field(model, "micro_to_large_number_rate").value(0)};
    };

    const auto [low_c_critical, low_large_number, low_rate] =
        run_case(low_pressure);
    const auto [high_c_critical, high_large_number, high_rate] =
        run_case(high_pressure);

    EXPECT_GT(high_c_critical, low_c_critical);
    EXPECT_GT(low_large_number, high_large_number);
    EXPECT_GT(low_rate, high_rate);
}

/**
 * @brief Constant and flat prescribed-pressure histories produce the same state.
 */
TEST(RadiolyticGasModelTest, ConstantAndPrescribedPressureModesRegressTogether)
{
    auto base_options = sheng_options();
    base_options.initial_micro_number_density = 1.0e8;
    base_options.initial_micro_moles = 1.0e-8;
    base_options.initial_dissolved_hydrogen = 1.0e3;
    base_options.microbubble_lifetime = 1.0e9;
    base_options.large_bubble_dissolution_time = 1.0e9;
    base_options.micro_to_large_conversion_coefficient = 1.0e-8;

    auto run_case = [&](SimpleFluid::RadiolyticPressureMode pressure_mode)
    {
        auto options = base_options;
        options.pressure_mode = pressure_mode;
        if (pressure_mode
            == SimpleFluid::RadiolyticPressureMode::PrescribedHistory)
        {
            options.pressure_history_times = {0.0, 1.0};
            options.pressure_history_values = {
                options.reference_pressure,
                options.reference_pressure};
        }
        auto mesh = make_single_cell_mesh();
        RadiolyticModelType model(mesh, options);
        FieldType temperature(mesh, 300.0, "temperature");
        FieldType pressure(mesh, 0.0, "pressure");
        FieldType power(mesh, 0.0, "qdot_fission");
        VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
        FaceFieldType flux(mesh, 0.0, "flux");
        auto material = make_water_properties(mesh);
        model.advance(
            0.25,
            1.0e-7,
            temperature,
            pressure,
            velocity,
            flux,
            material,
            &power);
        return std::array<double, 4>{
            model.absolute_pressure().value(0),
            model_field(model, "C_H2_crit").value(0),
            model.alpha_g().value(0),
            model.dissolved_hydrogen_inventory().value(0)
              + model.micro_moles().value(0)
              + model.large_moles().value(0)};
    };

    const auto constant =
        run_case(SimpleFluid::RadiolyticPressureMode::Constant);
    const auto prescribed =
        run_case(SimpleFluid::RadiolyticPressureMode::PrescribedHistory);
    for (size_t index = 0; index < constant.size(); ++index)
    {
        EXPECT_NEAR(
            prescribed[index],
            constant[index],
            std::max(1.0e-12, std::abs(constant[index]) * 1.0e-12));
    }
}

/**
 * @brief Constant and inertial pressure modes run from one case definition.
 */
TEST(RadiolyticGasModelTest, ConstantAndInertialPressureModesRunFromSameCase)
{
    auto base_options = sheng_options();
    base_options.initial_large_number_density = 1.0e6;
    base_options.initial_large_moles = 1.0e-7;
    base_options.initial_dissolved_hydrogen = 5.0;
    base_options.microbubble_lifetime = 1.0e-4;
    base_options.large_bubble_dissolution_time = 1.0e-4;
    base_options.micro_to_large_conversion_coefficient = 1.0e-8;
    base_options.liquid_thermal_expansion = 2.0e-4;

    auto run_case = [&](SimpleFluid::RadiolyticPressureMode pressure_mode)
    {
        auto options = base_options;
        options.pressure_mode = pressure_mode;
        auto mesh = make_single_cell_mesh();
        RadiolyticModelType model(mesh, options);
        FieldType temperature(mesh, 300.0, "temperature");
        FieldType pressure(mesh, 0.0, "pressure");
        FieldType power(mesh, 2.0e4, "qdot_fission");
        VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
        FaceFieldType flux(mesh, 0.0, "flux");
        auto material = make_water_properties(mesh);
        for (int step = 0; step < 2; ++step)
        {
            temperature.put_scalar(300.0 + 0.5 * step);
            ASSERT_NO_FATAL_FAILURE(advance_model(
                model,
                mesh,
                (step + 1) * 1.0e-5,
                1.0e-5,
                temperature,
                pressure,
                power,
                velocity,
                flux,
                material));
        }
        expect_state_fields_finite_bounded(model);
        EXPECT_TRUE(std::isfinite(model.absolute_pressure().value(0)));
        EXPECT_GE(
            model.absolute_pressure().value(0),
            options.minimum_absolute_pressure);
    };

    run_case(SimpleFluid::RadiolyticPressureMode::Constant);
    run_case(SimpleFluid::RadiolyticPressureMode::Inertial);
}

/**
 * @brief More local subcycles move stiff kinetics toward the fine solution.
 */
TEST(RadiolyticGasModelTest, StiffLocalKineticsConvergesWithSubcycles)
{
    auto run_case = [](int max_subcycles)
    {
        auto options = sheng_options();
        options.initial_dissolved_hydrogen = 1.0e3;
        options.initial_micro_number_density = 1.0e8;
        options.initial_micro_moles = 1.0e-8;
        options.initial_large_number_density = 1.0e6;
        options.initial_large_moles = 1.0e-7;
        options.microbubble_lifetime = 1.0e-7;
        options.large_bubble_dissolution_time = 1.0e-7;
        options.micro_to_large_conversion_coefficient = 1.0e-8;
        options.max_subcycles = max_subcycles;
        auto mesh = make_single_cell_mesh();
        RadiolyticModelType model(mesh, options);
        FieldType temperature(mesh, 300.0, "temperature");
        FieldType pressure(mesh, 0.0, "pressure");
        FieldType power(mesh, 1.0e5, "qdot_fission");
        VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
        FaceFieldType flux(mesh, 0.0, "flux");
        auto material = make_water_properties(mesh);
        model.advance(
            2.0e-6,
            2.0e-6,
            temperature,
            pressure,
            velocity,
            flux,
            material,
            &power);
        return std::array<double, 4>{
            model.dissolved_hydrogen_inventory().value(0),
            model.micro_moles().value(0),
            model.large_moles().value(0),
            model.alpha_g().value(0)};
    };
    auto error = [](const auto& value, const auto& reference)
    {
        double result = 0.0;
        for (size_t index = 0; index < value.size(); ++index)
        {
            result += std::abs(value[index] - reference[index]);
        }
        return result;
    };

    const auto coarse = run_case(1);
    const auto medium = run_case(5);
    const auto fine = run_case(100);
    const auto coarse_error = error(coarse, fine);
    const auto medium_error = error(medium, fine);

    EXPECT_GT(coarse_error, 0.0);
    EXPECT_LT(medium_error, coarse_error);
}

/**
 * @brief Concentration and void clipping report excluded inventory fields.
 */
TEST(RadiolyticGasModelTest, ClippedConcentrationAndVoidBoundsReportExcludedInventory)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.alpha_max = 0.01;
    options.max_concentration = 1.0;
    options.initial_dissolved_hydrogen = 5.0;
    options.initial_large_number_density = 1.0e12;
    options.initial_large_moles = 1.0;
    options.microbubble_lifetime = 1.0e9;
    options.large_bubble_dissolution_time = 1.0e9;
    options.micro_to_large_conversion_coefficient = 0.0;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    model.advance(
        1.0e-9,
        1.0e-9,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    EXPECT_DOUBLE_EQ(model.alpha_g().value(0), options.alpha_max);
    EXPECT_GT(model_field(model, "alpha_g_excess").value(0), 0.0);
    EXPECT_LE(model_field(model, "C_H2").value(0), options.max_concentration);
    EXPECT_GT(model_field(model, "I_H2_excluded").value(0), 0.0);
}

/**
 * @brief All pressure modes keep published state finite and bounded.
 */
TEST(RadiolyticGasModelTest, StateFieldsRemainFiniteNonNegativeAndBoundedAcrossPressureModes)
{
    const std::array<SimpleFluid::RadiolyticPressureMode, 4> modes{
        SimpleFluid::RadiolyticPressureMode::Constant,
        SimpleFluid::RadiolyticPressureMode::PrescribedHistory,
        SimpleFluid::RadiolyticPressureMode::Reconstructed,
        SimpleFluid::RadiolyticPressureMode::Inertial};

    for (const auto mode : modes)
    {
        auto mesh = make_single_cell_mesh();
        auto options = sheng_options();
        options.pressure_mode = mode;
        options.initial_dissolved_hydrogen = 1.0e3;
        options.initial_micro_number_density = 1.0e8;
        options.initial_micro_moles = 1.0e-8;
        options.initial_large_number_density = 1.0e6;
        options.initial_large_moles = 1.0e-7;
        options.liquid_thermal_expansion = 2.0e-4;
        if (mode == SimpleFluid::RadiolyticPressureMode::PrescribedHistory)
        {
            options.pressure_history_times = {0.0, 1.0};
            options.pressure_history_values = {1.0e5, 1.2e5};
        }
        RadiolyticModelType model(mesh, options);
        FieldType temperature(mesh, 300.0, "temperature");
        FieldType pressure(mesh, 1.0e4, "pressure");
        FieldType power(mesh, 1.0e5, "qdot_fission");
        VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
        FaceFieldType flux(mesh, 0.0, "flux");
        auto material = make_water_properties(mesh);

        for (int step = 0; step < 3; ++step)
        {
            temperature.put_scalar(300.0 + step);
            model.advance(
                (step + 1) * 1.0e-6,
                1.0e-6,
                temperature,
                pressure,
                velocity,
                flux,
                material,
                &power);
            expect_state_fields_finite_bounded(model);
        }
    }
}

/**
 * @brief BoussinesqSolver owns, publishes, and removes the optional gas model.
 */
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
