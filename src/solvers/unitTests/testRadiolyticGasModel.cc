/**
 * @file testRadiolyticGasModel.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for ideal and Sheng-style radiolytic gas models.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/RadiolyticGasModel.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
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
#include <limits>
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
 *
 * @return Assembled single-cell mesh.
 */
SimpleFluid::SP<MeshType> make_single_cell_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
}

/**
 * @brief Create water-like material fields for the radiolysis tests.
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

/**
 * @brief Baseline ideal-gas radiolysis options with bounded alpha source.
 *
 * @return Configured ideal-gas source options.
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
 *
 * @return Configured Sheng-model options.
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
 *
 * @param mesh Mesh providing the communicator.
 * @param local_value Local rank contribution.
 * @return Global sum across ranks.
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
 *
 * @param field Cell field to integrate.
 * @return Global volume integral.
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
 *
 * @param model Model publishing diagnostic fields.
 * @param name Requested field name.
 * @return Published field reference.
 * @throws std::runtime_error if the field is absent or null.
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
 *
 * @param field Field to inspect.
 * @param name Diagnostic name included in assertion messages.
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
 *
 * @param field Field to inspect.
 * @param name Diagnostic name included in assertion messages.
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
 *
 * @param model Model whose published state is inspected.
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
 *
 * @param options Radiolysis correlation options.
 * @param pressure Absolute pressure.
 * @param temperature Liquid temperature.
 * @return Critical dissolved-hydrogen concentration.
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
 *
 * @param model Model to advance.
 * @param mesh Mesh defining owned cells.
 * @param time End-of-step time.
 * @param time_step Physical time-step size.
 * @param temperature Temperature field.
 * @param pressure Gauge-pressure field.
 * @param power Fission power-density field.
 * @param velocity Liquid velocity field.
 * @param flux Liquid face-flux field.
 * @param material Material-property fields.
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
 * @brief Initial Sheng inventories publish derived void without a fake source.
 */
TEST(RadiolyticGasModelTest,
     InitialPopulationsAreReconstructedBeforeFirstKinetics)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_dissolved_hydrogen = 2.5;
    options.initial_micro_number_density = 1.0e10;
    options.initial_micro_moles = 1.0e-3;
    options.microbubble_lifetime = 1.0e30;
    options.micro_to_large_conversion_coefficient = 0.0;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    ASSERT_FALSE(model.initial_state_initialized());
    model.initialize_state(
        0.0, temperature, pressure, velocity, material);

    ASSERT_TRUE(model.initial_state_initialized());
    const auto micro_radius = model_field(model, "r_micro").value(0);
    const auto expected_raw_void =
        SimpleFluid::RadiolyticGasPhysics::bubble_void_fraction(
            options.initial_micro_number_density, micro_radius);
    const auto expected_void = std::clamp(
        expected_raw_void, options.alpha_min, options.alpha_max);
    ASSERT_GT(micro_radius, 0.0);
    ASSERT_GT(expected_void, options.alpha_min);
    EXPECT_NEAR(model.alpha_g().value(0), expected_void, 1.0e-14);
    EXPECT_NEAR(model.alpha_l().value(0), 1.0 - expected_void, 1.0e-14);
    EXPECT_NEAR(
        model.dissolved_hydrogen().value(0),
        options.initial_dissolved_hydrogen,
        1.0e-14);
    EXPECT_NEAR(
        model.dissolved_hydrogen_inventory().value(0),
        (1.0 - expected_void) * options.initial_dissolved_hydrogen,
        1.0e-14);
    EXPECT_DOUBLE_EQ(model.source_alpha_rad().value(0), 0.0);

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
    EXPECT_NEAR(model.alpha_g().value(0), expected_void, 1.0e-14);
    EXPECT_NEAR(model.source_alpha_rad().value(0), 0.0, 1.0e-14);
}

/** @brief Global inventory accessors keep dissolved and bubble H2 distinct. */
TEST(RadiolyticGasModelTest, SubmergedInventoryAccessorsResolveBothBubblePopulations)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_dissolved_hydrogen = 2.5;
    options.initial_micro_number_density = 2.0e8;
    options.initial_micro_moles = 3.0e-6;
    options.initial_large_number_density = 4.0e6;
    options.initial_large_moles = 5.0e-6;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    auto material = make_water_properties(mesh);

    model.initialize_state(0.0, temperature, pressure, velocity, material);

    const auto dissolved = global_integral(model.dissolved_hydrogen_inventory());
    const auto micro = global_integral(model.micro_moles());
    const auto large = global_integral(model.large_moles());
    const auto micro_volume = global_integral(model_field(model, "alpha_g_micro"));
    const auto large_volume = global_integral(model_field(model, "alpha_g_large"));

    EXPECT_DOUBLE_EQ(model.global_dissolved_hydrogen_moles(), dissolved);
    EXPECT_DOUBLE_EQ(model.global_microbubble_hydrogen_moles(), micro);
    EXPECT_DOUBLE_EQ(model.global_large_bubble_hydrogen_moles(), large);
    EXPECT_DOUBLE_EQ(model.global_submerged_bubble_hydrogen_moles(), micro + large);
    EXPECT_DOUBLE_EQ(model.global_submerged_hydrogen_moles(), dissolved + micro + large);
    EXPECT_NEAR(model.global_submerged_bubble_volume(), micro_volume + large_volume, 1.0e-15);
}

/** @brief Candidate EOS volume responds to pressure and stored temperature. */
TEST(RadiolyticGasModelTest, CandidateSubmergedBubbleVolumeRespondsToPressureAndTemperature)
{
    auto options = sheng_options();
    options.initial_micro_number_density = 2.0e8;
    options.initial_micro_moles = 3.0e-6;
    options.initial_large_number_density = 4.0e6;
    options.initial_large_moles = 5.0e-6;

    const auto initialized_volume = [&](double temperature_value)
    {
        auto mesh = make_single_cell_mesh();
        RadiolyticModelType model(mesh, options);
        FieldType temperature(mesh, temperature_value, "temperature");
        FieldType pressure(mesh, 0.0, "pressure");
        VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
        auto material = make_water_properties(mesh);
        EXPECT_THROW(model.evaluate_submerged_bubble_volume(options.reference_pressure), std::logic_error);
        model.initialize_state(0.0, temperature, pressure, velocity, material);
        const auto accepted = model.global_submerged_bubble_volume();
        const auto evaluated = model.evaluate_submerged_bubble_volume(options.reference_pressure);
        EXPECT_NEAR(evaluated, accepted, accepted * 1.0e-11);
        return std::pair{evaluated, model.evaluate_submerged_bubble_volume(2.0 * options.reference_pressure)};
    };

    const auto [cold_volume, compressed_cold_volume] = initialized_volume(300.0);
    const auto [hot_volume, compressed_hot_volume] = initialized_volume(360.0);
    EXPECT_GT(cold_volume, compressed_cold_volume);
    EXPECT_GT(hot_volume, compressed_hot_volume);
    EXPECT_GT(hot_volume, cold_volume);
}

/** @brief Dissolved H2 is excluded from every submerged bubble-volume API. */
TEST(RadiolyticGasModelTest, DissolvedHydrogenDoesNotOccupyBubbleVolume)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_dissolved_hydrogen = 100.0;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    auto material = make_water_properties(mesh);

    model.initialize_state(0.0, temperature, pressure, velocity, material);

    EXPECT_GT(model.global_dissolved_hydrogen_moles(), 0.0);
    EXPECT_DOUBLE_EQ(model.global_submerged_bubble_hydrogen_moles(), 0.0);
    EXPECT_DOUBLE_EQ(model.global_submerged_bubble_volume(), 0.0);
    EXPECT_DOUBLE_EQ(model.evaluate_submerged_bubble_volume(options.reference_pressure), 0.0);
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

    const auto statistics = model.last_statistics();
    EXPECT_NEAR(
        statistics.hydrogen_produced,
        options.hydrogen_yield_mol_per_j * 1.0e5 * time_step,
        1.0e-15);
    EXPECT_DOUBLE_EQ(statistics.cumulative_hydrogen_produced, statistics.hydrogen_produced);
    EXPECT_DOUBLE_EQ(model.cumulative_hydrogen_produced(), statistics.cumulative_hydrogen_produced);
    EXPECT_DOUBLE_EQ(statistics.cumulative_dissolved_hydrogen_outflow, 0.0);
    EXPECT_NEAR(statistics.inventory_error, 0.0, 1.0e-14);
    EXPECT_GT(statistics.hydrogen_after, 0.0);
    EXPECT_GT(model.alpha_g().value(0), 0.0);
    EXPECT_GE(model.dissolved_hydrogen_inventory().value(0), 0.0);
    EXPECT_GE(model.micro_moles().value(0), 0.0);

    model.advance(2.0 * time_step, time_step, temperature, pressure, velocity, flux, material, &power);
    const auto second = model.last_statistics();
    EXPECT_DOUBLE_EQ(second.cumulative_hydrogen_produced, statistics.hydrogen_produced + second.hydrogen_produced);
    EXPECT_DOUBLE_EQ(model.cumulative_hydrogen_produced(), second.cumulative_hydrogen_produced);

    model.configure(options);
    EXPECT_DOUBLE_EQ(model.cumulative_hydrogen_produced(), 0.0);
    EXPECT_DOUBLE_EQ(model.cumulative_dissolved_hydrogen_outflow(), 0.0);
    EXPECT_DOUBLE_EQ(model.cumulative_submerged_bubble_hydrogen_escaped(), 0.0);
    EXPECT_DOUBLE_EQ(model.last_statistics().cumulative_hydrogen_produced, 0.0);
}

/** @brief Sheng transport runs directly on MeshHandle/FieldStored storage. */
TEST(RadiolyticGasModelTest, NativeTwoPopulationStepConservesHydrogen)
{
    using Handle = SimpleFluid::MeshHandle<Pack>;
    auto cartesian =
        std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
                {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<Handle>(std::move(cartesian));
    auto options = sheng_options();
    SimpleFluid::RadiolyticGasModel<Pack, Handle> model(mesh, options);
    SimpleFluid::ScalarCellFieldStored<Pack> temperature(
        mesh, 300.0, "temperature");
    SimpleFluid::ScalarCellFieldStored<Pack> pressure(
        mesh, 0.0, "pressure");
    SimpleFluid::ScalarCellFieldStored<Pack> power(
        mesh, 1.0e5, "qdot_fission");
    SimpleFluid::VectorCellFieldStored<Pack> velocity(
        mesh, Handle::Vec3{}, "velocity");
    SimpleFluid::ScalarFaceFieldStored<Pack> flux(
        mesh, 0.0, "flux");
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions material_options;
    material_options.reference_density = 1000.0;
    material_options.density = 1000.0;
    material_options.specific_heat_capacity = 4200.0;
    material_options.dynamic_viscosity = 1.0e-3;
    material_options.thermal_conductivity = 0.6;
    SimpleFluid::MaterialPropertyFields<Pack, Handle> material(
        mesh, material_options, time_options);

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
    EXPECT_FALSE(mesh->legacy_mesh());
    EXPECT_EQ(model.alpha_g().mesh_ptr(), mesh);
    EXPECT_NEAR(
        statistics.hydrogen_produced,
        options.hydrogen_yield_mol_per_j * 1.0e5 * time_step,
        1.0e-15);
    EXPECT_NEAR(statistics.inventory_error, 0.0, 1.0e-14);
    EXPECT_GT(statistics.hydrogen_after, 0.0);
    EXPECT_GT(model.alpha_g().value(0), 0.0);
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
 * @brief Reconstructed pressure uses the solver's gauge variation in Pa.
 */
TEST(RadiolyticGasModelTest,
     ReconstructedPressureUsesPascalGaugeVariation)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    ASSERT_EQ(mesh->num_owned_cells(), 2U);

    auto options = ideal_options();
    options.pressure_mode =
        SimpleFluid::RadiolyticPressureMode::Reconstructed;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    pressure.set_owned_value(0, -1000.0);
    pressure.set_owned_value(1, 1000.0);
    pressure.sync_ghosts();
    FieldType power(mesh, 0.0, "qdot_fission");
    FieldType authoritative_alpha(mesh, 0.0, "authoritative_alpha_g");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);
    material.initialize_density(800.0);
    ASSERT_DOUBLE_EQ(material.density.value(0), 800.0);

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
        options.alpha_max);

    EXPECT_NEAR(model.absolute_pressure().value(0), 9.9e4, 1.0e-10);
    EXPECT_NEAR(model.absolute_pressure().value(1), 1.01e5, 1.0e-10);
}

/** @brief Coupled pressure offsets preserve reconstructed gauge variations. */
TEST(RadiolyticGasModelTest, AbsolutePressureOffsetSetterShiftsPublishedPressure)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    auto options = ideal_options();
    options.pressure_mode = SimpleFluid::RadiolyticPressureMode::Reconstructed;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    pressure.set_owned_value(0, -1000.0);
    pressure.set_owned_value(1, 1000.0);
    pressure.sync_ghosts();
    FieldType power(mesh, 0.0, "qdot_fission");
    FieldType authoritative_alpha(mesh, 0.0, "authoritative_alpha_g");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);
    model.advance(
        0.1, 0.1, temperature, pressure, velocity, flux, material, &power, authoritative_alpha, options.alpha_max);

    ASSERT_DOUBLE_EQ(model.absolute_pressure_offset(), options.reference_pressure);
    const auto old_variation = model.absolute_pressure().value(1) - model.absolute_pressure().value(0);
    model.set_absolute_pressure_offset(2.0e5);
    EXPECT_DOUBLE_EQ(model.absolute_pressure_offset(), 2.0e5);
    EXPECT_NEAR(model.absolute_pressure().value(0), 1.99e5, 1.0e-10);
    EXPECT_NEAR(model.absolute_pressure().value(1), 2.01e5, 1.0e-10);
    EXPECT_DOUBLE_EQ(model.absolute_pressure().value(1) - model.absolute_pressure().value(0), old_variation);

    model.advance(
        0.2, 0.1, temperature, pressure, velocity, flux, material, &power, authoritative_alpha, options.alpha_max);
    EXPECT_NEAR(model.absolute_pressure().value(0), 1.99e5, 1.0e-10);
    EXPECT_NEAR(model.absolute_pressure().value(1), 2.01e5, 1.0e-10);
    EXPECT_THROW(model.set_absolute_pressure_offset(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(model.set_absolute_pressure_offset(0.5), std::invalid_argument);

    auto prescribed_options = ideal_options();
    prescribed_options.pressure_mode = SimpleFluid::RadiolyticPressureMode::PrescribedHistory;
    prescribed_options.pressure_history_times = {0.0, 1.0};
    prescribed_options.pressure_history_values = {1.0e5, 1.0e5};
    RadiolyticModelType prescribed(mesh, prescribed_options);
    EXPECT_THROW(prescribed.set_absolute_pressure_offset(2.0e5), std::logic_error);
    EXPECT_THROW(prescribed.minimum_valid_absolute_pressure_offset(), std::logic_error);

    auto inertial_options = ideal_options();
    inertial_options.pressure_mode = SimpleFluid::RadiolyticPressureMode::Inertial;
    RadiolyticModelType inertial(mesh, inertial_options);
    EXPECT_THROW(inertial.set_absolute_pressure_offset(2.0e5), std::logic_error);
    EXPECT_THROW(inertial.minimum_valid_absolute_pressure_offset(), std::logic_error);
}

/** @brief Accepted offsets rebuild every pressure-dependent bubble field. */
TEST(RadiolyticGasModelTest, ReconstructedOffsetRefreshesTwoPopulationDiagnostics)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    auto options = sheng_options();
    options.pressure_mode = SimpleFluid::RadiolyticPressureMode::Reconstructed;
    options.minimum_absolute_pressure = 5.0e4;
    options.alpha_max = 5.0e-5;
    options.initial_micro_number_density = 2.0e8;
    options.initial_micro_moles = 3.0e-6;
    options.initial_large_number_density = 4.0e6;
    options.initial_large_moles = 5.0e-6;
    options.rise_velocity_mode = SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    options.constant_slip_velocity = 0.1;
    options.free_surface_patches = {"zmax"};
    options.microbubble_lifetime = 1.0e30;
    options.large_bubble_dissolution_time = 1.0e30;
    options.micro_to_large_conversion_coefficient = 0.0;
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 330.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    pressure.set_owned_value(0, -1000.0);
    pressure.set_owned_value(1, 1000.0);
    pressure.sync_ghosts();
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);
    model.initialize_state(0.0, temperature, pressure, velocity, material);
    model.advance(0.01, 0.01, temperature, pressure, velocity, flux, material, &power);

    EXPECT_NEAR(model.minimum_valid_absolute_pressure_offset(), options.minimum_absolute_pressure + 1000.0, 1.0e-10);
    EXPECT_THROW(model.evaluate_submerged_bubble_volume(5.05e4), std::invalid_argument);

    const auto micro_moles_before = model.global_microbubble_hydrogen_moles();
    const auto large_moles_before = model.global_large_bubble_hydrogen_moles();
    const auto escaped_before = model.last_statistics().cumulative_hydrogen_escaped;
    const auto count_escaped_before = model.last_statistics().cumulative_escaped_bubble_count;
    const auto bubble_hydrogen_escaped_before = model.cumulative_submerged_bubble_hydrogen_escaped();
    ASSERT_GT(escaped_before, 0.0);
    ASSERT_GT(count_escaped_before, 0.0);
    ASSERT_GT(bubble_hydrogen_escaped_before, 0.0);
    const auto old_micro_radius = model_field(model, "r_micro").value(0);
    const auto old_large_radius = model_field(model, "r_large").value(0);
    const auto old_critical_concentration = model_field(model, "C_H2_crit").value(0);
    constexpr double accepted_offset = 2.0e5;
    const auto candidate_raw_volume = model.evaluate_submerged_bubble_volume(accepted_offset);

    model.set_absolute_pressure_offset(accepted_offset);

    EXPECT_DOUBLE_EQ(model.global_microbubble_hydrogen_moles(), micro_moles_before);
    EXPECT_DOUBLE_EQ(model.global_large_bubble_hydrogen_moles(), large_moles_before);
    EXPECT_DOUBLE_EQ(model.last_statistics().cumulative_hydrogen_escaped, escaped_before);
    EXPECT_DOUBLE_EQ(model.last_statistics().cumulative_escaped_bubble_count, count_escaped_before);
    EXPECT_DOUBLE_EQ(model.cumulative_submerged_bubble_hydrogen_escaped(), bubble_hydrogen_escaped_before);
    EXPECT_NEAR(model.global_submerged_bubble_volume(), candidate_raw_volume, candidate_raw_volume * 1.0e-11);
    EXPECT_LT(model_field(model, "r_micro").value(0), old_micro_radius);
    EXPECT_LT(model_field(model, "r_large").value(0), old_large_radius);
    EXPECT_NE(model_field(model, "C_H2_crit").value(0), old_critical_concentration);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
        const auto raw = model_field(model, "alpha_g_raw").value(cell_lid);
        const auto component_sum =
            model_field(model, "alpha_g_micro").value(cell_lid) + model_field(model, "alpha_g_large").value(cell_lid);
        const auto bounded = model.alpha_g().value(cell_lid);
        EXPECT_NEAR(raw, component_sum, std::abs(raw) * 1.0e-13);
        EXPECT_DOUBLE_EQ(bounded, std::clamp(raw, options.alpha_min, options.alpha_max));
        EXPECT_DOUBLE_EQ(model.alpha_l().value(cell_lid), 1.0 - bounded);
        EXPECT_DOUBLE_EQ(model_field(model, "alpha_g_excess").value(cell_lid), std::max(raw - bounded, 0.0));
    }
    EXPECT_NEAR(model.last_statistics().void_volume, global_integral(model.alpha_g()), 1.0e-15);
}

/**
 * @brief Finite-Courant escape closes step and cumulative inventories.
 */
TEST(RadiolyticGasModelTest, FreeSurfaceEscapeAccumulatesGlobally)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_micro_number_density = 1.0e10;
    options.initial_micro_moles = 1.0e-6;
    options.rise_velocity_mode =
        SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    options.constant_slip_velocity = 10.0;
    options.free_surface_patches = {"zmax"};
    options.microbubble_lifetime = 1.0e30;
    options.large_bubble_dissolution_time = 1.0e30;
    options.micro_to_large_conversion_coefficient = 0.0;
    SimpleFluid::RadiolyticGasModel<Pack> model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    constexpr double time_step = 0.1;
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
    EXPECT_NEAR(first.hydrogen_escaped, 5.0e-7, 1.0e-14);
    EXPECT_DOUBLE_EQ(first.dissolved_hydrogen_outflow, 0.0);
    EXPECT_NEAR(first.microbubble_hydrogen_escaped, first.hydrogen_escaped, 1.0e-14);
    EXPECT_DOUBLE_EQ(first.large_bubble_hydrogen_escaped, 0.0);
    EXPECT_NEAR(first.submerged_bubble_hydrogen_escaped, first.hydrogen_escaped, 1.0e-14);
    EXPECT_NEAR(first.escaped_bubble_count, 5.0e9, 1.0e-2);
    EXPECT_NEAR(first.escaped_microbubble_count, first.escaped_bubble_count, 1.0e-2);
    EXPECT_DOUBLE_EQ(first.escaped_large_bubble_count, 0.0);
    EXPECT_NEAR(
        first.hydrogen_after + first.hydrogen_escaped,
        first.hydrogen_before + first.hydrogen_produced,
        1.0e-14);
    EXPECT_NEAR(first.inventory_error, 0.0, 1.0e-14);
    EXPECT_NEAR(
        time_step
          * global_integral(
              model_field(model, "H2_escape_molar_rate")),
        first.hydrogen_escaped,
        1.0e-14);
    EXPECT_NEAR(
        time_step
          * global_integral(
              model_field(model, "bubble_escape_number_rate")),
        first.escaped_bubble_count,
        1.0e-2);
    EXPECT_DOUBLE_EQ(
        first.cumulative_hydrogen_escaped,
        first.hydrogen_escaped);
    EXPECT_DOUBLE_EQ(first.cumulative_submerged_bubble_hydrogen_escaped, first.submerged_bubble_hydrogen_escaped);
    EXPECT_DOUBLE_EQ(first.cumulative_hydrogen_escaped,
        first.cumulative_dissolved_hydrogen_outflow + first.cumulative_submerged_bubble_hydrogen_escaped);
    EXPECT_DOUBLE_EQ(model.cumulative_submerged_bubble_hydrogen_escaped(), first.submerged_bubble_hydrogen_escaped);

    model.advance(
        2.0 * time_step,
        time_step,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);
    const auto second = model.last_statistics();
    EXPECT_LT(model.source_alpha_rad().value(0), 0.0);
    EXPECT_NEAR(
        second.cumulative_hydrogen_escaped,
        first.cumulative_hydrogen_escaped + second.hydrogen_escaped,
        1.0e-14);
    EXPECT_NEAR(
        second.cumulative_escaped_bubble_count,
        first.cumulative_escaped_bubble_count
          + second.escaped_bubble_count,
        1.0e-2);
    EXPECT_DOUBLE_EQ(second.cumulative_submerged_bubble_hydrogen_escaped,
        first.submerged_bubble_hydrogen_escaped + second.submerged_bubble_hydrogen_escaped);
    EXPECT_DOUBLE_EQ(
        model.cumulative_submerged_bubble_hydrogen_escaped(), second.cumulative_submerged_bubble_hydrogen_escaped);
}

/** @brief Dissolved boundary outflow is not reported as escaped bubbles. */
TEST(RadiolyticGasModelTest, DissolvedBoundaryOutflowIsSeparateFromBubbleEscape)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_dissolved_hydrogen = 1.0e-3;
    options.dissolved_transport = SimpleFluid::RadiolyticTransportMode::Advective;
    options.free_surface_patches = {"zmax"};
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        if (flux.is_owned_face(face_lid) && mesh->is_boundary_face(face_lid) &&
            mesh->boundary_batch_name(mesh->boundary_id(face_lid)) == "zmax")
        {
            flux.set_value(face_lid, 1.0);
        }
    }
    auto material = make_water_properties(mesh);

    model.advance(0.1, 0.1, temperature, pressure, velocity, flux, material, &power);

    const auto& statistics = model.last_statistics();
    EXPECT_GT(statistics.dissolved_hydrogen_outflow, 0.0);
    EXPECT_DOUBLE_EQ(statistics.microbubble_hydrogen_escaped, 0.0);
    EXPECT_DOUBLE_EQ(statistics.large_bubble_hydrogen_escaped, 0.0);
    EXPECT_DOUBLE_EQ(statistics.submerged_bubble_hydrogen_escaped, 0.0);
    EXPECT_DOUBLE_EQ(statistics.escaped_bubble_count, 0.0);
    EXPECT_DOUBLE_EQ(statistics.hydrogen_escaped, statistics.dissolved_hydrogen_outflow);
    EXPECT_DOUBLE_EQ(statistics.cumulative_dissolved_hydrogen_outflow, statistics.dissolved_hydrogen_outflow);
    EXPECT_DOUBLE_EQ(model.cumulative_dissolved_hydrogen_outflow(), statistics.dissolved_hydrogen_outflow);
    EXPECT_DOUBLE_EQ(statistics.cumulative_submerged_bubble_hydrogen_escaped, 0.0);
    EXPECT_DOUBLE_EQ(statistics.cumulative_hydrogen_escaped,
        statistics.cumulative_dissolved_hydrogen_outflow + statistics.cumulative_submerged_bubble_hydrogen_escaped);
    EXPECT_DOUBLE_EQ(model.cumulative_submerged_bubble_hydrogen_escaped(), 0.0);
    EXPECT_NEAR(statistics.inventory_error, 0.0, 1.0e-14);
}

/** @brief Positive escape below the old relative cutoff remains observable. */
TEST(RadiolyticGasModelTest, SmallPositiveEscapeIsNotDiscarded)
{
    auto mesh = make_single_cell_mesh();
    auto options = sheng_options();
    options.initial_dissolved_hydrogen = 1.0e9;
    options.max_concentration = 1.0e12;
    options.dissolved_transport = SimpleFluid::RadiolyticTransportMode::Advective;
    options.free_surface_patches = {"zmax"};
    RadiolyticModelType model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        if (flux.is_owned_face(face_lid) && mesh->is_boundary_face(face_lid) &&
            mesh->boundary_batch_name(mesh->boundary_id(face_lid)) == "zmax")
        {
            flux.set_value(face_lid, 1.0e-13);
        }
    }
    auto material = make_water_properties(mesh);

    model.advance(0.1, 0.1, temperature, pressure, velocity, flux, material, &power);

    const auto& statistics = model.last_statistics();
    const auto old_cutoff = 64.0 * std::numeric_limits<double>::epsilon() * statistics.hydrogen_before;
    EXPECT_GT(statistics.dissolved_hydrogen_outflow, 0.0);
    EXPECT_LT(statistics.dissolved_hydrogen_outflow, old_cutoff);
    EXPECT_DOUBLE_EQ(statistics.cumulative_dissolved_hydrogen_outflow, statistics.dissolved_hydrogen_outflow);
}

/** @brief Verify escape-rate diagnostics are localized to free-surface cells. */
TEST(RadiolyticGasModelTest, EscapeRateFieldsAreBoundaryLocalized)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(1, 1, 2));
    auto options = sheng_options();
    options.initial_micro_number_density = 1.0e10;
    options.initial_micro_moles = 1.0e-6;
    options.rise_velocity_mode =
        SimpleFluid::BubbleRiseVelocityMode::ConstantSlip;
    options.constant_slip_velocity = 1.0;
    options.free_surface_patches = {"zmax"};
    options.microbubble_lifetime = 1.0e30;
    options.large_bubble_dissolution_time = 1.0e30;
    options.micro_to_large_conversion_coefficient = 0.0;
    SimpleFluid::RadiolyticGasModel<Pack> model(mesh, options);
    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    constexpr double time_step = 0.1;
    model.advance(
        time_step,
        time_step,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    const auto& molar_rate =
        model_field(model, "H2_escape_molar_rate");
    const auto& number_rate =
        model_field(model, "bubble_escape_number_rate");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        if (mesh->cell_centroid(cell_lid).z > 1.0)
        {
            EXPECT_GT(molar_rate.value(cell_lid), 0.0);
            EXPECT_GT(number_rate.value(cell_lid), 0.0);
        }
        else
        {
            EXPECT_DOUBLE_EQ(molar_rate.value(cell_lid), 0.0);
            EXPECT_DOUBLE_EQ(number_rate.value(cell_lid), 0.0);
        }
    }
    EXPECT_NEAR(
        time_step * global_integral(molar_rate),
        model.last_statistics().hydrogen_escaped,
        1.0e-14);
    EXPECT_NEAR(
        time_step * global_integral(number_rate),
        model.last_statistics().escaped_bubble_count,
        1.0e-2);
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
    model.initialize_state(
        0.0, temperature, pressure, velocity, material);

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
    const auto bounded_volume = global_integral(model.alpha_g());
    const auto raw_volume = model.global_submerged_bubble_volume();
    const auto unrepresented_volume = model.global_unrepresented_bubble_volume();
    EXPECT_GT(raw_volume, bounded_volume);
    EXPECT_NEAR(unrepresented_volume, raw_volume - bounded_volume, raw_volume * 1.0e-12);
    EXPECT_NEAR(
        model.evaluate_submerged_bubble_volume(model.absolute_pressure_offset()), raw_volume, raw_volume * 1.0e-11);
    EXPECT_DOUBLE_EQ(model.global_large_bubble_hydrogen_moles(), global_integral(model.large_moles()));
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
 * @brief Solver startup publishes Sheng void before precursor initialization.
 */
TEST(RadiolyticGasModelTest,
     SolverPublishesInitialShengStateInEitherConfigurationOrder)
{
    auto options = sheng_options();
    options.initial_dissolved_hydrogen = 2.5;
    options.initial_micro_number_density = 1.0e10;
    options.initial_micro_moles = 1.0e-3;

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-6;
    time_options.reference_temperature = 300.0;
    auto model_options =
        SimpleFluid::BoussinesqModelOptions::legacy_defaults(
            time_options);
    model_options.reference_density = 1000.0;
    model_options.density = 1000.0;
    model_options.specific_heat_capacity = 4200.0;
    model_options.dynamic_viscosity = 1.0e-3;
    model_options.thermal_conductivity = 0.6;
    SimpleFluid::DelayedNeutronPrecursorOptions precursor_options;
    precursor_options.group_count = 1;
    precursor_options.initial_concentrations = {4.0};

    const auto expect_published =
        [&](const auto& radiolysis,
            const auto& scalar_void,
            const auto& precursors)
    {
        ASSERT_TRUE(radiolysis.initial_state_initialized());
        const auto alpha_g = radiolysis.alpha_g().value(0);
        ASSERT_GT(alpha_g, options.alpha_min);
        EXPECT_NEAR(
            scalar_void.alpha_g().value(0), alpha_g, 1.0e-14);
        EXPECT_NEAR(
            scalar_void.alpha_l().value(0), 1.0 - alpha_g, 1.0e-14);
        EXPECT_DOUBLE_EQ(
            radiolysis.source_alpha_rad().value(0), 0.0);
        EXPECT_DOUBLE_EQ(
            scalar_void.source_alpha_total().value(0), 0.0);
        EXPECT_NEAR(
            radiolysis.dissolved_hydrogen().value(0),
            options.initial_dissolved_hydrogen,
            1.0e-14);
        EXPECT_NEAR(
            radiolysis.dissolved_hydrogen_inventory().value(0),
            (1.0 - alpha_g) * options.initial_dissolved_hydrogen,
            1.0e-14);
        EXPECT_NEAR(
            precursors.concentration(0).value(0), 4.0, 1.0e-14);
        EXPECT_NEAR(
            precursors.liquid_inventory(0).value(0),
            (1.0 - alpha_g) * 4.0,
            1.0e-14);
    };

    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh, {}, time_options, {}, model_options);
        solver.initialize_heated_box(300.0, 300.0);
        const auto& radiolysis =
            solver.configure_radiolytic_gas(options);
        const auto& precursors =
            solver.configure_precursors(precursor_options);
        const auto* scalar_void =
            solver.find_scalar_void_fraction_model();
        ASSERT_NE(scalar_void, nullptr);
        expect_published(radiolysis, *scalar_void, precursors);
    }

    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoussinesqSolver<Pack> solver(
            mesh, {}, time_options, {}, model_options);
        const auto& radiolysis =
            solver.configure_radiolytic_gas(options);
        ASSERT_FALSE(radiolysis.initial_state_initialized());
        const auto& precursors =
            solver.configure_precursors(precursor_options);

        solver.initialize_heated_box(300.0, 300.0);

        const auto* scalar_void =
            solver.find_scalar_void_fraction_model();
        ASSERT_NE(scalar_void, nullptr);
        expect_published(radiolysis, *scalar_void, precursors);
    }
}

/**
 * @brief First temperature solve uses feedback from reconstructed initial void.
 */
TEST(RadiolyticGasModelTest,
     FirstPhysicalSolveUsesReconstructedShengVoidFeedback)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-3;
    time_options.reference_temperature = 300.0;
    time_options.gravity_x = 0.0;
    time_options.gravity_y = 0.0;
    time_options.gravity_z = 0.0;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = 1000.0;
    model_options.density = 1000.0;
    model_options.specific_heat_capacity = 1.0;
    model_options.dynamic_viscosity = 1.0e-3;
    model_options.thermal_conductivity = 0.0;
    SimpleFluid::BoussinesqSolver<Pack> solver(
        mesh, {}, time_options, {}, model_options);
    solver.initialize_heated_box(300.0, 300.0);

    SimpleFluid::FissionPowerSourceOptions fission;
    fission.profile = SimpleFluid::FissionPowerProfile::Constant;
    fission.power_density = 0.0;
    solver.configure_fission_power_source(fission);
    constexpr double power_density = 1.0e6;
    solver.add_temperature_source("startup_heating", power_density);

    auto options = sheng_options();
    options.initial_micro_number_density = 1.0e10;
    options.initial_micro_moles = 5.0;
    options.microbubble_lifetime = 1.0e30;
    options.micro_to_large_conversion_coefficient = 0.0;
    const auto& radiolysis =
        solver.configure_radiolytic_gas(options);
    const auto initial_alpha = radiolysis.alpha_g().value(0);
    ASSERT_GT(initial_alpha, 0.05);

    SimpleFluid::MaterialFeedbackOptions feedback;
    feedback.density_mode =
        SimpleFluid::DensityFeedbackMode::Mixture;
    feedback.reference_density = 1000.0;
    feedback.liquid_density = 1000.0;
    feedback.gas_density = 1.0;
    feedback.reference_dynamic_viscosity = 1.0e-3;
    solver.configure_material_feedback(feedback);

    solver.step();

    const auto initial_mixture_density =
        feedback.liquid_density * (1.0 - initial_alpha)
      + feedback.gas_density * initial_alpha;
    const auto expected_temperature =
        300.0
      + time_options.time_step * power_density
        / (initial_mixture_density
           * model_options.specific_heat_capacity);
    EXPECT_NEAR(
        solver.temperature().value(0), expected_temperature, 1.0e-10);
    EXPECT_GT(
        solver.temperature().value(0),
        300.0
          + time_options.time_step * power_density
            / (model_options.density
               * model_options.specific_heat_capacity));
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
