/**
 * @file testRadiolyticGasModelMultiRank.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief MPI consistency tests for the radiolytic gas model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/RadiolyticGasModel.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>

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
 * @brief Sheng two-population options used by distributed inventory tests.
 *
 * @return Configured Sheng-model options.
 */
SimpleFluid::RadiolyticGasOptions sheng_options()
{
    SimpleFluid::RadiolyticGasOptions options;
    options.mode =
        SimpleFluid::RadiolyticGasMode::Sheng2024TwoPopulation;
    options.hydrogen_yield_mol_per_j = 2.0e-7;
    options.gas_release_efficiency = 1.0;
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
    return options;
}

/**
 * @brief Create water-like material fields on the distributed test mesh.
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
 * @brief Sum a scalar diagnostic over all ranks in the mesh communicator.
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
 * @brief Compute a distributed cell-volume integral for a cell field.
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
 * @brief Verify that a replicated diagnostic has the same value on every rank.
 *
 * @param mesh Mesh providing the communicator.
 * @param value Replicated value to compare.
 */
void expect_same_on_all_ranks(const MeshType& mesh, double value)
{
    double minimum = 0.0;
    double maximum = 0.0;
    Teuchos::reduceAll(
        *mesh.owned_cell_map()->getComm(),
        Teuchos::REDUCE_MIN,
        1,
        &value,
        &minimum);
    Teuchos::reduceAll(
        *mesh.owned_cell_map()->getComm(),
        Teuchos::REDUCE_MAX,
        1,
        &value,
        &maximum);
    EXPECT_NEAR(
        maximum,
        minimum,
        std::max(1.0e-14, std::abs(maximum) * 1.0e-12));
}

} // namespace

/**
 * @brief Two-rank radiolysis update conserves global H2 and void diagnostics.
 */
TEST(RadiolyticGasModelMultiRankTest, ConservesGlobalHydrogenAndVoidInventory)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    auto options = sheng_options();
    options.microbubble_lifetime = 1.0e9;
    options.large_bubble_dissolution_time = 1.0e9;
    options.micro_to_large_conversion_coefficient = 0.0;
    RadiolyticModelType model(mesh, options);

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    constexpr double power_density = 1.0e5;
    FieldType power(mesh, power_density, "qdot_fission");
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

    double local_volume = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        local_volume += mesh->cell_volume(
            static_cast<Pack::local_ordinal_type>(owned));
    }
    const auto total_volume = global_sum(*mesh, local_volume);
    const auto expected_produced =
        options.hydrogen_yield_mol_per_j
      * power_density
      * total_volume
      * time_step;
    const auto& statistics = model.last_statistics();

    EXPECT_NEAR(
        statistics.hydrogen_produced,
        expected_produced,
        expected_produced * 1.0e-10);
    EXPECT_NEAR(
        statistics.inventory_error,
        0.0,
        std::max(1.0e-14, expected_produced * 1.0e-10));
    EXPECT_TRUE(std::isfinite(statistics.void_volume));
    EXPECT_GE(statistics.void_volume, 0.0);
    EXPECT_NEAR(
        statistics.void_volume,
        global_integral(model.alpha_g()),
        std::max(1.0e-14, statistics.void_volume * 1.0e-10));

    expect_same_on_all_ranks(*mesh, statistics.hydrogen_produced);
    expect_same_on_all_ranks(*mesh, statistics.inventory_error);
    expect_same_on_all_ranks(*mesh, statistics.void_volume);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_TRUE(std::isfinite(model.alpha_g().value(cell_lid)));
        EXPECT_GE(model.alpha_g().value(cell_lid), options.alpha_min);
        EXPECT_LE(model.alpha_g().value(cell_lid), options.alpha_max);
        EXPECT_TRUE(std::isfinite(
            model.dissolved_hydrogen_inventory().value(cell_lid)));
        EXPECT_GE(model.dissolved_hydrogen_inventory().value(cell_lid), 0.0);
        EXPECT_TRUE(std::isfinite(model.micro_moles().value(cell_lid)));
        EXPECT_GE(model.micro_moles().value(cell_lid), 0.0);
        EXPECT_TRUE(std::isfinite(model.large_moles().value(cell_lid)));
        EXPECT_GE(model.large_moles().value(cell_lid), 0.0);
    }
}

/** @brief Verify finite-Courant free-surface escape is globally conservative. */
TEST(RadiolyticGasModelMultiRankTest,
     ConservesFiniteCourantFreeSurfaceEscape)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

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
    RadiolyticModelType model(mesh, options);

    FieldType temperature(mesh, 300.0, "temperature");
    FieldType pressure(mesh, 0.0, "pressure");
    FieldType power(mesh, 0.0, "qdot_fission");
    VelocityFieldType velocity(mesh, MeshType::Vec3{}, "velocity");
    FaceFieldType flux(mesh, 0.0, "flux");
    auto material = make_water_properties(mesh);

    const auto count_before =
        global_integral(model.micro_number_density())
      + global_integral(model.large_number_density());
    constexpr double time_step = 0.025;
    model.advance(
        time_step,
        time_step,
        temperature,
        pressure,
        velocity,
        flux,
        material,
        &power);

    const auto count_after =
        global_integral(model.micro_number_density())
      + global_integral(model.large_number_density());
    const auto& statistics = model.last_statistics();
    const auto& molar_rate =
        *model.output_fields().at("H2_escape_molar_rate");
    const auto& number_rate =
        *model.output_fields().at("bubble_escape_number_rate");

    EXPECT_GT(statistics.hydrogen_escaped, 0.0);
    EXPECT_GT(statistics.escaped_bubble_count, 0.0);
    EXPECT_NEAR(
        statistics.hydrogen_after + statistics.hydrogen_escaped,
        statistics.hydrogen_before + statistics.hydrogen_produced,
        1.0e-13);
    EXPECT_NEAR(statistics.inventory_error, 0.0, 1.0e-13);
    EXPECT_NEAR(
        count_after + statistics.escaped_bubble_count,
        count_before,
        count_before * 1.0e-10);
    EXPECT_NEAR(
        time_step * global_integral(molar_rate),
        statistics.hydrogen_escaped,
        1.0e-13);
    EXPECT_NEAR(
        time_step * global_integral(number_rate),
        statistics.escaped_bubble_count,
        count_before * 1.0e-10);

    expect_same_on_all_ranks(*mesh, statistics.hydrogen_escaped);
    expect_same_on_all_ranks(*mesh, statistics.escaped_bubble_count);
    expect_same_on_all_ranks(*mesh, statistics.inventory_error);
    expect_same_on_all_ranks(
        *mesh, statistics.cumulative_hydrogen_escaped);
    expect_same_on_all_ranks(
        *mesh, statistics.cumulative_escaped_bubble_count);
}

/** @brief Verify clipped-cell diagnostics are reduced over every rank. */
TEST(RadiolyticGasModelMultiRankTest, ReducesClippedCellCountGlobally)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

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

    const auto expected_clipped = static_cast<int>(
        mesh->owned_cell_map()->getGlobalNumElements());
    EXPECT_EQ(
        model.last_statistics().clipped_cells, expected_clipped);
    expect_same_on_all_ranks(
        *mesh,
        static_cast<double>(model.last_statistics().clipped_cells));
}
