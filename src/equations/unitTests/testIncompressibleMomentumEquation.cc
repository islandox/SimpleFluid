/**
 * @file testIncompressibleMomentumEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for generic incompressible momentum transport.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/BoundaryConditions.hh"
#include "equations/IncompressibleMomentumEquation.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_single_hex_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
}

Pack::scalar_type local_matrix_entry(
    const Pack::matrix_type& matrix,
    MeshType::local_ordinal_type row,
    MeshType::local_ordinal_type column)
{
    const auto row_entries = matrix.getNumEntriesInLocalRow(row);
    typename Pack::matrix_type::nonconst_local_inds_host_view_type columns(
        "columns", row_entries);
    typename Pack::matrix_type::nonconst_values_host_view_type values(
        "values", row_entries);
    size_t num_entries = 0;
    matrix.getLocalRowCopy(row, columns, values, num_entries);

    Pack::scalar_type entry = 0.0;
    for (size_t i = 0; i < num_entries; ++i)
    {
        if (columns(i) == column)
        {
            entry += values(i);
        }
    }
    return entry;
}

} // namespace

/**
 * @brief Verify that a constant body-force source advances velocity
 *        linearly on a single-hex mesh with zero viscosity.
 */
TEST(IncompressibleMomentumEquationTest, AdvancesVelocityFromSource)
{
    auto mesh = make_single_hex_mesh();
    VectorFieldType velocity(mesh, "velocity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.1;
    options.kinematic_viscosity = 0.0;

    auto source =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {1.0, 2.0, 3.0};
    };

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    equation.advance_velocity(
        velocity, zero_fluxes, boundary_cache, options, velocity, source);

    EXPECT_NEAR(velocity.value(0).x, 0.1, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).y, 0.2, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).z, 0.3, 1.0e-12);
}

/**
 * @brief Verify the physical momentum advance with explicit density,
 *        dynamic viscosity, and acceleration on a single-hex mesh.
 */
TEST(IncompressibleMomentumEquationTest, AdvancesPhysicalMomentum)
{
    auto mesh = make_single_hex_mesh();
    VectorFieldType velocity(mesh, "velocity");
    FieldType dynamic_viscosity(mesh, 0.0, "dynamic_viscosity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.2;

    auto acceleration =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {-2.0, 1.0, 0.5};
    };

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    equation.advance_velocity_physical(
        velocity, zero_fluxes, boundary_cache, options, dynamic_viscosity,
        10.0, velocity, acceleration);

    EXPECT_NEAR(velocity.value(0).x, -0.4, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).y, 0.2, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).z, 0.1, 1.0e-12);
}

/** @brief Verifies that slip boundaries add no diffusive momentum diagonal. */
TEST(IncompressibleMomentumEquationTest,
     SlipBoundariesDoNotAddDiffusiveMomentumDiagonal)
{
    auto mesh = make_single_hex_mesh();
    VectorFieldType velocity(
        mesh,
        VectorFieldType::vec_type{1.0, 2.0, 3.0},
        "slip_velocity");
    velocity.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet boundary_conditions;
    for (const auto* name :
         {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundary_conditions.velocity[name] =
            {SimpleFluid::BoundaryConditionType::Slip, {}};
    }
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.25;
    options.kinematic_viscosity = 10.0;
    const auto transient = mesh->cell_volume(0) / options.time_step;

    SimpleFluid::IncompressibleMomentumEquation<Pack> equation(mesh);
    const auto constant_viscosity_system = equation.assemble_system(
        velocity, zero_fluxes, boundary_cache, options);
    EXPECT_NEAR(
        local_matrix_entry(*constant_viscosity_system.matrix, 0, 0),
        transient,
        1.0e-12);

    FieldType dynamic_viscosity(mesh, 10.0, "dynamic_viscosity");
    auto zero_acceleration =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {};
    };
    const auto physical_viscosity_system =
        equation.assemble_physical_system(
            velocity, zero_fluxes, boundary_cache, options,
            dynamic_viscosity, 1.0, zero_acceleration);
    EXPECT_NEAR(
        local_matrix_entry(*physical_viscosity_system.matrix, 0, 0),
        transient,
        1.0e-12);
}
