/**
 * @file testIncompressibleMomentumEquation.cc
 * @brief Unit tests for generic incompressible momentum transport.
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

} // namespace

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
