/**
 * @file testFluidSolverMultiRank.cc
 * @brief MPI tests for globally reduced pressure-velocity residual norms.
 */

#include <gtest/gtest.h>

#include "FVM/FaceFlux.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "solvers/FluidSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using VelocityFieldType = SimpleFluid::VectorCellField<Pack>;
using FaceFieldType = SimpleFluid::FaceField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class ExposedFluidSolver : public SimpleFluid::FluidSolver<Pack>
{
public:
    using SimpleFluid::FluidSolver<Pack>::FluidSolver;

    Pack::scalar_type update_norm(
        const VelocityFieldType& before,
        const VelocityFieldType& after) const
    {
        return velocity_update_norm(before, after);
    }
};

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

SimpleFluid::SP<MeshType> distributed_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(8, 8, 1, 0.125));
}

SimpleFluid::BoundaryConditionSet cavity_boundary_conditions()
{
    SimpleFluid::BoundaryConditionSet conditions;
    for (const auto* name : {"xmin", "xmax", "ymin"})
    {
        conditions.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    conditions.velocity["ymax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet,
        {1.0, 0.0, 0.0}};
    conditions.velocity["zmin"] = {
        SimpleFluid::BoundaryConditionType::Slip, {}};
    conditions.velocity["zmax"] = {
        SimpleFluid::BoundaryConditionType::Slip, {}};
    return conditions;
}

} // namespace

TEST(FluidSolverMultiRankTest, VelocityUpdateNormIsGlobal)
{
    auto mesh = distributed_mesh();
    if (mesh->owned_cell_map()->getComm()->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    ExposedFluidSolver solver(mesh, {});
    VelocityFieldType before(
        mesh, MeshType::Vec3{}, "velocity_before");
    VelocityFieldType after(
        mesh, MeshType::Vec3{}, "velocity_after");

    double local_squared_norm = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        const auto cell_gid =
            mesh->owned_cell_map()->getGlobalElement(cell_lid);
        const auto scale = static_cast<double>(cell_gid + 1);
        const MeshType::Vec3 value{
            scale, -0.5 * scale, 2.0 * scale};
        after.set_value(cell_lid, value);
        local_squared_norm +=
            value.dot(value) * mesh->cell_volume(cell_lid);
    }

    const auto expected =
        std::sqrt(global_sum(*mesh, local_squared_norm));
    ASSERT_GT(expected, 0.0);
    EXPECT_NEAR(
        solver.update_norm(before, after),
        expected,
        std::max(1.0e-14, expected * 1.0e-12));
}

TEST(FluidSolverMultiRankTest, CoupledContinuityResidualIsGlobal)
{
    auto mesh = distributed_mesh();
    const auto communicator = mesh->owned_cell_map()->getComm();
    if (communicator->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }

    const auto boundary_conditions = cavity_boundary_conditions();
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 2.0e-3;
    time_options.kinematic_viscosity = 1.0e-2;
    time_options.pressure_velocity_coupling =
        SimpleFluid::PressureVelocityCoupling::CoupledKrylov;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 300;
    linear_options.tolerance = 1.0e-10;

    SimpleFluid::FluidSolver<Pack> solver(
        mesh, boundary_conditions, time_options, linear_options);
    solver.step();
    ASSERT_TRUE(solver.last_step_statistics().converged);

    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundary_conditions);
    FaceFieldType face_fluxes(mesh, "independent_coupled_flux");
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        solver.velocity(),
        solver.pressure(),
        time_options.time_step,
        boundary_cache,
        face_fluxes);

    double local_squared_norm = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(owned);
        const auto balance =
            SimpleFluid::FVM::cell_flux_balance<Pack>(
                *mesh, face_fluxes, cell_lid);
        local_squared_norm += balance * balance;
    }

    const auto global_squared_norm =
        global_sum(*mesh, local_squared_norm);
    ASSERT_GT(global_squared_norm, 1.0e-24)
        << "The fixture must exercise a nonzero continuity norm.";
    const auto expected = std::sqrt(global_squared_norm);
    const auto actual =
        solver.last_pressure_velocity_residuals().continuity;
    EXPECT_NEAR(
        actual,
        expected,
        std::max(1.0e-14, expected * 1.0e-12));
}
