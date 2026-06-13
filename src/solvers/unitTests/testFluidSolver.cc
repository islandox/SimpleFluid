/**
 * @file testFluidSolver.cc
 * @brief Tests for the reusable incompressible fluid solver.
 */

#include <gtest/gtest.h>

#include "geometry/MeshFactory.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/FluidSolver.hh"
#include "utils/testing_environment.hh"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_single_cell_mesh()
{
    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("dimension", 3);
    database->set("mesh_size", SimpleFluid::real_t{1.0});
    database->set(
        "domain_type",
        static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    database->set("X", SimpleFluid::ArrReal{0.0, 1.0});
    database->set("Y", SimpleFluid::ArrReal{0.0, 1.0});
    database->set("Z", SimpleFluid::ArrReal{0.0, 1.0});
    database->set(
        "domain_exterior_face_types",
        SimpleFluid::ArrString{
            "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    return SimpleFluid::MeshFactory(database).template build<Pack>();
}

} // namespace

static_assert(std::is_base_of_v<
              SimpleFluid::FluidSolver<Pack>,
              SimpleFluid::BoussinesqSolver<Pack>>);

TEST(FluidSolverTest, AdvancesPressureVelocityWithoutThermalFields)
{
    auto mesh = make_single_cell_mesh();

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.steps = 2;
    time_options.kinematic_viscosity = 0.0;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-13;

    SimpleFluid::FluidSolver<Pack> solver(
        mesh, {}, time_options, linear_options);
    solver.run();

    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_NEAR(solver.time(), 0.2, 1.0e-14);
    EXPECT_NEAR(solver.pressure().value(0), 0.0, 1.0e-12);
    const auto velocity = solver.velocity().value(0);
    EXPECT_NEAR(velocity.x, 0.0, 1.0e-12);
    EXPECT_NEAR(velocity.y, 0.0, 1.0e-12);
    EXPECT_NEAR(velocity.z, 0.0, 1.0e-12);

    const auto residuals = solver.last_pressure_velocity_residuals();
    const auto statistics = solver.last_step_statistics();
    EXPECT_TRUE(statistics.converged);
    EXPECT_NEAR(statistics.momentum, residuals.momentum, 1.0e-12);
    EXPECT_NEAR(statistics.pressure, residuals.pressure, 1.0e-12);
    EXPECT_NEAR(statistics.continuity, residuals.continuity, 1.0e-12);
}

TEST(FluidSolverTest, WritesOnlyCoreFluidFields)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::FluidSolver<Pack> solver(mesh, {});

    const auto unique_id =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output_file =
        std::filesystem::temp_directory_path()
        / ("SimpleFluid_testFluidSolver_"
           + std::to_string(unique_id) + ".vtu");
    solver.write_solution_vtu(output_file.string());

    std::ifstream input(output_file);
    ASSERT_TRUE(input.good());
    const std::string contents(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    EXPECT_NE(contents.find("Name=\"pressure\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"velocity\""), std::string::npos);
    EXPECT_EQ(contents.find("Name=\"temperature\""), std::string::npos);

    std::filesystem::remove(output_file);
}

TEST(FluidSolverTest, SupportsCoupledKrylovPressureVelocitySolve)
{
    auto mesh = make_single_cell_mesh();

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.kinematic_viscosity = 0.0;
    time_options.pressure_velocity_coupling =
        SimpleFluid::PressureVelocityCoupling::CoupledKrylov;

    SimpleFluid::FluidSolver<Pack> solver(mesh, {}, time_options);
    solver.step();

    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_TRUE(solver.last_step_statistics().converged);
    EXPECT_EQ(solver.last_step_statistics().nonlinear_iterations, 1);
    EXPECT_TRUE(std::isfinite(
        solver.last_pressure_velocity_residuals().continuity));
}
