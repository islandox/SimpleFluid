/**
 * @file testFluidSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for the reusable incompressible fluid solver.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
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

SimpleFluid::SP<MeshType> make_two_cell_line_mesh()
{
    auto database = std::make_shared<SimpleFluid::Database>();
    database->set("dimension", 3);
    database->set("mesh_size", SimpleFluid::real_t{0.5});
    database->set(
        "domain_type",
        static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    database->set("X", SimpleFluid::ArrReal{0.0, 0.5, 1.0});
    database->set("Y", SimpleFluid::ArrReal{0.0, 1.0});
    database->set("Z", SimpleFluid::ArrReal{0.0, 1.0});
    database->set(
        "domain_exterior_face_types",
        SimpleFluid::ArrString{
            "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    return SimpleFluid::MeshFactory(database).template build<Pack>();
}

class TestFluidSolver : public SimpleFluid::FluidSolver<Pack>
{
public:
    using SimpleFluid::FluidSolver<Pack>::FluidSolver;

    SimpleFluid::LinearSolveSummary advance_momentum_once()
    {
        return advance_momentum();
    }
};

class WaterPressureFluidSolver : public SimpleFluid::FluidSolver<Pack>
{
public:
    using SimpleFluid::FluidSolver<Pack>::FluidSolver;

protected:
    Pack::scalar_type pressure_reference_density() const noexcept override
    {
        return 1000.0;
    }
};

} // namespace

static_assert(std::is_base_of_v<
              SimpleFluid::FluidSolver<Pack>,
              SimpleFluid::BoussinesqSolver<Pack>>);

/**
 * @brief Verify that FluidSolver advances pressure and velocity over two
 *        time steps on a single-cell mesh with zero viscosity.
 */
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

/**
 * @brief Confirm that the momentum predictor uses the previous pressure
 *        gradient, producing a linear velocity ramp on a two-cell mesh.
 */
TEST(FluidSolverTest, MomentumPredictorIncludesOldPressureGradient)
{
    auto mesh = make_two_cell_line_mesh();

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.kinematic_viscosity = 0.0;

    TestFluidSolver solver(mesh, {}, time_options);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        solver.pressure().set_value(
            cell_lid, mesh->cell_centroid(cell_lid).x);
    }
    mesh->sync_periodic_boundaries(solver.pressure());

    const auto summary = solver.advance_momentum_once();

    EXPECT_TRUE(summary.converged);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto velocity = solver.velocity().value(cell_lid);
        EXPECT_NEAR(velocity.x, -time_options.time_step, 1.0e-12);
        EXPECT_NEAR(velocity.y, 0.0, 1.0e-12);
        EXPECT_NEAR(velocity.z, 0.0, 1.0e-12);
    }
}

TEST(FluidSolverTest,
     SegregatedModesHonorPhysicalDirichletPressureBoundary)
{
    for (const auto coupling : {
             SimpleFluid::PressureVelocityCoupling::SIMPLE,
             SimpleFluid::PressureVelocityCoupling::PISO,
             SimpleFluid::PressureVelocityCoupling::PIMPLE})
    {
        auto mesh = make_single_cell_mesh();
        SimpleFluid::BoundaryConditionSet bcs;
        bcs.pressure["xmax"] = {
            SimpleFluid::BoundaryConditionType::Dirichlet, 1000.0};
        bcs.velocity["xmax"] = {
            SimpleFluid::BoundaryConditionType::Neumann, {}};
        for (const auto* name :
             {"xmin", "ymin", "ymax", "zmin", "zmax"})
        {
            bcs.velocity[name] = {
                SimpleFluid::BoundaryConditionType::NoSlip, {}};
        }

        SimpleFluid::TimeStepperOptions time_options;
        time_options.time_step = 0.1;
        time_options.kinematic_viscosity = 0.0;
        time_options.pressure_velocity_coupling = coupling;
        time_options.n_pressure_correctors = 2;
        time_options.n_outer_correctors = 1;

        SimpleFluid::LinearSolverOptions linear_options;
        linear_options.tolerance = 1.0e-12;
        WaterPressureFluidSolver solver(
            mesh, bcs, time_options, linear_options);
        solver.step();

        const auto corrections =
            coupling == SimpleFluid::PressureVelocityCoupling::SIMPLE
          ? 1
          : time_options.n_pressure_correctors;
        const auto remaining_fraction =
            std::pow(0.5, corrections);
        EXPECT_NEAR(
            solver.pressure().value(0),
            1000.0 * (1.0 - remaining_fraction),
            1.0e-8);
        const auto velocity = solver.velocity().value(0);
        EXPECT_NEAR(
            velocity.x, -0.1 * remaining_fraction, 1.0e-10);
        EXPECT_NEAR(velocity.y, 0.0, 1.0e-12);
        EXPECT_NEAR(velocity.z, 0.0, 1.0e-12);
        EXPECT_NEAR(
            solver.last_pressure_velocity_residuals().continuity,
            0.0,
            1.0e-12);
        EXPECT_TRUE(solver.last_step_statistics().converged);
    }
}

TEST(FluidSolverTest,
     CoupledKrylovHonorsDirichletPressureOutlet)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::BoundaryConditionSet bcs;
    bcs.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.velocity["xmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann, {}};
    for (const auto* name :
         {"xmin", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.1;
    time_options.kinematic_viscosity = 0.0;
    time_options.pressure_velocity_coupling =
        SimpleFluid::PressureVelocityCoupling::CoupledKrylov;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;
    linear_options.max_iterations = 200;
    SimpleFluid::FluidSolver<Pack> solver(
        mesh, bcs, time_options, linear_options);
    solver.step();

    EXPECT_TRUE(solver.last_step_statistics().converged);
    EXPECT_NEAR(solver.pressure().value(0), 1.0, 1.0e-10);
    const auto velocity = solver.velocity().value(0);
    EXPECT_NEAR(velocity.x, 0.0, 1.0e-10);
    EXPECT_NEAR(velocity.y, 0.0, 1.0e-10);
    EXPECT_NEAR(velocity.z, 0.0, 1.0e-10);
    EXPECT_NEAR(
        solver.last_pressure_velocity_residuals().continuity,
        0.0,
        1.0e-10);
}

/**
 * @brief Ensure VTU output contains pressure and velocity but excludes
 *        temperature when no thermal model is active.
 */
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

/**
 * @brief Validate a single coupled-Krylov pressure-velocity step on a
 *        single-cell mesh with zero viscosity.
 */
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
