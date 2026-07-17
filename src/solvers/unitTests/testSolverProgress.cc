/**
 * @file testSolverProgress.cc
 * @brief Unit tests for member-based solver progress reporting.
 */

#include <gtest/gtest.h>

#include "geometry/MeshFactory.hh"
#include "solvers/BoussinesqSolver.hh"
#include "solvers/FluidSolver.hh"
#include "solvers/SolverProgress.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SimpleFluid::SP<SimpleFluid::Mesh<Pack>> make_single_cell_mesh()
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
    return SimpleFluid::MeshFactory(database).build<Pack>();
}

SimpleFluid::FluidSolver<Pack> make_solver(int configured_steps = 2)
{
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.125;
    time_options.steps = configured_steps;
    time_options.kinematic_viscosity = 0.0;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;
    return SimpleFluid::FluidSolver<Pack>(
        make_single_cell_mesh(), {}, time_options, linear_options);
}

} // namespace

TEST(ProgressLineFormatterTest, FormatsCompleteStepSummary)
{
    SimpleFluid::FluidStepStatistics<double> statistics;
    statistics.converged = false;
    statistics.nonlinear_iterations = 2;
    statistics.linear_solves = 3;
    statistics.krylov_iterations = 17;
    statistics.achieved_tolerance = 1.25e-8;
    statistics.momentum = 2.5e-4;
    statistics.pressure = 3.5e-5;
    statistics.temperature = 4.5e-6;
    statistics.continuity = 5.5e-7;

    const SimpleFluid::ProgressLineFormatter formatter;
    EXPECT_EQ(
        formatter.format(4, 9, 0.5, statistics),
        "step=4/9 time=5.000000e-01 converged=no nonlinear=2 "
        "linear_solves=3 krylov_iterations=17 "
        "linear_tolerance=1.250000e-08 "
        "residuals(momentum=2.500000e-04,pressure=3.500000e-05,"
        "temperature=4.500000e-06,continuity=5.500000e-07)");
}

TEST(SolverProgressTest, RunWritesExactlyOneLinePerStep)
{
    auto solver = make_solver();
    std::ostringstream output;
    SimpleFluid::ProgressStream progress(output);

    solver.run(2, progress);

    const auto text = output.str();
    EXPECT_EQ(solver.step_index(), 2);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 2);
    EXPECT_NE(text.find("step=1/2 time=1.250000e-01 converged=yes"),
              std::string::npos);
    EXPECT_NE(text.find("step=2/2 time=2.500000e-01 converged=yes"),
              std::string::npos);
    EXPECT_NE(text.find("linear_solves="), std::string::npos);
    EXPECT_NE(text.find("residuals(momentum="), std::string::npos);
}

TEST(SolverProgressTest, StepUsesConfiguredTotalAndPreservesStreamFormatting)
{
    auto solver = make_solver(3);
    std::ostringstream output;
    output << std::fixed << std::setprecision(2);
    const auto flags = output.flags();
    const auto precision = output.precision();
    SimpleFluid::ProgressStream progress(output);

    solver.step(progress);

    EXPECT_NE(output.str().find("step=1/3"), std::string::npos);
    EXPECT_EQ(output.flags(), flags);
    EXPECT_EQ(output.precision(), precision);
}

TEST(SolverProgressTest, RunRejectsNegativeStepCount)
{
    auto solver = make_solver();
    std::ostringstream output;
    SimpleFluid::ProgressStream progress(output);
    EXPECT_THROW(solver.run(-1, progress), std::invalid_argument);
}

TEST(SolverProgressTest, ContinuedRunUsesAbsoluteFinalStep)
{
    auto solver = make_solver();
    solver.step();
    std::ostringstream output;
    SimpleFluid::ProgressStream progress(output);

    solver.run(2, progress);

    EXPECT_NE(output.str().find("step=2/3"), std::string::npos);
    EXPECT_NE(output.str().find("step=3/3"), std::string::npos);
    EXPECT_EQ(output.str().find("step=3/2"), std::string::npos);
}

TEST(SolverProgressTest, BoussinesqExposesProgressStepOverload)
{
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-3;
    time_options.steps = 1;
    time_options.thermal_diffusivity = 1.0e-2;
    time_options.kinematic_viscosity = 1.0e-2;

    SimpleFluid::BoussinesqSolver<Pack> solver(
        make_single_cell_mesh(), {}, time_options);
    solver.initialize_heated_box(1.0, 0.0);
    std::ostringstream output;
    SimpleFluid::ProgressStream progress(output);

    solver.step(progress);

    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_NE(output.str().find("step=1/1"), std::string::npos);
    EXPECT_NE(output.str().find("temperature="), std::string::npos);
}
