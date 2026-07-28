/**
 * @file testSteadyStateSearch.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for adaptive pseudo-transient steady-state search.
 * @version 0.1
 * @date 2026-07-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/MeshFactory.hh"
#include "solvers/FluidSolver.hh"
#include "solvers/SteadyStateSearch.hh"
#include "utils/testing_environment.hh"

#include <memory>
#include <sstream>
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
    database->set("domain_type", static_cast<int>(SimpleFluid::MeshFactory::DomainType::BOX));
    database->set("X", SimpleFluid::ArrReal{0.0, 1.0});
    database->set("Y", SimpleFluid::ArrReal{0.0, 1.0});
    database->set("Z", SimpleFluid::ArrReal{0.0, 1.0});
    database->set("domain_exterior_face_types",
                  SimpleFluid::ArrString{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    return SimpleFluid::MeshFactory(database).build<Pack>();
}

class ExposedFluidSolver : public SimpleFluid::FluidSolver<Pack>
{
public:
    using SimpleFluid::FluidSolver<Pack>::FluidSolver;

    void set_projected_flux(double value) { projected_face_fluxes().put_scalar(value); }
};

} // namespace

TEST(AdaptiveSteadyStateControllerTest, RampsByCourantAndRequiresSustainedUpdates)
{
    SimpleFluid::SteadyStateSearchOptions options;
    options.maximum_steps = 10;
    options.minimum_steps = 2;
    options.required_consecutive_steps = 2;
    options.relative_update_tolerance = 0.1;
    options.minimum_time_step = 0.01;
    options.maximum_time_step = 1.0;
    options.target_courant_number = 0.8;
    options.time_step_growth_factor = 2.0;
    options.time_step_reduction_factor = 0.5;
    options.rejection_recovery_steps = 2;
    options.rejection_time_step_safety_factor = 0.9;

    SimpleFluid::AdaptiveSteadyStateController controller(options, 0.1);
    const SimpleFluid::SteadyStateUpdateRates<double> small_updates{0.05, 0.04, 0.03};

    const auto first = controller.observe(0.1, 0.1, 0.1, small_updates, true);
    EXPECT_EQ(first.consecutive_converged_steps, 0);
    EXPECT_FALSE(first.steady);
    EXPECT_DOUBLE_EQ(first.next_time_step, 0.2);

    const auto second = controller.observe(0.3, 0.2, 0.4, small_updates, true);
    EXPECT_EQ(second.consecutive_converged_steps, 1);
    EXPECT_FALSE(second.steady);
    EXPECT_DOUBLE_EQ(second.next_time_step, 0.4);

    const auto third = controller.observe(0.7, 0.4, 0.8, small_updates, true);
    EXPECT_EQ(third.consecutive_converged_steps, 2);
    EXPECT_TRUE(third.steady);
    EXPECT_DOUBLE_EQ(third.next_time_step, 0.4);

    const auto failed = controller.observe(1.1, 0.4, 0.8, small_updates, false);
    EXPECT_EQ(failed.consecutive_converged_steps, 0);
    EXPECT_FALSE(failed.steady);
    EXPECT_DOUBLE_EQ(failed.next_time_step, 0.2);
    EXPECT_DOUBLE_EQ(controller.rejected_time_step(0.2), 0.1);
    SimpleFluid::AdaptiveSteadyStateController floor_controller(
        options, 0.1);
    EXPECT_DOUBLE_EQ(
        floor_controller.rejected_time_step(0.01), 0.01);

    const auto first_recovery =
        controller.observe(1.2, 0.1, 0.1, small_updates, true);
    EXPECT_DOUBLE_EQ(first_recovery.next_time_step, 0.1);
    const auto second_recovery =
        controller.observe(1.3, 0.1, 0.1, small_updates, true);
    EXPECT_DOUBLE_EQ(second_recovery.next_time_step, 0.1);
    const auto recovered =
        controller.observe(1.4, 0.1, 0.1, small_updates, true);
    EXPECT_DOUBLE_EQ(recovered.next_time_step, 0.18);

    EXPECT_DOUBLE_EQ(
        controller.rejected_time_step(0.18), 0.09);
    const auto lower_ceiling =
        controller.observe(1.49, 0.09, 0.1, small_updates, true);
    EXPECT_DOUBLE_EQ(lower_ceiling.next_time_step, 0.09);
}

TEST(AdaptiveSteadyStateControllerTest, RejectsInvalidControlsAndObservations)
{
    SimpleFluid::SteadyStateSearchOptions options;
    options.maximum_steps = 0;
    EXPECT_THROW(SimpleFluid::AdaptiveSteadyStateController(options, 1.0e-3),
                 std::invalid_argument);

    options = {};
    options.maximum_retries_per_step = -1;
    EXPECT_THROW(SimpleFluid::AdaptiveSteadyStateController(options, 1.0e-3),
                 std::invalid_argument);

    options = {};
    options.rejection_recovery_steps = -1;
    EXPECT_THROW(SimpleFluid::AdaptiveSteadyStateController(options, 1.0e-3),
                 std::invalid_argument);

    options = {};
    options.rejection_time_step_safety_factor = 1.0;
    EXPECT_THROW(SimpleFluid::AdaptiveSteadyStateController(options, 1.0e-3),
                 std::invalid_argument);

    options = {};
    options.maximum_steps = 10;
    options.minimum_steps = 8;
    options.required_consecutive_steps = 4;
    EXPECT_THROW(SimpleFluid::AdaptiveSteadyStateController(options, 1.0e-3),
                 std::invalid_argument);

    options = {};
    SimpleFluid::AdaptiveSteadyStateController controller(options, 1.0e-3);
    EXPECT_THROW(controller.observe(1.0e-3, 1.0e-3, -1.0, {0.0, 0.0, 0.0}, true),
                 std::invalid_argument);
    EXPECT_THROW(controller.rejected_time_step(0.0), std::invalid_argument);
}

TEST(SteadyStateFieldMonitorTest, ComputesVolumeWeightedRelativeUpdateRates)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::VectorCellField<Pack> velocity(
        mesh, SimpleFluid::VectorCellField<Pack>::vec_type{0.0, 0.0, 0.0}, "velocity");
    SimpleFluid::CellField<Pack> temperature(mesh, 10.0, "temperature");
    SimpleFluid::CellField<Pack> turbulence(mesh, 1.0, "turbulence");

    SimpleFluid::SteadyStateUpdateScales scales;
    scales.velocity = 1.0;
    scales.temperature = 2.0;
    scales.turbulence = 0.5;
    SimpleFluid::SteadyStateFieldMonitor<Pack> monitor(mesh, 10.0, scales);
    monitor.initialize(velocity, temperature, {&turbulence});

    velocity.set_owned_value(0, {2.0, 0.0, 0.0});
    temperature.set_owned_value(0, 14.0);
    turbulence.set_owned_value(0, 3.0);
    const auto rates = monitor.observe(2.0);

    EXPECT_NEAR(rates.velocity, 0.5, 1.0e-14);
    EXPECT_NEAR(rates.temperature, 0.5, 1.0e-14);
    EXPECT_NEAR(rates.turbulence, 1.0 / 3.0, 1.0e-14);
    EXPECT_NEAR(rates.maximum(), 0.5, 1.0e-14);

    const auto unchanged = monitor.observe(2.0);
    EXPECT_DOUBLE_EQ(unchanged.maximum(), 0.0);
}

TEST(FluidSolverAdaptiveTimeStepTest, UpdatesTimeStepAndComputesMaximumCourantNumber)
{
    auto mesh = make_single_cell_mesh();
    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 0.25;
    ExposedFluidSolver solver(mesh, {}, time_options);
    solver.set_projected_flux(2.0);

    EXPECT_DOUBLE_EQ(solver.time_step(), 0.25);
    EXPECT_NEAR(solver.maximum_courant_number(), 1.5, 1.0e-14);

    solver.set_time_step(0.125);
    EXPECT_DOUBLE_EQ(solver.time_step(), 0.125);
    EXPECT_NEAR(solver.maximum_courant_number(), 0.75, 1.0e-14);
    EXPECT_THROW(solver.set_time_step(0.0), std::invalid_argument);
}

TEST(SteadyStateProgressLineFormatterTest, FormatsAdaptiveAndPhysicalDiagnostics)
{
    SimpleFluid::SteadyStateStepStatistics<double> statistics;
    statistics.iteration = 3;
    statistics.maximum_steps = 20;
    statistics.consecutive_converged_steps = 2;
    statistics.required_consecutive_steps = 4;
    statistics.time = 0.7;
    statistics.time_step = 0.2;
    statistics.next_time_step = 0.25;
    statistics.maximum_courant_number = 0.6;
    statistics.update_rates = {1.0e-3, 2.0e-3, 3.0e-3};
    statistics.solver_converged = true;

    SimpleFluid::FluidStepStatistics<double> solver_statistics;
    solver_statistics.krylov_iterations = 17;
    solver_statistics.achieved_tolerance = 1.0e-9;
    const SimpleFluid::SteadyStateProgressLineFormatter formatter;
    const auto line = formatter.format(statistics, solver_statistics);

    EXPECT_NE(line.find("steady_step=3/20"), std::string::npos);
    EXPECT_NE(line.find("max_Co=6.000000e-01"), std::string::npos);
    EXPECT_NE(line.find("steady_window=2/4"), std::string::npos);
    EXPECT_NE(line.find("turbulence=3.000000e-03"), std::string::npos);
    EXPECT_NE(line.find("krylov_iterations=17"), std::string::npos);

    const auto retry_line =
        formatter.format_retry(4, 1, 3, 0.7, 0.2, 0.1, "transport did not converge");
    EXPECT_NE(retry_line.find("steady_step=4 rejected=yes retry=1/3"), std::string::npos);
    EXPECT_NE(retry_line.find("next_dt=1.000000e-01"), std::string::npos);
    EXPECT_NE(retry_line.find("transport did not converge"), std::string::npos);
}
