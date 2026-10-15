#include <gtest/gtest.h>
#include "solvers/SolverTimings.hh"
#include "solvers/ALEConvergenceStatistics.hh"
#include "utils/testing_environment.hh"

namespace
{
testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
}

TEST(SolverTimingsTest, AccumulatesCompletedInclusiveScopesAndReturnsIndependentSnapshot)
{
    SimpleFluid::SolverTimings timings;
    using Phase = SimpleFluid::SolverPhase;
    const auto initial = timings.snapshot();
    {
        const auto parent = timings.scope(Phase::Step);
        {
            const auto child = timings.scope(Phase::Temperature);
            EXPECT_EQ(timings.query(Phase::Temperature).calls, 0u);
        }
        EXPECT_EQ(timings.query(Phase::Step).calls, 0u);
        EXPECT_THROW(timings.reset(), std::logic_error);
    }
    const auto first = timings.query(Phase::Step);
    EXPECT_EQ(first.calls, 1u);
    EXPECT_GE(first.seconds, timings.query(Phase::Temperature).seconds);
    {
        const auto second = timings.scope(Phase::Step);
    }
    EXPECT_EQ(timings.query(Phase::Step).calls, 2u);
    EXPECT_GE(timings.query(Phase::Step).seconds, first.seconds);
    EXPECT_EQ(initial[static_cast<std::size_t>(Phase::Step)].calls, 0u);
    EXPECT_STREQ(first.name, "sf.step");
    EXPECT_EQ(first.depth, 0u);
    timings.reset();
    for (const auto& entry : timings.snapshot())
    {
        EXPECT_EQ(entry.calls, 0u);
        EXPECT_EQ(entry.seconds, 0.0);
    }
    EXPECT_STREQ(timings.query(Phase::Step).name, "sf.step");
    EXPECT_THROW(timings.query(Phase::Count), std::out_of_range);
    EXPECT_THROW((void)timings.scope(Phase::Count), std::out_of_range);
    EXPECT_NO_THROW(timings.reset());
}

TEST(SolverTimingsTest, ExceptionUnwindingCountsAttemptAndBalancesReset)
{
    SimpleFluid::SolverTimings timings;
    using Phase = SimpleFluid::SolverPhase;
    EXPECT_THROW(([&]
    {
        const auto parent = timings.scope(Phase::Step);
        const auto child = timings.scope(Phase::Temperature);
        throw std::runtime_error("rejected trial");
    })(), std::runtime_error);
    EXPECT_EQ(timings.query(Phase::Step).calls, 1u);
    EXPECT_EQ(timings.query(Phase::Temperature).calls, 1u);
    EXPECT_GE(timings.query(Phase::Step).seconds, timings.query(Phase::Temperature).seconds);
    EXPECT_NO_THROW(timings.reset());
}

TEST(ALEConvergenceStatisticsTest, RecordsJointAndSoleBlockersWithoutChangingGateComparisons)
{
    SimpleFluid::ALEConvergenceStatistics statistics;
    std::array<double,7> ratios{};
    std::array<bool,7> failed{};
    ratios[0] = 3.; ratios[6] = 2.; failed[0] = failed[6] = true;
    statistics.record(ratios,failed);
    ratios.fill(0.); failed.fill(false);
    // A rounded ratio of one must not erase the caller's exact failed comparison.
    ratios[6] = 1.; failed[6] = true;
    statistics.record(ratios,failed);
    failed.fill(false);
    statistics.record(ratios,failed);
    const auto snapshot = statistics;
    EXPECT_EQ(snapshot.checks,3u);
    EXPECT_EQ(snapshot.converged_checks,1u);
    EXPECT_EQ(snapshot.blocking_counts()[0],1u);
    EXPECT_EQ(snapshot.blocking_counts()[6],2u);
    EXPECT_EQ(snapshot.blocking_combinations[(1u << 0) | (1u << 6)],1u);
    EXPECT_EQ(snapshot.blocking_combinations[1u << 6],1u);
    EXPECT_EQ(snapshot.blocking_combinations[0],1u);
    EXPECT_DOUBLE_EQ(snapshot.maximum_normalized_residual[0],3.);
    EXPECT_DOUBLE_EQ(snapshot.maximum_normalized_residual[6],2.);
}
