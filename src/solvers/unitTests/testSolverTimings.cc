#include <gtest/gtest.h>
#include "solvers/SolverTimings.hh"
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
