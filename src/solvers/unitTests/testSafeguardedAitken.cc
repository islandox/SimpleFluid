#include <gtest/gtest.h>
#include "solvers/SafeguardedAitken.hh"

#include <array>
#include <limits>

using SimpleFluid::SafeguardedAitken;
using SimpleFluid::SafeguardedAitkenOptions;

namespace
{
double observe(SafeguardedAitken& acceleration, double residual)
{
    const std::array values{residual};
    return acceleration.finish(acceleration.prepare(values));
}
}

TEST(SafeguardedAitkenTest, AcceleratesContractingMapAndRetainsTinyResidualInformation)
{
    for (double scale : {1.0, 1e-25})
    {
        SafeguardedAitken acceleration({.initial_relaxation = .5});
        EXPECT_DOUBLE_EQ(observe(acceleration, scale), .5);
        EXPECT_DOUBLE_EQ(observe(acceleration, .5 * scale), 1.0);
        EXPECT_DOUBLE_EQ(observe(acceleration, .25 * scale), 1.0);
    }
}

TEST(SafeguardedAitkenTest, PartitionsAndZeroWeightedReplicasUseOneGlobalFactor)
{
    SafeguardedAitken serial({.initial_relaxation = .5});
    SafeguardedAitken first({.initial_relaxation = .5}), second({.initial_relaxation = .5}), empty({.initial_relaxation = .5});
    const std::array weights{2.0, 3.0};
    const std::array first_weights{2.0, 0.0};
    const std::array second_weights{3.0};
    for (double factor : {1.0, .75, .125})
    {
        const std::array all{factor, 2 * factor};
        const std::array first_values{factor, 1000 * factor};
        const std::array second_values{2 * factor};
        auto products = first.prepare(first_values, first_weights);
        const auto other = second.prepare(second_values, second_weights);
        const auto none = empty.prepare({});
        for (std::size_t i = 0; i < products.size(); ++i) products[i] += other[i] + none[i];
        const double expected = serial.finish(serial.prepare(all, weights));
        EXPECT_DOUBLE_EQ(first.finish(products), expected);
        EXPECT_DOUBLE_EQ(second.finish(products), expected);
        EXPECT_DOUBLE_EQ(empty.finish(products), expected);
    }
}

TEST(SafeguardedAitkenTest, DegeneracyGrowthAndInvalidSecantsUseFallbackAndRestart)
{
    SafeguardedAitken acceleration;
    EXPECT_DOUBLE_EQ(observe(acceleration, 1), 1);
    EXPECT_DOUBLE_EQ(observe(acceleration, 1), .5);
    EXPECT_DOUBLE_EQ(observe(acceleration, 3), .5);
    EXPECT_DOUBLE_EQ(observe(acceleration, 1.5), 1);
    EXPECT_DOUBLE_EQ(observe(acceleration, 1.6), .5);
    acceleration.reset();
    EXPECT_DOUBLE_EQ(observe(acceleration, 2), 1);
    const std::array values{1.0};
    auto products = acceleration.prepare(values);
    products[1] = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(acceleration.finish(products), .5);
    EXPECT_DOUBLE_EQ(observe(acceleration, .5), 1);
}

TEST(SafeguardedAitkenTest, BoundsAndResetApplyToIndependentProblems)
{
    SafeguardedAitken acceleration({.initial_relaxation = .6, .minimum_relaxation = .4,
                                   .maximum_relaxation = .8});
    EXPECT_DOUBLE_EQ(observe(acceleration, 1), .6);
    EXPECT_DOUBLE_EQ(observe(acceleration, .9), .8);
    EXPECT_DOUBLE_EQ(observe(acceleration, -.9), .4);
    acceleration.reset();
    EXPECT_DOUBLE_EQ(observe(acceleration, 9), .6);
}

TEST(SafeguardedAitkenTest, RejectsInvalidLocalInputsBeforeCallerReduction)
{
    EXPECT_THROW(SafeguardedAitken({.minimum_relaxation = .8}), std::invalid_argument);
    EXPECT_THROW(SafeguardedAitken({.maximum_residual_growth = .5}), std::invalid_argument);
    SafeguardedAitken acceleration;
    EXPECT_THROW(acceleration.finish({}), std::logic_error);
    const std::array invalid{std::numeric_limits<double>::quiet_NaN()};
    EXPECT_THROW(acceleration.prepare(invalid), std::invalid_argument);
    const std::array good{1.0}, negative{-1.0};
    const std::array mismatch{1.0, 2.0};
    EXPECT_THROW(acceleration.prepare(good, negative), std::invalid_argument);
    EXPECT_DOUBLE_EQ(observe(acceleration, 1.0), 1.0);
    EXPECT_THROW(acceleration.prepare(mismatch), std::invalid_argument);
    EXPECT_DOUBLE_EQ(observe(acceleration, .5), 1.0);
}
