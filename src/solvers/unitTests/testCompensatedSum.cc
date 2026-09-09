#include "utils/CompensatedSum.hh"

#include <gtest/gtest.h>

// Use double explicitly so Linux also exercises Apple Arm64 accumulator precision.
TEST(CompensatedSumTest, RetainsLargeMeshInventoryAtDoublePrecision)
{
    SimpleFluid::detail::CompensatedSum<double> mass;
    constexpr double density = 996.558076096375;
    for (int cell = 0; cell < 8000; ++cell)
        mass += density * 0.125;
    EXPECT_NEAR(mass.value(), density * 1000.0, 1e-9);
}

TEST(CompensatedSumTest, RetainsSmallContributionsAcrossCancellation)
{
    SimpleFluid::detail::CompensatedSum<double> sum;
    sum += 1.0;
    sum += 1e16;
    sum += 1.0;
    sum += -1e16;
    EXPECT_DOUBLE_EQ(sum.value(), 2.0);
}
