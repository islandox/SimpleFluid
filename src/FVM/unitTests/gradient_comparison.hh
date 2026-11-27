#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace SimpleFluid::test
{

// Independently evaluated face contributions can round differently before cancellation.
inline void expect_independent_gradients_near(
    const auto& actual, const auto& expected, double roundoff_units)
{
    const auto a = actual.owned_read_view(), b = expected.owned_read_view();
    ASSERT_EQ(a.extent(0), b.extent(0));
    ASSERT_EQ(a.extent(1), b.extent(1));
    for (size_t cell = 0; cell < a.extent(0); ++cell)
        for (size_t component = 0; component < a.extent(1); ++component)
        {
            const double actual_value = a(cell, component), expected_value = b(cell, component);
            ASSERT_TRUE(std::isfinite(actual_value) && std::isfinite(expected_value))
                << "cell " << cell << " component " << component;
            const auto scale = std::max({1., std::abs(actual_value), std::abs(expected_value)});
            EXPECT_NEAR(actual_value, expected_value, roundoff_units * std::numeric_limits<double>::epsilon() * scale)
                << "cell " << cell << " component " << component;
        }
}

} // namespace SimpleFluid::test
