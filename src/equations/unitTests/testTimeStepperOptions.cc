/**
 * @file testTimeStepperOptions.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for transient time-stepper options.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/TimeStepperOptions.hh"

namespace
{

/**
 * @brief Verifies default time-stepper options match expected physical smoke values.
 */
TEST(TimeStepperOptionsTest, DefaultsArePhysicalSmokeValues)
{
    const SimpleFluid::TimeStepperOptions options;

    EXPECT_DOUBLE_EQ(options.time_step, 1.0e-3);
    EXPECT_EQ(options.steps, 1);
    EXPECT_DOUBLE_EQ(options.thermal_diffusivity, 1.0e-3);
    EXPECT_DOUBLE_EQ(options.kinematic_viscosity, 1.0e-3);
    EXPECT_DOUBLE_EQ(options.thermal_expansion, 3.0e-3);
    EXPECT_DOUBLE_EQ(options.reference_temperature, 0.5);
}

TEST(TimeStepperOptionsTest, GravityVectorReflectsComponents)
{
    SimpleFluid::TimeStepperOptions options;
    options.gravity_x = -1.0;
    options.gravity_y = -2.0;
    options.gravity_z = -3.0;

    const auto gravity = options.gravity_vector();
    EXPECT_DOUBLE_EQ(gravity.x, -1.0);
    EXPECT_DOUBLE_EQ(gravity.y, -2.0);
    EXPECT_DOUBLE_EQ(gravity.z, -3.0);
}

} // namespace
