/**
 * @file testVec3.cc
 * @brief Unit tests for the vec3 3D vector class.
 */

#include <gtest/gtest.h>

#include "dataclass/vec3.hh"

#include <array>
#include <cmath>

namespace
{

using Vec3 = SimpleFluid::vec3<double>;

TEST(Vec3Test, ConstructionAndAccess)
{
    // Default zero-initialization
    {
        const Vec3 v;
        EXPECT_DOUBLE_EQ(v.x, 0.0);
        EXPECT_DOUBLE_EQ(v.y, 0.0);
        EXPECT_DOUBLE_EQ(v.z, 0.0);
    }
    // Component constructor
    {
        const Vec3 v(1.0, 2.0, 3.0);
        EXPECT_DOUBLE_EQ(v.x, 1.0);
        EXPECT_DOUBLE_EQ(v.y, 2.0);
        EXPECT_DOUBLE_EQ(v.z, 3.0);
    }
    // Array constructor
    {
        const Vec3 v(std::array<double, 3>{4.0, 5.0, 6.0});
        EXPECT_DOUBLE_EQ(v.x, 4.0);
        EXPECT_DOUBLE_EQ(v.y, 5.0);
        EXPECT_DOUBLE_EQ(v.z, 6.0);
    }
    // Indexed component access (read + write)
    {
        Vec3 v(1.0, 2.0, 3.0);
        const Vec3& cv = v;
        EXPECT_DOUBLE_EQ(cv.component(0), 1.0);
        EXPECT_DOUBLE_EQ(cv.component(1), 2.0);
        EXPECT_DOUBLE_EQ(cv.component(2), 3.0);
        v.component(1) = 5.0;
        EXPECT_DOUBLE_EQ(v.y, 5.0);
    }
}

TEST(Vec3Test, ArithmeticAndComparison)
{
    const Vec3 a(1.0, 2.0, 3.0);
    const Vec3 b(4.0, 6.0, 8.0);

    EXPECT_EQ(a + b, (Vec3{5.0, 8.0, 11.0}));
    EXPECT_EQ(a - b, (Vec3{-3.0, -4.0, -5.0}));
    EXPECT_EQ(a * 2.0, (Vec3{2.0, 4.0, 6.0}));
    EXPECT_EQ(b / 2.0, (Vec3{2.0, 3.0, 4.0}));

    EXPECT_EQ(a, a);
    EXPECT_NE(a, b);

    // Large and small values
    const Vec3 large(1.0e10, 1.0e10, 1.0e10);
    EXPECT_DOUBLE_EQ((large + large).x, 2.0e10);

    const Vec3 small(1.0e-10, 1.0e-10, 1.0e-10);
    EXPECT_NEAR(small.norm(), std::sqrt(3.0) * 1.0e-10, 1.0e-15);
}

TEST(Vec3Test, GeometricOperations)
{
    // Dot product
    EXPECT_DOUBLE_EQ(Vec3(1.0, 2.0, 3.0).dot(Vec3(4.0, 5.0, 6.0)), 32.0);
    EXPECT_DOUBLE_EQ(Vec3(1.0, 0.0, 0.0).dot(Vec3(0.0, 1.0, 0.0)), 0.0);

    // Cross product
    EXPECT_EQ(Vec3(1.0, 0.0, 0.0).cross(Vec3(0.0, 1.0, 0.0)),
              (Vec3{0.0, 0.0, 1.0}));
    EXPECT_EQ(Vec3(1.0, 2.0, 3.0).cross(Vec3(2.0, 4.0, 6.0)),
              (Vec3{0.0, 0.0, 0.0}));

    // Norm
    EXPECT_DOUBLE_EQ(Vec3(3.0, 4.0, 0.0).norm(), 5.0);
    EXPECT_DOUBLE_EQ(Vec3(1.0, 0.0, 0.0).norm(), 1.0);
    EXPECT_DOUBLE_EQ(Vec3(0.0, 0.0, 0.0).norm(), 0.0);
}

} // namespace
