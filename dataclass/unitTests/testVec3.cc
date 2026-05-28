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

TEST(Vec3Test, DefaultConstructorZeroInitializes)
{
    const Vec3 v;
    EXPECT_DOUBLE_EQ(v.x, 0.0);
    EXPECT_DOUBLE_EQ(v.y, 0.0);
    EXPECT_DOUBLE_EQ(v.z, 0.0);
}

TEST(Vec3Test, ComponentConstructorStoresValues)
{
    const Vec3 v(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(v.x, 1.0);
    EXPECT_DOUBLE_EQ(v.y, 2.0);
    EXPECT_DOUBLE_EQ(v.z, 3.0);
}

TEST(Vec3Test, ArrayConstructorStoresValues)
{
    const Vec3 v(std::array<double, 3>{4.0, 5.0, 6.0});
    EXPECT_DOUBLE_EQ(v.x, 4.0);
    EXPECT_DOUBLE_EQ(v.y, 5.0);
    EXPECT_DOUBLE_EQ(v.z, 6.0);
}

TEST(Vec3Test, ArithmeticOperators)
{
    const Vec3 a(1.0, 2.0, 3.0);
    const Vec3 b(4.0, 6.0, 8.0);

    EXPECT_EQ(a + b, (Vec3{5.0, 8.0, 11.0}));
    EXPECT_EQ(a - b, (Vec3{-3.0, -4.0, -5.0}));
    EXPECT_EQ(a * 2.0, (Vec3{2.0, 4.0, 6.0}));
    EXPECT_EQ(b / 2.0, (Vec3{2.0, 3.0, 4.0}));
}

TEST(Vec3Test, DotProduct)
{
    const Vec3 a(1.0, 2.0, 3.0);
    const Vec3 b(4.0, 5.0, 6.0);
    EXPECT_DOUBLE_EQ(a.dot(b), 32.0);
}

TEST(Vec3Test, DotProductOrthogonal)
{
    const Vec3 a(1.0, 0.0, 0.0);
    const Vec3 b(0.0, 1.0, 0.0);
    EXPECT_DOUBLE_EQ(a.dot(b), 0.0);
}

TEST(Vec3Test, CrossProduct)
{
    const Vec3 a(1.0, 0.0, 0.0);
    const Vec3 b(0.0, 1.0, 0.0);
    EXPECT_EQ(a.cross(b), (Vec3{0.0, 0.0, 1.0}));
}

TEST(Vec3Test, CrossProductCollinearReturnsZero)
{
    const Vec3 a(1.0, 2.0, 3.0);
    const Vec3 b(2.0, 4.0, 6.0);
    EXPECT_EQ(a.cross(b), (Vec3{0.0, 0.0, 0.0}));
}

TEST(Vec3Test, Norm)
{
    EXPECT_DOUBLE_EQ(Vec3(3.0, 4.0, 0.0).norm(), 5.0);
    EXPECT_DOUBLE_EQ(Vec3(1.0, 0.0, 0.0).norm(), 1.0);
    EXPECT_DOUBLE_EQ(Vec3(0.0, 0.0, 0.0).norm(), 0.0);
}

TEST(Vec3Test, ComparisonOperators)
{
    const Vec3 a(1.0, 2.0, 3.0);
    const Vec3 b(1.0, 2.0, 3.0);
    const Vec3 c(1.0, 2.0, 4.0);

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(Vec3Test, HandlesSmallAndLargeValues)
{
    const Vec3 large(1.0e10, 1.0e10, 1.0e10);
    EXPECT_DOUBLE_EQ((large + large).x, 2.0e10);

    const Vec3 small(1.0e-10, 1.0e-10, 1.0e-10);
    EXPECT_NEAR(small.norm(), std::sqrt(3.0) * 1.0e-10, 1.0e-15);
}

} // namespace
