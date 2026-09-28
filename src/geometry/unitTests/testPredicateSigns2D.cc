/** @file testPredicateSigns2D.cc
 * @brief Independent integer oracle checks for stored binary64 XY predicates.
 */
#include <gtest/gtest.h>

#include "geometry/mesh/PredicateSigns2D.hh"

#include <boost/multiprecision/cpp_int.hpp>

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>

namespace
{
using SimpleFluid::Meshes::detail::PredicateSign;
using SimpleFluid::Meshes::detail::orient2d_sign;
using SimpleFluid::Meshes::detail::incircle_sign;
using Vec3 = SimpleFluid::MeshUtils::Vec3;
using boost::multiprecision::cpp_int;

// Independent oracle: put every binary64 value on one fixed 2^-1074 grid,
// then evaluate ordinary integer determinants. Production uses dyadic
// operations with independently tracked exponents.
cpp_int integer_coordinate(double value)
{
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const int raw_exponent = static_cast<int>((bits >> 52) & 0x7ff);
    const auto fraction = bits & ((std::uint64_t{1} << 52) - 1);
    cpp_int n = raw_exponent == 0
        ? cpp_int(fraction) : cpp_int(fraction | (std::uint64_t{1} << 52));
    if (raw_exponent) n <<= raw_exponent - 1;
    return bits >> 63 ? -n : n;
}

PredicateSign oracle_sign(const cpp_int& n)
{
    return n > 0 ? PredicateSign::Positive
         : n < 0 ? PredicateSign::Negative : PredicateSign::Zero;
}

PredicateSign oracle_orientation(const Vec3& a, const Vec3& b, const Vec3& c)
{
    const auto ax = integer_coordinate(a.x), ay = integer_coordinate(a.y);
    const auto bx = integer_coordinate(b.x), by = integer_coordinate(b.y);
    const auto cx = integer_coordinate(c.x), cy = integer_coordinate(c.y);
    return oracle_sign(cpp_int((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)));
}

PredicateSign oracle_incircle(const Vec3& a, const Vec3& b,
                              const Vec3& c, const Vec3& p)
{
    const auto ax = integer_coordinate(a.x) - integer_coordinate(p.x);
    const auto ay = integer_coordinate(a.y) - integer_coordinate(p.y);
    const auto bx = integer_coordinate(b.x) - integer_coordinate(p.x);
    const auto by = integer_coordinate(b.y) - integer_coordinate(p.y);
    const auto cx = integer_coordinate(c.x) - integer_coordinate(p.x);
    const auto cy = integer_coordinate(c.y) - integer_coordinate(p.y);
    return oracle_sign(cpp_int((ax * ax + ay * ay) * (bx * cy - by * cx)
              + (bx * bx + by * by) * (cx * ay - cy * ax)
              + (cx * cx + cy * cy) * (ax * by - ay * bx)));
}

TEST(PredicateSigns2DTest, SmallTrianglesAreStrictlyInside)
{
    for (double h : {1.0e-2, 1.0e-4, 1.0e-5})
    {
        const Vec3 a{0, 0, 0}, b{h, 0, 0}, c{0, h, 0}, p{h / 2, h / 2, 0};
        EXPECT_EQ(orient2d_sign(a, b, c), PredicateSign::Positive);
        EXPECT_EQ(incircle_sign(a, b, c, p), PredicateSign::Positive);
        EXPECT_EQ(incircle_sign(a, b, c, p), oracle_incircle(a, b, c, p));
    }
}

TEST(PredicateSigns2DTest, ExactAndAdjacentCocircularValues)
{
    const Vec3 a{0, 0, 0}, b{1, 0, 0}, c{0, 1, 0};
    const std::array<Vec3, 3> points{{
        {1, 1, 0},
        {1, std::nextafter(1.0, 0.0), 0},
        {1, std::nextafter(1.0, std::numeric_limits<double>::infinity()), 0}}};
    const std::array signs{PredicateSign::Zero, PredicateSign::Positive,
                           PredicateSign::Negative};
    for (size_t i = 0; i < points.size(); ++i)
    {
        EXPECT_EQ(incircle_sign(a, b, c, points[i]), signs[i]);
        EXPECT_EQ(incircle_sign(a, c, b, points[i]),
                  static_cast<PredicateSign>(-static_cast<int>(signs[i])));
        EXPECT_EQ(incircle_sign(b, c, a, points[i]), signs[i]);
    }
}

TEST(PredicateSigns2DTest, CollinearPerturbationsAndPermutations)
{
    const Vec3 a{1, 1, 0}, b{2, 2, 0};
    const std::array<Vec3, 3> points{{
        {3, 3, 0},
        {3, std::nextafter(3.0, 4.0), 0},
        {3, std::nextafter(3.0, 2.0), 0}}};
    const std::array signs{PredicateSign::Zero, PredicateSign::Positive,
                           PredicateSign::Negative};
    for (size_t i = 0; i < points.size(); ++i)
    {
        EXPECT_EQ(orient2d_sign(a, b, points[i]), signs[i]);
        EXPECT_EQ(orient2d_sign(a, points[i], b),
                  static_cast<PredicateSign>(-static_cast<int>(signs[i])));
        EXPECT_EQ(orient2d_sign(b, points[i], a), signs[i]);
        EXPECT_EQ(orient2d_sign(a, b, points[i]),
                  oracle_orientation(a, b, points[i]));
    }
}

TEST(PredicateSigns2DTest, ExactScalingTranslationAndExtremeExponents)
{
    const Vec3 a{3, 4, 0}, b{-3, 4, 0}, c{-3, -4, 0};
    const Vec3 p{0, std::nextafter(-5.0, 0.0), 0};
    for (int exponent : {-300, -100, 0, 100, 300})
    {
        const auto transform = [&](Vec3 v) {
            return Vec3{std::ldexp(v.x, exponent),
                        std::ldexp(v.y, exponent), 0};
        };
        const auto aa = transform(a), bb = transform(b);
        const auto cc = transform(c), pp = transform(p);
        EXPECT_EQ(orient2d_sign(aa, bb, cc), oracle_orientation(aa, bb, cc));
        EXPECT_EQ(incircle_sign(aa, bb, cc, pp), oracle_incircle(aa, bb, cc, pp));
    }
    const Vec3 shifted_a{a.x + 1, a.y + 1, 0};
    const Vec3 shifted_b{b.x + 1, b.y + 1, 0};
    const Vec3 shifted_c{c.x + 1, c.y + 1, 0};
    const Vec3 shifted_p{p.x + 1, p.y + 1, 0};
    EXPECT_EQ(incircle_sign(shifted_a, shifted_b, shifted_c, shifted_p),
              oracle_incircle(shifted_a, shifted_b, shifted_c, shifted_p));
    EXPECT_EQ(incircle_sign(shifted_a, shifted_b, shifted_c, shifted_p),
              incircle_sign(a, b, c, p));
    const Vec3 tiny{std::numeric_limits<double>::denorm_min(), 0, 0};
    EXPECT_EQ(orient2d_sign({0, 0, 0}, tiny,
                            {0, std::numeric_limits<double>::denorm_min(), 0}),
              PredicateSign::Positive);
}

TEST(PredicateSigns2DTest, NearCocircularFallbackMatchesIntegerOracle)
{
    const Vec3 a{5, 0, 0}, b{3, 4, 0}, c{-3, 4, 0};
    for (const double y : {-5.0, std::nextafter(-5.0, 0.0),
                           std::nextafter(-5.0, -6.0)})
    {
        const Vec3 p{0, y, 0};
        EXPECT_EQ(incircle_sign(a, b, c, p), oracle_incircle(a, b, c, p));
    }
    EXPECT_THROW(orient2d_sign({0, 0, 0},
        {std::numeric_limits<double>::infinity(), 0, 0}, {0, 1, 0}),
        std::invalid_argument);
}

TEST(PredicateSigns2DTest, NonNearestArithmeticUsesExactFallback)
{
    const int previous = std::fegetround();
    ASSERT_NE(previous, -1);
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    const Vec3 a{1, 1, 0}, b{2, 2, 0};
    const Vec3 c{3, std::nextafter(3.0, 4.0), 0};
    EXPECT_EQ(orient2d_sign(a, b, c), oracle_orientation(a, b, c));
    const Vec3 p{0, std::nextafter(-5.0, 0.0), 0};
    EXPECT_EQ(incircle_sign({5, 0, 0}, {3, 4, 0}, {-3, 4, 0}, p),
              oracle_incircle({5, 0, 0}, {3, 4, 0}, {-3, 4, 0}, p));
    ASSERT_EQ(std::fesetround(previous), 0);
}
} // namespace
