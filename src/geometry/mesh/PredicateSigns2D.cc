/** @file PredicateSigns2D.cc
 * @brief Filtered exact signs for finite IEEE binary64 XY coordinates.
 *
 * The first-stage error bounds are from Jonathan Richard Shewchuk,
 * "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
 * Predicates", Discrete & Computational Geometry 18 (1997), predicates.c,
 * May 18 1996, public domain. See https://www.cs.cmu.edu/~quake/robust.html.
 * We use only its certified A filters; uncertain and out-of-filter-range
 * inputs fall through to exact signed integer arithmetic on decoded binary64
 * significands. No expansion arithmetic or mutable exactinit state is used.
 *
 * The filter needs round-to-nearest binary64 evaluation without fused
 * contraction. Its guarded exponent range prevents nonzero intermediate
 * overflow or underflow; x86 FTZ/DAZ modes also bypass the filter. CMake
 * compiles this file without fast math, FMA contraction, or LTO. The exact
 * fallback uses Boost.Multiprecision cpp_int (Boost Software License 1.0) and
 * supports every finite binary64 XY input, including subnormals and extreme
 * exponents. Nonfinite XY is rejected.
 */
#include "geometry/mesh/PredicateSigns2D.hh"

#include <boost/multiprecision/cpp_int.hpp>

#include <bit>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace SimpleFluid::Meshes::detail
{
namespace
{
static_assert(std::numeric_limits<double>::is_iec559
              && std::numeric_limits<double>::digits == 53
              && sizeof(double) == sizeof(std::uint64_t));
static_assert(FLT_EVAL_METHOD == 0,
              "Predicate filter requires binary64 evaluation of double expressions.");
static_assert(std::is_same_v<real_t, double>);

using boost::multiprecision::cpp_int;
using Vec3 = MeshUtils::Vec3;

PredicateSign sign_of(double value)
{
    return value > 0 ? PredicateSign::Positive
         : value < 0 ? PredicateSign::Negative : PredicateSign::Zero;
}

PredicateSign sign_of(const cpp_int& value)
{
    return value > 0 ? PredicateSign::Positive
         : value < 0 ? PredicateSign::Negative : PredicateSign::Zero;
}

void check_finite(const Vec3& a)
{
    if (!std::isfinite(a.x) || !std::isfinite(a.y))
        throw std::invalid_argument("PredicateSigns2D requires finite binary64 XY coordinates.");
}

// A conservative domain for the floating filter. Four factors of 2^101
// cannot overflow, and four nonzero factors of 2^-101 remain normal. Exact
// zero differences are allowed; uncertain determinants use the fallback.
bool filter_factor(double x)
{
    constexpr double lower = 0x1p-101;
    constexpr double upper = 0x1p101;
    return x == 0.0 || (std::isfinite(x) && std::abs(x) >= lower
                                          && std::abs(x) <= upper);
}

bool filter_coordinate(double x)
{
    return x == 0.0 || (std::abs(x) >= 0x1p-100
                     && std::abs(x) <= 0x1p100);
}

bool filter_environment()
{
    if (std::fegetround() != FE_TONEAREST) return false;
#if defined(__SSE2__)
    // MXCSR FTZ and DAZ invalidate the usual IEEE error analysis.
    if ((_mm_getcsr() & 0x8040U) != 0) return false;
#endif
    return true;
}

struct Dyadic
{
    cpp_int numerator;
    int exponent{};
};

Dyadic exact(double value)
{
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto fraction = bits & ((std::uint64_t{1} << 52) - 1);
    const auto raw_exponent = static_cast<int>((bits >> 52) & 0x7ff);
    cpp_int numerator = raw_exponent == 0
        ? cpp_int(fraction) : cpp_int(fraction | (std::uint64_t{1} << 52));
    if (bits >> 63) numerator = -numerator;
    return {std::move(numerator), raw_exponent == 0 ? -1074 : raw_exponent - 1075};
}

Dyadic subtract(const Dyadic& a, const Dyadic& b)
{
    const int exponent = std::min(a.exponent, b.exponent);
    return {(a.numerator << (a.exponent - exponent))
          - (b.numerator << (b.exponent - exponent)), exponent};
}

Dyadic add(const Dyadic& a, const Dyadic& b)
{
    const int exponent = std::min(a.exponent, b.exponent);
    return {(a.numerator << (a.exponent - exponent))
          + (b.numerator << (b.exponent - exponent)), exponent};
}

Dyadic multiply(const Dyadic& a, const Dyadic& b)
{
    return {a.numerator * b.numerator, a.exponent + b.exponent};
}

Dyadic square(const Dyadic& a) { return multiply(a, a); }

PredicateSign exact_orientation(const Vec3& a, const Vec3& b, const Vec3& c)
{
    const auto acx = subtract(exact(a.x), exact(c.x));
    const auto acy = subtract(exact(a.y), exact(c.y));
    const auto bcx = subtract(exact(b.x), exact(c.x));
    const auto bcy = subtract(exact(b.y), exact(c.y));
    return sign_of(subtract(multiply(acx, bcy), multiply(acy, bcx)).numerator);
}

PredicateSign exact_incircle(const Vec3& a, const Vec3& b,
                             const Vec3& c, const Vec3& p)
{
    const auto ax = subtract(exact(a.x), exact(p.x));
    const auto ay = subtract(exact(a.y), exact(p.y));
    const auto bx = subtract(exact(b.x), exact(p.x));
    const auto by = subtract(exact(b.y), exact(p.y));
    const auto cx = subtract(exact(c.x), exact(p.x));
    const auto cy = subtract(exact(c.y), exact(p.y));
    const auto alift = add(square(ax), square(ay));
    const auto blift = add(square(bx), square(by));
    const auto clift = add(square(cx), square(cy));
    const auto ab = subtract(multiply(ax, by), multiply(bx, ay));
    const auto bc = subtract(multiply(bx, cy), multiply(cx, by));
    const auto ca = subtract(multiply(cx, ay), multiply(ax, cy));
    return sign_of(add(add(multiply(alift, bc), multiply(blift, ca)),
                       multiply(clift, ab)).numerator);
}

} // namespace

PredicateSign orient2d_sign(const Vec3& a, const Vec3& b, const Vec3& c)
{
    check_finite(a); check_finite(b); check_finite(c);
    // Shewchuk's orient2d A filter, with epsilon = 2^-53. The strict
    // inequality ensures an uncertain rounded zero reaches the exact path.
    if (filter_environment()
        && filter_coordinate(a.x) && filter_coordinate(a.y)
        && filter_coordinate(b.x) && filter_coordinate(b.y)
        && filter_coordinate(c.x) && filter_coordinate(c.y))
    {
        const double acx = a.x - c.x, bcx = b.x - c.x;
        const double acy = a.y - c.y, bcy = b.y - c.y;
        if (filter_factor(acx) && filter_factor(bcx)
            && filter_factor(acy) && filter_factor(bcy))
        {
            const double left = acx * bcy, right = acy * bcx;
            const double determinant = left - right;
            const double permanent = std::abs(left) + std::abs(right);
            constexpr double epsilon = 0x1p-53;
            constexpr double bound = (3.0 + 16.0 * epsilon) * epsilon;
            if (std::abs(determinant) > bound * permanent)
                return sign_of(determinant);
        }
    }
    return exact_orientation(a, b, c);
}

PredicateSign incircle_sign(const Vec3& a, const Vec3& b,
                            const Vec3& c, const Vec3& p)
{
    check_finite(a); check_finite(b); check_finite(c); check_finite(p);
    // Shewchuk's incircle A filter. Every difference must be in the
    // conservative normal range before its certified bound is applied.
    if (filter_environment()
        && filter_coordinate(a.x) && filter_coordinate(a.y)
        && filter_coordinate(b.x) && filter_coordinate(b.y)
        && filter_coordinate(c.x) && filter_coordinate(c.y)
        && filter_coordinate(p.x) && filter_coordinate(p.y))
    {
        const double ax = a.x - p.x, ay = a.y - p.y;
        const double bx = b.x - p.x, by = b.y - p.y;
        const double cx = c.x - p.x, cy = c.y - p.y;
        if (filter_factor(ax) && filter_factor(ay)
            && filter_factor(bx) && filter_factor(by)
            && filter_factor(cx) && filter_factor(cy))
        {
            const double ab = ax * by - bx * ay;
            const double bc = bx * cy - cx * by;
            const double ca = cx * ay - ax * cy;
            const double alift = ax * ax + ay * ay;
            const double blift = bx * bx + by * by;
            const double clift = cx * cx + cy * cy;
            const double determinant = alift * bc + blift * ca + clift * ab;
            const double permanent =
                (std::abs(bx * cy) + std::abs(cx * by)) * alift
              + (std::abs(cx * ay) + std::abs(ax * cy)) * blift
              + (std::abs(ax * by) + std::abs(bx * ay)) * clift;
            constexpr double epsilon = 0x1p-53;
            constexpr double bound = (10.0 + 96.0 * epsilon) * epsilon;
            if (std::abs(determinant) > bound * permanent)
                return sign_of(determinant);
        }
    }
    return exact_incircle(a, b, c, p);
}

PredicateSign polygon_area_sign(const Arr<Vec3>& polygon)
{
    Dyadic area{0, 0};
    for (size_t i = 0; i < polygon.size(); ++i)
    {
        const auto& a = polygon[i];
        const auto& b = polygon[(i + 1) % polygon.size()];
        check_finite(a); check_finite(b);
        area = add(area, subtract(multiply(exact(a.x), exact(b.y)),
                                  multiply(exact(a.y), exact(b.x))));
    }
    return sign_of(area.numerator);
}
} // namespace SimpleFluid::Meshes::detail
