#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace SimpleFluid::PLIC::detail
{
/** A closed enclosure of the represented rational, never a replacement value.
 * Invalid intervals ask the caller to use its original exact predicate.
 * The fixed-size summaries own no rational/backend storage or heap allocations.
 */
struct CertifiedInterval
{
    double lower = 0, upper = 0;
    bool valid = false;
};
using CertifiedFilterPoint = std::array<CertifiedInterval, 3>;
struct CertifiedFilterPlane
{
    CertifiedFilterPoint normal, anchor;
    CertifiedInterval offset;
};

static inline bool certified_filter_environment()
{
    // The implementation relies on separately rounded binary64 operations.
    // GCC's verified IEC-559 indicator and unsafe-mode macros certify the
    // compiler mode. Clang does not advertise its individual unsafe modes,
    // so all Clang configurations conservatively use the exact fallback.
    // x86-64 SSE2 avoids x87 excess precision; volatile stores below also
    // prevent contraction of consecutive operations.
#if defined(__GNUC__) && !defined(__clang__) && defined(__GCC_IEC_559) && __GCC_IEC_559 >= 2 && defined(__x86_64__) && \
    defined(__SSE2__) && !defined(__FAST_MATH__) && (!defined(__FINITE_MATH_ONLY__) || __FINITE_MATH_ONLY__ == 0) &&   \
    !defined(__ASSOCIATIVE_MATH__) && !defined(__RECIPROCAL_MATH__) && !defined(__NO_SIGNED_ZEROS__) &&                \
    FLT_EVAL_METHOD == 0
    if constexpr (!std::numeric_limits<double>::is_iec559 || std::numeric_limits<double>::radix != 2 ||
                  std::numeric_limits<double>::digits != 53 || std::numeric_limits<double>::min_exponent != -1021 ||
                  std::numeric_limits<double>::max_exponent != 1024)
        return false;
    // These controls are thread-local. Reading them does not alter the process
    // environment. Ordinary filter arithmetic may set sticky exception flags;
    // it never changes control bits. Cached summaries cannot bypass a mode change.
    const unsigned csr = _mm_getcsr();
    unsigned short x87_control;
    __asm__ __volatile__("fnstcw %0" : "=m"(x87_control));
    // An unavailable filter must not trap while trying arithmetic that the Q
    // predicate never needed. Require all six SSE exception masks; also require
    // the x87 masks in case a supported libm implementation uses x87 internally.
    // Test these before calling further floating-point environment/library code.
    constexpr unsigned exception_masks = 0x3fu << 7;
    if ((csr & exception_masks) != exception_masks || (x87_control & 0x3fu) != 0x3fu)
        return false;
    return std::fegetround() == FE_TONEAREST && (csr & ((1u << 15) | (1u << 6) | (3u << 13))) == 0;
#else
    return false;
#endif
}

namespace certified_filter_impl
{
static inline bool normal_finite(double value)
{
    return std::isnormal(value) && std::isfinite(value);
}
static inline bool exact_zero(const CertifiedInterval& value)
{
    return value.valid && value.lower == 0 && value.upper == 0;
}
static inline CertifiedInterval zero()
{
    return {0, 0, true};
}

static inline CertifiedInterval outward(double lower, double upper)
{
    // For a finite normal correctly rounded result, its predecessor/successor
    // enclose the exact result. Reject zero, subnormal and nonfinite results,
    // including any outward endpoint that would enter those exceptional ranges.
    if (!normal_finite(lower) || !normal_finite(upper))
        return {};
    lower = std::nextafter(lower, -std::numeric_limits<double>::infinity());
    upper = std::nextafter(upper, std::numeric_limits<double>::infinity());
    if (!normal_finite(lower) || !normal_finite(upper))
        return {};
    return {lower, upper, true};
}
static inline CertifiedInterval negate(const CertifiedInterval& value)
{
    return value.valid ? CertifiedInterval{-value.upper, -value.lower, true} : CertifiedInterval{};
}
static inline CertifiedInterval add(const CertifiedInterval& a, const CertifiedInterval& b)
{
    if (!a.valid || !b.valid)
        return {};
    if (exact_zero(a))
        return b;
    if (exact_zero(b))
        return a;
    volatile double lower = a.lower + b.lower;
    volatile double upper = a.upper + b.upper;
    return outward(lower, upper);
}
static inline CertifiedInterval subtract(const CertifiedInterval& a, const CertifiedInterval& b)
{
    return add(a, negate(b));
}
static inline CertifiedInterval multiply(const CertifiedInterval& a, const CertifiedInterval& b)
{
    if (!a.valid || !b.valid)
        return {};
    if (exact_zero(a) || exact_zero(b))
        return zero();
    volatile double p00 = a.lower * b.lower;
    volatile double p01 = a.lower * b.upper;
    volatile double p10 = a.upper * b.lower;
    volatile double p11 = a.upper * b.upper;
    const std::array<double, 4> products{p00, p01, p10, p11};
    for (const auto product : products)
        if (!normal_finite(product))
            return {};
    const auto bounds = std::minmax_element(products.begin(), products.end());
    return outward(*bounds.first, *bounds.second);
}

struct LeadingInteger
{
    double lower, upper;
    std::size_t exponent;
};
template<class Backend> static LeadingInteger leading_integer(const Backend& magnitude)
{
    using Limb = boost::multiprecision::limb_type;
    constexpr std::size_t limb_bits = std::numeric_limits<Limb>::digits;
    const auto size = static_cast<std::size_t>(magnitude.size());
    Limb leading = magnitude.limbs()[size - 1];
    static_assert(limb_bits <= 64);
    const unsigned top_bits = 64u - std::countl_zero(static_cast<std::uint64_t>(leading));
    // cpp_int's canonical nonzero storage has a nonzero most-significant limb.
    // Its allocated limb count cannot overflow this bit index in addressable
    // storage (each limb occupies limb_bits/CHAR_BIT bytes).
    if (!top_bits || size - 1 > (std::numeric_limits<std::size_t>::max() - top_bits) / limb_bits)
        throw std::logic_error("Invalid exact rational integer storage.");
    const auto bits = (size - 1) * limb_bits + top_bits;
    const unsigned retained = bits < 53 ? static_cast<unsigned>(bits) : 53;
    std::uint64_t top = 0;
    if constexpr (limb_bits >= 53)
    {
        if (top_bits >= retained)
            top = static_cast<std::uint64_t>(leading >> (top_bits - retained));
        else
        {
            // Here size>=2 and 0<extra<=52<limb_bits. The top retained bits
            // occupy at most two limbs; no copied big integer is needed.
            const auto extra = retained - top_bits;
            top = (static_cast<std::uint64_t>(leading) << extra) |
                  static_cast<std::uint64_t>(magnitude.limbs()[size - 2] >> (limb_bits - extra));
        }
    }
    else
        for (unsigned i = 0; i < retained; ++i)
        {
            const auto bit = bits - 1 - i;
            top = (top << 1) | ((magnitude.limbs()[bit / limb_bits] >> (bit % limb_bits)) & Limb{1});
        }
    // For k=bits, r=retained and T=floor(M/2^(k-r)),
    // T*2^(1-r) <= M/2^(k-1) <= (T+1)*2^(1-r).
    // If k<=53 the lower value is exact. Both integers are <=2^53 and
    // normalization stays in [1,2], so conversion/scalbn are exact binary64.
    const auto lower = std::scalbn(static_cast<double>(top), 1 - static_cast<int>(retained));
    const auto upper = bits <= 53 ? lower : std::scalbn(static_cast<double>(top + 1), 1 - static_cast<int>(retained));
    return {lower, upper, bits - 1};
}

static inline CertifiedInterval rational_interval(const boost::multiprecision::cpp_rational& value)
{
    const auto& numerator = value.backend().num();
    const auto& denominator = value.backend().denom();
    if (!denominator.size() || denominator.sign() || (denominator.size() == 1 && denominator.limbs()[0] == 0))
        throw std::logic_error("Certified support filter requires a positive rational denominator.");
    if (numerator.size() == 1 && numerator.limbs()[0] == 0)
        return zero();
    const auto n = leading_integer(numerator), d = leading_integer(denominator);
    // Exponent cancellation is done before narrowing, so arbitrarily large
    // common numerator/denominator exponents require no temporary big integers.
    int exponent;
    if (n.exponent >= d.exponent)
    {
        const auto difference = n.exponent - d.exponent;
        if (difference > 2048)
            return {};
        exponent = static_cast<int>(difference);
    }
    else
    {
        const auto difference = d.exponent - n.exponent;
        if (difference > 2048)
            return {};
        exponent = -static_cast<int>(difference);
    }
    // Positive denominator: [n.lo/d.hi,n.hi/d.lo] encloses the normalized
    // exact quotient. Division operates in [1/2,2], safely away from underflow.
    volatile double lower = n.lower / d.upper;
    volatile double upper = n.upper / d.lower;
    auto quotient = outward(lower, upper);
    if (!quotient.valid)
        return {};
    int lower_exponent = 0, upper_exponent = 0;
    std::frexp(quotient.lower, &lower_exponent);
    std::frexp(quotient.upper, &upper_exponent);
    // Check the exact exponent before scaling. Testing only the rounded
    // result would miss a subnormal exact endpoint rounded up to min_normal.
    if (lower_exponent + exponent < std::numeric_limits<double>::min_exponent ||
        upper_exponent + exponent > std::numeric_limits<double>::max_exponent)
        return {};
    quotient.lower = std::scalbn(quotient.lower, exponent);
    quotient.upper = std::scalbn(quotient.upper, exponent);
    // Multiplication by a power of two is exact when the result is normal and
    // finite. Reject the entire conversion if either endpoint leaves that set.
    if (!normal_finite(quotient.lower) || !normal_finite(quotient.upper))
        return {};
    return numerator.sign() ? negate(quotient) : quotient;
}
static inline CertifiedFilterPoint point(const std::array<boost::multiprecision::cpp_rational, 3>& value)
{
    return {rational_interval(value[0]), rational_interval(value[1]), rational_interval(value[2])};
}
} // namespace certified_filter_impl

static inline CertifiedInterval certified_rational_interval(const boost::multiprecision::cpp_rational& value)
{
    return certified_filter_environment() ? certified_filter_impl::rational_interval(value) : CertifiedInterval{};
}
static inline CertifiedFilterPoint certified_filter_point(
    const std::array<boost::multiprecision::cpp_rational, 3>& value)
{
    return certified_filter_environment() ? certified_filter_impl::point(value) : CertifiedFilterPoint{};
}
static inline CertifiedFilterPlane certified_filter_plane(
    const std::array<boost::multiprecision::cpp_rational, 3>& normal,
    const std::array<boost::multiprecision::cpp_rational, 3>& anchor, const boost::multiprecision::cpp_rational& offset)
{
    if (!certified_filter_environment())
        return {};
    return {certified_filter_impl::point(normal), certified_filter_impl::point(anchor),
        certified_filter_impl::rational_interval(offset)};
}
static inline std::optional<int> try_certified_support_sign(
    const CertifiedFilterPlane& plane, const CertifiedFilterPoint& point)
{
    if (!certified_filter_environment())
        return std::nullopt;
    using namespace certified_filter_impl;
    auto sum = zero();
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        // Inactive axes contribute exactly zero even if their coordinates are
        // outside binary64. They must not force conversion into an exact Q.
        if (exact_zero(plane.normal[axis]))
            continue;
        sum = add(sum, multiply(plane.normal[axis], subtract(point[axis], plane.anchor[axis])));
        if (!sum.valid)
            return std::nullopt;
    }
    sum = subtract(sum, plane.offset);
    if (!sum.valid)
        return std::nullopt;
    if (sum.lower > 0)
        return 1;
    if (sum.upper < 0)
        return -1;
    return std::nullopt;
}
static inline std::optional<int> try_certified_support_sign(
    const CertifiedFilterPlane& plane, const std::array<boost::multiprecision::cpp_rational, 3>& point)
{
    if (!certified_filter_environment())
        return std::nullopt;
    return try_certified_support_sign(plane, certified_filter_impl::point(point));
}
} // namespace SimpleFluid::PLIC::detail
