/** @file BoundedInterval.hh
 * @brief Policy-based bounded floating-point interval arithmetic.
 */
#pragma once
#include "utils/DirectedFloatingPoint.hh"
#include <algorithm>
#include <array>
#include <numeric>

namespace SimpleFluid::detail::directed_fp
{
template<class Policy> struct Interval
{
    long double lo = 0, hi = 0, estimate = 0;
    explicit Interval(long double value = 0) : lo(value), hi(value), estimate(value) { Policy::finite(value); }
    Interval(long double lower, long double upper) : lo(lower), hi(upper), estimate(std::midpoint(lower, upper))
    {
        Policy::finite(lo);
        Policy::finite(hi);
        if (lo > hi)
            Policy::reject("invalid enclosure.");
    }
    bool zero() const { return lo == 0 && hi == 0; }
};

template<class Policy> Interval<Policy> operator-(Interval<Policy> a)
{
    return {-a.hi, -a.lo};
}

template<class Policy> Interval<Policy> operator+(Interval<Policy> a, Interval<Policy> b)
{
    Policy::interval_operation();
    return {directed_add<Policy>(a.lo, b.lo, false), directed_add<Policy>(a.hi, b.hi, true)};
}

template<class Policy> Interval<Policy> operator-(Interval<Policy> a, Interval<Policy> b)
{
    return a + -b;
}

template<class Policy> Interval<Policy> operator*(Interval<Policy> a, Interval<Policy> b)
{
    Policy::interval_operation();
    const std::array low{directed_multiply<Policy>(a.lo, b.lo, false), directed_multiply<Policy>(a.lo, b.hi, false),
        directed_multiply<Policy>(a.hi, b.lo, false), directed_multiply<Policy>(a.hi, b.hi, false)};
    const std::array high{directed_multiply<Policy>(a.lo, b.lo, true), directed_multiply<Policy>(a.lo, b.hi, true),
        directed_multiply<Policy>(a.hi, b.lo, true), directed_multiply<Policy>(a.hi, b.hi, true)};
    return {*std::min_element(low.begin(), low.end()), *std::max_element(high.begin(), high.end())};
}

template<class Policy> Interval<Policy> operator/(Interval<Policy> a, Interval<Policy> b)
{
    Policy::interval_operation();
    if (b.lo <= 0 && b.hi >= 0)
        Policy::reject("division by an enclosure containing zero.");
    const std::array low{directed_divide<Policy>(a.lo, b.lo, false), directed_divide<Policy>(a.lo, b.hi, false),
        directed_divide<Policy>(a.hi, b.lo, false), directed_divide<Policy>(a.hi, b.hi, false)};
    const std::array high{directed_divide<Policy>(a.lo, b.lo, true), directed_divide<Policy>(a.lo, b.hi, true),
        directed_divide<Policy>(a.hi, b.lo, true), directed_divide<Policy>(a.hi, b.hi, true)};
    return {*std::min_element(low.begin(), low.end()), *std::max_element(high.begin(), high.end())};
}

template<class Policy> Interval<Policy> scaled(Interval<Policy> a, int exponent)
{
    Policy::interval_operation();
    const auto lo = std::scalbn(a.lo, exponent), hi = std::scalbn(a.hi, exponent);
    Policy::finite(lo);
    Policy::finite(hi);
    if ((lo == 0 && a.lo != 0) || (hi == 0 && a.hi != 0))
        Policy::reject("nonzero enclosure below wider-FP range.");
    Interval<Policy> result(lo, hi);
    result.estimate = std::scalbn(a.estimate, exponent);
    return result; // A power-of-two scaling is exact in the admitted normal range.
}
} // namespace SimpleFluid::detail::directed_fp
