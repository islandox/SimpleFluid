/** @file BoundedInterval.hh
 * @brief Policy-based bounded floating-point interval arithmetic.
 * Policy::ordinary() is a compile-time choice; callers dispatch runtime modes
 * before entering a numerical kernel. Policy hooks retain validation and work
 * accounting without selecting arithmetic inside an interval operation.
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
    // Mode-specific kernels share the same persisted interval representation.
    // Rebinding a previously validated interval is a copy, not new arithmetic.
    template<class OtherPolicy>
    Interval(const Interval<OtherPolicy>& other) : lo(other.lo), hi(other.hi), estimate(other.estimate)
    {
    }
    bool zero() const { return lo == 0 && hi == 0; }
};

template<class Policy> Interval<Policy> operator-(Interval<Policy> a)
{
    if constexpr (Policy::ordinary())
        return Interval<Policy>(-a.estimate);
    return {-a.hi, -a.lo};
}

template<class Policy> Interval<Policy> operator+(Interval<Policy> a, Interval<Policy> b)
{
    Policy::interval_operation();
    if constexpr (Policy::ordinary())
    {
        Policy::operation();
        return Interval<Policy>(a.estimate + b.estimate);
    }
    return {directed_add<Policy>(a.lo, b.lo, false), directed_add<Policy>(a.hi, b.hi, true)};
}

template<class Policy> Interval<Policy> operator-(Interval<Policy> a, Interval<Policy> b)
{
    return a + -b;
}

template<class Policy> Interval<Policy> operator*(Interval<Policy> a, Interval<Policy> b)
{
    Policy::interval_operation();
    if constexpr (Policy::ordinary())
    {
        Policy::operation();
        return Interval<Policy>(a.estimate * b.estimate);
    }
    const std::array low{directed_multiply<Policy>(a.lo, b.lo, false), directed_multiply<Policy>(a.lo, b.hi, false),
        directed_multiply<Policy>(a.hi, b.lo, false), directed_multiply<Policy>(a.hi, b.hi, false)};
    const std::array high{directed_multiply<Policy>(a.lo, b.lo, true), directed_multiply<Policy>(a.lo, b.hi, true),
        directed_multiply<Policy>(a.hi, b.lo, true), directed_multiply<Policy>(a.hi, b.hi, true)};
    return {*std::min_element(low.begin(), low.end()), *std::max_element(high.begin(), high.end())};
}

template<class Policy> Interval<Policy> operator/(Interval<Policy> a, Interval<Policy> b)
{
    Policy::interval_operation();
    if constexpr (Policy::ordinary())
    {
        if (b.estimate == 0)
            Policy::reject("zero divisor in ordinary geometry.");
        Policy::operation();
        return Interval<Policy>(a.estimate / b.estimate);
    }
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
    if constexpr (Policy::ordinary())
    {
        Policy::operation();
        return Interval<Policy>(std::scalbn(a.estimate, exponent));
    }
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
