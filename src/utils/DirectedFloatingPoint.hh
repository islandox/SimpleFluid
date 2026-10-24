/** @file DirectedFloatingPoint.hh
 * @brief Directed scalar floating-point rounding with caller-owned policy hooks.
 */
#pragma once
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Directed floating-point arithmetic requires strict floating-point compilation."
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace SimpleFluid::detail::directed_fp
{
inline constexpr long double infinity = std::numeric_limits<long double>::infinity();

template<class Policy> long double directed_add(long double a, long double b, bool up)
{
    Policy::operation();
    const long double sum = a + b;
    Policy::finite(sum);
    const long double z = sum - a;
    const long double residual = (a - (sum - z)) + (b - z);
    Policy::finite(residual);
    if ((up && residual > 0) || (!up && residual < 0))
    {
        const auto result = std::nextafter(sum, up ? infinity : -infinity);
        Policy::finite(result);
        return result;
    }
    return sum;
}

template<class Policy> long double directed_multiply(long double a, long double b, bool up)
{
    Policy::operation();
    const long double product = a * b;
    Policy::finite(product);
    if (a != 0 && b != 0 && product == 0)
        Policy::reject("nonzero product below wider-FP range.");
    // In the bottom significand-width of the normal range, an FMA residual
    // can itself underflow. Outward rounding there must not infer exactness.
    if (product != 0 && std::abs(product) < std::scalbn(std::numeric_limits<long double>::min(),
                                                std::numeric_limits<long double>::digits))
        return std::nextafter(product, up ? infinity : -infinity);
    const long double residual = std::fma(a, b, -product);
    Policy::finite(residual);
    if ((up && residual > 0) || (!up && residual < 0))
    {
        const auto result = std::nextafter(product, up ? infinity : -infinity);
        Policy::finite(result);
        return result;
    }
    return product;
}

template<class Policy> long double directed_divide(long double a, long double b, bool up)
{
    Policy::operation();
    if (b == 0)
        Policy::reject("uncertain or zero divisor.");
    const long double quotient = a / b;
    Policy::finite(quotient);
    if (a != 0 && quotient == 0)
        Policy::reject("nonzero quotient below wider-FP range.");
    if (a != 0 &&
        std::abs(a) < std::scalbn(std::numeric_limits<long double>::min(), std::numeric_limits<long double>::digits))
        return std::nextafter(quotient, up ? infinity : -infinity);
    const long double residual = std::fma(-quotient, b, a);
    Policy::finite(residual);
    const bool above = (residual > 0) == (b > 0);
    if (residual != 0 && ((up && above) || (!up && !above)))
    {
        const auto result = std::nextafter(quotient, up ? infinity : -infinity);
        Policy::finite(result);
        return result;
    }
    return quotient;
}

} // namespace SimpleFluid::detail::directed_fp
