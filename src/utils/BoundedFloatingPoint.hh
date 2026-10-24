/** @file BoundedFloatingPoint.hh
 * @brief Error-free scalar transforms and bounded floating-point arithmetic.
 */
#pragma once
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Bounded floating-point arithmetic requires strict floating-point compilation."
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail::bounded_fp
{
inline long double upward(long double x)
{
    if (!std::isfinite(x) || x < 0)
        throw std::invalid_argument("Finite VoF arithmetic error is outside its qualified range.");
    return x == 0 ? 0 : std::nextafter(x, std::numeric_limits<long double>::infinity());
}
inline long double rounding(long double x)
{
    if (!std::isfinite(x))
        throw std::invalid_argument("Finite VoF arithmetic result is outside its qualified range.");
    return std::max(std::abs(std::nextafter(x, std::numeric_limits<long double>::infinity()) - x),
        std::abs(x - std::nextafter(x, -std::numeric_limits<long double>::infinity())));
}
inline long double bound_add(long double a, long double b)
{
    if (a == 0) return b;
    if (b == 0) return a;
    return upward(a + b);
}
inline long double bound_multiply(long double a, long double b)
{
    if (a == 0 || b == 0) return 0;
    const auto product = a * b;
    if (!std::isfinite(product))
        throw std::invalid_argument("Finite VoF arithmetic bound overflows.");
    return std::nextafter(product, std::numeric_limits<long double>::infinity());
}
inline std::pair<long double, long double> two_sum(long double a, long double b)
{
    const auto sum = a + b;
    if (!std::isfinite(sum))
        throw std::invalid_argument("Finite VoF sum exceeds qualified range.");
    const auto z = sum - a;
    return {sum, (a - (sum - z)) + (b - z)};
}
inline long double product_residual_bound(long double a, long double b, long double product)
{
    const auto residual = std::fma(a, b, -product);
    if (!std::isfinite(residual))
        throw std::invalid_argument("Finite VoF product residual exceeds qualified range.");
    if (product == 0 || std::abs(product) <
            std::scalbn(std::numeric_limits<long double>::min(), std::numeric_limits<long double>::digits))
        return bound_add(std::abs(residual), std::numeric_limits<long double>::denorm_min());
    return residual == 0 ? 0 : upward(std::abs(residual));
}
/** Value provides represented value and nonnegative absolute_error fields.
 * The caller validates its domain and retains its own operation accounting.
 */
template<class Value> inline Value add(Value a, Value b)
{
    if (a.value == 0 && a.absolute_error == 0) return b;
    if (b.value == 0 && b.absolute_error == 0) return a;
    const auto [value, residual] = two_sum(a.value, b.value);
    return {value, bound_add(bound_add(a.absolute_error, b.absolute_error), std::abs(residual))};
}
template<class Value> inline Value negate(Value a)
{ return {-a.value, a.absolute_error}; }
template<class Value> inline Value subtract(Value a, Value b)
{ return bounded_fp::add(a, bounded_fp::negate(b)); }
template<class Value> inline Value multiply(Value a, Value b)
{
    if ((a.value == 0 && a.absolute_error == 0) || (b.value == 0 && b.absolute_error == 0)) return {};
    const auto value = a.value * b.value;
    auto error = bound_add(bound_multiply(std::abs(a.value), b.absolute_error),
        bound_multiply(std::abs(b.value), a.absolute_error));
    error = bound_add(error, bound_multiply(a.absolute_error, b.absolute_error));
    return {value, bound_add(error, product_residual_bound(a.value, b.value, value))};
}
template<class Value> inline Value divide(Value a, Value b)
{
    const auto lower = b.absolute_error == 0 ? std::abs(b.value) :
        std::nextafter(std::abs(b.value) - b.absolute_error, -std::numeric_limits<long double>::infinity());
    if (!(lower > 0))
        throw std::invalid_argument("Finite VoF division has an unresolved denominator.");
    if (a.value == 0 && a.absolute_error == 0) return {};
    const auto value = a.value / b.value;
    if (!std::isfinite(value))
        throw std::invalid_argument("Finite VoF quotient exceeds qualified range.");
    const auto residual = std::fma(-value, b.value, a.value);
    long double residual_bound = residual == 0 ? 0 : upward(std::abs(residual));
    if (std::abs(a.value) < std::scalbn(std::numeric_limits<long double>::min(),
            std::numeric_limits<long double>::digits))
        residual_bound = bound_add(residual_bound, std::numeric_limits<long double>::denorm_min());
    const auto numerator = bound_add(bound_add(a.absolute_error,
        bound_multiply(std::abs(value), b.absolute_error)), residual_bound);
    const auto propagated = numerator == 0 ? 0 :
        std::nextafter(numerator / lower, std::numeric_limits<long double>::infinity());
    return {value, propagated};
}
} // namespace SimpleFluid::detail::bounded_fp
