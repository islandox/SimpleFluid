/** @file BoundedPhaseArithmetic.hh
 * @brief Fixed-storage floating-point phase arithmetic with propagated absolute error.
 */
#pragma once
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Bounded finite arithmetic requires strict caller floating-point compilation."
#endif
#include "geometry/BoundedConvexGeometry.hh"
#include "utils/CompensatedSum.hh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace SimpleFluid::VoF::finite_detail
{
using Value = Meshes::BoundedGeometryValue;
using Point = Meshes::BoundedPoint;
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
/** Error-free addition on the qualified round-to-nearest gradual-underflow route. */
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
inline void validate(Value v)
{
    if (!std::isfinite(v.value) || !std::isfinite(v.absolute_error) || v.absolute_error < 0)
        throw std::invalid_argument("Finite VoF arithmetic input is nonfinite or has an invalid error bound.");
}
inline Value add(Value a, Value b)
{
    Meshes::bounded_record_ledger_operations();
    validate(a); validate(b);
    if (a.value == 0 && a.absolute_error == 0) return b;
    if (b.value == 0 && b.absolute_error == 0) return a;
    const auto [value, residual] = two_sum(a.value, b.value);
    return {value, bound_add(bound_add(a.absolute_error, b.absolute_error), std::abs(residual))};
}
inline Value negate(Value a) { return {-a.value, a.absolute_error}; }
inline Value subtract(Value a, Value b) { return add(a, negate(b)); }
inline Value multiply(Value a, Value b)
{
    Meshes::bounded_record_ledger_operations();
    validate(a); validate(b);
    if ((a.value == 0 && a.absolute_error == 0) || (b.value == 0 && b.absolute_error == 0)) return {};
    const auto value = a.value * b.value;
    auto error = bound_add(bound_multiply(std::abs(a.value), b.absolute_error),
        bound_multiply(std::abs(b.value), a.absolute_error));
    error = bound_add(error, bound_multiply(a.absolute_error, b.absolute_error));
    return {value, bound_add(error, product_residual_bound(a.value, b.value, value))};
}
inline Value divide(Value a, Value b)
{
    Meshes::bounded_record_ledger_operations();
    validate(a); validate(b);
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
inline double upper_double(long double value)
{
    if (!std::isfinite(value) || value < 0 || value > std::numeric_limits<double>::max())
        throw std::invalid_argument("Finite VoF arithmetic bound has no finite publication.");
    auto result = static_cast<double>(value);
    if (static_cast<long double>(result) < value)
        result = std::nextafter(result, std::numeric_limits<double>::infinity());
    return result;
}
#include "FVM/details/BoundedPhaseCertification.inc"
/** Each term contributes its uncertainty even below binary64 range. */
class Sum
{
public:
    void add(Value value)
    {
        Meshes::bounded_record_ledger_operations();
        validate(value);
        if (value.value == 0 && value.absolute_error == 0) return;
        const auto [sum, residual] = two_sum(high_, value.value);
        const auto [correction, discarded] = two_sum(low_, residual);
        const auto normalized = two_sum(sum, correction);
        high_ = normalized.first;
        low_ = normalized.second;
        error_ = bound_add(bound_add(error_, value.absolute_error), std::abs(discarded));
        if (++count_ > (std::uint64_t(1) << 52))
            throw std::invalid_argument("Finite VoF accumulation operation bound is exhausted.");
    }
    [[nodiscard]] Value value() const
    {
        const auto [sum, residual] = two_sum(high_, low_);
        return {sum, bound_add(error_, std::abs(residual))};
    }
private:
    long double high_ = 0, low_ = 0, error_ = 0;
    std::uint64_t count_ = 0;
};

inline Point add(Point a, const Point& b)
{
    for (unsigned k = 0; k < 3; ++k)
    {
        auto v = add(Value{a.value[k], a.absolute_error[k]}, {b.value[k], b.absolute_error[k]});
        a.value[k] = v.value;
        a.absolute_error[k] = v.absolute_error;
    }
    return a;
}
inline Point scale(Point a, Value factor)
{
    for (unsigned k = 0; k < 3; ++k)
    {
        auto v = multiply(Value{a.value[k], a.absolute_error[k]}, factor);
        a.value[k] = v.value;
        a.absolute_error[k] = v.absolute_error;
    }
    return a;
}
inline Point subtract(Point a, Point b) { return add(a, scale(b, {-1, 0})); }
inline Point point(Meshes::GeometryPoint p) { return {{p[0], p[1], p[2]}, {0, 0, 0}}; }
inline Value dot(const Point& a, const Point& b)
{
    Sum result;
    for (unsigned k = 0; k < 3; ++k)
        result.add(multiply({a.value[k], a.absolute_error[k]}, {b.value[k], b.absolute_error[k]}));
    return result.value();
}
inline Point cross(const Point& a, const Point& b)
{
    Point result;
    for (unsigned k = 0; k < 3; ++k)
    {
        const auto i = (k + 1) % 3, j = (k + 2) % 3;
        const auto v = subtract(multiply({a.value[i], a.absolute_error[i]}, {b.value[j], b.absolute_error[j]}),
            multiply({a.value[j], a.absolute_error[j]}, {b.value[i], b.absolute_error[i]}));
        result.value[k] = v.value;
        result.absolute_error[k] = v.absolute_error;
    }
    return result;
}
} // namespace SimpleFluid::VoF::finite_detail
