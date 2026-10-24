/** @file BoundedPhaseArithmetic.hh
 * @brief Fixed-storage floating-point phase arithmetic with propagated absolute error.
 */
#pragma once
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Bounded finite arithmetic requires strict caller floating-point compilation."
#endif
#include "SimpleFluidExport.hh"
#include "geometry/BoundedConvexGeometry.hh"
#include "utils/CompensatedSum.hh"
#include "utils/BoundedFloatingPoint.hh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace SimpleFluid::VoF::finite_detail
{
using Value = Meshes::BoundedGeometryValue;
using Point = Meshes::BoundedPoint;
using SimpleFluid::detail::bounded_fp::upward;
using SimpleFluid::detail::bounded_fp::rounding;
using SimpleFluid::detail::bounded_fp::bound_add;
using SimpleFluid::detail::bounded_fp::bound_multiply;
/** Error-free addition on the qualified round-to-nearest gradual-underflow route. */
using SimpleFluid::detail::bounded_fp::two_sum;
using SimpleFluid::detail::bounded_fp::product_residual_bound;
inline void validate(Value v)
{
    if (!std::isfinite(v.value) || !std::isfinite(v.absolute_error) || v.absolute_error < 0)
        throw std::invalid_argument("Finite VoF arithmetic input is nonfinite or has an invalid error bound.");
}
SIMPLEFLUID_FVM_EXPORT Value add(Value a, Value b);
SIMPLEFLUID_FVM_EXPORT Value negate(Value a);
SIMPLEFLUID_FVM_EXPORT Value subtract(Value a, Value b);
SIMPLEFLUID_FVM_EXPORT Value multiply(Value a, Value b);
SIMPLEFLUID_FVM_EXPORT Value divide(Value a, Value b);
inline double upper_double(long double value)
{
    if (!std::isfinite(value) || value < 0 || value > std::numeric_limits<double>::max())
        throw std::invalid_argument("Finite VoF arithmetic bound has no finite publication.");
    auto result = static_cast<double>(value);
    if (static_cast<long double>(result) < value)
        result = std::nextafter(result, std::numeric_limits<double>::infinity());
    return result;
}
SIMPLEFLUID_FVM_EXPORT std::pair<double, double> publish(Value value);
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
