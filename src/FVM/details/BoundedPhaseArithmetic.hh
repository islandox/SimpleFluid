/** @file BoundedPhaseArithmetic.hh
 * @brief Mode-selected finite phase arithmetic and explicit bounded certification.
 */
#pragma once
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Finite arithmetic requires strict caller floating-point compilation."
#endif
#include "SimpleFluidExport.hh"
#include "geometry/BoundedConvexGeometry.hh"
#include "utils/CompensatedSum.hh"
#include "utils/BoundedFloatingPoint.hh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
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
    if (!std::isfinite(v.value) || !std::isfinite(v.absolute_error) || v.absolute_error < 0 ||
        (v.arithmetic_mode != ArithmeticMode::Ordinary && v.arithmetic_mode != ArithmeticMode::BoundedCertification))
        throw std::invalid_argument("Finite VoF arithmetic input is nonfinite or has an invalid arithmetic tag.");
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
inline Value make_value(ArithmeticMode mode, long double value, long double error = 0)
{
    Value result{value, mode == ArithmeticMode::Ordinary ? 0 : error, mode};
    if (!std::isfinite(error) || error < 0)
        throw std::invalid_argument("Finite VoF arithmetic input has an invalid error field.");
    validate(result);
    return result;
}
inline ArithmeticMode arithmetic_mode(Value a, Value b)
{
    validate(a);
    validate(b);
    if (a.arithmetic_mode == b.arithmetic_mode)
        return a.arithmetic_mode;
    // A certified singleton is an exact represented input constant. It can be
    // consumed by ordinary arithmetic without discarding input uncertainty.
    const auto& certified = a.arithmetic_mode == ArithmeticMode::BoundedCertification ? a : b;
    if (certified.absolute_error == 0)
        return ArithmeticMode::Ordinary;
    throw std::invalid_argument("Finite VoF arithmetic modes differ without an explicit conversion.");
}
inline long double lower(Value value)
{
    validate(value);
    return value.arithmetic_mode == ArithmeticMode::Ordinary ? value.value : value.lower();
}
inline long double upper(Value value)
{
    validate(value);
    return value.arithmetic_mode == ArithmeticMode::Ordinary ? value.value : value.upper();
}
inline bool closure_within(Value residual, long double tolerance, long double source_error = 0)
{
    validate(residual);
    if (!std::isfinite(tolerance) || tolerance < 0 || !std::isfinite(source_error) || source_error < 0)
        throw std::invalid_argument("Finite VoF closure tolerance is invalid.");
    if (residual.arithmetic_mode == ArithmeticMode::Ordinary)
        return std::abs(residual.value) <= tolerance;
    return std::abs(residual.value) <= residual.absolute_error &&
           residual.absolute_error <= bound_add(tolerance, source_error);
}
namespace certified
{
SIMPLEFLUID_FVM_EXPORT Value add(Value a, Value b);
SIMPLEFLUID_FVM_EXPORT Value negate(Value a);
SIMPLEFLUID_FVM_EXPORT Value subtract(Value a, Value b);
SIMPLEFLUID_FVM_EXPORT Value multiply(Value a, Value b);
SIMPLEFLUID_FVM_EXPORT Value divide(Value a, Value b);
SIMPLEFLUID_FVM_EXPORT void admit_receipt(Value amount, double relative_tolerance, const char* phase);
SIMPLEFLUID_FVM_EXPORT std::pair<double, double> publish(Value value);
} // namespace certified
namespace ordinary
{
SIMPLEFLUID_FVM_EXPORT Value add(Value a, Value b);
SIMPLEFLUID_FVM_EXPORT Value negate(Value a);
SIMPLEFLUID_FVM_EXPORT Value multiply(Value a, Value b);
SIMPLEFLUID_FVM_EXPORT Value divide(Value a, Value b);
SIMPLEFLUID_FVM_EXPORT void admit_receipt(Value amount, double relative_tolerance, const char*);
SIMPLEFLUID_FVM_EXPORT std::pair<double, double> publish(Value value);
} // namespace ordinary
inline Value add(Value a, Value b)
{
    return arithmetic_mode(a, b) == ArithmeticMode::Ordinary ? ordinary::add(a, b) : certified::add(a, b);
}
inline Value negate(Value a)
{
    validate(a);
    return a.arithmetic_mode == ArithmeticMode::Ordinary ? ordinary::negate(a) : certified::negate(a);
}
inline Value subtract(Value a, Value b) { return add(a, negate(b)); }
inline Value multiply(Value a, Value b)
{
    return arithmetic_mode(a, b) == ArithmeticMode::Ordinary ? ordinary::multiply(a, b) : certified::multiply(a, b);
}
inline Value divide(Value a, Value b)
{
    return arithmetic_mode(a, b) == ArithmeticMode::Ordinary ? ordinary::divide(a, b) : certified::divide(a, b);
}
inline void admit_receipt(Value value, double tolerance, const char* phase)
{
    validate(value);
    if (value.arithmetic_mode == ArithmeticMode::Ordinary)
        ordinary::admit_receipt(value, tolerance, phase);
    else
        certified::admit_receipt(value, tolerance, phase);
}
inline std::pair<double, double> publish(Value value)
{
    validate(value);
    return value.arithmetic_mode == ArithmeticMode::Ordinary ? ordinary::publish(value) : certified::publish(value);
}
/** Compensated accumulation; ordinary sums retain represented values without
 * claiming a certified radius. An explicit mode is used at state boundaries. */
class Sum
{
public:
    Sum() = default;
    explicit Sum(ArithmeticMode mode) : mode_(mode), explicit_mode_(true)
    {
        (void) make_value(mode, 0);
    }
    void add(Value value)
    {
        Meshes::bounded_record_ledger_operations();
        validate(value);
        if (!mode_)
        {
            if (value.arithmetic_mode == ArithmeticMode::BoundedCertification &&
                value.value == 0 && value.absolute_error == 0)
                return;
            mode_ = value.arithmetic_mode;
        }
        else if (*mode_ != value.arithmetic_mode)
        {
            if (*mode_ == ArithmeticMode::Ordinary && value.absolute_error == 0)
                value = make_value(*mode_, value.value);
            else if (!explicit_mode_ && value.arithmetic_mode == ArithmeticMode::Ordinary && !input_uncertainty_)
            {
                mode_ = ArithmeticMode::Ordinary;
                error_ = 0;
            }
            else
                throw std::invalid_argument("Finite VoF sum mixes arithmetic modes.");
        }
        if (value.value == 0 && value.absolute_error == 0) return;
        const auto [sum, residual] = two_sum(high_, value.value);
        const auto [correction, discarded] = two_sum(low_, residual);
        const auto normalized = two_sum(sum, correction);
        high_ = normalized.first;
        low_ = normalized.second;
        if (*mode_ == ArithmeticMode::BoundedCertification)
        {
            input_uncertainty_ |= value.absolute_error != 0;
            error_ = bound_add(bound_add(error_, value.absolute_error), std::abs(discarded));
        }
        if (++count_ > (std::uint64_t(1) << 52))
            throw std::invalid_argument("Finite VoF accumulation operation bound is exhausted.");
    }
    [[nodiscard]] Value value() const
    {
        const auto [sum, residual] = two_sum(high_, low_);
        const auto mode = mode_.value_or(ArithmeticMode::BoundedCertification);
        return make_value(mode, sum,
            mode == ArithmeticMode::BoundedCertification ? bound_add(error_, std::abs(residual)) : 0);
    }
private:
    long double high_ = 0, low_ = 0, error_ = 0;
    std::uint64_t count_ = 0;
    std::optional<ArithmeticMode> mode_;
    bool explicit_mode_ = false, input_uncertainty_ = false;
};
inline Value component(const Point& point, unsigned axis)
{
    return {point.value[axis], point.absolute_error[axis], point.arithmetic_mode};
}
inline Point add(Point a, const Point& b)
{
    const auto input_mode = a.arithmetic_mode;
    for (unsigned k = 0; k < 3; ++k)
    {
        const auto v = add(Value{a.value[k], a.absolute_error[k], input_mode}, component(b, k));
        a.value[k] = v.value;
        a.absolute_error[k] = v.absolute_error;
        a.arithmetic_mode = v.arithmetic_mode;
    }
    return a;
}
inline Point scale(Point a, Value factor)
{
    const auto input_mode = a.arithmetic_mode;
    for (unsigned k = 0; k < 3; ++k)
    {
        const auto v = multiply(Value{a.value[k], a.absolute_error[k], input_mode}, factor);
        a.value[k] = v.value;
        a.absolute_error[k] = v.absolute_error;
        a.arithmetic_mode = v.arithmetic_mode;
    }
    return a;
}
inline Point subtract(Point a, Point b) { return add(a, scale(b, {-1, 0})); }
inline Point point(Meshes::GeometryPoint p, ArithmeticMode mode = ArithmeticMode::BoundedCertification)
{
    return {{p[0], p[1], p[2]}, {0, 0, 0}, mode};
}
inline Value dot(const Point& a, const Point& b)
{
    Sum result;
    for (unsigned k = 0; k < 3; ++k)
        result.add(multiply(component(a, k), component(b, k)));
    return result.value();
}
inline Point cross(const Point& a, const Point& b)
{
    Point result;
    for (unsigned k = 0; k < 3; ++k)
    {
        const auto i = (k + 1) % 3, j = (k + 2) % 3;
        const auto v = subtract(multiply(component(a, i), component(b, j)),
            multiply(component(a, j), component(b, i)));
        result.value[k] = v.value;
        result.absolute_error[k] = v.absolute_error;
        result.arithmetic_mode = v.arithmetic_mode;
    }
    return result;
}

/** Fixed-mode arithmetic for numerical kernels. Runtime adapters above are
 * retained for externally tagged values and compatibility callers. */
template<ArithmeticMode Mode> struct Arithmetic
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    static void input(Value value)
    {
        validate(value);
        if (value.arithmetic_mode != Mode &&
            !(Mode == ArithmeticMode::Ordinary && value.absolute_error == 0))
            throw std::invalid_argument("Finite VoF arithmetic modes differ without an explicit conversion.");
    }
    static Value make_value(long double value, long double error = 0)
    {
        if (!std::isfinite(error) || error < 0)
            throw std::invalid_argument("Finite VoF arithmetic input has an invalid error field.");
        Value result{value, 0, Mode};
        if constexpr (Mode == ArithmeticMode::BoundedCertification) result.absolute_error = error;
        validate(result);
        return result;
    }
    static Value make_value(ArithmeticMode mode, long double value, long double error = 0)
    {
        if (mode != Mode)
            throw std::invalid_argument("Finite VoF value arithmetic mode differs from its kernel.");
        return make_value(value, error);
    }
    static long double lower(Value value)
    {
        input(value);
        if constexpr (Mode == ArithmeticMode::Ordinary) return value.value;
        else return value.template lower<Mode>();
    }
    static long double upper(Value value)
    {
        input(value);
        if constexpr (Mode == ArithmeticMode::Ordinary) return value.value;
        else return value.template upper<Mode>();
    }
    static double as_double(Value value)
    {
        validate(value);
        if (value.arithmetic_mode != Mode)
            throw std::invalid_argument("Finite VoF publication arithmetic mode differs from its kernel.");
        return value.template as_double<Mode>();
    }
    static bool closure_within(Value value, long double tolerance, long double source_error = 0)
    {
        input(value);
        if (!std::isfinite(tolerance) || tolerance < 0 || !std::isfinite(source_error) || source_error < 0)
            throw std::invalid_argument("Finite VoF closure tolerance is invalid.");
        if constexpr (Mode == ArithmeticMode::Ordinary) return std::abs(value.value) <= tolerance;
        else return std::abs(value.value) <= value.absolute_error &&
                    value.absolute_error <= bound_add(tolerance, source_error);
    }
    static Value add(Value a, Value b)
    {
        input(a); input(b);
        if constexpr (Mode == ArithmeticMode::Ordinary) return ordinary::add(a, b);
        else return certified::add(a, b);
    }
    static Value negate(Value a)
    {
        input(a);
        if constexpr (Mode == ArithmeticMode::Ordinary)
        {
            // Singleton provenance controls accounting only. Ordinary events
            // remain in the compiled FVM arithmetic domain.
            if (a.arithmetic_mode == ArithmeticMode::BoundedCertification)
                return {-a.value, a.absolute_error, a.arithmetic_mode};
            return ordinary::negate(a);
        }
        else return certified::negate(a);
    }
    static Value subtract(Value a, Value b) { return add(a, negate(b)); }
    static Value multiply(Value a, Value b)
    {
        input(a); input(b);
        if constexpr (Mode == ArithmeticMode::Ordinary) return ordinary::multiply(a, b);
        else return certified::multiply(a, b);
    }
    static Value divide(Value a, Value b)
    {
        input(a); input(b);
        if constexpr (Mode == ArithmeticMode::Ordinary) return ordinary::divide(a, b);
        else return certified::divide(a, b);
    }
    static void admit_receipt(Value value, double tolerance, const char* phase)
    {
        input(value);
        if constexpr (Mode == ArithmeticMode::Ordinary) ordinary::admit_receipt(value, tolerance, phase);
        else certified::admit_receipt(value, tolerance, phase);
    }
    static std::pair<double, double> publish(Value value)
    {
        input(value);
        if constexpr (Mode == ArithmeticMode::Ordinary) return ordinary::publish(value);
        else return certified::publish(value);
    }
    static Value component(const Point& p, unsigned axis)
    { return {p.value[axis], p.absolute_error[axis], p.arithmetic_mode}; }
    static Point point(Meshes::GeometryPoint p) { return {{p[0], p[1], p[2]}, {0, 0, 0}, Mode}; }
    static Point add(Point a, const Point& b)
    {
        const auto mode = a.arithmetic_mode;
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const auto value = add(Value{a.value[axis], a.absolute_error[axis], mode}, component(b, axis));
            a.value[axis] = value.value;
            a.absolute_error[axis] = value.absolute_error;
            a.arithmetic_mode = Mode;
        }
        return a;
    }
    static Point scale(Point a, Value factor)
    {
        const auto mode = a.arithmetic_mode;
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const auto value = multiply(Value{a.value[axis], a.absolute_error[axis], mode}, factor);
            a.value[axis] = value.value;
            a.absolute_error[axis] = value.absolute_error;
            a.arithmetic_mode = Mode;
        }
        return a;
    }
    static Point subtract(Point a, Point b) { return add(a, scale(b, make_value(-1))); }
    static Value dot(const Point& a, const Point& b);
    static Point cross(const Point& a, const Point& b)
    {
        Point result;
        result.arithmetic_mode = Mode;
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const auto i = (axis + 1) % 3, j = (axis + 2) % 3;
            const auto value = subtract(multiply(component(a, i), component(b, j)),
                multiply(component(a, j), component(b, i)));
            result.value[axis] = value.value;
            result.absolute_error[axis] = value.absolute_error;
        }
        return result;
    }
};

template<ArithmeticMode Mode> class SumFor
{
public:
    SumFor() = default;
    explicit SumFor(ArithmeticMode mode) : reserved_explicit_mode_(true)
    {
        if (mode != Mode)
            throw std::invalid_argument("Finite VoF sum arithmetic mode differs from its kernel.");
    }
    void add(Value value)
    {
        Meshes::bounded_record_ledger_operations();
        Arithmetic<Mode>::input(value);
        if (!reserved_explicit_mode_ && !reserved_input_uncertainty_ &&
            value.arithmetic_mode == ArithmeticMode::BoundedCertification &&
            value.value == 0 && value.absolute_error == 0)
            return; // Preserve the inferred sum's neutral certified singleton.
        if constexpr (Mode == ArithmeticMode::Ordinary)
            if (!reserved_explicit_mode_ && !reserved_input_uncertainty_ &&
                value.arithmetic_mode != Mode && value.value != 0)
                throw std::invalid_argument("An inferred ordinary kernel sum requires mode-tagged nonzero inputs.");
        reserved_input_uncertainty_ = true;
        if (value.value == 0 && value.absolute_error == 0) return;
        const auto [sum, residual] = two_sum(high_, value.value);
        const auto [correction, discarded] = two_sum(low_, residual);
        const auto normalized = two_sum(sum, correction);
        high_ = normalized.first;
        low_ = normalized.second;
        if constexpr (Mode == ArithmeticMode::BoundedCertification)
            error_ = bound_add(bound_add(error_, value.absolute_error), std::abs(discarded));
        if (++count_ > (std::uint64_t(1) << 52))
            throw std::invalid_argument("Finite VoF accumulation operation bound is exhausted.");
    }
    Value value() const
    {
        if (!reserved_explicit_mode_ && !reserved_input_uncertainty_)
            return {0, 0, ArithmeticMode::BoundedCertification};
        const auto [sum, residual] = two_sum(high_, low_);
        if constexpr (Mode == ArithmeticMode::BoundedCertification)
            return Arithmetic<Mode>::make_value(sum, bound_add(error_, std::abs(residual)));
        else return Arithmetic<Mode>::make_value(sum);
    }
private:
    long double high_ = 0, low_ = 0, error_ = 0;
    std::uint64_t count_ = 0;
    [[maybe_unused]] std::optional<ArithmeticMode> reserved_mode_;
    bool reserved_explicit_mode_ = false, reserved_input_uncertainty_ = false;
};

template<ArithmeticMode Mode> Value Arithmetic<Mode>::dot(const Point& a, const Point& b)
{
    SumFor<Mode> result;
    for (unsigned axis = 0; axis < 3; ++axis)
        result.add(multiply(component(a, axis), component(b, axis)));
    return result.value();
}
} // namespace SimpleFluid::VoF::finite_detail
