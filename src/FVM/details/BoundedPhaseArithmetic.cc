/** @file BoundedPhaseArithmetic.cc
 * @brief Compiled finite scalar arithmetic and phase publication policies.
 */
#include "FVM/details/BoundedPhaseArithmetic.hh"

namespace SimpleFluid::VoF::finite_detail
{
namespace certified
{
Value add(Value a, Value b)
{
    Meshes::bounded_record_ledger_operations();
    validate(a); validate(b);
    return SimpleFluid::detail::bounded_fp::add(a, b);
}
Value negate(Value a)
{
    return SimpleFluid::detail::bounded_fp::negate(a);
}
Value subtract(Value a, Value b)
{
    return add(a, negate(b));
}
Value multiply(Value a, Value b)
{
    Meshes::bounded_record_ledger_operations();
    validate(a); validate(b);
    return SimpleFluid::detail::bounded_fp::multiply(a, b);
}
Value divide(Value a, Value b)
{
    Meshes::bounded_record_ledger_operations();
    validate(a); validate(b);
    return SimpleFluid::detail::bounded_fp::divide(a, b);
}
/** Admit current-step construction/arithmetic error against this recipient's
 * own phase amount. This precedes binary64 publication: historical allowances
 * and final narrowing error are separate evidence, never a donor/bulk-scaled
 * substitute or a cumulative temporal error ceiling.
 */
void admit_receipt(Value amount, double relative_tolerance, const char* phase)
{
    validate(amount);
    if (!std::isfinite(relative_tolerance) || !(relative_tolerance > 0) || amount.value < 0)
        throw std::invalid_argument("Finite VoF recipient phase admission has invalid inputs.");
    if (amount.absolute_error == 0)
        return; // Includes a proven zero phase; no positive amount cutoff exists.
    const auto lower = std::nextafter(amount.value - amount.absolute_error,
        -std::numeric_limits<long double>::infinity());
    const auto target = lower > 0 ? std::nextafter(lower * static_cast<long double>(relative_tolerance),
        -std::numeric_limits<long double>::infinity()) : 0;
    if (!(lower > 0) || amount.absolute_error > target)
        throw std::invalid_argument(std::string("Finite VoF recipient ") + phase +
            " phase current-step geometry/arithmetic error exceeds its independent phase budget.");
}
std::pair<double, double> publish(Value value)
{
    if (!std::isfinite(value.value) || value.value < 0 || value.absolute_error < 0 ||
        (value.value == 0 && value.absolute_error != 0) ||
        (value.value != 0 && !(value.value - value.absolute_error > 0)))
        throw std::invalid_argument("Finite VoF phase positivity is unresolved.");
    const auto result = static_cast<double>(value.value);
    if (!std::isfinite(result) || (value.value != 0 && result == 0))
        throw std::invalid_argument("Finite VoF phase publication is outside binary64 range.");
    return {result, upper_double(bound_add(value.absolute_error, std::abs(value.value - result)))};
}
} // namespace certified
namespace ordinary
{
Value result(long double value)
{
    Meshes::bounded_record_ledger_operations();
    return Arithmetic<ArithmeticMode::Ordinary>::make_value(value);
}
Value add(Value a, Value b) { return result(a.value + b.value); }
Value negate(Value a) { return result(-a.value); }
Value multiply(Value a, Value b) { return result(a.value * b.value); }
Value divide(Value a, Value b)
{
    if (b.value == 0)
        throw std::invalid_argument("Ordinary VoF division has a zero denominator.");
    return result(a.value / b.value);
}
void admit_receipt(Value amount, double relative_tolerance, const char*)
{
    if (!std::isfinite(relative_tolerance) || !(relative_tolerance > 0) || amount.value < 0)
        throw std::invalid_argument("Ordinary VoF recipient phase admission has invalid inputs.");
}
std::pair<double, double> publish(Value value)
{
    if (value.value < 0)
        throw std::invalid_argument("Ordinary VoF phase amount is negative.");
    const auto narrowed = static_cast<double>(value.value);
    if (!std::isfinite(narrowed) || (value.value != 0 && narrowed == 0))
        throw std::invalid_argument("Ordinary VoF phase publication is outside binary64 range.");
    // The actual final narrowing discrepancy remains part of phase history.
    return {narrowed, upper_double(std::abs(value.value - static_cast<long double>(narrowed)))};
}
} // namespace ordinary
} // namespace SimpleFluid::VoF::finite_detail
