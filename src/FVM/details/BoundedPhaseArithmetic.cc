/** @file BoundedPhaseArithmetic.cc
 * @brief Compiled finite scalar arithmetic and phase publication.
 */
#include "FVM/details/BoundedPhaseArithmetic.hh"

namespace SimpleFluid::VoF::finite_detail
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
} // namespace SimpleFluid::VoF::finite_detail
