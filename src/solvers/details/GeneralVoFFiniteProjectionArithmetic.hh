/** @file GeneralVoFFiniteProjectionArithmetic.hh
 * @brief Operation-derived fixed-storage arithmetic for represented projection leaves.
 */
#pragma once
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include <array>
namespace SimpleFluid::detail::vof_projection_fp
{
using Value = PLIC::DynamicGeometryValue;
namespace fp = VoF::finite_detail;
/** Retain arithmetic uncertainty when an operator coefficient is cached in
 * binary64. Geometry and prepared gradients are represented input leaves;
 * discretization and linear-solver truncation are independent physical gates.
 */
template<ArithmeticMode Mode> inline Value binary64(Value value)
{
    fp::Arithmetic<Mode>::input(value);
    const auto represented = static_cast<double>(value.value);
    if (!std::isfinite(represented))
        throw std::invalid_argument("Finite projection coefficient exceeds binary64 range.");
    return fp::Arithmetic<Mode>::make_value(
        represented, Mode == ArithmeticMode::BoundedCertification
                         ? fp::bound_add(value.absolute_error, std::abs(value.value - represented))
                         : 0);
}
template<ArithmeticMode Mode> inline Value dot(const std::array<double, 3>& left, const std::array<double, 3>& right)
{
    using A = fp::Arithmetic<Mode>;
    auto result = A::multiply(A::make_value(left[0]), A::make_value(right[0]));
    for (unsigned axis = 1; axis < 3; ++axis)
        result = A::add(result, A::multiply(A::make_value(left[axis]), A::make_value(right[axis])));
    return result;
}
} // namespace SimpleFluid::detail::vof_projection_fp

namespace SimpleFluid::detail::vof_projection_fp
{
inline Value binary64(Value value)
{
    return value.arithmetic_mode == ArithmeticMode::Ordinary ? binary64<ArithmeticMode::Ordinary>(value)
                                                             : binary64<ArithmeticMode::BoundedCertification>(value);
}
inline Value dot(const std::array<double, 3>& left, const std::array<double, 3>& right, ArithmeticMode mode)
{
    (void) fp::make_value(mode, 0);
    return mode == ArithmeticMode::Ordinary ? dot<ArithmeticMode::Ordinary>(left, right)
                                            : dot<ArithmeticMode::BoundedCertification>(left, right);
}
} // namespace SimpleFluid::detail::vof_projection_fp
