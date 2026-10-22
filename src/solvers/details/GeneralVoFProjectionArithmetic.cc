/** @file GeneralVoFProjectionArithmetic.cc
 * @brief Operation-derived arithmetic enclosures for native projection fluxes.
 */
#include "solvers/details/GeneralVoFProjectionArithmetic.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace SimpleFluid::detail
{
namespace
{
using M = Meshes::ExactGeometryMeasure;
M product(const M& a, const M& b, std::size_t budget)
{
    if (!a.sign() || !b.sign())
        return {};
    return vof_exact::divide(a, vof_exact::divide(M::from_value(1), b, budget), budget);
}
M absolute(const M& a, std::size_t budget)
{
    return a.sign() < 0 ? vof_exact::scale(a, -1, budget) : a;
}
} // namespace
M general_vof_projection_divergence_limit(const M& initial_maximum, double dt, double absolute_tolerance,
    double relative_tolerance, double divergence_tolerance, std::size_t cap)
{
    if (initial_maximum.sign() < 0 || !std::isfinite(dt) || !(dt > 0) || !std::isfinite(absolute_tolerance) ||
        absolute_tolerance < 0 || !std::isfinite(relative_tolerance) || relative_tolerance < 0 ||
        !std::isfinite(divergence_tolerance) || !(divergence_tolerance > 0))
        throw std::runtime_error("Invalid fixed primary pressure reference.");
    const auto threshold = vof_exact::leased_add(vof_exact::leased_scale(M::from_value(dt), absolute_tolerance, cap),
        vof_exact::leased_scale(initial_maximum, relative_tolerance, cap), cap);
    const auto maximum = M::from_value(divergence_tolerance);
    return threshold.compare(maximum) < 0 ? threshold : maximum;
}
GeneralVoFRoundedValue GeneralVoFProjectionArithmetic::leaf(double value) const
{
    if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite projection arithmetic leaf.");
    auto exact = M::from_value(value);
    if (exact.storage_bytes() > budget_)
        throw std::runtime_error("Projection exact leaf exceeds scalar reserve.");
    return {value, exact, exact, 0};
}
GeneralVoFRoundedValue GeneralVoFProjectionArithmetic::apply(
    const GeneralVoFRoundedValue& a, const GeneralVoFRoundedValue& b, unsigned op) const
{
    namespace E = vof_exact;
    for (const auto* input : {&a, &b})
    {
        const auto represented = M::from_value(input->value);
        if (input->lower.compare(represented) > 0 || input->upper.compare(represented) < 0)
            throw std::runtime_error("Projection arithmetic interval excludes its represented value.");
    }
    if (a.operations > std::numeric_limits<std::uint64_t>::max() - 1 ||
        b.operations > std::numeric_limits<std::uint64_t>::max() - 1 - a.operations)
        throw std::runtime_error("Projection arithmetic operation count overflow.");
    if (op == 3 && b.lower.sign() <= 0 && b.upper.sign() >= 0)
        throw std::runtime_error("Projection arithmetic denominator interval contains zero.");
    const auto calculate = [&](const M& x, const M& y) -> M
    {
        if (op == 0)
            return E::add(x, y, budget_);
        if (op == 1)
            return E::add(x, E::scale(y, -1, budget_), budget_);
        if (op == 2)
            return product(x, y, budget_);
        return E::divide(x, y, budget_);
    };
    const double value = op == 0   ? a.value + b.value
                         : op == 1 ? a.value - b.value
                         : op == 2 ? a.value * b.value
                                   : a.value / b.value;
    if (!std::isfinite(value))
        throw std::runtime_error("Projection arithmetic operation overflows binary64.");
    const auto center = calculate(M::from_value(a.value), M::from_value(b.value));
    const auto represented = M::from_value(value);
    // This exact per-operation rounding error follows only the represented
    // operands of this operation. It does not inspect physical divergence or
    // Krylov residuals. Input arithmetic intervals are propagated separately.
    const auto rounding = absolute(E::add(center, E::scale(represented, -1, budget_), budget_), budget_);
    const std::array<M, 4> corners = {calculate(a.lower, b.lower), calculate(a.lower, b.upper),
        calculate(a.upper, b.lower), calculate(a.upper, b.upper)};
    auto lower = corners[0], upper = corners[0];
    for (const auto& c : corners)
    {
        if (c.compare(lower) < 0)
            lower = c;
        if (c.compare(upper) > 0)
            upper = c;
    }
    lower = E::add(lower, E::scale(rounding, -1, budget_), budget_);
    upper = E::add(upper, rounding, budget_);
    return {value, lower, upper, a.operations + b.operations + 1};
}
GeneralVoFRoundedValue GeneralVoFProjectionArithmetic::add(
    const GeneralVoFRoundedValue& a, const GeneralVoFRoundedValue& b) const
{
    return apply(a, b, 0);
}
GeneralVoFRoundedValue GeneralVoFProjectionArithmetic::subtract(
    const GeneralVoFRoundedValue& a, const GeneralVoFRoundedValue& b) const
{
    return apply(a, b, 1);
}
GeneralVoFRoundedValue GeneralVoFProjectionArithmetic::multiply(
    const GeneralVoFRoundedValue& a, const GeneralVoFRoundedValue& b) const
{
    return apply(a, b, 2);
}
GeneralVoFRoundedValue GeneralVoFProjectionArithmetic::divide(
    const GeneralVoFRoundedValue& a, const GeneralVoFRoundedValue& b) const
{
    return apply(a, b, 3);
}
M GeneralVoFProjectionArithmetic::radius(const GeneralVoFRoundedValue& value) const
{
    const auto center = M::from_value(value.value);
    const auto lower = absolute(vof_exact::add(center, vof_exact::scale(value.lower, -1, budget_), budget_), budget_);
    const auto upper = absolute(vof_exact::add(value.upper, vof_exact::scale(center, -1, budget_), budget_), budget_);
    return lower.compare(upper) > 0 ? lower : upper;
}
} // namespace SimpleFluid::detail
