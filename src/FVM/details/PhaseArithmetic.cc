/** @file PhaseArithmetic.cc
 * @brief Verified finite publication of transient exact phase and error measures.
 */
#include "FVM/details/PhaseArithmetic.hh"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace SimpleFluid::VoF::detail
{
using M = Meshes::ExactGeometryMeasure;
double phase_signed_value(const M& value)
{
    const auto result = static_cast<double>(value.enclosure().value);
    if (!std::isfinite(result))
        throw std::invalid_argument("Finite phase value exceeds binary64 range.");
    return result;
}
double phase_upper_bound(const M& value)
{
    if (value.sign() < 0)
        throw std::invalid_argument("Finite phase error bound is negative.");
    auto result = phase_signed_value(value);
    if (M::from_value(result).compare(value) < 0)
        result = std::nextafter(result, std::numeric_limits<double>::infinity());
    if (!std::isfinite(result) || M::from_value(result).compare(value) < 0)
        throw std::invalid_argument("Finite phase bound has no verified upward binary64 enclosure.");
    return result;
}
RoundedPhaseValue round_phase_residual(const M& value)
{
    const auto result = phase_signed_value(value);
    const auto difference = M::from_value(result) - value;
    const auto absolute = difference.sign() < 0 ? difference.scaled(-1) : difference;
    return {result, phase_upper_bound(absolute)};
}
RoundedPhaseValue round_phase_value(const M& value)
{
    if (value.sign() < 0)
        throw std::invalid_argument("Finite phase publication has negative inventory.");
    return round_phase_residual(value);
}

} // namespace SimpleFluid::VoF::detail
