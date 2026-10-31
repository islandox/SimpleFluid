/** @file PhasePublication.tcc
 * @brief Compiled policy-specific finite phase-conservation checks.
 */
#include "FVM/details/GeneralVoF/PhasePublication.hh"
#include <cmath>
#include <limits>
namespace SimpleFluid::VoF::general_vof_detail
{
template<PhaseConservationPolicy Policy>
PhaseConservationResult check_phase_conservation(
    const PhaseConservationSummary& global, int ranks, double relative_tolerance) noexcept
{
    const auto operations = 8.L * ranks + 16;
    const auto u = operations * std::numeric_limits<double>::epsilon();
    if (!(u < .25L))
        return PhaseConservationResult::ReductionBoundExhausted;
    for (unsigned phase = 0; phase < 2; ++phase)
    {
        const auto physical = [&]
        {
            if constexpr (Policy == PhaseConservationPolicy::Ordinary)
                return relative_tolerance * global[6 + phase];
            else
                return 0.L;
        }();
        // Ordinary keeps the physical product and its publication addition in binary64.
        const auto bound = physical + global[2 + phase] + u / (1 - u) * (global[4 + phase] + global[2 + phase]);
        if (!std::isfinite(global[phase]) || !std::isfinite(bound) || std::abs(global[phase]) > bound)
            return PhaseConservationResult::Nonconserved;
    }
    return PhaseConservationResult::Accepted;
}
} // namespace SimpleFluid::VoF::general_vof_detail
