/** @file VoFPhaseBudget.hh
 * @brief Roundoff-only conservative directional PLIC donor budgets.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace SimpleFluid::VoF::transport_detail
{
struct PhaseBudget
{
    double gas = 1.0, carrier = 1.0;
    [[nodiscard]] bool changes_flux() const noexcept { return gas < 1.0 || carrier < 1.0; }
};

inline PhaseBudget phase_budget(
    long double gas_out, long double carrier_out, double fraction, double volume, double phase_roundoff)
{
    const auto factor = [&](long double outgoing, long double available)
    {
        available = std::max(0.0L, available);
        if (outgoing <= available)
            return 1.0;
        if (outgoing - available > static_cast<long double>(phase_roundoff) * volume)
            throw std::runtime_error("Directional PLIC outflow materially exceeds its donor phase inventory.");
        // Leave a representable inward margin for each rounded face product.
        double value = static_cast<double>(available / outgoing);
        value = std::nextafter(std::nextafter(value, 0.0), 0.0);
        return std::max(0.0, value);
    };
    const long double gas = static_cast<long double>(fraction) * volume;
    const long double carrier = (1.0L - fraction) * volume;
    PhaseBudget result{factor(gas_out, gas), factor(carrier_out, carrier)};
    if (result.gas < 1.0 && result.carrier < 1.0)
        throw std::runtime_error("Directional PLIC outflow exhausts both phase budgets; reduce the timestep.");
    return result;
}

inline double budgeted_gas(double total, double gas, PhaseBudget factor)
{
    const double q = std::abs(total), g = std::abs(gas);
    double result = g;
    if (factor.gas < 1.0)
        result = std::nextafter(g * factor.gas, 0.0);
    else if (factor.carrier < 1.0)
    {
        const double carrier = std::nextafter((q - g) * factor.carrier, 0.0);
        result = q - carrier;
        // Subtraction can round the complementary carrier upward. Move only
        // the shared face partition until its actual rounded carrier is safe.
        while (q - result > carrier)
            result = std::nextafter(result, q);
    }
    return std::copysign(result, total);
}
} // namespace SimpleFluid::VoF::transport_detail
