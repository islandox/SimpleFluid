/** @file AMRRemapBudget.hh
 * @brief Compensated scalar budgets and fail-closed extensive products.
 */
#pragma once
#include "utils/CompensatedSum.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace SimpleFluid::VoF::detail
{
/** Scale mantissas before multiplying: neither an overflowing amount*volume
 * nor an underflowing volume ratio is formed as an intermediate. A nonzero
 * product outside the working type's range rejects before accumulation.
 */
template<class Real> Real weighted_amount(Real amount, Real numerator, Real denominator)
{
    if (amount == 0 || numerator == 0)
        return 0;
    if (!(numerator > 0) || !(denominator > 0))
        throw std::invalid_argument("Invalid AMR carrier weight.");
    int a = 0, n = 0, d = 0;
    const auto am = std::frexp(amount, &a), nm = std::frexp(numerator, &n), dm = std::frexp(denominator, &d);
    const auto value = std::ldexp(am * (nm / dm), a + n - d);
    if (!std::isfinite(value) || value == 0)
        throw std::invalid_argument("AMR weighted nonzero amount is unrepresentable in its accumulator.");
    return value;
}

struct ScalarBudgetSum
{
    std::array<double, 4> values{};
    long double arithmetic = 0;
};

/** Sum four scalar diagnostics per rank, never field values. Neumaier summation
 * at both levels has a first-order error independent of rank count; the explicit
 * second-order term covers the local/global operation counts. This uncertainty
 * goes on the residual side of the unchanged physical acceptance inequality.
 */
inline ScalarBudgetSum sum_rank_budgets(std::span<const std::array<double, 4>> ranks, long double cells)
{
    std::array<SimpleFluid::detail::CompensatedSum<double>, 4> sums;
    for (const auto& rank : ranks)
        for (std::size_t a = 0; a < 4; ++a)
        {
            if (!std::isfinite(rank[a]))
                throw std::invalid_argument("Nonfinite scaled AMR budget.");
            sums[a] += rank[a];
        }
    ScalarBudgetSum result;
    for (std::size_t a = 0; a < 4; ++a)
    {
        result.values[a] = sums[a].value();
        if (!std::isfinite(result.values[a]))
            throw std::invalid_argument("Unrepresentable scaled AMR diagnostic.");
    }
    const long double norm = std::max(result.values[2], result.values[3]);
    if (norm < 0)
        throw std::invalid_argument("Negative scaled AMR L1 budget.");
    if (norm > 0)
    {
        const long double epsilon = std::numeric_limits<double>::epsilon();
        result.arithmetic = (16 * epsilon + 64 * (cells + ranks.size()) * epsilon * epsilon) * norm +
                            cells * static_cast<long double>(std::numeric_limits<double>::denorm_min());
    }
    return result;
}
} // namespace SimpleFluid::VoF::detail
