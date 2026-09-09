/**
 * @file CompensatedSum.hh
 * @brief Roundoff-compensated local accumulation before scalar MPI reduction.
 */
#pragma once

#include <cmath>
#include <concepts>

namespace SimpleFluid::detail
{

/** Neumaier summation, including platforms where long double equals double. */
template<std::floating_point Scalar = long double> class CompensatedSum
{
public:
    void operator+=(Scalar value)
    {
        const auto next = d_sum + value;
        if (std::abs(d_sum) >= std::abs(value))
            d_correction += (d_sum - next) + value;
        else
            d_correction += (value - next) + d_sum;
        d_sum = next;
    }

    [[nodiscard]] Scalar value() const { return d_sum + d_correction; }

private:
    Scalar d_sum{};
    Scalar d_correction{};
};

} // namespace SimpleFluid::detail
