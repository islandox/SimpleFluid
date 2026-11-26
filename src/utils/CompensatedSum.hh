/**
 * @file CompensatedSum.hh
 * @brief Roundoff-compensated local accumulation before scalar MPI reduction.
 */
#pragma once

#include <array>
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

    void append(const CompensatedSum& other)
    {
        // Preserve a subtotal's correction when its leading value later cancels.
        const auto sum = other.d_sum;
        const auto correction = other.d_correction;
        *this += sum;
        *this += correction;
    }

    [[nodiscard]] Scalar value() const { return d_sum + d_correction; }

    /** Preserve both native components when handing off a raw subtotal. */
    [[nodiscard]] std::array<Scalar, 2> components() const noexcept { return {d_sum, d_correction}; }

private:
    Scalar d_sum{};
    Scalar d_correction{};
};

} // namespace SimpleFluid::detail
