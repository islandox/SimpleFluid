/** @file LowMachScaledFaceSum.hh
 * @brief Private compensated accumulation of signed native face rates.
 */
#pragma once

#include "utils/CompensatedSum.hh"
#include "utils/export_helper.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::Reacting::qbmm_face_detail
{

class SIMPLEFLUID_EQUATIONS_LOCAL ScaledFaceSum
{
    static constexpr int digits = std::numeric_limits<double>::digits;
    static constexpr int quantum = std::numeric_limits<double>::min_exponent - digits;

    struct Term
    {
        double fraction = 0;
        int exponent = 0;
    };

    // Keep the leading term inline. The lower terms are ordered from smallest
    // to largest, with gaps represented by the exponents alone.
    Term d_top;
    std::vector<Term> d_lower;

    static Term normalized(double value, int scale = 0)
    {
        int exponent;
        const auto fraction = std::frexp(value, &exponent);
        return {fraction, scale + exponent};
    }

    static Term operand(double value)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("Nonfinite QBMM face rate operand.");
        return normalized(value);
    }

    // Add two terms, returning the low-order residual. Alignment is confined
    // to one significand so a distant term never underflows during a merge.
    static Term merge(Term& large, Term small)
    {
        if (large.fraction == 0)
        {
            large = small;
            return {};
        }
        if (small.fraction == 0)
            return {};
        if (large.exponent < small.exponent ||
            (large.exponent == small.exponent && std::abs(large.fraction) < std::abs(small.fraction)))
            std::swap(large, small);
        const int gap = large.exponent - small.exponent;
        if (gap > digits + 1)
            return small;

        const double low = std::scalbn(small.fraction, -gap);
        const double high = large.fraction + low;
        SimpleFluid::detail::CompensatedSum<long double> residual;
        residual += large.fraction;
        residual += low;
        residual += -high;
        const int scale = large.exponent;
        large = normalized(high, scale);
        return normalized(static_cast<double>(residual.value()), scale);
    }

    void add(Term term)
    {
        if (term.fraction == 0)
            return;
        // Grow the expansion in place, retaining the compensation from each
        // merge even if a later merge cancels its entire leading term.
        std::size_t kept = 0;
        for (const auto part : d_lower)
        {
            const auto residual = merge(term, part);
            if (residual.fraction != 0)
                d_lower[kept++] = residual;
        }
        d_lower.resize(kept);
        const auto residual = merge(term, d_top);
        if (residual.fraction != 0)
            d_lower.push_back(residual);
        if (term.fraction == 0 && !d_lower.empty())
        {
            term = d_lower.back();
            d_lower.pop_back();
        }
        d_top = term;
    }

    static std::array<Term, 2> product(Term a, Term b)
    {
        if (a.fraction == 0 || b.fraction == 0)
            return {};
        const int exponent = a.exponent + b.exponent;
        const double high = a.fraction * b.fraction;
        if (std::abs(a.fraction) == .5 || std::abs(b.fraction) == .5)
            return {normalized(high, exponent), Term{}};
        return {normalized(high, exponent), normalized(std::fma(a.fraction, b.fraction, -high), exponent)};
    }

    void add_scaled(Term a, Term b)
    {
        const auto [high, low] = product(a, b);
        add(high);
        add(low);
    }

    [[nodiscard]] double leading_fraction() const
    {
        if (d_lower.empty())
            return d_top.fraction;

        // Cancellation can expose several significant bits from a lower term
        // even when it shares no significant bits with the leading term.
        SimpleFluid::detail::CompensatedSum<long double> estimate;
        estimate += d_top.fraction;
        for (auto part = d_lower.rbegin(); part != d_lower.rend(); ++part)
        {
            const int shift = part->exponent - d_top.exponent;
            if (shift < -2 * digits)
                break;
            estimate += std::scalbn(part->fraction, shift);
        }
        return static_cast<double>(estimate.value());
    }

    void add_weighted_terms(const ScaledFaceSum& rate, Term factor)
    {
        for (const auto part : rate.d_lower)
            add_scaled(part, factor);
        add_scaled(rate.d_top, factor);
    }

    [[nodiscard]] double rounded_with(Term denominator, const ScaledFaceSum* full_denominator = nullptr) const
    {
        if (denominator.fraction <= 0)
            throw std::invalid_argument("QBMM face rate divisor must be positive.");
        if (d_top.fraction == 0)
            return 0;

        const double sign = d_top.fraction < 0 ? -1. : 1.;
        const double estimate =
            std::scalbn(std::abs(leading_fraction()) / denominator.fraction, d_top.exponent - denominator.exponent);
        if (!full_denominator && d_lower.empty() && denominator.fraction == .5)
            return sign * estimate;

        // Compare in numerator units against the full denominator times the
        // candidate and then its midpoint. An exponent tag keeps half of the
        // smallest subnormal step available for the same comparison.
        const double maximum = std::numeric_limits<double>::max();
        double candidate = std::min(estimate, maximum);
        ScaledFaceSum residual = *this;
        const auto subtract = [&](Term factor)
        {
            factor.fraction *= -sign;
            if (full_denominator)
                residual.add_weighted_terms(*full_denominator, factor);
            else
                residual.add_scaled(denominator, factor);
        };
        subtract(normalized(candidate));
        while (residual.d_top.fraction != 0)
        {
            const auto center = normalized(candidate);
            const bool upward = residual.d_top.fraction * sign > 0;
            const int step = candidate == 0
                                 ? quantum
                                 : std::max(quantum, center.exponent - digits - (!upward && center.fraction == .5));
            const Term half_step{upward ? .5 : -.5, step};
            subtract(half_step);
            const double side = residual.d_top.fraction * sign * (upward ? 1 : -1);
            if (side < 0)
                break;
            if (side == 0)
            {
                // The integral number of steps survives a shift into a
                // binade with spacing two exactly when it is even.
                const double units = std::scalbn(candidate, -step);
                const double shift = std::scalbn(1., digits);
                const double shifted = shift + units;
                if (shifted - shift == units)
                    break;
            }
            if (upward && candidate == maximum)
                return sign * std::numeric_limits<double>::infinity();
            candidate += std::scalbn(upward ? 1. : -1., step);
            if (!full_denominator || side == 0)
                break;

            // The ratio of two rounded leading terms can be more than one
            // step away. Move the residual from the midpoint to the new center.
            subtract(half_step);
        }
        return sign * candidate;
    }

public:
    /** Append exact (significand, exponent) pairs for an owner reduction. */
    void append_scaled_terms(std::vector<double>& values) const
    {
        for (const auto term : d_lower)
        {
            values.push_back(term.fraction);
            values.push_back(term.exponent);
        }
        if (d_top.fraction != 0)
        {
            values.push_back(d_top.fraction);
            values.push_back(d_top.exponent);
        }
    }

    // Admit a finite significand whose binary64 value may require multiplication
    // by another operand before it can be stored as a double.
    void add_scaled_value(double fraction, int exponent)
    {
        auto term = operand(fraction);
        term.exponent += exponent;
        add(term);
    }

    /** Import each significand component of a host raw stock before rounding to binary64. */
    void add_unrounded_value(long double value)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("Nonfinite QBMM stock operand.");
        int exponent;
        auto fraction = std::frexp(value, &exponent);
        while (fraction != 0)
        {
            const auto leading = static_cast<double>(fraction);
            add_scaled_value(leading, exponent);
            fraction -= leading;
        }
    }

    [[nodiscard]] int sign() const noexcept { return (d_top.fraction > 0) - (d_top.fraction < 0); }

    void add_scaled_weighted(const ScaledFaceSum& rate, double fraction, int exponent)
    {
        auto factor = operand(fraction);
        factor.exponent += exponent;
        if (this == &rate)
        {
            const auto copy = rate;
            add_weighted_terms(copy, factor);
        }
        else
            add_weighted_terms(rate, factor);
    }

    void add_product(double lhs, double rhs, double weight = 1.)
    {
        const auto a = operand(lhs), b = operand(rhs), w = operand(weight);
        const auto [high, low] = product(a, b);
        add_scaled(high, w);
        add_scaled(low, w);
    }

    void add_weighted(const ScaledFaceSum& rate, double weight)
    {
        const auto factor = operand(weight);
        if (factor.fraction == 0)
            return;
        if (this == &rate)
        {
            const auto copy = rate;
            add_weighted_terms(copy, factor);
        }
        else
            add_weighted_terms(rate, factor);
    }

    [[nodiscard]] double rounded(double divisor = 1.) const { return rounded_with(operand(divisor)); }

    [[nodiscard]] double rounded(const ScaledFaceSum& positive_divisor) const
    {
        if (positive_divisor.d_lower.empty())
            return rounded_with(positive_divisor.d_top);
        const auto leading = normalized(positive_divisor.leading_fraction(), positive_divisor.d_top.exponent);
        return rounded_with(leading, &positive_divisor);
    }
};

struct MaterialVolumeRates
{
    double fraction;
    double volume_flux;
};

[[nodiscard]] SIMPLEFLUID_EQUATIONS_LOCAL inline MaterialVolumeRates material_volume_rates(
    const ScaledFaceSum& mass_density, double eos_density, double phase_flux)
{
    const double fraction = mass_density.rounded(eos_density);
    // For ordinary fractions the stored fraction remains the coefficient used
    // by pressure and transport. At the subnormal floor, apply the velocity
    // before rounding so an otherwise representable material flux survives.
    if (std::isnormal(fraction) || !std::isfinite(fraction))
        return {fraction, phase_flux * fraction};
    ScaledFaceSum volume_numerator;
    volume_numerator.add_weighted(mass_density, phase_flux);
    return {fraction, volume_numerator.rounded(eos_density)};
}

} // namespace SimpleFluid::Reacting::qbmm_face_detail
