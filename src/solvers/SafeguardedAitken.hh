/**
 * @file SafeguardedAitken.hh
 * @brief Weighted vector Aitken relaxation with caller-owned collective reductions.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace SimpleFluid
{
struct SafeguardedAitkenOptions
{
    double initial_relaxation = 1.0;
    double fallback_relaxation = 0.5;
    double minimum_relaxation = 0.1;
    double maximum_relaxation = 1.0;
    double maximum_residual_growth = 2.0;
};

/**
 * A history belongs to one fixed-point problem. Residual scaling, entry order,
 * and weights must stay fixed until reset(). Zero weights exclude replicated
 * entries; empty local vectors are supported on ranks without owned entries.
 *
 * prepare() performs local validation and allocation. Distributed callers must
 * collectively agree any exception before summing its products. Every rank then
 * calls finish() with the same global SUM, including ranks with empty vectors.
 * No method performs communication or changes a physical acceptance criterion.
 */
class SafeguardedAitken
{
public:
    // Previous residual dot difference, difference squared, previous squared,
    // current squared. All products include the caller's nonnegative weights.
    using Products = std::array<double, 4>;

    explicit SafeguardedAitken(SafeguardedAitkenOptions options = {})
        : d_options(options), d_relaxation(options.initial_relaxation)
    {
        const auto factor = [](double value)
        { return std::isfinite(value) && value > 0.0 && value <= 1.0; };
        if (!factor(options.minimum_relaxation) || !factor(options.maximum_relaxation) ||
            !factor(options.initial_relaxation) || !factor(options.fallback_relaxation) ||
            options.minimum_relaxation > options.maximum_relaxation ||
            options.initial_relaxation < options.minimum_relaxation ||
            options.initial_relaxation > options.maximum_relaxation ||
            options.fallback_relaxation < options.minimum_relaxation ||
            options.fallback_relaxation > options.maximum_relaxation ||
            !std::isfinite(options.maximum_residual_growth) || options.maximum_residual_growth < 1.0)
            throw std::invalid_argument("Invalid safeguarded Aitken relaxation options.");
    }

    Products prepare(std::span<const double> residual, std::span<const double> weights = {})
    {
        if (d_prepared)
            throw std::logic_error("Finish the prepared Aitken observation before preparing another.");
        if ((!weights.empty() && weights.size() != residual.size()) ||
            (d_has_previous && d_previous.size() != residual.size()))
            throw std::invalid_argument("Aitken residual and weight layouts must remain compatible.");
        std::array<long double, 4> products{};
        for (std::size_t i = 0; i < residual.size(); ++i)
        {
            const double weight = weights.empty() ? 1.0 : weights[i];
            if (!std::isfinite(residual[i]) || !std::isfinite(weight) || weight < 0.0)
                throw std::invalid_argument("Aitken requires finite residuals and nonnegative finite weights.");
            const long double current = residual[i];
            const long double previous = d_has_previous ? d_previous[i] : 0.0L;
            const long double difference = current - previous;
            products[0] += weight * previous * difference;
            products[1] += weight * difference * difference;
            products[2] += weight * previous * previous;
            products[3] += weight * current * current;
        }
        Products local{};
        for (std::size_t i = 0; i < local.size(); ++i)
        {
            local[i] = static_cast<double>(products[i]);
            if (!std::isfinite(local[i]))
                throw std::overflow_error("Aitken local residual products overflowed.");
        }
        d_pending.assign(residual.begin(), residual.end());
        d_prepared = true;
        return local;
    }

    /** Finish one observation without allocation; invalid secants use fallback. */
    double finish(const Products& global)
    {
        if (!d_prepared)
            throw std::logic_error("Aitken finish requires a prepared observation.");
        const bool finite = std::all_of(global.begin(), global.end(),
            [](double value) { return std::isfinite(value); }) &&
            global[1] >= 0.0 && global[2] >= 0.0 && global[3] >= 0.0;
        double next = d_options.fallback_relaxation;
        if (finite && !d_has_previous)
            next = d_options.initial_relaxation;
        else if (finite)
        {
            // Relative degeneracy is essential for small, physically meaningful
            // volume-rate residuals. An absolute epsilon would suppress Aitken.
            const double scale = std::max(global[2], global[3]);
            const double floor = 64.0 * std::numeric_limits<double>::epsilon() * scale;
            const bool bounded_growth = std::sqrt(global[3]) <=
                d_options.maximum_residual_growth * std::sqrt(global[2]);
            if (global[1] > floor && bounded_growth)
            {
                const double proposed = -d_relaxation * (global[0] / global[1]);
                if (std::isfinite(proposed) && proposed > 0.0)
                    next = std::clamp(proposed, d_options.minimum_relaxation, d_options.maximum_relaxation);
            }
        }
        // Restart a rejected secant from this observation and the fallback
        // factor. Never retain an older residual across a failed estimate.
        d_previous.swap(d_pending);
        d_has_previous = true;
        d_prepared = false;
        d_relaxation = next;
        return next;
    }

    void reset() noexcept
    {
        d_previous.clear();
        d_pending.clear();
        d_has_previous = false;
        d_prepared = false;
        d_relaxation = d_options.initial_relaxation;
    }

private:
    SafeguardedAitkenOptions d_options;
    std::vector<double> d_previous, d_pending;
    double d_relaxation;
    bool d_has_previous = false, d_prepared = false;
};
} // namespace SimpleFluid
