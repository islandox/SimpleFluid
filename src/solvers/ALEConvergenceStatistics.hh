/**
 * @file ALEConvergenceStatistics.hh
 * @brief Cumulative observations of the existing planar ALE acceptance gates.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace SimpleFluid
{
/** One completed corrector check, including rejected and replayed attempts. */
struct ALEConvergenceObservation
{
    static constexpr std::size_t gas_field_count = 5;
    static constexpr std::array<const char*, gas_field_count> gas_field_names{
        "dissolved_inventory", "micro_number_density", "micro_moles", "large_number_density", "large_moles"};
    std::uint64_t check = 0;
    int step = 0, corrector = 0;
    double time = 0, time_step = 0, gas_tolerance = 0;
    std::size_t blocking_mask = 0;
    std::array<double, gas_field_count> gas_relative_change{};
};

/** Diagnostic counters, independent of accepted/checkpoint state and MPI clocks.
 * A mask records every failed gate, including simultaneous blockers. Ratios are
 * for attribution only; the caller supplies the original boolean gate results.
 */
struct ALEConvergenceStatistics
{
    static constexpr std::size_t criterion_count = 7;
    static constexpr std::array<const char*, criterion_count> criterion_names{
        "thermal", "level", "target_change", "continuity", "pool_volume", "material", "gas"};
    std::uint64_t checks = 0, converged_checks = 0;
    std::array<std::uint64_t, 1u << criterion_count> blocking_combinations{};
    std::array<double, criterion_count> maximum_normalized_residual{};

    /** Record globally reduced residuals and the unchanged gate comparisons. */
    void record(const std::array<double, criterion_count>& normalized,
                const std::array<bool, criterion_count>& failed) noexcept
    {
        std::size_t mask = 0;
        for (std::size_t i = 0; i < criterion_count; ++i)
        {
            if (failed[i]) mask |= std::size_t{1} << i;
            const double value = std::isfinite(normalized[i]) ? std::abs(normalized[i])
                : std::numeric_limits<double>::infinity();
            maximum_normalized_residual[i] = std::max(maximum_normalized_residual[i], value);
        }
        ++checks;
        ++blocking_combinations[mask];
        if (mask == 0) ++converged_checks;
    }

    /** Number of checks where a criterion blocked acceptance, alone or jointly. */
    std::array<std::uint64_t, criterion_count> blocking_counts() const noexcept
    {
        std::array<std::uint64_t, criterion_count> result{};
        for (std::size_t mask = 1; mask < blocking_combinations.size(); ++mask)
            for (std::size_t i = 0; i < criterion_count; ++i)
                if (mask & (std::size_t{1} << i)) result[i] += blocking_combinations[mask];
        return result;
    }
};
} // namespace SimpleFluid
