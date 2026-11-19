/** @file ALESpeciesConvergence.hh
 * @brief Private arithmetic uncertainty for signed ALE source-net comparisons.
 */
#pragma once

#include "equations/SpeciesInventoryHistory.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <span>

namespace SimpleFluid::detail::ale_species
{
struct ConvergenceEntry
{
    double value = 0;
    long double cancellation_allowance = 0;

    ConvergenceEntry(double raw_value = 0, long double allowance = 0)
        : value(raw_value), cancellation_allowance(allowance)
    {
    }
};

/** Only the six source operands contribute. Physical escape is a separate event.
 * A binary64 rounded nonnegative operand has relative representation uncertainty
 * at most u/(1-u), u=epsilon/2. One epsilon per reported operand conservatively
 * encloses that uncertainty; compensated long-double sums retain its scale.
 * Both signs must be present: an uncancelled tiny source gets no allowance.
 * This is an arithmetic comparison allowance, never an inventory/source edit.
 */
inline long double source_cancellation_allowance(const Reacting::SpeciesCompartmentEvents& event)
{
    CompensatedSum<long double> allowance;
    bool positive = false, negative = false;
    const auto add = [&](double value, bool& present)
    {
        if (!std::isfinite(value) || value < 0)
            throw std::runtime_error("ALE source convergence operands must be finite and nonnegative.");
        present = present || value > 0;
        // Scale before summing so finite binary64 gross operands cannot overflow
        // even on platforms where long double has binary64 range/precision.
        allowance += static_cast<long double>(std::numeric_limits<double>::epsilon()) * value;
    };
    for (const auto value : {event.chemical_production, event.external_supply, event.interphase_inflow})
        add(value, positive);
    for (const auto value : {event.chemical_consumption, event.external_withdrawal, event.interphase_outflow})
        add(value, negative);
    return positive && negative ? allowance.value() : 0;
}

/** A solution subset has exactly the same source net as its sole allowed
 * Liquid/Dissolved compartment. Only this structural identity permits inheriting
 * the matching species/phase operand bound. With both phases allowed, internal
 * subset transfers must not inflate uncertainty for unrelated tiny source nets.
 * Gas subsets and missing exact provenance retain zero allowance.
 */
inline long double solution_subset_cancellation_allowance(const Reacting::SpeciesCatalog& catalog,
    Reacting::SpeciesId species, std::span<const Reacting::SpeciesCompartmentEvents> events)
{
    using Phase = Reacting::SpeciesPhase;
    const bool liquid = catalog.allows(species, Phase::Liquid);
    const bool dissolved = catalog.allows(species, Phase::Dissolved);
    if (liquid == dissolved)
        return 0;
    const auto phase = liquid ? Phase::Liquid : Phase::Dissolved;
    const Reacting::SpeciesCompartmentEvents* match = nullptr;
    for (const auto& event : events)
        if (event.species == species && event.phase == phase && event.population.empty())
        {
            if (match)
                throw std::logic_error("ALE solution-subset source provenance has duplicate compartments.");
            match = &event;
        }
    return match ? source_cancellation_allowance(*match) : 0;
}

inline double relative_change(const ConvergenceEntry& before, const ConvergenceEntry& after)
{
    if (!std::isfinite(before.value) || !std::isfinite(after.value) ||
        !std::isfinite(before.cancellation_allowance) || !std::isfinite(after.cancellation_allowance) ||
        before.cancellation_allowance < 0 || after.cancellation_allowance < 0)
        throw std::runtime_error("ALE species/event convergence state or allowance is invalid.");
    if (before.cancellation_allowance == 0 && after.cancellation_allowance == 0)
    {
        // Preserve the existing binary64 gate exactly for every stock, count,
        // gross event and uncancelled net; only bounded cancellation uses wide arithmetic.
        const auto scale = std::max(std::abs(before.value), std::abs(after.value));
        return scale == 0 ? 0 : std::abs(after.value - before.value) / scale;
    }
    const auto old_value = static_cast<long double>(before.value);
    const auto new_value = static_cast<long double>(after.value);
    const auto scale = std::max(std::abs(old_value), std::abs(new_value));
    if (scale == 0)
        return 0;
    const auto excess = std::max(0.L, std::abs(new_value - old_value) - before.cancellation_allowance -
                                       after.cancellation_allowance);
    return static_cast<double>(excess / scale);
}
} // namespace SimpleFluid::detail::ale_species
