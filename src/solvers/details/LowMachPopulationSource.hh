/** @file LowMachPopulationSource.hh
 * MPI-free explicit number events, raw endpoint proof, no inferred material/energy.
 */
#pragma once
#include "solvers/ReactingMixtureCoupler.hh"
#include "solvers/details/LowMachFaceSourceComposition.hh"
#include "utils/CompensatedSum.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace SimpleFluid::Reacting::population_source_detail
{
using Flow = LowMachMixtureSolver;
SIMPLEFLUID_SOLVERS_LOCAL inline double narrow(long double value)
{
    const double result = static_cast<double>(value);
    if (!std::isfinite(value) || !std::isfinite(result) || (value != 0 && result == 0))
        throw std::invalid_argument("Population event count is nonfinite or unrepresentable.");
    return result;
}
/** Independent before-birth donor reservation. Repeated event names accumulate;
 * gross deaths cannot borrow incoming births. Empty gas requires explicit death,
 * newly occupied gas requires explicit birth. No number is inferred from material.
 */
SIMPLEFLUID_SOLVERS_LOCAL inline MixturePopulationEventReceipt apply(std::span<const Flow::PhaseRoleDescriptor> roles,
    std::span<const std::optional<double>> before, std::span<const int> occupied,
    std::vector<BubbleGasPhysics::PopulationNumberEvent> events)
{
    if (roles.size() != before.size() || occupied.size() != roles.size())
        throw std::invalid_argument("Population event physical-phase layout differs.");
    using Sum = SimpleFluid::detail::CompensatedSum<long double>;
    std::vector<Sum> births(roles.size()), deaths(roles.size()), deltas(roles.size()), endpoints(roles.size()),
        margins(roles.size());
    for (std::size_t p = 0; p < roles.size(); ++p)
    {
        if (roles[p].role != Flow::PhaseRole::DispersedGas)
        {
            if (before[p])
                throw std::invalid_argument("Bulk role unexpectedly owns a population number.");
            continue;
        }
        if (!before[p] || !std::isfinite(*before[p]) || *before[p] < 0)
            throw std::invalid_argument("Population donor number is missing, negative or nonfinite.");
        endpoints[p] += *before[p];
        margins[p] += *before[p];
    }
    for (const auto& event : events)
    {
        const auto found = std::find_if(roles.begin(), roles.end(), [&](const auto& role)
            { return role.role == Flow::PhaseRole::DispersedGas && role.identity == event.population; });
        if (found == roles.end() || !std::isfinite(event.births) || !std::isfinite(event.deaths) || event.births < 0 ||
            event.deaths < 0)
            throw std::invalid_argument("Invalid explicit named population event.");
        const auto p = static_cast<std::size_t>(found - roles.begin());
        births[p] += event.births;
        deaths[p] += event.deaths;
        // Never collapse positive gross reports before signed cancellation or donor reservation.
        deltas[p] += event.births;
        deltas[p] += -static_cast<long double>(event.deaths);
        endpoints[p] += event.births;
        endpoints[p] += -static_cast<long double>(event.deaths);
        margins[p] += -static_cast<long double>(event.deaths); // Incoming births NEVER credit this margin.
    }
    MixturePopulationEventReceipt receipt;
    receipt.events = std::move(events);
    receipt.transported_numbers.assign(before.begin(), before.end());
    receipt.final_numbers.resize(roles.size());
    receipt.births.resize(roles.size());
    receipt.deaths.resize(roles.size());
    receipt.changes.resize(roles.size());
    for (std::size_t p = 0; p < roles.size(); ++p)
    {
        if (roles[p].role != Flow::PhaseRole::DispersedGas)
            continue;
        if (!std::isfinite(margins[p].value()) || margins[p].value() < 0)
            throw std::invalid_argument("Population deaths overdraw actual transported numbers.");
        const double final = narrow(endpoints[p].value()), change = narrow(deltas[p].value());
        if (final < 0 || (final > 0) != (occupied[p] != 0))
            throw std::invalid_argument("Postmaterial gas and explicitly supplied final population number disagree.");
        // This bound uses the actual endpoint, never the large incoming/outgoing count scale.
        // In particular before=1e20, death=1e20, birth=1 cannot round to a packet that returns zero.
        low_mach_faces::validate_source_endpoint(*before[p], change, final);
        receipt.final_numbers[p] = final;
        receipt.births[p] = narrow(births[p].value());
        receipt.deaths[p] = narrow(deaths[p].value());
        receipt.changes[p] = change;
    }
    return receipt;
}
} // namespace SimpleFluid::Reacting::population_source_detail
