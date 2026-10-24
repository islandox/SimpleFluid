/** @file
 * @brief Private owner-local raw stocks, never packed into halo or MPI reports.
 */
#pragma once
#include "solvers/details/LowMachFaceTransportAdapter.hh"
#include <cmath>
#include <limits>

namespace SimpleFluid::Reacting::low_mach_faces
{
using RawAmounts = std::vector<std::array<long double, 2>>;

SIMPLEFLUID_SOLVERS_LOCAL inline void add_raw_amount(std::array<long double, 2>& stock, long double delta)
{
    if (!std::isfinite(delta))
        throw std::invalid_argument("Nonfinite original material event.");
    const auto next = stock[0] + delta;
    stock[1] += std::abs(stock[0]) >= std::abs(delta) ? (stock[0] - next) + delta : (delta - next) + stock[0];
    stock[0] = next;
}
SIMPLEFLUID_SOLVERS_LOCAL inline std::size_t raw_amount_slot(const SpeciesCatalog& catalog,
    const CellSpeciesState& topology, SpeciesId species, const DiffusionCompartment& compartment)
{
    if (static_cast<std::size_t>(compartment.phase) >= static_cast<std::size_t>(SpeciesPhase::Count) ||
        (!compartment.population.empty() && compartment.phase != SpeciesPhase::Dispersed))
        throw std::invalid_argument("Raw material event has invalid compartment metadata.");
    const auto ns = catalog.species().size(), index = catalog.index(species);
    if (compartment.population.empty())
        return static_cast<std::size_t>(compartment.phase) * ns + index;
    for (std::size_t p = 0; p < topology.dispersedPopulations.size(); ++p)
        if (topology.dispersedPopulations[p].id == compartment.population)
            return topology.amounts.size() + p * ns + index;
    throw std::invalid_argument("Raw material event selects an absent population.");
}
SIMPLEFLUID_SOLVERS_LOCAL inline RawAmounts raw_amounts_from_state(const CellSpeciesState& state)
{
    RawAmounts result;
    for (double value : state.amounts)
        result.push_back({value, 0});
    for (const auto& population : state.dispersedPopulations)
        for (double value : population.amounts)
            result.push_back({value, 0});
    return result; // Explicit initial state only; do not replace a prior source's uncollapsed receipt.
}
SIMPLEFLUID_SOLVERS_LOCAL inline CellSpeciesState project_raw_inventory(
    const SpeciesCatalog& catalog, const CellSpeciesState& topology, std::span<const std::array<long double, 2>> raw)
{
    topology.validate(catalog);
    const auto count = topology.amounts.size() + topology.dispersedPopulations.size() * catalog.species().size();
    if (raw.size() != count)
        throw std::invalid_argument("Raw predecessor stock has the wrong schema dimensions.");
    auto result = topology;
    std::size_t index = 0;
    const auto project = [&]
    {
        const auto& parts = raw[index++];
        const auto value = parts[0] + parts[1];
        const double narrowed = static_cast<double>(value);
        if (!std::isfinite(parts[0]) || !std::isfinite(parts[1]) || !std::isfinite(value) || value < 0 ||
            !std::isfinite(narrowed) || (value != 0 && narrowed == 0))
            throw std::invalid_argument("Raw predecessor stock is negative, nonfinite or unrepresentable.");
        return narrowed;
    };
    for (auto& value : result.amounts)
        value = project();
    for (auto& population : result.dispersedPopulations)
        for (auto& value : population.amounts)
            value = project();
    result.validate(catalog);
    return result;
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_raw_inventory(
    const SpeciesCatalog& catalog, const CellSpeciesState& represented, const RawAmounts& raw)
{
    const auto projected = project_raw_inventory(catalog, represented, raw);
    bool same = projected.amounts == represented.amounts &&
                projected.dispersedPopulations.size() == represented.dispersedPopulations.size();
    for (std::size_t p = 0; same && p < projected.dispersedPopulations.size(); ++p)
        same = projected.dispersedPopulations[p].id == represented.dispersedPopulations[p].id &&
               projected.dispersedPopulations[p].amounts == represented.dispersedPopulations[p].amounts;
    if (!same)
        throw std::invalid_argument("Raw predecessor stock does not project to its actual represented state.");
}
SIMPLEFLUID_SOLVERS_LOCAL inline CellSpeciesState apply_raw_material(const SpeciesCatalog& catalog,
    const CellSpeciesState& predecessor, RawAmounts& raw, std::span<const MolarTerm> terms)
{
    validate_raw_inventory(catalog, predecessor, raw);
    auto margins = raw, next = raw;
    for (const auto& term : terms)
    {
        const auto slot = raw_amount_slot(catalog, predecessor, term.species, term.compartment);
        if (term.moles < 0)
            add_raw_amount(margins[slot], term.moles); // All outgoing original terms; no incoming credit.
        add_raw_amount(next[slot], term.moles);
    }
    for (const auto& stock : margins)
        if (!std::isfinite(stock[0]) || !std::isfinite(stock[1]) || stock[0] + stock[1] < 0)
            throw std::runtime_error("Original owned material packets exhaust their actual raw predecessor donor.");
    auto result = project_raw_inventory(catalog, predecessor, next);
    raw = std::move(next);
    return result;
}
} // namespace SimpleFluid::Reacting::low_mach_faces
