/** @file LowMachSourceStage.hh
 * @brief Private local source validation and reference-safe fixed-point residuals.
 */
#pragma once
#include "solvers/LowMachMixtureSolver.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Reacting::source_stage_detail
{
using Flow = LowMachMixtureSolver;
SIMPLEFLUID_SOLVERS_LOCAL inline void require(bool valid, const char* message)
{
    if (!valid)
        throw std::invalid_argument(message);
}
SIMPLEFLUID_SOLVERS_LOCAL inline long double mass(std::span<const double> moles, const SpeciesCatalog& catalog)
{
    long double total = 0;
    for (size_t s = 0; s < moles.size(); ++s)
        total += static_cast<long double>(moles[s]) * catalog.species()[s].molarMass;
    return total;
}
SIMPLEFLUID_SOLVERS_LOCAL inline double normalized(long double difference, long double scale, double tolerance)
{
    if (difference == 0)
        return 0;
    if (!(scale > 0) || !std::isfinite(scale) || !std::isfinite(difference))
        return std::numeric_limits<double>::infinity();
    return static_cast<double>(std::abs(difference) / scale / tolerance);
}
SIMPLEFLUID_SOLVERS_LOCAL inline double species_packet(
    std::span<const Flow::CellSources> cells, size_t c, size_t a, size_t s)
{
    return cells.empty() || cells[c].species_moles.empty() ? 0 : cells[c].species_moles[a][s];
}
SIMPLEFLUID_SOLVERS_LOCAL inline double energy_packet(std::span<const Flow::CellSources> cells, size_t c, size_t a)
{
    return cells.empty() || cells[c].internal_energy.empty() ? 0 : cells[c].internal_energy[a];
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_packets(
    std::span<const Flow::CellSources> sources, size_t cells, size_t phases, size_t species)
{
    require(sources.empty() || sources.size() == cells, "Source stage returned a different owned-cell layout.");
    for (const auto& source : sources)
    {
        require((source.species_moles.empty() || source.species_moles.size() == phases) &&
                    (source.internal_energy.empty() || source.internal_energy.size() == phases),
            "Source stage changed phase layout.");
        for (const auto& phase : source.species_moles)
        {
            require(phase.size() == species, "Source stage changed species layout.");
            for (const double value : phase)
                require(std::isfinite(value), "Nonfinite source-stage molar packet.");
        }
        for (const double value : source.internal_energy)
            require(std::isfinite(value), "Nonfinite source-stage energy packet.");
        for (size_t axis = 0; axis < 3; ++axis)
            require(std::isfinite(source.momentum.component(axis)), "Nonfinite source-stage impulse.");
    }
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_events(const std::vector<std::vector<double>>& events, size_t cells)
{
    require(events.empty() || events.size() == cells, "Source stage returned a different gross-event cell layout.");
    for (const auto& cell : events)
        for (const double value : cell)
            require(std::isfinite(value) && value >= 0, "Gross source events must be finite and nonnegative.");
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_event_shape(
    const std::vector<std::vector<double>>& current, const std::vector<std::vector<double>>& previous)
{
    require(current.size() == previous.size(), "Source stage changed its gross-event layout.");
    for (size_t c = 0; c < current.size(); ++c)
        require(current[c].size() == previous[c].size(), "Source stage changed its gross-event dimensions.");
}
SIMPLEFLUID_SOLVERS_LOCAL inline double event_error(
    const std::vector<std::vector<double>>& current, const std::vector<std::vector<double>>& previous, double tolerance)
{
    validate_event_shape(current, previous);
    double maximum = 0;
    for (size_t c = 0; c < current.size(); ++c)
        for (size_t i = 0; i < current[c].size(); ++i)
        {
            const long double now = current[c][i], old = previous[c][i];
            maximum = std::max(maximum, normalized(now - old, std::max(now, old), tolerance));
        }
    return maximum;
}
struct SIMPLEFLUID_SOLVERS_LOCAL CaloricReference
{
    long double energy = 0, capacity = 0;
    bool available = false;
};
SIMPLEFLUID_SOLVERS_LOCAL inline CaloricReference caloric_reference(
    const Flow::TransportPhaseInventory& transported, const PhaseThermalState& guess, const PhaseThermalState& accepted)
{
    const auto& catalog = guess.thermodynamics->catalog();
    const auto guess_mass = mass(guess.moles, catalog), accepted_mass = mass(accepted.moles, catalog);
    const auto stage_mass = mass(transported.moles, catalog);
    const auto* phase = guess_mass > 0 && guess.temperature > 0         ? &guess
                        : accepted_mass > 0 && accepted.temperature > 0 ? &accepted
                                                                        : nullptr;
    if (!phase)
        return {};
    const long double phase_mass = phase == &guess ? guess_mass : accepted_mass;
    long double total = 0;
    for (const double n : phase->moles)
        total += n;
    std::vector<double> x;
    x.reserve(phase->moles.size());
    for (const double n : phase->moles)
        x.push_back(static_cast<double>(static_cast<long double>(n) / total));
    const auto point = phase->thermodynamics->evaluate(phase->temperature, phase->absolute_pressure, x);
    CaloricReference result;
    result.energy = stage_mass > 0 ? static_cast<long double>(transported.internal_energy) / stage_mass
                                   : static_cast<long double>(phase->internal_energy) / phase_mass;
    result.capacity = phase_mass * phase->temperature * point.internal_energy_temperature;
    result.available = true;
    require(std::isfinite(result.energy) && std::isfinite(result.capacity) && result.capacity > 0,
        "Source-stage caloric reference/capacity is invalid.");
    return result;
}
SIMPLEFLUID_SOLVERS_LOCAL inline double source_error(std::span<const Flow::CellSources> current,
    std::span<const Flow::CellSources> previous, const Flow::TransportSnapshot& transported, const Flow::State& guess,
    const Flow::State& accepted, const Flow::Options& options)
{
    double maximum = 0;
    for (size_t c = 0; c < transported.cells.size(); ++c)
    {
        for (size_t a = 0; a < transported.cells[c].phases.size(); ++a)
        {
            const auto& catalog = guess.cells[c].phases[a].thermodynamics->catalog();
            long double mass_current = 0, mass_previous = 0;
            bool active = false;
            for (size_t s = 0; s < catalog.species().size(); ++s)
            {
                const long double now = species_packet(current, c, a, s), old = species_packet(previous, c, a, s);
                active = active || now != 0 || old != 0;
                maximum = std::max(maximum,
                    normalized(now - old, std::max(std::abs(now), std::abs(old)), options.source_relative_tolerance));
                mass_current += now * catalog.species()[s].molarMass;
                mass_previous += old * catalog.species()[s].molarMass;
            }
            const long double energy_current = energy_packet(current, c, a),
                              energy_previous = energy_packet(previous, c, a);
            active = active || energy_current != 0 || energy_previous != 0;
            if (!active)
                continue;
            const auto reference = caloric_reference(
                transported.cells[c].phases[a], guess.cells[c].phases[a], accepted.cells[c].phases[a]);
            if (!reference.available)
            {
                maximum = std::numeric_limits<double>::infinity();
                continue;
            }
            // A common additive mass-energy reference cancels before comparison.
            // No absolute one-joule floor, and no formation-offset-sized tolerance.
            const long double now = energy_current - mass_current * reference.energy;
            const long double old = energy_previous - mass_previous * reference.energy;
            const long double difference =
                (energy_current - energy_previous) - (mass_current - mass_previous) * reference.energy;
            maximum = std::max(
                maximum, normalized(difference, reference.capacity + std::abs(now) + std::abs(old),
                             std::min(options.source_relative_tolerance, options.conservation_relative_tolerance)));
        }
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const long double now = current.empty() ? 0 : current[c].momentum.component(axis);
            const long double old = previous.empty() ? 0 : previous[c].momentum.component(axis);
            maximum = std::max(maximum,
                normalized(now - old, std::max(std::abs(now), std::abs(old)), options.source_relative_tolerance));
        }
    }
    return maximum;
}
SIMPLEFLUID_SOLVERS_LOCAL inline double transport_error(const Flow::TransportSnapshot& current,
    const Flow::TransportSnapshot& previous, const Flow::State& guess, const Flow::State& accepted,
    const Flow::Options& options)
{
    require(current.cells.size() == previous.cells.size() &&
                current.accepted_generation == previous.accepted_generation &&
                current.old_geometry_epoch == previous.old_geometry_epoch &&
                current.new_geometry_epoch == previous.new_geometry_epoch && current.time_step == previous.time_step &&
                current.accepted_time == previous.accepted_time,
        "Transport snapshots do not belong to one immutable preview frame.");
    double maximum = 0;
    for (size_t c = 0; c < current.cells.size(); ++c)
    {
        require(current.old_control_volumes[c] == previous.old_control_volumes[c] &&
                    current.new_control_volumes[c] == previous.new_control_volumes[c],
            "Source corrector changed geometry authority.");
        for (size_t a = 0; a < current.cells[c].phases.size(); ++a)
        {
            const auto& now = current.cells[c].phases[a];
            const auto& old = previous.cells[c].phases[a];
            const auto& catalog = guess.cells[c].phases[a].thermodynamics->catalog();
            for (size_t s = 0; s < now.moles.size(); ++s)
                maximum = std::max(maximum,
                    normalized(static_cast<long double>(now.moles[s]) - old.moles[s],
                        std::max(std::abs(now.moles[s]), std::abs(old.moles[s])), options.source_relative_tolerance));
            const auto reference = caloric_reference(now, guess.cells[c].phases[a], accepted.cells[c].phases[a]);
            const long double delta_mass = mass(now.moles, catalog) - mass(old.moles, catalog);
            const long double delta_energy = static_cast<long double>(now.internal_energy) - old.internal_energy;
            if (reference.available)
                maximum = std::max(
                    maximum, normalized(delta_energy - delta_mass * reference.energy, reference.capacity,
                                 std::min(options.source_relative_tolerance, options.conservation_relative_tolerance)));
            else if (delta_energy != 0 || delta_mass != 0)
                maximum = std::numeric_limits<double>::infinity();
            maximum = std::max(maximum,
                normalized(static_cast<long double>(now.advected_material_volume) - old.advected_material_volume,
                    std::max(std::abs(now.advected_material_volume), std::abs(old.advected_material_volume)),
                    options.volume_relative_tolerance));
            require(!now.population_number && !old.population_number,
                "Number transport is not implemented by the source-only solver.");
        }
    }
    return maximum;
}
SIMPLEFLUID_SOLVERS_LOCAL inline double thermal_error(
    const Flow::State& current, const Flow::State& previous, const Flow::Options& options)
{
    double maximum = 0;
    for (size_t c = 0; c < current.cells.size(); ++c)
        for (size_t a = 0; a < current.cells[c].phases.size(); ++a)
        {
            const auto& now = current.cells[c].phases[a];
            const auto& old = previous.cells[c].phases[a];
            maximum =
                std::max(maximum, normalized(static_cast<long double>(now.temperature) - old.temperature,
                                      std::max(now.temperature, old.temperature), options.thermal_relative_tolerance));
            if (now.temperature > 0 || old.temperature > 0)
            {
                maximum = std::max(maximum,
                    normalized(static_cast<long double>(now.absolute_pressure) - old.absolute_pressure,
                        std::max(now.absolute_pressure, old.absolute_pressure), options.thermal_relative_tolerance));
                // This draft has no capillary pressure-offset model. Later radius
                // integration must replace this common-phase-pressure identity.
                const long double actual =
                    static_cast<long double>(current.thermodynamic_pressure) + current.cells[c].mechanical_pressure;
                const long double prior =
                    static_cast<long double>(previous.thermodynamic_pressure) + previous.cells[c].mechanical_pressure;
                maximum = std::max(maximum, normalized(actual - prior, std::max(std::abs(actual), std::abs(prior)),
                                                options.thermal_relative_tolerance));
                maximum = std::max(
                    maximum, normalized(static_cast<long double>(now.absolute_pressure) - actual,
                                 std::max(std::abs(actual), std::abs(static_cast<long double>(now.absolute_pressure))),
                                 options.thermal_relative_tolerance));
            }
        }
    return maximum;
}
} // namespace SimpleFluid::Reacting::source_stage_detail
