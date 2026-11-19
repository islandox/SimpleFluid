/** @file LowMachQbmmLawInputs.hh
 * @brief MPI-free native law snapshots and endpoint gates; callers own collective fences.
 */
#pragma once

#include "equations/QbmmLawInputs.hh"
#include "solvers/LowMachMixtureSolver.hh"

#include <numeric>

namespace SimpleFluid::Reacting::qbmm_native_law_detail
{
using Flow = LowMachMixtureSolver;

/** Applies also to count-only phases and native configurations with no QBMM schema. */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_drift(const Flow::CellState& cell)
{
    if (cell.drift_velocity.size() != cell.phases.size())
        throw std::invalid_argument("Native supplied phase drift layout differs from physical phases.");
    for (const auto& velocity : cell.drift_velocity)
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (!std::isfinite(velocity.component(axis)))
                throw std::invalid_argument("Native supplied phase drift must be finite, including count-only phases.");
}

SIMPLEFLUID_SOLVERS_LOCAL inline bool is_transport_gate(const QbmmLawMetadata& law)
{
    return law.gate == QbmmApplicabilityGate::RejectOutside && qbmm_is_transport_law(law);
}

/** The native state supplies a carrier-liquid basis only when its liquid slot is unique. */
SIMPLEFLUID_SOLVERS_LOCAL inline std::optional<std::size_t> carrier_liquid_index(
    std::span<const PhaseThermalState> phases)
{
    std::optional<std::size_t> liquid;
    for (std::size_t p = 0; p < phases.size(); ++p)
        if (phases[p].thermodynamics->kind() == EOSKind::Liquid)
        {
            if (liquid)
                return std::nullopt;
            liquid = p;
        }
    return liquid;
}

/** Correctors supply recovered EOS volumes from the exact viscosity inputs. Endpoint
 * callers without a recovered snapshot sample the physical state for its EOS volumes.
 */
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_transport_gates(const Flow::Options& options,
    const Flow::CellState& cell, double bulk_volume_m3, double time_step_s,
    std::span<const double> recovered_volumes = {})
{
    const auto& families = options.qbmm_families;
    const auto active_gate = [&](std::size_t p, const QbmmLawMetadata& law)
    { return is_transport_gate(law) && (!law.requires_population || cell.phase_numbers.at(p).value() > 0); };
    bool has_gate = false, needs_liquid = false;
    for (std::size_t p = 0; p < families.size(); ++p)
        if (families[p])
            for (const auto& law : families[p]->closure_versions().laws)
                if (active_gate(p, law))
                {
                    has_gate = true;
                    needs_liquid = needs_liquid || qbmm_uses_carrier_liquid(law);
                }
    if (!has_gate)
        return;

    const auto liquid = needs_liquid ? carrier_liquid_index(cell.phases) : std::nullopt;
    if (!recovered_volumes.empty() && recovered_volumes.size() != cell.phases.size())
        throw std::invalid_argument("Native transport law recovered volume layout differs from physical phases.");
    std::vector<double> volumes(cell.phases.size()), specific_volumes(cell.phases.size());
    double gas_volume = 0;
    std::optional<std::vector<double>> liquid_concentrations;
    std::optional<double> liquid_temperature;
    for (std::size_t p = 0; p < cell.phases.size(); ++p)
    {
        const auto& phase = cell.phases[p];
        if (!phase.thermodynamics || phase.moles.size() != phase.thermodynamics->catalog().species().size())
            throw std::invalid_argument("Native law input material layout differs from its thermodynamic catalog.");
        const bool gas = phase.thermodynamics->kind() == EOSKind::Gas;
        if (!gas && liquid != p)
            continue;
        const double total_moles = std::accumulate(phase.moles.begin(), phase.moles.end(), 0.);
        if (total_moles == 0)
            continue;
        // Use the native report's mass and composition arithmetic at inclusive range boundaries.
        double mass = 0;
        for (std::size_t s = 0; s < phase.moles.size(); ++s)
            mass += phase.moles[s] * phase.thermodynamics->catalog().species()[s].molarMass;
        if (recovered_volumes.empty())
        {
            auto composition = phase.moles;
            for (auto& amount : composition)
                amount /= total_moles;
            const auto point = phase.thermodynamics->evaluate(phase.temperature, phase.absolute_pressure, composition);
            volumes[p] = mass / point.density;
            specific_volumes[p] = 1 / point.density;
        }
        else
        {
            volumes[p] = recovered_volumes[p];
            specific_volumes[p] = volumes[p] / mass;
        }
        if (!std::isfinite(volumes[p]) || volumes[p] <= 0)
            throw std::invalid_argument("Native law input physical phase volume is not representable.");
        if (gas)
            gas_volume += volumes[p];
        else
        {
            liquid_concentrations = qbmm_phase_concentrations(phase.moles, volumes[p]);
            liquid_temperature = phase.temperature;
        }
    }
    for (std::size_t p = 0; p < families.size(); ++p)
    {
        const auto active = [&](const auto& law) { return active_gate(p, law); };
        if (!families[p] || std::ranges::none_of(families[p]->closure_versions().laws, active))
            continue;
        const auto& family = *families[p];
        const auto& phase = cell.phases.at(p);
        const QbmmCellView view{family.population(), family.catalog().fingerprint(), phase.moles,
            cell.phase_numbers.at(p).value(), cell.qbmm_shape.at(p).value(), family.fingerprint()};
        const auto quadrature = invert_qbmm_moments(qbmm_cell_moments(family, view));
        const std::optional<double> specific_volume =
            volumes[p] > 0 ? std::optional<double>(specific_volumes[p]) : std::nullopt;
        const auto& drift = cell.drift_velocity.at(p);
        const auto inputs = make_qbmm_law_inputs(
            family, {quadrature, qbmm_geometry(family, view, bulk_volume_m3, specific_volume), phase.moles,
                        bulk_volume_m3, volumes[p], phase.absolute_pressure, phase.temperature, time_step_s,
                        gas_volume / bulk_volume_m3, liquid_concentrations, liquid_temperature,
                        {drift.component(0), drift.component(1), drift.component(2)}});
        for (const auto& law : family.closure_versions().laws)
            if (active(law) && !evaluate_qbmm_applicability(law, inputs).gate_passed)
                throw std::invalid_argument("Native QBMM family '" + family.population() + "' law '" + law.identity +
                                            "' is outside its declared applicability or lacks a required input.");
    }
}
} // namespace SimpleFluid::Reacting::qbmm_native_law_detail
