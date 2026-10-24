/** Private flow metadata and reference-independent conditioned thermal convergence. */
#pragma once
#include "solvers/LowMachMixtureSolver.hh"
#include "solvers/details/LowMachSourceStage.hh"
#include "utils/CompensatedSum.hh"
#include <iomanip>
#include <locale>
#include <numeric>
#include <sstream>

namespace SimpleFluid::Reacting::conditioned_thermal_detail
{
using Flow = LowMachMixtureSolver;
SIMPLEFLUID_SOLVERS_LOCAL inline ThermalBatchIdentity identity(const Flow::TransportSnapshot& snapshot)
{
    return {snapshot.stage.provider_identity, snapshot.stage.provider_revision, snapshot.stage.invocation,
        static_cast<std::uint64_t>(snapshot.stage.corrector), snapshot.schema_fingerprint, snapshot.accepted_generation,
        snapshot.old_geometry_epoch, snapshot.new_geometry_epoch};
}
SIMPLEFLUID_SOLVERS_LOCAL inline std::string signature(const InterphaseConfiguration& config)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    auto text = [&](const std::string& value) { out << ':' << value.size() << ':' << value; };
    auto metadata = [&](const ThermalLawIdentity& law)
    {
        text(law.identity);
        out << ':' << law.revision << ':' << law.schema_fingerprint;
    };
    metadata(config.metadata);
    out << ':' << config.stages.size() << ':' << config.phase_enthalpy.size();
    for (const auto& stage : config.stages)
    {
        text(stage.identity);
        out << ':' << int(stage.kind) << ':' << stage.material.size() << ':' << stage.heat.size();
        for (const auto& slot : stage.material)
            out << ':' << slot.species << ':' << slot.donor << ':' << slot.recipient;
        for (const auto& slot : stage.heat)
            out << ':' << slot.first << ':' << slot.second;
    }
    for (const auto& law : config.phase_enthalpy)
    {
        out << ':' << bool(law);
        if (!law)
            continue;
        metadata(law->metadata);
        out << ':' << law->bounds.minimum_temperature << ':' << law->bounds.maximum_temperature << ':'
            << law->bounds.minimum_pressure << ':' << law->bounds.maximum_pressure << ':' << bool(law->evaluate);
    }
    out << ':' << bool(config.caloric_domain);
    if (config.caloric_domain)
    {
        metadata(config.caloric_domain->metadata);
        out << ':' << bool(config.caloric_domain->evaluate);
    }
    return out.str();
}
SIMPLEFLUID_SOLVERS_LOCAL inline void validate_batch(const PressureConditionedThermalBatch& batch,
    const Flow::SourceStageContext& context, std::span<const Flow::CellSources> sources)
{
    using source_stage_detail::require;
    require(batch.configuration && batch.source == identity(context.transport) &&
                batch.cells.size() == context.transport.cells.size() && sources.size() == batch.cells.size(),
        "Conditioned thermal batch identity/owned rows differ from its exact transport receipt.");
    for (size_t c = 0; c < batch.cells.size(); ++c)
    {
        const auto& plan = batch.cells[c];
        require(plan.configuration.get() == batch.configuration.get(),
            "A conditioned cell plan changed immutable law/slot ownership.");
        const auto np = context.transport.cells[c].phases.size();
        require(plan.pre_moles.size() == np && plan.final_moles.size() == np && plan.raw_static_energy.size() == np &&
                    plan.amounts.size() == batch.configuration->stages.size(),
            "Conditioned cell plan phase/event layout differs.");
        validate_interphase_configuration(*batch.configuration, context.thermodynamic_trial.cells.at(c).phases);
        for (size_t a = 0; a < np; ++a)
        {
            SimpleFluid::detail::CompensatedSum<long double> raw;
            for (auto value : plan.raw_static_energy[a])
            {
                require(std::isfinite(value), "Nonfinite raw static thermal channel.");
                raw += value;
            }
            const auto projection = static_cast<double>(raw.value());
            require(std::isfinite(projection) && projection == source_stage_detail::energy_packet(sources, c, a),
                "Conditioned raw static energy does not project to the exact provider-returned packet.");
            require(plan.final_moles[a].size() == context.transport.cells[c].phases[a].moles.size(),
                "Conditioned final species layout differs.");
            for (size_t i = 0; i < plan.final_moles[a].size(); ++i)
            {
                const auto before = context.transport.cells[c].phases[a].moles[i];
                const double actual = before + source_stage_detail::species_packet(sources, c, a, i),
                             expected = plan.final_moles[a][i];
                const long double scale =
                    std::abs(static_cast<long double>(actual)) + std::abs(static_cast<long double>(expected));
                require(std::isfinite(actual) && actual >= 0 && expected >= 0 && (actual == 0) == (expected == 0) &&
                            std::abs(static_cast<long double>(actual) - expected) <=
                                64 * std::numeric_limits<double>::epsilon() * scale,
                    "Conditioned material plan differs from the actual static packet endpoint.");
            }
        }
    }
}
/** Pure numerical reference gauge, not a phase-temperature assignment. A per-species
 * additive caloric shift cancels identically between raw energy and reference stock.
 * EOS provider failures stay hard errors; none is interpreted as a search signal.
 */
SIMPLEFLUID_SOLVERS_LOCAL inline long double reference_energy(
    const PhaseThermalState& phase, std::span<const double> moles)
{
    SimpleFluid::detail::CompensatedSum<long double> amount, mass;
    const auto& catalog = phase.thermodynamics->catalog();
    for (size_t i = 0; i < moles.size(); ++i)
    {
        amount += moles[i];
        mass += static_cast<long double>(moles[i]) * catalog.species()[i].molarMass;
    }
    if (amount.value() == 0)
        return 0;
    auto x = std::vector<double>(moles.begin(), moles.end());
    for (auto& value : x)
        value = static_cast<double>(value / amount.value());
    const auto& b = phase.thermodynamics->bounds();
    return mass.value() *
           phase.thermodynamics
               ->evaluate(b.minimum_temperature, std::midpoint(b.minimum_pressure, b.maximum_pressure), x)
               .internal_energy;
}
SIMPLEFLUID_SOLVERS_LOCAL inline double convergence(const PressureConditionedThermalBatch& current,
    const PressureConditionedThermalBatch& previous, std::span<const PressureConditionedThermalEvaluation> now,
    std::span<const PressureConditionedThermalEvaluation> old, const Flow::TransportSnapshot& transported,
    const Flow::TransportSnapshot& previous_transport, double relative_tolerance)
{
    using source_stage_detail::normalized;
    using source_stage_detail::require;
    require(signature(*current.configuration) == signature(*previous.configuration),
        "Conditioned stage metadata changed within one preview.");
    require(
        current.cells.size() == previous.cells.size() && now.size() == current.cells.size() && old.size() == now.size(),
        "Conditioned convergence row layout differs.");
    double maximum = 0;
    for (size_t c = 0; c < now.size(); ++c)
    {
        const auto& a = current.cells[c];
        const auto& b = previous.cells[c];
        require(a.amounts.size() == b.amounts.size(), "Conditioned event-stage dimensions changed.");
        for (size_t j = 0; j < a.amounts.size(); ++j)
        {
            require(a.amounts[j].size() == b.amounts[j].size(), "Conditioned immutable event-slot dimensions changed.");
            for (size_t i = 0; i < a.amounts[j].size(); ++i)
                maximum = std::max(
                    maximum, normalized(static_cast<long double>(a.amounts[j][i]) - b.amounts[j][i],
                                 std::max(std::abs(a.amounts[j][i]), std::abs(b.amounts[j][i])), relative_tolerance));
        }
        const long double caloric =
            now[c].caloric_scale + old[c].caloric_scale + now[c].work_magnitude + old[c].work_magnitude;
        for (size_t p = 0; p < a.pre_moles.size(); ++p)
        {
            SimpleFluid::detail::CompensatedSum<long double> raw_change, evaluated_change;
            for (auto value : a.raw_static_energy[p])
                raw_change += value;
            for (auto value : b.raw_static_energy[p])
                raw_change += -value;
            // Remove only the numerical reference-gauge composition change, keeping all physical sensible heat.
            raw_change += -reference_energy(now[c].phases[p], a.pre_moles[p]);
            raw_change += reference_energy(now[c].phases[p], transported.cells[c].phases[p].moles);
            raw_change += reference_energy(old[c].phases[p], b.pre_moles[p]);
            raw_change += -reference_energy(old[c].phases[p], previous_transport.cells[c].phases[p].moles);
            for (auto value : now[c].raw_applied_phase_energy[p])
                evaluated_change += value;
            for (auto value : old[c].raw_applied_phase_energy[p])
                evaluated_change += -value;
            evaluated_change += -reference_energy(now[c].phases[p], now[c].phases[p].moles);
            evaluated_change += reference_energy(now[c].phases[p], transported.cells[c].phases[p].moles);
            evaluated_change += reference_energy(old[c].phases[p], old[c].phases[p].moles);
            evaluated_change += -reference_energy(old[c].phases[p], previous_transport.cells[c].phases[p].moles);
            maximum = std::max(maximum, normalized(raw_change.value(), caloric, relative_tolerance));
            maximum = std::max(maximum, normalized(evaluated_change.value(), caloric, relative_tolerance));
        }
    }
    return maximum;
}
} // namespace SimpleFluid::Reacting::conditioned_thermal_detail
