#pragma once

#include "equations/PressureConditionedThermal.hh"

namespace SimpleFluid::Reacting::conditioned_thermal_detail
{
// The optional Kelvin radii describe how the final phase temperatures were
// resolved. They are transient inputs to the next source corrector.
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL PressureConditionedThermalEvaluation evaluate_with_resolution(
    const FrozenInterphasePlan& plan, std::span<const PhaseThermalState> final_templates,
    std::span<const double> advected_energy, std::span<const double> advected_volumes, double pressure,
    PhaseThermalMode mode, ThermalEvaluationControls controls, std::span<double> temperature_resolution);
}
