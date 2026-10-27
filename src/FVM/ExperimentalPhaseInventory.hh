/** @file ExperimentalPhaseInventory.hh
 * @brief Explicitly opt-in unqualified exact phase/provenance payload.
 */
#pragma once
#include "FVM/PhaseVolumes.hh"
#include "geometry/swept_volume/SweptGeometry.hh"
namespace SimpleFluid::VoF
{
/** Allocated only under ExperimentalExact; finite mode has a null shared pointer. */
struct ExperimentalPhaseError
{
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> difference, admitted_bound;
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> native_conversion, phase_input_bound;
    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> fit_residual, fit_bound, transferred_bound,
        fit_allocation_bound;
};
struct ExperimentalPhaseInventory
{
    SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> phases;
    ExperimentalPhaseError error;
};
} // namespace SimpleFluid::VoF
