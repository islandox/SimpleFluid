/** @file ExperimentalPhaseInventory.hh
 * @brief Explicitly opt-in unqualified exact phase/provenance payload.
 */
#pragma once
#include "FVM/PhaseVolumes.hh"
#include "geometry/ExactSweptGeometry.hh"
namespace SimpleFluid::VoF
{
/** Allocated only under ExperimentalExact; finite mode has a null shared pointer. */
struct ExperimentalPhaseError
{
    Meshes::ExactGeometryMeasure difference, admitted_bound;
    Meshes::ExactGeometryMeasure native_conversion, phase_input_bound;
    Meshes::ExactGeometryMeasure fit_residual, fit_bound, transferred_bound, fit_allocation_bound;
};
struct ExperimentalPhaseInventory
{
    Meshes::ExactPhaseMeasures phases;
    ExperimentalPhaseError error;
};
} // namespace SimpleFluid::VoF
