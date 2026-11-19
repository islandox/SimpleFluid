/** @file LowMachQbmmKineticTransport.hh
 * @brief Synchronized fixed-grid kinetic advection of the native QBMM families.
 */
#pragma once

#include "solvers/details/LowMachQbmmExplicitTransport.hh"

namespace SimpleFluid::Reacting::qbmm_kinetic_detail
{
using qbmm_explicit_detail::FaceIntegrals;
using qbmm_explicit_detail::Flow;
using qbmm_explicit_detail::TransportResult;

/** Updates only the bound families in the private corrector candidate. All
 * ranks take the same substeps. Carrier flux and the thermodynamic guess are
 * borrowed for this corrector and stay fixed throughout the interval.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL TransportResult transport(const SP<const Flow::Mesh>& mesh,
    const Flow::Options& options, const Flow::State& accepted, const Flow::State& thermodynamic_guess,
    Flow::State& candidate, const Flow::Traits::scalar_face_type& carrier_flux,
    const std::vector<std::vector<double>>& accepted_material_volumes, double dt, size_t corrector);
} // namespace SimpleFluid::Reacting::qbmm_kinetic_detail
