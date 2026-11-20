/** @file LowMachQbmmLinearTransport.hh
 * @brief Fixed-grid realizable linear SSPRK2 advection for native QBMM families.
 */
#pragma once

#include "solvers/details/LowMachQbmmExplicitTransport.hh"

namespace SimpleFluid::Reacting::qbmm_linear_detail
{
using qbmm_explicit_detail::Flow;
using qbmm_explicit_detail::TransportResult;

/** Validate the full immutable XY base of a polygonal disk or annulus. */
SIMPLEFLUID_SOLVERS_LOCAL void validate_cylindrical_base(const Flow::Mesh::SemiStructured& mesh);

enum class TraceLimiter
{
    Outgoing,
    AllIncident
};

[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL TransportResult transport(const SP<const Flow::Mesh>& mesh,
    const Flow::Options& options, const Flow::State& accepted, const Flow::State& thermodynamic_guess,
    Flow::State& candidate, const Flow::Traits::scalar_face_type& carrier_flux,
    const std::vector<std::vector<double>>& accepted_material_volumes, double dt, size_t corrector,
    TraceLimiter limiter = TraceLimiter::Outgoing);
} // namespace SimpleFluid::Reacting::qbmm_linear_detail
