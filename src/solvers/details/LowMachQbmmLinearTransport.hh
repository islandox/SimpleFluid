/** @file LowMachQbmmLinearTransport.hh
 * @brief Realizable linear SSPRK2 advection for native QBMM families.
 */
#pragma once

#include "solvers/details/LowMachQbmmExplicitTransport.hh"

#include <span>

namespace SimpleFluid::Reacting::qbmm_linear_detail
{
using qbmm_explicit_detail::Flow;
using qbmm_explicit_detail::TransportResult;

enum class TraceLimiter
{
    Outgoing,
    AllIncident
};

struct SIMPLEFLUID_SOLVERS_LOCAL PlanarFaceGeometry
{
    Flow::Mesh::Vec3 owner_to_neighbor, owner_to_face, neighbor_to_face;
};

/** Capture accepted interior face offsets in local order, including partition faces. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL std::vector<PlanarFaceGeometry> capture_planar_face_geometry(
    const Flow::Mesh& mesh);

struct SIMPLEFLUID_SOLVERS_LOCAL PlanarTransportGeometry
{
    const FVM::ALEControlVolumeState& ale;
    std::span<const PlanarFaceGeometry> old_faces;
};

/** The frozen carrier flux is absolute. Planar transport assembles its mesh-relative flux from the trial. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL TransportResult transport(const SP<const Flow::Mesh>& mesh,
    const Flow::Options& options, const Flow::State& accepted, const Flow::State& thermodynamic_guess,
    Flow::State& candidate, const Flow::Traits::scalar_face_type& carrier_flux,
    const std::vector<std::vector<double>>& accepted_material_volumes, double dt, size_t corrector,
    TraceLimiter limiter = TraceLimiter::Outgoing, const PlanarTransportGeometry* planar = nullptr);
} // namespace SimpleFluid::Reacting::qbmm_linear_detail
