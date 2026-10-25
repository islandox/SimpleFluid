/** @file NodeVelocity.hh
 * @brief Owner-authenticated trajectory node velocities from native cell fields.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "FVM/GeometricVoF.hh"
#include "SimpleFluidExport.hh"
#include <span>
#include <vector>

namespace SimpleFluid::detail::general_vof
{
/** Collective, including empty owners. Borrows the source phase publication and
 * native maps; never modifies fields or publishes a solver state. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL std::vector<VoF::GeneralVoFNodeVelocity> node_velocity(const MeshHandle<>& mesh,
    const VoF::GeneralVoF& phases, std::span<const VoF::Vector> owned_velocity, std::size_t byte_budget);
} // namespace SimpleFluid::detail::general_vof
