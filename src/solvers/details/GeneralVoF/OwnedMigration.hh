/** @file OwnedMigration.hh
 * @brief Detached Ordinary accepted-state ownership migration.
 */
#pragma once
#include "solvers/details/GeneralVoF/Remap.hh"
namespace SimpleFluid::detail::general_vof
{
/** Uses an already authenticated native migration plan; no publication occurs.
 * Retains accepted curvature while rebuilding the rank-owned capillary catalog.
 * Partition-bound prepared execution caches are absent on the new State;
 * closure_storage_bytes is reset because it describes old workspace allocation,
 * rather than physical inventory. Other accepted diagnostics remain unchanged. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL PreparedRemap prepare_owned_migration(
    const Resources&, const State&, const VoF::GeneralVoF::Hierarchy::Plan&);
} // namespace SimpleFluid::detail::general_vof
