/** @file DistributedCartesianKernel.hh
 * @brief Detached native Cartesian block recipes and exact local dyadic kernels.
 */
#pragma once
#include "geometry/details/NativeAllocationEnvelope.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include <cstddef>
#include <optional>
namespace SimpleFluid::Meshes::AMR
{
/** Copies only the compact grid description. Cartesian periodic axes retain
 * exact image certificates and matching block partner requirements. */
[[nodiscard]] std::shared_ptr<const AdaptiveProviderKernel> cartesian_adaptive_kernel(AMRGrid);
/** Explicit trusted capability. Adapters default to unsupported; enum tags or
 * parsed descriptors cannot promise the Cartesian constructor semantics. */
[[nodiscard]] std::optional<AMRGrid> trusted_cartesian_incidence_grid(const AdaptiveProviderKernel&) noexcept;
namespace adaptive_detail
{
/** Contact support replay uses the same concrete numeric kernel without
 * allocating publication schema or granting publication authority. */
[[nodiscard]] SIMPLEFLUID_LOCAL std::shared_ptr<const AdaptiveProviderKernel> cartesian_incidence_replay_kernel(AMRGrid);
[[nodiscard]] SIMPLEFLUID_LOCAL geometry_detail::NativeTrackedOwner<const AdaptiveProviderKernel>
cartesian_incidence_replay_kernel(AMRGrid, const geometry_detail::NativeConstructionAdmission&);
/** Actual object payload of the private concrete type issued by this factory;
 * diagnostic only, with implementation-defined shared control overhead excluded. */
[[nodiscard]] SIMPLEFLUID_LOCAL std::size_t native_cartesian_kernel_payload_bytes() noexcept;
} // namespace adaptive_detail
/** Construct one procedural cell record; repack shares its actual block recipe
 * with other locally resident cells in that block. No source hierarchy is held. */
[[nodiscard]] AdaptiveLeaf cartesian_adaptive_leaf(const AMRGrid&, AdaptiveCellKey, AdaptiveBytes domain);
} // namespace SimpleFluid::Meshes::AMR
