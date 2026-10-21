/** @file DistributedCartesianKernel.hh
 * @brief Detached native Cartesian block recipes and exact local dyadic kernels.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
namespace SimpleFluid::Meshes::AMR
{
/** Copies only the compact grid description. Cartesian periodic axes retain
 * exact image certificates and matching block partner requirements. */
[[nodiscard]] std::shared_ptr<const AdaptiveProviderKernel> cartesian_adaptive_kernel(AMRGrid);
/** Construct one procedural cell record; repack shares its actual block recipe
 * with other locally resident cells in that block. No source hierarchy is held. */
[[nodiscard]] AdaptiveLeaf cartesian_adaptive_leaf(const AMRGrid&, AdaptiveCellKey, AdaptiveBytes domain);
} // namespace SimpleFluid::Meshes::AMR
