/** @file DistributedMixedKernel.hh
 * @brief Detached mixed-provider dispatch and needed original-root join aliases.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
namespace SimpleFluid::Meshes::AMR
{
struct AdaptiveJoinRule
{
    AdaptiveBytes original_support, canonical_support;
    ExactPlanarFacePtr original_root_selection;
};
struct AdaptiveBoundaryRule
{
    AdaptiveBytes original_support;
    ExactPlanarFacePtr original_root_selection;
    std::string boundary; ///< Accepted physical destination; empty means unnamed.
};
/** The immutable route is a participant/root identity, never a current native
 * fragment ordinal. Selections contain only needed original-root supports;
 * current active face memberships are not a detached-kernel catalog. */
struct AdaptiveJoinCapsule
{
    std::uint64_t route = 0;
    std::vector<AdaptiveJoinRule> rules;
    std::vector<AdaptiveBoundaryRule> boundaries;
};
[[nodiscard]] AdaptiveBytes encode_join_capsule(const AdaptiveJoinCapsule&, std::size_t byte_budget);
[[nodiscard]] AdaptiveJoinCapsule decode_join_capsule(std::span<const char>, std::size_t byte_budget);

struct DistributedMixedRoute
{
    std::uint64_t identity = 0, original_region = 0, region = 0;
    AdaptiveProviderKind provider = AdaptiveProviderKind::Polyhedral;
    std::shared_ptr<const AdaptiveProviderKernel> kernel;
    AdaptiveBytes compact_root_schema;
    bool periodic = false;
};
/** Exact-path stationary bootstrap. Unsupported analytic/nonconvex cells reject
 * here and must use the separate whole-model FrozenNative migration path. */
[[nodiscard]] AdaptiveBootstrap detach_static_participant(const MultiRegionMesh::Region&, std::uint64_t provider_root,
    std::uint64_t region, std::uint64_t material, std::span<const std::uint64_t> owned_native_cells,
    AdaptiveBytes domain);
/** Every route owns a pure value/configuration kernel. No route may retain the
 * bootstrap MixedAMR, native global catalog or adaptive participant object. */
[[nodiscard]] std::shared_ptr<const AdaptiveProviderKernel> mixed_adaptive_kernel(
    std::vector<DistributedMixedRoute> routes, std::size_t byte_budget);
} // namespace SimpleFluid::Meshes::AMR
