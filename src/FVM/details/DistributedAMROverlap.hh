/** @file DistributedAMROverlap.hh
 * @brief Shared bounded one-generation ancestry/owner-authenticated overlap bootstrap.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveOverlap.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
namespace SimpleFluid::VoF::detail
{
[[nodiscard]] std::shared_ptr<const Meshes::AMR::AdaptiveOverlapPlan> build_adaptive_overlap(
    const Meshes::AMR::DistributedAdaptiveHierarchy& source, const Meshes::AMR::DistributedAdaptiveHierarchy& target,
    std::size_t byte_budget, std::size_t leaf_fetch_budget = 0);
}
