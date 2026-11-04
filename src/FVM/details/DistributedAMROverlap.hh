/** @file DistributedAMROverlap.hh
 * @brief Shared bounded one-generation ancestry/owner-authenticated overlap bootstrap.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveOverlap.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
namespace SimpleFluid::VoF::detail
{
/** Owner-authenticated ancestry candidates only. Numerical clients must still
 * establish geometric coverage with their own arithmetic/certificate policy. */
struct AdaptiveOverlapCatalog
{
    Meshes::AMR::HierarchyObservation source_observation, target_observation;
    std::vector<Meshes::AMR::AdaptiveCellBinding> source_cells, target_cells;
    std::vector<Meshes::AMR::AdaptiveAncestryRelation> relations;
};
/** Authenticated procedural/native leaf metadata, with no eager convex cell
 * construction. Explicit input certificates remain their existing owners. */
struct AdaptiveOverlapCellMetadata
{
    Meshes::AMR::AdaptiveLeaf leaf;
    std::uint64_t gid = 0;
    int owner = -1;
};
struct AdaptiveOverlapMetadataCatalog
{
    Meshes::AMR::HierarchyObservation source_observation, target_observation;
    std::vector<AdaptiveOverlapCellMetadata> source_cells, target_cells;
    std::vector<Meshes::AMR::AdaptiveAncestryRelation> relations;
};
[[nodiscard]] AdaptiveOverlapMetadataCatalog build_adaptive_overlap_metadata_catalog(
    const Meshes::AMR::DistributedAdaptiveHierarchy& source, const Meshes::AMR::DistributedAdaptiveHierarchy& target,
    std::size_t byte_budget, std::size_t leaf_fetch_budget = 0);
[[nodiscard]] AdaptiveOverlapCatalog build_adaptive_overlap_catalog(
    const Meshes::AMR::DistributedAdaptiveHierarchy& source, const Meshes::AMR::DistributedAdaptiveHierarchy& target,
    std::size_t byte_budget, std::size_t leaf_fetch_budget = 0);
[[nodiscard]] std::shared_ptr<const Meshes::AMR::AdaptiveOverlapPlan> build_adaptive_overlap(
    const Meshes::AMR::DistributedAdaptiveHierarchy& source, const Meshes::AMR::DistributedAdaptiveHierarchy& target,
    std::size_t byte_budget, std::size_t leaf_fetch_budget = 0);
}
