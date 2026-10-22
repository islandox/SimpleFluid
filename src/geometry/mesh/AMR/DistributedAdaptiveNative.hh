/** @file DistributedAdaptiveNative.hh
 * @brief Publish an authority-certified local draft through actual native maps.
 */
#pragma once
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include "parallel/CompositePartition.hh"
namespace SimpleFluid::Meshes::AMR
{
class DistributedAdaptiveHierarchy;
struct AdaptiveNativePublication
{
    std::shared_ptr<const CompositePartition<>> partition;
    std::shared_ptr<const MeshHandle<>> mesh;
    std::map<AdaptiveCellKey, std::uint64_t> owned_gids;
    std::map<std::uint64_t, AdaptiveCellKey> local_keys;
    MeshStorageReport storage;
};
/** The draft includes complete full owned/ghost shells, metric-only outer
 * neighbors and exact physical node proofs. Only compact original-root metadata
 * is common across ranks; no active global region/cell catalog is constructed. */
[[nodiscard]] AdaptiveNativePublication assemble_adaptive_native(AdaptiveNativeDraft, std::span<const char> root_schema,
    Teuchos::RCP<const Teuchos::Comm<int>>, std::size_t byte_budget);
/** Fetch complete owner-authenticated first-ring and explicitly requested full
 * ghost records, plus metric-only outer neighbors. No recursive halo closure or
 * global catalog is constructed. The native publisher uses this identical path. */
[[nodiscard]] AdaptiveNativeDraft fetch_adaptive_native_draft(
    const DistributedAdaptiveHierarchy&, std::span<const AdaptiveCellKey> additional_full_ghosts = {});
/** Owner-authenticated first-ring and caller-declared full ghosts, with outer
 * metric neighbors. Additional keys are bounded by the hierarchy byte budget. */
[[nodiscard]] AdaptiveNativePublication assemble_adaptive_native(const DistributedAdaptiveHierarchy&,
    std::span<const char> root_schema, std::span<const AdaptiveCellKey> additional_full_ghosts = {});
} // namespace SimpleFluid::Meshes::AMR
