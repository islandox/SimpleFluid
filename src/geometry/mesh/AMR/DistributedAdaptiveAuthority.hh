/** @file DistributedAdaptiveAuthority.hh
 * @brief Sparse exact shared-surface overlay and incident-edge decisions.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include "parallel/SparseOwnerExchange.hh"
namespace SimpleFluid::Meshes::AMR
{
/** Payload counters for the exact edge phase, excluding transport framing. */
struct AdaptiveAuthorityStats
{
    std::size_t edge_request_bytes = 0, edge_reply_bytes = 0, edge_reply_points = 0;
    std::uint64_t global_faces = 0, global_nodes = 0;
};
/** Lossless owner/ghost tile packet retaining exact image and both physical node loops. */
[[nodiscard]] AdaptiveBytes encode_adaptive_tile(const AdaptiveTile&);
[[nodiscard]] AdaptiveTile decode_adaptive_tile(std::span<const char>);
/** All owned leaf support occurrences participate. Authorities require complete
 * opposite interior coverage, never infer exterior from missing remote input.
 * Returned tiles touch a local owned cell and retain exact propagated vertices.
 * This is transient interface work, not a retained global cell certificate set. */
[[nodiscard]] std::vector<AdaptiveTile> decide_adaptive_tiles(std::span<const AdaptiveLeaf> owned,
    Teuchos::RCP<const Teuchos::Comm<int>> communicator, std::size_t byte_budget, std::size_t tile_budget,
    AdaptiveAuthorityStats* statistics = nullptr, std::vector<AdaptiveNode>* node_catalog = nullptr);
} // namespace SimpleFluid::Meshes::AMR
