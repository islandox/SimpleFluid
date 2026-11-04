/** @file ClosedContactIndex.hh
 * @brief Private owner-local dynamic closed-range broad phase; no MPI state.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include <limits>
#include <memory>
#include <optional>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct ClosedContactEntry
{
    AdaptiveCellKey key;
    AdaptiveBytes domain;
    std::uint64_t revision = 0;
    int owner = -1;
    std::vector<AdaptiveContactImage> images;
};
struct ClosedContactIndexStatistics
{
    std::size_t entries = 0, images = 0, queries = 0, visited_nodes = 0, scanned_entries = 0;
    /** Admission diagnostics are monotone; the cap applies to simultaneous
     * retained storage and operation scratch, with checked actual capacities. */
    std::size_t admitted_bytes = 0, retained_bytes = 0, peak_bytes = 0;
};
struct ClosedContactCover
{
    AdaptiveBytes domain;
    AdaptiveContactBounds bounds;
    int owner = -1;
    std::size_t images = 0;
};
/** Compact metadata only: the caller retains provider leaves and exact contact
 * records. Queries return pointers valid until that key is erased/replaced.
 * A height-balanced AABB tree indexes individual periodic images. Result
 * vectors become caller-owned on return; callers account overlapping returned
 * buffers in their enclosing budget. Repeated queries do not consume live
 * capacity after their result buffers are destroyed. */
class SIMPLEFLUID_LOCAL ClosedContactIndex
{
public:
    explicit ClosedContactIndex(std::size_t byte_budget);
    ~ClosedContactIndex();
    ClosedContactIndex(ClosedContactIndex&&) noexcept;
    ClosedContactIndex& operator=(ClosedContactIndex&&) noexcept;
    ClosedContactIndex(const ClosedContactIndex&) = delete;
    ClosedContactIndex& operator=(const ClosedContactIndex&) = delete;
    /** Rebuild candidate metadata in canonical cell-key order. The cap applies
     * to candidate retained storage and its construction scratch; the caller
     * separately accounts the still-live source. Query counters are inherited,
     * while admission statistics describe the new candidate's construction. */
    [[nodiscard]] ClosedContactIndex fork(std::size_t byte_budget) const;
    /** Rebind the index cap to the caller's current available credit. Failure
     * leaves the previous cap unchanged. The retained index must fit. */
    void rebind_budget(std::size_t byte_budget);
    /** Allocation-free spatial union across all indexed images, irrespective of
     * owner or domain. Empty indexes have no bounds. */
    [[nodiscard]] std::optional<AdaptiveContactBounds> aggregate_bounds() const noexcept;
    void insert(ClosedContactEntry);
    void erase(const AdaptiveCellKey&);
    void replace(ClosedContactEntry);
    [[nodiscard]] const ClosedContactEntry* find(const AdaptiveCellKey&) const;
    /** Connected-domain equality is required. Results are deduplicated and
     * sorted by the complete canonical typed cell key. Closed bounds use <.
     * scratch_cap admits old and new result buffers during capacity growth;
     * callers subtract their other live operation buffers before supplying it. */
    [[nodiscard]] std::vector<const ClosedContactEntry*> query(std::span<const char> domain,
        std::span<const AdaptiveContactImage> images,
        std::size_t scratch_cap = std::numeric_limits<std::size_t>::max());
    /** Bounded-size actual subtree fragments, split at owner/domain boundaries.
     * Empty owners emit nothing; disconnected ownership remains fragmented. */
    [[nodiscard]] std::vector<ClosedContactCover> ownership_covers(
        std::size_t maximum_fragment_images = 8, std::size_t scratch_cap = std::numeric_limits<std::size_t>::max());
    /** Bounded actual-subtree frontier. Mixed owner/domain subtrees are always
     * split; homogeneous subtrees are split only until the requested target is
     * reached. A mixed tree may require more than the target. Bounds are
     * conservative and every returned subtree has exactly one owner/domain. */
    [[nodiscard]] std::vector<ClosedContactCover> ownership_frontier_covers(
        std::size_t target_fragments = 8, std::size_t scratch_cap = std::numeric_limits<std::size_t>::max());
    [[nodiscard]] const ClosedContactIndexStatistics& statistics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
[[nodiscard]] SIMPLEFLUID_LOCAL bool closed_contact_overlap(
    const AdaptiveContactBounds&, const AdaptiveContactBounds&) noexcept;
[[nodiscard]] SIMPLEFLUID_LOCAL bool closed_contact_overlap(
    const AdaptiveContactImage&, const AdaptiveContactImage&) noexcept;
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
