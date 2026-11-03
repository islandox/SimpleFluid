/** @file ClosedContactIndex.hh
 * @brief Private owner-local dynamic closed-range broad phase; no MPI state.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include <limits>
#include <memory>

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
    void insert(ClosedContactEntry);
    void erase(const AdaptiveCellKey&);
    void replace(ClosedContactEntry);
    [[nodiscard]] const ClosedContactEntry* find(const AdaptiveCellKey&) const;
    /** Connected-domain equality is required. Results are deduplicated and
     * sorted by the complete canonical typed cell key. Closed bounds use <.
     * scratch_cap admits old and new result buffers during capacity growth;
     * callers subtract their other live operation buffers before supplying it. */
    [[nodiscard]] std::vector<const ClosedContactEntry*> query(
        std::span<const char> domain, std::span<const AdaptiveContactImage> images,
        std::size_t scratch_cap = std::numeric_limits<std::size_t>::max());
    /** Bounded-size actual subtree fragments, split at owner/domain boundaries.
     * Empty owners emit nothing; disconnected ownership remains fragmented. */
    [[nodiscard]] std::vector<ClosedContactCover> ownership_covers(std::size_t maximum_fragment_images = 8,
        std::size_t scratch_cap = std::numeric_limits<std::size_t>::max());
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
