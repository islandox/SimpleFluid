/** @file DistributedAdaptiveHierarchy.hh
 * @brief Source-pinned owner-local adaptation with split-family authority.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
#include "parallel/CompositePartition.hh"

#include <map>
#include <optional>
namespace SimpleFluid::Meshes::AMR
{
struct AdaptiveBalancePlan;
struct DistributedAdaptiveOptions
{
    std::size_t byte_budget = 64 * 1024 * 1024, maximum_owned = 65536, maximum_tiles = 262144;
    unsigned maximum_closure_rounds = 64;
    double maximum_span_ratio = 2;
};
/** Inputs must already be detached from bootstrap hierarchy/provider state.
 * Leaf values may share immutable compact procedural recipes locally. Saved
 * family capsules are retained only at their actual anchor-descendant owner. */
class DistributedAdaptiveHierarchy
{
public:
    struct Data;
    struct Storage
    {
        std::size_t owned_leaves = 0, ghost_leaves = 0, family_capsules = 0, directory_entries = 0;
        std::size_t owned_bytes = 0, ancestor_bytes = 0, directory_bytes = 0, procedural_bytes = 0;
        std::size_t native_bytes = 0, native_map_entries = 0;
    };
    DistributedAdaptiveHierarchy(std::vector<AdaptiveLeaf> owned, std::vector<AdaptiveFamily> families,
        std::shared_ptr<const AdaptiveProviderKernel> kernel, Teuchos::RCP<const Teuchos::Comm<int>> communicator,
        DistributedAdaptiveOptions options = {});
    [[nodiscard]] const std::map<AdaptiveCellKey, AdaptiveLeaf>& owned() const noexcept;
    [[nodiscard]] const std::map<AdaptiveBytes, AdaptiveFamily>& families() const noexcept;
    /** Capacity-based retained allocation estimate, shared components counted
     * once; excludes allocator/control-block overhead and is distinct from RSS. */
    [[nodiscard]] Storage storage() const;
    [[nodiscard]] std::uint64_t topology_generation() const noexcept;
    [[nodiscard]] std::uint64_t partition_generation() const noexcept;
    [[nodiscard]] const std::shared_ptr<const void>& identity() const noexcept;
    /** Collective sparse owner lookup; address is the current observed native
     * GID. Optional cap presence must agree collectively; values may differ by rank.
     * These transient caps are positive local application leases no
     * larger than options().byte_budget; exhaustion rejects collectively.
     * Omitting the cap preserves the original service budget. */
    [[nodiscard]] std::vector<AdaptiveOwner> lookup_cells(
        std::span<const AdaptiveCellKey>, bool require_all = true, std::optional<std::size_t> transient_cap = {}) const;
    [[nodiscard]] std::vector<AdaptiveOwner> lookup_families(
        std::span<const AdaptiveBytes>, bool require_all = true, std::optional<std::size_t> transient_cap = {}) const;
    [[nodiscard]] AdaptiveBytes configuration_descriptor() const;
    /** Preserves key/material/revision/exact shape. Native descriptors may be
     * repacked to the requested subset: returned native_cell is valid only in
     * its returned recipe. Resolve tile faces by persistent support+geometry,
     * never reuse an owner's ephemeral native face/cell ordinal. Explicitly
     * leased fetches retain original immutable recipes and ordinals instead of
     * constructing new subset views before packet admission. */
    [[nodiscard]] std::vector<AdaptiveLeaf> fetch_leaves(
        std::span<const AdaptiveCellKey>, std::optional<std::size_t> transient_cap = {}) const;
    [[nodiscard]] std::vector<AdaptiveFamily> fetch_families(
        std::span<const AdaptiveBytes>, std::optional<std::size_t> transient_cap = {}) const;
    [[nodiscard]] const Teuchos::RCP<const Teuchos::Comm<int>>& communicator() const noexcept;
    [[nodiscard]] const DistributedAdaptiveOptions& options() const noexcept;
    [[nodiscard]] std::uint64_t global_cells() const noexcept;
    [[nodiscard]] PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const AdaptiveLeaf&) const;
    [[nodiscard]] AdaptiveCellMetrics metrics(const AdaptiveLeaf&) const;
    [[nodiscard]] std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf&) const;
    [[nodiscard]] HierarchyObservation observation() const noexcept;
    [[nodiscard]] bool geometry_unchanged() const noexcept;
    [[nodiscard]] const std::shared_ptr<const CompositePartition<>>& native_partition() const noexcept;
    [[nodiscard]] std::shared_ptr<const DistributedAdaptiveHierarchy> with_partition(
        std::shared_ptr<const CompositePartition<>>, Teuchos::RCP<const DefaultTpetraTypes::map_type> owned_map) const;
    struct NativeMigrationDiagnostics
    {
        std::uint64_t remote_sent_payload_bytes = 0, remote_received_payload_bytes = 0;
        std::uint64_t remote_sent_packets = 0, remote_received_packets = 0;
        std::uint64_t remote_sent_peers = 0, remote_received_peers = 0, self_payload_bytes = 0;
        /** Packet capacities plus transport application planning/framing scratch;
         * excludes provider decode/state/maps/halo/directory buffers and MPI. */
        std::size_t peak_native_packet_buffers = 0;
    };
    class Plan
    {
    public:
        [[nodiscard]] const NativeMigrationDiagnostics& native_migration() const noexcept { return native_migration_; }
        [[nodiscard]] std::uint64_t actual_migration_application_bytes() const noexcept
        {
            return native_migration_.remote_sent_payload_bytes + 8 * native_migration_.remote_sent_packets +
                   16 * native_migration_.remote_sent_peers;
        }
        [[nodiscard]] std::uint64_t collective_identity() const noexcept { return transaction_; }
        [[nodiscard]] const std::shared_ptr<const DistributedAdaptiveHierarchy>& candidate() const noexcept
        {
            return candidate_;
        }

    private:
        friend class DistributedAdaptiveHierarchy;
        std::shared_ptr<const void> source_;
        HierarchyObservation observation_;
        NativeMigrationDiagnostics native_migration_;
        std::uint64_t topology_ = 0, partition_ = 0, transaction_ = 0;
        std::shared_ptr<const DistributedAdaptiveHierarchy> candidate_;
    };
    /** Collective, but local marks need not agree across ranks. A family request
     * may originate on any owner; its complete source children are fetched from
     * their actual owners before exact inverse. No legacy global adapt call. */
    [[nodiscard]] Plan plan_adaptation(
        std::span<const AdaptiveMark> refine, std::span<const AdaptiveBytes> coarsen = {}) const;
    [[nodiscard]] std::shared_ptr<const DistributedAdaptiveHierarchy> apply(const Plan&) const;
    [[nodiscard]] std::shared_ptr<const DistributedAdaptiveHierarchy> apply(
        const Plan&, std::shared_ptr<const DistributedAdaptiveHierarchy> bound_candidate) const;
    /** One destination per locally owned leaf in map order; capsules follow
     * their actual anchor owner. Empty ranks participate in every phase. */
    [[nodiscard]] Plan plan_migration(std::span<const int> destinations) const;
    [[nodiscard]] Plan plan_migration(const AdaptiveBalancePlan&) const;
    /** Materialize only transient local support data for canonical authorities. */
    [[nodiscard]] AdaptiveNativeDraft native_draft() const;

private:
    explicit DistributedAdaptiveHierarchy(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;
};
} // namespace SimpleFluid::Meshes::AMR
