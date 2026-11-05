/** @file DistributedAdaptiveIncidence.hh
 * @brief Private incremental canonical support and edge incidence session.
 */
#pragma once
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/mesh/AMR/ContactPointPageView.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct Budget;
class IdentityBudget;
class CartesianLineRangeAdmission;
class OwnedSupportMembership;
class PooledLineCompletionSnapshotAdmission;
struct IncidencePools;
using IncidenceRowHandle = std::shared_ptr<const ContactIncidenceRow>;
using OwnedLeafLookup = std::function<const AdaptiveLeaf&(const AdaptiveCellKey&)>;

struct IncidenceCellVersion
{
    AdaptiveCellKey key;
    std::uint64_t leaf_revision = 0;
    std::uint64_t incidence_generation = 0;
};
struct IncidenceDelta
{
    std::vector<IncidenceCellVersion> retired;
    std::vector<IncidenceRowHandle> changed;
};

/** All methods are collective. The arena and identity ledger outlive this
 * session; accepted and candidate sessions use separately admitted arenas.
 * A lookup is borrowed for the duration of the collective call only. */
class SIMPLEFLUID_LOCAL DistributedAdaptiveIncidenceSession
{
public:
    DistributedAdaptiveIncidenceSession(const AdaptiveProviderKernel&, Teuchos::RCP<const Teuchos::Comm<int>>,
        AdaptiveClosureOptions, ContactHaloContext, Budget&, IdentityBudget&, std::shared_ptr<Budget> = {});
    ~DistributedAdaptiveIncidenceSession();
    DistributedAdaptiveIncidenceSession(DistributedAdaptiveIncidenceSession&&) noexcept;
    DistributedAdaptiveIncidenceSession& operator=(DistributedAdaptiveIncidenceSession&&) noexcept;
    DistributedAdaptiveIncidenceSession(const DistributedAdaptiveIncidenceSession&) = delete;
    DistributedAdaptiveIncidenceSession& operator=(const DistributedAdaptiveIncidenceSession&) = delete;

    /** Construct an isolated candidate, charging shared bodies in its own
     * arena. Mutation copies only dirty buckets and never changes this state. */
    [[nodiscard]] std::unique_ptr<DistributedAdaptiveIncidenceSession> fork(
        Budget&, IdentityBudget&, std::shared_ptr<Budget> = {}) const;
    [[nodiscard]] std::vector<IncidenceRowHandle> bootstrap(std::span<const AdaptiveLeaf>, const OwnedLeafLookup&);
    [[nodiscard]] IncidenceDelta apply_delta(std::span<const IncidenceCellVersion> retired,
        std::span<const AdaptiveCellKey> inserted, ContactHaloContext next, const OwnedLeafLookup&);
    [[nodiscard]] const ContactHaloStatistics& statistics() const noexcept;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    [[nodiscard]] bool owns_point_page_credit(const void*) const noexcept;
    [[nodiscard]] bool admits_source_metadata_owner(const void*, std::size_t) const noexcept;
    void for_each_borrowed_source_admission(const std::function<void(const void*, std::size_t)>&) const;
    void for_each_borrowed_overlap_credit(const std::function<void(const void*, std::size_t, bool)>&) const;
    [[nodiscard]] std::size_t source_arena_overlap_metadata_bytes() const noexcept;
    [[nodiscard]] const IncidencePointDirectory<>& point_pool() const noexcept;
    void for_each_borrowed_payload(
        const std::function<void(const ContactPayloadAllocation&)>&, bool include_object = false) const;
    void for_each_borrowed_geometry_ledger(const std::function<void(const PLIC::GeometryAllocationLedger&)>&) const;
    void for_each_borrowed_plane_identity(
        const std::function<void(const PLIC::PlaneSupportIdentity<ArithmeticMode::Ordinary>&)>&) const;

private:
    struct Impl;
    static PooledLineCompletionSnapshotAdmission source_result_admission(
        std::shared_ptr<const ContactSourcePointSnapshot>, const IncidencePools&);
    static ContactSourcePointAdmission source_point_admission(AMRGrid, ContactHaloContext, int, std::uint64_t, unsigned,
        std::size_t, std::shared_ptr<const CartesianIncidenceSource>, std::shared_ptr<const CartesianSourceManifestSet>,
        std::shared_ptr<const ContactSourcePointSnapshot>);
    static CartesianSourceManifestAdmission source_manifest_admission(AMRGrid, ContactHaloContext, std::uint64_t,
        std::size_t, std::shared_ptr<const CartesianSourceInventory>, const OwnedSupportMembership*);
    static CartesianLineRangeAdmission source_line_admission(std::shared_ptr<const CartesianSourceManifestSet>,
        std::shared_ptr<const ContactSourcePointSnapshot>, ContactHaloContext, std::uint64_t, unsigned);
    explicit DistributedAdaptiveIncidenceSession(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
