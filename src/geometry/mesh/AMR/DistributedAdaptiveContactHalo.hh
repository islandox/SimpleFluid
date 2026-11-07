/** @file DistributedAdaptiveContactHalo.hh
 * @brief Private exact-incidence spatial closure session.
 */
#pragma once
#include "utils/export_helper.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/mesh/AMR/AdaptiveContactQuery.hh"
#include "geometry/mesh/AMR/ContactHaloContext.hh"
#include "geometry/mesh/AMR/ContactPayloadCensus.hh"
#include "geometry/mesh/AMR/ContactPointArray.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes::AMR
{
class AdaptiveLeafLayout;
class AdaptiveLeafTransition;
class DistributedAdaptiveHierarchy;
}

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class DistributedAdaptiveContactHalo;
/** Borrow the accepted canonical cache for internal diagnostics. The caller
 * must retain the hierarchy throughout use of the returned reference. */
[[nodiscard]] const DistributedAdaptiveContactHalo& contact_cache(const DistributedAdaptiveHierarchy&);
struct ContactIncidenceRow
{
    AdaptiveCellKey key;
    std::uint64_t revision = 0;
    AdaptiveBytes domain;
    ContactPointArray points;
    std::uint64_t incidence_generation = 0;
};
struct Budget;
/** Owns a result whose allocation remains charged to the originating halo.
 * Destroy the lease before destroying that halo or its allocation arena. */
class SIMPLEFLUID_LOCAL ContactMarkLease
{
public:
    ContactMarkLease(std::vector<AdaptiveMark>, Budget&, std::size_t charged) noexcept;
    ~ContactMarkLease();
    ContactMarkLease(ContactMarkLease&&) noexcept;
    ContactMarkLease& operator=(ContactMarkLease&&) noexcept;
    ContactMarkLease(const ContactMarkLease&) = delete;
    ContactMarkLease& operator=(const ContactMarkLease&) = delete;
    [[nodiscard]] std::span<const AdaptiveMark> marks() const noexcept { return marks_; }

private:
    void reset() noexcept;
    std::vector<AdaptiveMark> marks_;
    Budget* budget_ = nullptr;
    std::size_t charged_ = 0;
};
/** Additive external residency tied only to the allocation arena. It never
 * retains a halo, provider, hierarchy, partition, or mesh. */
class SIMPLEFLUID_LOCAL ContactExternalResidencyLease
{
public:
    ContactExternalResidencyLease() = default;
    ContactExternalResidencyLease(std::shared_ptr<Budget>, std::shared_ptr<std::size_t>);
    ~ContactExternalResidencyLease();
    ContactExternalResidencyLease(ContactExternalResidencyLease&&) noexcept;
    ContactExternalResidencyLease& operator=(ContactExternalResidencyLease&&) noexcept;
    ContactExternalResidencyLease(const ContactExternalResidencyLease&) = delete;
    ContactExternalResidencyLease& operator=(const ContactExternalResidencyLease&) = delete;
    /** Admission preserves old/new overlap until the caller publishes then
     * shrinks to the new resident union. Cumulative allocation stays monotonic. */
    void resize(std::size_t);
    /** Transfer a charge already present in this arena from caller scratch. */
    void adopt_precharged(std::size_t);
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::size_t available() const noexcept
    {
        return arena_ && arena_->used <= arena_->limit ? arena_->limit - arena_->used : 0;
    }

private:
    std::shared_ptr<Budget> arena_;
    std::shared_ptr<std::size_t> total_;
    std::size_t bytes_ = 0;
};
struct ContactRetiredLeaf
{
    AdaptiveCellKey key;
    std::uint64_t revision = 0;
};
using ContactOwnedLeafLookup = std::function<const AdaptiveLeaf&(const AdaptiveCellKey&)>;
enum class ContactHaloMessageKind : std::uint64_t
{
    Cover,
    DirectoryQuery,
    LeafCovers,
    Points,
    Fan,
    Records,
    RecordData,
    DirtyPoints
};
using ContactHaloBeforeSend =
    std::function<void(const ContactHaloContext&, ContactHaloMessageKind, std::vector<OwnerPacket>&)>;
struct ContactHaloStatistics
{
    std::size_t owned_records = 0, remote_records = 0, exact_point_queries = 0, empty_point_replies = 0;
    std::size_t matched_occurrences = 0, logical_contact_pairs = 0, unique_cell_pairs = 0;
    std::size_t span_pair_evaluations = 0, trace_pair_evaluations = 0, query_bytes = 0, record_bytes = 0;
    std::size_t record_fetches = 0, decoded_record_replies = 0, point_query_pages = 0, record_pages = 0;
    std::size_t pages = 0, work = 0, retained_bytes = 0, peak_bytes = 0, cumulative_bytes = 0;
    std::size_t bootstrap_owned_visits = 0, delta_owned_visits = 0, full_owned_scans_after_bootstrap = 0;
    std::size_t contact_record_pages_copied = 0, contact_row_pages_copied = 0, contact_work_pages_copied = 0;
    std::size_t contact_index_pages_copied = 0, contact_index_split_records = 0;
    std::size_t contact_cover_pages_copied = 0, contact_cover_page_bytes = 0;
    std::size_t contact_catalog_pages_copied = 0, contact_trace_pages_copied = 0;
    std::size_t contact_owner_admission_work = 0, contact_shared_owner_bytes = 0, contact_owner_index_bytes = 0;
    std::size_t contact_query_pages_copied = 0, contact_identity_rebinds = 0, contact_frozen_groups = 0;
    std::size_t dirty_supports = 0, dirty_support_members = 0, dirty_lines = 0, dirty_subscriptions = 0;
    std::size_t recomputed_tiles = 0;
    std::size_t point_groups = 0, group_private_metadata_bytes = 0, group_shared_body_bytes = 0;
    std::size_t reused_point_credit_metadata_bytes = 0, reused_point_body_bytes = 0;
    /** Effective source path observations; no timing or physical qualification. */
    std::size_t source_owner_manifests = 0, source_membership_cells = 0;
    std::size_t source_coordinate_points = 0, explicit_coordinate_points = 0;
    std::size_t source_tile_records = 0, explicit_tile_ring_slots = 0;
    std::size_t source_line_records = 0, source_line_nonforward_records = 0, source_line_families = 0;
    std::size_t source_axis_maps = 0, source_result_selectors = 0;
    std::size_t certified_line_witnesses = 0;
    std::size_t explicit_line_records = 0, stride_line_records = 0;
    std::size_t retained_explicit_line_records = 0, retained_stride_line_records = 0, retained_source_line_records = 0;
    std::size_t source_point_recaptures = 0, source_point_recaptured_records = 0;
    std::size_t source_point_recapture_admitted_bytes = 0;
    std::size_t active_saved_layers = 0, retained_saved_layers = 0;
    std::size_t retained_explicit_layer_input_slots = 0, retained_explicit_layer_output_slots = 0;
    std::size_t retained_explicit_layer_edge_result_slots = 0;
    std::size_t retained_certified_layer_input_slots = 0, retained_certified_layer_output_slots = 0;
    std::size_t retained_certified_layer_edge_result_slots = 0, retained_certified_layer_descriptor_bytes = 0;
    std::size_t retained_layer_endpoint_identity_slots = 0;
    std::size_t cached_completion_requests = 0, cached_completion_remote_requests = 0;
    std::size_t cached_completion_received = 0, retained_completion_owners = 0;
    std::size_t retained_expanded_completion_result_slots = 0;
};
struct ContactFanView;
class AdaptiveSubfaceSnapshot;
class AdaptiveContactSnapshot;
struct ContactHaloBorrowedGroup
{
    const void* record;
    std::size_t slot;
    std::uint64_t generation;
    const AdaptiveBytes* domain;
    ContactPointIdentity normalized;
    std::size_t normalized_slot;
    std::uint64_t normalized_generation;
    const ContactFanView* fan;
    std::size_t cover;
    bool live, retired;
};
struct ContactHaloBorrowedFact
{
    const void* record;
    std::size_t slot;
    std::uint64_t generation;
    std::size_t pins;
    bool retired;
    const void* physical_page;
    std::uint64_t captured;
    ContactPointIdentity value;
    const void* source_snapshot = nullptr;
    std::uint64_t source_captured = 0;
};
struct ContactHaloBorrowedBinding
{
    const void* record;
    std::size_t slot;
    std::uint64_t generation;
    std::size_t pins;
    bool retired;
    const ContactIncidenceRow* row;
    std::uint64_t incidence_generation;
    std::size_t record_id;
};
/** A session owns complete incidence and only exact-match remote records. Its
 * deterministic rank-cover tree contains bounds, never a global leaf catalogue.
 * All methods involving communication are collective, including empty owners.
 * Canonical source leaves are borrowed only during bootstrap. */
class SIMPLEFLUID_LOCAL DistributedAdaptiveContactHalo
{
public:
    DistributedAdaptiveContactHalo(const AdaptiveProviderKernel&, Teuchos::RCP<const Teuchos::Comm<int>>,
        AdaptiveClosureOptions, ContactHaloContext = {}, ContactHaloBeforeSend = {});
    ~DistributedAdaptiveContactHalo();
    DistributedAdaptiveContactHalo(DistributedAdaptiveContactHalo&&) noexcept;
    DistributedAdaptiveContactHalo& operator=(DistributedAdaptiveContactHalo&&) noexcept;
    DistributedAdaptiveContactHalo(const DistributedAdaptiveContactHalo&) = delete;
    DistributedAdaptiveContactHalo& operator=(const DistributedAdaptiveContactHalo&) = delete;
    /** caller_live_bytes is admitted before the fork allocates; the caller
     * releases that charge through allocation() after its owners die. */
    [[nodiscard]] std::unique_ptr<DistributedAdaptiveContactHalo> fork_candidate(
        ContactHaloContext, std::size_t caller_live_bytes = 0) const;
    void bootstrap(std::span<const AdaptiveLeaf>, std::vector<ContactIncidenceRow>);
    void bootstrap_canonical(std::span<const AdaptiveLeaf>, const ContactOwnedLeafLookup&,
        std::shared_ptr<const AdaptiveLeafLayout> = {});
    void apply_delta(std::span<const ContactRetiredLeaf>, std::span<const AdaptiveCellKey>, ContactHaloContext,
        const ContactOwnedLeafLookup&);
    [[nodiscard]] Budget& allocation() noexcept;
    [[nodiscard]] ContactExternalResidencyLease external_residency_lease() const;
    void bind_external_retained_bytes(std::size_t) noexcept;
    [[nodiscard]] std::size_t cache_storage_bytes() const noexcept;
    [[nodiscard]] std::vector<AdaptiveMark> required_refinements();
    [[nodiscard]] ContactMarkLease lease_required_refinements();
    /** Visit the complete canonical contact witnesses of one owned leaf.
     * Synchronous and local. The callback must keep the halo quiescent. */
    void visit_closed_contacts(const AdaptiveCellKey&, const ContactWitnessVisitor&) const;
    [[nodiscard]] const std::shared_ptr<const AdaptiveSubfaceSnapshot>& subface_snapshot() const;
    /** Freeze after closure with the pinned layout or a hierarchy-issued
     * transition that supplies all replacement leaf pages. */
    [[nodiscard]] std::shared_ptr<const AdaptiveContactSnapshot> seal_contacts(
        std::shared_ptr<const AdaptiveLeafLayout>, std::span<const AdaptiveLeaf* const> changed_leaf_pages = {});
    [[nodiscard]] std::shared_ptr<const AdaptiveContactSnapshot> seal_contacts(const AdaptiveLeafTransition&);
    void for_each_incidence_row(const std::function<void(const ContactIncidenceRow&)>&) const;
    /** Synchronous borrowed census. The caller keeps this controller alive and
     * prevents mutation; callbacks must not retain references or copy owners. */
    void for_each_borrowed_contact_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;
    /** PLIC geometry/identity pins are inspected separately from known C++
     * contact payload. Caller holds this halo and all borrowed roots quiescent. */
    void for_each_borrowed_geometry_ledger(const std::function<void(const PLIC::GeometryAllocationLedger&)>&) const;
    void for_each_borrowed_plane_identity(
        const std::function<void(const PLIC::PlaneSupportIdentity<ArithmeticMode::Ordinary>&)>&) const;
    void for_each_borrowed_contact_face(
        const std::function<void(const PLIC::PlanarFace<ArithmeticMode::Ordinary>&)>&) const;
    void for_each_borrowed_source_admission(const std::function<void(const void*, std::size_t)>&) const;
    void for_each_borrowed_overlap_credit(const std::function<void(const void*, std::size_t, bool)>&) const;
    [[nodiscard]] std::size_t source_arena_overlap_metadata_bytes() const noexcept;
    [[nodiscard]] std::size_t current_fork_overlap_bytes() const noexcept;
    void for_each_borrowed_contact_group(const std::function<void(const ContactHaloBorrowedGroup&)>&) const;
    void for_each_borrowed_contact_fact(const std::function<void(const ContactHaloBorrowedFact&)>&) const;
    void for_each_borrowed_contact_binding(const std::function<void(const ContactHaloBorrowedBinding&)>&) const;
    [[nodiscard]] const ContactHaloStatistics& statistics() const noexcept;

private:
    [[nodiscard]] std::shared_ptr<const AdaptiveContactSnapshot> seal_contacts_impl(
        const std::shared_ptr<const AdaptiveLeafLayout>&, const AdaptiveLeafTransition*);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
/** Canonical support/edge authorities are used once to establish complete
 * physical incidence, including legacy rounded edge-index admissions. Tile
 * owners are retired before the owning normalized rows are constructed. */
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<ContactIncidenceRow> bootstrap_adaptive_contact_rows(
    std::span<const AdaptiveLeaf>, const AdaptiveProviderKernel&, Teuchos::RCP<const Teuchos::Comm<int>>,
    AdaptiveClosureOptions = {});
/** Explicit test oracle. Never selected by the spatial production route. */
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<AdaptiveMark> legacy_required_adaptive_refinements(
    std::span<const AdaptiveLeaf>, const AdaptiveProviderKernel&, Teuchos::RCP<const Teuchos::Comm<int>>,
    AdaptiveClosureOptions = {});
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
