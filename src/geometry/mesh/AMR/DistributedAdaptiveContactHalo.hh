/** @file DistributedAdaptiveContactHalo.hh
 * @brief Private exact-incidence spatial closure session.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/mesh/AMR/ContactPointArray.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct ContactIncidenceRow
{
    AdaptiveCellKey key;
    std::uint64_t revision = 0;
    AdaptiveBytes domain;
    ContactPointArray points;
    std::uint64_t incidence_generation = 0;
};
struct ContactHaloContext
{
    std::uint64_t epoch = 0, topology = 0, partition = 0, round = 0, transaction = 0;
    auto operator<=>(const ContactHaloContext&) const = default;
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
    std::size_t dirty_supports = 0, dirty_support_members = 0, dirty_lines = 0, dirty_subscriptions = 0;
    std::size_t recomputed_tiles = 0;
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
    void bootstrap_canonical(std::span<const AdaptiveLeaf>, const ContactOwnedLeafLookup&);
    void apply_delta(std::span<const ContactRetiredLeaf>, std::span<const AdaptiveCellKey>, ContactHaloContext,
        const ContactOwnedLeafLookup&);
    [[nodiscard]] Budget& allocation() noexcept;
    void bind_external_retained_bytes(std::size_t) noexcept;
    [[nodiscard]] std::size_t cache_storage_bytes() const noexcept;
    [[nodiscard]] std::vector<AdaptiveMark> required_refinements();
    [[nodiscard]] ContactMarkLease lease_required_refinements();
    void for_each_incidence_row(const std::function<void(const ContactIncidenceRow&)>&) const;
    [[nodiscard]] const ContactHaloStatistics& statistics() const noexcept;

private:
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
