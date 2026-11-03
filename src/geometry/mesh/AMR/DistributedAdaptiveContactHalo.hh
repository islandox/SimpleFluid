/** @file DistributedAdaptiveContactHalo.hh
 * @brief Private exact-incidence spatial closure session.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <functional>
#include <memory>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
using ContactPointIdentity = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
struct ContactPointIncidence
{
    ContactPointIdentity normalized, physical;
};
struct ContactIncidenceRow
{
    AdaptiveCellKey key;
    std::uint64_t revision = 0;
    AdaptiveBytes domain;
    std::vector<ContactPointIncidence> points;
};
struct ContactHaloContext
{
    std::uint64_t epoch = 0, topology = 0, partition = 0, round = 0;
    auto operator<=>(const ContactHaloContext&) const = default;
};
enum class ContactHaloMessageKind : std::uint64_t
{
    Cover, DirectoryQuery, LeafCovers, Points, Fan, Records, RecordData
};
using ContactHaloBeforeSend = std::function<void(
    const ContactHaloContext&, ContactHaloMessageKind, std::vector<OwnerPacket>&)>;
struct ContactHaloStatistics
{
    std::size_t owned_records = 0, remote_records = 0, exact_point_queries = 0, empty_point_replies = 0;
    std::size_t matched_occurrences = 0, logical_contact_pairs = 0, unique_cell_pairs = 0;
    std::size_t span_pair_evaluations = 0, trace_pair_evaluations = 0, query_bytes = 0, record_bytes = 0;
    std::size_t record_fetches = 0, decoded_record_replies = 0, point_query_pages = 0, record_pages = 0;
    std::size_t pages = 0, work = 0, retained_bytes = 0, peak_bytes = 0, cumulative_bytes = 0;
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
    void bootstrap(std::span<const AdaptiveLeaf>, std::vector<ContactIncidenceRow>);
    [[nodiscard]] std::vector<AdaptiveMark> required_refinements();
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
