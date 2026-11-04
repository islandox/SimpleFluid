/** @file DistributedAdaptiveIncidence.hh
 * @brief Private incremental canonical support and edge incidence session.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/ContactPointPageView.hh"
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct Budget;
class IdentityBudget;
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
    [[nodiscard]] const IdentityPool<ContactPointIdentity>& point_pool() const noexcept;

private:
    struct Impl;
    explicit DistributedAdaptiveIncidenceSession(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
