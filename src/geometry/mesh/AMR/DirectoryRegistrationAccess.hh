/** @file DirectoryRegistrationAccess.hh
 * @brief Private hierarchy projection and source-fact admission capabilities.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveFamilyLayout.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"

#include <functional>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Borrowed exact live source overlap. Never stored in a directory. */
class DirectorySourceCredit
{
    friend class ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy;
    friend class DirectoryRegistrationAccess;
    friend class ::SimpleFluid::Meshes::AMR::DistributedAncestryDirectory;
    DirectorySourceCredit(const DistributedAncestryDirectory&, Budget&, std::size_t&);
    const DistributedAncestryDirectory* predecessor_;
    std::shared_ptr<const void> root_;
    Budget* arena_;
    std::size_t* held_;
    std::size_t facts_;
};

/** A current family root or a synchronously validated bootstrap map issues
 * the revision marker. Generic public registrations cannot create this token. */
class FamilyDirectoryProjection
{
    friend class ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy;
    friend class DirectoryRegistrationAccess;
    friend class ::SimpleFluid::Meshes::AMR::DistributedAncestryDirectory;
    using Bootstrap = std::map<AdaptiveBytes, AdaptiveFamily>;
    FamilyDirectoryProjection(std::shared_ptr<const AdaptiveFamilyLayout> root, std::uint64_t topology)
        : root_(std::move(root)), topology_(topology)
    {
    }
    FamilyDirectoryProjection(const Bootstrap& rows, std::uint64_t topology) : bootstrap_(&rows), topology_(topology) {}
    [[nodiscard]] bool marker(const AdaptiveOwner&, int rank) const;
    [[nodiscard]] std::size_t size() const noexcept { return root_ ? root_->size() : bootstrap_->size(); }

    std::shared_ptr<const AdaptiveFamilyLayout> root_;
    const Bootstrap* bootstrap_ = nullptr;
    std::uint64_t topology_;
};

struct DirectoryPersistenceStatistics
{
    std::size_t fact_bytes = 0, numeric_bytes = 0, shared_fact_bytes = 0, overlap_bytes = 0;
    std::size_t created_pages = 0, created_page_bytes = 0, fact_edits = 0;
};

class DirectoryFactPages;
class DirectoryRegistrationAccess
{
public:
    [[nodiscard]] static DistributedAncestryDirectory derive(std::span<const AdaptiveOwner>,
        Teuchos::RCP<const Teuchos::Comm<int>>, std::size_t budget, std::size_t transient, const DirectorySourceCredit*,
        const FamilyDirectoryProjection* = nullptr);
    /** Reconcile the exact shared surviving facts with the live outer lease. */
    static void transfer_shared(
        const DirectorySourceCredit&, const DistributedAncestryDirectory&, Budget&, std::size_t& candidate_held);
    static void finish_overlap(const DistributedAncestryDirectory&) noexcept;
    [[nodiscard]] static DirectoryPersistenceStatistics statistics(const DistributedAncestryDirectory&) noexcept;
    /** Borrow exact facts for diagnostics while the directory is pinned. */
    [[nodiscard]] static const DirectoryFactPages* facts(const DistributedAncestryDirectory&) noexcept;
    [[nodiscard]] static const DistributedAncestryDirectory& cells(const DistributedAdaptiveHierarchy&);
    [[nodiscard]] static const DistributedAncestryDirectory& families(const DistributedAdaptiveHierarchy&);
    [[nodiscard]] static const std::shared_ptr<const DistributedAncestryDirectory>& cell_owner(
        const DistributedAdaptiveHierarchy&);
    static void visit_storage(
        const DistributedAncestryDirectory&, const std::function<void(const void*, std::size_t, bool numeric)>&);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
