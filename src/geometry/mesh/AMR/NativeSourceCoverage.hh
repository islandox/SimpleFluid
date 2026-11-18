/** Closed lazy source membership and detached physical-owner subset coverage. */
#pragma once
#include "geometry/mesh/AMR/AdaptiveLeafBodyUsage.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/PersistentOrderedPages.hh"
#include "geometry/PLIC/details/NativePaidExtent.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class NativeSourceOwnerStore;
class NativeSourceCoverage;
class NativeSourceCoverageTransaction;
struct NativeSourceCoveragePool;
/** One actual canonical physical owner. It observes issuer and owner weakly;
 * its admitted lease contains only this owner's bytes, never a whole source
 * floor, membership index, layout or controller. */
class SIMPLEFLUID_LOCAL NativeSourceOwnerCoverage
{
public:
    ~NativeSourceOwnerCoverage();
    [[nodiscard]] bool matches(std::uint8_t, const std::shared_ptr<const void>&, std::size_t) const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] const ContactExternalResidencyLease& lease() const noexcept { return lease_; }
    [[nodiscard]] const auto& allocation() const noexcept { return allocation_; }
private:
    friend class NativeSourceCoverageTransaction;
    friend class NativeSourceOwnerStore;
    ContactExternalResidencyLease lease_;
    std::weak_ptr<NativeSourceCoveragePool> issuer_;
    std::weak_ptr<const void> owner_;
    PLIC::details::NativeWeakOwnerObservation owner_fixed_;
    const void* address_ = nullptr;
    std::size_t bytes_ = 0;
    std::uint8_t kind_ = 0;
    bool published_ = false, prepared_ = false;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> issuer_allocation_;
};
/** A native wrapper can hold this object before its shared geometry fields.
 * Releasing the wrapper releases only its actual selected owners' coverage. */
class SIMPLEFLUID_LOCAL NativeSourceSubsetCoverage
{
public:
    [[nodiscard]] std::span<const std::shared_ptr<const NativeSourceOwnerCoverage>> owners() const noexcept
    { return owners_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] const auto& allocation() const noexcept { return allocation_; }
    [[nodiscard]] const auto& metadata() const noexcept { return metadata_; }
private:
    friend class NativeSourceCoverageTransaction;
    friend class NativeSourceOwnerStore;
    geometry_detail::NativeConstructorStorageOwner metadata_;
    std::vector<std::shared_ptr<const NativeSourceOwnerCoverage>> owners_;
    std::size_t bytes_ = 0;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
};
/** Current SourceStrong-only membership. Its immutable body tree and role are
 * reused across a new arena without materializing per-owner origin rows. */
class SIMPLEFLUID_LOCAL NativeSourceCoverage
{
public:
    [[nodiscard]] bool contains(std::uint8_t, const std::shared_ptr<const void>&, std::size_t) const;
    [[nodiscard]] bool same_membership(const NativeSourceCoverage&) const noexcept;
    void visit_delta(const NativeSourceCoverage*, PLIC::details::NativePaidExtentGroup::MemberSink) const;
    [[nodiscard]] std::size_t covered_bytes() const noexcept { return covered_; }
    [[nodiscard]] const auto& floor() const noexcept { return floor_; }
    [[nodiscard]] const auto& allocation() const noexcept { return allocation_; }
    void require_current() const;
private:
    friend class NativeSourceOwnerStore;
    friend class NativeSourceCoverageTransaction;
    AdaptiveLeafBodyUsage body_;
    PLIC::details::OwnerManifestPair members_;
    std::array<std::shared_ptr<const NativeSourceCoverage>, 8> parts_{};
    std::size_t part_count_ = 0;
    bool manifest_membership_ = false;
    AdaptiveLeafBodyRole role_ = AdaptiveLeafBodyRole::Leaf;
    std::shared_ptr<ContactExternalResidencyLease> floor_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> floor_allocation_;
    std::shared_ptr<NativeSourceCoveragePool> pool_;
    std::size_t covered_ = 0;
    std::uint64_t revision_ = 0;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
};
/** Private source cache index, separate from SourceStrong membership. It holds
 * weak subset receipts and never enters any persistent native output. */
class SIMPLEFLUID_LOCAL NativeSourceCoverageBroker
{
    struct Row
    {
        std::uint8_t kind;
        const void* address;
        std::weak_ptr<const NativeSourceOwnerCoverage> coverage;
        PLIC::details::NativeWeakOwnerObservation coverage_fixed;
    };
    struct Less
    {
        bool operator()(const Row& a, const Row& b) const noexcept
        { return a.kind != b.kind ? a.kind < b.kind : std::less<const void*>{}(a.address, b.address); }
    };
    using Pages = persistent_ordered_detail::OrderedPages<Row, Less>;
public:
    [[nodiscard]] std::size_t storage_bytes() const noexcept { return root_ ? root_->subtree_bytes : 0; }
    void project(PLIC::details::GeometryOwnerGraphEditor&, bool retain,
        PLIC::details::NativeGeometryAccounting*) const;
private:
    friend class NativeSourceCoverageTransaction;
    Pages::Ptr root_;
    bool pending_ = false;
};
/** All subset allocations/index edits and lease checks are staged. Commit only
 * transfers existing same-arena coverage and persistent root controls. */
class SIMPLEFLUID_LOCAL NativeSourceCoverageTransaction
{
    struct Batch;
public:
    NativeSourceCoverageTransaction(NativeSourceCoverageBroker&, geometry_detail::NativeConstructionAdmission);
    ~NativeSourceCoverageTransaction();
    NativeSourceCoverageTransaction(const NativeSourceCoverageTransaction&) = delete;
    [[nodiscard]] std::shared_ptr<const NativeSourceSubsetCoverage> collect(
        const std::shared_ptr<const NativeSourceCoverage>&,
        std::span<const PLIC::details::geometry_demand::OwnerRecord>,
        const PLIC::details::NativeOwnerReceipts* physical_receipts = nullptr);
    [[nodiscard]] NativeSourceCoverageBroker staged_broker() const;
    void prepare();
    void commit() noexcept;
private:
    NativeSourceCoverageBroker& broker_;
    geometry_detail::NativeConstructionAdmission admission_;
    NativeSourceCoverageBroker::Pages::Ptr staged_;
    geometry_detail::NativeConstructorVector<std::shared_ptr<Batch>> batches_;
    bool prepared_ = false, committed_ = false, failed_ = false;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
