/** Closed current-source ownership and disjoint source accounting. */
#pragma once
#include "geometry/PLIC/details/NativeGeometryAccounting.hh"
#include "geometry/mesh/AMR/AdaptiveLeafPages.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/NativePublicationOwnerGraph.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/AMR/NativeSourceCoverage.hh"
#include <optional>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct NativePublicationIndexes;
class NativePublicationReceipt;
struct NativeSourceBoundInput
{
    using Map = Teuchos::RCP<const DefaultTpetraTypes::map_type>;
    std::shared_ptr<const CompositePartition<>> partition;
    std::shared_ptr<const Map> map_token;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> map_token_allocation;
};
struct NativeSourcePublicationValues
{
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> cache_allocation;
    std::shared_ptr<const NativePublicationReceipt> receipt;
    NativeSourceBoundInput input;
    NativePublicationSourceObservations source;
    NativeSourceCoverageBroker broker;
};
class NativeSourceCoverage;
class NativeSourceSubsetCoverage;
class AdaptiveLeafBodyUsage;
enum class AdaptiveLeafBodyRole;
class ContactExternalResidencyLease;
class AdaptiveSubfaceSnapshot;
class DistributedAdaptiveContactHalo;
using NativeContactSideIndexes = std::array<PLIC::details::geometry_demand::GeometryOwnerManifest, 4>;
struct NativeContactSourceCoverage
{
    PLIC::details::OwnerManifestPair owners;
    PLIC::details::NativeOwnerReceipts paid;
    NativeContactSideIndexes side_indexes;
    std::array<std::shared_ptr<const NativeSourceCoverage>, 2> coverage;
    std::optional<std::size_t> projected_admission_bytes;
};
struct SourceAccountingPartition
{
    std::size_t exclusive_source_admission = 0, exclusive_contact_admission = 0;
    std::size_t projected_source_admission = 0, projected_contact_admission = 0;
    [[nodiscard]] std::size_t residual_bytes() const;
    /** The caller names the exact projected component, never deduces an
     * overlap from equal totals or naked allocation addresses. */
    static SourceAccountingPartition split(std::size_t source, std::size_t projected_source,
        std::size_t contacts, std::size_t projected_contacts);
};
/** This store owns only its current detached physical manifest. No controller,
 * predecessor store, global union or source layout is retained by its root. */
class SIMPLEFLUID_LOCAL NativeSourceOwnerStore
{
    struct Root
    {
        PLIC::details::OwnerManifestPair owners;
        PLIC::details::NativeOwnerReceipts paid;
        std::shared_ptr<const NativeSourceCoverage> coverage;
        std::shared_ptr<const AdaptiveSubfaceSnapshot> subfaces;
        NativeContactSideIndexes contact_side_indexes;
        PLIC::details::NativePaidExtentGroup group;
        std::array<PLIC::details::OwnerManifestPair, 4> components;
        std::array<PLIC::details::NativeOwnerReceipts, 4> component_paid;
        std::optional<std::size_t> contact_projected_admission;
        std::shared_ptr<const void> contact_arena;
        std::shared_ptr<const NativePublicationIndexes> indexes;
        NativeSourcePublicationValues publication_values;
        std::array<PLIC::details::GeometryNativeAdministration, 2> input_administration;
        std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
    };
public:
    [[nodiscard]] const PLIC::details::GeometryOwnerRootBinding& binding() const noexcept { return binding_; }
    [[nodiscard]] const PLIC::details::OwnerManifestPair& owners() const noexcept;
    [[nodiscard]] const std::shared_ptr<const NativePublicationIndexes>& indexes() const noexcept;
    [[nodiscard]] const NativeSourceBoundInput& bound_input() const noexcept;
    [[nodiscard]] const NativePublicationSourceObservations& observations() const noexcept;
    [[nodiscard]] const std::array<PLIC::details::GeometryNativeAdministration, 2>& input_administration() const noexcept;
    [[nodiscard]] const std::shared_ptr<const NativeSourceCoverage>& coverage() const noexcept;
    [[nodiscard]] const std::shared_ptr<const AdaptiveSubfaceSnapshot>& subfaces() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return !root_; }
    void require_current() const;
    [[nodiscard]] const std::array<PLIC::details::OwnerManifestPair, 4>& components() const noexcept;
    [[nodiscard]] const std::array<PLIC::details::NativeOwnerReceipts, 4>& component_paid() const noexcept;
    /** Exact physical contact roots, subface control and arena authenticate the
     * legacy financial component removed from the caller's fork prefix. */
    [[nodiscard]] std::size_t contact_credit(const DistributedAdaptiveContactHalo&) const;
    [[nodiscard]] static PLIC::details::OwnerManifestPair single(
        PLIC::details::geometry_demand::GeometryOwnerManifest);
    /** Components are certified maintained body/contact/kernel/index graphs.
     * Shared B+ controls prune owner set deltas; no warm census is performed. */
    [[nodiscard]] static NativeSourceOwnerStore assemble(const NativeSourceOwnerStore& previous,
        std::array<PLIC::details::OwnerManifestPair, 4>, const geometry_detail::NativeConstructionAdmission&,
        PLIC::details::NativeGeometryAccounting* = nullptr,
        std::shared_ptr<const NativePublicationIndexes> indexes = {},
        std::array<PLIC::details::GeometryNativeAdministration, 2> input_administration = {},
        std::shared_ptr<const NativeSourceCoverage> coverage = {},
        std::shared_ptr<const AdaptiveSubfaceSnapshot> subfaces = {},
        std::optional<std::array<PLIC::details::NativeOwnerReceipts, 4>> component_paid = {},
        std::optional<std::size_t> contact_projected_admission = {}, std::shared_ptr<const void> contact_arena = {},
        std::optional<NativeSourcePublicationValues> publication_values = {},
        std::optional<NativeContactSideIndexes> contact_side_indexes = {});
    [[nodiscard]] static std::shared_ptr<const NativeSourceCoverage> body_coverage(const AdaptiveLeafBodyUsage&,
        AdaptiveLeafBodyRole, std::shared_ptr<const ContactExternalResidencyLease>,
        const geometry_detail::NativeConstructionAdmission&, std::uint64_t revision,
        std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> floor_allocation = {});
    [[nodiscard]] static std::shared_ptr<const NativeSourceCoverage> manifest_coverage(
        PLIC::details::OwnerManifestPair, std::shared_ptr<const ContactExternalResidencyLease>,
        const geometry_detail::NativeConstructionAdmission&, std::uint64_t revision,
        std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> floor_allocation = {});
    [[nodiscard]] static std::shared_ptr<const NativeSourceCoverage> composite_coverage(
        const std::shared_ptr<const NativeSourceCoverage>& previous,
        std::span<const std::shared_ptr<const NativeSourceCoverage>>,
        const geometry_detail::NativeConstructionAdmission&, std::uint64_t revision);
    [[nodiscard]] static PLIC::details::NativePaidExtentGroup coverage_group(
        const std::shared_ptr<const NativeSourceCoverage>&);
    static void project_coverage(const std::shared_ptr<const NativeSourceCoverage>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain, PLIC::details::NativeGeometryAccounting* = nullptr);
    static void project_coverage_descriptor(const std::shared_ptr<const NativeSourceCoverage>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain, PLIC::details::NativeGeometryAccounting* = nullptr);
    static void project_subset(const std::shared_ptr<const NativeSourceSubsetCoverage>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain, PLIC::details::NativeGeometryAccounting* = nullptr);
    static void record_subset(const std::shared_ptr<const NativeSourceSubsetCoverage>&,
        PLIC::details::GeometryOwnerGraphEditor&, bool retain = true);
    /** One leaf's actual direct shared edges. Keys/domain/ancestry/row storage
     * remain in the explicitly disjoint exclusive source extent. */
    [[nodiscard]] static geometry_detail::NativeAllocationMirror mirror(const std::shared_ptr<Budget>&);
    [[nodiscard]] static std::size_t publication_mirror_bytes(const std::shared_ptr<Budget>&);
    static void leaf(const AdaptiveLeaf&, PLIC::details::GeometryOwnerGraphEditor&, bool retain,
        bool prepaid = false, PLIC::details::NativeGeometryAccounting* = nullptr);
    /** Bootstrap/default source edits use the existing geometry work domain.
     * The two completed graphs must have been paired before editing. */
    static PLIC::details::OwnerManifestPair seal_default(PLIC::details::GeometryOwnerGraphEditor&,
        PLIC::details::GeometryOwnerGraphEditor&, std::size_t overlap);
    /** Issuance happens privately. The caller stages this binding into the
     * current union and swaps the store only after collective agreement. */
    [[nodiscard]] static NativeSourceOwnerStore staged(PLIC::details::OwnerManifestPair,
        PLIC::details::NativeGeometryAccounting&);
    [[nodiscard]] PLIC::details::GeometryOwnerUnion replace_in(const NativeSourceOwnerStore& accepted,
        const PLIC::details::GeometryOwnerUnion&, PLIC::details::NativeGeometryAccounting&,
        PLIC::details::GeometryOwnerUnionWork* = nullptr) const;
    [[nodiscard]] PLIC::details::GeometryOwnerUnion replace_in(const NativeSourceOwnerStore& accepted,
        const PLIC::details::GeometryOwnerUnion&, PLIC::GeometryAllocationLedger&, PLIC::GeometryAllocationLedger&,
        std::size_t& used, std::size_t limit, PLIC::details::GeometryOwnerUnionWork* = nullptr) const;
    void swap(NativeSourceOwnerStore&) noexcept;
private:
    friend struct ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy::Data;
    friend class NativeSourceCoverageTransaction;
    static void seal_coverage_floor(const std::shared_ptr<NativeSourceCoverage>&,
        const geometry_detail::NativeConstructionAdmission&);
    static void project_coverage_pool(const std::shared_ptr<NativeSourceCoveragePool>&,
        const geometry_detail::NativeConstructorOwnerSink&);
    [[nodiscard]] static PLIC::details::NativeWeakOwnerObservation observe_coverage(
        const std::shared_ptr<const NativeSourceOwnerCoverage>&);
    [[nodiscard]] static geometry_detail::NativeAllocationMirror make_mirror(
        const std::shared_ptr<Budget>&, bool publication);
    [[nodiscard]] static PLIC::details::SealedSourcePrefix prefix_for_refresh(
        const DistributedAdaptiveHierarchy::Data&);
    static bool validate_owner_coverage(const std::shared_ptr<const void>&,
        const std::shared_ptr<const void>& physical, const std::shared_ptr<const void>& arena,
        std::uint8_t kind, std::size_t bytes, std::size_t snapshot, std::uint64_t revision);
    static void project_owner_coverage(const std::shared_ptr<const void>&,
        const geometry_detail::NativeConstructorOwnerSink&);
    static void inspect_owner_coverage(const std::shared_ptr<const void>&,
        const PLIC::details::NativeFinancialInspectionSink&);
    friend class NativePublicationOperation;
    std::shared_ptr<const Root> root_;
    PLIC::details::GeometryOwnerRootBinding binding_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
