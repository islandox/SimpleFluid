/** @file AdaptiveGeometrySnapshot.hh
 * @brief Immutable mesh-owned geometry roots and authenticated source bindings.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/details/GeometryOwnerManifest.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>
namespace SimpleFluid::PLIC::details
{
struct CellCatalogDemandWork;
class GeometryOwnerGraphEditor;
struct GeometryOwnerGraphWork;
} // namespace SimpleFluid::PLIC::details
namespace SimpleFluid::Meshes::AMR
{
namespace adaptive_detail
{
class GeometrySnapshotAccess;
struct Budget;
} // namespace adaptive_detail
namespace geometry_snapshot_detail
{
struct Builder;
struct PublicationSource;
class Bindings;
inline constexpr std::size_t prepared_cartesian_page_size = 64;
/** One immutable prepared State page: exact recipe control, 64-cell block and
 * the actual source-only physical page. */
using PreparedOwnerManifest = PLIC::details::geometry_demand::GeometryOwnerManifest;
struct PreparedCartesianPage
{
    std::shared_ptr<const NativeFragmentDescriptor> recipe;
    std::shared_ptr<const PLIC::CartesianCellSource> source;
    std::uint64_t first = 0, rows = 0;
};
/** Immutable persistent exact recipe/page and source-context page index. The
 * value lives inside its owning snapshot header; only the shared index pages
 * are separate allocations. */
class PreparedCartesianPages
{
public:
    [[nodiscard]] const PreparedCartesianPage* find(
        const std::shared_ptr<const NativeFragmentDescriptor>&, std::uint64_t first) const;
    [[nodiscard]] bool contains_source(const std::shared_ptr<const PLIC::CartesianCellSource>&) const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t insertion_byte_bound() const;
    [[nodiscard]] std::size_t insertion_dependency_edge_bound() const;
    [[nodiscard]] std::size_t insertion_dependency_owner_bound() const;
    [[nodiscard]] static PreparedCartesianPages appended(
        const PreparedCartesianPages*, PreparedCartesianPage, adaptive_detail::Budget&);
    void visit_pages(const std::function<void(const PreparedCartesianPage&)>&) const;
    void visit_dependencies(PLIC::details::GeometryOwnerGraphEditor&, bool retain) const;
    void visit_metadata(const std::function<void(const std::shared_ptr<const void>&, std::size_t)>&) const;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;

private:
    std::shared_ptr<const void> root_;
};
} // namespace geometry_snapshot_detail
class DistributedAdaptiveHierarchy;
class AdaptiveLeafLayout;
enum class GeometrySnapshotPurpose
{
    State,
    OriginalSources
};
/** Named controls only, excluding floating-point exception status and padding. */
struct GeometryConstructionControls
{
    int rounding = -1;
    std::uint64_t x87_control = 0, simd_control = 0;
    bool supported = false;
    bool operator==(const GeometryConstructionControls&) const = default;
};
[[nodiscard]] GeometryConstructionControls geometry_construction_controls();
/** Only join-bearing explicit Polyhedral providers consume supplied surfaces. */
struct GeometrySnapshotSurface
{
    AdaptiveSurface surface;
};
/** Exact recipe and surface comparisons at snapshot admission. */
struct GeometrySnapshotBinding
{
    [[nodiscard]] static bool same_auxiliary_recipes(const AdaptiveLeaf&, const AdaptiveLeaf&);
    [[nodiscard]] static bool same_auxiliary_capsules(const AdaptiveLeaf&, const AdaptiveLeaf&);
};
struct GeometryEvaluationPolicy
{
    bool original_integrals = false, ordinary_integrals = false, bounded_integrals = false;
};
/** No mutable hierarchy, partition, field, or solver is retained by a snapshot. */
class AdaptiveGeometrySnapshot : public std::enable_shared_from_this<AdaptiveGeometrySnapshot>
{
public:
    using OrdinaryCatalog = PLIC::CellGeometryCatalog<ArithmeticMode::Ordinary>;
    using BoundedCatalog = PLIC::CellGeometryCatalog<ArithmeticMode::BoundedCertification>;
    /** Exact owner-local active layout, independent of catalog selection and
     * numerical ordinals. A weak identity does not make geometry own topology. */
    [[nodiscard]] bool matches_layout(const std::shared_ptr<const AdaptiveLeafLayout>& layout) const noexcept
    {
        return layout && source_layout_.lock() == layout && !source_layout_.owner_before(layout) &&
               !layout.owner_before(source_layout_);
    }
    [[nodiscard]] const auto& ordinary_catalog() const noexcept { return ordinary_; }
    [[nodiscard]] const auto& finite_ordinary_catalog() const noexcept { return finite_ordinary_; }
    [[nodiscard]] const auto& bounded_catalog() const noexcept { return bounded_; }
    class NativeMetrics
    {
    public:
        [[nodiscard]] std::size_t size() const noexcept;
        [[nodiscard]] bool empty() const noexcept { return size() == 0; }
        [[nodiscard]] AdaptiveCellMetrics at(std::size_t slot) const;
        [[nodiscard]] AdaptiveCellMetrics operator[](std::size_t slot) const { return at(slot); }

    private:
        friend class AdaptiveGeometrySnapshot;
        explicit NativeMetrics(const AdaptiveGeometrySnapshot& snapshot) : snapshot_(&snapshot) {}
        const AdaptiveGeometrySnapshot* snapshot_;
    };
    [[nodiscard]] NativeMetrics native_metrics() const noexcept { return NativeMetrics(*this); }
    [[nodiscard]] const auto& construction_controls() const noexcept { return controls_; }
    [[nodiscard]] bool stable_controls() const noexcept { return stable_controls_; }
    [[nodiscard]] ArithmeticMode arithmetic_mode() const noexcept { return mode_; }
    [[nodiscard]] GeometrySnapshotPurpose purpose() const noexcept { return purpose_; }
    [[nodiscard]] GeometryEvaluationPolicy evaluation_policy(std::size_t) const;
    [[nodiscard]] std::optional<std::size_t> find(const AdaptiveLeaf&) const;
    /** Snapshot and binding-page bytes. Use retained_bytes() to deduplicate and
     * include descriptors, certificates and catalog payload owners. */
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    /** Borrow actual metadata owners without changing reference counts. */
    void visit_external_owners(const std::function<void(const void*, std::size_t)>&) const;
    void visit_prepared_cartesian_sources(const std::function<void(const PLIC::CartesianCellSource&)>&) const;
    void visit_authoritative_geometries(
        const std::function<void(const PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>&)>&) const;
    void visit_authoritative_surfaces(const std::function<void(const GeometrySnapshotSurface&)>&) const;
    void retain(
        PLIC::GeometryAllocationLedger&, PLIC::GeometryAllocationLedger&, std::size_t& used, std::size_t limit) const;
    [[nodiscard]] std::size_t retained_bytes() const;

private:
    friend class DistributedAdaptiveHierarchy;
    friend class adaptive_detail::GeometrySnapshotAccess;
    friend struct geometry_snapshot_detail::Builder;
    AdaptiveGeometrySnapshot() = default;
    std::weak_ptr<const AdaptiveLeafLayout> source_layout_;
    std::shared_ptr<const OrdinaryCatalog> ordinary_, finite_ordinary_;
    std::shared_ptr<const BoundedCatalog> bounded_;
    std::shared_ptr<const geometry_snapshot_detail::Bindings> bindings_;
    std::optional<geometry_snapshot_detail::PreparedCartesianPages> prepared_cartesian_;
    GeometryConstructionControls controls_;
    ArithmeticMode mode_ = ArithmeticMode::Ordinary;
    GeometrySnapshotPurpose purpose_ = GeometrySnapshotPurpose::State;
    bool stable_controls_ = false;
    /** Payload-only physical graph, excluding header and these index graphs. */
    std::array<geometry_snapshot_detail::PreparedOwnerManifest, 2> payload_owners_;
    std::array<std::size_t, 9> owner_work_{};
    void record_owner_work(const PLIC::details::GeometryOwnerGraphWork&);
    void update_owner_manifests(const AdaptiveGeometrySnapshot*, PLIC::details::GeometryOwnerGraphEditor&,
        PLIC::details::GeometryOwnerGraphEditor&);
    void visit_payload_dependencies(PLIC::details::GeometryOwnerGraphEditor& ordinary,
        PLIC::details::GeometryOwnerGraphEditor& bounded, bool retain) const;
    /** Prepare the 64-cell block of `requested` (or an explicitly named block of
     * the same recipe and region) as an optional immutable State source page. */
    bool prepare_cartesian_page(const DistributedAdaptiveHierarchy&, const AdaptiveLeaf&, std::size_t& used,
        std::size_t operation_limit, std::size_t optional_headroom, std::uint64_t block_first,
        PLIC::GeometryAllocationLedger&);
    void retain_prepared_cartesian_sources(PLIC::GeometryAllocationLedger&, std::size_t& used, std::size_t limit) const;
    /** Physical prepared page slot for a cell, if the page covers it. */
    struct PreparedSourceSlot
    {
        std::shared_ptr<const void> source;
        std::uint64_t row = 0;
        bool valid = false;
    };
    [[nodiscard]] PreparedSourceSlot prepared_slot(const AdaptiveLeaf&) const;
    /** Adopt prepared page sources for already selected cells. */
    void rebind_prepared_slots(std::span<const std::size_t> ordinals, std::span<const PreparedSourceSlot> physical,
        PLIC::GeometryAllocationLedger& ordinary_owners, std::size_t& used, std::size_t limit);
    /** Exact immutable owner set of this snapshot, excluding its own header.
     * `bounded` selects the bounded-certification ledger's owners. */
    [[nodiscard]] geometry_snapshot_detail::PreparedOwnerManifest owner_manifest(
        adaptive_detail::Budget& budget, bool bounded = false) const;
    void owner_manifests(adaptive_detail::Budget&, geometry_snapshot_detail::PreparedOwnerManifest& ordinary,
        geometry_snapshot_detail::PreparedOwnerManifest& bounded, bool payload_prepaid = false) const;
    /** Internal construction. The result is published only after complete admission. */
    [[nodiscard]] static std::shared_ptr<AdaptiveGeometrySnapshot> build(const DistributedAdaptiveHierarchy&,
        std::span<const AdaptiveLeaf* const>, ArithmeticMode, GeometrySnapshotPurpose, std::size_t& used,
        std::size_t limit, PLIC::GeometryAllocationLedger&, PLIC::GeometryAllocationLedger&,
        std::shared_ptr<const AdaptiveGeometrySnapshot> previous = {},
        std::span<const geometry_snapshot_detail::PublicationSource> publication = {},
        PLIC::details::CellCatalogDemandWork* demand_work = nullptr);
};
} // namespace SimpleFluid::Meshes::AMR
