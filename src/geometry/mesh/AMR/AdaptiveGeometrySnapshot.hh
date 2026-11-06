/** @file AdaptiveGeometrySnapshot.hh
 * @brief Immutable mesh-owned geometry roots and authenticated source bindings.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>
namespace SimpleFluid::Meshes::AMR
{
namespace geometry_snapshot_detail
{
struct Builder;
struct PublicationSource;
class Bindings;
struct PreparedCartesianPages;
} // namespace geometry_snapshot_detail
class DistributedAdaptiveHierarchy;
class AdaptiveLeafLayout;
namespace adaptive_detail
{
class GeometrySnapshotAccess;
}
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
    std::shared_ptr<const geometry_snapshot_detail::PreparedCartesianPages> prepared_cartesian_;
    GeometryConstructionControls controls_;
    ArithmeticMode mode_ = ArithmeticMode::Ordinary;
    GeometrySnapshotPurpose purpose_ = GeometrySnapshotPurpose::State;
    bool stable_controls_ = false;
    bool prepare_cartesian_page(const DistributedAdaptiveHierarchy&, const AdaptiveLeaf&, std::size_t& used,
        std::size_t operation_limit, std::size_t optional_headroom);
    void retain_prepared_cartesian_sources(PLIC::GeometryAllocationLedger&, std::size_t& used, std::size_t limit) const;
    /** Internal construction. The result is published only after complete admission. */
    [[nodiscard]] static std::shared_ptr<AdaptiveGeometrySnapshot> build(const DistributedAdaptiveHierarchy&,
        std::span<const AdaptiveLeaf* const>, ArithmeticMode, GeometrySnapshotPurpose, std::size_t& used,
        std::size_t limit, PLIC::GeometryAllocationLedger&, PLIC::GeometryAllocationLedger&,
        std::shared_ptr<const AdaptiveGeometrySnapshot> previous = {},
        std::span<const geometry_snapshot_detail::PublicationSource> publication = {});
};
} // namespace SimpleFluid::Meshes::AMR
