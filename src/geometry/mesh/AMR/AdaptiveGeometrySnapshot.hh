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
#include <map>
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
} // namespace geometry_snapshot_detail
class DistributedAdaptiveHierarchy;
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
/** A detached binding pins only the immutable recipe and original shell. */
struct GeometrySnapshotBinding
{
    AdaptiveCellKey key;
    AdaptiveBytes domain;
    std::uint64_t material = 0, revision = 0, native_cell = 0;
    AdaptiveProviderKind provider = AdaptiveProviderKind::Polyhedral;
    std::shared_ptr<const NativeFragmentDescriptor> native_recipe;
    std::shared_ptr<const AdaptiveBytes> join_recipe;
    std::shared_ptr<const AdaptiveFactorRecipe> factor_recipe;
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry;
    std::vector<GeometrySnapshotSurface> surfaces;
    [[nodiscard]] bool matches(const AdaptiveLeaf&) const;
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
    [[nodiscard]] const auto& ordinary_catalog() const noexcept { return ordinary_; }
    [[nodiscard]] const auto& finite_ordinary_catalog() const noexcept { return finite_ordinary_; }
    [[nodiscard]] const auto& bounded_catalog() const noexcept { return bounded_; }
    [[nodiscard]] const auto& native_metrics() const noexcept { return native_metrics_; }
    [[nodiscard]] const auto& construction_controls() const noexcept { return controls_; }
    [[nodiscard]] bool stable_controls() const noexcept { return stable_controls_; }
    [[nodiscard]] ArithmeticMode arithmetic_mode() const noexcept { return mode_; }
    [[nodiscard]] GeometrySnapshotPurpose purpose() const noexcept { return purpose_; }
    [[nodiscard]] const GeometryEvaluationPolicy& evaluation_policy(std::size_t slot) const
    {
        return evaluation_policy_.at(slot);
    }
    [[nodiscard]] std::optional<std::size_t> find(const AdaptiveLeaf&) const;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
    /** Borrow actual metadata owners without changing reference counts. */
    void visit_external_owners(const std::function<void(const void*, std::size_t)>&) const;
    void visit_authoritative_surfaces(const std::function<void(const GeometrySnapshotSurface&)>&) const;
    void retain(
        PLIC::GeometryAllocationLedger&, PLIC::GeometryAllocationLedger&, std::size_t& used, std::size_t limit) const;
    [[nodiscard]] std::size_t retained_bytes() const;

private:
    friend class DistributedAdaptiveHierarchy;
    friend class adaptive_detail::GeometrySnapshotAccess;
    friend struct geometry_snapshot_detail::Builder;
    AdaptiveGeometrySnapshot() = default;
    std::shared_ptr<const OrdinaryCatalog> ordinary_, finite_ordinary_;
    std::shared_ptr<const BoundedCatalog> bounded_;
    std::vector<AdaptiveCellMetrics> native_metrics_;
    std::vector<GeometryEvaluationPolicy> evaluation_policy_;
    std::vector<GeometrySnapshotBinding> bindings_;
    std::map<AdaptiveCellKey, std::size_t> binding_slots_;
    GeometryConstructionControls controls_;
    ArithmeticMode mode_ = ArithmeticMode::Ordinary;
    GeometrySnapshotPurpose purpose_ = GeometrySnapshotPurpose::State;
    bool stable_controls_ = false;
    /** Internal construction. The result is published only after complete admission. */
    [[nodiscard]] static std::shared_ptr<const AdaptiveGeometrySnapshot> build(const DistributedAdaptiveHierarchy&,
        std::span<const AdaptiveLeaf* const>, ArithmeticMode, GeometrySnapshotPurpose, std::size_t& used,
        std::size_t limit, PLIC::GeometryAllocationLedger&, PLIC::GeometryAllocationLedger&,
        std::shared_ptr<const AdaptiveGeometrySnapshot> previous = {},
        std::span<const geometry_snapshot_detail::PublicationSource> publication = {});
};
} // namespace SimpleFluid::Meshes::AMR
