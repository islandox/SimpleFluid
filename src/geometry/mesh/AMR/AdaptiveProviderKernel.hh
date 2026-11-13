/** @file AdaptiveProviderKernel.hh
 * @brief Pure local adaptation kernels; no communicator or replicated hierarchy.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include <array>
#include <optional>
namespace SimpleFluid::Meshes::AMR
{
struct AMRGrid;
struct AdaptiveSplitResult
{
    AdaptiveFamily family;
    std::vector<AdaptiveLeaf> children;
};
struct AdaptiveBootstrap
{
    std::vector<AdaptiveLeaf> owned;
    std::vector<AdaptiveSourceCell> source_cells;
    /** Bootstrap-temporary original support prototypes only for needed roots;
     * consumers must not retain a global source hierarchy/catalog. */
    std::vector<AdaptiveLeaf> support_roots;
    std::vector<AdaptiveFamily> families;
    std::shared_ptr<const class AdaptiveProviderKernel> kernel;
};
/** Comparable provider trace support. A face restricts the trace to exact
 * contact points on that support; an empty face applies to the whole leaf. */
struct AdaptiveTraceFace
{
    AdaptiveBytes stem;
    unsigned depth = 0;
    AdaptiveSplit split = AdaptiveSplit::Isotropic;
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> face;
    bool terminal = true; ///< At least one matching trace must be its actual terminal cap.
};
struct AdaptiveTrace
{
    AdaptiveCellKey key;
    std::vector<AdaptiveTraceFace> faces;
};
/** Actual Cartesian provider factory input under the current caller controls.
 * These are raw coordinate expressions, before any PLIC frame/normalization. */
struct AdaptiveCartesianGeometryBounds
{
    PLIC::GeometryPoint lower{}, upper{};
};
struct AdaptiveCellMetrics
{
    double volume = 0;
    PLIC::GeometryPoint centroid{};
};
/** Closed broad-phase ranges in canonical world coordinates. Equality at an
 * endpoint is contact; these ranges must never use an interior-overlap test. */
struct AdaptiveContactBounds
{
    std::array<long double, 3> lower{}, upper{};
};
enum class AdaptiveContactNormalization
{
    Unsupported,
    PhysicalIdentity,
    PeriodicCartesian
};
/** Explicit provider promise, independent of the legacy contact-point API.
 * Union preservation permits retaining ancestor ownership covers after local
 * splits/coarsens; it does not authorize skipping the exact contact predicate. */
struct AdaptiveContactCapabilities
{
    AdaptiveContactNormalization normalization = AdaptiveContactNormalization::Unsupported;
    bool split_union_preserving = false, coarsen_union_preserving = false;
};
struct AdaptiveContactLogicalBox
{
    AdaptiveBytes frame; ///< Complete grid semantics, never a display-coordinate identity.
    std::array<std::uint64_t, 3> lower{}, upper{}, period{};
    bool exact_embedding = false; ///< Integer separation proves physical separation only under this promise.
};
struct AdaptiveContactImage
{
    AdaptiveContactBounds bounds;
    /** Physical-to-contact mapping for this image, including periodic corners. */
    PLIC::DisplacementIdentity<ArithmeticMode::Ordinary> physical_to_contact;
    std::optional<AdaptiveContactLogicalBox> logical;
};
struct AdaptiveProductFamilySelector
{
    AdaptiveCellKey parent;
    AdaptiveSplit split;
    bool operator==(const AdaptiveProductFamilySelector&) const = default;
};

class AdaptiveProviderKernel
{
public:
    virtual ~AdaptiveProviderKernel() = default;
    /** Canonical complete compact kernel semantics for collective agreement. */
    [[nodiscard]] virtual AdaptiveBytes descriptor() const = 0;
    /** Authenticate the supplied native descriptor/ordinal against typed ancestry
     * and actual exact shape before publication or owner migration. */
    virtual void validate(const AdaptiveLeaf&) const;
    [[nodiscard]] virtual AdaptiveCellMetrics metrics(const AdaptiveLeaf&) const;
    [[nodiscard]] virtual PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const AdaptiveLeaf&) const = 0;
    [[nodiscard]] virtual std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf&) const = 0;
    /** Opt-in successful-path capacity envelope for publication materialization.
     * Zero preserves legacy expanded admission. Implementations must include
     * all output owners and simultaneous construction scratch; caller admits
     * this envelope before invoking compact_native_materialize. */
    [[nodiscard]] virtual std::size_t compact_native_materialization_bound(const AdaptiveLeaf&) const { return 0; }
    [[nodiscard]] virtual AdaptiveLeaf compact_native_materialize(const AdaptiveLeaf&) const;
    [[nodiscard]] virtual PLIC::PointIdentity<ArithmeticMode::Ordinary> contact_point(
        const AdaptiveLeaf&, const PLIC::PointIdentity<ArithmeticMode::Ordinary>&) const;
    /** Opt-in spatial-search contract. The default remains unsupported, so an
     * existing custom contact_point override cannot silently use wrong bounds. */
    [[nodiscard]] virtual AdaptiveContactCapabilities contact_capabilities(const AdaptiveLeaf&) const { return {}; }
    /** Complete conservative closed images in contact_point coordinates.
     * Supported identity providers use the canonical actual convex hull. */
    [[nodiscard]] virtual std::vector<AdaptiveContactImage> closed_contact_images(const AdaptiveLeaf&) const;
    [[nodiscard]] virtual std::vector<AdaptiveCellKey> matching_partners(const AdaptiveLeaf&) const;
    [[nodiscard]] virtual AdaptiveTrace traces(const AdaptiveLeaf&) const;
    [[nodiscard]] virtual AdaptiveSplit closure_split(const AdaptiveLeaf&, std::size_t axis) const;
    /** Needed product factor-parent capsules, synthesized independently of
     * historical XY/Z split order; anchor is this actual descendant. */
    [[nodiscard]] virtual std::vector<AdaptiveFamily> ancestry_families(const AdaptiveLeaf&) const { return {}; }
    /** Repack an actual owned selection into shared procedural fragments and
     * prune factor DAGs before migration/publication. Default explicit path. */
    [[nodiscard]] virtual std::vector<AdaptiveLeaf> repack(std::span<const AdaptiveLeaf> leaves) const;
    [[nodiscard]] virtual std::vector<AdaptiveCellKey> refinement_group(const AdaptiveCellKey&, AdaptiveSplit) const;
    [[nodiscard]] virtual std::vector<AdaptiveBytes> coarsening_group(const AdaptiveFamily&) const;
    [[nodiscard]] virtual AdaptiveSplitResult split(const AdaptiveLeaf&, AdaptiveSplit) const = 0;
    /** Explicit opt-in box/recipe contract. The default is unsupported, even
     * for a provider labelled Cartesian. A supporting kernel guarantees its
     * raw bounds match geometry() under current controls and its native metrics
     * retain the Cartesian recipe's binary64 expressions. The capability query
     * performs no allocation or coordinate arithmetic. */
    [[nodiscard]] virtual bool has_native_cartesian_geometry() const noexcept { return false; }
    [[nodiscard]] virtual std::optional<AdaptiveCartesianGeometryBounds> native_cartesian_geometry_bounds(
        const AdaptiveLeaf&) const
    {
        return {};
    }
    /** Explicit opt-in to the exact Cartesian incidence-source contract.
     * Default unsupported; enum tags, names and descriptor parsing are not
     * authorization. Successful providers copy their complete fixed grid. */
    [[nodiscard]] virtual bool native_cartesian_incidence_grid(AMRGrid&) const noexcept { return false; }
    /** Complete immediate source membership is verified by distributed family
     * authority before this exact local inverse is evaluated. */
    [[nodiscard]] AdaptiveLeaf coarsen(const AdaptiveFamily&, std::span<const AdaptiveLeaf>) const;
    /** Opt in when repack preserves every selected leaf's persistent identity
     * and certified shape without the rest of the active selection. */
    [[nodiscard]] virtual bool supports_leaf_local_repack() const noexcept { return false; }
    /** The same exact factor parents as ancestry_families. Built-in Product
     * routes produce selectors without constructing any parent geometry. */
    [[nodiscard]] virtual std::vector<AdaptiveProductFamilySelector> ancestry_family_selectors(
        const AdaptiveLeaf&) const;
    [[nodiscard]] virtual AdaptiveFamily ancestry_family(
        const AdaptiveLeaf&, const AdaptiveProductFamilySelector&) const;
    /** Opt in only when changing the saved parent's revision cannot change
     * its provider validation, geometry, or complete immediate children.
     * A null parent asks whether the promise holds for every Product parent
     * of this actual kernel, without constructing a routed parent copy. */
    [[nodiscard]] virtual bool product_family_proof_revision_invariant(const AdaptiveLeaf* = nullptr) const
    {
        return false;
    }
};
/** Allocation-free explicit provider capability. A provider label or custom
 * wrapper does not grant this opt-in contract. Virtual dispatch preserves the
 * actual provider's promise across statically embedded library images. */
[[nodiscard]] bool has_native_cartesian_geometry(const AdaptiveProviderKernel&) noexcept;
/** Same ancestry checks, coordinate expression and order as geometry().
 * Returns empty for unsupported kernels; performs no allocation. */
[[nodiscard]] std::optional<AdaptiveCartesianGeometryBounds> native_cartesian_geometry_bounds(
    const AdaptiveProviderKernel&, const AdaptiveLeaf&);
[[nodiscard]] std::shared_ptr<const AdaptiveProviderKernel> convex_adaptive_kernel(
    unsigned maximum_depth, double minimum_scaled_volume);
/** Exact full family identity includes typed parent and factor split. */
[[nodiscard]] AdaptiveBytes adaptive_family_key(const AdaptiveCellKey&, AdaptiveSplit);
} // namespace SimpleFluid::Meshes::AMR
