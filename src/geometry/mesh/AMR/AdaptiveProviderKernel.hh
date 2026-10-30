/** @file AdaptiveProviderKernel.hh
 * @brief Pure local adaptation kernels; no communicator or replicated hierarchy.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
namespace SimpleFluid::Meshes::AMR
{
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
struct AdaptiveCellMetrics
{
    double volume = 0;
    PLIC::GeometryPoint centroid{};
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
    [[nodiscard]] virtual PLIC::PointIdentity<ArithmeticMode::Ordinary> contact_point(
        const AdaptiveLeaf&, const PLIC::PointIdentity<ArithmeticMode::Ordinary>&) const;
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
    /** Complete immediate source membership is verified by distributed family
     * authority before this exact local inverse is evaluated. */
    [[nodiscard]] AdaptiveLeaf coarsen(const AdaptiveFamily&, std::span<const AdaptiveLeaf>) const;
};
[[nodiscard]] std::shared_ptr<const AdaptiveProviderKernel> convex_adaptive_kernel(
    unsigned maximum_depth, double minimum_scaled_volume);
/** Exact full family identity includes typed parent and factor split. */
[[nodiscard]] AdaptiveBytes adaptive_family_key(const AdaptiveCellKey&, AdaptiveSplit);
} // namespace SimpleFluid::Meshes::AMR
