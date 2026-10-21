/** @file AdaptiveOverlap.hh
 * @brief Certified sparse overlaps selected by explicit cell ancestry.
 */
#pragma once
#include "geometry/ConvexCellGeometry.hh"
#include "geometry/mesh/AMR/AdaptiveHierarchy.hh"

namespace SimpleFluid::Meshes::AMR
{
struct AdaptiveCellBinding
{
    AdaptiveCellKey key;
    std::uint64_t gid = 0, material = 0;
    ConvexGeometryPtr geometry;
};

struct AdaptiveAncestryRelation
{
    enum class Kind
    {
        Unchanged,
        Refinement,
        Coarsening,
        ProductIntersection ///< Opposite strict base/Z ancestry directions.
    };
    Kind kind = Kind::Unchanged;
    std::uint64_t source_gid = 0, target_gid = 0;
};

struct AdaptiveOverlapRecord
{
    AdaptiveCellKey source_key, target_key;
    std::uint64_t source_gid = 0, target_gid = 0;
    AdaptiveAncestryRelation::Kind relation = AdaptiveAncestryRelation::Kind::Unchanged;
    ConvexGeometryPtr intersection;
    GeometryMoments moments;
};

/** Geometry-only local certificate. Inputs are caller-supplied compact catalogs
 * and explicit ancestry pairs, never an implicit all-pairs mesh search.
 * Every supplied source/target cell must be exactly covered without overlap.
 * Distribution and solver physical state are deliberately separate consumers.
 */
class AdaptiveOverlapPlan
{
public:
    [[nodiscard]] static std::shared_ptr<const AdaptiveOverlapPlan> build(HierarchyObservation source,
        HierarchyObservation target, std::vector<AdaptiveCellBinding> source_cells,
        std::vector<AdaptiveCellBinding> target_cells, std::span<const AdaptiveAncestryRelation> relations);
    [[nodiscard]] const HierarchyObservation& source_observation() const noexcept { return source_; }
    [[nodiscard]] const HierarchyObservation& target_observation() const noexcept { return target_; }
    [[nodiscard]] const std::vector<AdaptiveCellBinding>& source_cells() const noexcept { return source_cells_; }
    [[nodiscard]] const std::vector<AdaptiveCellBinding>& target_cells() const noexcept { return target_cells_; }
    [[nodiscard]] const std::vector<AdaptiveOverlapRecord>& records() const noexcept { return records_; }
    /** Local exact preflight; collective consumers must aggregate failure. */
    void validate_observations(const HierarchyObservation& source, const HierarchyObservation& target) const;

private:
    HierarchyObservation source_, target_;
    std::vector<AdaptiveCellBinding> source_cells_, target_cells_;
    std::vector<AdaptiveOverlapRecord> records_;
};
} // namespace SimpleFluid::Meshes::AMR
