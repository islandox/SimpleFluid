/** @file PolyhedralAMR.hh
 * @brief Immutable bounded-bootstrap convex refinement with canonical atomic faces.
 */
#pragma once
#include "geometry/PLIC/PolyhedralComplex.hh"
#include "geometry/mesh/AMR/AdaptiveOverlap.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "parallel/CompositePartition.hh"

namespace SimpleFluid::Meshes::AMR
{
/** Already embedded native planar-convex input. Region/material membership is
 * explicit per original native cell and independent of topology fragments.
 * Stage 4 admits one native region without joins and one physical region ID;
 * per-root materials may differ. Empty membership arrays select zero for every root. Planning retains bounded
 * replicated canonical geometry; native partition providers remain sparse.
 */
struct PolyhedralAMRGrid
{
    SP<const MultiRegionMesh> mesh;
    std::vector<std::uint64_t> physical_regions, materials;
    unsigned max_depth = 8;
    double minimum_scaled_volume = 1e-12;
    std::uint64_t maximum_cells = 4096, maximum_face_tiles = 65536;
};

class PolyhedralAMR final : public AdaptiveHierarchy
{
public:
    struct Data;
    explicit PolyhedralAMR(PolyhedralAMRGrid grid);
    [[nodiscard]] HierarchyObservation observation() const noexcept override;
    [[nodiscard]] bool geometry_unchanged() const noexcept;
    [[nodiscard]] const SP<const MultiRegionMesh>& native_geometry() const noexcept override;
    [[nodiscard]] const SP<const MultiRegionMesh>& geometry() const noexcept;
    [[nodiscard]] const std::shared_ptr<const CompositePartition<>>& native_partition() const noexcept;
    [[nodiscard]] std::shared_ptr<const PolyhedralAMR> with_partition(
        std::shared_ptr<const CompositePartition<>>) const;
    [[nodiscard]] AdaptiveCellKey cell_key(std::uint64_t cell) const override;
    [[nodiscard]] std::uint64_t cell_material(std::uint64_t cell) const override;
    [[nodiscard]] std::size_t cell_fragment(std::uint64_t cell) const override;
    [[nodiscard]] PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> cell_geometry(std::uint64_t cell) const;
    [[nodiscard]] const std::shared_ptr<const PLIC::PolyhedralComplex<ArithmeticMode::Ordinary>>&
    complex() const noexcept;

    [[nodiscard]] AdaptiveBootstrap detach_owned(
        std::span<const std::uint64_t> owned_cells, AdaptiveBytes domain) const;

    /** Collective source leaf ordinals; coarsening requires the complete
     * immediate recorded sibling family in the original source. Every face
     * complex is rebuilt against surviving fine neighbors before publication.
     * Conservative preflight budgets bound upper cell/face-overlay growth.
     */
    [[nodiscard]] std::shared_ptr<const PolyhedralAMR> adapt(
        const std::vector<std::size_t>& refine = {}, const std::vector<AdaptiveCellKey>& coarsen = {}) const;
    class AdaptationPlan
    {
    public:
        const HierarchyObservation& source_observation() const noexcept { return source_; }
        const std::shared_ptr<const PolyhedralAMR>& candidate() const noexcept { return candidate_; }

    private:
        friend class PolyhedralAMR;
        HierarchyObservation source_;
        std::shared_ptr<const PolyhedralAMR> candidate_;
        std::uint64_t collective_id_ = 0;
    };
    [[nodiscard]] AdaptationPlan plan_adaptation(
        const std::vector<std::size_t>& refine = {}, const std::vector<AdaptiveCellKey>& coarsen = {}) const;
    [[nodiscard]] std::shared_ptr<const PolyhedralAMR> apply_adaptation(const AdaptationPlan&) const;
    [[nodiscard]] std::vector<AdaptiveAncestryRelation> overlap_relations(const PolyhedralAMR& target) const;
    [[nodiscard]] std::vector<AdaptiveAncestryRelation> overlap_relations(
        const PolyhedralAMR& target, const std::vector<std::uint64_t>& targets) const;

private:
    explicit PolyhedralAMR(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;
};
} // namespace SimpleFluid::Meshes::AMR
