/** @file ExtrudedAMR.hh
 * @brief Immutable conforming whole-column base and common-Z adaptation.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveOverlap.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "geometry/mesh/ExtrudedRegionProviders.hh"
#include "parallel/CompositePartition.hh"

namespace SimpleFluid::Meshes::AMR
{
struct ExtrudedAMRGrid
{
    std::vector<PLIC::GeometryPoint> xy_nodes;
    std::vector<std::vector<unsigned>> base_cells;
    std::vector<SemiStructMeshTopo::BoundaryEdge> boundaries;
    std::vector<double> z_edges;
    std::map<std::string, std::string> boundary_aliases;
    /** Optional local edge ordinal for each root triangle; defaults to longest
     * represented edge. Adjacent root triangles must mark shared edges compatibly. */
    std::vector<std::size_t> triangle_reference_edges;
    std::uint64_t region_id = 0, material_id = 0;
    unsigned max_depth = 12;
    double minimum_scaled_area = 1e-14;
};

class ExtrudedAMR final : public AdaptiveHierarchy
{
public:
    struct Data;
    explicit ExtrudedAMR(ExtrudedAMRGrid grid);
    [[nodiscard]] HierarchyObservation observation() const noexcept override;
    [[nodiscard]] bool geometry_unchanged() const noexcept;
    [[nodiscard]] const SP<const MultiRegionMesh>& native_geometry() const noexcept override;
    [[nodiscard]] const SP<const MultiRegionMesh>& geometry() const noexcept;
    [[nodiscard]] const std::shared_ptr<const CompositePartition<>>& native_partition() const noexcept;
    [[nodiscard]] std::shared_ptr<const ExtrudedAMR> with_partition(std::shared_ptr<const CompositePartition<>>) const;
    [[nodiscard]] AdaptiveCellKey cell_key(std::uint64_t cell) const override;
    [[nodiscard]] std::uint64_t cell_material(std::uint64_t cell) const override;
    [[nodiscard]] std::size_t cell_fragment(std::uint64_t cell) const override;
    [[nodiscard]] const std::vector<BaseCellKey>& base_keys() const noexcept;
    [[nodiscard]] const std::vector<BaseEdgeKey>& base_edge_keys(std::size_t base_cell) const;
    [[nodiscard]] const std::vector<AxialIntervalKey>& interval_keys() const noexcept;
    [[nodiscard]] PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> cell_geometry(std::uint64_t cell) const;
    [[nodiscard]] std::vector<ProductContactViolation> contact_violations(
        double maximum_span_ratio, unsigned maximum_trace_jump = 1, std::size_t contact_budget = 65536) const;
    [[nodiscard]] AdaptiveBootstrap detach_owned(
        std::span<const std::uint64_t> owned_cells, AdaptiveBytes domain) const;
    [[nodiscard]] AdaptiveBytes root_descriptor() const;
    struct StorageCounts
    {
        std::size_t base_cells, base_vertices, base_edges, z_edges, point_recipes, z_recipes, base_history_nodes,
            axial_history_nodes;
    };
    [[nodiscard]] StorageCounts storage_counts() const noexcept;
    /** Conservative retained allocation bound, including exact dynamic state;
     * shared native geometry is counted once, maps/transient certificates excluded. */
    [[nodiscard]] MeshStorageReport storage_report() const;

    /** Collective marks refer to this hierarchy's base/interval ordinals.
     * Complete immediate sibling families are required for factor coarsening.
     * Closure propagates XY edge splits through whole columns. */
    [[nodiscard]] std::shared_ptr<const ExtrudedAMR> adapt(const std::vector<std::size_t>& refine_base = {},
        const std::vector<std::size_t>& refine_z = {}, const std::vector<BaseCellKey>& coarsen_base = {},
        const std::vector<AxialIntervalKey>& coarsen_z = {}) const;
    class AdaptationPlan
    {
    public:
        const HierarchyObservation& source_observation() const noexcept { return source_; }
        const std::shared_ptr<const ExtrudedAMR>& candidate() const noexcept { return candidate_; }

    private:
        friend class ExtrudedAMR;
        HierarchyObservation source_;
        std::shared_ptr<const ExtrudedAMR> candidate_;
        std::uint64_t collective_id_ = 0;
    };
    [[nodiscard]] AdaptationPlan plan_adaptation(const std::vector<std::size_t>& refine_base = {},
        const std::vector<std::size_t>& refine_z = {}, const std::vector<BaseCellKey>& coarsen_base = {},
        const std::vector<AxialIntervalKey>& coarsen_z = {}) const;
    [[nodiscard]] std::shared_ptr<const ExtrudedAMR> apply_adaptation(const AdaptationPlan&) const;
    /** Enumerate ancestry-compatible product pairs on demand, without creating
     * or retaining a global base*layers shell catalog. An empty explicit target
     * selection means zero records, including on empty MPI owners.
     * Complete-family overlap certification remains an explicit caller operation. */
    [[nodiscard]] std::vector<AdaptiveAncestryRelation> overlap_relations(const ExtrudedAMR& target) const;
    [[nodiscard]] std::vector<AdaptiveAncestryRelation> overlap_relations(
        const ExtrudedAMR& target, const std::vector<std::uint64_t>& target_cells) const;

private:
    explicit ExtrudedAMR(std::shared_ptr<const Data> data);
    std::shared_ptr<const Data> data_;
};
} // namespace SimpleFluid::Meshes::AMR
