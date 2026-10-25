/** @file MixedAMR.hh
 * @brief Bounded adaptive mixed-provider assembly with exact nested native joins.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveOverlap.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/LocalProductAMR.hh"
#include "geometry/mesh/AMR/PolyhedralAMR.hh"
#include "parallel/CompositePartition.hh"
namespace SimpleFluid::Meshes::AMR
{
struct MixedStaticParticipant
{
    MultiRegionMesh::Region region;
    std::uint64_t physical_region = 0, material = 0, provider_root = 0;
};
struct MixedPatchJoin
{
    // Optional convenience product, local_products, adaptive providers, then static providers.
    std::size_t first_participant = 0, second_participant = 0;
    std::string first_patch, second_patch;
};
struct MixedParticipantMarks
{
    std::size_t participant = 0;
    std::vector<AdaptiveCellKey> refine; ///< Source leaf cells for convex/Cartesian adapters.
    std::vector<ProductCellAncestry> refine_xy, refine_z, coarsen_xy, coarsen_z;
    std::vector<AdaptiveCellKey> coarsen_polyhedral;
    std::vector<AMRBlock> coarsen_cartesian;
};
struct MixedAMRGrid
{
    std::shared_ptr<const LocalProductAMR> product; ///< Optional first local product.
    std::vector<std::shared_ptr<const LocalProductAMR>> local_products;
    std::uint64_t physical_region = 0;
    std::vector<std::shared_ptr<const AdaptiveHierarchy>> adaptive;
    std::vector<MixedStaticParticipant> stationary;
    std::vector<MixedPatchJoin> joins;
    /** Maximum physical directional width ratio across supported closed
     * face/edge/vertex contacts. External coarse neighbors may refine; unsupported
     * local or conservative mixed-corner proof failures reject atomically. */
    double maximum_span_ratio = 2;
    unsigned maximum_closure_rounds = 12;
    /** Explicit external fluid joins. Contacts belong to separate assemblies. */
    std::vector<MultiRegionMesh::Interface> interfaces;
    std::uint64_t maximum_cells = 4096, maximum_interface_tiles = 65536;
};
/** Internal product seams use complete native patches and exact whole-face
 * nesting. Comparable local dyadic traces permit one generation difference,
 * including edge/vertex contacts; contact proof work is explicitly bounded.
 * Crossed traces reject rather than creating mortar polygons. Static
 * participants retain their native providers and never accept refinement marks.
 * Geometry/remap is qualified only for individually certified planar cells;
 * unsupported static geometry rejects the entire remap before publication.
 */
class MixedAMR final : public AdaptiveHierarchy
{
public:
    struct Data;
    explicit MixedAMR(MixedAMRGrid);
    const SP<const MultiRegionMesh>& native_geometry() const noexcept override;
    const SP<const MultiRegionMesh>& geometry() const noexcept { return native_geometry(); }
    HierarchyObservation observation() const noexcept override;
    bool geometry_unchanged() const noexcept;
    AdaptiveCellKey cell_key(std::uint64_t) const override;
    std::uint64_t cell_material(std::uint64_t) const override;
    std::size_t cell_fragment(std::uint64_t) const override;
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> cell_geometry(std::uint64_t) const;
    const std::shared_ptr<const CompositePartition<>>& native_partition() const noexcept;
    std::shared_ptr<const MixedAMR> with_partition(std::shared_ptr<const CompositePartition<>>) const;
    /** Convenience factor transaction on the optional first product. Stable
     * physical-patch joins are rebuilt automatically; unsupported static splits
     * or incompatible native traces reject the whole transaction.
     */
    std::shared_ptr<const MixedAMR> adapt(const std::vector<ProductCellAncestry>& refine_xy = {},
        const std::vector<ProductCellAncestry>& refine_z = {}, const std::vector<ProductCellAncestry>& coarsen_xy = {},
        const std::vector<ProductCellAncestry>& coarsen_z = {},
        const std::vector<AdaptiveCellKey>& static_marks = {}) const;
    std::shared_ptr<const MixedAMR> adapt_participants(const std::vector<MixedParticipantMarks>&) const;
    class AdaptationPlan
    {
    public:
        const HierarchyObservation& source_observation() const noexcept { return source_; }
        const std::shared_ptr<const MixedAMR>& candidate() const noexcept { return candidate_; }

    private:
        friend class MixedAMR;
        HierarchyObservation source_;
        std::shared_ptr<const MixedAMR> candidate_;
    };
    AdaptationPlan plan_adaptation(const std::vector<ProductCellAncestry>& refine_xy = {},
        const std::vector<ProductCellAncestry>& refine_z = {}, const std::vector<ProductCellAncestry>& coarsen_xy = {},
        const std::vector<ProductCellAncestry>& coarsen_z = {}) const;
    AdaptationPlan plan_participants(const std::vector<MixedParticipantMarks>&) const;
    std::shared_ptr<const MixedAMR> apply_adaptation(const AdaptationPlan&) const;
    std::vector<AdaptiveAncestryRelation> overlap_relations(const MixedAMR&) const;
    std::vector<AdaptiveAncestryRelation> overlap_relations(
        const MixedAMR&, const std::vector<std::uint64_t>& targets) const;
    [[nodiscard]] AdaptiveBootstrap detach_owned(std::span<const std::uint64_t> owned_cells, AdaptiveBytes domain,
        std::size_t byte_budget = 64 * 1024 * 1024) const;
    struct StorageBreakdown
    {
        struct Participant
        {
            const void* identity = nullptr;
            MeshStorageReport report;
            bool includes_complete_planner = false;
        };
        MeshStorageReport native_assembly, assembly_metadata;
        std::optional<MeshStorageReport>
            resident_partition_geometry; ///< Maps/partition planning allocations are not included.
        /** Independently measured retained states. These overlap native_assembly
         * through shared providers and MUST NOT be added to it. Cartesian and
         * polyhedral entries report their native catalog only, explicitly
         * excluding private planner ancestry. No complete memory total/RSS is claimed. */
        std::vector<Participant> retained_participants;
    };
    [[nodiscard]] StorageBreakdown storage_breakdown() const;

private:
    explicit MixedAMR(std::shared_ptr<const Data> d) : data_(std::move(d)) {}
    std::shared_ptr<const Data> data_;
};
/** Contact metadata does not add native pressure connectivity. Each side is a
 * separate connected fluid mesh; contact closure remains an external operation.
 */
struct MixedContactDomains
{
    struct Contact
    {
        std::size_t first_domain, second_domain;
        std::string first_patch, second_patch;
    };
    std::vector<std::shared_ptr<const MixedAMR>> domains;
    std::vector<Contact> contacts;
    void validate() const;
};
} // namespace SimpleFluid::Meshes::AMR
