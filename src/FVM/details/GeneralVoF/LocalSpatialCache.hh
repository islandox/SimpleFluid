/** @file LocalSpatialCache.hh
 * @brief Private immutable retained-domain indexes, shared across phase publications.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
namespace local_spatial_detail
{
constexpr auto mode = ArithmeticMode::Ordinary;
using Wide = std::array<long double, 3>;
constexpr auto infinity = std::numeric_limits<long double>::infinity();
struct SIMPLEFLUID_FVM_LOCAL Box
{
    Wide low{infinity, infinity, infinity}, high{-infinity, -infinity, -infinity};
    void add(const Wide&);
    void add(const Box&);
    bool overlaps(const Box&) const;
};
/** Balanced median index. Spatial ordering only schedules conservative AABB tests;
 * every returned list is restored to its canonical source order. */
class SIMPLEFLUID_FVM_LOCAL Index
{
    struct Node
    {
        Box box;
        std::size_t begin = 0, end = 0, left = 0, right = 0;
    };
    const std::vector<Box>& boxes_;
    std::vector<std::size_t> order_;
    std::vector<Node> nodes_;
    std::size_t build(std::size_t, std::size_t, SimpleFluid::geometry::swept_volume::Work&);
    void visit(std::size_t, const Box&, std::vector<std::size_t>&, SimpleFluid::geometry::swept_volume::Work&, GeneralVoFTransportDiagnostics&) const;

public:
    explicit Index(const std::vector<Box>& boxes) : boxes_(boxes) {}
    void initialize(SimpleFluid::geometry::swept_volume::Work&);
    std::vector<std::size_t> query(const Box&, SimpleFluid::geometry::swept_volume::Work&, GeneralVoFTransportDiagnostics&) const;
};
struct Frontier
{
    PLIC::PlanarFacePtr<mode> geometry;
    bool first;
};
} // namespace local_spatial_detail
/** Immutable retained-domain geometry, independent of trial phases and traces. */
struct SIMPLEFLUID_FVM_LOCAL LocalSpatialCache
{
    struct CellIdentity
    {
        Key key;
        std::uint64_t gid;
        int owner;
        PLIC::DynamicConvexGeometryPtr geometry;
    };
    AMR::HierarchyObservation observation, supplied_observation;
    std::vector<CellIdentity> cells;
    std::vector<local_spatial_detail::Box> cell_boxes, frontier_boxes;
    std::vector<local_spatial_detail::Frontier> frontiers;
    local_spatial_detail::Index donor_index{cell_boxes}, frontier_index{frontier_boxes};
    std::size_t bytes = 0;
    LocalSpatialCache() = default;
    LocalSpatialCache(const LocalSpatialCache&) = delete;
};
/** The topology slot is only a synchronization/lifetime anchor. Exact source
 * observations and the retained cell catalog additionally authenticate reuse.
 * Every consumer pays the same conservative retained-storage charge; only an
 * actual build pays the geometry traversal and sorting operation counts. */
using SpatialCacheAdmission = void (*)(const GeneralVoF::Data&, std::size_t);
SIMPLEFLUID_FVM_LOCAL std::shared_ptr<const LocalSpatialCache> retained_spatial_cache(
    const GeneralVoF::Data&, SimpleFluid::geometry::swept_volume::Work&, GeneralVoFTransportDiagnostics&, SpatialCacheAdmission = nullptr);
} // namespace SimpleFluid::VoF::general_vof_detail
