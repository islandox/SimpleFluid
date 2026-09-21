/** @file SweptRZRegionProviders.hh
 * @brief Compact planar polygon sectors swept from a radius-height template.
 */
#pragma once

#include "geometry/mesh/ExtrudedRegionProviders.hh"

namespace SimpleFluid::Meshes
{
/**
 * @brief Immutable polygonal angular sweep with O(base + angles) storage.
 *
 * Input nodes store radius in x, physical elevation in y, and zero in z.
 * Base triangles/quadrilaterals must be strictly convex and CLOCKWISE in
 * that radius-height plane. Increasing theta then gives positive physical
 * HEX_8/WEDGE_6 connectivity. Straight edges join consecutive angular planes;
 * the resulting cells have planar faces, not curved cylindrical metrics.
 *
 * The topology uses its usual base-cell/axial-layer ordinals, with angular
 * layers replacing axial layers. Its nonperiodic caps may be joined by a
 * coincident explicit self-interface in MultiRegionMesh. Physical Z motion
 * belongs to the enclosing composite's common affine map.
 */
class SweptRZGeometry
{
public:
    using Vec3 = MeshUtils::Vec3;
    SweptRZGeometry(std::shared_ptr<const ExtrudedTopology> topology,
                    Arr<Vec3> rz_nodes, ArrReal theta_edges);
    SweptRZGeometry& operator=(const SweptRZGeometry&) = delete;
    SweptRZGeometry& operator=(SweptRZGeometry&&) = delete;

    RegionLayout layout() const { return d_topology->layout(); }
    const void* storage_identity() const { return this; }
    std::uint64_t geometry_epoch() const noexcept { return 0; }
    real_t cell_volume(size_t cell) const;
    Vec3 cell_centroid(size_t cell) const;
    real_t face_area(size_t face) const;
    Vec3 face_centroid(size_t face) const;
    Vec3 face_area_vector(size_t face) const;
    Vec3 node_coordinates(size_t node) const;
    MeshStorageReport storage_report() const;

    const Arr<Vec3>& rz_nodes() const noexcept { return d_nodes; }
    const ArrReal& theta_edges() const noexcept { return d_theta; }
    const std::shared_ptr<const ExtrudedTopology>& topology_ptr() const noexcept { return d_topology; }
    EntityRange<EntityRange<uint64_t>> rz_cell_nodes() const;
    std::array<real_t, 2> axial_bounds() const noexcept { return d_bounds; }

private:
    struct BaseMetrics
    {
        real_t area, radial_moment, radial_second_over_first, radial_height_over_first;
        real_t centroid_radius, centroid_height;
    };
    Vec3 physical_node(unsigned base_node, unsigned angle) const;
    std::shared_ptr<const ExtrudedTopology> d_topology;
    Arr<Vec3> d_nodes;
    ArrReal d_theta;
    Arr<BaseMetrics> d_metrics;
    std::array<real_t, 2> d_bounds;
};

using SweptRZRegion = IsoRegion<ExtrudedTopology, SweptRZGeometry>;
SweptRZRegion swept_rz_region(std::string name,
    std::shared_ptr<const ExtrudedTopology> topology,
    Arr<MeshUtils::Vec3> rz_nodes, ArrReal theta_edges);
static_assert(GeometryProvider<SweptRZGeometry>);
} // namespace SimpleFluid::Meshes
