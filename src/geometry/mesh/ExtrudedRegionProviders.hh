/**
 * @file ExtrudedRegionProviders.hh
 * @author islandox
 * @brief Shared immutable extrusion topology with independently owned XY/Z geometry.
 * @version 0.1
 * @date 2026-09-10
 * @copyright Copyright (c) 2026
 */
#pragma once
#include "geometry/mesh/RegionProviders.hh"
namespace SimpleFluid::Meshes
{
/** @brief A single immutable base template, factored through the existing axial indexer. */
class ExtrudedTopology
{
public:
    using index_type_pack=UnstructuredMeshIndexTypes;
    using ID=uint64_t;
    explicit ExtrudedTopology(SemiStructMeshTopo topology);
    ExtrudedTopology(unsigned nodes, const Arr<Arr<unsigned>>& cells, unsigned layers,
        const Arr<SemiStructMeshTopo::BoundaryEdge>& boundaries={});
    ExtrudedTopology& operator=(const ExtrudedTopology&)=delete;
    ExtrudedTopology& operator=(ExtrudedTopology&&)=delete;
    RegionLayout layout() const;
    const SemiStructuredIndexer& indexer() const { return d_topology.indexer(); }
    const SemiStructMeshTopo& native_topology() const { return d_topology; }
    const void* storage_identity() const { return this; }
    MeshStorageReport storage_report() const { return {.topology=d_topology.storage_bytes()-sizeof(d_topology)}; }
    EntityRange<ID> base_cell_nodes(size_t c) const;
    EntityRange<ID> cell_faces(size_t c) const;
    EntityRange<ID> cell_nodes(size_t c) const;
    EntityRange<ID> face_nodes(size_t f) const;
    ID owner_cell(size_t f) const;
    ID neighbor_cell(size_t f) const;
    MeshUtils::CellType cell_type(size_t c) const;
    int boundary_id(size_t f) const { return d_topology.boundary_id(indexer().face_id(f)); }
    auto boundary_batch_ids() const { return d_topology.boundary_batch_ids(); }
    const std::string& boundary_batch_name(int b) const { return d_topology.boundary_batch_name(b); }
private:
    SemiStructMeshTopo d_topology;
};
/**
 * @brief Independent straight-extruded geometry with O(base + layers) storage.
 *
 * XY nodes use z=0 and the supplied template's exact numbering and CCW loops.
 * Z coordinates are finite and strictly increasing. Base metrics are cached
 * once; no cell or face metrics are expanded through layers.
 */
class ExtrudedGeometry
{
public:
    using Vec3=MeshUtils::Vec3;
    ExtrudedGeometry(std::shared_ptr<const ExtrudedTopology> topology, Arr<Vec3> xy_nodes, ArrReal z_edges);
    ExtrudedGeometry& operator=(const ExtrudedGeometry&)=delete;
    ExtrudedGeometry& operator=(ExtrudedGeometry&&)=delete;
    RegionLayout layout() const { return d_topology->layout(); }
    const void* storage_identity() const { return this; }
    std::uint64_t geometry_epoch() const noexcept { return 0; }
    real_t cell_volume(size_t c) const;
    Vec3 cell_centroid(size_t c) const;
    real_t face_area(size_t f) const;
    Vec3 face_centroid(size_t f) const;
    Vec3 face_area_vector(size_t f) const;
    Vec3 node_coordinates(size_t n) const;
    MeshStorageReport storage_report() const;
    const Arr<Vec3>& xy_nodes() const { return d_nodes; }
    const ArrReal& z_edges() const { return d_z; }
    EntityRange<EntityRange<uint64_t>> xy_cell_nodes() const;
private:
    std::shared_ptr<const ExtrudedTopology> d_topology;
    Arr<Vec3> d_nodes, d_cell_centers, d_edge_centers, d_edge_normals;
    ArrReal d_z, d_areas, d_lengths;
};
using ExtrudedRegion=IsoRegion<ExtrudedTopology,ExtrudedGeometry>;
ExtrudedRegion extruded_region(std::string name, std::shared_ptr<const ExtrudedTopology> topology,
    Arr<MeshUtils::Vec3> xy_nodes, ArrReal z_edges);
static_assert(TopologyProvider<ExtrudedTopology>);
static_assert(GeometryProvider<ExtrudedGeometry>);
} // namespace SimpleFluid::Meshes
