/**
 * @file RegionProviders.hh
 * @author islandox
 * @brief Concrete topology and geometry contracts for static mesh regions.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#pragma once

#include "geometry/mesh/MeshRegion.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "geometry/mesh/UnstructuredMesh.hh"

#include <memory>
#include <ranges>

namespace SimpleFluid::Meshes
{
/** @brief Procedural HEX_8 topology backed by a shared immutable orthogonal template. */
class RectilinearTopology : public GeometryExecutionGuard
{
public:
    using index_type_pack = UnstructuredMeshIndexTypes;
    using ID = uint64_t;
    explicit RectilinearTopology(std::shared_ptr<const OrthoMeshTopo> topology);
    RegionLayout layout() const;
    EntityRange<ID> cell_faces(size_t c) const;
    EntityRange<ID> cell_nodes(size_t c) const;
    EntityRange<ID> face_nodes(size_t f) const;
    ID owner_cell(size_t f) const;
    ID neighbor_cell(size_t f) const;
    MeshUtils::CellType cell_type(size_t) const { return MeshUtils::CellType::HEXAHEDRON; }
    int boundary_id(size_t f) const { return d_topology->boundary_id(indexer().face_id(f)); }
    auto boundary_batch_ids() const { return d_topology->boundary_batch_ids(); }
    const std::string& boundary_batch_name(int b) const { return d_topology->boundary_batch_name(b); }
    const OrthogonalIndexer& indexer() const { return d_topology->indexer(); }
    const OrthoMeshTopo& native_topology() const { return *d_topology; }
    const void* storage_identity() const { return d_topology.get(); }
    MeshStorageReport storage_report() const;
private:
    std::shared_ptr<const OrthoMeshTopo> d_topology;
};

/**
 * @brief Independently owned Cartesian coordinate arrays; metrics are computed per query.
 *
 * This is geometry only: it stores an arithmetic indexer and axis coordinates,
 * with no topology object or per-cell/per-face metrics. It is immutable after
 * construction and may be paired with a shared RectilinearTopology.
 */
class RectilinearGeometry
{
public:
    using Vec3 = MeshUtils::Vec3;
    explicit RectilinearGeometry(Vec3D<ArrReal> edges);
    const Vec3D<ArrReal>& cell_edges() const noexcept { return d_edges; }
    RegionLayout layout() const;
    real_t cell_volume(size_t c) const;
    Vec3 cell_centroid(size_t c) const;
    real_t face_area(size_t f) const;
    Vec3 face_centroid(size_t f) const;
    Vec3 face_area_vector(size_t f) const;
    Vec3 node_coordinates(size_t n) const;
    std::uint64_t geometry_epoch() const noexcept { return 0; }
    const void* storage_identity() const { return this; }
    MeshStorageReport storage_report() const;
    RectilinearGeometry& operator=(const RectilinearGeometry&) = delete;
    RectilinearGeometry& operator=(RectilinearGeometry&&) = delete;
private:
    Vec3D<ArrReal> d_edges;
    OrthogonalIndexer d_indexer;
};

using CartesianRegion = IsoRegion<RectilinearTopology, RectilinearGeometry>;
/** @brief Build Cartesian providers without constructing a full native geometry mesh. */
CartesianRegion cartesian_region(std::string name, Vec3D<ArrReal> edges);

static_assert(TopologyProvider<RectilinearTopology>);
static_assert(GeometryProvider<RectilinearGeometry>);
static_assert(TopologyProvider<NativeRegionProvider<SemiStructuredXY_Z>>);
static_assert(GeometryProvider<NativeRegionProvider<SemiStructuredXY_Z>>);
static_assert(TopologyProvider<NativeRegionProvider<UnstructuredMesh>>);
static_assert(GeometryProvider<NativeRegionProvider<UnstructuredMesh>>);
} // namespace SimpleFluid::Meshes

#include "geometry/mesh/RegionProviders.ipp"
