/**
 * @file RegionProviders.hh
 * @author islandox
 * @brief Concrete topology and geometry contracts for static mesh regions.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#pragma once

#include "geometry/mesh/EntityRange.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "geometry/mesh/UnstructuredMesh.hh"

#include <memory>
#include <ranges>

namespace SimpleFluid::Meshes
{
/** @brief Allocated capacities in bytes, excluding allocator headers and shared_ptr control blocks. */
struct MeshStorageReport
{
    size_t topology = 0;
    size_t geometry = 0;
    size_t interfaces = 0;
    size_t indexing = 0;
    size_t compatibility = 0;
    size_t boundaries = 0;
    size_t objects = 0;
    size_t indexing_estimate = 0;
    size_t required_maps_estimate = 0;
    size_t output_only = 0;
    size_t measured_bytes() const noexcept
    {
        return topology + geometry + interfaces + indexing + compatibility + boundaries + objects;
    }
};

/** @brief Ordinal layout and orientation compatibility; explicit templates require identical identity. */
struct RegionLayout
{
    enum class Family { Rectilinear, Extruded, Explicit, Cylindrical };
    Family family;
    size_t cells, faces, nodes;
    std::array<size_t, 3> extents;
    const void* template_identity = nullptr;
    std::array<bool,3> periodic{};
    bool operator==(const RegionLayout&) const = default;
};

/** @brief Topology supplies owner-oriented incidence and reversible entity ordinals. */
template<class T>
concept TopologyProvider = MeshIndexTypePack<typename T::index_type_pack>
    && requires(const T& t, size_t c, size_t f, size_t n)
{
    { t.layout() } -> std::same_as<RegionLayout>;
    { t.storage_identity() } -> std::same_as<const void*>;
    { t.cell_faces(c) } -> std::ranges::random_access_range;
    { t.owner_cell(f) } -> std::convertible_to<size_t>;
    { t.neighbor_cell(f) } -> std::convertible_to<size_t>;
    { t.cell_nodes(c) } -> std::ranges::random_access_range;
    { t.face_nodes(f) } -> std::ranges::random_access_range;
    { t.cell_type(c) } -> std::same_as<MeshUtils::CellType>;
};

/** @brief Geometry supplies physical metrics in a declared ordinal layout without incidence storage. */
template<class G>
concept GeometryProvider = requires(const G& g, size_t c, size_t f, size_t n)
{
    { g.layout() } -> std::same_as<RegionLayout>;
    { g.storage_identity() } -> std::same_as<const void*>;
    { g.cell_volume(c) } -> std::convertible_to<real_t>;
    { g.cell_centroid(c) } -> std::same_as<MeshUtils::Vec3>;
    { g.face_area(f) } -> std::convertible_to<real_t>;
    { g.face_centroid(f) } -> std::same_as<MeshUtils::Vec3>;
    { g.face_area_vector(f) } -> std::same_as<MeshUtils::Vec3>;
    { g.node_coordinates(n) } -> std::same_as<MeshUtils::Vec3>;
    { g.geometry_epoch() } -> std::convertible_to<std::uint64_t>;
};

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

/** @brief Zero-copy ordinal topology/geometry adapter retaining the exact native object. */
template<class Native>
class NativeRegionProvider : public GeometryExecutionGuard
{
public:
    using index_type_pack = UnstructuredMeshIndexTypes;
    using ID = uint64_t;
    using Vec3 = MeshUtils::Vec3;
    explicit NativeRegionProvider(std::shared_ptr<const Native> mesh) : d_mesh(std::move(mesh))
    {
        if (!d_mesh) throw std::invalid_argument("Region provider requires a non-null native mesh.");
        if (d_mesh->num_owned_cells() != d_mesh->num_cells() || d_mesh->num_owned_faces() != d_mesh->num_faces())
            throw std::invalid_argument("A region must be standalone serial native geometry.");
    }
    RegionLayout layout() const
    {
        if constexpr (std::same_as<Native, OrthogonalCartesian3D> || std::same_as<Native, OrthogonalCylindrial3D>)
        {
            const auto& n = d_mesh->indexer().num_cells_per_dim;
            const auto& p=d_mesh->indexer().periodic_dimensions;
            return {std::same_as<Native,OrthogonalCartesian3D>?RegionLayout::Family::Rectilinear:RegionLayout::Family::Cylindrical,
                d_mesh->num_cells(),d_mesh->num_faces(),d_mesh->num_nodes(),{n[0],n[1],n[2]},nullptr,{p[0],p[1],p[2]}};
        }
        else if constexpr (std::same_as<Native, SemiStructuredXY_Z>)
        {
            const auto& i = d_mesh->indexer();
            return {RegionLayout::Family::Extruded, d_mesh->num_cells(), d_mesh->num_faces(),
                d_mesh->num_nodes(), {i.num_cells_per_layer, i.num_side_faces_per_layer, i.num_layers}, d_mesh.get()};
        }
        else
            return {RegionLayout::Family::Explicit, d_mesh->num_cells(), d_mesh->num_faces(),
                d_mesh->num_nodes(), {}, d_mesh.get()};
    }
    EntityRange<ID> cell_faces(size_t c) const
    {
        const auto& faces = d_mesh->cell_faces(d_mesh->cell_id(c));
        return {this, c, faces.size(), [](const void* source, size_t c, size_t i) -> ID
        {
            const auto& m = *static_cast<const NativeRegionProvider*>(source)->d_mesh;
            const auto& faces = m.cell_faces(m.cell_id(c));
            return m.face_local_id(faces[i]);
        }};
    }
    ID owner_cell(size_t f) const { return d_mesh->cell_local_id(d_mesh->owner_cell(d_mesh->face_id(f))); }
    ID neighbor_cell(size_t f) const
    {
        const auto c = d_mesh->neighbor_cell(d_mesh->face_id(f));
        return c == Native::invalid_cell_id() ? UnstructuredMesh::invalid_ordinal : d_mesh->cell_local_id(c);
    }
    real_t cell_volume(size_t c) const { return d_mesh->cell_volume(d_mesh->cell_id(c)); }
    Vec3 cell_centroid(size_t c) const { return d_mesh->cell_centroid(d_mesh->cell_id(c)); }
    real_t face_area(size_t f) const { return d_mesh->face_area(d_mesh->face_id(f)); }
    Vec3 face_centroid(size_t f) const { return d_mesh->face_centroid(d_mesh->face_id(f)); }
    Vec3 face_area_vector(size_t f) const { return d_mesh->face_area_vector(d_mesh->face_id(f)); }
    Vec3 node_coordinates(size_t n) const { return d_mesh->node_coordinates(d_mesh->node_id(n)); }
    std::uint64_t geometry_epoch() const { return mesh_geometry_epoch(*d_mesh); }
    int boundary_id(size_t f) const { return d_mesh->boundary_id(d_mesh->face_id(f)); }
    auto boundary_batch_ids() const { return d_mesh->boundary_batch_ids(); }
    const std::string& boundary_batch_name(int b) const { return d_mesh->boundary_batch_name(b); }
    EntityRange<ID> cell_nodes(size_t c) const;
    EntityRange<ID> face_nodes(size_t f) const;
    MeshUtils::CellType cell_type(size_t c) const;
    const Native& native() const { return *d_mesh; }
    const void* storage_identity() const { return d_mesh.get(); }
    MeshStorageReport storage_report() const;
private:
    std::shared_ptr<const Native> d_mesh;
};

/**
 * @brief A caller-supplied common topology and compatible independent geometry.
 *
 * Identity and boundary metadata describe a topology region, not a material or
 * MPI partition. Explicit/factored adapters require the same native template;
 * rectilinear providers validate all indexing extents. The object retains both
 * const providers and checks exact layout and revision before static use.
 */
template<TopologyProvider T, GeometryProvider G>
class IsoRegion
{
public:
    IsoRegion(std::string name, std::shared_ptr<const T> topology, std::shared_ptr<const G> geometry)
        : d_name(std::move(name)), d_topology(std::move(topology)), d_geometry(std::move(geometry))
    {
        if (d_name.empty() || !d_topology || !d_geometry)
            throw std::invalid_argument("IsoRegion requires a name and live providers.");
        d_layout = d_topology->layout();
        if (d_layout != d_geometry->layout())
            throw std::invalid_argument("IsoRegion topology/geometry numbering or entity extents mismatch.");
        d_epoch = d_geometry->geometry_epoch();
        d_geometry_identity = d_geometry->storage_identity();
        d_topology_identity = d_topology->storage_identity();
    }
    const std::string& name() const { return d_name; }
    const T& topology() const { return *d_topology; }
    const G& geometry() const { return *d_geometry; }
    const RegionLayout& layout() const { return d_layout; }
    void validate_static() const
    {
        if (d_geometry_identity != d_geometry->storage_identity() || d_topology_identity != d_topology->storage_identity()
            || d_epoch != d_geometry->geometry_epoch() || d_layout != d_geometry->layout()
            || d_layout != d_topology->layout())
            throw std::logic_error("Static region constituent changed; rebuild MultiRegionMesh and its fields/operators.");
    }
private:
    std::string d_name;
    std::shared_ptr<const T> d_topology;
    std::shared_ptr<const G> d_geometry;
    RegionLayout d_layout{};
    std::uint64_t d_epoch = 0;
    const void* d_geometry_identity = nullptr;
    const void* d_topology_identity = nullptr;
};

using CartesianRegion = IsoRegion<RectilinearTopology, RectilinearGeometry>;
template<class Native> using NativeIsoRegion = IsoRegion<NativeRegionProvider<Native>, NativeRegionProvider<Native>>;

/** @brief Retain one native object with two const, zero-copy provider observations. */
template<class Native>
auto native_region(std::string name, std::shared_ptr<Native> mesh)
{
    using N = std::remove_const_t<Native>;
    auto provider = std::make_shared<const NativeRegionProvider<N>>(std::move(mesh));
    return NativeIsoRegion<N>(std::move(name), provider, provider);
}

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
