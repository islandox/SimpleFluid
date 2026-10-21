/** @file MeshRegion.hh
 * @brief General static mesh-region contracts and retained native adapters.
 */
#pragma once

#include "geometry/GeometryEpoch.hh"
#include "geometry/GeometryExecutionGuard.hh"
#include "geometry/MeshUtils.hh"
#include "geometry/mesh/EntityRange.hh"
#include "geometry/mesh/MeshIndexTypes.hh"

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace SimpleFluid::Meshes
{
class OrthogonalCartesian3D;
class OrthogonalCylindrial3D;
class SemiStructuredXY_Z;
class UnstructuredMesh;

/** A native face ordinal qualified by its topology-region ordinal. */
struct RegionFace
{
    size_t region = 0;
    uint64_t face = 0;
    auto operator<=>(const RegionFace&) const = default;
};
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

/** @brief Zero-copy ordinal topology/geometry adapter retaining the exact native object.
 * The four built-in native meshes are instantiated in MeshRegion.cc. Other
 * native types can include MeshRegion.tcc in their own instantiation unit.
 */
template<class Native>
class NativeRegionProvider : public GeometryExecutionGuard
{
public:
    using index_type_pack = UnstructuredMeshIndexTypes;
    using ID = uint64_t;
    using Vec3 = MeshUtils::Vec3;
    explicit NativeRegionProvider(std::shared_ptr<const Native> mesh);
    RegionLayout layout() const;
    EntityRange<ID> cell_faces(size_t c) const;
    ID owner_cell(size_t f) const { return d_mesh->cell_local_id(d_mesh->owner_cell(d_mesh->face_id(f))); }
    ID neighbor_cell(size_t f) const;
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
    IsoRegion(std::string name, std::shared_ptr<const T> topology, std::shared_ptr<const G> geometry);
    const std::string& name() const { return d_name; }
    const T& topology() const { return *d_topology; }
    const G& geometry() const { return *d_geometry; }
    const RegionLayout& layout() const { return d_layout; }
    void validate_static() const;
private:
    std::string d_name;
    std::shared_ptr<const T> d_topology;
    std::shared_ptr<const G> d_geometry;
    RegionLayout d_layout{};
    std::uint64_t d_epoch = 0;
    const void* d_geometry_identity = nullptr;
    const void* d_topology_identity = nullptr;
};

/** @brief Retain one native object with two const, zero-copy provider observations. */
template<class Native>
auto native_region(std::string name, std::shared_ptr<Native> mesh);

template<class Native> using NativeIsoRegion = IsoRegion<NativeRegionProvider<Native>, NativeRegionProvider<Native>>;

} // namespace SimpleFluid::Meshes

#include "geometry/mesh/MeshRegion.ipp"
