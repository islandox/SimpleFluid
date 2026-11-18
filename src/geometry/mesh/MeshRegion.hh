/** @file MeshRegion.hh
 * @brief General static mesh-region contracts and retained native adapters.
 */
#pragma once

#include "geometry/GeometryEpoch.hh"
#include "geometry/GeometryExecutionGuard.hh"
#include "geometry/MeshUtils.hh"
#include "geometry/mesh/EntityRange.hh"
#include "geometry/mesh/MeshIndexTypes.hh"
#include "geometry/mesh/RegionProviderConcepts.hh"

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

namespace SimpleFluid::geometry_detail
{
class NativeAllocationEnvelope;
}
namespace SimpleFluid::Meshes
{
namespace details { class NativeLegacyCartesianAccess; }
class NativeFragmentAccess;
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

/** @brief Zero-copy ordinal topology/geometry adapter retaining the exact native object.
 * The four built-in native meshes are instantiated in MeshRegion.cc. Other
 * native types can include MeshRegion.tcc in their own instantiation unit.
 */
template<class Native> class NativeRegionProvider : public GeometryExecutionGuard
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
    friend class NativeFragmentAccess;
    friend class details::NativeLegacyCartesianAccess;
    std::shared_ptr<const Native> d_mesh;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> d_native_allocation;
};

/**
 * @brief A caller-supplied common topology and compatible independent geometry.
 *
 * Identity and boundary metadata describe a topology region, not a material or
 * MPI partition. Explicit/factored adapters require the same native template;
 * rectilinear providers validate all indexing extents. The object retains both
 * const providers and checks exact layout and revision before static use.
 */
template<TopologyProvider T, GeometryProvider G> class IsoRegion
{
public:
    IsoRegion(std::string name, std::shared_ptr<const T> topology, std::shared_ptr<const G> geometry);
    const std::string& name() const { return d_name; }
    const T& topology() const { return *d_topology; }
    const G& geometry() const { return *d_geometry; }
    const RegionLayout& layout() const { return d_layout; }
    void validate_static() const;

private:
    friend class NativeFragmentAccess;
    friend class details::NativeLegacyCartesianAccess;
    std::string d_name;
    std::shared_ptr<const T> d_topology;
    std::shared_ptr<const G> d_geometry;
    RegionLayout d_layout{};
    std::uint64_t d_epoch = 0;
    const void* d_geometry_identity = nullptr;
    const void* d_topology_identity = nullptr;
};

/** @brief Retain one native object with two const, zero-copy provider observations. */
template<class Native> auto native_region(std::string name, std::shared_ptr<Native> mesh);

template<class Native> using NativeIsoRegion = IsoRegion<NativeRegionProvider<Native>, NativeRegionProvider<Native>>;

} // namespace SimpleFluid::Meshes

#include "geometry/mesh/MeshRegion.ipp"
