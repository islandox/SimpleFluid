/** @file DecomposedSemiStructuredTopology.hh
 * @brief Compact local XY/Z products and exact adjacency halos for a decomposed extrusion.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/mesh/SemiStructMeshTopo.hh"
#include <array>
#include <limits>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace SimpleFluid::Meshes
{
/** Source topology stays shared and immutable. One rank-local XY union is reused
 * by nested radius views and axial bands; no 3D geometry-ID lookup table exists.
 * Owned cells form one XY-part/Z-slab product. The halo is exactly dXY+dZ<=depth.
 */
class SIMPLEFLUID_PUBLIC_TYPE DecomposedSemiStructuredTopology
{
public:
    using Ordinal = SemiStructuredIndexer::Ordinal;
    using CellID = SemiStructuredIndexer::CellID;
    using FaceID = SemiStructuredIndexer::FaceID;
    using NodeID = SemiStructuredIndexer::NodeID;
    struct Band
    {
        size_t z_begin = 0, z_end = 0, view = 0, offset = 0, xy_size = 0;
        unsigned orientation = 0;
        bool owned = false;
        size_t count() const noexcept { return (z_end - z_begin) * xy_size; }
    };
    static constexpr size_t invalid_local_id() noexcept { return std::numeric_limits<size_t>::max(); }

    /** Construct from the shared source, its graph ownership and factory XY order.
     * Rank is z_slab*xy_parts+xy_part. Empty XY or Z ownership gives an empty
     * local container. All inputs are rank-local; collective validation belongs
     * to the MeshHandle construction boundary.
     */
    DecomposedSemiStructuredTopology(std::shared_ptr<const SemiStructMeshTopo> source,
        std::span<const size_t> xy_owners, std::span<const size_t> xy_order, size_t rank, size_t xy_parts,
        size_t z_parts, size_t halo_depth);
    const SemiStructMeshTopo& source_topology() const noexcept { return *d_source; }
    size_t num_owned_cells() const noexcept { return d_owned_cells; }
    size_t num_local_cells() const noexcept { return d_local_cells; }
    size_t num_owned_faces() const noexcept { return d_owned_faces; }
    size_t num_local_faces() const noexcept { return d_local_faces; }
    size_t num_local_nodes() const noexcept { return d_local_nodes; }
    CellID source_cell(size_t local) const;
    FaceID source_face(size_t local) const;
    NodeID source_node(size_t local) const;
    size_t cell_local_id(CellID source) const noexcept;
    size_t face_local_id(FaceID source) const noexcept;
    size_t node_local_id(NodeID source) const noexcept;
    size_t owner_cell(size_t face) const;
    size_t neighbor_cell(size_t face) const;
    int boundary_id(size_t face) const;
    EntityRange<size_t> cell_faces(size_t cell) const;
    std::span<const Ordinal> xy_cells() const noexcept { return d_xy_cells; }
    std::span<const Ordinal> xy_sides() const noexcept { return d_xy_sides; }
    std::span<const Ordinal> xy_nodes() const noexcept { return d_xy_nodes; }
    size_t xy_view_count() const noexcept { return d_views.size(); }
    size_t xy_view_size(size_t view) const { return d_views.at(view).positions.size(); }
    EntityRange<Ordinal> view_source_ids(size_t view) const;
    std::span<const Band> cell_bands() const noexcept { return d_cells; }
    std::span<const Band> face_bands() const noexcept { return d_faces; }
    std::span<const Band> node_bands() const noexcept { return d_nodes; }
    size_t storage_bytes() const noexcept;
    size_t lookup_storage_estimate() const noexcept;

private:
    enum class Kind : unsigned
    {
        Cell,
        Side,
        Node
    };
    struct View
    {
        Kind kind;
        size_t radius;
        unsigned ownership;
        std::vector<size_t> positions;
    };
    struct Group
    {
        size_t begin = 0, end = 0;
    };
    std::shared_ptr<const SemiStructMeshTopo> d_source;
    std::vector<Ordinal> d_xy_cells, d_xy_sides, d_xy_nodes;
    std::vector<size_t> d_cell_distance, d_side_distance, d_node_distance;
    std::vector<unsigned char> d_side_owned;
    std::unordered_map<Ordinal, size_t> d_cell_position, d_side_position, d_node_position;
    std::vector<View> d_views;
    std::vector<Band> d_cells, d_faces, d_nodes;
    std::array<Group, 2> d_cell_groups{};
    std::array<Group, 4> d_face_groups{};
    std::array<Group, 1> d_node_groups{};
    size_t d_owned_cells = 0, d_local_cells = 0, d_owned_faces = 0, d_local_faces = 0, d_local_nodes = 0;
    SIMPLEFLUID_LOCAL size_t view(Kind kind, size_t radius, unsigned ownership = 0);
    SIMPLEFLUID_LOCAL Ordinal source_entity(size_t view, size_t entry) const;
    SIMPLEFLUID_LOCAL size_t position_in_view(size_t view, Ordinal source) const noexcept;
    SIMPLEFLUID_LOCAL size_t lookup(const std::vector<Band>& bands, std::span<const Group> groups, Ordinal source,
        size_t z, unsigned orientation) const noexcept;
    SIMPLEFLUID_LOCAL static const Band& band_for_local(const std::vector<Band>& bands, size_t local);
};
} // namespace SimpleFluid::Meshes
