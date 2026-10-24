#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/mesh/EntityRange.hh"
#include "geometry/mesh/OrthoMeshTopo.hh"
#include <array>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes
{
/** Compact rectangular ownership and exact face-adjacency halos. Source geometry
 * remains shared; storage contains disjoint logical boxes, never entity ID maps.
 * Owned IDs are I-fastest (faces first by orientation). Ghosts and nodes
 * concatenate deterministic disjoint boxes, with I-fastest IDs inside each box.
 */
class SIMPLEFLUID_PUBLIC_TYPE DecomposedOrthogonalTopology
{
public:
    using Indexer = OrthogonalIndexer;
    using Ordinal = Indexer::Ordinal;
    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    using NodeID = Indexer::NodeID;
    using Coordinates = std::array<size_t, 3>;
    struct Box
    {
        Coordinates begin{}, end{};
        size_t offset = 0;
        size_t count() const noexcept { return (end[0] - begin[0]) * (end[1] - begin[1]) * (end[2] - begin[2]); }
    };
    static constexpr size_t invalid_local_id() noexcept { return std::numeric_limits<size_t>::max(); }
    /** Half-open owned bounds; any empty axis yields an empty view. Halo zero
     * retains owned cells and their complete incident face/node sets.
     */
    DecomposedOrthogonalTopology(
        std::shared_ptr<const OrthoMeshTopo> source, Coordinates owned_begin, Coordinates owned_end, size_t halo_depth);
    const OrthoMeshTopo& source_topology() const noexcept { return *d_source; }
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
    /** Disjoint coverage boxes. Ghost/node offsets define local concatenation
     * order; owned face boxes describe coverage only, since owned faces retain
     * orientation-grouped I-fastest order across periodic seam splits.
     */
    std::span<const Box> cell_boxes(bool owned) const noexcept { return d_cells[owned ? 0 : 1]; }
    std::span<const Box> face_boxes(unsigned orientation, bool owned) const
    {
        return d_faces.at(owned ? 0 : 1).at(orientation);
    }
    std::span<const Box> node_boxes() const noexcept { return d_nodes; }
    size_t storage_bytes() const noexcept;
    size_t lookup_storage_estimate() const noexcept { return 0; }

private:
    using Boxes = std::vector<Box>;
    std::shared_ptr<const OrthoMeshTopo> d_source;
    std::array<Boxes, 2> d_cells;
    std::array<std::array<Boxes, 3>, 2> d_faces;
    Boxes d_nodes;
    Coordinates d_owned_begin{}, d_owned_end{};
    std::array<Coordinates, 3> d_owned_face_shapes{};
    std::array<std::array<size_t, 4>, 2> d_face_offsets{};
    size_t d_owned_cells = 0, d_local_cells = 0, d_owned_faces = 0, d_local_faces = 0, d_local_nodes = 0;
    SIMPLEFLUID_LOCAL static Boxes subtract(Box box, const Box& remove);
    SIMPLEFLUID_LOCAL static void insert(Boxes& boxes, Box box);
    SIMPLEFLUID_LOCAL static size_t count(const Boxes& boxes) noexcept;
    SIMPLEFLUID_LOCAL static void finalize(Boxes& boxes) noexcept;
    SIMPLEFLUID_LOCAL static size_t flatten(Coordinates p, Coordinates n) noexcept;
    SIMPLEFLUID_LOCAL static Coordinates expand(size_t id, Coordinates n) noexcept;
    SIMPLEFLUID_LOCAL size_t owned_face_local_id(FaceID source) const noexcept;
    SIMPLEFLUID_LOCAL FaceID owned_source_face(size_t local) const;
    SIMPLEFLUID_LOCAL static size_t lookup(const Boxes& boxes, Coordinates p) noexcept;
    SIMPLEFLUID_LOCAL static Coordinates select(const Boxes& boxes, size_t local);
    SIMPLEFLUID_LOCAL void add_wrapped(Boxes& boxes, Box box, Coordinates shape) const;
    SIMPLEFLUID_LOCAL Coordinates shape(unsigned kind, unsigned orientation = 0) const noexcept;
};
} // namespace SimpleFluid::Meshes
