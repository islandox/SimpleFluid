/**
 * @file SemiStructMeshTopo.hh
 * @brief Topology for meshes formed by extruding a two-dimensional mesh.
 */

#pragma once

#include "dataclass/typedefs.hh"
#include "geometry/mesh/BoundaryFacePatch.hh"
#include "geometry/mesh/SemiStructuredIndexer.hh"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace SimpleFluid::Meshes
{

/**
 * @brief Connectivity and boundary patches for a layered semi-structured mesh.
 *
 * Base cells are supplied as consistently oriented node loops. Their edges
 * are deduplicated into side faces, then repeated through every axial layer.
 */
class SemiStructMeshTopo
{
public:
    using Indexer = SemiStructuredIndexer;
    using Ordinal = Indexer::Ordinal;
    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    using BoundaryPatch = BoundaryFacePatch<FaceID>;
    using BoundaryPatchMap = std::unordered_map<int, BoundaryPatch>;
    using BoundaryNames = std::unordered_map<int, std::string>;
    using CellPatch = std::vector<CellID>;
    using NeighborCells = std::vector<CellID>;

    static constexpr int invalid_boundary_id = -1;
    static constexpr Ordinal invalid_ordinal =
        static_cast<Ordinal>(-1);

    struct BoundaryEdge
    {
        Ordinal node0 = invalid_ordinal;
        Ordinal node1 = invalid_ordinal;
        std::string patch_name = "side";
    };

    struct SideFace
    {
        std::array<Ordinal, 2> nodes{};
        Ordinal owner = invalid_ordinal;
        Ordinal neighbor = invalid_ordinal;
        int boundary_id = invalid_boundary_id;
    };

    SemiStructMeshTopo() = default;
    SemiStructMeshTopo(
        Ordinal nodes_per_layer,
        const Arr<Arr<Ordinal>>& cell_nodes,
        Ordinal layers,
        const Arr<BoundaryEdge>& boundary_edges = {},
        bool axial_periodic = false);

    const Indexer& indexer() const noexcept { return d_indexer; }
    const Arr<SideFace>& side_faces() const noexcept
    {
        return d_side_faces;
    }
    const SideFace& side_face(Ordinal side_face_id) const
    {
        return d_side_faces.at(side_face_id);
    }
    const Arr<Ordinal>& cell_side_faces(Ordinal base_cell_id) const
    {
        return d_cell_side_faces.at(base_cell_id);
    }

    std::vector<FaceID> cell_faces(CellID cell_id) const;
    CellID owner_cell(FaceID face_id) const noexcept;
    CellID neighbor_cell(FaceID face_id) const noexcept;
    const CellPatch& interior_cell_patch() const noexcept
    {
        return d_interior_cell_patch;
    }
    NeighborCells neighbor_cells(CellID cell_id) const
    {
        const auto& axial_neighbors = d_axial_neighbors[cell_id.k];
        const auto& base_neighbors = d_base_neighbor_cells[cell_id.ij];

        NeighborCells result;
        result.reserve(axial_neighbors.num + base_neighbors.size());
        for (unsigned k = 0; k < axial_neighbors.num; ++k)
        {
            result.push_back(
                {cell_id.ij, axial_neighbors.indices[k]});
        }
        for (const auto base_neighbor : base_neighbors)
        {
            result.push_back({base_neighbor, cell_id.k});
        }
        return result;
    }

    bool is_boundary_face(FaceID face_id) const noexcept;
    int boundary_id(FaceID face_id) const noexcept;
    const std::string& boundary_patch_name(int patch_id) const;
    const BoundaryPatch& boundary_face_patch(int patch_id) const;
    const BoundaryPatchMap& boundary_patches() const noexcept
    {
        return d_boundary_patches;
    }
    std::vector<int> boundary_patch_ids() const;
    int num_boundary_patches() const noexcept;

private:
    struct FaceCells
    {
        Ordinal owner = invalid_ordinal;
        Ordinal neighbor = invalid_ordinal;
    };

    struct AxialNeighbors
    {
        unsigned num{};
        std::array<Ordinal, 2> indices{};
    };

    void build_base_topology(
        Ordinal nodes_per_layer,
        const Arr<Arr<Ordinal>>& cell_nodes,
        const Arr<BoundaryEdge>& boundary_edges);
    void initialize_face_adjacency();
    void initialize_cell_adjacency();
    void initialize_boundary_patches();

    Indexer d_indexer;
    Arr<SideFace> d_side_faces;
    Arr<Arr<Ordinal>> d_cell_side_faces;
    std::array<Arr<FaceCells>, 2> d_face_cells_per_orientation;
    CellPatch d_interior_cell_patch;
    Arr<Arr<Ordinal>> d_base_neighbor_cells;
    Arr<AxialNeighbors> d_axial_neighbors;
    BoundaryNames d_boundary_names;
    BoundaryPatchMap d_boundary_patches;
};

} // namespace SimpleFluid::Meshes
