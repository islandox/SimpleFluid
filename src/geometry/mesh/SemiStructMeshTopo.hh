/**
 * @file SemiStructMeshTopo.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Topology for meshes formed by extruding a two-dimensional mesh.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "dataclass/typedefs.hh"
#include "geometry/mesh/BoundaryFaceBatch.hh"
#include "geometry/mesh/EntityRange.hh"
#include "geometry/mesh/SemiStructuredIndexer.hh"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace SimpleFluid::Meshes
{

/**
 * @brief Connectivity and boundary batches for a layered semi-structured mesh.
 *
 * Base cells are supplied as consistently oriented node loops. Their edges
 * are deduplicated into side faces, then repeated through every axial layer.
 * `cell_side_faces()` follows each base cell's node-loop order, and both
 * ordinal accessors are bounds checked. `interior_cell_batch()` contains only
 * cells with a neighbor across every face. `neighbor_cells()` returns axial
 * neighbors before base-topology neighbors.
 */
class SemiStructMeshTopo
{
public:
    using Indexer = SemiStructuredIndexer;
    using Ordinal = Indexer::Ordinal;
    using CellID = Indexer::CellID;
    using FaceID = Indexer::FaceID;
    struct BoundaryBatch
    {
        int id = -1;
        EntityRange<FaceID> face_lids;
    };
    using BoundaryBatchMap = std::unordered_map<int, BoundaryBatch>;
    using BoundaryNames = std::unordered_map<int, std::string>;
    using CellBatch = std::vector<CellID>;
    using NeighborCells = std::vector<CellID>;

    static constexpr int invalid_boundary_id = -1;
    static constexpr Ordinal invalid_ordinal = static_cast<Ordinal>(-1);

    /** @brief Base-topology edge assigned to a named side boundary. */
    struct BoundaryEdge
    {
        Ordinal node0 = invalid_ordinal;
        Ordinal node1 = invalid_ordinal;
        std::string batch_name = "side";
    };

    /** @brief Base edge connectivity and adjacency for an extruded side face. */
    struct SideFace
    {
        std::array<Ordinal, 2> nodes{};
        Ordinal owner = invalid_ordinal;
        Ordinal neighbor = invalid_ordinal;
        int boundary_id = invalid_boundary_id;
    };

    SemiStructMeshTopo() = default;
    SemiStructMeshTopo(Ordinal nodes_per_layer, const Arr<Arr<Ordinal>>& cell_nodes, Ordinal layers,
        const Arr<BoundaryEdge>& boundary_edges = {}, bool axial_periodic = false);

    size_t storage_bytes() const noexcept;
    size_t base_storage_bytes() const noexcept;
    const void* base_storage_identity() const noexcept { return d_base.get(); }
    /** Arithmetic layer view sharing immutable base connectivity. */
    SemiStructMeshTopo with_layers(Ordinal layers) const;
    const Indexer& indexer() const noexcept { return d_indexer; }
    const Arr<SideFace>& side_faces() const noexcept { return d_base->side_faces; }
    const SideFace& side_face(Ordinal side_face_id) const { return d_base->side_faces.at(side_face_id); }
    const Arr<Ordinal>& cell_side_faces(Ordinal base_cell_id) const { return d_base->cell_side_faces.at(base_cell_id); }

    EntityRange<FaceID> cell_faces(CellID cell_id) const;
    CellID owner_cell(FaceID face_id) const noexcept;
    CellID neighbor_cell(FaceID face_id) const noexcept;
    EntityRange<CellID> interior_cell_batch() const;
    NeighborCells neighbor_cells(CellID cell_id) const
    {
        const auto& base_neighbors = d_base->base_neighbor_cells[cell_id.ij];
        NeighborCells result;
        result.reserve(2 + base_neighbors.size());
        if (d_indexer.axial_periodic)
        {
            result.push_back({cell_id.ij, cell_id.k ? cell_id.k - 1 : d_indexer.num_layers - 1});
            result.push_back({cell_id.ij, (cell_id.k + 1) % d_indexer.num_layers});
        }
        else
        {
            if (cell_id.k)
                result.push_back({cell_id.ij, cell_id.k - 1});
            if (cell_id.k + 1 < d_indexer.num_layers)
                result.push_back({cell_id.ij, cell_id.k + 1});
        }
        for (const auto base_neighbor : base_neighbors)
        {
            result.push_back({base_neighbor, cell_id.k});
        }
        return result;
    }

    bool is_boundary_face(FaceID face_id) const noexcept;
    int boundary_id(FaceID face_id) const noexcept;
    const std::string& boundary_batch_name(int batch_id) const;
    BoundaryBatch boundary_face_batch(int batch_id) const;
    BoundaryBatchMap boundary_batches() const
    {
        BoundaryBatchMap result;
        for (const auto& [id, faces] : d_base->boundary_base_faces)
            result.emplace(id, boundary_face_batch(id));
        return result;
    }
    std::vector<int> boundary_batch_ids() const;
    int num_boundary_batches() const noexcept;

private:
    /** @brief Owner and neighbor ordinals for one face. */
    struct FaceCells
    {
        Ordinal owner = invalid_ordinal;
        Ordinal neighbor = invalid_ordinal;
    };

    /** @brief Neighbor layer indices available for one axial layer. */
    struct AxialNeighbors
    {
        unsigned num{};
        std::array<Ordinal, 2> indices{};
    };

    void build_base_topology(
        Ordinal nodes_per_layer, const Arr<Arr<Ordinal>>& cell_nodes, const Arr<BoundaryEdge>& boundary_edges);
    void initialize_face_adjacency();
    void initialize_cell_adjacency();
    void initialize_boundary_batches();

    struct BaseData
    {
        Arr<SideFace> side_faces;
        Arr<Arr<Ordinal>> cell_side_faces;
        Arr<Ordinal> interior_base_cells;
        Arr<Arr<Ordinal>> base_neighbor_cells;
        BoundaryNames boundary_names;
        std::unordered_map<int, Arr<Ordinal>> boundary_base_faces;
    };
    Indexer d_indexer;
    std::shared_ptr<BaseData> d_base = std::make_shared<BaseData>();
};

} // namespace SimpleFluid::Meshes
