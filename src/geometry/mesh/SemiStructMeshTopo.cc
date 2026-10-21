/**
 * @file SemiStructMeshTopo.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Layered semi-structured mesh topology implementation.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/SemiStructMeshTopo.hh"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace SimpleFluid::Meshes
{
namespace
{

using EdgeKey = std::array<SemiStructMeshTopo::Ordinal, 2>;

/** @brief Hash an ordered base-topology edge key. */
struct EdgeKeyHash
{
    size_t operator()(const EdgeKey& key) const noexcept
    {
        return static_cast<size_t>(key[0]) ^ (static_cast<size_t>(key[1]) << 1);
    }
};

/**
 * @brief Create a normalized (min, max) edge key from two node ordinals.
 *
 * @param node0 First node ordinal.
 * @param node1 Second node ordinal.
 * @return EdgeKey with the smaller ordinal first.
 */
EdgeKey edge_key(SemiStructMeshTopo::Ordinal node0, SemiStructMeshTopo::Ordinal node1)
{
    return node0 < node1 ? EdgeKey{node0, node1} : EdgeKey{node1, node0};
}

} // namespace

/**
 * @brief Construct a semi-structured topology from base nodes, cells, layers, and boundary edges.
 *
 * @param nodes_per_layer Number of nodes per XY layer.
 * @param cell_nodes Node-ordered cell connectivity for a single layer.
 * @param layers Number of axial extrusion layers.
 * @param boundary_edges Side-face boundary edge definitions.
 * @param axial_periodic Whether the axial direction wraps periodically.
 * @throws std::invalid_argument if any size parameter is zero.
 * @throws std::overflow_error if cell or face counts exceed the ordinal type.
 */
SemiStructMeshTopo::SemiStructMeshTopo(Ordinal nodes_per_layer, const Arr<Arr<Ordinal>>& cell_nodes, Ordinal layers,
    const Arr<BoundaryEdge>& boundary_edges, bool axial_periodic)
{
    if (nodes_per_layer == 0)
    {
        throw std::invalid_argument("Semi-structured topology requires at least one base node.");
    }
    if (cell_nodes.empty())
    {
        throw std::invalid_argument("Semi-structured topology requires at least one base cell.");
    }
    if (layers == 0)
    {
        throw std::invalid_argument("Semi-structured topology requires at least one layer.");
    }
    if (cell_nodes.size() > static_cast<size_t>(std::numeric_limits<Ordinal>::max()))
    {
        throw std::overflow_error("Semi-structured base-cell count exceeds its ID type.");
    }

    build_base_topology(nodes_per_layer, cell_nodes, boundary_edges);
    if (d_base->side_faces.size() > static_cast<size_t>(std::numeric_limits<Ordinal>::max()))
    {
        throw std::overflow_error("Semi-structured side-face count exceeds its ID type.");
    }

    d_indexer = Indexer(static_cast<Ordinal>(cell_nodes.size()), static_cast<Ordinal>(d_base->side_faces.size()),
        nodes_per_layer, layers, axial_periodic);
    initialize_face_adjacency();
    initialize_cell_adjacency();
    initialize_boundary_batches();
}

/**
 * @brief Return all faces bounding a cell.
 *
 * Returns the lower axial face, upper axial face (or wrapped periodic
 * equivalent), and the side faces inherited from the base 2D topology.
 *
 * @param id Cell identifier.
 * @return Independent indexed view of all bounding face IDs.
 */
EntityRange<SemiStructMeshTopo::FaceID> SemiStructMeshTopo::cell_faces(CellID id) const
{
    return {this, d_indexer.cell_ordinal(id), d_base->cell_side_faces.at(id.ij).size() + 2,
        [](const void* source, size_t cell, size_t face) -> FaceID
        {
            const auto& topo = *static_cast<const SemiStructMeshTopo*>(source);
            const auto id = topo.indexer().cell_id(cell);
            if (face == 0)
                return {id.ij, id.k, Indexer::AXIAL};
            if (face == 1)
            {
                const auto upper = topo.indexer().axial_periodic ? (id.k + 1) % topo.indexer().num_layers : id.k + 1;
                return {id.ij, upper, Indexer::AXIAL};
            }
            return {topo.cell_side_faces(id.ij)[face - 2], id.k, Indexer::SIDE};
        }};
}

EntityRange<SemiStructMeshTopo::CellID> SemiStructMeshTopo::interior_cell_batch() const
{
    const auto layers =
        d_indexer.axial_periodic ? d_indexer.num_layers : (d_indexer.num_layers > 2 ? d_indexer.num_layers - 2 : 0);
    return {this, 0, d_base->interior_base_cells.size() * layers,
        [](const void* source, size_t, size_t ordinal) -> CellID
        {
            const auto& topo = *static_cast<const SemiStructMeshTopo*>(source);
            const auto base_count = topo.d_base->interior_base_cells.size();
            return {topo.d_base->interior_base_cells[ordinal % base_count],
                static_cast<Ordinal>(ordinal / base_count + (topo.indexer().axial_periodic ? 0 : 1))};
        }};
}

/**
 * @brief Return the owner cell of a face.
 *
 * For axial faces the owner is the cell on the lower-k side. For side faces
 * the orientation follows the base-topology owner.
 *
 * @param id Face identifier.
 * @return CellID of the owning cell.
 */
auto SemiStructMeshTopo::owner_cell(FaceID id) const noexcept -> CellID
{
    if (id.orientation == Indexer::SIDE)
        return {d_base->side_faces[id.ij].owner, id.k};
    const auto owner = id.k ? id.k - 1 : (d_indexer.axial_periodic ? d_indexer.num_layers - 1 : 0);
    return {id.ij, owner};
}

/**
 * @brief Return the neighbor cell of an interior face, or an invalid cell for boundaries.
 *
 * @param id Face identifier.
 * @return CellID of the neighboring cell, or a default CellID if the face is on a boundary.
 */
auto SemiStructMeshTopo::neighbor_cell(FaceID id) const noexcept -> CellID
{
    if (id.orientation == Indexer::SIDE)
    {
        auto neighbor = d_base->side_faces[id.ij].neighbor;
        return neighbor == invalid_ordinal ? CellID{} : CellID{neighbor, id.k};
    }
    if (!d_indexer.axial_periodic && (id.k == 0 || id.k == d_indexer.num_layers))
        return {};
    return {id.ij, id.k};
}

/**
 * @brief Return true if the face lies on a domain boundary.
 *
 * @param face_id Face identifier.
 * @return true if the face has a boundary batch ID.
 */
bool SemiStructMeshTopo::is_boundary_face(FaceID face_id) const noexcept
{
    return boundary_id(face_id) != invalid_boundary_id;
}

/**
 * @brief Return the boundary batch ID for a face, or invalid_boundary_id for interior faces.
 *
 * @param id Face identifier.
 * @return Boundary batch ID or invalid_boundary_id.
 */
int SemiStructMeshTopo::boundary_id(FaceID id) const noexcept
{
    if (id.orientation == Indexer::AXIAL)
    {
        if (d_indexer.axial_periodic)
        {
            return invalid_boundary_id;
        }
        if (id.k == 0)
        {
            return 0;
        }
        if (id.k == d_indexer.num_layers)
        {
            return 1;
        }
        return invalid_boundary_id;
    }
    return d_base->side_faces[id.ij].boundary_id;
}

/**
 * @brief Return the human-readable name of a boundary batch.
 *
 * @param batch_id Boundary batch ID.
 * @return Const reference to the batch name string.
 * @throws std::out_of_range if @p batch_id is not registered.
 */
const std::string& SemiStructMeshTopo::boundary_batch_name(int batch_id) const
{
    const auto iter = d_base->boundary_names.find(batch_id);
    if (iter == d_base->boundary_names.end())
    {
        throw std::out_of_range("Requested boundary batch is not found.");
    }
    return iter->second;
}

/**
 * @brief Return the face list for a boundary batch.
 *
 * @param batch_id Boundary batch ID.
 * @return Boundary batch with an independent base/layer face view.
 * @throws std::out_of_range if @p batch_id is not registered.
 */
SemiStructMeshTopo::BoundaryBatch SemiStructMeshTopo::boundary_face_batch(int batch_id) const
{
    const auto& base = d_base->boundary_base_faces.at(batch_id);
    const size_t count = batch_id < 2 ? d_indexer.num_cells_per_layer : base.size() * d_indexer.num_layers;
    return {batch_id,
        {this, static_cast<size_t>(batch_id), count, [](const void* source, size_t batch, size_t i) -> FaceID
            {
                const auto& t = *static_cast<const SemiStructMeshTopo*>(source);
                if (batch < 2)
                    return {static_cast<Ordinal>(i), batch == 0 ? 0U : t.indexer().num_layers, Indexer::AXIAL};
                const auto& base = t.d_base->boundary_base_faces.at(static_cast<int>(batch));
                return {
                    base[i / t.indexer().num_layers], static_cast<Ordinal>(i % t.indexer().num_layers), Indexer::SIDE};
            }}};
}

/**
 * @brief Return a vector of all registered boundary batch IDs.
 *
 * @return Vector of batch IDs in no particular order.
 */
std::vector<int> SemiStructMeshTopo::boundary_batch_ids() const
{
    std::vector<int> ids;
    ids.reserve(d_base->boundary_base_faces.size());
    for (const auto& [id, batch] : d_base->boundary_base_faces)
    {
        (void) batch;
        ids.push_back(id);
    }
    return ids;
}

/**
 * @brief Return the number of registered boundary batches.
 *
 * @return Count of boundary batches.
 */
int SemiStructMeshTopo::num_boundary_batches() const noexcept
{
    return static_cast<int>(d_base->boundary_base_faces.size());
}

/** @brief Precompute owner and neighbor cells for axial and side faces. */
void SemiStructMeshTopo::initialize_face_adjacency() {}

/** @brief Precompute base and axial cell neighbors and interior cells. */
void SemiStructMeshTopo::initialize_cell_adjacency()
{
    d_base->base_neighbor_cells.resize(d_indexer.num_cells_per_layer);
    for (Ordinal cell = 0; cell < d_indexer.num_cells_per_layer; ++cell)
    {
        const auto& side_faces = d_base->cell_side_faces[cell];
        auto& neighbors = d_base->base_neighbor_cells[cell];
        neighbors.reserve(side_faces.size());
        for (const auto side_face_id : side_faces)
        {
            const auto& side_face = d_base->side_faces[side_face_id];
            if (cell == side_face.owner)
            {
                if (side_face.neighbor != invalid_ordinal)
                {
                    neighbors.push_back(side_face.neighbor);
                }
            }
            else
            {
                neighbors.push_back(side_face.owner);
            }
        }
    }

    for (Ordinal cell = 0; cell < d_indexer.num_cells_per_layer; ++cell)
    {
        if (d_base->base_neighbor_cells[cell].size() == d_base->cell_side_faces[cell].size())
            d_base->interior_base_cells.push_back(cell);
    }
}

/**
 * @brief Derive unique base edges, adjacency, and side-boundary IDs.
 * @param nodes_per_layer Number of nodes in the base topology.
 * @param cell_nodes Counter-clockwise node loop for each base cell.
 * @param boundary_edges Optional named exterior base edges.
 * @throws std::invalid_argument If connectivity, orientation, manifoldness,
 *         or boundary tags are invalid.
 */
void SemiStructMeshTopo::build_base_topology(
    Ordinal nodes_per_layer, const Arr<Arr<Ordinal>>& cell_nodes, const Arr<BoundaryEdge>& boundary_edges)
{
    std::unordered_map<EdgeKey, std::string, EdgeKeyHash> boundary_tags;
    for (const auto& boundary : boundary_edges)
    {
        if (boundary.node0 >= nodes_per_layer || boundary.node1 >= nodes_per_layer || boundary.node0 == boundary.node1)
        {
            throw std::invalid_argument("Semi-structured boundary edge has invalid node IDs.");
        }
        if (boundary.batch_name.empty() || boundary.batch_name == "zmin" || boundary.batch_name == "zmax")
        {
            throw std::invalid_argument("Semi-structured boundary edge has an invalid batch name.");
        }
        if (!boundary_tags.emplace(edge_key(boundary.node0, boundary.node1), boundary.batch_name).second)
        {
            throw std::invalid_argument("Semi-structured boundary edge is specified more than once.");
        }
    }

    d_base->cell_side_faces.resize(cell_nodes.size());
    std::unordered_map<EdgeKey, Ordinal, EdgeKeyHash> side_face_ids;
    for (Ordinal cell = 0; cell < cell_nodes.size(); ++cell)
    {
        const auto& nodes = cell_nodes[cell];
        if (nodes.size() < 3)
        {
            throw std::invalid_argument("Semi-structured base cells require at least three nodes.");
        }

        std::unordered_set<Ordinal> unique_nodes;
        auto& cell_side_faces = d_base->cell_side_faces[cell];
        cell_side_faces.reserve(nodes.size());
        for (size_t side = 0; side < nodes.size(); ++side)
        {
            const auto node0 = nodes[side];
            const auto node1 = nodes[(side + 1) % nodes.size()];
            if (node0 >= nodes_per_layer || node1 >= nodes_per_layer || node0 == node1 ||
                !unique_nodes.insert(node0).second)
            {
                throw std::invalid_argument("Semi-structured base cell has invalid node "
                                            "connectivity.");
            }

            const auto key = edge_key(node0, node1);
            const auto existing = side_face_ids.find(key);
            if (existing == side_face_ids.end())
            {
                const auto side_face = static_cast<Ordinal>(d_base->side_faces.size());
                side_face_ids.emplace(key, side_face);
                d_base->side_faces.push_back({{{node0, node1}}, cell, invalid_ordinal, invalid_boundary_id});
                cell_side_faces.push_back(side_face);
                continue;
            }

            auto& side_face = d_base->side_faces[existing->second];
            if (side_face.neighbor != invalid_ordinal || side_face.nodes[0] != node1 || side_face.nodes[1] != node0)
            {
                throw std::invalid_argument("Semi-structured base topology is non-manifold or "
                                            "has inconsistent cell orientation.");
            }
            side_face.neighbor = cell;
            cell_side_faces.push_back(existing->second);
        }
    }

    d_base->boundary_names.emplace(0, "zmin");
    d_base->boundary_names.emplace(1, "zmax");
    std::unordered_map<std::string, int> boundary_ids{{"zmin", 0}, {"zmax", 1}};
    int next_boundary_id = 2;
    std::unordered_set<EdgeKey, EdgeKeyHash> used_boundary_tags;

    for (auto& side_face : d_base->side_faces)
    {
        const auto key = edge_key(side_face.nodes[0], side_face.nodes[1]);
        const auto tag = boundary_tags.find(key);
        if (side_face.neighbor != invalid_ordinal)
        {
            if (tag != boundary_tags.end())
            {
                throw std::invalid_argument("Semi-structured boundary tag refers to an "
                                            "interior edge.");
            }
            continue;
        }

        const std::string batch_name = tag == boundary_tags.end() ? "side" : tag->second;
        if (tag != boundary_tags.end())
        {
            used_boundary_tags.insert(key);
        }

        const auto existing_id = boundary_ids.find(batch_name);
        if (existing_id != boundary_ids.end())
        {
            side_face.boundary_id = existing_id->second;
            continue;
        }

        side_face.boundary_id = next_boundary_id;
        boundary_ids.emplace(batch_name, next_boundary_id);
        d_base->boundary_names.emplace(next_boundary_id, batch_name);
        ++next_boundary_id;
    }

    if (used_boundary_tags.size() != boundary_tags.size())
    {
        throw std::invalid_argument("Semi-structured boundary tag does not match an exterior "
                                    "edge.");
    }
}

/** @brief Materialize axial and side boundary-face batches for all layers. */
void SemiStructMeshTopo::initialize_boundary_batches()
{
    for (const auto& [id, name] : d_base->boundary_names)
        if (!d_indexer.axial_periodic || id > 1)
            d_base->boundary_base_faces.emplace(id, Arr<Ordinal>{});
    for (Ordinal f = 0; f < d_base->side_faces.size(); ++f)
        if (d_base->side_faces[f].boundary_id != invalid_boundary_id)
            d_base->boundary_base_faces.at(d_base->side_faces[f].boundary_id).push_back(f);
}

SemiStructMeshTopo SemiStructMeshTopo::with_layers(Ordinal layers) const
{
    if (!layers)
        throw std::invalid_argument("Shared extrusion layer view must be nonempty.");
    auto result = *this;
    result.d_indexer = Indexer(d_indexer.num_cells_per_layer, d_indexer.num_side_faces_per_layer,
        d_indexer.num_nodes_per_layer, layers, d_indexer.axial_periodic);
    return result;
}
size_t SemiStructMeshTopo::base_storage_bytes() const noexcept
{
    size_t bytes = sizeof(BaseData) + d_base->side_faces.capacity() * sizeof(SideFace) +
                   d_base->cell_side_faces.capacity() * sizeof(Arr<Ordinal>) +
                   d_base->base_neighbor_cells.capacity() * sizeof(Arr<Ordinal>) +
                   d_base->interior_base_cells.capacity() * sizeof(Ordinal);
    for (const auto& row : d_base->cell_side_faces)
        bytes += row.capacity() * sizeof(Ordinal);
    for (const auto& row : d_base->base_neighbor_cells)
        bytes += row.capacity() * sizeof(Ordinal);
    for (const auto& [id, faces] : d_base->boundary_base_faces)
        bytes += sizeof(decltype(d_base->boundary_base_faces)::value_type) + 4 * sizeof(void*) +
                 faces.capacity() * sizeof(Ordinal);
    for (const auto& [id, name] : d_base->boundary_names)
        bytes += sizeof(decltype(d_base->boundary_names)::value_type) + 4 * sizeof(void*) + name.capacity();
    return bytes;
}
size_t SemiStructMeshTopo::storage_bytes() const noexcept
{
    return sizeof(*this) + base_storage_bytes();
}
} // namespace SimpleFluid::Meshes
