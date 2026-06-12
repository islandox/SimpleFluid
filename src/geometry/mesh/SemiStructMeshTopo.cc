/**
 * @file SemiStructMeshTopo.cc
 * @brief Layered semi-structured mesh topology implementation.
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

struct EdgeKeyHash
{
    size_t operator()(const EdgeKey& key) const noexcept
    {
        return static_cast<size_t>(key[0])
             ^ (static_cast<size_t>(key[1]) << 1);
    }
};

EdgeKey edge_key(
    SemiStructMeshTopo::Ordinal node0,
    SemiStructMeshTopo::Ordinal node1)
{
    return node0 < node1
         ? EdgeKey{node0, node1}
         : EdgeKey{node1, node0};
}

} // namespace

SemiStructMeshTopo::SemiStructMeshTopo(
    Ordinal nodes_per_layer,
    const Arr<Arr<Ordinal>>& cell_nodes,
    Ordinal layers,
    const Arr<BoundaryEdge>& boundary_edges,
    bool axial_periodic)
{
    if (nodes_per_layer == 0)
    {
        throw std::invalid_argument(
            "Semi-structured topology requires at least one base node.");
    }
    if (cell_nodes.empty())
    {
        throw std::invalid_argument(
            "Semi-structured topology requires at least one base cell.");
    }
    if (layers == 0)
    {
        throw std::invalid_argument(
            "Semi-structured topology requires at least one layer.");
    }
    if (cell_nodes.size()
        > static_cast<size_t>(std::numeric_limits<Ordinal>::max()))
    {
        throw std::overflow_error(
            "Semi-structured base-cell count exceeds its ID type.");
    }

    build_base_topology(nodes_per_layer, cell_nodes, boundary_edges);
    if (d_side_faces.size()
        > static_cast<size_t>(std::numeric_limits<Ordinal>::max()))
    {
        throw std::overflow_error(
            "Semi-structured side-face count exceeds its ID type.");
    }

    d_indexer = Indexer(
        static_cast<Ordinal>(cell_nodes.size()),
        static_cast<Ordinal>(d_side_faces.size()),
        nodes_per_layer,
        layers,
        axial_periodic);
    initialize_face_adjacency();
    initialize_cell_adjacency();
    initialize_boundary_batches();
}

std::vector<SemiStructMeshTopo::FaceID>
SemiStructMeshTopo::cell_faces(CellID id) const
{
    const auto& side_faces = d_cell_side_faces[id.ij];
    std::vector<FaceID> faces;
    faces.reserve(side_faces.size() + 2);

    const auto upper_k =
        d_indexer.axial_periodic && id.k + 1 == d_indexer.num_layers
      ? 0
      : id.k + 1;
    faces.push_back({id.ij, id.k, Indexer::AXIAL});
    faces.push_back({id.ij, upper_k, Indexer::AXIAL});
    for (const auto side_face : side_faces)
    {
        faces.push_back({side_face, id.k, Indexer::SIDE});
    }
    return faces;
}

auto SemiStructMeshTopo::owner_cell(FaceID id) const noexcept -> CellID
{
    const auto coordinate =
        id.orientation == Indexer::AXIAL ? id.k : id.ij;
    const auto owner =
        d_face_cells_per_orientation[id.orientation][coordinate].owner;
    return id.orientation == Indexer::AXIAL
         ? CellID{id.ij, owner}
         : CellID{owner, id.k};
}

auto SemiStructMeshTopo::neighbor_cell(FaceID id) const noexcept -> CellID
{
    const auto coordinate =
        id.orientation == Indexer::AXIAL ? id.k : id.ij;
    const auto neighbor =
        d_face_cells_per_orientation[id.orientation][coordinate].neighbor;
    return neighbor == invalid_ordinal
         ? CellID{}
         : id.orientation == Indexer::AXIAL
             ? CellID{id.ij, neighbor}
             : CellID{neighbor, id.k};
}

bool SemiStructMeshTopo::is_boundary_face(FaceID face_id) const noexcept
{
    return boundary_id(face_id) != invalid_boundary_id;
}

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
    return d_side_faces[id.ij].boundary_id;
}

const std::string&
SemiStructMeshTopo::boundary_batch_name(int batch_id) const
{
    const auto iter = d_boundary_names.find(batch_id);
    if (iter == d_boundary_names.end())
    {
        throw std::out_of_range("Requested boundary batch is not found.");
    }
    return iter->second;
}

const SemiStructMeshTopo::BoundaryBatch&
SemiStructMeshTopo::boundary_face_batch(int batch_id) const
{
    const auto iter = d_boundary_batches.find(batch_id);
    if (iter == d_boundary_batches.end())
    {
        throw std::out_of_range("Requested boundary batch is not found.");
    }
    return iter->second;
}

std::vector<int> SemiStructMeshTopo::boundary_batch_ids() const
{
    std::vector<int> ids;
    ids.reserve(d_boundary_batches.size());
    for (const auto& [id, batch] : d_boundary_batches)
    {
        (void)batch;
        ids.push_back(id);
    }
    return ids;
}

int SemiStructMeshTopo::num_boundary_batches() const noexcept
{
    return static_cast<int>(d_boundary_batches.size());
}

void SemiStructMeshTopo::initialize_face_adjacency()
{
    auto& axial_faces =
        d_face_cells_per_orientation[Indexer::AXIAL];
    axial_faces.resize(d_indexer.num_node_layers);
    if (d_indexer.axial_periodic)
    {
        for (Ordinal face = 0;
             face < d_indexer.num_layers;
             ++face)
        {
            axial_faces[face] = {
                face == 0 ? d_indexer.num_layers - 1 : face - 1,
                face};
        }
    }
    else
    {
        axial_faces[0] = {0, invalid_ordinal};
        for (Ordinal face = 1;
             face < d_indexer.num_layers;
             ++face)
        {
            axial_faces[face] = {face - 1, face};
        }
        axial_faces[d_indexer.num_layers] = {
            d_indexer.num_layers - 1,
            invalid_ordinal};
    }

    auto& side_faces =
        d_face_cells_per_orientation[Indexer::SIDE];
    side_faces.reserve(d_side_faces.size());
    for (const auto& side_face : d_side_faces)
    {
        side_faces.push_back(
            {side_face.owner, side_face.neighbor});
    }
}

void SemiStructMeshTopo::initialize_cell_adjacency()
{
    d_base_neighbor_cells.resize(d_indexer.num_cells_per_layer);
    for (Ordinal cell = 0;
         cell < d_indexer.num_cells_per_layer;
         ++cell)
    {
        const auto& side_faces = d_cell_side_faces[cell];
        auto& neighbors = d_base_neighbor_cells[cell];
        neighbors.reserve(side_faces.size());
        for (const auto side_face_id : side_faces)
        {
            const auto& side_face = d_side_faces[side_face_id];
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

    const auto layers = d_indexer.num_layers;
    d_axial_neighbors.resize(layers);
    if (layers == 1)
    {
        if (d_indexer.axial_periodic)
        {
            d_axial_neighbors[0] = {2, {0, 0}};
        }
    }
    else
    {
        for (Ordinal layer = 1; layer < layers - 1; ++layer)
        {
            d_axial_neighbors[layer] =
                {2, {layer - 1, layer + 1}};
        }
        if (d_indexer.axial_periodic)
        {
            d_axial_neighbors[0] = {2, {layers - 1, 1}};
            d_axial_neighbors[layers - 1] =
                {2, {layers - 2, 0}};
        }
        else
        {
            d_axial_neighbors[0] = {1, {1}};
            d_axial_neighbors[layers - 1] =
                {1, {layers - 2}};
        }
    }

    d_interior_cell_batch.reserve(d_indexer.total_cells());
    for (Ordinal layer = 0; layer < layers; ++layer)
    {
        if (d_axial_neighbors[layer].num != 2)
        {
            continue;
        }
        for (Ordinal cell = 0;
             cell < d_indexer.num_cells_per_layer;
             ++cell)
        {
            if (d_base_neighbor_cells[cell].size()
                == d_cell_side_faces[cell].size())
            {
                d_interior_cell_batch.push_back({cell, layer});
            }
        }
    }
}

void SemiStructMeshTopo::build_base_topology(
    Ordinal nodes_per_layer,
    const Arr<Arr<Ordinal>>& cell_nodes,
    const Arr<BoundaryEdge>& boundary_edges)
{
    std::unordered_map<EdgeKey, std::string, EdgeKeyHash> boundary_tags;
    for (const auto& boundary : boundary_edges)
    {
        if (boundary.node0 >= nodes_per_layer
            || boundary.node1 >= nodes_per_layer
            || boundary.node0 == boundary.node1)
        {
            throw std::invalid_argument(
                "Semi-structured boundary edge has invalid node IDs.");
        }
        if (boundary.patch_name.empty()
            || boundary.patch_name == "zmin"
            || boundary.patch_name == "zmax")
        {
            throw std::invalid_argument(
                "Semi-structured boundary edge has an invalid patch name.");
        }
        if (!boundary_tags.emplace(
                edge_key(boundary.node0, boundary.node1),
                boundary.patch_name).second)
        {
            throw std::invalid_argument(
                "Semi-structured boundary edge is specified more than once.");
        }
    }

    d_cell_side_faces.resize(cell_nodes.size());
    std::unordered_map<EdgeKey, Ordinal, EdgeKeyHash> side_face_ids;
    for (Ordinal cell = 0; cell < cell_nodes.size(); ++cell)
    {
        const auto& nodes = cell_nodes[cell];
        if (nodes.size() < 3)
        {
            throw std::invalid_argument(
                "Semi-structured base cells require at least three nodes.");
        }

        std::unordered_set<Ordinal> unique_nodes;
        auto& cell_side_faces = d_cell_side_faces[cell];
        cell_side_faces.reserve(nodes.size());
        for (size_t side = 0; side < nodes.size(); ++side)
        {
            const auto node0 = nodes[side];
            const auto node1 = nodes[(side + 1) % nodes.size()];
            if (node0 >= nodes_per_layer
                || node1 >= nodes_per_layer
                || node0 == node1
                || !unique_nodes.insert(node0).second)
            {
                throw std::invalid_argument(
                    "Semi-structured base cell has invalid node "
                    "connectivity.");
            }

            const auto key = edge_key(node0, node1);
            const auto existing = side_face_ids.find(key);
            if (existing == side_face_ids.end())
            {
                const auto side_face =
                    static_cast<Ordinal>(d_side_faces.size());
                side_face_ids.emplace(key, side_face);
                d_side_faces.push_back(
                    {{{node0, node1}}, cell, invalid_ordinal,
                     invalid_boundary_id});
                cell_side_faces.push_back(side_face);
                continue;
            }

            auto& side_face = d_side_faces[existing->second];
            if (side_face.neighbor != invalid_ordinal
                || side_face.nodes[0] != node1
                || side_face.nodes[1] != node0)
            {
                throw std::invalid_argument(
                    "Semi-structured base topology is non-manifold or "
                    "has inconsistent cell orientation.");
            }
            side_face.neighbor = cell;
            cell_side_faces.push_back(existing->second);
        }
    }

    d_boundary_names.emplace(0, "zmin");
    d_boundary_names.emplace(1, "zmax");
    std::unordered_map<std::string, int> boundary_ids{
        {"zmin", 0},
        {"zmax", 1}};
    int next_boundary_id = 2;
    std::unordered_set<EdgeKey, EdgeKeyHash> used_boundary_tags;

    for (auto& side_face : d_side_faces)
    {
        const auto key =
            edge_key(side_face.nodes[0], side_face.nodes[1]);
        const auto tag = boundary_tags.find(key);
        if (side_face.neighbor != invalid_ordinal)
        {
            if (tag != boundary_tags.end())
            {
                throw std::invalid_argument(
                    "Semi-structured boundary tag refers to an "
                    "interior edge.");
            }
            continue;
        }

        const std::string patch_name =
            tag == boundary_tags.end() ? "side" : tag->second;
        if (tag != boundary_tags.end())
        {
            used_boundary_tags.insert(key);
        }

        const auto existing_id = boundary_ids.find(patch_name);
        if (existing_id != boundary_ids.end())
        {
            side_face.boundary_id = existing_id->second;
            continue;
        }

        side_face.boundary_id = next_boundary_id;
        boundary_ids.emplace(patch_name, next_boundary_id);
        d_boundary_names.emplace(next_boundary_id, patch_name);
        ++next_boundary_id;
    }

    if (used_boundary_tags.size() != boundary_tags.size())
    {
        throw std::invalid_argument(
            "Semi-structured boundary tag does not match an exterior "
            "edge.");
    }
}

void SemiStructMeshTopo::initialize_boundary_batches()
{
    for (const auto& [batch_id, name] : d_boundary_names)
    {
        static_cast<void>(name);
        if (d_indexer.axial_periodic
            && (batch_id == 0 || batch_id == 1))
        {
            continue;
        }
        d_boundary_batches.emplace(
            batch_id,
            BoundaryBatch{batch_id, {}});
    }

    if (!d_indexer.axial_periodic)
    {
        for (Ordinal cell = 0;
             cell < d_indexer.num_cells_per_layer;
             ++cell)
        {
            d_boundary_batches.at(0).face_lids.push_back(
                {cell, 0, Indexer::AXIAL});
            d_boundary_batches.at(1).face_lids.push_back(
                {cell, d_indexer.num_layers, Indexer::AXIAL});
        }
    }

    for (Ordinal side_face = 0;
         side_face < d_side_faces.size();
         ++side_face)
    {
        const auto batch_id = d_side_faces[side_face].boundary_id;
        if (batch_id == invalid_boundary_id)
        {
            continue;
        }
        for (Ordinal layer = 0;
             layer < d_indexer.num_layers;
             ++layer)
        {
            d_boundary_batches.at(batch_id).face_lids.push_back(
                {side_face, layer, Indexer::SIDE});
        }
    }
}

} // namespace SimpleFluid::Meshes
