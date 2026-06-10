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
    initialize_cell_adjacency();
    initialize_boundary_patches();
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
    if (id.orientation == Indexer::AXIAL)
    {
        if (id.k == 0 && d_indexer.axial_periodic)
        {
            return {id.ij, d_indexer.num_layers - 1};
        }
        return {id.ij, id.k == 0 ? 0U : id.k - 1};
    }
    return {d_side_faces[id.ij].owner, id.k};
}

auto SemiStructMeshTopo::neighbor_cell(FaceID id) const noexcept -> CellID
{
    if (id.orientation == Indexer::AXIAL)
    {
        return d_indexer.axial_periodic
                || (id.k != 0 && id.k != d_indexer.num_layers)
             ? CellID{id.ij, id.k}
             : CellID{};
    }

    const auto neighbor = d_side_faces[id.ij].neighbor;
    return neighbor == invalid_ordinal
         ? CellID{}
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
SemiStructMeshTopo::boundary_patch_name(int patch_id) const
{
    const auto patch = d_boundary_names.find(patch_id);
    if (patch == d_boundary_names.end())
    {
        throw std::out_of_range("Requested boundary patch is not found.");
    }
    return patch->second;
}

const SemiStructMeshTopo::BoundaryPatch&
SemiStructMeshTopo::boundary_face_patch(int patch_id) const
{
    const auto patch = d_boundary_patches.find(patch_id);
    if (patch == d_boundary_patches.end())
    {
        throw std::out_of_range("Requested boundary patch is not found.");
    }
    return patch->second;
}

std::vector<int> SemiStructMeshTopo::boundary_patch_ids() const
{
    std::vector<int> ids;
    ids.reserve(d_boundary_patches.size());
    for (const auto& [id, patch] : d_boundary_patches)
    {
        (void)patch;
        ids.push_back(id);
    }
    return ids;
}

int SemiStructMeshTopo::num_boundary_patches() const noexcept
{
    return static_cast<int>(d_boundary_patches.size());
}

void SemiStructMeshTopo::initialize_cell_adjacency()
{
    d_neighbor_cells.resize(d_indexer.total_cells());
    d_interior_cell_patch.reserve(d_indexer.total_cells());

    for (size_t local_id = 0;
         local_id < d_indexer.total_cells();
         ++local_id)
    {
        const auto cell = d_indexer.cell_id(local_id);
        const auto& side_faces = d_cell_side_faces[cell.ij];
        auto& neighbors = d_neighbor_cells[local_id];
        neighbors.reserve(side_faces.size() + 2);

        if (cell.k > 0)
        {
            neighbors.push_back({cell.ij, cell.k - 1});
        }
        else if (d_indexer.axial_periodic)
        {
            neighbors.push_back(
                {cell.ij, d_indexer.num_layers - 1});
        }

        if (cell.k + 1 < d_indexer.num_layers)
        {
            neighbors.push_back({cell.ij, cell.k + 1});
        }
        else if (d_indexer.axial_periodic)
        {
            neighbors.push_back({cell.ij, 0});
        }

        for (const auto side_face_id : side_faces)
        {
            const auto& side_face = d_side_faces[side_face_id];
            if (cell.ij == side_face.owner)
            {
                if (side_face.neighbor != invalid_ordinal)
                {
                    neighbors.push_back(
                        {side_face.neighbor, cell.k});
                }
            }
            else
            {
                neighbors.push_back({side_face.owner, cell.k});
            }
        }

        if (neighbors.size() == side_faces.size() + 2)
        {
            d_interior_cell_patch.push_back(cell);
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

void SemiStructMeshTopo::initialize_boundary_patches()
{
    for (const auto& [patch_id, name] : d_boundary_names)
    {
        static_cast<void>(name);
        if (d_indexer.axial_periodic
            && (patch_id == 0 || patch_id == 1))
        {
            continue;
        }
        d_boundary_patches.emplace(
            patch_id,
            BoundaryPatch{patch_id, {}});
    }

    if (!d_indexer.axial_periodic)
    {
        for (Ordinal cell = 0;
             cell < d_indexer.num_cells_per_layer;
             ++cell)
        {
            d_boundary_patches.at(0).face_lids.push_back(
                {cell, 0, Indexer::AXIAL});
            d_boundary_patches.at(1).face_lids.push_back(
                {cell, d_indexer.num_layers, Indexer::AXIAL});
        }
    }

    for (Ordinal side_face = 0;
         side_face < d_side_faces.size();
         ++side_face)
    {
        const auto patch_id = d_side_faces[side_face].boundary_id;
        if (patch_id == invalid_boundary_id)
        {
            continue;
        }
        for (Ordinal layer = 0;
             layer < d_indexer.num_layers;
             ++layer)
        {
            d_boundary_patches.at(patch_id).face_lids.push_back(
                {side_face, layer, Indexer::SIDE});
        }
    }
}

} // namespace SimpleFluid::Meshes
