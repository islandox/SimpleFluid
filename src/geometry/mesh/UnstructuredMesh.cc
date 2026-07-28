/**
 * @file UnstructuredMesh.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief UnstructuredMesh topology and geometry construction.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/UnstructuredMesh.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace SimpleFluid::Meshes
{
namespace
{

using Ordinal = UnstructuredMesh::Ordinal;
using FaceKey = Arr<Ordinal>;

/** @brief Hash a sorted variable-length face-node key. */
struct FaceKeyHash
{
    size_t operator()(const FaceKey& key) const noexcept
    {
        size_t seed = key.size();
        for (const auto value : key)
        {
            seed ^= static_cast<size_t>(value) + 0x9e3779b9u
                  + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

/** @brief Boundary identifier and name recovered from input definitions. */
struct BoundaryTag
{
    int id = UnstructuredMesh::invalid_boundary_id;
    std::string name;
};

/**
 * @brief Validate that a entity count fits in the mesh's ordinal type.
 *
 * @param count The number of entities.
 * @param quantity Human-readable entity name for error messages.
 * @throws std::overflow_error if @p count exceeds the ordinal maximum.
 */
void validate_count(size_t count, const char* quantity)
{
    if (count >= UnstructuredMesh::invalid_ordinal)
    {
        throw std::overflow_error(
            std::string("UnstructuredMesh ") + quantity
            + " count exceeds its ID type.");
    }
}

/**
 * @brief Create a normalized (sorted) face key from a set of node ordinals.
 *
 * @param node_ids Face node ordinals.
 * @return Sorted node-ordinal array suitable for hash-based deduplication.
 */
FaceKey face_key(Arr<Ordinal> node_ids)
{
    std::ranges::sort(node_ids);
    return node_ids;
}

/**
 * @brief Map the number of face nodes to the corresponding FaceType enum.
 *
 * @param node_count Number of nodes on the face (3 for triangle, 4 for quad).
 * @return The FaceType enumeration value.
 * @throws std::invalid_argument if @p node_count is unsupported.
 */
auto face_type_from_node_count(size_t node_count)
    -> UnstructuredMesh::FaceType
{
    if (node_count == 3)
    {
        return UnstructuredMesh::FaceType::TRIANGLE;
    }
    if (node_count == 4)
    {
        return UnstructuredMesh::FaceType::QUAD;
    }
    throw std::invalid_argument(
        "UnstructuredMesh face has unsupported node count.");
}

/**
 * @brief Return the canonical side-face node-ordinal lists for a cell type.
 *
 * @param type The cell type (HEXAHEDRON or TRIPRISM).
 * @return Per-face node-ordinal arrays following the cell's local node numbering.
 * @throws std::invalid_argument if @p type is unsupported.
 */
Arr<Arr<Ordinal>> side_node_ordinals(UnstructuredMesh::CellType type)
{
    using CellType = UnstructuredMesh::CellType;
    if (type == CellType::HEXAHEDRON)
    {
        return {
            {0, 1, 2, 3},
            {4, 5, 6, 7},
            {0, 4, 5, 1},
            {1, 5, 6, 2},
            {2, 6, 7, 3},
            {3, 7, 4, 0}};
    }
    if (type == CellType::TRIPRISM)
    {
        return {
            {0, 1, 2},
            {3, 4, 5},
            {0, 3, 4, 1},
            {1, 4, 5, 2},
            {2, 5, 3, 0}};
    }
    throw std::invalid_argument(
        "UnstructuredMesh cell type is unsupported.");
}

/**
 * @brief Return the expected total node count for a cell type.
 *
 * @param type The cell type (HEXAHEDRON: 8, TRIPRISM: 6).
 * @return The number of nodes.
 * @throws std::invalid_argument if @p type is unsupported.
 */
size_t expected_node_count(UnstructuredMesh::CellType type)
{
    using CellType = UnstructuredMesh::CellType;
    if (type == CellType::HEXAHEDRON)
    {
        return 8;
    }
    if (type == CellType::TRIPRISM)
    {
        return 6;
    }
    throw std::invalid_argument(
        "UnstructuredMesh cell type is unsupported.");
}

/**
 * @brief Generate a default boundary batch name from a numeric ID.
 *
 * @param id The boundary batch ID.
 * @return A string of the form "boundary_N".
 */
std::string default_boundary_name(int id)
{
    return "boundary_" + std::to_string(id);
}

/**
 * @brief Build a face-key to boundary-tag map from boundary face definitions.
 *
 * Validates uniqueness of boundary names per ID and deduplicates face keys.
 *
 * @param boundary_faces Boundary face definitions.
 * @return Map from sorted node-ordinal key to boundary ID and name.
 * @throws std::invalid_argument if duplicate names, keys, or node IDs are found.
 */
std::unordered_map<FaceKey, BoundaryTag, FaceKeyHash>
make_boundary_tags(
    const Arr<UnstructuredMesh::BoundaryFaceDefinition>& boundary_faces)
{
    std::unordered_map<FaceKey, BoundaryTag, FaceKeyHash> tags;
    std::unordered_map<int, std::string> names;
    for (const auto& boundary : boundary_faces)
    {
        if (boundary.boundary_id
            == UnstructuredMesh::invalid_boundary_id)
        {
            throw std::invalid_argument(
                "UnstructuredMesh boundary face uses the invalid "
                "boundary ID.");
        }
        if (boundary.node_ids.size() != 3
            && boundary.node_ids.size() != 4)
        {
            throw std::invalid_argument(
                "UnstructuredMesh boundary face must have three or "
                "four nodes.");
        }

        const auto name = boundary.name.empty()
            ? default_boundary_name(boundary.boundary_id)
            : boundary.name;
        const auto [name_iter, inserted_name] =
            names.emplace(boundary.boundary_id, name);
        if (!inserted_name && name_iter->second != name)
        {
            throw std::invalid_argument(
                "UnstructuredMesh boundary ID is assigned multiple "
                "names.");
        }

        auto key = face_key(boundary.node_ids);
        if (std::ranges::adjacent_find(key) != key.end())
        {
            throw std::invalid_argument(
                "UnstructuredMesh boundary face repeats a node ID.");
        }
        if (!tags.emplace(
                std::move(key),
                BoundaryTag{boundary.boundary_id, name}).second)
        {
            throw std::invalid_argument(
                "UnstructuredMesh boundary face is specified more "
                "than once.");
        }
    }
    return tags;
}

} // namespace

/**
 * @brief Construct an unstructured mesh from nodes, cell definitions, and optional boundary faces.
 *
 * @param nodes Node coordinate array.
 * @param cells Cell definitions with type and connectivity.
 * @param boundary_faces Boundary face definitions with node sets and tags.
 * @throws std::invalid_argument if input validation fails.
 */
UnstructuredMesh::UnstructuredMesh(
    const Arr<Vec3>& nodes,
    const Arr<CellDefinition>& cells,
    const Arr<BoundaryFaceDefinition>& boundary_faces)
    : d_nodes(nodes)
{
    validate_input(cells, boundary_faces);
    initialize_cells(cells);
    build_faces(boundary_faces);
    compute_face_geometry();
    update_counts();
}

/**
 * @brief Construct an unstructured mesh with explicit owned-entity counts for partitioned use.
 *
 * @param nodes Node coordinate array.
 * @param cells Cell definitions with type and connectivity.
 * @param boundary_faces Boundary face definitions with node sets and tags.
 * @param num_owned_cells Number of cells owned by this rank.
 * @param num_owned_faces Number of faces owned by this rank.
 * @throws std::invalid_argument if owned counts exceed local counts.
 */
UnstructuredMesh::UnstructuredMesh(
    const Arr<Vec3>& nodes,
    const Arr<CellDefinition>& cells,
    const Arr<BoundaryFaceDefinition>& boundary_faces,
    size_t num_owned_cells,
    size_t num_owned_faces)
    : UnstructuredMesh(nodes, cells, boundary_faces)
{
    if (num_owned_cells > Base::d_num_cells
        || num_owned_faces > Base::d_num_faces)
    {
        throw std::invalid_argument(
            "UnstructuredMesh owned entity count exceeds its local count.");
    }
    Base::d_num_owned_cells = num_owned_cells;
    Base::d_num_owned_faces = num_owned_faces;
}

/**
 * @brief Move geometry, topology, and cached counts from another mesh.
 * @param other Mesh whose storage is transferred.
 * @return This mesh after assignment.
 */
UnstructuredMesh& UnstructuredMesh::operator=(
    UnstructuredMesh&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }

    d_nodes = std::move(other.d_nodes);
    d_cells = std::move(other.d_cells);
    d_faces = std::move(other.d_faces);
    d_boundary_names = std::move(other.d_boundary_names);
    d_boundary_batches = std::move(other.d_boundary_batches);
    d_indexer = other.d_indexer;

    Base::d_num_local_cells = other.d_num_local_cells;
    Base::d_num_owned_cells = other.d_num_owned_cells;
    Base::d_num_cells = other.d_num_cells;
    Base::d_num_faces = other.d_num_faces;
    Base::d_num_owned_faces = other.d_num_owned_faces;
    Base::d_num_nodes = other.d_num_nodes;
    return *this;
}

/**
 * @brief Validate cell and boundary-face inputs before mesh construction.
 *
 * Checks node count, cell connectivity, node coordinate finiteness,
 * and boundary-face node references.
 *
 * @param cells Cell definitions to validate.
 * @param boundary_faces Boundary face definitions to validate.
 * @throws std::invalid_argument on any validation failure.
 */
void UnstructuredMesh::validate_input(
    const Arr<CellDefinition>& cells,
    const Arr<BoundaryFaceDefinition>& boundary_faces) const
{
    if (d_nodes.empty())
    {
        throw std::invalid_argument(
            "UnstructuredMesh requires at least one node.");
    }
    if (cells.empty())
    {
        throw std::invalid_argument(
            "UnstructuredMesh requires at least one cell.");
    }
    validate_count(d_nodes.size(), "node");
    validate_count(cells.size(), "cell");

    for (const auto& node : d_nodes)
    {
        if (!std::isfinite(node.x)
            || !std::isfinite(node.y)
            || !std::isfinite(node.z))
        {
            throw std::invalid_argument(
                "UnstructuredMesh contains a non-finite node "
                "coordinate.");
        }
    }

    for (const auto& cell : cells)
    {
        const auto expected_nodes = expected_node_count(cell.type);
        if (cell.node_ids.size() != expected_nodes)
        {
            throw std::invalid_argument(
                "UnstructuredMesh cell has the wrong node count for "
                "its type.");
        }

        std::unordered_set<NodeID> unique_nodes;
        for (const auto node_id : cell.node_ids)
        {
            if (node_id >= d_nodes.size()
                || !unique_nodes.insert(node_id).second)
            {
                throw std::invalid_argument(
                    "UnstructuredMesh cell has invalid node "
                    "connectivity.");
            }
        }
    }

    for (const auto& boundary : boundary_faces)
    {
        for (const auto node_id : boundary.node_ids)
        {
            if (node_id >= d_nodes.size())
            {
                throw std::invalid_argument(
                    "UnstructuredMesh boundary face references an "
                    "invalid node ID.");
            }
        }
    }
}

/**
 * @brief Populate the internal cell array from cell definitions.
 *
 * Computes cell centroids and volumes, and validates non-positive volumes.
 *
 * @param cells Cell definitions to store.
 * @throws std::invalid_argument if any cell has non-positive volume.
 */
void UnstructuredMesh::initialize_cells(
    const Arr<CellDefinition>& cells)
{
    d_cells.clear();
    d_cells.reserve(cells.size());
    for (const auto& cell : cells)
    {
        CellInfo info;
        info.type = cell.type;
        info.node_ids = cell.node_ids;

        Arr<Vec3> coords;
        coords.reserve(info.node_ids.size());
        for (const auto node_id : info.node_ids)
        {
            coords.push_back(d_nodes[node_id]);
        }
        info.center = MeshUtils::average(coords);

        if (info.type == CellType::HEXAHEDRON)
        {
            info.volume = MeshUtils::hex_volume(coords);
        }
        else if (info.type == CellType::TRIPRISM)
        {
            info.volume = MeshUtils::wedge_volume(coords);
        }

        if (!(info.volume > 0.0) || !std::isfinite(info.volume))
        {
            throw std::invalid_argument(
                "UnstructuredMesh cell has non-positive volume.");
        }

        d_cells.push_back(std::move(info));
    }
}

/**
 * @brief Derive interior and boundary faces from cell connectivity.
 *
 * Deduplicates faces by sorted node set, tags boundary faces, and populates
 * the per-face owner/neighbor adjacency and boundary batch maps.
 *
 * @param boundary_faces Boundary face definitions.
 */
void UnstructuredMesh::build_faces(
    const Arr<BoundaryFaceDefinition>& boundary_faces)
{
    d_faces.clear();
    d_boundary_names.clear();
    d_boundary_batches.clear();

    auto boundary_tags = make_boundary_tags(boundary_faces);
    std::unordered_set<FaceKey, FaceKeyHash> matched_boundary_tags;
    std::unordered_map<FaceKey, FaceID, FaceKeyHash> face_ids;

    for (CellID cell_id = 0; cell_id < d_cells.size(); ++cell_id)
    {
        auto& cell = d_cells[cell_id];
        cell.face_ids.clear();
        const auto side_ordinals = side_node_ordinals(cell.type);
        cell.face_ids.reserve(side_ordinals.size());

        for (const auto& side : side_ordinals)
        {
            Arr<NodeID> face_nodes;
            face_nodes.reserve(side.size());
            for (const auto ordinal : side)
            {
                face_nodes.push_back(cell.node_ids[ordinal]);
            }

            const auto key = face_key(face_nodes);
            const auto existing = face_ids.find(key);
            if (existing == face_ids.end())
            {
                validate_count(d_faces.size(), "face");
                const auto face_id =
                    static_cast<FaceID>(d_faces.size());
                FaceInfo face;
                face.type = face_type_from_node_count(face_nodes.size());
                face.node_ids = std::move(face_nodes);
                face.owner = cell_id;

                d_faces.push_back(std::move(face));
                face_ids.emplace(key, face_id);
                cell.face_ids.push_back(face_id);
                continue;
            }

            auto& face = d_faces[existing->second];
            if (face.neighbor != invalid_ordinal)
            {
                throw std::invalid_argument(
                    "UnstructuredMesh contains a non-manifold face.");
            }
            face.neighbor = cell_id;
            cell.face_ids.push_back(existing->second);
        }
    }

    for (FaceID face_id = 0; face_id < d_faces.size(); ++face_id)
    {
        auto& face = d_faces[face_id];
        const auto key = face_key(face.node_ids);
        const auto tag = boundary_tags.find(key);
        if (face.neighbor != invalid_ordinal)
        {
            if (tag != boundary_tags.end())
            {
                throw std::invalid_argument(
                    "UnstructuredMesh boundary tag refers to an "
                    "interior face.");
            }
            continue;
        }
        if (tag == boundary_tags.end())
        {
            continue;
        }

        matched_boundary_tags.insert(key);
        face.boundary_id = tag->second.id;
        d_boundary_names.emplace(tag->second.id, tag->second.name);

        auto& batch = d_boundary_batches[tag->second.id];
        batch.id = tag->second.id;
        batch.face_lids.push_back(face_id);
    }

    if (matched_boundary_tags.size() != boundary_tags.size())
    {
        throw std::invalid_argument(
            "UnstructuredMesh boundary tag does not match an exterior "
            "face.");
    }
}

/**
 * @brief Compute face centroids, areas, and owner-oriented unit normals.
 * @throws std::invalid_argument If a face is degenerate or non-finite.
 */
void UnstructuredMesh::compute_face_geometry()
{
    for (auto& face : d_faces)
    {
        Arr<Vec3> coords;
        coords.reserve(face.node_ids.size());
        for (const auto node_id : face.node_ids)
        {
            coords.push_back(d_nodes[node_id]);
        }

        face.center = MeshUtils::average(coords);
        auto area_vector = MeshUtils::face_area_vector(coords);
        face.area = area_vector.norm();
        if (!(face.area > 0.0) || !std::isfinite(face.area))
        {
            throw std::invalid_argument(
                "UnstructuredMesh contains a degenerate face.");
        }

        auto normal = area_vector / face.area;
        const auto owner_to_face =
            face.center - d_cells[face.owner].center;
        if (normal.dot(owner_to_face) < 0.0)
        {
            normal = normal * -1.0;
        }
        face.normal = normal;
    }
}

/** @brief Synchronize the identity indexer and base-class entity counts. */
void UnstructuredMesh::update_counts()
{
    d_indexer = {
        static_cast<Ordinal>(d_cells.size()),
        static_cast<Ordinal>(d_faces.size()),
        static_cast<Ordinal>(d_nodes.size())};

    Base::d_num_cells = d_cells.size();
    Base::d_num_local_cells = d_cells.size();
    Base::d_num_owned_cells = d_cells.size();
    Base::d_num_faces = d_faces.size();
    Base::d_num_owned_faces = d_faces.size();
    Base::d_num_nodes = d_nodes.size();
}

} // namespace SimpleFluid::Meshes
