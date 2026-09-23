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
#include <map>
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
    if (node_count > 4)
    {
        return UnstructuredMesh::FaceType::POLYGON;
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
        if (boundary.node_ids.size() < 3)
        {
            throw std::invalid_argument(
                "UnstructuredMesh boundary face must have at least three nodes.");
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

bool opposite_loops(const Arr<Ordinal>& a, const Arr<Ordinal>& b)
{
    if (a.size() != b.size()) return false;
    const auto first = std::ranges::find(b, a.front());
    if (first == b.end()) return false;
    const auto start = static_cast<size_t>(first - b.begin());
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[(start + b.size() - i) % b.size()]) return false;
    return true;
}

/** Validate a simple planar loop in its dominant coordinate projection. */
void validate_polygon(const Arr<UnstructuredMesh::Vec3>& x)
{
    using Vec = UnstructuredMesh::Vec3;
    real_t scale = 0.0;
    for (const auto& p : x) scale = std::max(scale, (p - x[0]).norm());
    const auto area_vector = MeshUtils::face_area_vector(x);
    const auto area = area_vector.norm();
    constexpr auto relative = 256.0 * std::numeric_limits<real_t>::epsilon();
    const auto distance_tolerance = relative * scale;
    const auto area_tolerance = relative * scale * scale;
    if (!(area > area_tolerance) || !std::isfinite(area))
        throw std::invalid_argument("UnstructuredMesh contains a degenerate polygon.");
    const auto normal = area_vector / area;
    size_t drop = 0;
    for (size_t k = 1; k < 3; ++k)
        if (std::abs(normal.component(k)) > std::abs(normal.component(drop))) drop = k;
    const auto u = (drop + 1) % 3, v = (drop + 2) % 3;
    const auto turn = [u, v](const Vec& a, const Vec& b, const Vec& c) {
        return (b.component(u) - a.component(u)) * (c.component(v) - a.component(v))
             - (b.component(v) - a.component(v)) * (c.component(u) - a.component(u));
    };
    const auto within = [u, v, distance_tolerance](const Vec& a, const Vec& b, const Vec& p) {
        return p.component(u) >= std::min(a.component(u), b.component(u)) - distance_tolerance
            && p.component(u) <= std::max(a.component(u), b.component(u)) + distance_tolerance
            && p.component(v) >= std::min(a.component(v), b.component(v)) - distance_tolerance
            && p.component(v) <= std::max(a.component(v), b.component(v)) + distance_tolerance;
    };
    for (size_t i = 0; i < x.size(); ++i)
    {
        const auto& a = x[i];
        const auto& b = x[(i + 1) % x.size()];
        if (std::abs((a - x[0]).dot(normal)) > distance_tolerance)
            throw std::invalid_argument("UnstructuredMesh polyhedral face is not planar.");
        if (!((b - a).norm() > distance_tolerance))
            throw std::invalid_argument("UnstructuredMesh polygon has a zero-length edge.");
        const auto& previous = x[(i + x.size() - 1) % x.size()];
        if (std::abs(turn(previous, a, b)) <= area_tolerance
            && (previous - a).dot(b - a) > 0.0)
            throw std::invalid_argument("UnstructuredMesh polygon backtracks along an edge.");
        for (size_t j = i + 1; j < x.size(); ++j)
        {
            if (j == i + 1 || (i == 0 && j + 1 == x.size())) continue;
            const auto& c = x[j];
            const auto& d = x[(j + 1) % x.size()];
            const auto ab_c = turn(a, b, c), ab_d = turn(a, b, d);
            const auto cd_a = turn(c, d, a), cd_b = turn(c, d, b);
            const auto straddles = [area_tolerance](real_t p, real_t q) {
                return (p > area_tolerance && q < -area_tolerance)
                    || (p < -area_tolerance && q > area_tolerance);
            };
            if ((straddles(ab_c, ab_d) && straddles(cd_a, cd_b))
                || (std::abs(ab_c) <= area_tolerance && within(a, b, c))
                || (std::abs(ab_d) <= area_tolerance && within(a, b, d))
                || (std::abs(cd_a) <= area_tolerance && within(c, d, a))
                || (std::abs(cd_b) <= area_tolerance && within(c, d, b)))
                throw std::invalid_argument("UnstructuredMesh polygon loop intersects itself.");
        }
    }
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
    build_faces(boundary_faces, cells);
    compute_face_geometry();
    compute_polyhedral_cell_geometry();
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
    set_owned_counts(num_owned_cells, num_owned_faces);
}

void UnstructuredMesh::set_owned_counts(size_t num_owned_cells, size_t num_owned_faces)
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

UnstructuredMesh::UnstructuredMesh(
    const Arr<Vec3>& nodes, const PolyhedralTopology& topology)
    : d_nodes(nodes)
{
    validate_count(nodes.size(), "node");
    validate_count(topology.num_cells, "cell");
    validate_count(topology.faces.size(), "face");
    for (const auto& node : nodes)
        if (!std::isfinite(node.x) || !std::isfinite(node.y) || !std::isfinite(node.z))
            throw std::invalid_argument("UnstructuredMesh contains non-finite coordinates.");
    d_cells.resize(topology.num_cells);
    for (auto& cell : d_cells) cell.type = CellType::POLYHEDRON;
    std::unordered_set<FaceKey, FaceKeyHash> unique_faces;
    for (const auto& input : topology.faces)
    {
        if (input.node_ids.size() < 3 || input.owner >= topology.num_cells
            || (input.neighbor != invalid_ordinal && input.neighbor >= topology.num_cells)
            || input.owner == input.neighbor)
            throw std::invalid_argument("UnstructuredMesh explicit face has invalid connectivity.");
        const auto key = face_key(input.node_ids);
        if (key.back() >= nodes.size() || std::ranges::adjacent_find(key) != key.end()
            || !unique_faces.insert(key).second)
            throw std::invalid_argument("UnstructuredMesh explicit face has repeated or invalid connectivity.");
        if (input.neighbor != invalid_ordinal && input.boundary_id != invalid_boundary_id)
            throw std::invalid_argument("UnstructuredMesh interior face cannot have a boundary tag.");
        FaceInfo face;
        face.type = face_type_from_node_count(input.node_ids.size());
        face.node_ids = input.node_ids;
        face.owner = input.owner;
        face.neighbor = input.neighbor;
        face.boundary_id = input.boundary_id;
        const auto face_id = static_cast<FaceID>(d_faces.size());
        d_cells[face.owner].face_ids.push_back(face_id);
        if (face.neighbor != invalid_ordinal) d_cells[face.neighbor].face_ids.push_back(face_id);
        if (face.boundary_id != invalid_boundary_id)
        {
            const auto name = input.name.empty() ? default_boundary_name(face.boundary_id) : input.name;
            const auto [it, inserted] = d_boundary_names.emplace(face.boundary_id, name);
            if (!inserted && it->second != name)
                throw std::invalid_argument("UnstructuredMesh boundary ID is assigned multiple names.");
            auto& batch = d_boundary_batches[face.boundary_id];
            batch.id = face.boundary_id;
            batch.face_lids.push_back(face_id);
        }
        d_faces.push_back(std::move(face));
    }
    for (auto& cell : d_cells)
    {
        std::unordered_set<NodeID> seen;
        for (const auto face : cell.face_ids)
            for (const auto node : d_faces[face].node_ids)
                if (seen.insert(node).second) cell.node_ids.push_back(node);
    }
    compute_face_geometry();
    compute_polyhedral_cell_geometry();
    update_counts();
}

UnstructuredMesh::UnstructuredMesh(
    const Arr<Vec3>& nodes, const PolyhedralTopology& topology,
    size_t num_owned_cells, size_t num_owned_faces)
    : UnstructuredMesh(nodes, topology)
{
    set_owned_counts(num_owned_cells, num_owned_faces);
}

/**
 * @brief Move geometry, topology, and cached counts from another mesh.
 * @param other Mesh whose storage is transferred.
 * @return This mesh after assignment.
 */
UnstructuredMesh& UnstructuredMesh::operator=(
    UnstructuredMesh&& other)
{
    if (this == &other)
    {
        return *this;
    }

    GeometryExecutionGuard::operator=(std::move(other));
    d_geometry_state = std::move(other.d_geometry_state);
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
        if (cell.type == CellType::POLYHEDRON)
        {
            if (cell.face_node_ids.size() < 4)
                throw std::invalid_argument("UnstructuredMesh polyhedron requires at least four faces.");
            std::unordered_set<NodeID> face_nodes;
            for (const auto& loop : cell.face_node_ids)
            {
                auto key = face_key(loop);
                if (key.size() < 3 || key.back() >= d_nodes.size()
                    || std::ranges::adjacent_find(key) != key.end())
                    throw std::invalid_argument("UnstructuredMesh polyhedron has invalid face nodes.");
                face_nodes.insert(loop.begin(), loop.end());
            }
            if (!cell.node_ids.empty())
            {
                const std::unordered_set<NodeID> supplied(cell.node_ids.begin(), cell.node_ids.end());
                if (supplied.size() != cell.node_ids.size() || supplied != face_nodes)
                    throw std::invalid_argument("UnstructuredMesh polyhedron node list does not match its faces.");
            }
            continue;
        }
        if (!cell.face_node_ids.empty())
            throw std::invalid_argument("UnstructuredMesh element cell cannot specify polyhedral faces.");
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
        if (info.type == CellType::POLYHEDRON)
        {
            if (info.node_ids.empty())
            {
                std::unordered_set<NodeID> seen;
                for (const auto& face : cell.face_node_ids)
                    for (const auto node : face)
                        if (seen.insert(node).second) info.node_ids.push_back(node);
            }
            d_cells.push_back(std::move(info));
            continue;
        }

        Arr<Vec3> coords;
        coords.reserve(info.node_ids.size());
        for (const auto node_id : info.node_ids)
        {
            coords.push_back(d_nodes[node_id]);
        }
        if (info.type == CellType::HEXAHEDRON)
        {
            info.volume = MeshUtils::hex_volume(coords);
            if (info.volume > 0.0) info.center = MeshUtils::hex_centroid(coords);
        }
        else if (info.type == CellType::TRIPRISM)
        {
            info.volume = MeshUtils::wedge_volume(coords);
            if (info.volume > 0.0) info.center = MeshUtils::wedge_centroid(coords);
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
    const Arr<BoundaryFaceDefinition>& boundary_faces,
    const Arr<CellDefinition>& cells)
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
        auto loops = cells[cell_id].face_node_ids;
        if (cell.type != CellType::POLYHEDRON)
        {
            for (const auto& side : side_node_ordinals(cell.type))
            {
                Arr<NodeID> loop;
                Arr<Vec3> points;
                for (const auto ordinal : side)
                {
                    loop.push_back(cell.node_ids[ordinal]);
                    points.push_back(d_nodes[cell.node_ids[ordinal]]);
                }
                if (MeshUtils::face_area_vector(points).dot(MeshUtils::average(points) - cell.center) < 0.0)
                    std::ranges::reverse(loop);
                loops.push_back(std::move(loop));
            }
        }
        cell.face_ids.reserve(loops.size());
        for (auto& face_nodes : loops)
        {

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
            if (face.neighbor != invalid_ordinal || face.owner == cell_id)
            {
                throw std::invalid_argument(
                    "UnstructuredMesh contains a non-manifold face.");
            }
            if (!opposite_loops(face.node_ids, face_nodes))
                throw std::invalid_argument("UnstructuredMesh shared face loops have inconsistent orientation.");
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

        const bool polyhedral = d_cells[face.owner].type == CellType::POLYHEDRON
            || (face.neighbor != invalid_ordinal && d_cells[face.neighbor].type == CellType::POLYHEDRON);
        if (polyhedral) validate_polygon(coords);
        face.center = polyhedral ? MeshUtils::polygon_centroid(coords) : MeshUtils::face_centroid(coords);
        auto area_vector = MeshUtils::face_area_vector(coords);
        face.area = area_vector.norm();
        if (!(face.area > 0.0) || !std::isfinite(face.area))
        {
            throw std::invalid_argument(
                "UnstructuredMesh contains a degenerate face.");
        }

        auto normal = area_vector / face.area;
        if (!polyhedral && normal.dot(face.center - d_cells[face.owner].center) < 0.0)
        {
            normal = normal * -1.0;
        }
        face.normal = normal;
    }
}

void UnstructuredMesh::compute_polyhedral_cell_geometry()
{
    using Edge = std::pair<NodeID, NodeID>;
    struct Incidence
    {
        int balance = 0;
        Arr<size_t> faces;
    };
    for (CellID cell_id = 0; cell_id < d_cells.size(); ++cell_id)
    {
        auto& cell = d_cells[cell_id];
        if (cell.type != CellType::POLYHEDRON) continue;
        if (cell.face_ids.size() < 4 || cell.node_ids.size() < 4)
            throw std::invalid_argument("UnstructuredMesh polyhedron requires at least four faces and nodes.");

        std::map<Edge, Incidence> edges;
        std::unordered_map<NodeID, Arr<size_t>> vertex_faces;
        Arr<Vec3> nodes;
        for (const auto node : cell.node_ids) nodes.push_back(d_nodes[node]);
        // Shift all moment calculations near the cell to limit cancellation.
        const auto reference = nodes[0] + [&] {
            Vec3 sum{};
            for (const auto& node : nodes) sum = sum + (node - nodes[0]);
            return sum / static_cast<real_t>(nodes.size());
        }();
        real_t scale = 0.0;
        for (const auto& node : nodes) scale = std::max(scale, (node - reference).norm());
        long double volume = 0.0L;
        std::array<long double, 3> moment{};
        for (size_t local_face = 0; local_face < cell.face_ids.size(); ++local_face)
        {
            const auto& face = d_faces[cell.face_ids[local_face]];
            const int orientation = face.owner == cell_id ? 1 : -1;
            const auto& loop = face.node_ids;
            for (size_t i = 0; i < loop.size(); ++i)
            {
                auto a = loop[i], b = loop[(i + 1) % loop.size()];
                const auto key = std::minmax(a, b);
                auto& edge = edges[{key.first, key.second}];
                edge.balance += (a < b ? 1 : -1) * orientation;
                edge.faces.push_back(local_face);
                vertex_faces[a].push_back(local_face);
            }
            const auto a = d_nodes[loop[0]] - reference;
            for (size_t i = 1; i + 1 < loop.size(); ++i)
            {
                const auto b = d_nodes[loop[i]] - reference;
                const auto c = d_nodes[loop[i + 1]] - reference;
                const long double tetra = static_cast<long double>(a.dot(b.cross(c)))
                                        * orientation / 6.0L;
                volume += tetra;
                for (size_t k = 0; k < 3; ++k)
                    moment[k] += tetra * (static_cast<long double>(a.component(k))
                                          + b.component(k) + c.component(k)) / 4.0L;
            }
        }
        Arr<Arr<size_t>> adjacent(cell.face_ids.size());
        for (const auto& [edge, incidence] : edges)
        {
            if (incidence.faces.size() != 2 || incidence.balance != 0)
                throw std::invalid_argument("UnstructuredMesh polyhedron is not a closed oriented edge manifold.");
            const auto a = incidence.faces[0], b = incidence.faces[1];
            adjacent[a].push_back(b);
            adjacent[b].push_back(a);
        }
        const auto visit = [&adjacent](size_t first, const std::unordered_set<size_t>& allowed) {
            std::unordered_set<size_t> reached{first};
            Arr<size_t> stack{first};
            while (!stack.empty())
            {
                const auto current = stack.back();
                stack.pop_back();
                for (const auto next : adjacent[current])
                    if (allowed.contains(next) && reached.insert(next).second) stack.push_back(next);
            }
            return reached.size();
        };
        std::unordered_set<size_t> all;
        for (size_t f = 0; f < cell.face_ids.size(); ++f) all.insert(f);
        if (visit(0, all) != all.size())
            throw std::invalid_argument("UnstructuredMesh polyhedron surface is disconnected.");
        // Two disks meeting only at a vertex pass the edge test but do not form
        // a vertex manifold. Every incident-face fan must also be connected.
        for (const auto& [vertex, incident] : vertex_faces)
        {
            std::unordered_set<size_t> allowed(incident.begin(), incident.end());
            if (visit(incident.front(), allowed) != allowed.size())
                throw std::invalid_argument("UnstructuredMesh polyhedron has a non-manifold vertex.");
        }
        const auto tolerance = 256.0L * std::numeric_limits<real_t>::epsilon()
                             * scale * scale * scale;
        if (!(volume > tolerance) || !std::isfinite(volume))
            throw std::invalid_argument("UnstructuredMesh polyhedron has non-positive or degenerate signed volume.");
        cell.volume = static_cast<real_t>(volume);
        for (size_t k = 0; k < 3; ++k)
            cell.center.component(k) = reference.component(k) + static_cast<real_t>(moment[k] / volume);
        if (!std::isfinite(cell.volume) || !std::isfinite(cell.center.x)
            || !std::isfinite(cell.center.y) || !std::isfinite(cell.center.z))
            throw std::invalid_argument("UnstructuredMesh polyhedral geometry is not finite.");
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
