/** @file ExplicitRegionProvider.hh
 * @brief Immutable rank-local explicit topology with original region ordinals.
 */
#pragma once

#include "geometry/mesh/RegionProviders.hh"
#include <map>
#include <cmath>
#include <set>

namespace SimpleFluid::Meshes
{
/**
 * The layout describes the global region, while records contain only visible
 * entities. Face adjacency always contains native global cell IDs, including
 * neighbors outside the local halo. Additional cell metrics permit resolving
 * outer halo faces without treating a partition cut as a physical boundary.
 */
class ExplicitRegionProvider : public GeometryExecutionGuard
{
public:
    using index_type_pack = UnstructuredMeshIndexTypes;
    using ID = uint64_t;
    using Vec3 = MeshUtils::Vec3;
    struct Cell
    {
        ID id{};
        MeshUtils::CellType type = MeshUtils::CellType::INVALID;
        std::vector<ID> faces, nodes;
        Vec3 centroid{};
        real_t volume{};
    };
    struct Face
    {
        ID id{}, owner{}, neighbor = UnstructuredMesh::invalid_ordinal;
        std::vector<ID> nodes;
        Vec3 centroid{}, area_vector{};
        int boundary = -1;
    };
    struct Node { ID id{}; Vec3 point{}; };

    ExplicitRegionProvider(RegionLayout layout, std::vector<Cell> cells,
                           std::vector<Face> faces, std::vector<Node> nodes,
                           std::map<int, std::string> boundaries)
        : d_layout(layout), d_boundaries(std::move(boundaries))
    {
        if (layout.family != RegionLayout::Family::Explicit || !layout.cells || !layout.faces || !layout.nodes
            || layout.cells == UnstructuredMesh::invalid_ordinal || layout.faces == UnstructuredMesh::invalid_ordinal
            || layout.nodes == UnstructuredMesh::invalid_ordinal || layout.extents != std::array<size_t, 3>{}
            || layout.periodic != std::array<bool, 3>{})
            throw std::invalid_argument("Explicit region requires explicit global layout.");
        for (const auto& [id, name] : d_boundaries)
            if (id == UnstructuredMesh::invalid_boundary_id || name.empty())
                throw std::invalid_argument("Invalid explicit region boundary metadata.");
        d_layout.template_identity = this;
        for (auto& c : cells)
        {
            if (c.id >= layout.cells || !(c.volume > 0) || !std::isfinite(c.volume)
                || !finite(c.centroid) || !supported(c.type)
                || c.faces.empty() != c.nodes.empty())
                throw std::invalid_argument("Invalid explicit region cell record.");
            if (!c.faces.empty())
            {
                const auto faces = std::set<ID>(c.faces.begin(), c.faces.end());
                const auto nodes = std::set<ID>(c.nodes.begin(), c.nodes.end());
                if (faces.size() != c.faces.size() || nodes.size() != c.nodes.size()
                    || faces.size() < 4 || nodes.size() < 4
                    || *faces.rbegin() >= layout.faces || *nodes.rbegin() >= layout.nodes
                    || (c.type == MeshUtils::CellType::HEXAHEDRON && (faces.size() != 6 || nodes.size() != 8))
                    || (c.type == MeshUtils::CellType::TRIPRISM && (faces.size() != 5 || nodes.size() != 6)))
                    throw std::invalid_argument("Invalid explicit region cell connectivity.");
            }
            if (!d_cells.emplace(c.id, std::move(c)).second)
                throw std::invalid_argument("Duplicate explicit region cell ID.");
        }
        for (auto& n : nodes)
            if (n.id >= layout.nodes || !finite(n.point)
                || !d_nodes.emplace(n.id, n.point).second)
                throw std::invalid_argument("Invalid explicit region node record.");
        for (auto& f : faces)
        {
            if (f.id >= layout.faces || f.owner >= layout.cells
                || (f.neighbor != UnstructuredMesh::invalid_ordinal && f.neighbor >= layout.cells)
                || f.owner == f.neighbor || f.nodes.size() < 3
                || !finite(f.centroid) || !finite(f.area_vector) || !(f.area_vector.norm() > 0)
                || !std::isfinite(f.area_vector.norm())
                || !d_cells.contains(f.owner)
                || (f.neighbor != UnstructuredMesh::invalid_ordinal && !d_cells.contains(f.neighbor)))
                throw std::invalid_argument("Invalid explicit region face record.");
            if (std::set<ID>(f.nodes.begin(), f.nodes.end()).size() != f.nodes.size()
                || (f.boundary != UnstructuredMesh::invalid_boundary_id
                    && (f.neighbor != UnstructuredMesh::invalid_ordinal || !d_boundaries.contains(f.boundary))))
                throw std::invalid_argument("Invalid explicit region face nodes or boundary.");
            for (auto n : f.nodes) if (!d_nodes.contains(n))
                throw std::invalid_argument("Explicit face references an unavailable node.");
            if (!d_faces.emplace(f.id, std::move(f)).second)
                throw std::invalid_argument("Duplicate explicit region face ID.");
        }
        for (const auto& [id, c] : d_cells)
        {
            std::set<ID> incident_nodes;
            std::map<std::pair<ID, ID>, std::pair<unsigned, int>> edges;
            for (auto f : c.faces)
            {
                const auto entry = d_faces.find(f);
                if (entry == d_faces.end()) throw std::invalid_argument("Explicit cell references an unavailable face.");
                const auto& face = entry->second;
                if (face.owner != id && face.neighbor != id)
                    throw std::invalid_argument("Explicit cell-face incidence is inconsistent.");
                incident_nodes.insert(face.nodes.begin(), face.nodes.end());
                for (size_t i = 0; i < face.nodes.size(); ++i)
                {
                    const auto a = face.nodes[i], b = face.nodes[(i + 1) % face.nodes.size()];
                    auto& edge = edges[std::minmax(a, b)];
                    ++edge.first;
                    edge.second += (a < b ? 1 : -1) * (face.owner == id ? 1 : -1);
                }
            }
            for (auto n : c.nodes) if (!d_nodes.contains(n))
                throw std::invalid_argument("Explicit cell references an unavailable node.");
            if (incident_nodes != std::set<ID>(c.nodes.begin(), c.nodes.end()))
                throw std::invalid_argument("Explicit cell nodes do not match its faces.");
            for (const auto& [key, incidence] : edges)
                if (incidence.first != 2 || incidence.second != 0)
                    throw std::invalid_argument("Explicit cell faces do not form a closed oriented surface.");
        }
        for (const auto& [id, face] : d_faces)
            for (auto c : {face.owner, face.neighbor})
                if (c != UnstructuredMesh::invalid_ordinal && !d_cells.at(c).faces.empty()
                    && std::ranges::find(d_cells.at(c).faces, id) == d_cells.at(c).faces.end())
                    throw std::invalid_argument("Explicit face is missing from an incident cell.");
    }

    RegionLayout layout() const { return d_layout; }
    const void* storage_identity() const { return this; }
    std::uint64_t geometry_epoch() const noexcept { return 0; }
    EntityRange<ID> cell_faces(size_t c) const { return range(d_cells.at(c).faces); }
    EntityRange<ID> cell_nodes(size_t c) const { return range(d_cells.at(c).nodes); }
    EntityRange<ID> face_nodes(size_t f) const { return range(d_faces.at(f).nodes); }
    ID owner_cell(size_t f) const { return d_faces.at(f).owner; }
    ID neighbor_cell(size_t f) const { return d_faces.at(f).neighbor; }
    MeshUtils::CellType cell_type(size_t c) const { return d_cells.at(c).type; }
    real_t cell_volume(size_t c) const { return d_cells.at(c).volume; }
    Vec3 cell_centroid(size_t c) const { return d_cells.at(c).centroid; }
    real_t face_area(size_t f) const { return d_faces.at(f).area_vector.norm(); }
    Vec3 face_centroid(size_t f) const { return d_faces.at(f).centroid; }
    Vec3 face_area_vector(size_t f) const { return d_faces.at(f).area_vector; }
    Vec3 node_coordinates(size_t n) const { return d_nodes.at(n); }
    int boundary_id(size_t f) const { return d_faces.at(f).boundary; }
    std::vector<int> boundary_batch_ids() const
    {
        std::vector<int> result;
        for (const auto& [id, name] : d_boundaries) result.push_back(id);
        return result;
    }
    const std::string& boundary_batch_name(int b) const { return d_boundaries.at(b); }
    size_t resident_cells() const noexcept { return d_cells.size(); }
    size_t resident_faces() const noexcept { return d_faces.size(); }
    size_t resident_nodes() const noexcept { return d_nodes.size(); }
    bool has_face(ID face) const noexcept { return d_faces.contains(face); }
    MeshStorageReport storage_report() const
    {
        MeshStorageReport report;
        report.topology = d_cells.size() * sizeof(Cell) + d_faces.size() * sizeof(Face);
        for (const auto& [id, c] : d_cells)
            report.topology += (c.faces.capacity() + c.nodes.capacity()) * sizeof(ID);
        for (const auto& [id, f] : d_faces) report.topology += f.nodes.capacity() * sizeof(ID);
        report.geometry = d_nodes.size() * sizeof(Node);
        // Ordered-map node overhead is an estimate, like existing index maps.
        report.indexing_estimate = (d_cells.size() + d_faces.size() + d_nodes.size()) * 3 * sizeof(void*);
        for (const auto& [id, name] : d_boundaries) report.boundaries += sizeof(id) + name.capacity();
        return report;
    }
private:
    static bool supported(MeshUtils::CellType type)
    {
        return type == MeshUtils::CellType::HEXAHEDRON || type == MeshUtils::CellType::TRIPRISM
            || type == MeshUtils::CellType::POLYHEDRON;
    }
    static bool finite(Vec3 p) { return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z); }
    static EntityRange<ID> range(const std::vector<ID>& values)
    {
        return {&values, 0, values.size(), [](const void* p, size_t, size_t i)
        { return (*static_cast<const std::vector<ID>*>(p))[i]; }};
    }
    RegionLayout d_layout;
    std::map<ID, Cell> d_cells;
    std::map<ID, Face> d_faces;
    std::map<ID, Vec3> d_nodes;
    std::map<int, std::string> d_boundaries;
};
using DistributedExplicitRegion = IsoRegion<ExplicitRegionProvider, ExplicitRegionProvider>;
static_assert(TopologyProvider<ExplicitRegionProvider>);
static_assert(GeometryProvider<ExplicitRegionProvider>);
} // namespace SimpleFluid::Meshes
