/** @file ExplicitRegionProvider.cc
 * @brief Sparse authoritative exact explicit-provider bindings and metric validation.
 */
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include "geometry/ExactPolyhedralComplex.hh"
#include <algorithm>

namespace SimpleFluid::Meshes
{
namespace
{
bool same(MeshUtils::Vec3 a, MeshUtils::Vec3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
} // namespace
ExplicitRegionProvider::ExplicitRegionProvider(RegionLayout layout, std::vector<Cell> cells, std::vector<Face> faces,
    std::vector<Node> nodes, std::map<int, std::string> boundaries,
    std::shared_ptr<const ExactPolyhedralComplex> construction)
    : ExplicitRegionProvider(layout, std::move(cells), std::move(faces), std::move(nodes), std::move(boundaries))
{
    if (!construction || layout.cells != construction->global_cells() || layout.faces != construction->global_faces() ||
        layout.nodes != construction->global_nodes() || d_cells.size() != construction->cell_ids().size() ||
        d_faces.size() != construction->face_ids().size() || d_nodes.size() != construction->node_ids().size())
        throw std::invalid_argument("Exact explicit construction differs from native residency/layout.");
    for (const auto id : construction->cell_ids())
    {
        const auto& cell = d_cells.at(id);
        if (cell.volume != construction->cell_volume(id) || !same(cell.centroid, construction->cell_centroid(id)) ||
            cell.faces != construction->cell_faces(id) || cell.nodes != construction->cell_nodes(id) ||
            (!cell.faces.empty()) != construction->has_full_cell(id))
            throw std::invalid_argument(
                "Exact explicit cell topology or metrics differ from authoritative construction.");
    }
    for (const auto id : construction->face_ids())
    {
        const auto& face = d_faces.at(id);
        const auto& surface = construction->face_surface(id);
        if (face.owner != construction->owner_cell(id) || face.neighbor != construction->neighbor_cell(id) ||
            face.nodes != construction->face_nodes(id) || !same(face.centroid, construction->face_centroid(id)) ||
            !same(face.area_vector, construction->face_area_vector(id)) ||
            (surface.exterior && !surface.boundary.empty()
                    ? (!d_boundaries.contains(face.boundary) || d_boundaries.at(face.boundary) != surface.boundary)
                    : face.boundary != UnstructuredMesh::invalid_boundary_id))
            throw std::invalid_argument("Exact explicit face topology, patch or metrics differ from construction.");
    }
    for (const auto id : construction->node_ids())
        if (!same(d_nodes.at(id), construction->node_coordinates(id)))
            throw std::invalid_argument("Exact explicit displayed node differs from its authoritative construction.");
    d_construction = std::move(construction);
}

std::shared_ptr<const ExplicitRegionProvider> ExplicitRegionProvider::from_exact_complex(
    std::shared_ptr<const ExactPolyhedralComplex> construction)
{
    if (!construction || construction->resident_cell_ids().size() != construction->global_cells())
        throw std::invalid_argument("Sparse exact explicit factory requires an explicit global boundary schema.");
    std::set<std::string> names;
    for (const auto face : construction->face_ids())
        if (const auto& surface = construction->face_surface(face); surface.exterior && !surface.boundary.empty())
            names.insert(surface.boundary);
    std::map<int, std::string> boundaries;
    int next = 0;
    for (const auto& name : names)
        boundaries.emplace(next++, name);
    return from_exact_complex(std::move(construction), std::move(boundaries));
}
std::shared_ptr<const ExplicitRegionProvider> ExplicitRegionProvider::from_exact_complex(
    std::shared_ptr<const ExactPolyhedralComplex> construction, std::map<int, std::string> boundaries)
{
    if (!construction)
        throw std::invalid_argument("Exact explicit provider requires a construction.");
    std::map<std::string, int> ids;
    for (const auto& [id, name] : boundaries)
        if (!ids.emplace(name, id).second)
            throw std::invalid_argument("Repeated exact explicit patch name.");
    std::vector<Cell> cells;
    std::vector<Face> faces;
    std::vector<Node> nodes;
    for (const auto c : construction->cell_ids())
        cells.push_back({c, MeshUtils::CellType::POLYHEDRON, construction->cell_faces(c), construction->cell_nodes(c),
            construction->cell_centroid(c), construction->cell_volume(c)});
    for (const auto f : construction->face_ids())
    {
        const auto& surface = construction->face_surface(f);
        if (surface.exterior && !surface.boundary.empty() && !ids.contains(surface.boundary))
            throw std::invalid_argument("Exact exterior patch is missing from the explicit global schema.");
        const int boundary = surface.exterior && !surface.boundary.empty() ? ids.at(surface.boundary)
                                                                           : UnstructuredMesh::invalid_boundary_id;
        faces.push_back({f, construction->owner_cell(f), construction->neighbor_cell(f), construction->face_nodes(f),
            construction->face_centroid(f), construction->face_area_vector(f), boundary});
    }
    for (const auto n : construction->node_ids())
        nodes.push_back({n, construction->node_coordinates(n)});
    RegionLayout layout{};
    layout.family = RegionLayout::Family::Explicit;
    layout.cells = construction->global_cells();
    layout.faces = construction->global_faces();
    layout.nodes = construction->global_nodes();
    return std::make_shared<const ExplicitRegionProvider>(
        layout, std::move(cells), std::move(faces), std::move(nodes), std::move(boundaries), std::move(construction));
}
ConvexGeometryPtr ExplicitRegionProvider::certified_cell_geometry(std::size_t cell) const
{
    if (!d_construction)
        return {};
    if (!d_construction->has_full_cell(cell))
        throw std::invalid_argument("Metric-only neighbor has no resident exact cell certificate.");
    return d_construction->cell_geometry(cell);
}
ConvexGeometryPtr ExplicitRegionProvider::atomic_cell_geometry(std::size_t cell) const
{
    if (!d_construction)
        return {};
    if (!d_construction->has_full_cell(cell))
        throw std::invalid_argument("Metric-only neighbor has no resident atomic certificate.");
    return d_construction->atomic_cell_geometry(cell);
}
real_t ExplicitRegionProvider::face_area(std::size_t face) const
{
    return d_construction ? d_construction->face_area(face) : d_faces.at(face).area_vector.norm();
}
std::size_t ExplicitRegionProvider::exact_storage_bytes() const
{
    return d_construction ? d_construction->storage_bytes() : 0;
}
} // namespace SimpleFluid::Meshes
