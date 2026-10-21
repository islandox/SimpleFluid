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
std::shared_ptr<const ExplicitRegionProvider> ExplicitRegionProvider::from_certified_records(RegionLayout layout,
    std::vector<Cell> cells, std::vector<Face> faces, std::vector<Node> nodes, std::map<int, std::string> boundaries,
    std::map<ID, ConvexGeometryPtr> cell_certificates, std::map<ID, ExactPlanarFacePtr> face_certificates,
    const std::map<ID, std::vector<char>>& node_certificates)
{
    auto provider = std::make_shared<ExplicitRegionProvider>(
        layout, std::move(cells), std::move(faces), std::move(nodes), std::move(boundaries));
    if (provider->d_nodes.size() != node_certificates.size() || provider->d_faces.size() != face_certificates.size())
        throw std::invalid_argument("Certified explicit fragment requires every resident node/face proof.");
    for (const auto& [id, node] : provider->d_nodes)
    {
        const auto point = exact_point_display(node_certificates.at(id));
        if (node.x != point[0] || node.y != point[1] || node.z != point[2])
            throw std::invalid_argument("Certified explicit displayed node differs from exact authority point.");
    }
    for (const auto& [id, face] : provider->d_faces)
    {
        const auto certificate = face_certificates.at(id);
        if (!certificate || !same(face.centroid, certificate->centroid()) ||
            !same(face.area_vector, certificate->area_vector()))
            throw std::invalid_argument("Certified explicit face metrics differ from exact authority.");
        const auto vertices = certificate->vertex_identities();
        if (vertices.size() != face.nodes.size())
            throw std::invalid_argument("Certified explicit face incidence size mismatch.");
        bool matches = false;
        for (std::size_t start = 0; start < vertices.size() && !matches; ++start)
        {
            bool valid = true;
            for (std::size_t i = 0; i < vertices.size(); ++i)
                valid = valid && node_certificates.at(face.nodes[i]) == vertices[(i + start) % vertices.size()];
            matches = valid;
        }
        if (!matches)
            throw std::invalid_argument("Certified explicit face nodes differ from ordered exact incidence.");
    }
    for (const auto& [id, cell] : provider->d_cells)
        if (!cell.faces.empty())
        {
            const auto certificate = cell_certificates.at(id);
            if (!certificate || certificate->moments().volume.as_double() != cell.volume)
                throw std::invalid_argument("Certified explicit cell volume differs from native record.");
            const auto center = certificate->centroid();
            if (cell.centroid.x != center[0] || cell.centroid.y != center[1] || cell.centroid.z != center[2])
                throw std::invalid_argument("Certified explicit cell center differs from native record.");
            std::vector<ExactPlanarFacePtr> outward;
            for (auto face : cell.faces)
            {
                auto geometry = face_certificates.at(face);
                if (provider->d_faces.at(face).owner != id)
                    geometry = geometry->reversed();
                outward.push_back(std::move(geometry));
            }
            if (!ExactPlanarFace::cell_shell(outward)->equivalent(*certificate))
                throw std::invalid_argument("Certified explicit outward shell differs from canonical cell.");
        }
    for (const auto& [id, certificate] : cell_certificates)
        if (!certificate || !provider->d_cells.contains(id) || provider->d_cells.at(id).faces.empty())
            throw std::invalid_argument("Certified explicit cell proof lacks full native residency.");
    provider->d_cell_certificates = std::move(cell_certificates);
    provider->d_face_certificates = std::move(face_certificates);
    return provider;
}
ConvexGeometryPtr ExplicitRegionProvider::certified_cell_geometry(std::size_t cell) const
{
    if (!d_cell_certificates.empty())
        return d_cell_certificates.at(cell);
    if (!d_construction)
        return {};
    if (!d_construction->has_full_cell(cell))
        throw std::invalid_argument("Metric-only neighbor has no resident exact cell certificate.");
    return d_construction->cell_geometry(cell);
}
ConvexGeometryPtr ExplicitRegionProvider::atomic_cell_geometry(std::size_t cell) const
{
    if (!d_cell_certificates.empty())
    {
        std::vector<ExactPlanarFacePtr> faces;
        for (auto face : d_cells.at(cell).faces)
        {
            auto geometry = d_face_certificates.at(face);
            if (d_faces.at(face).owner != cell)
                geometry = geometry->reversed();
            faces.push_back(std::move(geometry));
        }
        if (faces.empty())
            throw std::invalid_argument("Metric-only explicit neighbor has no atomic cell certificate.");
        return ExactPlanarFace::cell_shell(faces);
    }
    if (!d_construction)
        return {};
    if (!d_construction->has_full_cell(cell))
        throw std::invalid_argument("Metric-only neighbor has no resident atomic certificate.");
    return d_construction->atomic_cell_geometry(cell);
}
ExactPlanarFacePtr ExplicitRegionProvider::certified_face_geometry(std::size_t face) const
{
    if (!d_face_certificates.empty())
        return d_face_certificates.at(face);
    return d_construction ? d_construction->face_geometry(face) : ExactPlanarFacePtr{};
}
real_t ExplicitRegionProvider::native_area_magnitude(Vec3 vector)
{
    const auto ordinary = vector.norm();
    if (ordinary > 0 && std::isfinite(ordinary))
        return ordinary;
    if (!finite(vector) || (vector.x == 0 && vector.y == 0 && vector.z == 0))
        return ordinary;
    // Preserve ordinary-range arithmetic. Squared components can underflow or
    // overflow although the finite nonzero vector's magnitude is representable.
    return std::hypot(vector.x, vector.y, vector.z);
}
real_t ExplicitRegionProvider::face_area(std::size_t face) const
{
    if (!d_face_certificates.empty())
        return d_face_certificates.at(face)->area().as_double();
    return d_construction ? d_construction->face_area(face) : native_area_magnitude(d_faces.at(face).area_vector);
}
std::size_t ExplicitRegionProvider::exact_storage_bytes() const
{
    if (d_construction)
        return d_construction->storage_bytes();
    std::size_t bytes = 0;
    std::set<const void*> seen;
    for (const auto& [id, cell] : d_cell_certificates)
    {
        bytes += sizeof(decltype(d_cell_certificates)::value_type) + 4 * sizeof(void*);
        if (seen.insert(cell.get()).second)
            bytes += cell->storage_bytes();
    }
    for (const auto& [id, face] : d_face_certificates)
    {
        bytes += sizeof(decltype(d_face_certificates)::value_type) + 4 * sizeof(void*);
        if (seen.insert(face.get()).second)
            bytes += face->encode_incidence().size();
    }
    return bytes;
}
} // namespace SimpleFluid::Meshes
