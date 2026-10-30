#include "geometry/PLIC/PolyhedralComplex.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include <algorithm>
#include <map>
#include <set>

namespace SimpleFluid::PLIC
{
namespace ordinary_complex_detail
{
using namespace details::ordinary_native;
using ID = std::uint64_t;
using Complex = PolyhedralComplex<ArithmeticMode::Ordinary>;
using CellGeometry = ConvexCellGeometry<ArithmeticMode::Ordinary>;
using CellPtr = ConvexGeometryPtr<ArithmeticMode::Ordinary>;
using Face = PlanarFace<ArithmeticMode::Ordinary>;
using FacePtr = PlanarFacePtr<ArithmeticMode::Ordinary>;
using CellKey = Meshes::AMR::AdaptiveCellKey;
using Ancestry = Meshes::AMR::ConvexCellAncestry;
constexpr ID absent = std::numeric_limits<ID>::max();
struct CellRow
{
    CellKey key;
    CellPtr geometry;
    double volume = 0;
    MeshUtils::Vec3 center{};
    std::vector<ID> faces, nodes;
};
struct FaceRow
{
    PolyhedralSurfaceUse surface;
    ID owner = absent, neighbor = absent;
    FacePtr geometry;
    std::vector<ID> nodes;
};
struct NodeRow
{
    std::uint64_t region = 0;
    WidePoint point{};
};
void write_surface(Writer& w, const PolyhedralSurfaceUse& use)
{
    w.word(static_cast<ID>(use.key.kind));
    w.word(use.key.region);
    w.word(use.key.creator.root);
    w.word(use.key.creator.path.size());
    for (auto id : use.key.creator.path)
        w.word(id);
    w.word(use.key.ordinal);
    w.word(use.orientation == 1 ? 1 : 0);
    w.word(use.exterior);
    w.packet(std::span<const char>(use.boundary.data(), use.boundary.size()));
}
PolyhedralSurfaceUse read_surface(Reader& r)
{
    PolyhedralSurfaceUse use;
    const auto kind = r.word();
    if (kind > 1)
        reject("invalid native surface kind.");
    use.key.kind = static_cast<PolyhedralSurfaceKey::Kind>(kind);
    use.key.region = r.word();
    use.key.creator.root = r.word();
    const auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        use.key.creator.path.push_back(r.word());
    use.key.ordinal = r.word();
    const auto orientation = r.word(), exterior = r.word();
    if (orientation > 1 || exterior > 1)
        reject("invalid native surface orientation.");
    use.orientation = orientation ? 1 : -1;
    use.exterior = exterior;
    const auto name = r.packet();
    use.boundary.assign(name.begin(), name.end());
    return use;
}
void write_key(Writer& w, const CellKey& key)
{
    const auto* ancestry = std::get_if<Ancestry>(&key.ancestry);
    if (!ancestry)
        reject("native polyhedral complex needs convex ancestry.");
    w.word(key.region);
    w.word(ancestry->root);
    w.word(ancestry->path.size());
    for (auto p : ancestry->path)
        w.word(p);
}
CellKey read_key(Reader& r)
{
    CellKey key;
    key.region = r.word();
    Ancestry ancestry;
    ancestry.root = r.word();
    const auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        ancestry.path.push_back(r.word());
    key.ancestry = std::move(ancestry);
    return key;
}
MeshUtils::Vec3 vector(const WidePoint& p)
{
    const auto v = display(p);
    return {v[0], v[1], v[2]};
}
MeshUtils::Vec3 vector(GeometryPoint p)
{
    return {p[0], p[1], p[2]};
}
} // namespace ordinary_complex_detail

using namespace ordinary_complex_detail;
struct PolyhedralComplex<ArithmeticMode::Ordinary>::Impl
{
    ID global_cells = 0, global_faces = 0, global_nodes = 0;
    std::map<ID, CellRow> cells;
    std::map<ID, FaceRow> faces;
    std::map<ID, NodeRow> nodes;
    std::vector<ID> cell_ids, resident_ids, face_ids, node_ids;
};

namespace ordinary_complex_detail
{
CellPtr atomic_geometry(const Complex::Impl& data, ID cell)
{
    std::vector<FacePtr> faces;
    for (auto id : data.cells.at(cell).faces)
    {
        const auto& face = data.faces.at(id);
        faces.push_back(face.owner == cell ? face.geometry : face.geometry->reversed());
    }
    return Face::cell_shell(faces);
}
void validate(Complex::Impl& data)
{
    data.cell_ids.clear();
    data.resident_ids.clear();
    data.face_ids.clear();
    data.node_ids.clear();
    std::set<CellKey> keys;
    std::set<ID> used_cells, used_faces, used_nodes;
    for (auto& [id, cell] : data.cells)
    {
        if (id >= data.global_cells || !std::get_if<Ancestry>(&cell.key.ancestry) || !keys.insert(cell.key).second ||
            !std::isfinite(cell.volume) || !(cell.volume > 0) || !std::isfinite(cell.center.x) ||
            !std::isfinite(cell.center.y) || !std::isfinite(cell.center.z))
            reject("invalid native complex cell catalog.");
        data.cell_ids.push_back(id);
        if (cell.geometry)
        {
            data.resident_ids.push_back(id);
            if (cell.geometry->volume().as_double() != cell.volume || vector(cell.geometry->centroid()) != cell.center)
                reject("native cell metric and construction disagree.");
        }
        else if (!cell.faces.empty() || !cell.nodes.empty())
            reject("metric-only native cell retains topology.");
    }
    for (auto& [id, face] : data.faces)
    {
        if (id >= data.global_faces || !face.geometry || !data.cells.contains(face.owner) ||
            (face.neighbor != absent && (!data.cells.contains(face.neighbor) || face.owner >= face.neighbor)) ||
            face.surface.exterior != (face.neighbor == absent) ||
            (face.surface.orientation != 1 && face.surface.orientation != -1) ||
            (!face.surface.exterior && !face.surface.boundary.empty()))
            reject("invalid native complex face incidence.");
        const auto& polygon = face.geometry->points();
        if (polygon.size() != face.nodes.size() || polygon.size() < 3)
            reject("native face node loop does not match its geometry.");
        std::set<ID> unique;
        for (std::size_t i = 0; i < polygon.size(); ++i)
        {
            const auto n = face.nodes[i];
            if (!data.nodes.contains(n) || !unique.insert(n).second ||
                data.nodes.at(n).region != face.surface.key.region || data.nodes.at(n).point != polygon[i])
                reject("native face has foreign/repeated node identity.");
            used_nodes.insert(n);
        }
        const auto normal = face.geometry->area_vector(), center = face.geometry->centroid();
        if (!((center - data.cells.at(face.owner).center).dot(normal) > 0))
            reject("native face has nonpositive owner distance.");
        if (face.neighbor != absent && !((center - data.cells.at(face.neighbor).center).dot(normal) < 0))
            reject("native face has nonnegative neighbor distance.");
        used_cells.insert(face.owner);
        if (face.neighbor != absent)
            used_cells.insert(face.neighbor);
        data.face_ids.push_back(id);
    }
    for (const auto& [id, cell] : data.cells)
        if (cell.geometry)
        {
            std::set<ID> local_faces, local_nodes;
            for (auto f : cell.faces)
            {
                if (!data.faces.contains(f) || !local_faces.insert(f).second)
                    reject("native cell has missing/repeated face incidence.");
                const auto& face = data.faces.at(f);
                if (face.owner != id && face.neighbor != id)
                    reject("native cell has foreign face incidence.");
                used_faces.insert(f);
                local_nodes.insert(face.nodes.begin(), face.nodes.end());
            }
            if (std::vector<ID>(local_nodes.begin(), local_nodes.end()) != cell.nodes)
                reject("native cell node catalog differs from its faces.");
            if (!atomic_geometry(data, id)->equivalent(*cell.geometry))
                reject("native atomic faces changed the source cell shell.");
        }
    for (const auto& [id, face] : data.faces)
        for (auto c : {face.owner, face.neighbor})
            if (c != absent && data.cells.at(c).geometry &&
                std::find(data.cells.at(c).faces.begin(), data.cells.at(c).faces.end(), id) ==
                    data.cells.at(c).faces.end())
                reject("native face lacks reciprocal resident-cell incidence.");
    if (used_cells.size() != data.cells.size() || used_faces.size() != data.faces.size() ||
        used_nodes.size() != data.nodes.size())
        reject("native sparse complex contains orphan records.");
    std::set<std::pair<ID, GeometryPoint>> displayed;
    for (const auto& [id, node] : data.nodes)
    {
        if (id >= data.global_nodes || !displayed.insert({node.region, display(node.point)}).second)
            reject("native node identities collide in published coordinates.");
        data.node_ids.push_back(id);
    }
}
} // namespace ordinary_complex_detail

std::shared_ptr<const Complex> Complex::make(std::vector<PolyhedralComplexCell<ArithmeticMode::Ordinary>> cells)
{
    if (cells.empty())
        reject("native polyhedral complex requires at least one cell.");
    auto data = std::make_shared<Impl>();
    data->global_cells = cells.size();
    struct Incidence
    {
        ID cell;
        FacePtr geometry;
        PolyhedralSurfaceUse surface;
    };
    std::map<PolyhedralSurfaceKey, std::vector<Incidence>> surfaces;
    std::set<CellKey> keys;
    for (const auto& cell : cells)
    {
        if (!cell.geometry || cell.gid >= cells.size() || data->cells.contains(cell.gid) ||
            !keys.insert(cell.key).second || cell.surfaces.size() != cell.geometry->num_faces())
            reject("invalid native complex source cell binding.");
        data->cells.emplace(cell.gid, CellRow{cell.key, cell.geometry, cell.geometry->volume().as_double(),
                                          vector(cell.geometry->centroid()), {}, {}});
        for (std::size_t f = 0; f < cell.surfaces.size(); ++f)
        {
            const auto& surface = cell.surfaces[f];
            if (surface.key.region != cell.key.region || (surface.orientation != 1 && surface.orientation != -1))
                reject("invalid native surface lineage or orientation.");
            auto face = cell.geometry->face_geometry(f);
            if (surface.orientation < 0)
                face = face->reversed();
            surfaces[surface.key].push_back({cell.gid, std::move(face), surface});
        }
    }
    // Retain the sparse broad phase; boxes only reject disjoint candidates.
    struct Box
    {
        std::size_t cell;
        WidePoint lo, hi;
    };
    std::vector<Box> boxes;
    for (std::size_t c = 0; c < cells.size(); ++c)
    {
        const auto points = cells[c].geometry->points();
        Box box{c, points.front().value, points.front().value};
        for (const auto& point : points)
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                box.lo[axis] = std::min(box.lo[axis], point.value[axis]);
                box.hi[axis] = std::max(box.hi[axis], point.value[axis]);
            }
        boxes.push_back(box);
    }
    std::sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) { return a.lo[0] < b.lo[0]; });
    std::vector<std::size_t> active;
    for (std::size_t i = 0; i < boxes.size(); ++i)
    {
        const auto& a = boxes[i];
        std::erase_if(active, [&](auto j) { return boxes[j].hi[0] <= a.lo[0]; });
        for (auto j : active)
        {
            const auto& b = boxes[j];
            if (std::max(a.lo[1], b.lo[1]) < std::min(a.hi[1], b.hi[1]) &&
                std::max(a.lo[2], b.lo[2]) < std::min(a.hi[2], b.hi[2]) &&
                cells[a.cell].geometry->intersect(*cells[b.cell].geometry))
                reject("native complex source interiors overlap.");
        }
        active.push_back(i);
    }
    struct Tile
    {
        PolyhedralSurfaceUse surface;
        ID owner, neighbor;
        FacePtr canonical;
    };
    std::vector<Tile> tiles;
    for (const auto& [key, entries] : surfaces)
    {
        const auto normal = entries.front().geometry->area_vector();
        const auto exterior = entries.front().surface.exterior;
        const auto boundary = entries.front().surface.boundary;
        std::vector<std::size_t> plus, minus;
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            const auto& entry = entries[i];
            if (!(entry.geometry->area_vector().dot(normal) > 0) || entry.surface.exterior != exterior ||
                entry.surface.boundary != boundary)
                reject("native ancestral surface has inconsistent orientation or patch.");
            (entry.surface.orientation > 0 ? plus : minus).push_back(i);
        }
        for (const auto* side : {&plus, &minus})
            for (std::size_t i = 0; i < side->size(); ++i)
                for (std::size_t j = 0; j < i; ++j)
                    if (entries[(*side)[i]].geometry->intersection(*entries[(*side)[j]].geometry))
                        reject("native ancestral face patches overlap on one side.");
        if (exterior)
        {
            if (!plus.empty() && !minus.empty())
                reject("native boundary surface has opposite cell-side orientations.");
            for (const auto& entry : entries)
                tiles.push_back({entry.surface, entry.cell, absent, entry.geometry});
            continue;
        }
        if (plus.empty() || minus.empty())
            reject("native interior surface lacks an incident side.");
        std::vector<std::vector<FacePtr>> coverage(entries.size());
        for (const auto p : plus)
            for (const auto n : minus)
            {
                auto intersection = entries[p].geometry->intersection(*entries[n].geometry);
                if (!intersection)
                    continue;
                if (entries[p].cell == entries[n].cell)
                    reject("native convex cell is its own interface neighbor.");
                coverage[p].push_back(intersection);
                coverage[n].push_back(intersection);
                const auto owner = std::min(entries[p].cell, entries[n].cell);
                const auto neighbor = std::max(entries[p].cell, entries[n].cell);
                tiles.push_back({owner == entries[p].cell ? entries[p].surface : entries[n].surface, owner, neighbor,
                    std::move(intersection)});
            }
        for (std::size_t i = 0; i < entries.size(); ++i)
            entries[i].geometry->certify_partition(coverage[i]);
    }
    std::sort(tiles.begin(), tiles.end(),
        [](const auto& a, const auto& b)
        {
            if (a.surface.key != b.surface.key)
                return a.surface.key < b.surface.key;
            return canonical_face_key(*a.canonical) < canonical_face_key(*b.canonical);
        });
    std::map<ID, std::set<PointIdentity<ArithmeticMode::Ordinary>>> candidates;
    for (const auto& tile : tiles)
        for (auto point : tile.canonical->vertex_identities())
            candidates[tile.surface.key.region].insert(std::move(point));
    std::map<std::pair<ID, PointIdentity<ArithmeticMode::Ordinary>>, ID> node_ids;
    for (auto& tile : tiles)
    {
        const auto& points = candidates.at(tile.surface.key.region);
        const std::vector<PointIdentity<ArithmeticMode::Ordinary>> all(points.begin(), points.end());
        tile.canonical = tile.canonical->with_edge_vertices(all);
        if (tile.surface.orientation < 0)
            tile.canonical = tile.canonical->reversed();
        for (auto point : tile.canonical->vertex_identities())
            node_ids.emplace(std::pair{tile.surface.key.region, std::move(point)}, 0);
    }
    ID next = 0;
    for (auto& [key, id] : node_ids)
    {
        id = next++;
        data->nodes.emplace(id, NodeRow{key.first, read_point(details::GeometryIdentityAccess::key(key.second))});
    }
    data->global_nodes = next;
    for (std::size_t f = 0; f < tiles.size(); ++f)
    {
        auto& tile = tiles[f];
        FaceRow row{tile.surface, tile.owner, tile.neighbor, tile.canonical, {}};
        for (const auto& point : tile.canonical->vertex_identities())
            row.nodes.push_back(node_ids.at({tile.surface.key.region, point}));
        data->faces.emplace(f, std::move(row));
        data->cells.at(tile.owner).faces.push_back(f);
        if (tile.neighbor != absent)
            data->cells.at(tile.neighbor).faces.push_back(f);
    }
    data->global_faces = tiles.size();
    for (auto& [id, cell] : data->cells)
    {
        std::set<ID> nodes;
        for (auto f : cell.faces)
            nodes.insert(data->faces.at(f).nodes.begin(), data->faces.at(f).nodes.end());
        cell.nodes.assign(nodes.begin(), nodes.end());
    }
    validate(*data);
    return std::shared_ptr<const Complex>(new Complex(std::move(data)));
}

std::uint64_t Complex::global_cells() const noexcept
{
    return impl_->global_cells;
}
std::uint64_t Complex::global_faces() const noexcept
{
    return impl_->global_faces;
}
std::uint64_t Complex::global_nodes() const noexcept
{
    return impl_->global_nodes;
}
const std::vector<ID>& Complex::cell_ids() const noexcept
{
    return impl_->cell_ids;
}
const std::vector<ID>& Complex::resident_cell_ids() const noexcept
{
    return impl_->resident_ids;
}
const std::vector<ID>& Complex::face_ids() const noexcept
{
    return impl_->face_ids;
}
const std::vector<ID>& Complex::node_ids() const noexcept
{
    return impl_->node_ids;
}
bool Complex::has_full_cell(ID id) const noexcept
{
    return impl_->cells.contains(id) && bool(impl_->cells.at(id).geometry);
}
const std::vector<ID>& Complex::cell_faces(ID id) const
{
    return impl_->cells.at(id).faces;
}
const std::vector<ID>& Complex::cell_nodes(ID id) const
{
    return impl_->cells.at(id).nodes;
}
const std::vector<ID>& Complex::face_nodes(ID id) const
{
    return impl_->faces.at(id).nodes;
}
ID Complex::owner_cell(ID id) const
{
    return impl_->faces.at(id).owner;
}
ID Complex::neighbor_cell(ID id) const
{
    return impl_->faces.at(id).neighbor;
}
const CellKey& Complex::cell_key(ID id) const
{
    return impl_->cells.at(id).key;
}
const PolyhedralSurfaceUse& Complex::face_surface(ID id) const
{
    return impl_->faces.at(id).surface;
}
MeshUtils::Vec3 Complex::node_coordinates(ID id) const
{
    return vector(impl_->nodes.at(id).point);
}
double Complex::cell_volume(ID id) const
{
    return impl_->cells.at(id).volume;
}
MeshUtils::Vec3 Complex::cell_centroid(ID id) const
{
    return impl_->cells.at(id).center;
}
double Complex::face_area(ID id) const
{
    return impl_->faces.at(id).geometry->area().as_double();
}
MeshUtils::Vec3 Complex::face_centroid(ID id) const
{
    return impl_->faces.at(id).geometry->centroid();
}
MeshUtils::Vec3 Complex::face_area_vector(ID id) const
{
    return impl_->faces.at(id).geometry->area_vector();
}
CellPtr Complex::cell_geometry(ID id) const
{
    const auto& cell = impl_->cells.at(id);
    if (!cell.geometry)
        reject("metric-only native neighbor has no retained cell geometry.");
    return cell.geometry;
}
CellPtr Complex::atomic_cell_geometry(ID id) const
{
    return atomic_geometry(*impl_, id);
}
FacePtr Complex::face_geometry(ID id) const
{
    return impl_->faces.at(id).geometry;
}
std::vector<char> Complex::face_identity(ID id) const
{
    const auto& face = impl_->faces.at(id);
    Writer w;
    auto use = face.surface;
    use.orientation = 1;
    write_surface(w, use);
    w.packet(canonical_face_key(*(face.surface.orientation == 1 ? face.geometry : face.geometry->reversed())));
    return std::move(w.bytes);
}
std::vector<char> Complex::encode() const
{
    return encode(impl_->resident_ids);
}
std::vector<char> Complex::encode(std::span<const ID> visible) const
{
    std::set<ID> full, cells, faces, nodes;
    for (auto c : visible)
    {
        if (!has_full_cell(c) || !full.insert(c).second)
            reject("invalid native sparse cell selection.");
        cells.insert(c);
        faces.insert(impl_->cells.at(c).faces.begin(), impl_->cells.at(c).faces.end());
    }
    for (auto f : faces)
    {
        const auto& face = impl_->faces.at(f);
        cells.insert(face.owner);
        if (face.neighbor != absent)
            cells.insert(face.neighbor);
        nodes.insert(face.nodes.begin(), face.nodes.end());
    }
    Writer w;
    w.word(0x53464650434f4d32ULL);
    w.word(1);
    w.word(impl_->global_cells);
    w.word(impl_->global_faces);
    w.word(impl_->global_nodes);
    w.word(cells.size());
    for (auto c : cells)
    {
        const auto& cell = impl_->cells.at(c);
        w.word(c);
        write_key(w, cell.key);
        w.real(cell.volume);
        w.real(cell.center.x);
        w.real(cell.center.y);
        w.real(cell.center.z);
        w.word(full.contains(c));
        if (full.contains(c))
        {
            w.packet(cell.geometry->encode_certificate());
            w.word(cell.faces.size());
            for (auto f : cell.faces)
                w.word(f);
            w.word(cell.nodes.size());
            for (auto n : cell.nodes)
                w.word(n);
        }
    }
    w.word(faces.size());
    for (auto f : faces)
    {
        const auto& face = impl_->faces.at(f);
        w.word(f);
        write_surface(w, face.surface);
        w.word(face.owner);
        w.word(face.neighbor);
        w.packet(face.geometry->encode_incidence());
        w.word(face.nodes.size());
        for (auto n : face.nodes)
            w.word(n);
    }
    w.word(nodes.size());
    for (auto n : nodes)
    {
        w.word(n);
        w.word(impl_->nodes.at(n).region);
        w.point(impl_->nodes.at(n).point);
    }
    return std::move(w.bytes);
}
std::shared_ptr<const Complex> Complex::decode(std::span<const char> packet)
{
    Reader r{packet};
    if (r.word() != 0x53464650434f4d32ULL || r.word() != 1)
        reject("foreign native complex packet mode/version.");
    auto data = std::make_shared<Impl>();
    data->global_cells = r.word();
    data->global_faces = r.word();
    data->global_nodes = r.word();
    if (!data->global_cells || !data->global_faces || !data->global_nodes)
        reject("native complex global catalog is empty.");
    auto count = r.count();
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto id = r.word();
        CellRow cell;
        cell.key = read_key(r);
        cell.volume = GeometryValue<ArithmeticMode::Ordinary>{r.real()}.as_double();
        const auto center = display(r.point());
        cell.center = vector(center);
        const auto full = r.word();
        if (full > 1)
            reject("invalid native cell residency flag.");
        if (full)
        {
            cell.geometry = CellGeometry::decode_certificate(r.packet());
            auto n = r.count();
            for (std::size_t j = 0; j < n; ++j)
                cell.faces.push_back(r.word());
            n = r.count();
            for (std::size_t j = 0; j < n; ++j)
                cell.nodes.push_back(r.word());
        }
        if (!data->cells.emplace(id, std::move(cell)).second)
            reject("duplicate native cell ID.");
    }
    count = r.count();
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto id = r.word();
        FaceRow face;
        face.surface = read_surface(r);
        face.owner = r.word();
        face.neighbor = r.word();
        face.geometry = Face::decode(r.packet());
        const auto n = r.count();
        for (std::size_t j = 0; j < n; ++j)
            face.nodes.push_back(r.word());
        if (!data->faces.emplace(id, std::move(face)).second)
            reject("duplicate native face ID.");
    }
    count = r.count();
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto id = r.word(), region = r.word();
        if (!data->nodes.emplace(id, NodeRow{region, r.point()}).second)
            reject("duplicate native node ID.");
    }
    r.finish();
    validate(*data);
    return std::shared_ptr<const Complex>(new Complex(std::move(data)));
}
std::size_t Complex::storage_bytes() const
{
    std::size_t bytes = sizeof(*this) + sizeof(Impl);
    std::set<const CellGeometry*> geometry;
    for (const auto& [id, cell] : impl_->cells)
    {
        bytes += sizeof(cell) + sizeof(ID) + 4 * sizeof(void*) +
                 (cell.faces.capacity() + cell.nodes.capacity()) * sizeof(ID);
        bytes += std::get<Ancestry>(cell.key.ancestry).path.capacity() * sizeof(ID);
        if (cell.geometry && geometry.insert(cell.geometry.get()).second)
            bytes += cell.geometry->storage_bytes();
    }
    for (const auto& [id, face] : impl_->faces)
        bytes += sizeof(face) + sizeof(ID) + 4 * sizeof(void*) + face.nodes.capacity() * sizeof(ID) +
                 face.surface.boundary.capacity() + face.surface.key.creator.path.capacity() * sizeof(ID) +
                 face.geometry->storage_bytes();
    bytes += impl_->nodes.size() * (sizeof(NodeRow) + sizeof(ID) + 4 * sizeof(void*));
    bytes += (impl_->cell_ids.capacity() + impl_->resident_ids.capacity() + impl_->face_ids.capacity() +
                 impl_->node_ids.capacity()) *
             sizeof(ID);
    return bytes;
}
} // namespace SimpleFluid::PLIC
