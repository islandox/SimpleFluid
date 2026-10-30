#include "geometry/PLIC/details/exact/ConvexCellGeometryInternal.hh"

namespace SimpleFluid::PLIC
{
namespace exact_geometry_detail
{
P polygon_vector(const Polygon& polygon)
{
    P area{};
    if (polygon.size() < 3)
        return area;
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
        area = add(area, cross(sub(polygon[i], polygon[0]), sub(polygon[i + 1], polygon[0])));
    return mul(area, Q(1, 2));
}

Polygon minimal_polygon(Polygon polygon)
{
    bool changed = true;
    while (changed && polygon.size() >= 3)
    {
        changed = false;
        for (std::size_t i = 0; i < polygon.size(); ++i)
            if (zero(cross(sub(polygon[i], polygon[(i + polygon.size() - 1) % polygon.size()]),
                    sub(polygon[(i + 1) % polygon.size()], polygon[i]))))
            {
                polygon.erase(polygon.begin() + i);
                changed = true;
                break;
            }
    }
    if (polygon.size() < 3 || zero(polygon_vector(polygon)))
        return {};
    auto first = std::min_element(polygon.begin(), polygon.end());
    std::rotate(polygon.begin(), first, polygon.end());
    return polygon;
}

Polygon polygon_intersection(Polygon polygon, const Polygon& clipper, const P& normal)
{
    for (std::size_t e = 0; e < clipper.size() && !polygon.empty(); ++e)
    {
        const Plane plane{cross(sub(clipper[(e + 1) % clipper.size()], clipper[e]), normal), clipper[e]};
        Polygon result;
        auto append = [&](const P& p)
        {
            if (result.empty() || result.back() != p)
                result.push_back(p);
        };
        for (std::size_t i = 0; i < polygon.size(); ++i)
        {
            const auto& a = polygon[i];
            const auto& b = polygon[(i + 1) % polygon.size()];
            const Q da = distance(plane, a), db = distance(plane, b);
            if (da <= 0)
                append(a);
            if ((da < 0 && db > 0) || (da > 0 && db < 0))
                append(add(a, mul(sub(b, a), Q(da / (da - db)))));
        }
        if (result.size() > 1 && result.front() == result.back())
            result.pop_back();
        polygon = std::move(result);
    }
    return minimal_polygon(std::move(polygon));
}

FaceMoment polygon_moments(const Polygon& polygon, std::size_t axis, const P& origin)
{
    FaceMoment result{};
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
    {
        const auto a = sub(polygon[0], origin), b = sub(polygon[i], origin), c = sub(polygon[i + 1], origin);
        const Q area = cross(sub(b, a), sub(c, a))[axis] / 2;
        result.area += area;
        result.first = add(result.first, mul(add(add(a, b), c), Q(area / 3)));
    }
    return result;
}

P polygon_center(const Polygon& polygon)
{
    const auto normal = polygon_vector(polygon);
    std::size_t axis = 0;
    while (axis < 3 && normal[axis] == 0)
        ++axis;
    if (axis == 3)
        throw std::invalid_argument("Exact face has zero area.");
    const auto m = polygon_moments(polygon, axis, polygon[0]);
    return add(polygon[0], mul(m.first, Q(1 / m.area)));
}

void surface_write(WireWriter& w, const PolyhedralSurfaceUse& surface)
{
    w.word(static_cast<std::uint64_t>(surface.key.kind));
    w.word(surface.key.region);
    w.word(surface.key.creator.root);
    w.word(surface.key.creator.path.size());
    for (auto v : surface.key.creator.path)
        w.word(v);
    w.word(surface.key.ordinal);
    w.word(surface.orientation == 1 ? 1 : 0);
    w.word(surface.exterior);
    w.word(surface.boundary.size());
    for (unsigned char ch : surface.boundary)
        w.data.push_back(static_cast<char>(ch));
}

PolyhedralSurfaceUse surface_read(WireReader& r)
{
    PolyhedralSurfaceUse surface;
    auto kind = r.word();
    if (kind > 1)
        throw std::invalid_argument("Unknown exact surface ancestry kind.");
    surface.key.kind = static_cast<PolyhedralSurfaceKey::Kind>(kind);
    surface.key.region = r.word();
    surface.key.creator.root = r.word();
    auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        surface.key.creator.path.push_back(r.word());
    surface.key.ordinal = r.word();
    auto orientation = r.word(), exterior = r.word();
    if (orientation > 1 || exterior > 1)
        throw std::invalid_argument("Invalid exact surface incidence.");
    surface.orientation = orientation ? 1 : -1;
    surface.exterior = exterior;
    n = r.word();
    if (n > r.data.size() - r.position)
        throw std::invalid_argument("Invalid exact boundary string.");
    surface.boundary.assign(r.data.begin() + r.position, r.data.begin() + r.position + n);
    r.position += n;
    return surface;
}

void cell_key_write(WireWriter& w, const ::SimpleFluid::Meshes::AMR::AdaptiveCellKey& key)
{
    const auto* ancestry = std::get_if<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(&key.ancestry);
    if (!ancestry)
        throw std::invalid_argument("Exact polyhedral complex needs convex-cell ancestry.");
    w.word(key.region);
    w.word(ancestry->root);
    w.word(ancestry->path.size());
    for (auto p : ancestry->path)
        w.word(p);
}

::SimpleFluid::Meshes::AMR::AdaptiveCellKey cell_key_read(WireReader& r)
{
    ::SimpleFluid::Meshes::AMR::AdaptiveCellKey key;
    key.region = r.word();
    ::SimpleFluid::Meshes::AMR::ConvexCellAncestry ancestry;
    ancestry.root = r.word();
    auto n = r.count();
    for (std::size_t i = 0; i < n; ++i)
        ancestry.path.push_back(r.word());
    key.ancestry = std::move(ancestry);
    return key;
}

void complex_validate(PolyhedralComplex<ArithmeticMode::Exact>::Impl& impl)
{
    impl.cell_ids.clear();
    impl.resident_ids.clear();
    impl.face_ids.clear();
    impl.node_ids.clear();
    std::set<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> cell_keys;
    for (auto& [id, cell] : impl.cells)
    {
        if (id >= impl.global_cells ||
            !std::get_if<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(&cell.key.ancestry) ||
            !std::isfinite(cell.volume) || !(cell.volume > 0) || !std::isfinite(cell.center.x) ||
            !std::isfinite(cell.center.y) || !std::isfinite(cell.center.z))
            throw std::invalid_argument("Invalid exact complex cell metric/catalog record.");
        if (!cell_keys.insert(cell.key).second)
            throw std::invalid_argument("Exact complex cell ancestry is duplicated.");
        impl.cell_ids.push_back(id);
        if (cell.geometry)
            impl.resident_ids.push_back(id);
        else if (!cell.faces.empty() || !cell.nodes.empty())
            throw std::invalid_argument("Metric-only exact cell cannot retain topology.");
    }
    std::set<ComplexID> used_nodes, used_faces, used_cells;
    for (auto& [id, face] : impl.faces)
    {
        if (id >= impl.global_faces || !impl.cells.contains(face.owner) ||
            (face.neighbor != absent_cell && (!impl.cells.contains(face.neighbor) || face.owner >= face.neighbor)) ||
            face.surface.exterior != (face.neighbor == absent_cell) ||
            (face.surface.orientation != 1 && face.surface.orientation != -1))
            throw std::invalid_argument("Invalid exact complex face incidence/catalog.");
        if (!face.surface.exterior && !face.surface.boundary.empty())
            throw std::invalid_argument("Exact complex physical boundary identity is inconsistent.");
        used_cells.insert(face.owner);
        if (face.neighbor != absent_cell)
            used_cells.insert(face.neighbor);
        Polygon polygon;
        std::set<ComplexID> unique;
        for (auto node : face.nodes)
        {
            if (!impl.nodes.contains(node) || !unique.insert(node).second ||
                impl.nodes.at(node).region != face.surface.key.region)
                throw std::invalid_argument("Invalid exact face vertex identity.");
            polygon.push_back(impl.nodes.at(node).point);
            used_nodes.insert(node);
        }
        if (polygon.size() < 3)
            throw std::invalid_argument("Exact face has too few vertices.");
        const auto area = polygon_vector(polygon);
        if (zero(area))
            throw std::invalid_argument("Exact face is degenerate.");
        const Plane plane{area, polygon[0]};
        for (const auto& p : polygon)
            if (distance(plane, p) != 0)
                throw std::invalid_argument("Exact complex face is nonplanar.");
        for (std::size_t e = 0; e < polygon.size(); ++e)
            for (const auto& p : polygon)
                if (dot(area, cross(sub(polygon[(e + 1) % polygon.size()], polygon[e]), sub(p, polygon[e]))) < 0)
                    throw std::invalid_argument("Exact face is nonsimple or nonconvex.");
        face.area = square_root(dot(area, area)).as_double();
        face.vector = rounded_point(area);
        face.center = rounded_point(polygon_center(polygon));
        const auto owner_distance = (face.center - impl.cells.at(face.owner).center).dot(face.vector);
        if (!std::isfinite(owner_distance) || !(owner_distance > 0))
            throw std::invalid_argument("Exact face has an unresolved signed native owner distance.");
        if (face.neighbor != absent_cell)
        {
            const auto distance = (face.center - impl.cells.at(face.neighbor).center).dot(face.vector);
            if (!std::isfinite(distance) || !(distance < 0))
                throw std::invalid_argument("Exact face has an unresolved signed native neighbor distance.");
        }
        impl.face_ids.push_back(id);
    }
    for (auto& [id, cell] : impl.cells)
        if (cell.geometry)
        {
            Poly shell;
            std::map<ComplexID, std::size_t> vertices;
            std::set<ComplexID> face_ids, nodes;
            for (auto face_id : cell.faces)
            {
                if (!impl.faces.contains(face_id) || !face_ids.insert(face_id).second)
                    throw std::invalid_argument("Exact resident cell has missing/repeated face.");
                const auto& face = impl.faces.at(face_id);
                if (face.owner != id && face.neighbor != id)
                    throw std::invalid_argument("Exact resident cell has nonincident face.");
                used_faces.insert(face_id);
                Face loop;
                for (auto node : face.nodes)
                {
                    const auto [it, inserted] = vertices.emplace(node, shell.vertices.size());
                    if (inserted)
                        shell.vertices.push_back(impl.nodes.at(node).point);
                    loop.push_back(it->second);
                    nodes.insert(node);
                }
                if (face.owner != id)
                    std::reverse(loop.begin(), loop.end());
                shell.faces.push_back(std::move(loop));
            }
            validate(shell);
            // Access to the authoritative shell is checked by the member caller;
            // closure and moments are certified here without narrowing vertices.
            std::vector<ComplexID> expected_nodes(nodes.begin(), nodes.end());
            if (cell.nodes != expected_nodes)
                throw std::invalid_argument("Exact cell node catalog disagrees with its incident faces.");
        }
    for (const auto& [face_id, face] : impl.faces)
        for (const auto incident : {face.owner, face.neighbor})
            if (incident != absent_cell && impl.cells.at(incident).geometry &&
                std::find(impl.cells.at(incident).faces.begin(), impl.cells.at(incident).faces.end(), face_id) ==
                    impl.cells.at(incident).faces.end())
                throw std::invalid_argument("Exact face is missing reciprocal membership in its full incident cell.");
    if (used_faces.size() != impl.faces.size() || used_nodes.size() != impl.nodes.size() ||
        used_cells.size() != impl.cells.size())
        throw std::invalid_argument("Exact sparse complex contains orphan cell/face/node records.");
    std::set<std::pair<std::uint64_t, GeometryPoint>> display_points;
    for (const auto& [id, node] : impl.nodes)
    {
        if (id >= impl.global_nodes)
            throw std::invalid_argument("Exact complex node ID exceeds global catalog.");
        const auto display = rounded_point(node.point);
        if (!display_points.insert({node.region, {display.x, display.y, display.z}}).second)
            throw std::invalid_argument("Distinct exact explicit vertices collide in native display coordinates.");
        impl.node_ids.push_back(id);
    }
}
} // namespace exact_geometry_detail
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{
std::shared_ptr<const PolyhedralComplex<ArithmeticMode::Exact>> PolyhedralComplex<ArithmeticMode::Exact>::make(
    std::vector<PolyhedralComplexCell<ArithmeticMode::Exact>> input)
{
    if (input.empty())
        throw std::invalid_argument("Exact complex root needs at least one actual convex cell.");
    auto impl = std::make_shared<Impl>();
    impl->global_cells = input.size();
    struct Incident
    {
        ComplexID cell;
        Polygon polygon;
        PolyhedralSurfaceUse use;
    };
    std::map<PolyhedralSurfaceKey, std::vector<Incident>> surfaces;
    std::set<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> keys;
    std::map<std::uint64_t, std::set<P>> candidate_vertices;
    struct Box
    {
        ComplexID cell;
        P lo, hi;
    };
    std::vector<Box> boxes;
    for (const auto& cell : input)
    {
        if (!cell.geometry || cell.gid >= input.size() || impl->cells.contains(cell.gid) ||
            !keys.insert(cell.key).second || cell.surfaces.size() != cell.geometry->impl_->poly.faces.size() ||
            !std::get_if<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(&cell.key.ancestry))
            throw std::invalid_argument("Exact complex cell binding/topology is invalid.");
        const auto& poly = cell.geometry->impl_->poly;
        const auto m = exact_moments(poly, point(cell.geometry->impl_->origin));
        const auto center = add(point(cell.geometry->impl_->origin), mul(m.first, Q(1 / m.volume)));
        impl->cells.emplace(cell.gid,
            ComplexCellRow{cell.key, cell.geometry, measured(m.volume).as_double(), rounded_point(center), {}, {}});
        Box box{cell.gid, poly.vertices[0], poly.vertices[0]};
        for (const auto& p : poly.vertices)
        {
            candidate_vertices[cell.key.region].insert(p);
            for (std::size_t a = 0; a < 3; ++a)
            {
                box.lo[a] = std::min(box.lo[a], p[a]);
                box.hi[a] = std::max(box.hi[a], p[a]);
            }
        }
        boxes.push_back(std::move(box));
        for (std::size_t f = 0; f < poly.faces.size(); ++f)
        {
            const auto& use = cell.surfaces[f];
            if (use.key.region != cell.key.region || (use.orientation != 1 && use.orientation != -1))
                throw std::invalid_argument("Surface lineage/region/orientation is invalid.");
            Polygon polygon;
            for (auto index : poly.faces[f])
                polygon.push_back(poly.vertices[index]);
            if (use.orientation < 0)
                std::reverse(polygon.begin(), polygon.end());
            surfaces[use.key].push_back({cell.gid, minimal_polygon(std::move(polygon)), use});
        }
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
                intersection(
                    impl->cells.at(a.cell).geometry->impl_->poly, impl->cells.at(b.cell).geometry->impl_->poly))
                throw std::invalid_argument("Exact complex root/leaf interiors overlap.");
        }
        active.push_back(i);
    }
    struct Tile
    {
        PolyhedralSurfaceUse use;
        ComplexID owner, neighbor;
        Polygon canonical;
    };
    std::vector<Tile> tiles;
    for (const auto& [key, entries] : surfaces)
    {
        const auto normal = polygon_vector(entries[0].polygon);
        const auto anchor = entries[0].polygon[0];
        const bool exterior = entries[0].use.exterior;
        const auto boundary = entries[0].use.boundary;
        std::vector<std::size_t> plus, minus;
        for (std::size_t i = 0; i < entries.size(); ++i)
        {
            const auto& entry = entries[i];
            if (entry.polygon.empty() || dot(polygon_vector(entry.polygon), normal) <= 0 ||
                entry.use.exterior != exterior || entry.use.boundary != boundary)
                throw std::invalid_argument("Exact ancestral surface has inconsistent normal/patch membership.");
            for (const auto& p : entry.polygon)
                if (dot(normal, sub(p, anchor)) != 0)
                    throw std::invalid_argument("One exact surface ancestry refers to different represented planes.");
            (entry.use.orientation > 0 ? plus : minus).push_back(i);
        }
        auto certify_side = [&](const auto& side)
        {
            for (std::size_t i = 0; i < side.size(); ++i)
                for (std::size_t j = 0; j < i; ++j)
                    if (!polygon_intersection(entries[side[i]].polygon, entries[side[j]].polygon, normal).empty())
                        throw std::invalid_argument("Same-side ancestral face patches overlap.");
        };
        certify_side(plus);
        certify_side(minus);
        if (exterior)
        {
            if (!plus.empty() && !minus.empty())
                throw std::invalid_argument("Physical boundary ancestry has both cell-side orientations.");
            for (const auto& e : entries)
                tiles.push_back({e.use, e.cell, absent_cell, e.polygon});
        }
        else
        {
            if (plus.empty() || minus.empty())
                throw std::invalid_argument("Interior ancestral surface has a missing incident side.");
            std::vector<std::vector<Polygon>> coverage(entries.size());
            for (auto p : plus)
                for (auto n : minus)
                {
                    auto polygon = polygon_intersection(entries[p].polygon, entries[n].polygon, normal);
                    if (polygon.empty())
                        continue;
                    if (entries[p].cell == entries[n].cell)
                        throw std::invalid_argument("A convex cell cannot be its own interface neighbor.");
                    coverage[p].push_back(polygon);
                    coverage[n].push_back(polygon);
                    const auto owner = std::min(entries[p].cell, entries[n].cell),
                               neighbor = std::max(entries[p].cell, entries[n].cell);
                    auto use = entries[p].cell == owner ? entries[p].use : entries[n].use;
                    tiles.push_back({std::move(use), owner, neighbor, std::move(polygon)});
                }
            std::size_t axis = 0;
            while (normal[axis] == 0)
                ++axis;
            for (std::size_t i = 0; i < entries.size(); ++i)
            {
                const auto origin = entries[i].polygon[0];
                const auto expected = polygon_moments(entries[i].polygon, axis, origin);
                FaceMoment sum{};
                for (const auto& piece : coverage[i])
                {
                    auto m = polygon_moments(piece, axis, origin);
                    sum.area += m.area;
                    sum.first = add(sum.first, m.first);
                }
                if (sum.area != expected.area || sum.first != expected.first)
                    throw std::invalid_argument("Exact shared-face overlay has a real represented gap.");
            }
        }
    }
    std::sort(tiles.begin(), tiles.end(),
        [](const auto& a, const auto& b)
        {
            if (a.use.key != b.use.key)
                return a.use.key < b.use.key;
            return a.canonical < b.canonical;
        });
    for (const auto& tile : tiles)
        for (const auto& p : tile.canonical)
            candidate_vertices[tile.use.key.region].insert(p);
    std::vector<Polygon> polygons;
    for (const auto& tile : tiles)
    {
        Polygon polygon;
        for (std::size_t e = 0; e < tile.canonical.size(); ++e)
        {
            const auto& a = tile.canonical[e];
            const auto& b = tile.canonical[(e + 1) % tile.canonical.size()];
            const auto d = sub(b, a);
            const Q length = dot(d, d);
            std::vector<std::pair<Q, P>> interior;
            for (const auto& p : candidate_vertices[tile.use.key.region])
            {
                const auto offset = sub(p, a);
                const Q along = dot(offset, d);
                if (zero(cross(offset, d)) && along > 0 && along < length)
                    interior.emplace_back(along, p);
            }
            std::sort(interior.begin(), interior.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            polygon.push_back(a);
            for (const auto& [along, p] : interior)
                polygon.push_back(p);
        }
        if (tile.use.orientation < 0)
            std::reverse(polygon.begin(), polygon.end());
        polygons.push_back(std::move(polygon));
    }
    std::map<std::pair<std::uint64_t, P>, ComplexID> node_ids;
    for (std::size_t f = 0; f < tiles.size(); ++f)
        for (const auto& p : polygons[f])
            node_ids.emplace(std::pair{tiles[f].use.key.region, p}, 0);
    ComplexID node_id = 0;
    for (auto& [key, id] : node_ids)
    {
        id = node_id;
        impl->nodes.emplace(node_id++, ComplexNodeRow{key.first, key.second});
    }
    impl->global_nodes = node_id;
    for (std::size_t f = 0; f < tiles.size(); ++f)
    {
        const auto& tile = tiles[f];
        ComplexFaceRow face;
        face.surface = tile.use;
        face.owner = tile.owner;
        face.neighbor = tile.neighbor;
        for (const auto& p : polygons[f])
            face.nodes.push_back(node_ids.at({tile.use.key.region, p}));
        impl->faces.emplace(f, std::move(face));
        impl->cells.at(tile.owner).faces.push_back(f);
        if (tile.neighbor != absent_cell)
            impl->cells.at(tile.neighbor).faces.push_back(f);
    }
    impl->global_faces = tiles.size();
    for (auto& [id, cell] : impl->cells)
    {
        std::set<ComplexID> nodes;
        for (auto face : cell.faces)
            for (auto node : impl->faces.at(face).nodes)
                nodes.insert(node);
        cell.nodes.assign(nodes.begin(), nodes.end());
    }
    complex_validate(*impl);
    for (const auto& [id, cell] : impl->cells)
    {
        Poly reconstructed;
        std::map<ComplexID, std::size_t> vertices;
        for (auto face_id : cell.faces)
        {
            const auto& face = impl->faces.at(face_id);
            Face loop;
            for (auto n : face.nodes)
            {
                auto [it, inserted] = vertices.emplace(n, reconstructed.vertices.size());
                if (inserted)
                    reconstructed.vertices.push_back(impl->nodes.at(n).point);
                loop.push_back(it->second);
            }
            if (face.owner != id)
                std::reverse(loop.begin(), loop.end());
            reconstructed.faces.push_back(std::move(loop));
        }
        if (!contained(reconstructed, cell.geometry->impl_->poly) ||
            !contained(cell.geometry->impl_->poly, reconstructed))
            throw std::invalid_argument("Atomic face closure changed the canonical convex cell shell.");
    }
    return std::shared_ptr<const PolyhedralComplex<ArithmeticMode::Exact>>(
        new PolyhedralComplex<ArithmeticMode::Exact>(std::move(impl)));
}

std::uint64_t PolyhedralComplex<ArithmeticMode::Exact>::global_cells() const noexcept
{
    return impl_->global_cells;
}

std::uint64_t PolyhedralComplex<ArithmeticMode::Exact>::global_faces() const noexcept
{
    return impl_->global_faces;
}

std::uint64_t PolyhedralComplex<ArithmeticMode::Exact>::global_nodes() const noexcept
{
    return impl_->global_nodes;
}

const std::vector<ComplexID>& PolyhedralComplex<ArithmeticMode::Exact>::cell_ids() const noexcept
{
    return impl_->cell_ids;
}

const std::vector<ComplexID>& PolyhedralComplex<ArithmeticMode::Exact>::resident_cell_ids() const noexcept
{
    return impl_->resident_ids;
}

const std::vector<ComplexID>& PolyhedralComplex<ArithmeticMode::Exact>::face_ids() const noexcept
{
    return impl_->face_ids;
}

const std::vector<ComplexID>& PolyhedralComplex<ArithmeticMode::Exact>::node_ids() const noexcept
{
    return impl_->node_ids;
}

bool PolyhedralComplex<ArithmeticMode::Exact>::has_full_cell(ComplexID id) const noexcept
{
    return impl_->cells.contains(id) && bool(impl_->cells.at(id).geometry);
}

const std::vector<ComplexID>& PolyhedralComplex<ArithmeticMode::Exact>::cell_faces(ComplexID id) const
{
    return impl_->cells.at(id).faces;
}

const std::vector<ComplexID>& PolyhedralComplex<ArithmeticMode::Exact>::cell_nodes(ComplexID id) const
{
    return impl_->cells.at(id).nodes;
}

const std::vector<ComplexID>& PolyhedralComplex<ArithmeticMode::Exact>::face_nodes(ComplexID id) const
{
    return impl_->faces.at(id).nodes;
}

ComplexID PolyhedralComplex<ArithmeticMode::Exact>::owner_cell(ComplexID id) const
{
    return impl_->faces.at(id).owner;
}

ComplexID PolyhedralComplex<ArithmeticMode::Exact>::neighbor_cell(ComplexID id) const
{
    return impl_->faces.at(id).neighbor;
}

const ::SimpleFluid::Meshes::AMR::AdaptiveCellKey& PolyhedralComplex<ArithmeticMode::Exact>::cell_key(
    ComplexID id) const
{
    return impl_->cells.at(id).key;
}

const PolyhedralSurfaceUse& PolyhedralComplex<ArithmeticMode::Exact>::face_surface(ComplexID id) const
{
    return impl_->faces.at(id).surface;
}

MeshUtils::Vec3 PolyhedralComplex<ArithmeticMode::Exact>::node_coordinates(ComplexID id) const
{
    return rounded_point(impl_->nodes.at(id).point);
}

double PolyhedralComplex<ArithmeticMode::Exact>::cell_volume(ComplexID id) const
{
    return impl_->cells.at(id).volume;
}

MeshUtils::Vec3 PolyhedralComplex<ArithmeticMode::Exact>::cell_centroid(ComplexID id) const
{
    return impl_->cells.at(id).center;
}

double PolyhedralComplex<ArithmeticMode::Exact>::face_area(ComplexID id) const
{
    return impl_->faces.at(id).area;
}

MeshUtils::Vec3 PolyhedralComplex<ArithmeticMode::Exact>::face_centroid(ComplexID id) const
{
    return impl_->faces.at(id).center;
}

MeshUtils::Vec3 PolyhedralComplex<ArithmeticMode::Exact>::face_area_vector(ComplexID id) const
{
    return impl_->faces.at(id).vector;
}

ConvexGeometryPtr<ArithmeticMode::Exact> PolyhedralComplex<ArithmeticMode::Exact>::cell_geometry(ComplexID id) const
{
    const auto& geometry = impl_->cells.at(id).geometry;
    if (!geometry)
        throw std::invalid_argument("Metric-only exact complex neighbor has no resident cell certificate.");
    return geometry;
}

std::vector<char> PolyhedralComplex<ArithmeticMode::Exact>::face_identity(ComplexID id) const
{
    const auto& face = impl_->faces.at(id);
    WireWriter w;
    auto surface = face.surface;
    surface.orientation = 1;
    surface_write(w, surface);
    Polygon polygon;
    for (auto node : face.nodes)
        polygon.push_back(impl_->nodes.at(node).point);
    if (face.surface.orientation < 0)
        std::reverse(polygon.begin(), polygon.end());
    polygon = minimal_polygon(std::move(polygon));
    w.word(polygon.size());
    for (const auto& p : polygon)
        for (const auto& q : p)
            w.rational(q);
    return std::move(w.data);
}

std::vector<char> PolyhedralComplex<ArithmeticMode::Exact>::encode() const
{
    return encode(impl_->resident_ids);
}

std::vector<char> PolyhedralComplex<ArithmeticMode::Exact>::encode(std::span<const ComplexID> visible) const
{
    std::set<ComplexID> full, cells, faces, nodes;
    for (auto id : visible)
    {
        if (!has_full_cell(id) || !full.insert(id).second)
            throw std::invalid_argument("Exact complex sparse selection is missing/duplicate/nonresident.");
        cells.insert(id);
        for (auto f : impl_->cells.at(id).faces)
            faces.insert(f);
    }
    for (auto f : faces)
    {
        const auto& face = impl_->faces.at(f);
        cells.insert(face.owner);
        if (face.neighbor != absent_cell)
            cells.insert(face.neighbor);
        for (auto n : face.nodes)
            nodes.insert(n);
    }
    WireWriter w;
    w.word(0x5346504f4c594331ULL);
    w.word(2);
    w.word(impl_->global_cells);
    w.word(impl_->global_faces);
    w.word(impl_->global_nodes);
    w.word(cells.size());
    for (auto id : cells)
    {
        const auto& cell = impl_->cells.at(id);
        w.word(id);
        cell_key_write(w, cell.key);
        w.real(cell.volume);
        w.real(cell.center.x);
        w.real(cell.center.y);
        w.real(cell.center.z);
        w.word(full.contains(id));
        if (full.contains(id))
        {
            const auto certificate = cell.geometry->encode_certificate();
            w.word(certificate.size());
            w.data.insert(w.data.end(), certificate.begin(), certificate.end());
            w.word(cell.faces.size());
            for (auto f : cell.faces)
                w.word(f);
            w.word(cell.nodes.size());
            for (auto n : cell.nodes)
                w.word(n);
        }
    }
    w.word(faces.size());
    for (auto id : faces)
    {
        const auto& face = impl_->faces.at(id);
        w.word(id);
        surface_write(w, face.surface);
        w.word(face.owner);
        w.word(face.neighbor);
        w.word(face.nodes.size());
        for (auto n : face.nodes)
            w.word(n);
    }
    w.word(nodes.size());
    for (auto id : nodes)
    {
        const auto& node = impl_->nodes.at(id);
        w.word(id);
        w.word(node.region);
        for (const auto& q : node.point)
            w.rational(q);
    }
    return std::move(w.data);
}

std::shared_ptr<const PolyhedralComplex<ArithmeticMode::Exact>> PolyhedralComplex<ArithmeticMode::Exact>::decode(
    std::span<const char> packet)
{
    WireReader r{packet, 0, true};
    if (r.word() != 0x5346504f4c594331ULL || r.word() != 2)
        throw std::invalid_argument("Unknown exact complex packet version/tag.");
    auto impl = std::make_shared<Impl>();
    impl->global_cells = r.word();
    impl->global_faces = r.word();
    impl->global_nodes = r.word();
    if (!impl->global_cells || !impl->global_faces || !impl->global_nodes)
        throw std::invalid_argument("Exact complex global catalog is empty.");
    auto count = r.count();
    for (std::size_t i = 0; i < count; ++i)
    {
        auto id = r.word();
        ComplexCellRow cell;
        cell.key = cell_key_read(r);
        cell.volume = r.real();
        cell.center = {r.real(), r.real(), r.real()};
        auto full = r.word();
        if (full > 1)
            throw std::invalid_argument("Invalid exact complex residency flag.");
        if (full)
        {
            auto n = r.word();
            if (n > r.data.size() - r.position)
                throw std::invalid_argument("Invalid exact cell certificate length.");
            cell.geometry =
                ConvexCellGeometry<ArithmeticMode::Exact>::decode_certificate(r.data.subspan(r.position, n));
            r.position += n;
            n = r.count();
            for (std::size_t j = 0; j < n; ++j)
                cell.faces.push_back(r.word());
            n = r.count();
            for (std::size_t j = 0; j < n; ++j)
                cell.nodes.push_back(r.word());
            const auto m = exact_moments(cell.geometry->impl_->poly, point(cell.geometry->impl_->origin));
            const auto center = rounded_point(add(point(cell.geometry->impl_->origin), mul(m.first, Q(1 / m.volume))));
            if (cell.volume != measured(m.volume).as_double() || cell.center.x != center.x ||
                cell.center.y != center.y || cell.center.z != center.z)
                throw std::invalid_argument("Exact cell packet metric/certificate mismatch.");
        }
        if (!impl->cells.emplace(id, std::move(cell)).second)
            throw std::invalid_argument("Repeated exact complex cell ID.");
    }
    count = r.count();
    for (std::size_t i = 0; i < count; ++i)
    {
        auto id = r.word();
        ComplexFaceRow face;
        face.surface = surface_read(r);
        face.owner = r.word();
        face.neighbor = r.word();
        auto n = r.count();
        for (std::size_t j = 0; j < n; ++j)
            face.nodes.push_back(r.word());
        if (!impl->faces.emplace(id, std::move(face)).second)
            throw std::invalid_argument("Repeated exact complex face ID.");
    }
    count = r.count();
    for (std::size_t i = 0; i < count; ++i)
    {
        auto id = r.word();
        ComplexNodeRow node;
        node.region = r.word();
        for (auto& q : node.point)
            q = r.rational();
        if (!impl->nodes.emplace(id, std::move(node)).second)
            throw std::invalid_argument("Repeated exact complex node ID.");
    }
    r.finish();
    complex_validate(*impl);
    for (const auto& [id, cell] : impl->cells)
        if (cell.geometry)
        {
            Poly poly;
            std::map<ComplexID, std::size_t> ids;
            for (auto f : cell.faces)
            {
                const auto& face = impl->faces.at(f);
                Face loop;
                for (auto n : face.nodes)
                {
                    const auto [it, inserted] = ids.emplace(n, poly.vertices.size());
                    if (inserted)
                        poly.vertices.push_back(impl->nodes.at(n).point);
                    loop.push_back(it->second);
                }
                if (face.owner != id)
                    std::reverse(loop.begin(), loop.end());
                poly.faces.push_back(std::move(loop));
            }
            if (!contained(poly, cell.geometry->impl_->poly) || !contained(cell.geometry->impl_->poly, poly))
                throw std::invalid_argument("Exact sparse atomic topology changed its canonical cell geometry.");
        }
    return std::shared_ptr<const PolyhedralComplex<ArithmeticMode::Exact>>(
        new PolyhedralComplex<ArithmeticMode::Exact>(std::move(impl)));
}

std::size_t PolyhedralComplex<ArithmeticMode::Exact>::storage_bytes() const
{
    std::size_t bytes = sizeof(PolyhedralComplex<ArithmeticMode::Exact>) + sizeof(Impl);
    std::set<const ConvexCellGeometry<ArithmeticMode::Exact>*> certificates;
    for (const auto& [id, cell] : impl_->cells)
    {
        bytes += sizeof(decltype(impl_->cells)::value_type) + 4 * sizeof(void*) +
                 (cell.faces.capacity() + cell.nodes.capacity()) * sizeof(ComplexID);
        if (cell.geometry && certificates.insert(cell.geometry.get()).second)
            bytes += cell.geometry->storage_bytes();
        bytes += std::get<::SimpleFluid::Meshes::AMR::ConvexCellAncestry>(cell.key.ancestry).path.capacity() *
                 sizeof(ComplexID);
    }
    for (const auto& [id, face] : impl_->faces)
        bytes += sizeof(decltype(impl_->faces)::value_type) + 4 * sizeof(void*) +
                 face.nodes.capacity() * sizeof(ComplexID) + face.surface.boundary.capacity() +
                 face.surface.key.creator.path.capacity() * sizeof(ComplexID);
    for (const auto& [id, node] : impl_->nodes)
    {
        bytes += sizeof(decltype(impl_->nodes)::value_type) + 4 * sizeof(void*);
        for (const auto& q : node.point)
            bytes += (q.backend().num().capacity() + q.backend().denom().capacity()) *
                     sizeof(boost::multiprecision::limb_type);
    }
    bytes += (impl_->cell_ids.capacity() + impl_->resident_ids.capacity() + impl_->face_ids.capacity() +
                 impl_->node_ids.capacity()) *
             sizeof(ComplexID);
    return bytes;
}

std::size_t ConvexCellGeometry<ArithmeticMode::Exact>::storage_bytes() const
{
    std::size_t bytes = sizeof(ConvexCellGeometry<ArithmeticMode::Exact>) + sizeof(Impl) +
                        impl_->poly.vertices.capacity() * sizeof(P) + impl_->poly.faces.capacity() * sizeof(Face);
    auto add_q = [&](const Q& q)
    {
        bytes +=
            (q.backend().num().capacity() + q.backend().denom().capacity()) * sizeof(boost::multiprecision::limb_type);
    };
    for (const auto& p : impl_->poly.vertices)
        for (const auto& q : p)
            add_q(q);
    for (const auto& face : impl_->poly.faces)
        bytes += face.capacity() * sizeof(std::size_t);
    if (impl_->product)
    {
        bytes += impl_->product->base.capacity() * sizeof(P);
        for (const auto& p : impl_->product->base)
            for (const auto& q : p)
                add_q(q);
        add_q(impl_->product->lower);
        add_q(impl_->product->upper);
    }
    return bytes;
}

ConvexGeometryPtr<ArithmeticMode::Exact> ConvexCellGeometry<ArithmeticMode::Exact>::from_native_cell_atomic(
    const MultiRegionMesh& mesh, std::uint64_t cell)
{
    return from_geometry(*ConvexCellGeometry<ArithmeticMode::Ordinary>::from_native_cell_atomic(mesh, cell));
}

ConvexGeometryPtr<ArithmeticMode::Exact> ExtrudedConstruction<ArithmeticMode::Exact>::atomic_cell_geometry(
    std::size_t cell) const
{
    const auto id = impl_->topology->indexer().cell_id(cell);
    P center = impl_->base->centers.at(id.ij);
    center[2] = (impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k)] +
                    impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + id.k + 1)]) /
                2;
    auto geometry = std::make_shared<ConvexCellGeometry<ArithmeticMode::Exact>::Impl>();
    std::map<std::uint64_t, std::size_t> ids;
    for (auto face : impl_->topology->cell_faces(cell))
    {
        Face loop;
        for (auto node : impl_->topology->face_nodes(face))
        {
            auto [it, inserted] = ids.emplace(node, geometry->poly.vertices.size());
            if (inserted)
            {
                const auto n = impl_->topology->indexer().node_id(node);
                auto p = impl_->base->points[impl_->base->active.at(n.ij)];
                p[2] = impl_->axis->values->z[impl_->axis->zactive.at(impl_->begin + n.k)];
                geometry->poly.vertices.push_back(std::move(p));
            }
            loop.push_back(it->second);
        }
        const auto area = area_vector(geometry->poly, loop);
        const auto direction = dot(area, sub(geometry->poly.vertices[loop[0]], center));
        if (direction == 0)
            throw std::invalid_argument("Exact product face has unresolved interior visibility.");
        if (direction < 0)
            std::reverse(loop.begin(), loop.end());
        geometry->poly.faces.push_back(std::move(loop));
    }
    validate(geometry->poly);
    const auto origin = rounded_point(geometry->poly.vertices[0]);
    geometry->origin = {origin.x, origin.y, origin.z};
    geometry->product = product_description(geometry->poly);
    return ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry<ArithmeticMode::Exact>(std::move(geometry)));
}

ConvexGeometryPtr<ArithmeticMode::Exact> PolyhedralComplex<ArithmeticMode::Exact>::atomic_cell_geometry(
    std::uint64_t cell_id) const
{
    const auto& cell = impl_->cells.at(cell_id);
    if (!cell.geometry)
        throw std::invalid_argument("Metric-only neighbor has no exact atomic shell.");
    auto geometry = std::make_shared<ConvexCellGeometry<ArithmeticMode::Exact>::Impl>();
    std::map<std::uint64_t, std::size_t> ids;
    for (auto face_id : cell.faces)
    {
        const auto& face = impl_->faces.at(face_id);
        Face loop;
        for (auto node : face.nodes)
        {
            auto [it, inserted] = ids.emplace(node, geometry->poly.vertices.size());
            if (inserted)
                geometry->poly.vertices.push_back(impl_->nodes.at(node).point);
            loop.push_back(it->second);
        }
        if (face.owner != cell_id)
            std::reverse(loop.begin(), loop.end());
        geometry->poly.faces.push_back(std::move(loop));
    }
    validate(geometry->poly);
    geometry->origin = cell.geometry->impl_->origin;
    geometry->product = product_description(geometry->poly);
    return ConvexGeometryPtr<ArithmeticMode::Exact>(new ConvexCellGeometry<ArithmeticMode::Exact>(std::move(geometry)));
}

} // namespace SimpleFluid::PLIC
