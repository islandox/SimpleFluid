#include "geometry/details/BoundedConvexGeometryInternal.hh"

namespace SimpleFluid::Meshes
{
namespace bounded_geometry_detail
{
bool incident(const Vertex& v, const PlanePtr& plane)
{
    return std::find(v.incident.begin(), v.incident.end(), plane) != v.incident.end();
}

BoundedGeometrySection section(
    const BoundedConvexGeometry::Impl& poly, const std::vector<Vertex>& vertices, const std::vector<std::size_t>& loop)
{
    BoundedGeometrySection result;
    for (const auto i : loop)
        result.vertices.push_back(
            bounded(scaled(vertices[i].coordinates, poly.frame.exponent) + point(poly.frame.origin)));
    if (loop.size() < 3)
        return result;
    const auto& a = vertices[loop[0]].coordinates;
    Point area_vector;
    for (std::size_t i = 1; i + 1 < loop.size(); ++i)
        area_vector = area_vector + cross(vertices[loop[i]].coordinates - a, vertices[loop[i + 1]].coordinates - a);
    area_vector = area_vector * Interval(.5L);
    const auto squared_area = dot(area_vector, area_vector);
    if (squared_area.zero())
        return result; // A proven line/point contact retains its vertices.
    result.area = bounded(scaled(square_root(squared_area), 2 * poly.frame.exponent));
    Point weighted;
    Interval total_weight;
    for (std::size_t i = 1; i + 1 < loop.size(); ++i)
    {
        const auto& b = vertices[loop[i]].coordinates;
        const auto& c = vertices[loop[i + 1]].coordinates;
        const auto weight = dot(cross(b - a, c - a), area_vector);
        total_weight = total_weight + weight;
        weighted = weighted + (a + b + c) * (weight / Interval(3));
    }
    result.centroid =
        bounded(scaled(weighted * (Interval(1) / total_weight), poly.frame.exponent) + point(poly.frame.origin));
    return result;
}
} // namespace bounded_geometry_detail
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
BoundedClipResult BoundedConvexGeometry::clip(const GeometryHalfspace& input, bool measure_section) const
{
    Operation scope(impl_->limits);
    if (!nonzero(point(input.normal)))
        reject("clipping requires a nonzero plane normal.");
    // Reuse the original, oriented source expression, never a comparison of
    // rounded vertex coordinates with a geometrically nearby plane.
    for (const auto& vertex : impl_->vertices)
        for (const auto& existing : vertex.incident)
            if (existing->input)
                if (const int orientation = matching_input(input, *existing->input))
                {
                    auto source = std::make_shared<Impl>();
                    source->surfaces.push_back({existing, orientation});
                    return clip_impl(source, 0, measure_section);
                }
    auto plane = std::make_shared<Plane>();
    plane->frame = impl_->frame;
    plane->normal = point(input.normal);
    plane->anchor = scaled(point(input.anchor) - point(impl_->frame.origin), -impl_->frame.exponent);
    plane->offset = scaled(Interval(input.offset), -impl_->frame.exponent);
    plane->input = input;
    auto source = std::make_shared<Impl>();
    source->surfaces.push_back({plane, 1});
    return clip_impl(source, 0, measure_section);
}

BoundedClipResult BoundedConvexGeometry::clip_by_face(
    const BoundedConvexGeometry& source, std::size_t face, bool measure_section) const
{
    Operation scope(impl_->limits);
    if (face >= source.impl_->faces.size())
        reject("invalid source face index.");
    return clip_impl(source.impl_, face, measure_section);
}

BoundedClipResult BoundedConvexGeometry::clip_impl(
    const std::shared_ptr<const Impl>& source, std::size_t plane_id, bool measure_section) const
{
    const auto support = source->surfaces.at(plane_id);
    const auto& plane = support.plane;
    std::vector<Interval> distances;
    std::vector<int> signs;
    bool negative = false, positive = false;
    for (const auto& vertex : impl_->vertices)
    {
        auto d = distance(vertex, impl_->frame, plane) * Interval(support.orientation);
        const auto s = sign(d);
        distances.push_back(d);
        signs.push_back(s);
        negative = negative || s < 0;
        positive = positive || s > 0;
    }
    BoundedClipResult result;
    if (!negative || !positive)
    {
        // A support face/edge/point is a proven zero-volume contact. Preserve
        // its section dimension rather than manufacturing a positive piece.
        std::vector<std::size_t> contact;
        std::map<std::pair<std::size_t, std::size_t>, int> contact_edges;
        for (const auto& face : impl_->faces)
            if (std::all_of(face.begin(), face.end(), [&](auto i) { return signs[i] == 0; }))
                for (std::size_t i = 0; i < face.size(); ++i)
                {
                    const auto a = face[i], b = face[(i + 1) % face.size()];
                    contact_edges[std::minmax(a, b)] += a < b ? 1 : -1;
                }
        std::map<std::size_t, std::set<std::size_t>> boundary;
        for (const auto& [edge, coefficient] : contact_edges)
            if (coefficient)
            {
                if (std::abs(coefficient) != 1)
                    reject("support section has repeated oriented boundary coverage.");
                boundary[edge.first].insert(edge.second);
                boundary[edge.second].insert(edge.first);
            }
        if (!boundary.empty())
        {
            for (const auto& [vertex, neighbors] : boundary)
                if (neighbors.size() != 2)
                    reject("support section is not a closed boundary.");
            const auto first = boundary.begin()->first;
            auto previous = first, current = *boundary.at(first).begin();
            contact.push_back(first);
            while (current != first)
            {
                if (contact.size() >= boundary.size())
                    reject("support section boundary repeats a vertex.");
                contact.push_back(current);
                const auto& next = boundary.at(current);
                const auto following = *next.begin() == previous ? *next.rbegin() : *next.begin();
                previous = current;
                current = following;
            }
            if (contact.size() != boundary.size())
                reject("support section has disconnected atomic boundaries.");
        }
        if (contact.empty())
            for (std::size_t i = 0; i < signs.size(); ++i)
                if (signs[i] == 0)
                    contact.push_back(i);
        if (measure_section)
            result.section = section(*impl_, impl_->vertices, contact);
        auto self = BoundedConvexGeometryPtr(new BoundedConvexGeometry(impl_));
        if (!positive)
            result.inside = self;
        if (!negative)
            result.outside = self;
        // A valid positive-volume shell cannot lie entirely on this plane.
        if (!negative && !positive)
            reject("positive-volume shell has entirely unresolved support.");
        return result;
    }

    std::vector<Vertex> vertices = impl_->vertices;
    // Authenticate every zero-distance input incidence on the new shared plane.
    for (std::size_t i = 0; i < signs.size(); ++i)
        if (signs[i] == 0)
        {
            if (!incident(vertices[i], plane))
                vertices[i].incident.push_back(plane);
            for (const auto& key : plane->authorities)
                if (std::find(vertices[i].authorities.begin(), vertices[i].authorities.end(), key) ==
                    vertices[i].authorities.end())
                    vertices[i].authorities.push_back(key);
        }
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> crossings;
    const auto crossing = [&](std::size_t a, std::size_t b)
    {
        const std::pair<std::size_t, std::size_t> key{std::min(a, b), std::max(a, b)};
        if (auto found = crossings.find(key); found != crossings.end())
            return found->second;
        a = key.first;
        b = key.second;
        const auto fraction = distances[a] / (distances[a] - distances[b]);
        if (!(fraction.lo > 0 && fraction.hi < 1))
            reject("edge intersection parameter is unresolved.");
        Vertex vertex;
        const auto denominator = distances[a] - distances[b];
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            // Delay the common division. A nonbinary edge parameter can
            // produce an exactly representable coordinate; rounding the
            // parameter first would destroy that cancellation and its support.
            vertex.coordinates[axis] =
                (vertices[b].coordinates[axis] * distances[a] - vertices[a].coordinates[axis] * distances[b]) /
                denominator;
        }
        for (const auto& p : vertices[a].incident)
            if (incident(vertices[b], p))
                vertex.incident.push_back(p);
        if (!incident(vertex, plane))
            vertex.incident.push_back(plane);
        for (const auto& key : vertices[a].authorities)
            if (std::find(vertices[b].authorities.begin(), vertices[b].authorities.end(), key) !=
                vertices[b].authorities.end())
                vertex.authorities.push_back(key);
        for (const auto& key : plane->authorities)
            if (std::find(vertex.authorities.begin(), vertex.authorities.end(), key) == vertex.authorities.end())
                vertex.authorities.push_back(key);
        if (vertices.size() >= impl_->limits.maximum_vertices)
            reject("intersection vertex cap exhausted.");
        const auto index = vertices.size();
        vertices.push_back(std::move(vertex));
        crossings.emplace(key, index);
        return index;
    };
    std::array<std::vector<Face>, 2> output_faces;
    std::array<std::vector<Surface>, 2> output_surfaces;
    std::set<std::pair<std::size_t, std::size_t>> cap_edges;
    for (std::size_t f = 0; f < impl_->faces.size(); ++f)
    {
        const auto& face = impl_->faces[f];
        for (unsigned outside = 0; outside < 2; ++outside)
        {
            Face loop;
            for (std::size_t i = 0; i < face.size(); ++i)
            {
                const auto a = face[i], b = face[(i + 1) % face.size()];
                if (outside ? signs[a] >= 0 : signs[a] <= 0)
                    loop.push_back(a);
                if (signs[a] * signs[b] < 0)
                    loop.push_back(crossing(a, b));
            }
            if (loop.size() >= 3 &&
                !std::all_of(loop.begin(), loop.end(), [&](auto i) { return incident(vertices[i], plane); }))
            {
                // A source face can meet the cut in an edge containing three
                // or more propagated collinear vertices. It is not a positive
                // area output face; the retained adjacent face contributes the
                // same authenticated edge chain to the cap.
                output_faces[outside].push_back(loop);
                output_surfaces[outside].push_back(impl_->surfaces[f]);
                if (!outside)
                    for (std::size_t i = 0; i < loop.size(); ++i)
                    {
                        const auto a = loop[i], b = loop[(i + 1) % loop.size()];
                        if (a != b && incident(vertices[a], plane) && incident(vertices[b], plane))
                            cap_edges.insert(std::minmax(a, b));
                    }
            }
        }
    }
    std::map<std::size_t, std::set<std::size_t>> adjacency;
    for (const auto& [a, b] : cap_edges)
    {
        adjacency[a].insert(b);
        adjacency[b].insert(a);
    }
    if (adjacency.size() < 3)
        reject("positive complementary cut has no closed section.");
    for (const auto& [vertex, neighbors] : adjacency)
        if (neighbors.size() != 2)
            reject("section edge incidence is not one closed cycle.");
    Face cap;
    const auto first = adjacency.begin()->first;
    std::size_t previous = first, current = *adjacency.at(first).begin();
    cap.push_back(first);
    while (current != first)
    {
        if (cap.size() >= adjacency.size())
            reject("section cycle repeats a vertex.");
        cap.push_back(current);
        const auto& neighbors = adjacency.at(current);
        const auto next = *neighbors.begin() == previous ? *neighbors.rbegin() : *neighbors.begin();
        previous = current;
        current = next;
    }
    if (cap.size() != adjacency.size())
        reject("section has disconnected boundaries.");
    Point cap_normal;
    for (std::size_t i = 1; i + 1 < cap.size(); ++i)
        cap_normal = cap_normal + cross(vertices[cap[i]].coordinates - vertices[cap[0]].coordinates,
                                      vertices[cap[i + 1]].coordinates - vertices[cap[0]].coordinates);
    const auto orientation = sign(dot(cap_normal, plane->normal) * Interval(support.orientation));
    if (!orientation)
        reject("section orientation is degenerate.");
    if (orientation < 0)
        std::reverse(cap.begin(), cap.end());
    if (measure_section)
        result.section = section(*impl_, vertices, cap);
    output_faces[0].push_back(cap);
    output_surfaces[0].push_back(support);
    std::reverse(cap.begin(), cap.end());
    output_faces[1].push_back(cap);
    output_surfaces[1].push_back({plane, -support.orientation});
    auto split = std::make_shared<Split>();
    split->parent = impl_->ancestry;
    const auto child = [&](unsigned outside)
    {
        auto poly = std::make_shared<Impl>();
        poly->frame = impl_->frame;
        poly->limits = impl_->limits;
        poly->ancestry = std::make_shared<Ancestry>(Ancestry{impl_->ancestry, split, outside != 0});
        std::set<std::size_t> used;
        for (const auto& face : output_faces[outside])
            used.insert(face.begin(), face.end());
        std::map<std::size_t, std::size_t> remap;
        for (const auto i : used)
        {
            remap.emplace(i, poly->vertices.size());
            poly->vertices.push_back(vertices[i]);
        }
        poly->faces = std::move(output_faces[outside]);
        poly->surfaces = std::move(output_surfaces[outside]);
        for (auto& face : poly->faces)
            for (auto& i : face)
                i = remap.at(i);
        finish(*poly);
        return BoundedConvexGeometryPtr(new BoundedConvexGeometry(std::move(poly)));
    };
    result.inside = child(0);
    result.outside = child(1);
    return result;
}

BoundedPhaseMeasures BoundedConvexGeometry::phases(const GeometryHalfspace& gas) const
{
    Operation scope(impl_->limits);
    const auto split = clip(gas, false);
    return {volume(), split.inside ? split.inside->volume() : BoundedGeometryValue{},
        split.outside ? split.outside->volume() : BoundedGeometryValue{}};
}

BoundedConvexGeometryPtr BoundedConvexGeometry::intersect(const BoundedConvexGeometry& other) const
{
    Operation scope(impl_->limits);
    ++statistics.intersections;
    auto result = BoundedConvexGeometryPtr(new BoundedConvexGeometry(impl_));
    for (std::size_t face = 0; result && face < other.impl_->faces.size(); ++face)
        result = result->clip_by_face(other, face).inside;
    return result;
}

BoundedConvexGeometryPtr BoundedConvexGeometry::translated(GeometryPoint displacement) const
{
    Operation scope(impl_->limits);
    auto result = std::make_shared<Impl>(*impl_);
    result->moment_cache = std::make_shared<MomentCache>();
    // Translation is a new source expression, with a fresh structural root.
    // Both vertex and support anchors carry the same displacement enclosure.
    result->ancestry = std::make_shared<Ancestry>();
    std::map<PlanePtr, PlanePtr> planes;
    for (auto& vertex : result->vertices)
        for (auto& plane : vertex.incident)
        {
            if (!planes.contains(plane))
            {
                auto shifted = std::make_shared<Plane>(*plane);
                shifted->anchor = shifted->anchor + scaled(point(displacement), -shifted->frame.exponent);
                shifted->input.reset();
                shifted->authorities.clear();
                planes.emplace(plane, shifted);
            }
            plane = planes.at(plane);
        }
    for (auto& surface : result->surfaces)
        surface.plane = planes.at(surface.plane);
    for (auto& vertex : result->vertices)
    {
        vertex.coordinates = vertex.coordinates + scaled(point(displacement), -result->frame.exponent);
        vertex.authorities.clear();
    }
    finish(*result);
    return BoundedConvexGeometryPtr(new BoundedConvexGeometry(std::move(result)));
}
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
namespace bounded_geometry_detail
{
bool structural_cover(
    const std::shared_ptr<const Ancestry>& root, const std::vector<std::shared_ptr<const Ancestry>>& leaves)
{
    if (leaves.empty())
        return false;
    if (leaves.size() > std::numeric_limits<std::size_t>::max() / 2)
        reject("structural proof node count overflows.");
    // A complete binary complement tree has fewer than twice as many total
    // nodes as leaves. Charge the whole proof before allocating its worklist;
    // private split identity, not numerical equality, authenticates siblings.
    active_work->charge(leaves.size(),
        2 * sizeof(std::shared_ptr<const Ancestry>) + sizeof(const Split*) + sizeof(unsigned) + 4 * sizeof(void*));
    std::vector<std::shared_ptr<const Ancestry>> pending;
    pending.reserve(2 * leaves.size());
    pending.insert(pending.end(), leaves.begin(), leaves.end());
    std::map<const Split*, unsigned> siblings;
    unsigned roots = 0;
    for (std::size_t i = 0; i < pending.size(); ++i)
    {
        active_work->pair();
        const auto& node = pending[i];
        if (node == root)
        {
            if (++roots > 1)
                return false;
            continue;
        }
        if (!node || !node->parent || !node->split || node->split->parent != node->parent)
            return false;
        const auto split = node->split.get();
        const unsigned side = node->outside ? 2 : 1;
        auto& covered = siblings[split];
        if (covered & side)
            return false; // Duplicate/ancestor-overlapping supplied pieces.
        covered |= side;
        if (covered == 3)
        {
            siblings.erase(split);
            pending.push_back(node->parent);
        }
    }
    return roots == 1 && siblings.empty();
}
} // namespace bounded_geometry_detail
} // namespace SimpleFluid::Meshes
