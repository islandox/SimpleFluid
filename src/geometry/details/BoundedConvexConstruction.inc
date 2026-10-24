std::shared_ptr<BoundedConvexGeometry::Impl> import(GeometryPoint origin, std::span<const BoundedPoint> vertices,
    const std::vector<Face>& faces, BoundedGeometryLimits limits, bool authenticated_planarity)
{
    if (vertices.size() < 4 || faces.size() < 4 || vertices.size() > limits.maximum_vertices ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive convex shell size.");
    for (const auto& face : faces)
        active_work->charge(face.size(), 4 * sizeof(std::size_t) + 4 * sizeof(void*));
    auto result = std::make_shared<BoundedConvexGeometry::Impl>();
    result->frame.origin = origin;
    result->limits = limits;
    long double extent = 0;
    std::vector<Point> input;
    for (const auto& vertex : vertices)
    {
        input.push_back(point(vertex));
        for (const auto x : input.back())
            extent = std::max({extent, std::abs(x.lo), std::abs(x.hi)});
    }
    if (!(extent > 0))
        reject("degenerate shell extent.");
    std::frexp(extent, &result->frame.exponent);
    for (std::size_t i = 0; i < input.size(); ++i)
        result->vertices.push_back({scaled(input[i], -result->frame.exponent), {}, vertices[i].supports()});
    result->faces = faces;
    std::map<std::pair<std::size_t, std::size_t>, std::pair<unsigned, int>> edges;
    for (const auto& face : faces)
    {
        if (face.size() < 3)
            reject("face has fewer than three vertices.");
        std::set<std::size_t> distinct;
        for (std::size_t i = 0; i < face.size(); ++i)
        {
            const auto a = face[i], b = face[(i + 1) % face.size()];
            if (a >= vertices.size() || b >= vertices.size() || !distinct.insert(a).second)
                reject("invalid or repeated face vertex.");
            auto& edge = edges[std::minmax(a, b)];
            ++edge.first;
            edge.second += a < b ? 1 : -1;
        }
        auto plane = std::make_shared<Plane>();
        plane->frame = result->frame;
        plane->anchor = result->vertices[face[0]].coordinates;
        std::size_t second = 1, third = 2;
        for (; third < face.size(); ++third)
        {
            plane->normal = cross(result->vertices[face[second]].coordinates - plane->anchor,
                result->vertices[face[third]].coordinates - plane->anchor);
            if (nonzero(plane->normal))
                break;
        }
        if (third == face.size())
            reject("face normal unresolved or degenerate.");
        std::vector<BoundedPoint> face_points;
        for (const auto index : face)
            face_points.push_back(vertices[index]);
        plane->authorities = bounded_common_supports(face_points);
        for (const auto index : face)
        {
            const auto& coordinate = result->vertices[index].coordinates;
            if (!authenticated_planarity && index != face[0] && index != face[second] && index != face[third] &&
                !dot(plane->normal, coordinate - plane->anchor).zero())
                reject("face coplanarity unresolved; an enclosure containing zero is not coplanarity.");
            result->vertices[index].incident.push_back(plane);
        }
        result->surfaces.push_back({plane, 1});
    }
    for (const auto& [key, count] : edges)
        if (count.first != 2 || count.second != 0)
            reject("shell is not an oriented closed two-manifold.");
    validate_connected_boundary(vertices.size(), result->faces);
    for (std::size_t f = 0; f < result->faces.size(); ++f)
    {
        const auto& face = result->faces[f];
        const auto& plane = result->surfaces[f].plane;
        for (const auto& vertex : result->vertices)
            if (sign(distance(vertex, result->frame, plane)) > 0)
                reject("shell is not convex/outward.");
        // Each edge bounds a convex face, including a collinear prefix.
        for (std::size_t i = 0; i < face.size(); ++i)
        {
            const auto& a = result->vertices[face[i]].coordinates;
            const auto& b = result->vertices[face[(i + 1) % face.size()]].coordinates;
            for (const auto index : face)
            {
                if (index == face[i] || index == face[(i + 1) % face.size()])
                    continue;
                const auto orientation = dot(cross(b - a, result->vertices[index].coordinates - a), plane->normal);
                if (orientation.hi < 0)
                    reject("nonconvex face loop.");
                if (orientation.lo < 0 && !orientation.zero())
                {
                    // Source authentication establishes planar convex loops;
                    // arbitrary bounded public shells need a decisive turn.
                    if (!authenticated_planarity)
                        reject("face convexity unresolved.");
                }
            }
        }
    }
    finish(*result);
    return result;
}
