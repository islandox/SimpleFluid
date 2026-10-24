Integrals integrate(const BoundedConvexGeometry::Impl& poly, bool moments)
{
    Point pivot;
    for (const auto& vertex : poly.vertices)
        pivot = pivot + vertex.coordinates;
    pivot = pivot * (Interval(1) / Interval(static_cast<long double>(poly.vertices.size())));
    Integrals sum;
    Accumulator volume_sum;
    std::array<Accumulator, 3> first_sum;
    // Enclosures accumulate independently of the compensated central estimate;
    // no rounded volume equality or skipped uncertain triangle supplies proof.
    for (const auto& face : poly.faces)
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            const auto& a = poly.vertices[face[0]].coordinates;
            const auto& b = poly.vertices[face[i]].coordinates;
            const auto& c = poly.vertices[face[i + 1]].coordinates;
            const auto determinant = dot(a - pivot, cross(b - pivot, c - pivot));
            if (!moments)
                volume_sum.add(determinant);
            if (moments)
            {
                const auto first = (pivot + a + b + c) * determinant;
                for (std::size_t axis = 0; axis < 3; ++axis)
                    first_sum[axis].add(first[axis]);
            }
        }
    // All tetrahedra have the same fixed integration denominator. Accumulate
    // numerators first, so representable cancellations are not obscured by
    // independently rounded divisions. This is bounded FP, not rational work.
    sum.volume = moments ? poly.volume : scaled(volume_sum.result() / Interval(6), 3 * poly.frame.exponent);
    if (moments)
        for (std::size_t axis = 0; axis < 3; ++axis)
            sum.first[axis] = scaled(first_sum[axis].result() / Interval(24), 4 * poly.frame.exponent);
    return sum;
}
void finish(BoundedConvexGeometry::Impl& poly)
{
    if (poly.vertices.size() > poly.limits.maximum_vertices || poly.faces.size() > poly.limits.maximum_faces)
        reject("vertex/face cap exhausted.");
    active_work->charge(1, sizeof(MomentCache));
    active_work->charge(poly.vertices.size(), sizeof(Vertex) + sizeof(BoundedPoint));
    active_work->charge(poly.faces.size(), sizeof(Face) + sizeof(Surface));
    poly.displayed.clear();
    for (const auto& v : poly.vertices)
    {
        auto displayed = bounded(scaled(v.coordinates, poly.frame.exponent) + point(poly.frame.origin));
        BoundedGeometryAccess::seal(displayed, v.authorities);
        active_work->charge(bounded_point_storage_bytes(displayed) - sizeof(BoundedPoint), 1);
        poly.displayed.push_back(std::move(displayed));
        active_work->charge(v.incident.size(), sizeof(PlanePtr));
        active_work->charge(v.authorities.size(), sizeof(BoundedPlaneAuthority));
        for (const auto& key : v.authorities)
            active_work->charge(key.key.size(), sizeof(std::uint64_t));
    }
    std::set<PlanePtr> retained_planes;
    for (const auto& v : poly.vertices)
        retained_planes.insert(v.incident.begin(), v.incident.end());
    for (const auto& plane : retained_planes)
    {
        active_work->charge(1, sizeof(Plane));
        active_work->charge(plane->authorities.size(), sizeof(BoundedPlaneAuthority));
        for (const auto& key : plane->authorities)
            active_work->charge(key.key.size(), sizeof(std::uint64_t));
    }
    for (const auto& f : poly.faces)
        active_work->charge(f.size(), sizeof(std::size_t));
    ++statistics.volume_evaluations;
    poly.volume = integrate(poly, false).volume;
    if (!(poly.volume.lo > 0))
        reject("strictly positive volume cannot be established.");
    ++statistics.constructions;
}
void validate_connected_boundary(std::size_t vertex_count, const std::vector<Face>& faces)
{
    active_work->charge(vertex_count, sizeof(char));
    active_work->charge(faces.size(), sizeof(std::vector<std::size_t>) + sizeof(char));
    std::vector<char> vertices_used(vertex_count, false), reached(faces.size(), false);
    std::vector<std::vector<std::size_t>> adjacent(faces.size());
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> first_face;
    for (std::size_t f = 0; f < faces.size(); ++f)
        for (std::size_t i = 0; i < faces[f].size(); ++i)
        {
            const auto a = faces[f][i], b = faces[f][(i + 1) % faces[f].size()];
            vertices_used.at(a) = true;
            const std::pair<std::size_t, std::size_t> edge{std::min(a, b), std::max(a, b)};
            auto [position, inserted] = first_face.emplace(edge, f);
            if (!inserted)
            {
                adjacent[f].push_back(position->second);
                adjacent[position->second].push_back(f);
            }
        }
    if (std::find(vertices_used.begin(), vertices_used.end(), false) != vertices_used.end())
        reject("shell contains unused vertices.");
    std::vector<std::size_t> pending{0};
    reached.front() = true;
    for (std::size_t i = 0; i < pending.size(); ++i)
        for (const auto neighbor : adjacent[pending[i]])
            if (!reached[neighbor])
            {
                reached[neighbor] = true;
                pending.push_back(neighbor);
            }
    if (pending.size() != faces.size())
        reject("shell has disconnected or multiply covered boundary components.");
}
