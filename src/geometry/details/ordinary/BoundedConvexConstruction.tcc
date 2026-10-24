/** @file ordinary/BoundedConvexConstruction.tcc
 * @brief Ordinary convex construction and compensated nominal integration.
 */
namespace SimpleFluid::Meshes::bounded_geometry_detail
{
namespace ordinary_kernel SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode> Integrals integrate_ordinary(const StoredGeometry& poly, bool moments)
{
    static_assert(Mode == kernel_mode);
    if (poly.vertices.size() == 4 && poly.faces.size() == 4 &&
        std::all_of(poly.faces.begin(), poly.faces.end(), [](const auto& face) { return face.size() == 3; }))
    {
        Integrals result;
        if (!moments)
        {
            const auto& anchor = poly.vertices.front().coordinates;
            for (const auto& face : poly.faces)
                if (std::find(face.begin(), face.end(), 0) == face.end())
                {
                    result.volume = scaled<Mode>(dot<Mode>(poly.vertices[face[0]].coordinates - anchor,
                                                     cross<Mode>(poly.vertices[face[1]].coordinates - anchor,
                                                         poly.vertices[face[2]].coordinates - anchor)) /
                                                     Scalar(6),
                        3 * poly.frame.exponent);
                    break;
                }
        }
        else
        {
            Point center;
            for (const auto& vertex : poly.vertices)
                center = center + vertex.coordinates;
            result.volume = poly.volume;
            result.first = scaled<Mode>(center * Scalar(.25L), poly.frame.exponent) * poly.volume;
        }
        return result;
    }
    Point pivot;
    for (const auto& vertex : poly.vertices)
        pivot = pivot + vertex.coordinates;
    pivot = pivot * (Scalar(1) / Scalar(static_cast<long double>(poly.vertices.size())));
    Accumulator<Mode> volume;
    std::array<Accumulator<Mode>, 3> first;
    for (const auto& face : poly.faces)
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            const auto& a = poly.vertices[face[0]].coordinates;
            const auto& b = poly.vertices[face[i]].coordinates;
            const auto& c = poly.vertices[face[i + 1]].coordinates;
            const auto determinant = dot<Mode>(a - pivot, cross<Mode>(b - pivot, c - pivot));
            volume.add(determinant);
            if (moments)
            {
                const auto weighted = (pivot + a + b + c) * determinant;
                for (std::size_t axis = 0; axis < 3; ++axis)
                    first[axis].add(weighted[axis]);
            }
        }
    Integrals result;
    result.volume = moments ? Scalar(poly.volume) : scaled<Mode>(volume.result() / Scalar(6), 3 * poly.frame.exponent);
    if (moments)
        for (std::size_t axis = 0; axis < 3; ++axis)
            result.first[axis] = scaled<Mode>(first[axis].result() / Scalar(24), 4 * poly.frame.exponent);
    return result;
}

template<ArithmeticMode Mode> Integrals integrate(const StoredGeometry& poly, bool moments)
{
    static_assert(Mode == kernel_mode);
    return integrate_ordinary<Mode>(poly, moments);
}

template<ArithmeticMode Mode> void finish(StoredGeometry& poly, bool partition_proof, bool positive_construction)
{
    static_assert(Mode == kernel_mode);
    if (poly.vertices.size() > std::min(poly.limits.maximum_vertices, active_work->limits.maximum_vertices) ||
        poly.faces.size() > std::min(poly.limits.maximum_faces, active_work->limits.maximum_faces))
        reject("vertex/face cap exhausted.");
    active_work->charge(1, sizeof(MomentCache));
    active_work->charge(poly.vertices.size(), sizeof(Vertex) + sizeof(BoundedPoint));
    active_work->charge(poly.faces.size(), sizeof(Face) + sizeof(Surface));
    poly.displayed.clear();
    for (const auto& v : poly.vertices)
    {
        auto displayed =
            bounded<Mode>(scaled<Mode>(v.coordinates, poly.frame.exponent) + point<Mode>(poly.frame.origin));
        BoundedGeometryAccess::seal(displayed, v.authorities);
        active_work->charge(bounded_point_storage_bytes(displayed) - sizeof(BoundedPoint), 1);
        poly.displayed.push_back(std::move(displayed));
        active_work->charge(v.incident.size(), sizeof(PlanePtr));
        active_work->charge(v.authorities.size(), sizeof(BoundedPlaneAuthority));
        active_work->charge(BoundedGeometryAccess::identity_bytes(v.source_identity), 1);
        for (const auto& key : v.authorities)
            active_work->charge(key.key.size(), sizeof(std::uint64_t));
    }
    std::set<PlanePtr> retained_planes;
    for (const auto& v : poly.vertices)
        retained_planes.insert(v.incident.begin(), v.incident.end());
    for (const auto& plane : retained_planes)
    {
        active_work->charge(1, sizeof(Plane));
        active_work->charge(plane->source_vertices.capacity(), sizeof(std::weak_ptr<const BoundedPointProvenance>));
        active_work->charge(plane->authorities.size(), sizeof(BoundedPlaneAuthority));
        for (const auto& key : plane->authorities)
            active_work->charge(key.key.size(), sizeof(std::uint64_t));
    }
    for (const auto& f : poly.faces)
        active_work->charge(f.size(), sizeof(std::size_t));
    // Local coordinate bounds retain the full spatial extent. Compare
    // in one local frame later, avoiding cancellation at large global origins.
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        auto lo = poly.vertices.front().coordinates[axis].value;
        auto hi = poly.vertices.front().coordinates[axis].value;
        for (const auto& vertex : poly.vertices)
        {
            lo = std::min(lo, vertex.coordinates[axis].value);
            hi = std::max(hi, vertex.coordinates[axis].value);
        }
        poly.bounds[axis] = {lo, hi};
    }
    poly.partition_proof_only = partition_proof;
    if (!partition_proof)
    {
        ++statistics.volume_evaluations;
        poly.volume = integrate<Mode>(poly, false).volume;
        if (!(poly.volume.value > 0))
            reject("strictly positive volume cannot be established.");
    }
    ++statistics.constructions;
}

template<ArithmeticMode Mode> void validate_connected_boundary(std::size_t vertex_count, const std::vector<Face>& faces)
{
    static_assert(Mode == kernel_mode);
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

template<ArithmeticMode Mode>
std::shared_ptr<StoredGeometry> import(GeometryPoint origin, std::span<const BoundedPoint> vertices,
    const std::vector<Face>& faces, BoundedGeometryLimits limits, bool authenticated_planarity,
    std::span<const std::shared_ptr<const BoundedPointProvenance>> identities)
{
    static_assert(Mode == kernel_mode);
    if (!identities.empty() && identities.size() != vertices.size())
        reject("source point identity count differs from the imported shell.");
    if (vertices.size() < 4 || faces.size() < 4 || vertices.size() > limits.maximum_vertices ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive convex shell size.");
    for (const auto& face : faces)
        active_work->charge(face.size(), 4 * sizeof(std::size_t) + 4 * sizeof(void*));
    auto result = std::make_shared<StoredGeometry>();
    result->frame.origin = origin;
    result->reference_origin = origin;
    result->limits = limits;
    long double extent = 0;
    std::vector<Point> input;
    for (const auto& vertex : vertices)
    {
        input.push_back(point<Mode>(vertex));
        for (const auto x : input.back())
            extent = std::max(extent, std::abs(x.value));
    }
    if (!(extent > 0))
        reject("degenerate shell extent.");
    std::frexp(extent, &result->frame.exponent);
    for (std::size_t i = 0; i < input.size(); ++i)
        result->vertices.push_back({scaled<Mode>(input[i], -result->frame.exponent), {}, vertices[i].supports(),
            identities.empty() ? nullptr : identities[i]});
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
            plane->normal = cross<Mode>(result->vertices[face[second]].coordinates - plane->anchor,
                result->vertices[face[third]].coordinates - plane->anchor);
            if (nonzero<Mode>(plane->normal))
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
            result->vertices[index].incident.push_back(plane);
        }
        // Issue point incidences only after this face's planarity is proved.
        // Tokens identify original global constructions, not rounded values.
        for (const auto index : face)
            if (const auto& identity = result->vertices[index].source_identity)
            {
                if (plane->source_vertices.empty())
                    plane->source_vertices.reserve(face.size());
                plane->source_vertices.push_back(identity);
            }
        result->surfaces.push_back({plane, 1});
    }
    for (const auto& [key, count] : edges)
        if (count.first != 2 || count.second != 0)
            reject("shell is not an oriented closed two-manifold.");
    validate_connected_boundary<Mode>(vertices.size(), result->faces);
    for (std::size_t f = 0; f < result->faces.size(); ++f)
    {
        const auto& face = result->faces[f];
        const auto& plane = result->surfaces[f].plane;
        // The native source already owns its convex face topology. Ordinary
        // import does not re-prove its rounded coplanarity or winding.
        if (authenticated_planarity)
            continue;
        for (const auto& vertex : result->vertices)
            if (sign<Mode>(distance<Mode>(vertex, result->frame, plane)) > 0)
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
                const auto orientation =
                    dot<Mode>(cross<Mode>(b - a, result->vertices[index].coordinates - a), plane->normal);
                if (orientation.value < 0)
                    reject("nonconvex face loop.");
            }
        }
    }
    finish<Mode>(*result);
    return result;
}

template<ArithmeticMode Mode> BoundedPoint bounded_point_average(std::span<const BoundedPoint> vertices)
{
    static_assert(Mode == kernel_mode);
    Operation scope(
        mode_limits(vertices.empty() ? ArithmeticMode::BoundedCertification : vertices.front().arithmetic_mode));
    if (vertices.empty())
        reject("empty affine average.");
    std::array<Accumulator<Mode>, 3> sum;
    for (const auto& vertex : vertices)
    {
        const auto input = point<Mode>(vertex);
        for (std::size_t axis = 0; axis < 3; ++axis)
            sum[axis].add(input[axis]);
    }
    Point average;
    for (std::size_t axis = 0; axis < 3; ++axis)
        average[axis] = sum[axis].result() / Scalar(static_cast<long double>(vertices.size()));
    auto result = bounded<Mode>(average);
    BoundedGeometryAccess::seal(result, bounded_common_supports(vertices));
    return result;
}

template<ArithmeticMode Mode>
BoundedPoint bounded_point_interpolate(const BoundedPoint& a, const BoundedPoint& b, long double fraction)
{
    static_assert(Mode == kernel_mode);
    require_mode(a.arithmetic_mode, b.arithmetic_mode);
    Operation scope(mode_limits(a.arithmetic_mode));
    auto result = bounded<Mode>(point<Mode>(a) + (point<Mode>(b) - point<Mode>(a)) * Scalar(fraction));
    const std::array vertices{a, b};
    BoundedGeometryAccess::seal(result, bounded_common_supports(vertices));
    return result;
}

template<ArithmeticMode Mode> long double ValueKernel<Mode>::lower() const
{
    static_assert(Mode == kernel_mode);
    reject("ordinary values do not carry certified bounds.");
}

template<ArithmeticMode Mode> long double ValueKernel<Mode>::upper() const
{
    static_assert(Mode == kernel_mode);
    reject("ordinary values do not carry certified bounds.");
}

template<ArithmeticMode Mode> double ValueKernel<Mode>::as_double() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(mode_limits(arithmetic_mode));
    finite(value);
    const auto result = static_cast<double>(value);
    if (!std::isfinite(result) || (result == 0 && value != 0))
        reject("ordinary value outside publication range.");
    return result;
}

template<ArithmeticMode Mode> bool ValueKernel<Mode>::relative_error_at_most(long double target, double tolerance) const
{
    static_assert(Mode == kernel_mode);
    reject("ordinary values do not carry certified relative errors.");
}

template<ArithmeticMode Mode> bool bounded_points_identical(const BoundedPoint& a, const BoundedPoint& b)
{
    static_assert(Mode == kernel_mode);
    require_mode(a.arithmetic_mode, b.arithmetic_mode);
    Operation scope(mode_limits(a.arithmetic_mode));
    (void) point<Mode>(a);
    (void) point<Mode>(b);
    if (BoundedGeometryAccess::same_source(a, b))
        return true;
    return a.value == b.value &&
           std::all_of(a.absolute_error.begin(), a.absolute_error.end(), [](auto x) { return x == 0; }) &&
           std::all_of(b.absolute_error.begin(), b.absolute_error.end(), [](auto x) { return x == 0; });
}

template<ArithmeticMode Mode> BoundedGeometryValue bounded_orientation(const std::array<BoundedPoint, 4>& vertices)
{
    static_assert(Mode == kernel_mode);
    Operation scope(mode_limits(vertices.front().arithmetic_mode));
    std::array<Point, 4> p;
    for (std::size_t i = 0; i < 4; ++i)
        p[i] = point<Mode>(vertices[i]);
    if (!bounded_common_supports(vertices).empty())
        return bounded<Mode>(Scalar{}); // Authenticated original incidence proves coplanarity.
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (BoundedGeometryAccess::same_source(vertices[i], vertices[j]))
                return bounded<Mode>(Scalar{});
    // An identical exact input vertex establishes zero, unlike identical
    // distinct coordinates from unrelated constructions.
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (same<Mode>(p[i], p[j]))
                return bounded<Mode>(Scalar{});
    Point a = p[1] - p[0], b = p[2] - p[0], c = p[3] - p[0];
    long double extent = 0;
    for (const auto* vector : {&a, &b, &c})
        for (const auto x : *vector)
            extent = std::max(extent, std::abs(x.value));
    if (extent == 0)
        return bounded<Mode>(Scalar{});
    int exponent;
    std::frexp(extent, &exponent);
    return bounded<Mode>(scaled<Mode>(
        dot<Mode>(scaled<Mode>(a, -exponent), cross<Mode>(scaled<Mode>(b, -exponent), scaled<Mode>(c, -exponent))),
        3 * exponent));
}

template<ArithmeticMode Mode>
BoundedConvexGeometryPtr GeometryKernel<Mode>::make(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    if (vertices.size() < 4 || vertices.size() > limits.maximum_vertices || faces.size() < 4 ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive shell size before conversion allocation.");
    active_work->charge(vertices.size(), sizeof(BoundedPoint));
    std::vector<BoundedPoint> relative;
    for (const auto& vertex : vertices)
        relative.push_back(bounded<Mode>(point<Mode>(vertex) - point<Mode>(vertices.front())));
    return BoundedConvexGeometryPtr(
        new BoundedConvexGeometry(import<Mode>(vertices.front(), relative, faces, limits, false)));
}

template<ArithmeticMode Mode>
BoundedConvexGeometryPtr GeometryKernel<Mode>::make_bounded(
    std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    if (vertices.size() < 4 || vertices.size() > limits.maximum_vertices || faces.size() < 4 ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive shell size before conversion allocation.");
    active_work->charge(vertices.size(), sizeof(BoundedPoint));
    GeometryPoint origin;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        origin[axis] = static_cast<double>(vertices.front().value[axis]);
        if (!std::isfinite(origin[axis]))
            reject("input anchor outside binary64 source-coordinate range.");
    }
    std::vector<BoundedPoint> relative;
    active_work->charge(vertices.size(), sizeof(std::shared_ptr<const BoundedPointProvenance>));
    std::vector<std::shared_ptr<const BoundedPointProvenance>> identities;
    identities.reserve(vertices.size());
    for (const auto& vertex : vertices)
    {
        const auto input = converted_point<Mode>(vertex, limits.arithmetic_mode);
        identities.push_back(BoundedGeometryAccess::identity(input));
        auto p = bounded<Mode>(point<Mode>(input) - point<Mode>(origin));
        BoundedGeometryAccess::seal(p, input.supports());
        relative.push_back(std::move(p));
    }
    return BoundedConvexGeometryPtr(
        new BoundedConvexGeometry(import<Mode>(origin, relative, faces, limits, false, identities)));
}

template<ArithmeticMode Mode>
BoundedConvexGeometryPtr GeometryKernel<Mode>::make_bounded_local(GeometryPoint origin,
    std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    if (vertices.size() < 4 || vertices.size() > limits.maximum_vertices || faces.size() < 4 ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive local shell size before conversion allocation.");
    active_work->charge(vertices.size(), sizeof(BoundedPoint));
    std::vector<BoundedPoint> unsealed;
    for (const auto& p : vertices)
    {
        const auto input = converted_point<Mode>(p, limits.arithmetic_mode);
        unsealed.emplace_back(input.value, input.absolute_error, limits.arithmetic_mode);
    }
    // Public relative coordinates do not authenticate the global supports of a
    // point copied from a different frame. The private source importer carries
    // both frame and original support authority together.
    return BoundedConvexGeometryPtr(new BoundedConvexGeometry(import<Mode>(origin, unsealed, faces, limits, false)));
}

template<ArithmeticMode Mode>
BoundedConvexGeometryPtr GeometryKernel<Mode>::from_geometry(
    const ConvexCellGeometry& source, BoundedGeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    ++statistics.source_imports;
    if (source.storage_bytes() > limits.byte_budget)
        reject("source authority exceeds bootstrap storage cap.");
    const auto shell = source.numerical_shell();
    std::vector<BoundedPoint> relative;
    for (const auto& vertex : shell.relative_vertices)
    {
        BoundedPoint p;
        p.arithmetic_mode = limits.arithmetic_mode;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            p.value[axis] = vertex[axis].value;
            p.absolute_error[axis] = vertex[axis].absolute_error;
        }
        relative.push_back(p);
    }
    if (shell.face_support_identities.size() != shell.outward_faces.size())
        reject("source authority omitted original face identities.");
    for (std::size_t f = 0; f < shell.outward_faces.size(); ++f)
    {
        const auto& bytes = shell.face_support_identities[f];
        if (bytes.empty())
            reject("empty original source support authority.");
        BoundedPlaneAuthority authority;
        authority.key.push_back(bytes.size());
        authority.key.resize(1 + (bytes.size() + 7) / 8, 0);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            authority.key[1 + i / 8] |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i]))
                                        << (8 * (i % 8));
        active_work->charge(authority.key.size(), sizeof(std::uint64_t));
        for (const auto index : shell.outward_faces[f])
        {
            auto supports = relative.at(index).supports();
            supports.push_back(authority);
            BoundedGeometryAccess::seal(relative.at(index), std::move(supports));
        }
    }
    return BoundedConvexGeometryPtr(
        new BoundedConvexGeometry(import<Mode>(shell.origin, relative, shell.outward_faces, limits, true)));
}

template<ArithmeticMode Mode>
BoundedConvexGeometryPtr GeometryKernel<Mode>::box(
    GeometryPoint lower, GeometryPoint upper, BoundedGeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    std::array<GeometryPoint, 8> vertices;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!(upper[axis] > lower[axis]))
            reject("box requires positive represented spans.");
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t axis = 0; axis < 3; ++axis)
            vertices[i][axis] = (i >> axis) & 1 ? upper[axis] : lower[axis];
    return make(vertices, {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}}, limits);
}

template<ArithmeticMode Mode>
BoundedConvexGeometryPtr GeometryKernel<Mode>::tetrahedron(
    const std::array<BoundedPoint, 4>& vertices, BoundedGeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    auto input = vertices;
    for (auto& p : input)
        p = converted_point<Mode>(p, limits.arithmetic_mode);
    const auto determinant = scalar_value<Mode>(bounded_orientation<Mode>(input));
    const auto orientation = sign<Mode>(determinant);
    if (!orientation)
        return {};
    std::vector<Face> faces{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    if (orientation < 0)
        for (auto& face : faces)
            std::reverse(face.begin(), face.end());
    return make_bounded(input, faces, limits);
}

template<ArithmeticMode Mode> BoundedGeometryValue GeometryKernel<Mode>::volume() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    if (impl_->partition_proof_only)
        reject("private partition witness has no volume certificate.");
    return bounded<Mode>(impl_->volume);
}

template<ArithmeticMode Mode> BoundedGeometryMoments GeometryKernel<Mode>::moments() const
{
    static_assert(Mode == kernel_mode);
    return moments(impl_->reference_origin);
}

template<ArithmeticMode Mode> BoundedGeometryMoments GeometryKernel<Mode>::moments(GeometryPoint origin) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    if (impl_->partition_proof_only)
        reject("private partition witness has no moment certificate.");
    std::call_once(impl_->moment_cache->prepared,
        [&]
        {
            ++statistics.moment_evaluations;
            impl_->moment_cache->integrals = integrate<Mode>(*impl_, true);
        });
    auto result = *impl_->moment_cache->integrals;
    const auto shift = point<Mode>(impl_->frame.origin) - point<Mode>(origin);
    result.first = result.first + shift * result.volume;
    BoundedGeometryMoments output;
    output.origin = origin;
    output.volume = bounded<Mode>(result.volume);
    for (std::size_t axis = 0; axis < 3; ++axis)
        output.first_moment[axis] = bounded<Mode>(result.first[axis]);
    return output;
}

template<ArithmeticMode Mode> GeometryPoint GeometryKernel<Mode>::centroid() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    const auto m = moments();
    GeometryPoint result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto coordinate =
            scalar_value<Mode>(m.first_moment[axis]) / impl_->volume + Scalar(impl_->reference_origin[axis]);
        result[axis] = static_cast<double>(bounded<Mode>(coordinate).value);
        if (!std::isfinite(result[axis]))
            reject("centroid outside publication range.");
    }
    return result;
}

template<ArithmeticMode Mode> std::vector<BoundedPoint> GeometryKernel<Mode>::local_points() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    active_work->charge(impl_->vertices.size(), sizeof(BoundedPoint));
    std::vector<BoundedPoint> result;
    result.reserve(impl_->vertices.size());
    for (const auto& vertex : impl_->vertices)
    {
        auto p = bounded<Mode>(scaled<Mode>(vertex.coordinates, impl_->frame.exponent) +
                               (point<Mode>(impl_->frame.origin) - point<Mode>(impl_->reference_origin)));
        // This public point is relative to origin(), not a global source
        // point. Only the private framed importer may carry global authority.
        result.push_back(std::move(p));
    }
    return result;
}

template<ArithmeticMode Mode> std::array<BoundedGeometryValue, 3> GeometryKernel<Mode>::axis_spans() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    std::array<BoundedGeometryValue, 3> result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        Scalar minimum = impl_->vertices.front().coordinates[axis], maximum = minimum;
        for (const auto& vertex : impl_->vertices)
        {
            const auto x = vertex.coordinates[axis];
            minimum = Scalar(std::min(minimum.value, x.value));
            maximum = Scalar(std::max(maximum.value, x.value));
        }
        result[axis] = bounded<Mode>(scaled<Mode>(maximum - minimum, impl_->frame.exponent));
    }
    return result;
}

template<ArithmeticMode Mode>
BoundedSupport GeometryKernel<Mode>::support(GeometryPoint normal, GeometryPoint anchor) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    if (!nonzero<Mode>(point<Mode>(normal)))
        reject("support requires a nonzero normal.");
    const auto offset = point<Mode>(impl_->frame.origin) - point<Mode>(anchor);
    std::optional<Scalar> minimum, maximum;
    for (const auto& vertex : impl_->vertices)
    {
        const auto projection =
            dot<Mode>(point<Mode>(normal), scaled<Mode>(vertex.coordinates, impl_->frame.exponent) + offset);
        minimum = minimum ? Scalar(std::min(minimum->value, projection.value)) : projection;
        maximum = maximum ? Scalar(std::max(maximum->value, projection.value)) : projection;
    }
    return {bounded<Mode>(*minimum), bounded<Mode>(*maximum)};
}

template Integrals integrate_ordinary<kernel_mode>(const StoredGeometry& poly, bool moments);
template Integrals integrate<kernel_mode>(const StoredGeometry& poly, bool moments);
template void finish<kernel_mode>(StoredGeometry& poly, bool partition_proof, bool positive_construction);
template void validate_connected_boundary<kernel_mode>(std::size_t vertex_count, const std::vector<Face>& faces);
template std::shared_ptr<StoredGeometry> import<kernel_mode>(GeometryPoint origin,
    std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits,
    bool authenticated_planarity, std::span<const std::shared_ptr<const BoundedPointProvenance>> identities);
template BoundedPoint bounded_point_average<kernel_mode>(std::span<const BoundedPoint> vertices);
template BoundedPoint bounded_point_interpolate<kernel_mode>(
    const BoundedPoint& a, const BoundedPoint& b, long double fraction);
template long double ValueKernel<kernel_mode>::lower() const;
template long double ValueKernel<kernel_mode>::upper() const;
template double ValueKernel<kernel_mode>::as_double() const;
template bool ValueKernel<kernel_mode>::relative_error_at_most(long double target, double tolerance) const;
template bool bounded_points_identical<kernel_mode>(const BoundedPoint& a, const BoundedPoint& b);
template BoundedGeometryValue bounded_orientation<kernel_mode>(const std::array<BoundedPoint, 4>& vertices);
template BoundedConvexGeometryPtr GeometryKernel<kernel_mode>::make(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits);
template BoundedConvexGeometryPtr GeometryKernel<kernel_mode>::make_bounded(
    std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits);
template BoundedConvexGeometryPtr GeometryKernel<kernel_mode>::make_bounded_local(GeometryPoint origin,
    std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits);
template BoundedConvexGeometryPtr GeometryKernel<kernel_mode>::from_geometry(
    const ConvexCellGeometry& source, BoundedGeometryLimits limits);
template BoundedConvexGeometryPtr GeometryKernel<kernel_mode>::box(
    GeometryPoint lower, GeometryPoint upper, BoundedGeometryLimits limits);
template BoundedConvexGeometryPtr GeometryKernel<kernel_mode>::tetrahedron(
    const std::array<BoundedPoint, 4>& vertices, BoundedGeometryLimits limits);
template BoundedGeometryValue GeometryKernel<kernel_mode>::volume() const;
template BoundedGeometryMoments GeometryKernel<kernel_mode>::moments() const;
template BoundedGeometryMoments GeometryKernel<kernel_mode>::moments(GeometryPoint origin) const;
template GeometryPoint GeometryKernel<kernel_mode>::centroid() const;
template std::vector<BoundedPoint> GeometryKernel<kernel_mode>::local_points() const;
template std::array<BoundedGeometryValue, 3> GeometryKernel<kernel_mode>::axis_spans() const;
template BoundedSupport GeometryKernel<kernel_mode>::support(GeometryPoint normal, GeometryPoint anchor) const;
} // namespace ordinary_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes::bounded_geometry_detail
