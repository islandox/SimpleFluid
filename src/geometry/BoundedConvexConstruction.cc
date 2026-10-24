#include "geometry/details/BoundedConvexGeometryInternal.hh"

namespace SimpleFluid::Meshes
{
namespace bounded_geometry_detail
{
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
} // namespace bounded_geometry_detail
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
std::vector<BoundedPlaneAuthority> bounded_common_supports(std::span<const BoundedPoint> points)
{
    if (points.empty())
        return {};
    std::vector<BoundedPlaneAuthority> result;
    for (const auto& key : points.front().supports())
        if (!key.key.empty() &&
            std::all_of(points.begin() + 1, points.end(), [&](const auto& p)
                { return std::find(p.supports().begin(), p.supports().end(), key) != p.supports().end(); }))
            result.push_back(key);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

BoundedPoint bounded_point_average(std::span<const BoundedPoint> vertices)
{
    Operation scope({});
    if (vertices.empty())
        reject("empty affine average.");
    std::array<Accumulator, 3> sum;
    for (const auto& vertex : vertices)
    {
        const auto input = point(vertex);
        for (std::size_t axis = 0; axis < 3; ++axis)
            sum[axis].add(input[axis]);
    }
    Point average;
    for (std::size_t axis = 0; axis < 3; ++axis)
        average[axis] = sum[axis].result() / Interval(static_cast<long double>(vertices.size()));
    auto result = bounded(average);
    BoundedGeometryAccess::seal(result, bounded_common_supports(vertices));
    return result;
}

BoundedPoint bounded_point_interpolate(const BoundedPoint& a, const BoundedPoint& b, long double fraction)
{
    Operation scope({});
    auto result = bounded(point(a) + (point(b) - point(a)) * Interval(fraction));
    const std::array vertices{a, b};
    BoundedGeometryAccess::seal(result, bounded_common_supports(vertices));
    return result;
}

bool bounded_geometry_environment()
{
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__SSE2__) &&                           \
    !defined(__FAST_MATH__) && (!defined(__FINITE_MATH_ONLY__) || __FINITE_MATH_ONLY__ == 0)
    if constexpr (!std::numeric_limits<long double>::is_iec559 || std::numeric_limits<long double>::radix != 2 ||
                  std::numeric_limits<long double>::digits < 64 ||
                  std::numeric_limits<long double>::max_exponent < 16384 || FLT_EVAL_METHOD != 0)
        return false;
    const unsigned csr = _mm_getcsr();
    unsigned short control;
    __asm__ __volatile__("fnstcw %0" : "=m"(control));
    return std::fegetround() == FE_TONEAREST && (csr & ((1u << 15) | (1u << 6) | (3u << 13))) == 0 &&
           (csr & (0x3fu << 7)) == (0x3fu << 7) && (control & 0x3fu) == 0x3fu && (control & (3u << 8)) == (3u << 8) &&
           (control & (3u << 10)) == 0;
#else
    return false;
#endif
}

BoundedGeometryStatistics bounded_geometry_statistics()
{
    return statistics;
}

void reset_bounded_geometry_statistics()
{
    statistics = {};
}

void bounded_record_ledger_operations(std::uint64_t count)
{
    if (count > std::numeric_limits<std::uint64_t>::max() - statistics.ledger_operations)
        reject("ledger instrumentation counter overflow.");
    statistics.ledger_operations += count;
}

long double BoundedGeometryValue::lower() const
{
    Operation scope({});
    return enclosure(*this).lo;
}

long double BoundedGeometryValue::upper() const
{
    Operation scope({});
    return enclosure(*this).hi;
}

double BoundedGeometryValue::as_double() const
{
    Operation scope({});
    const auto interval = enclosure(*this);
    const auto result = static_cast<double>(value);
    if (!std::isfinite(result) || (result == 0 && (interval.lo != 0 || interval.hi != 0)))
        reject("publication would lose a nonzero/uncertain amount.");
    return result;
}

bool BoundedGeometryValue::relative_error_at_most(long double target, double tolerance) const
{
    Operation scope({});
    if (!std::isfinite(target) || !std::isfinite(tolerance) || tolerance < 0)
        reject("invalid relative acceptance target.");
    const auto error = enclosure(*this) - Interval(target);
    if (target == 0)
        return error.zero();
    const auto allowance = Interval(std::abs(target)) * Interval(tolerance);
    return std::max(std::abs(error.lo), std::abs(error.hi)) <= allowance.lo;
}

std::size_t bounded_point_storage_bytes(const BoundedPoint& p)
{
    return BoundedGeometryAccess::storage_bytes(p);
}

bool bounded_points_identical(const BoundedPoint& a, const BoundedPoint& b)
{
    Operation scope({});
    (void) point(a);
    (void) point(b);
    if (BoundedGeometryAccess::same_source(a, b))
        return true;
    return a.value == b.value &&
           std::all_of(a.absolute_error.begin(), a.absolute_error.end(), [](auto x) { return x == 0; }) &&
           std::all_of(b.absolute_error.begin(), b.absolute_error.end(), [](auto x) { return x == 0; });
}

BoundedGeometryValue bounded_orientation(const std::array<BoundedPoint, 4>& vertices)
{
    Operation scope({});
    std::array<Point, 4> p;
    for (std::size_t i = 0; i < 4; ++i)
        p[i] = point(vertices[i]);
    if (!bounded_common_supports(vertices).empty())
        return {}; // Authenticated original incidence proves coplanarity.
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (BoundedGeometryAccess::same_source(vertices[i], vertices[j]))
                return {};
    // An identical exact input vertex establishes zero, unlike identical
    // nonzero-width coordinate enclosures from unrelated constructions.
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (same(p[i], p[j]) && std::all_of(p[i].begin(), p[i].end(), [](auto x) { return x.lo == x.hi; }))
                return {};
    Point a = p[1] - p[0], b = p[2] - p[0], c = p[3] - p[0];
    long double extent = 0;
    for (const auto* vector : {&a, &b, &c})
        for (const auto x : *vector)
            extent = std::max({extent, std::abs(x.lo), std::abs(x.hi)});
    if (extent == 0)
        return {};
    int exponent;
    std::frexp(extent, &exponent);
    return bounded(scaled(dot(scaled(a, -exponent), cross(scaled(b, -exponent), scaled(c, -exponent))), 3 * exponent));
}

BoundedConvexGeometry::BoundedConvexGeometry(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

BoundedConvexGeometryPtr BoundedConvexGeometry::make(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits)
{
    Operation scope(limits);
    if (vertices.size() < 4 || vertices.size() > limits.maximum_vertices || faces.size() < 4 ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive shell size before conversion allocation.");
    active_work->charge(vertices.size(), sizeof(BoundedPoint));
    std::vector<BoundedPoint> relative;
    for (const auto& vertex : vertices)
        relative.push_back(bounded(point(vertex) - point(vertices.front())));
    return BoundedConvexGeometryPtr(
        new BoundedConvexGeometry(import(vertices.front(), relative, faces, limits, false)));
}

BoundedConvexGeometryPtr BoundedConvexGeometry::make_bounded(
    std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits)
{
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
    for (const auto& vertex : vertices)
    {
        auto p = bounded(point(vertex) - point(origin));
        BoundedGeometryAccess::seal(p, vertex.supports());
        relative.push_back(std::move(p));
    }
    return BoundedConvexGeometryPtr(new BoundedConvexGeometry(import(origin, relative, faces, limits, false)));
}

BoundedConvexGeometryPtr BoundedConvexGeometry::make_bounded_local(GeometryPoint origin,
    std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits)
{
    Operation scope(limits);
    if (vertices.size() < 4 || vertices.size() > limits.maximum_vertices || faces.size() < 4 ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive local shell size before conversion allocation.");
    active_work->charge(vertices.size(), sizeof(BoundedPoint));
    std::vector<BoundedPoint> unsealed;
    for (const auto& p : vertices)
        unsealed.emplace_back(p.value, p.absolute_error);
    // Public relative coordinates do not authenticate the global supports of a
    // point copied from a different frame. The private source importer carries
    // both frame and original support authority together.
    return BoundedConvexGeometryPtr(new BoundedConvexGeometry(import(origin, unsealed, faces, limits, false)));
}

BoundedConvexGeometryPtr BoundedConvexGeometry::from_geometry(
    const ConvexCellGeometry& source, BoundedGeometryLimits limits)
{
    Operation scope(limits);
    ++statistics.source_imports;
    if (source.storage_bytes() > limits.byte_budget)
        reject("source authority exceeds bootstrap storage cap.");
    const auto shell = source.numerical_shell();
    std::vector<BoundedPoint> relative;
    for (const auto& vertex : shell.relative_vertices)
    {
        BoundedPoint p;
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
        new BoundedConvexGeometry(import(shell.origin, relative, shell.outward_faces, limits, true)));
}

BoundedConvexGeometryPtr BoundedConvexGeometry::box(
    GeometryPoint lower, GeometryPoint upper, BoundedGeometryLimits limits)
{
    std::array<GeometryPoint, 8> vertices;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!(upper[axis] > lower[axis]))
            reject("box requires positive represented spans.");
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t axis = 0; axis < 3; ++axis)
            vertices[i][axis] = (i >> axis) & 1 ? upper[axis] : lower[axis];
    return make(vertices, {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}}, limits);
}

BoundedConvexGeometryPtr BoundedConvexGeometry::tetrahedron(
    const std::array<BoundedPoint, 4>& vertices, BoundedGeometryLimits limits)
{
    Operation scope(limits);
    const auto determinant = enclosure(bounded_orientation(vertices));
    const auto orientation = sign(determinant);
    if (!orientation)
        return {};
    std::vector<Face> faces{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    if (orientation < 0)
        for (auto& face : faces)
            std::reverse(face.begin(), face.end());
    return make_bounded(vertices, faces, limits);
}

BoundedGeometryValue BoundedConvexGeometry::volume() const
{
    Operation scope(impl_->limits);
    return bounded(impl_->volume);
}

BoundedGeometryMoments BoundedConvexGeometry::moments() const
{
    return moments(impl_->frame.origin);
}

BoundedGeometryMoments BoundedConvexGeometry::moments(GeometryPoint origin) const
{
    Operation scope(impl_->limits);
    std::call_once(impl_->moment_cache->prepared,
        [&]
        {
            ++statistics.moment_evaluations;
            impl_->moment_cache->integrals = integrate(*impl_, true);
        });
    auto result = *impl_->moment_cache->integrals;
    const auto shift = point(impl_->frame.origin) - point(origin);
    result.first = result.first + shift * result.volume;
    BoundedGeometryMoments output;
    output.origin = origin;
    output.volume = bounded(result.volume);
    for (std::size_t axis = 0; axis < 3; ++axis)
        output.first_moment[axis] = bounded(result.first[axis]);
    return output;
}

GeometryPoint BoundedConvexGeometry::centroid() const
{
    Operation scope(impl_->limits);
    const auto m = moments();
    GeometryPoint result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto coordinate = enclosure(m.first_moment[axis]) / impl_->volume + Interval(impl_->frame.origin[axis]);
        result[axis] = static_cast<double>(bounded(coordinate).value);
        if (!std::isfinite(result[axis]))
            reject("centroid outside publication range.");
    }
    return result;
}

GeometryPoint BoundedConvexGeometry::origin() const
{
    return impl_->frame.origin;
}

const std::vector<BoundedPoint>& BoundedConvexGeometry::points() const
{
    return impl_->displayed;
}

std::vector<BoundedPoint> BoundedConvexGeometry::local_points() const
{
    Operation scope(impl_->limits);
    std::vector<BoundedPoint> result;
    for (const auto& vertex : impl_->vertices)
    {
        auto p = bounded(scaled(vertex.coordinates, impl_->frame.exponent));
        // This public point is relative to origin(), not a global source
        // point. Only the private framed importer may carry global authority.
        result.push_back(std::move(p));
    }
    return result;
}

const std::vector<Face>& BoundedConvexGeometry::faces() const
{
    return impl_->faces;
}

std::array<BoundedGeometryValue, 3> BoundedConvexGeometry::axis_spans() const
{
    Operation scope(impl_->limits);
    std::array<BoundedGeometryValue, 3> result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        Interval minimum = impl_->vertices.front().coordinates[axis], maximum = minimum;
        for (const auto& vertex : impl_->vertices)
        {
            const auto x = vertex.coordinates[axis];
            minimum = {std::min(minimum.lo, x.lo), std::min(minimum.hi, x.hi)};
            maximum = {std::max(maximum.lo, x.lo), std::max(maximum.hi, x.hi)};
        }
        result[axis] = bounded(scaled(maximum - minimum, impl_->frame.exponent));
    }
    return result;
}

BoundedSupport BoundedConvexGeometry::support(GeometryPoint normal, GeometryPoint anchor) const
{
    Operation scope(impl_->limits);
    if (!nonzero(point(normal)))
        reject("support requires a nonzero normal.");
    const auto offset = point(impl_->frame.origin) - point(anchor);
    std::optional<Interval> minimum, maximum;
    for (const auto& vertex : impl_->vertices)
    {
        const auto projection = dot(point(normal), scaled(vertex.coordinates, impl_->frame.exponent) + offset);
        minimum =
            minimum ? Interval(std::min(minimum->lo, projection.lo), std::min(minimum->hi, projection.hi)) : projection;
        maximum =
            maximum ? Interval(std::max(maximum->lo, projection.lo), std::max(maximum->hi, projection.hi)) : projection;
    }
    return {bounded(*minimum), bounded(*maximum)};
}
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
namespace bounded_geometry_detail
{
int matching_input(const GeometryHalfspace& a, const GeometryHalfspace& b)
{
    if (a.anchor != b.anchor)
        return 0;
    bool equal = a.offset == b.offset, reversed = a.offset == -b.offset;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        equal = equal && a.normal[axis] == b.normal[axis];
        reversed = reversed && a.normal[axis] == -b.normal[axis];
    }
    return equal ? 1 : reversed ? -1 : 0;
}
} // namespace bounded_geometry_detail
} // namespace SimpleFluid::Meshes
