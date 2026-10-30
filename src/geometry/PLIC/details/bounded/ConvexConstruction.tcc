/** Compiled certification geometry kernels. Included only by the owning implementation unit.
 * Directed interval arithmetic and retained certificates remain confined to this
 * provider; Mode templates retain the compiled implementation boundary.
 */
namespace SimpleFluid::PLIC::details
{
namespace certification_kernel SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode> Integrals integrate(const StoredGeometry& poly, bool moments)
{
    static_assert(Mode == kernel_mode);
    if (poly.vertices.size() == 4 && poly.faces.size() == 4 &&
        std::all_of(poly.faces.begin(), poly.faces.end(), [](const auto& face) { return face.size() == 3; }))
    {
        // A tetrahedron is one cone from any vertex to its opposite outward
        // face. Intersect those four equivalent enclosures instead of summing
        // cones about an interval-valued average with duplicated dependencies.
        // In a thin tetrahedron the useful short edges can depend on the anchor;
        // every anchor remains bounded work and no positive piece is discarded.
        Integrals result;
        if (!moments)
        {
            std::optional<Interval> determinant;
            for (std::size_t pivot = 0; pivot < 4; ++pivot)
                for (const auto& face : poly.faces)
                {
                    if (std::find(face.begin(), face.end(), pivot) != face.end())
                        continue; // This face and pivot are structurally coplanar.
                    const auto& p = poly.vertices[pivot].coordinates;
                    const auto a = poly.vertices[face[0]].coordinates - p;
                    const auto b = poly.vertices[face[1]].coordinates - p;
                    const auto c = poly.vertices[face[2]].coordinates - p;
                    const auto candidate = dot<Mode>(a, cross<Mode>(b, c));
                    if (determinant)
                    {
                        const auto lo = std::max(determinant->lo(), candidate.lo());
                        const auto hi = std::min(determinant->hi(), candidate.hi());
                        if (lo > hi)
                            reject("equivalent tetrahedron volume enclosures are disjoint.");
                        determinant = Interval(lo, hi);
                    }
                    else
                        determinant = candidate;
                }
            result.volume = scaled<Mode>(*determinant / Interval(6), 3 * poly.frame.exponent);
        }
        else
        {
            Point center;
            for (const auto& vertex : poly.vertices)
                center = center + vertex.coordinates;
            result.volume = poly.volume;
            result.first = scaled<Mode>(center * Interval(.25L), poly.frame.exponent) * result.volume;
        }
        return result;
    }
    Point pivot;
    for (const auto& vertex : poly.vertices)
        pivot = pivot + vertex.coordinates;
    pivot = pivot * (Interval(1) / Interval(static_cast<long double>(poly.vertices.size())));
    Integrals sum;
    Accumulator<Mode> volume_sum;
    std::array<Accumulator<Mode>, 3> first_sum;
    // Enclosures accumulate independently of the compensated central estimate;
    // no rounded volume equality or skipped uncertain triangle supplies proof.
    for (const auto& face : poly.faces)
        for (std::size_t i = 1; i + 1 < face.size(); ++i)
        {
            const auto& a = poly.vertices[face[0]].coordinates;
            const auto& b = poly.vertices[face[i]].coordinates;
            const auto& c = poly.vertices[face[i + 1]].coordinates;
            const auto determinant = dot<Mode>(a - pivot, cross<Mode>(b - pivot, c - pivot));
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
    sum.volume = scaled<Mode>(volume_sum.result() / Interval(6), 3 * poly.frame.exponent);
    if (moments)
        for (std::size_t axis = 0; axis < 3; ++axis)
            sum.first[axis] = scaled<Mode>(first_sum[axis].result() / Interval(24), 4 * poly.frame.exponent);
    if (!(sum.volume.lo() > 0))
    {
        const auto common = [](Interval a, Interval b)
        {
            const auto lo = std::max(a.lo(), b.lo()), hi = std::min(a.hi(), b.hi());
            if (lo > hi)
                reject("equivalent polyhedral integral enclosures are disjoint.");
            return Interval(lo, hi);
        };
        // Thin clipped shells can lose the dependency of the shared mean
        // pivot. Retry with each actual vertex, bounded by the admitted vertex
        // count and the same scalar-work cap. Already-positive enclosures do
        // not pay for these routes. Every route integrates the same shell.
        for (std::size_t anchor = 0; anchor < poly.vertices.size(); ++anchor)
        {
            const auto& p = poly.vertices[anchor].coordinates;
            Accumulator<Mode> anchored_volume;
            std::array<Accumulator<Mode>, 3> anchored_first;
            for (const auto& face : poly.faces)
            {
                if (std::find(face.begin(), face.end(), anchor) != face.end())
                    continue; // The entire admitted planar face contains p.
                for (std::size_t i = 1; i + 1 < face.size(); ++i)
                {
                    const auto a = poly.vertices[face[0]].coordinates - p;
                    const auto b = poly.vertices[face[i]].coordinates - p;
                    const auto c = poly.vertices[face[i + 1]].coordinates - p;
                    const auto determinant = dot<Mode>(a, cross<Mode>(b, c));
                    anchored_volume.add(determinant);
                    if (moments)
                    {
                        const auto first = (a + b + c) * determinant;
                        for (std::size_t axis = 0; axis < 3; ++axis)
                            anchored_first[axis].add(first[axis]);
                    }
                }
            }
            const auto volume = scaled<Mode>(anchored_volume.result() / Interval(6), 3 * poly.frame.exponent);
            sum.volume = common(sum.volume, volume);
            if (moments)
            {
                // All moment candidates refer to the numerical frame origin.
                // Restore the anchor shift using the already certified volume.
                const auto shift = scaled<Mode>(p, poly.frame.exponent) * poly.volume;
                for (std::size_t axis = 0; axis < 3; ++axis)
                    sum.first[axis] = common(sum.first[axis],
                        scaled<Mode>(anchored_first[axis].result() / Interval(24), 4 * poly.frame.exponent) +
                            shift[axis]);
            }
        }
    }
    if (moments)
    {
        sum.volume = poly.volume;
        // A positive convex body's centroid lies in its coordinate bounds.
        // This remains a quantitative enclosure when volume has a structural
        // positivity proof but no positive numerical lower bound.
        const auto box_moment = scaled<Mode>(interval_box<Mode>(poly.bounds), poly.frame.exponent) * poly.volume;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto lo = std::max(sum.first[axis].lo(), box_moment[axis].lo());
            const auto hi = std::min(sum.first[axis].hi(), box_moment[axis].hi());
            if (lo > hi)
                reject("convex moment bound contradicts its integral enclosure.");
            sum.first[axis] = Interval(lo, hi);
        }
    }
    return sum;
}

template<ArithmeticMode Mode> void finish(StoredGeometry& poly, bool partition_proof, bool positive_construction)
{
    static_assert(Mode == kernel_mode);
    if (poly.vertices.size() > std::min(poly.limits.maximum_vertices, active_work->limits.maximum_vertices) ||
        poly.faces.size() > std::min(poly.limits.maximum_faces, active_work->limits.maximum_faces))
        reject("vertex/face cap exhausted.");
    active_work->charge(1, sizeof(MomentCache));
    active_work->charge(poly.vertices.size(), sizeof(Vertex) + sizeof(DynamicPoint));
    active_work->charge(poly.faces.size(), sizeof(Face) + sizeof(Surface));
    poly.displayed.clear();
    for (auto& v : poly.vertices)
    {
        if (v.source_identity)
            v.authorities.share_if_equal(v.source_identity->support_payload);
        auto displayed =
            bounded<Mode>(scaled<Mode>(v.coordinates, poly.frame.exponent) + point<Mode>(poly.frame.origin));
        if (v.source_identity || poly.construction || poly.numerical_only)
            DynamicGeometryAccess::seal_export(displayed, v.source_identity, v.authorities.values(), poly.numerical_only);
        else
            DynamicGeometryAccess::seal(displayed, v.authorities.values());
        DynamicGeometryAccess::retain_supports(DynamicGeometryAccess::identity(displayed), active_work->allocations,
            active_work->bytes, active_work->limits.byte_budget);
        active_work->charge(DynamicGeometryAccess::storage_bytes(displayed, false) - sizeof(DynamicPoint), 1);
        poly.displayed.push_back(std::move(displayed));
        active_work->charge(v.incident.size(), sizeof(PlanePtr));
        active_work->charge(v.authorities.size(), sizeof(PlaneAuthority));
        DynamicGeometryAccess::retain_supports(
            v.source_identity, active_work->allocations, active_work->bytes, active_work->limits.byte_budget);
        active_work->charge(DynamicGeometryAccess::identity_bytes(v.source_identity, false), 1);
        for (const auto& key : v.authorities)
            active_work->charge(key.key.size(), sizeof(std::uint64_t));
    }
    std::set<PlanePtr> retained_planes;
    for (const auto& v : poly.vertices)
        retained_planes.insert(v.incident.begin(), v.incident.end());
    for (const auto& plane : retained_planes)
    {
        active_work->charge(1, sizeof(Plane));
        active_work->charge(plane->source_vertices.capacity(), sizeof(std::weak_ptr<const DynamicPointProvenance>));
        active_work->charge(plane->authorities.size(), sizeof(PlaneAuthority));
        for (const auto& key : plane->authorities)
            active_work->charge(key.key.size(), sizeof(std::uint64_t));
    }
    for (const auto& f : poly.faces)
        active_work->charge(f.size(), sizeof(std::size_t));
    // Local coordinate bounds retain the full construction enclosure. Compare
    // in one local frame later, avoiding cancellation at large global origins.
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        auto lo = poly.vertices.front().coordinates[axis].lo();
        auto hi = poly.vertices.front().coordinates[axis].hi();
        for (const auto& vertex : poly.vertices)
        {
            lo = std::min(lo, vertex.coordinates[axis].lo());
            hi = std::max(hi, vertex.coordinates[axis].hi());
        }
        poly.bounds[axis] = {lo, hi};
    }
    poly.partition_proof_only = partition_proof;
    if (!partition_proof)
    {
        ++statistics.volume_evaluations;
        poly.volume = integrate<Mode>(poly, false).volume;
        if (!(poly.volume.lo() > 0))
        {
            if (!positive_construction)
                reject("strictly positive volume cannot be established.");
            if (!(poly.volume.hi() > 0))
                reject("positive construction contradicts its volume enclosure.");
            // Strict opposite signs on a full-dimensional parent prove
            // positive child interior. They do not supply a positive numeric
            // lower bound. Intersect only with the proved nonnegative domain;
            // retain the full upper uncertainty for every quantitative gate.
            poly.volume = Interval(0, poly.volume.hi());
        }
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
std::shared_ptr<StoredGeometry> import(GeometryPoint origin, std::span<const DynamicPoint> vertices,
    const std::vector<Face>& faces, GeometryLimits limits, bool authenticated_planarity,
    std::span<const std::shared_ptr<const DynamicPointProvenance>> identities)
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
            extent = std::max({extent, std::abs(x.lo()), std::abs(x.hi())});
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
        std::vector<DynamicPoint> face_points;
        for (const auto index : face)
            face_points.push_back(vertices[index]);
        plane->authorities = dynamic_common_supports(face_points);
        for (const auto index : face)
        {
            const auto& coordinate = result->vertices[index].coordinates;
            if (!authenticated_planarity && index != face[0] && index != face[second] && index != face[third] &&
                !dot<Mode>(plane->normal, coordinate - plane->anchor).zero())
                reject("face coplanarity unresolved; an enclosure containing zero is not coplanarity.");
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
                if (orientation.hi() < 0)
                    reject("nonconvex face loop.");
                if (orientation.lo() < 0 && !orientation.zero())
                {
                    // Source authentication establishes planar convex loops;
                    // arbitrary bounded public shells need a decisive turn.
                    if (!authenticated_planarity)
                        reject("face convexity unresolved.");
                }
            }
        }
    }
    finish<Mode>(*result);
    return result;
}

template<ArithmeticMode Mode> DynamicPoint dynamic_point_average(std::span<const DynamicPoint> vertices)
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
        average[axis] = sum[axis].result() / Interval(static_cast<long double>(vertices.size()));
    auto result = bounded<Mode>(average);
    DynamicGeometryAccess::seal(result, dynamic_common_supports(vertices));
    return result;
}

template<ArithmeticMode Mode>
DynamicPoint dynamic_point_interpolate(const DynamicPoint& a, const DynamicPoint& b, long double fraction)
{
    static_assert(Mode == kernel_mode);
    require_mode(a.arithmetic_mode, b.arithmetic_mode);
    Operation scope(mode_limits(a.arithmetic_mode));
    auto result = bounded<Mode>(point<Mode>(a) + (point<Mode>(b) - point<Mode>(a)) * Interval(fraction));
    const std::array vertices{a, b};
    DynamicGeometryAccess::seal(result, dynamic_common_supports(vertices));
    return result;
}

template<ArithmeticMode Mode> long double ValueKernel<Mode>::lower() const
{
    static_assert(Mode == kernel_mode);
    if (arithmetic_mode != ArithmeticMode::BoundedCertification)
        reject("ordinary values do not carry certified bounds.");
    Operation scope(mode_limits(arithmetic_mode));
    return enclosure<Mode>(*this).lo();
}

template<ArithmeticMode Mode> long double ValueKernel<Mode>::upper() const
{
    static_assert(Mode == kernel_mode);
    if (arithmetic_mode != ArithmeticMode::BoundedCertification)
        reject("ordinary values do not carry certified bounds.");
    Operation scope(mode_limits(arithmetic_mode));
    return enclosure<Mode>(*this).hi();
}

template<ArithmeticMode Mode> double ValueKernel<Mode>::as_double() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(mode_limits(arithmetic_mode));
    const auto interval = enclosure<Mode>(*this);
    const auto result = static_cast<double>(value);
    if (!std::isfinite(result) || (result == 0 && (interval.lo() != 0 || interval.hi() != 0)))
        reject("publication would lose a nonzero/uncertain amount.");
    return result;
}

template<ArithmeticMode Mode> bool ValueKernel<Mode>::relative_error_at_most(long double target, double tolerance) const
{
    static_assert(Mode == kernel_mode);
    if (arithmetic_mode != ArithmeticMode::BoundedCertification)
        reject("ordinary values do not carry certified relative errors.");
    Operation scope(mode_limits(arithmetic_mode));
    if (!std::isfinite(target) || !std::isfinite(tolerance) || tolerance < 0)
        reject("invalid relative acceptance target.");
    const auto error = enclosure<Mode>(*this) - Interval(target);
    if (target == 0)
        return error.zero();
    const auto allowance = Interval(std::abs(target)) * Interval(tolerance);
    return std::max(std::abs(error.lo()), std::abs(error.hi())) <= allowance.lo();
}

template<ArithmeticMode Mode> bool dynamic_points_identical(const DynamicPoint& a, const DynamicPoint& b)
{
    static_assert(Mode == kernel_mode);
    require_mode(a.arithmetic_mode, b.arithmetic_mode);
    Operation scope(mode_limits(a.arithmetic_mode));
    (void) point<Mode>(a);
    (void) point<Mode>(b);
    if (DynamicGeometryAccess::same_source(a, b))
        return true;
    return a.value == b.value &&
           std::all_of(a.absolute_error.begin(), a.absolute_error.end(), [](auto x) { return x == 0; }) &&
           std::all_of(b.absolute_error.begin(), b.absolute_error.end(), [](auto x) { return x == 0; });
}

template<ArithmeticMode Mode> DynamicGeometryValue dynamic_orientation(const std::array<DynamicPoint, 4>& vertices)
{
    static_assert(Mode == kernel_mode);
    Operation scope(mode_limits(vertices.front().arithmetic_mode));
    std::array<Point, 4> p;
    for (std::size_t i = 0; i < 4; ++i)
        p[i] = point<Mode>(vertices[i]);
    if (!dynamic_common_supports(vertices).empty())
        return bounded<Mode>(Interval{}); // Authenticated original incidence proves coplanarity.
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (DynamicGeometryAccess::same_source(vertices[i], vertices[j]))
                return bounded<Mode>(Interval{});
    // An identical exact input vertex establishes zero, unlike identical
    // nonzero-width coordinate enclosures from unrelated constructions.
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (same<Mode>(p[i], p[j]) &&
                std::all_of(p[i].begin(), p[i].end(), [](auto x) { return x.lo() == x.hi(); }))
                return bounded<Mode>(Interval{});
    Point a = p[1] - p[0], b = p[2] - p[0], c = p[3] - p[0];
    long double extent = 0;
    for (const auto* vector : {&a, &b, &c})
        for (const auto x : *vector)
            extent = std::max({extent, std::abs(x.lo()), std::abs(x.hi())});
    if (extent == 0)
        return bounded<Mode>(Interval{});
    int exponent;
    std::frexp(extent, &exponent);
    return bounded<Mode>(scaled<Mode>(
        dot<Mode>(scaled<Mode>(a, -exponent), cross<Mode>(scaled<Mode>(b, -exponent), scaled<Mode>(c, -exponent))),
        3 * exponent));
}

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr GeometryKernel<Mode>::make(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    if (vertices.size() < 4 || vertices.size() > limits.maximum_vertices || faces.size() < 4 ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive shell size before conversion allocation.");
    active_work->charge(vertices.size(), sizeof(DynamicPoint));
    std::vector<DynamicPoint> relative;
    active_work->charge(vertices.size(), sizeof(std::shared_ptr<const DynamicPointProvenance>));
    std::vector<std::shared_ptr<const DynamicPointProvenance>> identities;
    identities.reserve(vertices.size());
    for (const auto& vertex : vertices)
    {
        DynamicPoint source{{vertex[0], vertex[1], vertex[2]}, {}, Mode};
        active_work->charge(1, sizeof(DynamicPointProvenance));
        DynamicGeometryAccess::seal_represented_source(source, {});
        identities.push_back(DynamicGeometryAccess::identity(source));
        relative.push_back(bounded<Mode>(point<Mode>(vertex) - point<Mode>(vertices.front())));
    }
    return DynamicConvexGeometryPtr(
        new DynamicConvexGeometry(import<Mode>(vertices.front(), relative, faces, limits, false, identities)));
}

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr GeometryKernel<Mode>::make_bounded(
    std::span<const DynamicPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    if (vertices.size() < 4 || vertices.size() > limits.maximum_vertices || faces.size() < 4 ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive shell size before conversion allocation.");
    active_work->charge(vertices.size(), sizeof(DynamicPoint));
    GeometryPoint origin;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        origin[axis] = static_cast<double>(vertices.front().value[axis]);
        if (!std::isfinite(origin[axis]))
            reject("input anchor outside binary64 source-coordinate range.");
    }
    std::vector<DynamicPoint> relative;
    active_work->charge(vertices.size(), sizeof(std::shared_ptr<const DynamicPointProvenance>));
    std::vector<std::shared_ptr<const DynamicPointProvenance>> identities;
    identities.reserve(vertices.size());
    for (const auto& vertex : vertices)
    {
        const auto input = converted_point<Mode>(vertex, limits.arithmetic_mode);
        identities.push_back(DynamicGeometryAccess::identity(input));
        auto p = bounded<Mode>(point<Mode>(input) - point<Mode>(origin));
        DynamicGeometryAccess::seal(p, input.supports());
        relative.push_back(std::move(p));
    }
    return DynamicConvexGeometryPtr(
        new DynamicConvexGeometry(import<Mode>(origin, relative, faces, limits, false, identities)));
}

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr GeometryKernel<Mode>::make_bounded_local(
    GeometryPoint origin, std::span<const DynamicPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    if (vertices.size() < 4 || vertices.size() > limits.maximum_vertices || faces.size() < 4 ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive local shell size before conversion allocation.");
    active_work->charge(vertices.size(), sizeof(DynamicPoint));
    std::vector<DynamicPoint> unsealed;
    for (const auto& p : vertices)
    {
        const auto input = converted_point<Mode>(p, limits.arithmetic_mode);
        unsealed.emplace_back(input.value, input.absolute_error, limits.arithmetic_mode);
    }
    // Public relative coordinates do not authenticate the global supports of a
    // point copied from a different frame. The private source importer carries
    // both frame and original support authority together.
    return DynamicConvexGeometryPtr(new DynamicConvexGeometry(import<Mode>(origin, unsealed, faces, limits, false)));
}

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr GeometryKernel<Mode>::from_snapshot(
    const GeometrySourceSnapshot& source, GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    ++statistics.source_imports;
    charge_source_preparation(source);
    const auto& shell = source.shell;
    std::vector<DynamicPoint> relative;
    for (const auto& vertex : shell.relative_vertices)
    {
        DynamicPoint p;
        p.arithmetic_mode = limits.arithmetic_mode;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            p.value[axis] = vertex[axis].value;
            p.absolute_error[axis] = 0;
        }
        relative.push_back(p);
    }
    // Ordinary source topology is not an interval certificate. Treat the
    // represented local coordinates as fresh inputs and independently establish
    // planarity/convexity; inherited source IDs cannot force a zero residual.
    active_work->charge(relative.size(), sizeof(std::shared_ptr<const DynamicPointProvenance>));
    std::vector<std::shared_ptr<const DynamicPointProvenance>> identities;
    identities.reserve(relative.size());
    for (const auto& point : relative)
    {
        auto global = bounded<Mode>(certification_kernel::point<Mode>(point) +
                                    certification_kernel::point<Mode>(shell.origin));
        active_work->charge(1, sizeof(DynamicPointProvenance));
        DynamicGeometryAccess::seal_represented_source(global, {});
        identities.push_back(DynamicGeometryAccess::identity(global));
    }
    return DynamicConvexGeometryPtr(
        new DynamicConvexGeometry(import<Mode>(shell.origin, relative, shell.outward_faces, limits, false, identities)));
}

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr GeometryKernel<Mode>::box(GeometryPoint lower, GeometryPoint upper, GeometryLimits limits)
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
DynamicConvexGeometryPtr GeometryKernel<Mode>::tetrahedron(
    const std::array<DynamicPoint, 4>& vertices, GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    auto input = vertices;
    for (auto& p : input)
        p = converted_point<Mode>(p, limits.arithmetic_mode);
    const auto determinant = enclosure<Mode>(dynamic_orientation<Mode>(input));
    const auto orientation = sign<Mode>(determinant);
    if (!orientation)
        return {};
    std::vector<Face> faces{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    if (orientation < 0)
        for (auto& face : faces)
            std::reverse(face.begin(), face.end());
    return make_bounded(input, faces, limits);
}

template<ArithmeticMode Mode> DynamicGeometryValue GeometryKernel<Mode>::volume() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    if (impl_->partition_proof_only)
        reject("private partition witness has no volume certificate.");
    return bounded<Mode>(impl_->volume);
}

template<ArithmeticMode Mode> DynamicGeometryMoments GeometryKernel<Mode>::moments() const
{
    static_assert(Mode == kernel_mode);
    return moments(impl_->reference_origin);
}

template<ArithmeticMode Mode> DynamicGeometryMoments GeometryKernel<Mode>::moments(GeometryPoint origin) const
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
    DynamicGeometryMoments output;
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
            enclosure<Mode>(m.first_moment[axis]) / impl_->volume + Interval(impl_->reference_origin[axis]);
        result[axis] = static_cast<double>(bounded<Mode>(coordinate).value);
        if (!std::isfinite(result[axis]))
            reject("centroid outside publication range.");
    }
    return result;
}

template<ArithmeticMode Mode> std::vector<DynamicPoint> GeometryKernel<Mode>::local_points() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    active_work->charge(impl_->vertices.size(), sizeof(DynamicPoint));
    std::vector<DynamicPoint> result;
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

template<ArithmeticMode Mode> std::array<DynamicGeometryValue, 3> GeometryKernel<Mode>::axis_spans() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    std::array<DynamicGeometryValue, 3> result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        Interval minimum = impl_->vertices.front().coordinates[axis], maximum = minimum;
        for (const auto& vertex : impl_->vertices)
        {
            const auto x = vertex.coordinates[axis];
            minimum = {std::min(minimum.lo(), x.lo()), std::min(minimum.hi(), x.hi())};
            maximum = {std::max(maximum.lo(), x.lo()), std::max(maximum.hi(), x.hi())};
        }
        result[axis] = bounded<Mode>(scaled<Mode>(maximum - minimum, impl_->frame.exponent));
    }
    return result;
}

template<ArithmeticMode Mode>
DynamicSupport GeometryKernel<Mode>::support(GeometryPoint normal, GeometryPoint anchor) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    if (!nonzero<Mode>(point<Mode>(normal)))
        reject("support requires a nonzero normal.");
    const auto offset = point<Mode>(impl_->frame.origin) - point<Mode>(anchor);
    std::optional<Interval> minimum, maximum;
    for (const auto& vertex : impl_->vertices)
    {
        const auto projection =
            dot<Mode>(point<Mode>(normal), scaled<Mode>(vertex.coordinates, impl_->frame.exponent) + offset);
        minimum = minimum ? Interval(std::min(minimum->lo(), projection.lo()), std::min(minimum->hi(), projection.hi()))
                          : projection;
        maximum = maximum ? Interval(std::max(maximum->lo(), projection.lo()), std::max(maximum->hi(), projection.hi()))
                          : projection;
    }
    return {bounded<Mode>(*minimum), bounded<Mode>(*maximum)};
}

template Integrals integrate<kernel_mode>(const StoredGeometry& poly, bool moments);
template void finish<kernel_mode>(StoredGeometry& poly, bool partition_proof, bool positive_construction);
template void validate_connected_boundary<kernel_mode>(std::size_t vertex_count, const std::vector<Face>& faces);
template std::shared_ptr<StoredGeometry> import<kernel_mode>(GeometryPoint origin,
    std::span<const DynamicPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits,
    bool authenticated_planarity, std::span<const std::shared_ptr<const DynamicPointProvenance>> identities);
template DynamicPoint dynamic_point_average<kernel_mode>(std::span<const DynamicPoint> vertices);
template DynamicPoint dynamic_point_interpolate<kernel_mode>(
    const DynamicPoint& a, const DynamicPoint& b, long double fraction);
template long double ValueKernel<kernel_mode>::lower() const;
template long double ValueKernel<kernel_mode>::upper() const;
template double ValueKernel<kernel_mode>::as_double() const;
template bool ValueKernel<kernel_mode>::relative_error_at_most(long double target, double tolerance) const;
template bool dynamic_points_identical<kernel_mode>(const DynamicPoint& a, const DynamicPoint& b);
template DynamicGeometryValue dynamic_orientation<kernel_mode>(const std::array<DynamicPoint, 4>& vertices);
template DynamicConvexGeometryPtr GeometryKernel<kernel_mode>::make(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits);
template DynamicConvexGeometryPtr GeometryKernel<kernel_mode>::make_bounded(
    std::span<const DynamicPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits);
template DynamicConvexGeometryPtr GeometryKernel<kernel_mode>::make_bounded_local(GeometryPoint origin,
    std::span<const DynamicPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits);
template DynamicConvexGeometryPtr GeometryKernel<kernel_mode>::from_snapshot(
    const GeometrySourceSnapshot& source, GeometryLimits limits);
template DynamicConvexGeometryPtr GeometryKernel<kernel_mode>::box(
    GeometryPoint lower, GeometryPoint upper, GeometryLimits limits);
template DynamicConvexGeometryPtr GeometryKernel<kernel_mode>::tetrahedron(
    const std::array<DynamicPoint, 4>& vertices, GeometryLimits limits);
template DynamicGeometryValue GeometryKernel<kernel_mode>::volume() const;
template DynamicGeometryMoments GeometryKernel<kernel_mode>::moments() const;
template DynamicGeometryMoments GeometryKernel<kernel_mode>::moments(GeometryPoint origin) const;
template GeometryPoint GeometryKernel<kernel_mode>::centroid() const;
template std::vector<DynamicPoint> GeometryKernel<kernel_mode>::local_points() const;
template std::array<DynamicGeometryValue, 3> GeometryKernel<kernel_mode>::axis_spans() const;
template DynamicSupport GeometryKernel<kernel_mode>::support(GeometryPoint normal, GeometryPoint anchor) const;
} // namespace certification_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
