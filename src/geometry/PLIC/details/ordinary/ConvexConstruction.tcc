/** @file ordinary/ConvexConstruction.tcc
 * @brief Ordinary convex construction and compensated nominal integration.
 */
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include <sstream>
#include "geometry/PLIC/details/ordinary/FixedBoxGeometry.hh"

namespace SimpleFluid::PLIC::details
{
namespace SIMPLEFLUID_LOCAL ordinary_kernel
{
template<ArithmeticMode Mode> void prepare_finish(DraftGeometry& poly, bool partition_proof)
{
    static_assert(Mode == kernel_mode);
    if (poly.vertices.size() > std::min(poly.limits.maximum_vertices, active_work->limits.maximum_vertices) ||
        poly.faces.size() > std::min(poly.limits.maximum_faces, active_work->limits.maximum_faces))
        reject("vertex/face cap exhausted.");
    active_work->charge(1, sizeof(MomentCache));
    if (!poly.moment_cache)
        poly.moment_cache = std::make_shared<MomentCache>();
    if (!poly.buffers_prepaid)
    {
        active_work->charge(poly.vertices.capacity(), sizeof(Vertex));
        active_work->charge(poly.faces.capacity(), sizeof(Face));
        active_work->charge(poly.surfaces.capacity(), sizeof(Surface));
    }
    // Draft rebuilds invalidate all derived caches together. In particular,
    // a copied shell must never clear the immutable parent's shared cache.
    // A populated cache has a completed once_flag and cannot be cleared in
    // place. Only a unique, never-prepared cache is reusable by a fresh draft.
    if (!poly.moment_cache || !poly.moment_cache.unique() || poly.moment_cache->integrals)
        poly.moment_cache = std::make_shared<MomentCache>();
    poly.volume = StorageScalar{};
    active_work->charge(1, sizeof(PointExportCache));
    poly.point_export_cache = std::make_shared<PointExportCache>();
    if (std::fegetenv(&poly.point_export_cache->environment) != 0)
        reject("cannot capture global-coordinate arithmetic environment.");
    active_work->charge(poly.vertices.size(), sizeof(DynamicStoredPointRef));
    poly.point_export_cache->vertex_provenance.resize(poly.vertices.size());
    if (poly.point_export_cache->vertex_provenance.capacity() > poly.vertices.size())
        active_work->charge(poly.point_export_cache->vertex_provenance.capacity() - poly.vertices.size(),
            sizeof(DynamicStoredPointRef));
    poly.export_source.reset();
    poly.native_export_source.reset();
    // Recipe attachment after finish must not change the point publication policy.
    poly.export_seal = bool(poly.construction) || poly.numerical_only;
    for (auto& v : poly.vertices)
    {
        // Validate physical-coordinate range now without allocating a publication.
        (void) bounded<Mode>(scaled<Mode>(v.coordinates, poly.frame.exponent) + point<Mode>(poly.frame.origin));
        if (!poly.buffers_prepaid)
            active_work->charge(v.incident.capacity(), sizeof(PlanePtr));
        for (const auto& key : resolve_authorities(poly.point_context, v.authorities))
        {
            OriginalSupportAuthorityAccess::validate_boundary(key);
            active_work->allocations.retain(key, active_work->bytes, active_work->limits.byte_budget);
        }
    }
    if (poly.point_context)
        active_work->allocations.template retain_snapshot<Mode>(
            poly.point_context, active_work->bytes, active_work->limits.byte_budget);
    std::set<PlanePtr> retained_planes;
    for (const auto& v : poly.vertices)
        for (const auto& plane : v.incident)
            if (!retained_planes.contains(plane))
            {
                active_work->charge(1, sizeof(PlanePtr) + 4 * sizeof(void*));
                retained_planes.insert(plane);
            }
    for (const auto& plane : retained_planes)
    {
        if (!poly.planes_prepaid)
            active_work->charge(1, sizeof(Plane));
        if (plane->native_support)
            active_work->retain_identity(*plane->native_support);
        if (!poly.planes_prepaid)
            active_work->charge(plane->source_vertices.capacity(), sizeof(StoredPointRef<Mode>));
        for (const auto& key : resolve_authorities(plane->point_context, plane->authorities))
            active_work->allocations.retain(key, active_work->bytes, active_work->limits.byte_budget);
    }
    if (!poly.buffers_prepaid)
        for (const auto& f : poly.faces)
            active_work->charge(f.capacity(), sizeof(std::size_t));
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
}

template<ArithmeticMode Mode> void finish(DraftGeometry& poly, bool partition_proof, bool positive_construction)
{
    static_assert(Mode == kernel_mode);
    prepare_finish<Mode>(poly, partition_proof);
    if (!partition_proof)
    {
        ++statistics.volume_evaluations;
        poly.volume = integrate<Mode>(poly, false).volume;
        represented_positive_volume(poly.volume);
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
    std::span<const PointProvenanceView> identities, PointSnapshotBuilder<Mode>* point_builder)
{
    static_assert(Mode == kernel_mode);
    if (!identities.empty() && identities.size() != vertices.size())
        reject("source point identity count differs from the imported shell.");
    if (vertices.size() < 4 || faces.size() < 4 || vertices.size() > limits.maximum_vertices ||
        faces.size() > limits.maximum_faces)
        reject("invalid or excessive convex shell size.");
    for (const auto& face : faces)
        active_work->charge(face.size(), 4 * sizeof(std::size_t) + 4 * sizeof(void*));
    std::optional<PointSnapshotBuilder<Mode>> local_points;
    if (!point_builder)
        local_points.emplace();
    auto& points = point_builder ? *point_builder : *local_points;
    active_work->charge(1, sizeof(DraftGeometry) + sizeof(Ancestry));
    auto result = std::make_shared<DraftGeometry>();
    result->reference_origin = origin;
    result->limits = limits;
    std::vector<Point> input;
    for (const auto& vertex : vertices)
        input.push_back(point<Mode>(vertex));
    result->frame = represented_source_frame(origin, std::span<const Point>(input));
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        const auto authority = !identities.empty() && identities[i] ? identities[i].supports() : vertices[i].supports();
        result->vertices.push_back(
            {scaled<Mode>(input[i], -result->frame.exponent), {}, points.import_authority_range(authority),
                identities.empty() ? StoredPointRef<Mode>{} : points.import(identities[i])});
    }
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
        plane->point_context = points.context();
        plane->frame = result->frame;
        plane->anchor = result->vertices[face[0]].coordinates;
        const auto face_normal = represented_source_face_normal(*result, face);
        plane->normal = face_normal.normal;
        // The witness still comes only from original source coordinates.
        const auto& defining = face_normal.defining;
        std::array<Point, 3> original_points;
        std::array<std::array<long double, 3>, 3> original_errors;
        bool known_normal = true;
        for (std::size_t i = 0; i < defining.size(); ++i)
        {
            const auto index = defining[i];
            original_points[i] = result->vertices[index].coordinates;
            if (!authenticated_planarity && !identities.empty() && identities[index] && identities[index].constructed())
            {
                known_normal = false;
                break;
            }
            const auto error =
                source_point_error(vertices[index], identities.empty() ? PointProvenanceView{} : identities[index],
                    result->frame, original_points[i], authenticated_planarity);
            if (!error)
            {
                known_normal = false;
                break;
            }
            original_errors[i] = *error;
        }
        if (known_normal)
            plane->normal_error = source_normal_error(original_points, original_errors, plane->normal);
        const auto common_authorities = source_face_authorities(face, vertices, identities);
        plane->authorities = points.authority_range(DynamicGeometryAccess::admit_incidence<Mode>(common_authorities));
        std::vector<details::ordinary_native::WidePoint> native_points;
        bool native_representable = true;
        for (const auto index : face)
        {
            if (!identities.empty() && identities[index])
                native_points.push_back(identities[index].value());
            else
            {
                const auto point =
                    details::ordinary_native::add(vertices[index].value, details::ordinary_native::wide(origin));
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    const long double a = vertices[index].value[axis], b = origin[axis], sum = point[axis];
                    const auto recovered = sum - a;
                    if ((a - (sum - recovered)) + (b - recovered) != 0)
                        native_representable = false;
                }
                native_points.push_back(point);
            }
        }
        if (native_representable)
            plane->native_support = details::ordinary_native::make_source_support(native_points);
        if (!plane->authorities.empty())
            if (const auto* support = details::OriginalSupportAuthorityAccess::get(
                    resolve_authorities(plane->point_context, plane->authorities).front()))
                plane->native_support = *support;
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
            if (sign<Mode>(distance<Mode>(vertex, result->frame, plane, points.context())) > 0)
                reject("shell is not convex/outward.");
        represented_source_face_winding(*result, face, plane->normal);
    }
    result->point_context = points.freeze();
    finish<Mode>(*result);
    return freeze_geometry<Mode>(std::move(*result));
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
        average[axis] = sum[axis].result() / Scalar(static_cast<long double>(vertices.size()));
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
    auto result = bounded<Mode>(point<Mode>(a) + (point<Mode>(b) - point<Mode>(a)) * Scalar(fraction));
    const std::array vertices{a, b};
    DynamicGeometryAccess::seal(result, dynamic_common_supports(vertices));
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
        return bounded<Mode>(Scalar{}); // Authenticated original incidence proves coplanarity.
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (DynamicGeometryAccess::same_source(vertices[i], vertices[j]))
                return bounded<Mode>(Scalar{});
    const auto determinant = ordinary_predicates::orientation(
        {predicate_point(p[0]), predicate_point(p[1]), predicate_point(p[2]), predicate_point(p[3])});
    (void) ordinary_predicates::require_sign(determinant);
    return bounded<Mode>(Scalar(determinant.value));
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
    PointSnapshotBuilder<Mode> points;
    std::vector<PointProvenanceView> identities;
    active_work->charge(vertices.size(), sizeof(PointProvenanceView));
    for (const auto& vertex : vertices)
    {
        DynamicPoint original{{vertex[0], vertex[1], vertex[2]}, {}, ArithmeticMode::Ordinary};
        identities.push_back(points.native_source(original.value, original.absolute_error, {}));
        relative.push_back(represented_source_relative(vertex, vertices.front()));
    }
    return DynamicConvexGeometryPtr(
        new DynamicConvexGeometry(import<Mode>(vertices.front(), relative, faces, limits, false, identities, &points)));
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
    active_work->charge(vertices.size(), sizeof(PointProvenanceView));
    PointSnapshotBuilder<Mode> points;
    std::vector<PointProvenanceView> identities;
    identities.reserve(vertices.size());
    for (const auto& vertex : vertices)
    {
        auto input = converted_point<Mode>(vertex, limits.arithmetic_mode);
        const auto original = DynamicGeometryAccess::identity(input);
        identities.push_back(original ? resolve_point(points.context(), points.import(original))
                                      : points.native_source(input.value, input.absolute_error, {}));
        auto p = bounded<Mode>(point<Mode>(input) - point<Mode>(origin));
        relative.push_back(std::move(p));
    }
    return DynamicConvexGeometryPtr(
        new DynamicConvexGeometry(import<Mode>(origin, relative, faces, limits, false, identities, &points)));
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
    if (shell.face_support_identities.size() != shell.outward_faces.size())
        reject("source authority omitted original face identities.");
    std::vector<std::vector<PlaneAuthority>> source_supports;
    active_work->charge(relative.size(), sizeof(std::vector<PlaneAuthority>));
    source_supports.reserve(relative.size());
    if (source_supports.capacity() > relative.size())
        active_work->charge(source_supports.capacity() - relative.size(), sizeof(std::vector<PlaneAuthority>));
    source_supports.resize(relative.size());
    std::vector<std::size_t> degrees;
    active_work->charge(relative.size(), sizeof(std::size_t));
    degrees.reserve(relative.size());
    if (degrees.capacity() > relative.size())
        active_work->charge(degrees.capacity() - relative.size(), sizeof(std::size_t));
    degrees.resize(relative.size());
    for (const auto& face : shell.outward_faces)
        for (const auto index : face)
            ++degrees.at(index);
    for (std::size_t i = 0; i < relative.size(); ++i)
    {
        active_work->charge(degrees[i], sizeof(PlaneAuthority));
        source_supports[i].reserve(degrees[i]);
        if (source_supports[i].capacity() > degrees[i])
            active_work->charge(source_supports[i].capacity() - degrees[i], sizeof(PlaneAuthority));
    }
    for (std::size_t f = 0; f < shell.outward_faces.size(); ++f)
    {
        const auto& identity = shell.face_support_identities[f];
        auto authority = details::original_support_authority(identity);
        active_work->allocations.retain(authority, active_work->bytes, active_work->limits.byte_budget);
        for (const auto index : shell.outward_faces[f])
            source_supports.at(index).push_back(authority);
    }
    // This is an explicit native-authority import, not a numerical atom decode.
    // Retain its original global point definitions so generated points can
    // later identify native source edges, including independently rebuilt roots.
    PointSnapshotBuilder<Mode> points;
    active_work->charge(relative.size(), sizeof(PointProvenanceView));
    std::vector<PointProvenanceView> identities;
    identities.reserve(relative.size());
    if (!source.original_points.empty() && source.original_points.size() != relative.size())
        reject("source authority point snapshot count differs from its shell.");
    for (std::size_t index = 0; index < relative.size(); ++index)
    {
        const auto& point = relative[index];
        auto global = bounded<Mode>(ordinary_kernel::point<Mode>(point) + ordinary_kernel::point<Mode>(shell.origin));
        auto& supports = source_supports[index];
        std::sort(supports.begin(), supports.end());
        supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
        PointProvenanceView facts;
        if (!source.original_points.empty())
        {
            const auto original = DynamicGeometryAccess::borrowed_original_source(
                resolve_point(source.point_context, source.original_points[index]));
            if (original && original.arithmetic_mode() == Mode &&
                original.local_ref().role == PointProvenanceRole::Native && original.value() == global.value &&
                original.absolute_error() == global.absolute_error)
                facts = original;
        }
        identities.push_back(points.native_source(
            global.value, global.absolute_error, DynamicGeometryAccess::admit_incidence<Mode>(supports), facts));
    }
    return DynamicConvexGeometryPtr(new DynamicConvexGeometry(
        import<Mode>(shell.origin, relative, shell.outward_faces, limits, true, identities, &points)));
}

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr GeometryKernel<Mode>::box(GeometryPoint lower, GeometryPoint upper, GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    const auto vertices = fixed_box_corners(lower, upper);
    return make(vertices, fixed_box_face_rows(), limits);
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
    const auto determinant = scalar_value<Mode>(dynamic_orientation<Mode>(input));
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
            impl_->moment_cache->ready.store(true, std::memory_order_release);
        });
    return represented_moments(*impl_->moment_cache->integrals, impl_->frame, origin);
}

template<ArithmeticMode Mode> GeometryPoint GeometryKernel<Mode>::centroid() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    return represented_centroid(moments(), impl_->volume, impl_->reference_origin);
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

template<ArithmeticMode Mode>
void GeometryKernel<Mode>::visit_local_coordinates(
    void* context, void (*visitor)(void*, std::size_t, const PointCoordinates<Mode>&)) const
{
    static_assert(Mode == kernel_mode);
    for (std::size_t index = 0; index < impl_->vertices.size(); ++index)
    {
        const auto coordinates = scaled<Mode>(impl_->vertices[index].coordinates, impl_->frame.exponent) +
                                 (point<Mode>(impl_->frame.origin) - point<Mode>(impl_->reference_origin));
        PointCoordinates<Mode> result;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto value = bounded<Mode>(coordinates[axis]);
            result.value[axis] = value.value;
            if constexpr (Mode == ArithmeticMode::BoundedCertification)
                result.absolute_error[axis] = value.absolute_error;
        }
        visitor(context, index, result);
    }
}

template<ArithmeticMode Mode> std::array<DynamicGeometryValue, 3> GeometryKernel<Mode>::axis_spans() const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    return represented_axis_spans(*impl_);
}

template<ArithmeticMode Mode>
DynamicSupport GeometryKernel<Mode>::support(GeometryPoint normal, GeometryPoint anchor) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    return represented_support(*impl_, normal, anchor);
}

template Integrals integrate_ordinary<kernel_mode>(const StoredGeometry& poly, bool moments);
template Integrals integrate_ordinary<kernel_mode>(const DraftGeometry& poly, bool moments);
template Integrals integrate<kernel_mode>(const StoredGeometry& poly, bool moments);
template Integrals integrate<kernel_mode>(const DraftGeometry& poly, bool moments);
template void prepare_finish<kernel_mode>(DraftGeometry& poly, bool partition_proof);
template void finish<kernel_mode>(DraftGeometry& poly, bool partition_proof, bool positive_construction);
template void validate_connected_boundary<kernel_mode>(std::size_t vertex_count, const std::vector<Face>& faces);
template std::shared_ptr<StoredGeometry> import<kernel_mode>(GeometryPoint origin,
    std::span<const DynamicPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits,
    bool authenticated_planarity, std::span<const PointProvenanceView> identities,
    PointSnapshotBuilder<kernel_mode>* point_builder);
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
template void GeometryKernel<kernel_mode>::visit_local_coordinates(
    void*, void (*)(void*, std::size_t, const PointCoordinates<kernel_mode>&)) const;
template std::array<DynamicGeometryValue, 3> GeometryKernel<kernel_mode>::axis_spans() const;
template DynamicSupport GeometryKernel<kernel_mode>::support(GeometryPoint normal, GeometryPoint anchor) const;
} // namespace ordinary_kernel
} // namespace SimpleFluid::PLIC::details
