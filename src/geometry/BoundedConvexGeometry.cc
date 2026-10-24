/** @file BoundedConvexGeometry.cc
 * @brief Bounded wider-FP construction and structural convex partitions.
 */
#include "geometry/BoundedConvexGeometry.hh"
#include "geometry/BoundedPlanarFace.hh"
#include "geometry/ExactPlanarFace.hh"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace SimpleFluid::Meshes
{
struct BoundedPointProvenance
{
    std::array<long double, 3> value, absolute_error;
    std::vector<BoundedPlaneAuthority> supports;
};
class BoundedGeometryAccess
{
public:
    static std::size_t storage_bytes(const BoundedPoint& p)
    {
        std::size_t bytes = sizeof(BoundedPoint);
        if (p.provenance_)
        {
            bytes +=
                sizeof(BoundedPointProvenance) + p.provenance_->supports.capacity() * sizeof(BoundedPlaneAuthority);
            for (const auto& key : p.provenance_->supports)
                bytes += key.key.capacity() * sizeof(std::uint64_t);
        }
        return bytes;
    }
    static bool same_source(const BoundedPoint& a, const BoundedPoint& b)
    {
        return a.provenance_ && a.provenance_ == b.provenance_ && a.provenance_->value == a.value &&
               a.provenance_->absolute_error == a.absolute_error && b.provenance_->value == b.value &&
               b.provenance_->absolute_error == b.absolute_error;
    }
    static void seal(BoundedPoint& point, std::vector<BoundedPlaneAuthority> supports)
    {
        std::sort(supports.begin(), supports.end());
        supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
        point.provenance_ = std::make_shared<BoundedPointProvenance>(
            BoundedPointProvenance{point.value, point.absolute_error, std::move(supports)});
    }
};
const std::vector<BoundedPlaneAuthority>& BoundedPoint::supports() const
{
    static const std::vector<BoundedPlaneAuthority> empty;
    if (!provenance_ || provenance_->value != value || provenance_->absolute_error != absolute_error)
        return empty;
    return provenance_->supports;
}
namespace
{
#include "geometry/details/BoundedConvexArithmetic.inc"
#include "geometry/details/BoundedConvexPredicates.inc"
struct Accumulator
{
    Interval enclosure;
    long double sum = 0, correction = 0;
    void add(Interval term)
    {
        enclosure = enclosure + term;
        const auto x = term.estimate, next = sum + x;
        operation();
        correction += std::abs(sum) >= std::abs(x) ? (sum - next) + x : (x - next) + sum;
        finite(next);
        finite(correction);
        sum = next;
    }
    Interval result() const
    {
        auto result = enclosure;
        result.estimate = sum + correction;
        return result;
    }
};
struct Integrals
{
    Interval volume;
    Point first;
};
struct MomentCache
{
    std::once_flag prepared;
    std::optional<Integrals> integrals;
};
} // namespace

struct BoundedConvexGeometry::Impl
{
    Frame frame;
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    std::vector<Surface> surfaces;
    std::vector<BoundedPoint> displayed;
    BoundedGeometryLimits limits;
    Interval volume;
    std::shared_ptr<const Ancestry> ancestry = std::make_shared<Ancestry>();
    std::shared_ptr<MomentCache> moment_cache = std::make_shared<MomentCache>();
};
namespace
{
#include "geometry/details/BoundedConvexMeasures.inc"
#include "geometry/details/BoundedConvexConstruction.inc"
} // namespace
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
namespace
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
} // namespace

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
#include "geometry/details/BoundedConvexClipping.inc"
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
#include "geometry/details/BoundedConvexPartitions.inc"
std::size_t BoundedConvexGeometry::storage_bytes() const
{
    std::size_t bytes = sizeof(Impl) + sizeof(MomentCache) + impl_->vertices.capacity() * sizeof(Vertex) +
                        impl_->faces.capacity() * sizeof(Face) + impl_->surfaces.capacity() * sizeof(Surface) +
                        impl_->displayed.capacity() * sizeof(BoundedPoint);
    std::set<PlanePtr> planes;
    for (const auto& vertex : impl_->vertices)
    {
        bytes += vertex.incident.capacity() * sizeof(PlanePtr) +
                 vertex.authorities.capacity() * sizeof(BoundedPlaneAuthority);
        for (const auto& key : vertex.authorities)
            bytes += key.key.capacity() * sizeof(std::uint64_t);
        planes.insert(vertex.incident.begin(), vertex.incident.end());
    }
    for (const auto& face : impl_->faces)
        bytes += face.capacity() * sizeof(std::size_t);
    for (const auto& p : impl_->displayed)
        bytes += bounded_point_storage_bytes(p) - sizeof(BoundedPoint);
    for (const auto& plane : planes)
    {
        bytes += sizeof(Plane) + plane->authorities.capacity() * sizeof(BoundedPlaneAuthority);
        for (const auto& key : plane->authorities)
            bytes += key.key.capacity() * sizeof(std::uint64_t);
    }
    std::set<const Split*> splits;
    for (auto node = impl_->ancestry; node; node = node->parent)
    {
        bytes += sizeof(Ancestry);
        if (node->split && splits.insert(node->split.get()).second)
            bytes += sizeof(Split);
    }
    return bytes;
}
namespace
{
BoundedPoint source_point_fan(std::span<const ConvexCellGeometry* const> sources, std::span<const char> identity)
{
    Operation scope({});
    ++statistics.source_imports;
    if (sources.empty() || identity.size() > active_work->limits.byte_budget)
        reject("source point export requires a capped authenticated source fan.");
    std::size_t faces = 0;
    for (const auto* source : sources)
    {
        if (!source || source->storage_bytes() > active_work->limits.byte_budget ||
            source->num_faces() > active_work->limits.maximum_faces - faces)
            reject("source point authority exceeds bootstrap storage/fan cap.");
        faces += source->num_faces();
    }
    const auto coordinates = numerical_point(identity);
    BoundedPoint result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result.value[axis] = coordinates[axis].value;
        result.absolute_error[axis] = coordinates[axis].absolute_error;
    }
    std::vector<BoundedPlaneAuthority> supports;
    for (const auto* source : sources)
        for (std::size_t f = 0; f < source->num_faces(); ++f)
        {
            const auto face = source->face_geometry(f);
            if (!face->contains_exact_point(identity))
                continue;
            const auto bytes = exact_planar_support(*face);
            BoundedPlaneAuthority authority;
            authority.key.push_back(bytes.size());
            authority.key.resize(1 + (bytes.size() + 7) / 8, 0);
            for (std::size_t i = 0; i < bytes.size(); ++i)
                authority.key[1 + i / 8] |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i]))
                                            << (8 * (i % 8));
            active_work->charge(authority.key.size(), sizeof(std::uint64_t));
            supports.push_back(std::move(authority));
        }
    BoundedGeometryAccess::seal(result, std::move(supports));
    return result;
}
} // namespace
BoundedPoint bounded_source_point(const ConvexCellGeometry& source, std::span<const char> identity)
{
    const std::array<const ConvexCellGeometry*, 1> sources{&source};
    return source_point_fan(sources, identity);
}
BoundedPoint bounded_source_point(std::span<const ConvexGeometryPtr> sources, std::span<const char> identity)
{
    Operation scope({});
    if (sources.size() > active_work->limits.maximum_faces)
        reject("source point fan exceeds bootstrap source-count cap.");
    active_work->charge(sources.size(), sizeof(const ConvexCellGeometry*));
    std::vector<const ConvexCellGeometry*> pointers;
    for (const auto& source : sources)
        pointers.push_back(source.get());
    return source_point_fan(pointers, identity);
}
BoundedPoint bounded_point_displaced(const BoundedPoint& source, const BoundedPoint& displacement)
{
    Operation scope({});
    const auto delta = point(displacement);
    if (std::all_of(delta.begin(), delta.end(), [](auto x) { return x.zero(); }))
        return source;
    return bounded(point(source) + delta);
}
#include "geometry/details/BoundedConvexTrajectories.inc"

#include "geometry/details/BoundedConvexPackets.inc"

} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
const void* bounded_geometry_statistics_domain()
{
    return &statistics;
}
} // namespace SimpleFluid::Meshes

#include "geometry/details/BoundedPlanarFace.inc"
