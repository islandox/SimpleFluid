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
thread_local BoundedGeometryStatistics statistics;
[[noreturn]] void reject(const char* reason)
{
    ++statistics.rejections;
    throw std::invalid_argument(std::string("Bounded FP geometry: ") + reason);
}
struct Work
{
    BoundedGeometryLimits limits;
    std::size_t operations = 0, bytes = 0, pairs = 0, operation_limit = 0;
    explicit Work(BoundedGeometryLimits requested) : limits(requested)
    {
        operation_limit = limits.maximum_operations;
        if (!operation_limit)
        {
            // Admission visits O(F*V^2) face turns; an intersection performs
            // at most F complementary clips, each O(F*V). This independent
            // checked guard bounds their scalar interval work. It is not a
            // reinterpretation or increase of any public sweep/trajectory cap.
            const auto checked_add = [](std::size_t a, std::size_t b)
            {
                if (b > std::numeric_limits<std::size_t>::max() - a)
                    reject("derived scalar-work guard overflow.");
                return a + b;
            };
            const auto checked_multiply = [](std::size_t a, std::size_t b)
            {
                if (b && a > std::numeric_limits<std::size_t>::max() / b)
                    reject("derived scalar-work guard overflow.");
                return a * b;
            };
            const auto v = limits.maximum_vertices, f = limits.maximum_faces;
            operation_limit = checked_add(1024,
                checked_multiply(256, checked_multiply(v, checked_multiply(f, checked_add(checked_add(v, f), 1)))));
        }
    }
    void operation()
    {
        ++statistics.interval_operations;
        if (++operations > operation_limit)
            reject("interval scalar operation cap exhausted.");
    }
    void charge(std::size_t count, std::size_t size)
    {
        if (size && count > (limits.byte_budget - bytes) / size)
            reject("construction storage cap exhausted.");
        bytes += count * size;
    }
    void pair()
    {
        if (++pairs > limits.maximum_pair_tests)
            reject("pair-test cap exhausted.");
    }
};
thread_local Work* active_work = nullptr;
struct Operation
{
    Work work;
    Work* previous;
    explicit Operation(BoundedGeometryLimits limits) : work{limits}, previous(active_work)
    {
        if (!bounded_geometry_environment())
            reject("unsupported floating-point environment; no exact fallback.");
        if (!limits.maximum_vertices || !limits.maximum_faces || !limits.maximum_pair_tests || !limits.byte_budget)
            reject("zero resource cap.");
        if (!previous)
        {
            active_work = &work;
            ++statistics.wider_routes;
        }
    }
    ~Operation() { active_work = previous; }
};
void operation()
{
    ++statistics.floating_operations;
}
void interval_operation()
{
    if (active_work)
        active_work->operation();
    else
        ++statistics.interval_operations;
}
void finite(long double value)
{
    if (!std::isfinite(value) || (value != 0 && !std::isnormal(value)))
        reject("arithmetic range unresolved (overflow or subnormal wider-FP result).");
}
constexpr long double infinity = std::numeric_limits<long double>::infinity();
long double directed_add(long double a, long double b, bool up)
{
    operation();
    const long double sum = a + b;
    finite(sum);
    const long double z = sum - a;
    const long double residual = (a - (sum - z)) + (b - z);
    finite(residual);
    if ((up && residual > 0) || (!up && residual < 0))
    {
        const auto result = std::nextafter(sum, up ? infinity : -infinity);
        finite(result);
        return result;
    }
    return sum;
}
long double directed_multiply(long double a, long double b, bool up)
{
    operation();
    const long double product = a * b;
    finite(product);
    if (a != 0 && b != 0 && product == 0)
        reject("nonzero product below wider-FP range.");
    // In the bottom significand-width of the normal range, an FMA residual
    // can itself underflow. Outward rounding there must not infer exactness.
    if (product != 0 && std::abs(product) < std::scalbn(std::numeric_limits<long double>::min(),
                                                std::numeric_limits<long double>::digits))
        return std::nextafter(product, up ? infinity : -infinity);
    const long double residual = std::fma(a, b, -product);
    finite(residual);
    if ((up && residual > 0) || (!up && residual < 0))
    {
        const auto result = std::nextafter(product, up ? infinity : -infinity);
        finite(result);
        return result;
    }
    return product;
}
long double directed_divide(long double a, long double b, bool up)
{
    operation();
    if (b == 0)
        reject("uncertain or zero divisor.");
    const long double quotient = a / b;
    finite(quotient);
    if (a != 0 && quotient == 0)
        reject("nonzero quotient below wider-FP range.");
    if (a != 0 &&
        std::abs(a) < std::scalbn(std::numeric_limits<long double>::min(), std::numeric_limits<long double>::digits))
        return std::nextafter(quotient, up ? infinity : -infinity);
    const long double residual = std::fma(-quotient, b, a);
    finite(residual);
    const bool above = (residual > 0) == (b > 0);
    if (residual != 0 && ((up && above) || (!up && !above)))
    {
        const auto result = std::nextafter(quotient, up ? infinity : -infinity);
        finite(result);
        return result;
    }
    return quotient;
}
struct Interval
{
    long double lo = 0, hi = 0, estimate = 0;
    explicit Interval(long double value = 0) : lo(value), hi(value), estimate(value) { finite(value); }
    Interval(long double lower, long double upper) : lo(lower), hi(upper), estimate(std::midpoint(lower, upper))
    {
        finite(lo);
        finite(hi);
        if (lo > hi)
            reject("invalid enclosure.");
    }
    bool zero() const { return lo == 0 && hi == 0; }
};
Interval operator-(Interval a)
{
    return {-a.hi, -a.lo};
}
Interval operator+(Interval a, Interval b)
{
    interval_operation();
    return {directed_add(a.lo, b.lo, false), directed_add(a.hi, b.hi, true)};
}
Interval operator-(Interval a, Interval b)
{
    return a + -b;
}
Interval operator*(Interval a, Interval b)
{
    interval_operation();
    const std::array low{directed_multiply(a.lo, b.lo, false), directed_multiply(a.lo, b.hi, false),
        directed_multiply(a.hi, b.lo, false), directed_multiply(a.hi, b.hi, false)};
    const std::array high{directed_multiply(a.lo, b.lo, true), directed_multiply(a.lo, b.hi, true),
        directed_multiply(a.hi, b.lo, true), directed_multiply(a.hi, b.hi, true)};
    return {*std::min_element(low.begin(), low.end()), *std::max_element(high.begin(), high.end())};
}
Interval operator/(Interval a, Interval b)
{
    interval_operation();
    if (b.lo <= 0 && b.hi >= 0)
        reject("division by an enclosure containing zero.");
    const std::array low{directed_divide(a.lo, b.lo, false), directed_divide(a.lo, b.hi, false),
        directed_divide(a.hi, b.lo, false), directed_divide(a.hi, b.hi, false)};
    const std::array high{directed_divide(a.lo, b.lo, true), directed_divide(a.lo, b.hi, true),
        directed_divide(a.hi, b.lo, true), directed_divide(a.hi, b.hi, true)};
    return {*std::min_element(low.begin(), low.end()), *std::max_element(high.begin(), high.end())};
}
Interval scaled(Interval a, int exponent)
{
    interval_operation();
    const auto lo = std::scalbn(a.lo, exponent), hi = std::scalbn(a.hi, exponent);
    finite(lo);
    finite(hi);
    if ((lo == 0 && a.lo != 0) || (hi == 0 && a.hi != 0))
        reject("nonzero enclosure below wider-FP range.");
    Interval result(lo, hi);
    result.estimate = std::scalbn(a.estimate, exponent);
    return result; // A power-of-two scaling is exact in the admitted normal range.
}
Interval enclosure(BoundedGeometryValue a)
{
    if (!(a.absolute_error >= 0))
        reject("invalid input error.");
    return Interval(a.value) + Interval(-a.absolute_error, a.absolute_error);
}
BoundedGeometryValue bounded(Interval a)
{
    const auto middle =
        std::isfinite(a.estimate) && a.estimate >= a.lo && a.estimate <= a.hi ? a.estimate : std::midpoint(a.lo, a.hi);
    const auto error = std::max(directed_add(middle, -a.lo, true), directed_add(a.hi, -middle, true));
    return {middle, error};
}
using Point = std::array<Interval, 3>;
Point point(const BoundedPoint& a)
{
    Point result;
    for (std::size_t axis = 0; axis < 3; ++axis)
        result[axis] = enclosure({a.value[axis], a.absolute_error[axis]});
    return result;
}
Point point(GeometryPoint a)
{
    return {Interval(a[0]), Interval(a[1]), Interval(a[2])};
}
BoundedPoint bounded(const Point& a)
{
    BoundedPoint result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto value = bounded(a[axis]);
        result.value[axis] = value.value;
        result.absolute_error[axis] = value.absolute_error;
    }
    return result;
}
Point operator+(const Point& a, const Point& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Point operator-(const Point& a, const Point& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Point operator*(const Point& a, Interval b)
{
    return {a[0] * b, a[1] * b, a[2] * b};
}
Point scaled(const Point& a, int e)
{
    return {scaled(a[0], e), scaled(a[1], e), scaled(a[2], e)};
}
Interval dot(const Point& a, const Point& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point cross(const Point& a, const Point& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
bool same(const Point& a, const Point& b)
{
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (a[axis].lo != b[axis].lo || a[axis].hi != b[axis].hi)
            return false;
    return true;
}
bool nonzero(const Point& a)
{
    return std::any_of(a.begin(), a.end(), [](auto x) { return x.lo > 0 || x.hi < 0; });
}
int sign(Interval a)
{
    if (a.lo > 0)
        return 1;
    if (a.hi < 0)
        return -1;
    if (a.zero())
        return 0;
    reject("support/topology sign unresolved from original coordinate enclosures.");
}
struct Frame
{
    GeometryPoint origin{};
    int exponent = 0;
};
Point transform(const Point& p, const Frame& from, const Frame& to)
{
    if (from.origin == to.origin && from.exponent == to.exponent)
        return p;
    return scaled(scaled(p, from.exponent) + (point(from.origin) - point(to.origin)), -to.exponent);
}
struct Plane
{
    Frame frame;
    Point normal, anchor;
    Interval offset;
    std::optional<GeometryHalfspace> input;
    std::vector<BoundedPlaneAuthority> authorities;
};
using PlanePtr = std::shared_ptr<const Plane>;
struct Vertex
{
    Point coordinates;
    std::vector<PlanePtr> incident;
    std::vector<BoundedPlaneAuthority> authorities;
};
using Face = std::vector<std::size_t>;
struct Split;
struct Ancestry
{
    std::shared_ptr<const Ancestry> parent;
    std::shared_ptr<const Split> split;
    bool outside = false;
};
struct Split
{
    std::shared_ptr<const Ancestry> parent;
};
struct Surface
{
    PlanePtr plane;
    int orientation = 1;
};
int matching_input(const GeometryHalfspace&, const GeometryHalfspace&);
Interval distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane)
{
    ++statistics.support_distances;
    if (std::find(vertex.incident.begin(), vertex.incident.end(), plane) != vertex.incident.end())
        return Interval{};
    for (const auto& key : plane->authorities)
        if (std::find(vertex.authorities.begin(), vertex.authorities.end(), key) != vertex.authorities.end())
            return Interval{};
    if (plane->input)
        for (const auto& original : vertex.incident)
            if (original->input && matching_input(*original->input, *plane->input))
                return Interval{}; // The identical original expression, not a rounded plane fit.
    return dot(plane->normal, transform(vertex.coordinates, frame, plane->frame) - plane->anchor) - plane->offset;
}
Interval square_root(Interval x)
{
    if (x.lo < 0)
        reject("area square root is unresolved.");
    const auto lo = std::sqrt(x.lo), hi = std::sqrt(x.hi);
    finite(lo);
    finite(hi);
    if (x.hi == 0)
        return Interval{};
    if (lo == 0)
        reject("positive area below wider-FP range.");
    return {std::nextafter(lo, -infinity), std::nextafter(hi, infinity)};
}
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
namespace
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
} // namespace
void BoundedConvexGeometry::certify_partition(
    std::span<const BoundedConvexGeometryPtr> pieces, double relative_budget) const
{
    Operation scope(impl_->limits);
    ++statistics.partition_checks;
    if (!std::isfinite(relative_budget) || relative_budget < 0)
        reject("invalid partition physical budget.");
    std::vector<std::shared_ptr<const Ancestry>> leaves;
    Interval total;
    for (const auto& piece : pieces)
    {
        if (!piece)
            reject("null partition piece.");
        leaves.push_back(piece->impl_->ancestry);
        total = total + piece->impl_->volume;
    }
    const bool structural = structural_cover(impl_->ancestry, leaves);
    if (structural)
        ++statistics.structural_partition_reuses;
    else
    {
        for (const auto& piece : pieces)
            for (const auto& surface : impl_->surfaces)
                for (const auto& vertex : piece->impl_->vertices)
                {
                    active_work->pair();
                    const auto d = distance(vertex, piece->impl_->frame, surface.plane) * Interval(surface.orientation);
                    if (sign(d) > 0)
                        reject("partition piece escapes its parent.");
                }
        for (std::size_t i = 0; i < pieces.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
            {
                active_work->pair();
                if (pieces[i]->intersect(*pieces[j]))
                    reject("partition pieces have overlapping interiors.");
            }
    }
    const auto residual = total - impl_->volume;
    const auto allowance = Interval(relative_budget) * Interval(impl_->volume.lo);
    if (std::max(std::abs(residual.lo), std::abs(residual.hi)) > allowance.lo)
        reject("partition closure exceeds its physical budget.");
}
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
BoundedPoint bounded_backtrace_constrained(const BoundedPoint& source, GeometryPoint velocity, double timestep,
    std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits)
{
    Operation scope(limits);
    if (!std::isfinite(timestep) || timestep < 0 || walls.size() > 256)
        reject("invalid or excessive constrained trajectory inputs.");
    const auto input = point(source);
    auto displacement = point(velocity) * Interval(timestep);
    const bool zero_motion = std::all_of(displacement.begin(), displacement.end(), [](auto x) { return x.zero(); });
    if (walls.empty())
    {
        if (zero_motion)
            return source;
        return bounded(input - displacement);
    }
    std::vector<Point> basis;
    std::vector<Interval> squared;
    std::vector<BoundedPlaneAuthority> retained;
    std::set<BoundedPlaneAuthority> independent_supports;
    for (const auto& wall : walls)
    {
        active_work->pair();
        if (!wall.geometry || wall.face >= wall.geometry->impl_->surfaces.size())
            reject("invalid wall support constraint.");
        const auto& plane = wall.geometry->impl_->surfaces[wall.face].plane;
        bool authenticated = false;
        for (const auto& key : plane->authorities)
            if (std::find(source.supports().begin(), source.supports().end(), key) != source.supports().end())
            {
                authenticated = true;
                retained.push_back(key);
            }
        if (!authenticated)
        {
            const auto local = scaled(input - point(plane->frame.origin), -plane->frame.exponent);
            if (!dot(plane->normal, local - plane->anchor).zero())
                reject("wall constraint lacks original point/support incidence.");
        }
        // Every catalog entry is authenticated above, including duplicates.
        // A collision-free original support identity proves equal planes even
        // when independently scaled normals have different numerical bounds.
        const bool repeated_support = std::any_of(plane->authorities.begin(), plane->authorities.end(),
            [&](const auto& key) { return independent_supports.contains(key); });
        independent_supports.insert(plane->authorities.begin(), plane->authorities.end());
        if (zero_motion || repeated_support || basis.size() == 3)
            continue; // The proven rank-three tangent space already is {0}.
        auto normal = plane->normal;
        for (std::size_t j = 0; j < basis.size(); ++j)
            normal = normal - basis[j] * (dot(normal, basis[j]) / squared[j]);
        const auto norm = dot(normal, normal);
        if (norm.zero())
            continue;
        if (!(norm.lo > 0) || basis.size() == 3)
            reject("wall constraint rank/conditioning unresolved.");
        basis.push_back(normal);
        squared.push_back(norm);
    }
    if (zero_motion)
        return source; // All catalog incidences were validated; no projection is needed.
    if (basis.size() == 3)
    {
        // A proven rank-three tangent space is exactly {0}; retaining the
        // original point is the structural construction, not a rounded snap.
        return source; // Preserve original construction identity as well as its coordinates.
    }
    auto projected = displacement;
    for (std::size_t j = 0; j < basis.size(); ++j)
        projected = projected - basis[j] * (dot(displacement, basis[j]) / squared[j]);
    // Verify the enclosure of every original wall residual before attaching
    // its incidence. The downstream geometry/CFL budget consumes the full
    // propagated coordinate error; this does not introduce a wall tolerance.
    for (const auto& wall : walls)
    {
        const auto& normal = wall.geometry->impl_->surfaces[wall.face].plane->normal;
        const auto residual = dot(normal, projected);
        if (residual.lo > 0 || residual.hi < 0)
            reject("projected trajectory does not enclose original wall tangency.");
    }
    auto result = bounded(input - projected);
    std::sort(retained.begin(), retained.end());
    retained.erase(std::unique(retained.begin(), retained.end()), retained.end());
    BoundedGeometryAccess::seal(result, std::move(retained));
    return result;
}

namespace
{
struct NumericalWriter
{
    std::vector<char> bytes;
    std::size_t limit;
    explicit NumericalWriter(std::size_t budget) : limit(budget) {}
    void word(std::uint64_t x)
    {
        if (bytes.size() > limit || limit - bytes.size() < 8)
            reject("numerical packet exceeds storage cap.");
        for (unsigned byte = 0; byte < 8; ++byte)
            bytes.push_back(static_cast<char>((x >> (8 * byte)) & 255));
    }
    void real(long double x)
    {
        finite(x);
        word(std::signbit(x) && x != 0 ? 1 : 0);
        int exponent = 0;
        auto fraction = std::frexp(std::abs(x), &exponent);
        word(static_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
        constexpr unsigned chunks = (std::numeric_limits<long double>::digits + 31) / 32;
        for (unsigned i = 0; i < chunks; ++i)
        {
            fraction = std::scalbn(fraction, 32);
            const auto chunk = static_cast<std::uint64_t>(fraction);
            word(chunk);
            fraction -= static_cast<long double>(chunk);
        }
        if (fraction != 0)
            reject("wider-FP significand cannot be serialized losslessly.");
    }
    void interval(Interval x)
    {
        real(x.lo);
        real(x.hi);
    }
    void coordinates(const Point& p)
    {
        for (const auto x : p)
            interval(x);
    }
    void frame(const Frame& f)
    {
        for (const auto x : f.origin)
            real(x);
        word(static_cast<std::uint64_t>(static_cast<std::int64_t>(f.exponent)));
    }
    void authorities(const std::vector<BoundedPlaneAuthority>& keys)
    {
        word(keys.size());
        for (const auto& key : keys)
        {
            word(key.key.size());
            for (const auto x : key.key)
                word(x);
        }
    }
};
struct NumericalReader
{
    std::span<const char> bytes;
    std::size_t cursor = 0;
    explicit NumericalReader(std::span<const char> input) : bytes(input) {}
    std::uint64_t word()
    {
        if (cursor > bytes.size() || bytes.size() - cursor < 8)
            reject("truncated numerical geometry packet.");
        std::uint64_t x = 0;
        for (unsigned byte = 0; byte < 8; ++byte)
            x |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[cursor++])) << (8 * byte);
        return x;
    }
    std::size_t count(std::size_t maximum, std::size_t element)
    {
        const auto n = word();
        if (n > maximum || (element && n > (bytes.size() - cursor) / element))
            reject("numerical packet count exceeds remaining bytes/resource cap.");
        active_work->charge(static_cast<std::size_t>(n), element);
        return static_cast<std::size_t>(n);
    }
    long double real()
    {
        const auto negative = word();
        const auto exponent_word = word();
        const auto exponent = std::bit_cast<std::int64_t>(exponent_word);
        if (negative > 1 || exponent < std::numeric_limits<long double>::min_exponent - 1 ||
            exponent > std::numeric_limits<long double>::max_exponent)
            reject("invalid numerical packet scalar exponent/sign.");
        constexpr unsigned chunks = (std::numeric_limits<long double>::digits + 31) / 32;
        long double significand = 0;
        for (unsigned i = 0; i < chunks; ++i)
        {
            const auto chunk = word();
            if (chunk > std::numeric_limits<std::uint32_t>::max())
                reject("invalid numerical packet significand.");
            significand = std::scalbn(significand, 32) + static_cast<long double>(chunk);
        }
        const auto value = std::scalbn(significand, static_cast<int>(exponent) - 32 * chunks);
        finite(value);
        if (significand != 0 && value == 0)
            reject("numerical packet loses nonzero coordinates.");
        return negative ? -value : value;
    }
    Interval interval()
    {
        const auto lo = real(), hi = real();
        return {lo, hi};
    }
    Point coordinates() { return {interval(), interval(), interval()}; }
    Frame frame()
    {
        Frame f;
        for (auto& x : f.origin)
        {
            const auto original = real();
            x = static_cast<double>(original);
            if (!std::isfinite(x) || static_cast<long double>(x) != original)
                reject("source frame anchor is not its original binary64 value.");
        }
        const auto exponent = std::bit_cast<std::int64_t>(word());
        if (exponent < -32768 || exponent > 32768)
            reject("invalid local scaling exponent.");
        f.exponent = static_cast<int>(exponent);
        return f;
    }
    std::vector<BoundedPlaneAuthority> authorities(
        const std::set<BoundedPlaneAuthority>& registry, BoundedGeometryLimits limits)
    {
        std::vector<BoundedPlaneAuthority> result;
        const auto n = count(limits.maximum_faces, sizeof(std::uint64_t));
        for (std::size_t i = 0; i < n; ++i)
        {
            BoundedPlaneAuthority key;
            const auto words = count(limits.byte_budget / sizeof(std::uint64_t), sizeof(std::uint64_t));
            if (!words)
                reject("empty numerical support authority.");
            for (std::size_t j = 0; j < words; ++j)
                key.key.push_back(word());
            if (!registry.contains(key))
                reject("foreign or absent original source support authority.");
            result.push_back(std::move(key));
        }
        std::sort(result.begin(), result.end());
        if (std::adjacent_find(result.begin(), result.end()) != result.end())
            reject("duplicate support authority in numerical packet.");
        return result;
    }
};
} // namespace
std::vector<char> BoundedConvexGeometry::encode(
    const BoundedGeometryPacketContext& context, std::size_t byte_budget) const
{
    Operation scope(impl_->limits);
    NumericalWriter writer(std::min(byte_budget, impl_->limits.byte_budget));
    writer.word(0x5346424f554e4431ULL);
    writer.word(1);
    writer.word(std::numeric_limits<long double>::digits);
    writer.word(std::numeric_limits<long double>::max_exponent);
    for (const auto pin : context.publication)
        writer.word(pin);
    writer.frame(impl_->frame);
    std::map<PlanePtr, std::size_t> plane_ids;
    std::vector<PlanePtr> planes;
    for (const auto& vertex : impl_->vertices)
        for (const auto& plane : vertex.incident)
            if (!plane_ids.contains(plane))
            {
                plane_ids.emplace(plane, planes.size());
                planes.push_back(plane);
            }
    writer.word(planes.size());
    for (const auto& plane : planes)
    {
        writer.frame(plane->frame);
        writer.coordinates(plane->normal);
        writer.coordinates(plane->anchor);
        writer.interval(plane->offset);
        writer.word(plane->input ? 1 : 0);
        if (plane->input)
        {
            for (const auto x : plane->input->normal)
                writer.real(x);
            for (const auto x : plane->input->anchor)
                writer.real(x);
            writer.real(plane->input->offset);
        }
        auto keys = plane->authorities;
        std::sort(keys.begin(), keys.end());
        writer.authorities(keys);
    }
    writer.word(impl_->vertices.size());
    for (const auto& vertex : impl_->vertices)
    {
        writer.coordinates(vertex.coordinates);
        writer.word(vertex.incident.size());
        for (const auto& plane : vertex.incident)
            writer.word(plane_ids.at(plane));
        auto keys = vertex.authorities;
        std::sort(keys.begin(), keys.end());
        writer.authorities(keys);
    }
    writer.word(impl_->faces.size());
    for (std::size_t f = 0; f < impl_->faces.size(); ++f)
    {
        const auto& face = impl_->faces[f];
        writer.word(face.size());
        for (const auto index : face)
            writer.word(index);
        writer.word(plane_ids.at(impl_->surfaces[f].plane));
        writer.word(impl_->surfaces[f].orientation > 0 ? 1 : 0);
    }
    active_work->charge(writer.bytes.size(), sizeof(char));
    return std::move(writer.bytes);
}
BoundedConvexGeometryPtr BoundedConvexGeometry::decode(std::span<const char> packet,
    const BoundedGeometryPacketContext& expected, std::span<const BoundedConvexGeometryPtr> authenticated_sources,
    BoundedGeometryLimits limits)
{
    Operation scope(limits);
    if (packet.size() > limits.byte_budget || authenticated_sources.empty())
        reject("numerical decode requires capped payload and authenticated source registry.");
    std::set<BoundedPlaneAuthority> registry;
    std::map<BoundedPlaneAuthority, PlanePtr> registry_planes;
    for (const auto& source : authenticated_sources)
    {
        if (!source)
            reject("null authenticated source registry entry.");
        for (const auto& vertex : source->impl_->vertices)
            for (const auto& plane : vertex.incident)
                for (const auto& key : plane->authorities)
                {
                    if (registry.insert(key).second)
                    {
                        active_work->charge(2, sizeof(BoundedPlaneAuthority) + 4 * sizeof(void*));
                        active_work->charge(2 * key.key.size(), sizeof(std::uint64_t));
                        registry_planes.emplace(key, plane);
                    }
                }
    }
    NumericalReader reader(packet);
    if (reader.word() != 0x5346424f554e4431ULL || reader.word() != 1 ||
        reader.word() != std::numeric_limits<long double>::digits ||
        reader.word() != std::numeric_limits<long double>::max_exponent)
        reject("unknown or incompatible numerical geometry packet.");
    for (const auto pin : expected.publication)
        if (reader.word() != pin)
            reject("stale or foreign numerical geometry publication/trial pins.");
    auto result = std::make_shared<Impl>();
    result->frame = reader.frame();
    result->limits = limits;
    std::vector<PlanePtr> planes;
    const auto plane_count = reader.count(limits.maximum_faces + limits.maximum_vertices, sizeof(PlanePtr));
    for (std::size_t i = 0; i < plane_count; ++i)
    {
        auto plane = std::make_shared<Plane>();
        plane->frame = reader.frame();
        plane->normal = reader.coordinates();
        plane->anchor = reader.coordinates();
        plane->offset = reader.interval();
        if (!nonzero(plane->normal))
            reject("numerical packet plane normal is unresolved.");
        const auto has_input = reader.word();
        if (has_input > 1)
            reject("invalid numerical plane provenance tag.");
        if (has_input)
        {
            GeometryHalfspace h;
            for (auto& x : h.normal)
            {
                const auto input = reader.real();
                x = static_cast<double>(input);
                if (static_cast<long double>(x) != input)
                    reject("plane input provenance is not original binary64.");
            }
            for (auto& x : h.anchor)
            {
                const auto input = reader.real();
                x = static_cast<double>(input);
                if (static_cast<long double>(x) != input)
                    reject("plane input anchor is not original binary64.");
            }
            const auto offset = reader.real();
            h.offset = static_cast<double>(offset);
            if (static_cast<long double>(h.offset) != offset || !nonzero(point(h.normal)))
                reject("invalid original plane input.");
            const auto expected_normal = point(h.normal);
            const auto expected_anchor = scaled(point(h.anchor) - point(plane->frame.origin), -plane->frame.exponent);
            const auto expected_offset = scaled(Interval(h.offset), -plane->frame.exponent);
            if (!same(plane->normal, expected_normal) || !same(plane->anchor, expected_anchor) ||
                plane->offset.lo != expected_offset.lo || plane->offset.hi != expected_offset.hi)
                reject("plane expression disagrees with its original input provenance.");
            plane->input = h;
        }
        plane->authorities = reader.authorities(registry, limits);
        // Known opaque identity is only an index. Its supplied numerical
        // expression must remain compatible with the pinned original support.
        for (const auto& key : plane->authorities)
        {
            const auto& original = registry_planes.at(key);
            const auto reference_anchor = transform(original->anchor, original->frame, plane->frame);
            const auto offset_residual = dot(plane->normal, reference_anchor - plane->anchor) - plane->offset;
            if (offset_residual.lo > 0 || offset_residual.hi < 0)
                reject("numerical source support expression differs from its registry.");
            const auto parallel = cross(plane->normal, original->normal);
            for (const auto component : parallel)
                if (component.lo > 0 || component.hi < 0)
                    reject("numerical source normal differs from its registry.");
        }
        planes.push_back(std::move(plane));
    }
    const auto vertices = reader.count(limits.maximum_vertices, sizeof(Vertex));
    if (vertices < 4)
        reject("numerical packet has too few vertices.");
    for (std::size_t i = 0; i < vertices; ++i)
    {
        Vertex vertex;
        vertex.coordinates = reader.coordinates();
        const auto count = reader.count(plane_count, sizeof(std::uint64_t));
        std::set<std::size_t> distinct;
        for (std::size_t j = 0; j < count; ++j)
        {
            const auto index = reader.word();
            if (index >= planes.size() || !distinct.insert(static_cast<std::size_t>(index)).second)
                reject("invalid or repeated numerical vertex/plane incidence.");
            vertex.incident.push_back(planes[index]);
        }
        vertex.authorities = reader.authorities(registry, limits);
        for (const auto& plane : vertex.incident)
        {
            const auto raw =
                dot(plane->normal, transform(vertex.coordinates, result->frame, plane->frame) - plane->anchor) -
                plane->offset;
            if (raw.lo > 0 || raw.hi < 0)
                reject("numerical vertex misses its claimed original construction support.");
        }
        for (const auto& key : vertex.authorities)
        {
            const auto& original = registry_planes.at(key);
            const auto raw = dot(original->normal,
                                 transform(vertex.coordinates, result->frame, original->frame) - original->anchor) -
                             original->offset;
            if (raw.lo > 0 || raw.hi < 0)
                reject("numerical point claims a foreign source support incidence.");
        }
        result->vertices.push_back(std::move(vertex));
    }
    const auto faces = reader.count(limits.maximum_faces, sizeof(Face));
    if (faces < 4)
        reject("numerical packet has too few faces.");
    std::map<std::pair<std::size_t, std::size_t>, std::pair<unsigned, int>> edges;
    for (std::size_t f = 0; f < faces; ++f)
    {
        Face face;
        const auto n = reader.count(vertices, sizeof(std::uint64_t));
        if (n < 3)
            reject("numerical packet face is degenerate.");
        std::set<std::size_t> distinct;
        for (std::size_t j = 0; j < n; ++j)
        {
            const auto index = reader.word();
            if (index >= vertices || !distinct.insert(static_cast<std::size_t>(index)).second)
                reject("invalid numerical face loop.");
            face.push_back(static_cast<std::size_t>(index));
        }
        const auto plane = reader.word(), outward = reader.word();
        if (plane >= planes.size() || outward > 1)
            reject("invalid numerical face support/orientation.");
        result->surfaces.push_back({planes[plane], outward ? 1 : -1});
        for (std::size_t j = 0; j < face.size(); ++j)
        {
            if (!incident(result->vertices[face[j]], planes[plane]))
                reject("numerical face lacks original vertex/support incidence.");
            const auto a = face[j], b = face[(j + 1) % face.size()];
            auto& edge = edges[std::minmax(a, b)];
            ++edge.first;
            edge.second += a < b ? 1 : -1;
        }
        result->faces.push_back(std::move(face));
    }
    if (reader.cursor != packet.size())
        reject("trailing numerical geometry packet bytes.");
    for (const auto& [edge, count] : edges)
        if (count.first != 2 || count.second != 0)
            reject("numerical packet shell is not a closed oriented manifold.");
    validate_connected_boundary(result->vertices.size(), result->faces);
    for (std::size_t f = 0; f < result->faces.size(); ++f)
    {
        const auto& surface = result->surfaces[f];
        for (const auto& vertex : result->vertices)
            if (sign(distance(vertex, result->frame, surface.plane) * Interval(surface.orientation)) > 0)
                reject("numerical packet violates convex containment.");
        const auto& face = result->faces[f];
        Point normal;
        for (std::size_t j = 1; j + 1 < face.size(); ++j)
            normal = normal + cross(result->vertices[face[j]].coordinates - result->vertices[face[0]].coordinates,
                                  result->vertices[face[j + 1]].coordinates - result->vertices[face[0]].coordinates);
        if (sign(dot(normal, surface.plane->normal) * Interval(surface.orientation)) <= 0)
            reject("numerical packet face winding is not outward.");
        for (std::size_t j = 0; j < face.size(); ++j)
        {
            const auto& a = result->vertices[face[j]].coordinates;
            const auto& b = result->vertices[face[(j + 1) % face.size()]].coordinates;
            for (const auto index : face)
                if (dot(cross(b - a, result->vertices[index].coordinates - a), normal).hi < 0)
                    reject("numerical packet contains a nonconvex face loop.");
        }
    }
    finish(*result);
    return BoundedConvexGeometryPtr(new BoundedConvexGeometry(std::move(result)));
}

} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
const void* bounded_geometry_statistics_domain()
{
    return &statistics;
}
} // namespace SimpleFluid::Meshes

#include "geometry/details/BoundedPlanarFace.inc"
