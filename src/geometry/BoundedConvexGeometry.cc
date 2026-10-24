#include "geometry/details/BoundedConvexGeometryInternal.hh"

namespace SimpleFluid::Meshes
{
const std::vector<BoundedPlaneAuthority>& BoundedPoint::supports() const
{
    static const std::vector<BoundedPlaneAuthority> empty;
    if (!provenance_ || provenance_->value != value || provenance_->absolute_error != absolute_error)
        return empty;
    return provenance_->supports;
}
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{
namespace bounded_geometry_detail
{
thread_local BoundedGeometryStatistics statistics;

[[noreturn]] void reject(const char* reason)
{
    ++statistics.rejections;
    throw std::invalid_argument(std::string("Bounded FP geometry: ") + reason);
}

thread_local Work* active_work = nullptr;

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

long double directed_add(long double a, long double b, bool up)
{
    return ::SimpleFluid::detail::directed_fp::directed_add<ScalarPolicy>(a, b, up);
}

long double directed_multiply(long double a, long double b, bool up)
{
    return ::SimpleFluid::detail::directed_fp::directed_multiply<ScalarPolicy>(a, b, up);
}

long double directed_divide(long double a, long double b, bool up)
{
    return ::SimpleFluid::detail::directed_fp::directed_divide<ScalarPolicy>(a, b, up);
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

Point transform(const Point& p, const Frame& from, const Frame& to)
{
    if (from.origin == to.origin && from.exponent == to.exponent)
        return p;
    return scaled(scaled(p, from.exponent) + (point(from.origin) - point(to.origin)), -to.exponent);
}

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
} // namespace bounded_geometry_detail
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
std::size_t BoundedGeometryAccess::storage_bytes(const BoundedPoint& p)
{
    std::size_t bytes = sizeof(BoundedPoint);
    if (p.provenance_)
    {
        bytes += sizeof(BoundedPointProvenance) + p.provenance_->supports.capacity() * sizeof(BoundedPlaneAuthority);
        for (const auto& key : p.provenance_->supports)
            bytes += key.key.capacity() * sizeof(std::uint64_t);
    }
    return bytes;
}

bool BoundedGeometryAccess::same_source(const BoundedPoint& a, const BoundedPoint& b)
{
    return a.provenance_ && a.provenance_ == b.provenance_ && a.provenance_->value == a.value &&
           a.provenance_->absolute_error == a.absolute_error && b.provenance_->value == b.value &&
           b.provenance_->absolute_error == b.absolute_error;
}

void BoundedGeometryAccess::seal(BoundedPoint& point, std::vector<BoundedPlaneAuthority> supports)
{
    std::sort(supports.begin(), supports.end());
    supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
    point.provenance_ = std::make_shared<BoundedPointProvenance>(
        BoundedPointProvenance{point.value, point.absolute_error, std::move(supports)});
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
namespace bounded_geometry_detail
{
Work::Work(BoundedGeometryLimits requested) : limits(requested)
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
        operation_limit = checked_add(
            1024, checked_multiply(256, checked_multiply(v, checked_multiply(f, checked_add(checked_add(v, f), 1)))));
    }
}

void Work::operation()
{
    ++statistics.interval_operations;
    if (++operations > operation_limit)
        reject("interval scalar operation cap exhausted.");
}

void Work::charge(std::size_t count, std::size_t size)
{
    if (size && count > (limits.byte_budget - bytes) / size)
        reject("construction storage cap exhausted.");
    bytes += count * size;
}

void Work::pair()
{
    if (++pairs > limits.maximum_pair_tests)
        reject("pair-test cap exhausted.");
}

Operation::Operation(BoundedGeometryLimits limits) : work{limits}, previous(active_work)
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

void Accumulator::add(Interval term)
{
    enclosure = enclosure + term;
    const auto x = term.estimate, next = sum + x;
    operation();
    correction += std::abs(sum) >= std::abs(x) ? (sum - next) + x : (x - next) + sum;
    finite(next);
    finite(correction);
    sum = next;
}

Interval Accumulator::result() const
{
    auto result = enclosure;
    result.estimate = sum + correction;
    return result;
}
} // namespace bounded_geometry_detail
} // namespace SimpleFluid::Meshes
