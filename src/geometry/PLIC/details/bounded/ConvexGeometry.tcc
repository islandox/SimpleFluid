/** Compiled certification geometry kernels. Included only by the owning implementation unit.
 * Directed interval arithmetic and retained certificates remain confined to this
 * provider; Mode templates retain the compiled implementation boundary.
 */
#include "geometry/PLIC/GeometryError.hh"

namespace SimpleFluid::PLIC::details
{
namespace certification_kernel SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode> long double directed_add(long double a, long double b, bool up)
{
    static_assert(Mode == kernel_mode);
    return ::SimpleFluid::detail::directed_fp::directed_add<ArithmeticPolicy<Mode>>(a, b, up);
}

template<ArithmeticMode Mode> long double directed_multiply(long double a, long double b, bool up)
{
    static_assert(Mode == kernel_mode);
    return ::SimpleFluid::detail::directed_fp::directed_multiply<ArithmeticPolicy<Mode>>(a, b, up);
}

template<ArithmeticMode Mode> long double directed_divide(long double a, long double b, bool up)
{
    static_assert(Mode == kernel_mode);
    return ::SimpleFluid::detail::directed_fp::directed_divide<ArithmeticPolicy<Mode>>(a, b, up);
}

template<ArithmeticMode Mode> Interval enclosure(DynamicGeometryValue a)
{
    static_assert(Mode == kernel_mode);
    if (!(a.absolute_error >= 0))
        reject("invalid input error.");
    return Interval(a.value) + Interval(-a.absolute_error, a.absolute_error);
}

template<ArithmeticMode Mode> DynamicGeometryValue bounded(Interval a)
{
    static_assert(Mode == kernel_mode);
    const auto middle = std::isfinite(a.estimate) && a.estimate >= a.lo() && a.estimate <= a.hi()
                            ? a.estimate
                            : std::midpoint(a.lo(), a.hi());
    const auto error = std::max(directed_add<Mode>(middle, -a.lo(), true), directed_add<Mode>(a.hi(), -middle, true));
    return {middle, error};
}

template<ArithmeticMode Mode> Point point(const DynamicPoint& a)
{
    static_assert(Mode == kernel_mode);
    require_mode(a.arithmetic_mode, active_work->limits.arithmetic_mode);
    Point result;
    for (std::size_t axis = 0; axis < 3; ++axis)
        result[axis] = enclosure<Mode>({a.value[axis], a.absolute_error[axis], a.arithmetic_mode});
    return result;
}

template<ArithmeticMode Mode> Point point(GeometryPoint a)
{
    static_assert(Mode == kernel_mode);
    return {Interval(a[0]), Interval(a[1]), Interval(a[2])};
}

template<ArithmeticMode Mode> DynamicPoint bounded(const Point& a)
{
    static_assert(Mode == kernel_mode);
    DynamicPoint result;
    result.arithmetic_mode = active_work->limits.arithmetic_mode;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto value = bounded<Mode>(a[axis]);
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

template<ArithmeticMode Mode> Point scaled(const Point& a, int e)
{
    static_assert(Mode == kernel_mode);
    return {scaled<Mode>(a[0], e), scaled<Mode>(a[1], e), scaled<Mode>(a[2], e)};
}

template<ArithmeticMode Mode> Interval dot(const Point& a, const Point& b)
{
    static_assert(Mode == kernel_mode);
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<ArithmeticMode Mode> Point cross(const Point& a, const Point& b)
{
    static_assert(Mode == kernel_mode);
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

template<ArithmeticMode Mode> bool same(const Point& a, const Point& b)
{
    static_assert(Mode == kernel_mode);
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (a[axis].lo() != b[axis].lo() || a[axis].hi() != b[axis].hi())
            return false;
    return true;
}

template<ArithmeticMode Mode> bool nonzero(const Point& a)
{
    static_assert(Mode == kernel_mode);
    return std::any_of(a.begin(), a.end(), [](auto x) { return x.lo() > 0 || x.hi() < 0; });
}

template<ArithmeticMode Mode> int sign(Interval a)
{
    static_assert(Mode == kernel_mode);
    if (a.lo() > 0)
        return 1;
    if (a.hi() < 0)
        return -1;
    if (a.zero())
        return 0;
    ++statistics.rejections;
    throw GeometryError(GeometryErrorCode::UnresolvedArithmetic,
        "Bounded FP geometry: support/topology sign unresolved from original coordinate enclosures.");
}

template<ArithmeticMode Mode> Point transform(const Point& p, const Frame& from, const Frame& to)
{
    static_assert(Mode == kernel_mode);
    if (from.origin == to.origin && from.exponent == to.exponent)
        return p;
    return scaled<Mode>(
        scaled<Mode>(p, from.exponent) + (point<Mode>(from.origin) - point<Mode>(to.origin)), -to.exponent);
}

template<ArithmeticMode Mode> CoordinateBox transform(const CoordinateBox& box, const Frame& from, const Frame& to)
{
    static_assert(Mode == kernel_mode);
    CoordinateBox result;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        Point point_box;
        point_box[axis] = Interval(box[axis].lo, box[axis].hi);
        const auto converted = transform<Mode>(point_box, from, to)[axis];
        result[axis] = {converted.lo(), converted.hi()};
    }
    return result;
}

template<ArithmeticMode Mode> Point interval_box(const CoordinateBox& box)
{
    static_assert(Mode == kernel_mode);
    return {Interval(box[0].lo, box[0].hi), Interval(box[1].lo, box[1].hi), Interval(box[2].lo, box[2].hi)};
}

template<ArithmeticMode Mode> DynamicPoint converted_point(const DynamicPoint& input, ArithmeticMode mode)
{
    static_assert(Mode == kernel_mode);
    if (input.arithmetic_mode == ArithmeticMode::Exact || mode == ArithmeticMode::Exact)
        reject("Exact arithmetic is not a finite geometry mode.");
    require_mode(mode, Mode);
    if (input.arithmetic_mode != mode)
        reject("ordinary point cannot be upgraded to a bounded certificate.");
    return input;
}

template<ArithmeticMode Mode> Interval raw_distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane)
{
    static_assert(Mode == kernel_mode);
    // Move the plane anchor into the vertex frame before forming the small
    // displacement. Moving the vertex into a distant plane frame first would
    // lose thin-fragment coordinates in the addition to that distant anchor.
    const auto local_anchor = transform<Mode>(plane->anchor, plane->frame, frame);
    return dot<Mode>(
               plane->normal, scaled<Mode>(vertex.coordinates - local_anchor, frame.exponent - plane->frame.exponent)) -
           plane->offset;
}

template<ArithmeticMode Mode> Interval distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane)
{
    static_assert(Mode == kernel_mode);
    ++statistics.support_distances;
    if (incident<Mode>(vertex, plane))
        return Interval{};
    for (const auto& key : plane->authorities)
        if (std::find(vertex.authorities.begin(), vertex.authorities.end(), key) != vertex.authorities.end())
            return Interval{};
    if (plane->input)
        for (const auto& original : vertex.incident)
            if (original->input && matching_input<Mode>(*original->input, *plane->input))
                return Interval{}; // The identical original expression, not a rounded plane fit.
    auto result = raw_distance<Mode>(vertex, frame, plane);
    if (result.lo() > 0 || result.hi() < 0 || result.zero())
        return result;
    const auto anchor = transform<Mode>(plane->anchor, plane->frame, frame);
    const auto offset = scaled<Mode>(plane->offset, plane->frame.exponent - frame.exponent);
    for (const auto& original : vertex.incident)
    {
        std::size_t axis = 3;
        long double largest = 0;
        for (std::size_t candidate = 0; candidate < 3; ++candidate)
        {
            const auto n = original->normal[candidate];
            const auto magnitude = n.lo() > 0 ? n.lo() : n.hi() < 0 ? -n.hi() : 0;
            if (magnitude > largest)
            {
                axis = candidate;
                largest = magnitude;
            }
        }
        if (axis == 3 || plane->normal[axis].zero())
            continue;
        // The original incident residual is identically zero. Eliminate its
        // pivot coordinate algebraically before interval evaluation, retaining
        // the dependency that independent coordinate bounds cannot express.
        // Coefficients remain enclosures; no rounded plane is made authoritative.
        const auto ratio = plane->normal[axis] / original->normal[axis];
        const auto original_anchor = transform<Mode>(original->anchor, original->frame, frame);
        auto reduced = ratio * (scaled<Mode>(original->offset, original->frame.exponent - frame.exponent) +
                                   dot<Mode>(original->normal, original_anchor - anchor)) -
                       offset;
        for (std::size_t other = 0; other < 3; ++other)
            if (other != axis)
                reduced = reduced + (plane->normal[other] - ratio * original->normal[other]) *
                                        (vertex.coordinates[other] - anchor[other]);
        reduced = scaled<Mode>(reduced, frame.exponent - plane->frame.exponent);
        const auto lo = std::max(result.lo(), reduced.lo()), hi = std::min(result.hi(), reduced.hi());
        if (lo > hi)
            reject("retained incident equation gives an inconsistent support residual.");
        result = Interval(lo, hi);
        if (result.lo() > 0 || result.hi() < 0 || result.zero())
            return result;
    }
    return result;
}

template<ArithmeticMode Mode> void tighten_incident_coordinates(Vertex& vertex, const Frame& frame)
{
    static_assert(Mode == kernel_mode);
    // Each retained incidence is an equation of the original construction.
    // Intersect its interval solution with the existing coordinate enclosure;
    // this propagates shared constraints without snapping a rounded point or
    // discarding source uncertainty. Three fixed sweeps bound the work, and
    // every arithmetic operation uses the enclosing transaction's scalar cap.
    for (unsigned sweep = 0; sweep < 3; ++sweep)
    {
        bool changed = false;
        for (const auto& plane : vertex.incident)
        {
            std::size_t axis = 3;
            long double largest = 0;
            for (std::size_t candidate = 0; candidate < 3; ++candidate)
            {
                const auto n = plane->normal[candidate];
                const auto magnitude = n.lo() > 0 ? n.lo() : n.hi() < 0 ? -n.hi() : 0;
                if (magnitude > largest)
                {
                    axis = candidate;
                    largest = magnitude;
                }
            }
            if (axis == 3 || vertex.coordinates[axis].lo() == vertex.coordinates[axis].hi())
                continue;
            const auto anchor = transform<Mode>(plane->anchor, plane->frame, frame);
            auto rhs = scaled<Mode>(plane->offset, plane->frame.exponent - frame.exponent);
            for (std::size_t other = 0; other < 3; ++other)
                if (other != axis && !plane->normal[other].zero())
                    rhs = rhs - plane->normal[other] * (vertex.coordinates[other] - anchor[other]);
            const auto solved = anchor[axis] + rhs / plane->normal[axis];
            auto& coordinate = vertex.coordinates[axis];
            const auto lo = std::max(coordinate.lo(), solved.lo()), hi = std::min(coordinate.hi(), solved.hi());
            if (lo > hi)
                reject("retained incident equations have inconsistent coordinate enclosures.");
            if (lo != coordinate.lo() || hi != coordinate.hi())
            {
                coordinate = Interval(lo, hi);
                changed = true;
            }
        }
        if (!changed)
            break;
    }
}

template<ArithmeticMode Mode> Interval square_root(Interval x)
{
    static_assert(Mode == kernel_mode);
    if (x.lo() < 0)
        reject("area square root is unresolved.");
    const auto lo = std::sqrt(x.lo()), hi = std::sqrt(x.hi());
    finite(lo);
    finite(hi);
    if (x.hi() == 0)
        return Interval{};
    if (lo == 0)
        reject("positive area below wider-FP range.");
    return {std::nextafter(lo, -infinity), std::nextafter(hi, infinity)};
}

template<ArithmeticMode Mode> void Accumulator<Mode>::add(Interval term)
{
    static_assert(Mode == kernel_mode);
    enclosure = enclosure + term;
    const auto x = term.estimate, next = sum + x;
    operation();
    correction += std::abs(sum) >= std::abs(x) ? (sum - next) + x : (x - next) + sum;
    finite(next);
    finite(correction);
    sum = next;
}

template<ArithmeticMode Mode> Interval Accumulator<Mode>::result() const
{
    static_assert(Mode == kernel_mode);
    auto result = enclosure;
    result.estimate = sum + correction;
    return result;
}

template long double directed_add<kernel_mode>(long double a, long double b, bool up);
template long double directed_multiply<kernel_mode>(long double a, long double b, bool up);
template long double directed_divide<kernel_mode>(long double a, long double b, bool up);
template Interval enclosure<kernel_mode>(DynamicGeometryValue a);
template DynamicGeometryValue bounded<kernel_mode>(Interval a);
template Point point<kernel_mode>(const DynamicPoint& a);
template Point point<kernel_mode>(GeometryPoint a);
template DynamicPoint bounded<kernel_mode>(const Point& a);
template Point scaled<kernel_mode>(const Point& a, int e);
template Interval dot<kernel_mode>(const Point& a, const Point& b);
template Point cross<kernel_mode>(const Point& a, const Point& b);
template bool same<kernel_mode>(const Point& a, const Point& b);
template bool nonzero<kernel_mode>(const Point& a);
template int sign<kernel_mode>(Interval a);
template Point transform<kernel_mode>(const Point& p, const Frame& from, const Frame& to);
template CoordinateBox transform<kernel_mode>(const CoordinateBox& box, const Frame& from, const Frame& to);
template Point interval_box<kernel_mode>(const CoordinateBox& box);
template DynamicPoint converted_point<kernel_mode>(const DynamicPoint& input, ArithmeticMode mode);
template Interval raw_distance<kernel_mode>(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template Interval distance<kernel_mode>(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template void tighten_incident_coordinates<kernel_mode>(Vertex& vertex, const Frame& frame);
template Interval square_root<kernel_mode>(Interval x);
template void Accumulator<kernel_mode>::add(Interval term);
template Interval Accumulator<kernel_mode>::result() const;
} // namespace certification_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
