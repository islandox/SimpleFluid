/** @file ordinary/ConvexGeometry.tcc
 * @brief Ordinary scalar geometry arithmetic and nominal-value conversion.
 */
#include "geometry/PLIC/details/ordinary/Predicates.hh"

namespace SimpleFluid::PLIC::details
{
namespace ordinary_kernel SIMPLEFLUID_LOCAL
{

namespace source_witness
{
void charge()
{
    ArithmeticPolicy<kernel_mode>::scalar_operation();
    operation();
}
long double add(long double a, long double b)
{
    charge();
    return a + b;
}
long double multiply(long double a, long double b)
{
    charge();
    return a * b;
}
long double scale(long double a, int exponent)
{
    charge();
    return std::scalbn(a, exponent);
}
long double upward(long double a)
{
    if (!std::isfinite(a))
        return infinity;
    charge();
    return std::nextafter(a, infinity);
}
long double add_up(long double a, long double b)
{
    if (a == 0)
        return b;
    if (b == 0)
        return a;
    return upward(add(a, b));
}
long double multiply_up(long double a, long double b)
{
    if (a == 0 || b == 0)
        return 0;
    return upward(multiply(a, b));
}
long double scale_up(long double a, int exponent)
{
    return a == 0 ? 0 : upward(scale(a, exponent));
}
// A full adjacent spacing bounds one correctly rounded result, including
// gradual underflow. No relative-error division or cancellation enters a bound.
long double rounding_gap(long double value)
{
    if (!std::isfinite(value))
        return infinity;
    charge();
    const auto next = std::nextafter(value, infinity);
    charge();
    const auto previous = std::nextafter(value, -infinity);
    if (!std::isfinite(next) || !std::isfinite(previous))
        return infinity;
    return std::max(add(next, -value), add(value, -previous));
}
long double sum_gap(long double a, long double b, long double result)
{
    return a == 0 || b == 0 || a == -b ? 0 : rounding_gap(result);
}
long double product_gap(long double a, long double b, long double result)
{
    return a == 0 || b == 0 ? 0 : rounding_gap(result);
}
long double product_error(long double a, long double b, long double ea, long double eb, long double product)
{
    return add_up(add_up(multiply_up(std::abs(a), eb), multiply_up(std::abs(b), ea)),
        add_up(multiply_up(ea, eb), product_gap(a, b, product)));
}
} // namespace source_witness

std::optional<std::array<long double, 3>> source_point_error(const DynamicPoint& relative,
    const DynamicPointProvenance* original, const Frame& frame, const Point& stored, bool authenticated_shell)
{
    using namespace source_witness;
    if (!certification_environment() || (!authenticated_shell && !original) ||
        (original && original->arithmetic_mode != ArithmeticMode::Ordinary))
        return std::nullopt;
    std::array<long double, 3> error;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        long double e;
        if (authenticated_shell)
            e = relative.absolute_error[axis];
        else
        {
            e = original->absolute_error[axis];
            if (!(e >= 0) || !std::isfinite(e))
                return std::nullopt;
            // The private importer formed relative from this original token.
            // Reject a foreign expression instead of inferring it from proximity.
            const auto difference = add(original->value[axis], -static_cast<long double>(frame.origin[axis]));
            if (difference != relative.value[axis])
                return std::nullopt;
            e = add_up(e, sum_gap(original->value[axis], -static_cast<long double>(frame.origin[axis]), difference));
        }
        if (!(e >= 0) || !std::isfinite(e))
            return std::nullopt;
        if (scale(relative.value[axis], -frame.exponent) != stored[axis].value)
            return std::nullopt;
        error[axis] =
            add_up(scale_up(e, -frame.exponent), relative.value[axis] == 0 ? 0 : rounding_gap(stored[axis].value));
        if (!std::isfinite(error[axis]))
            return std::nullopt;
    }
    return error;
}

std::optional<long double> source_normal_error(
    const std::array<Point, 3>& vertices, const std::array<std::array<long double, 3>, 3>& errors, const Point& normal)
{
    using namespace source_witness;
    if (!certification_environment())
        return std::nullopt;
    for (const auto& point_error : errors)
        for (const auto error : point_error)
            if (!(error >= 0) || !std::isfinite(error))
                return std::nullopt;
    std::array<long double, 3> u, v, eu, ev;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto a = vertices[0][axis].value, b = vertices[1][axis].value, c = vertices[2][axis].value;
        u[axis] = add(b, -a);
        v[axis] = add(c, -a);
        eu[axis] = add_up(add_up(errors[1][axis], errors[0][axis]), sum_gap(b, -a, u[axis]));
        ev[axis] = add_up(add_up(errors[2][axis], errors[0][axis]), sum_gap(c, -a, v[axis]));
    }
    long double error = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto j = (axis + 1) % 3, k = (axis + 2) % 3;
        const auto p = multiply(u[j], v[k]), q = multiply(u[k], v[j]);
        if (!std::isfinite(normal[axis].value) || add(p, -q) != normal[axis].value)
            return std::nullopt;
        const auto component_error =
            add_up(add_up(product_error(u[j], v[k], eu[j], ev[k], p), product_error(u[k], v[j], eu[k], ev[j], q)),
                sum_gap(p, -q, normal[axis].value));
        if (!std::isfinite(component_error))
            return std::nullopt;
        error = std::max(error, component_error);
    }
    return error;
}

bool independent_source_normals(const Plane& a, const Plane& b, const Plane& c)
{
    using namespace source_witness;
    if (!certification_environment())
        return false;
    const std::array<const Plane*, 3> planes{&a, &b, &c};
    std::array<std::array<long double, 3>, 3> normals, errors;
    for (std::size_t row = 0; row < 3; ++row)
    {
        const auto& plane = *planes[row];
        if (!plane.normal_error || !std::isfinite(*plane.normal_error) || *plane.normal_error < 0)
            return false;
        long double maximum = 0;
        for (const auto x : plane.normal)
            maximum = std::max(maximum, std::abs(x.value));
        if (!(maximum > 0) || !std::isfinite(maximum))
            return false;
        int exponent;
        std::frexp(maximum, &exponent);
        const auto error = scale_up(*plane.normal_error, -exponent);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            normals[row][axis] = scale(plane.normal[axis].value, -exponent);
            errors[row][axis] = add_up(error, plane.normal[axis].value == 0 ? 0 : rounding_gap(normals[row][axis]));
            if (!std::isfinite(normals[row][axis]) || !std::isfinite(errors[row][axis]))
                return false;
        }
    }
    constexpr std::array<std::array<std::size_t, 3>, 6> permutations{
        {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {1, 0, 2}, {2, 1, 0}}};
    long double determinant = 0, error = 0;
    for (std::size_t term = 0; term < permutations.size(); ++term)
    {
        const auto& permutation = permutations[term];
        std::array<long double, 3> x, e;
        for (std::size_t row = 0; row < 3; ++row)
        {
            x[row] = normals[row][permutation[row]];
            e[row] = errors[row][permutation[row]];
        }
        const auto product = multiply(x[0], x[1]);
        const auto value = multiply(product, x[2]);
        auto term_error =
            add_up(multiply_up(std::abs(x[2]), product_gap(x[0], x[1], product)), product_gap(product, x[2], value));
        // Expand all seven input-error terms directly; subtracting two
        // product magnitudes would erase the very error being bounded.
        for (unsigned mask = 1; mask < 8; ++mask)
        {
            long double expansion = 1;
            for (unsigned row = 0; row < 3; ++row)
                expansion = multiply_up(expansion, mask & (1u << row) ? e[row] : std::abs(x[row]));
            term_error = add_up(term_error, expansion);
        }
        const auto signed_value = term < 3 ? value : -value;
        const auto next = add(determinant, signed_value);
        error = add_up(error, add_up(term_error, sum_gap(determinant, signed_value, next)));
        determinant = next;
    }
    return std::isfinite(determinant) && std::isfinite(error) && std::abs(determinant) > error;
}

template<ArithmeticMode Mode> Scalar scalar_value(DynamicGeometryValue a)
{
    static_assert(Mode == kernel_mode);
    return Scalar(a.value);
}

template<ArithmeticMode Mode> DynamicGeometryValue bounded(Scalar a)
{
    static_assert(Mode == kernel_mode);
    return {a.value, 0, ArithmeticMode::Ordinary};
}

template<ArithmeticMode Mode> Point point(const DynamicPoint& a)
{
    static_assert(Mode == kernel_mode);
    require_mode(a.arithmetic_mode, active_work->limits.arithmetic_mode);
    Point result;
    for (std::size_t axis = 0; axis < 3; ++axis)
        result[axis] = scalar_value<Mode>({a.value[axis], a.absolute_error[axis], a.arithmetic_mode});
    return result;
}

template<ArithmeticMode Mode> Point point(GeometryPoint a)
{
    static_assert(Mode == kernel_mode);
    return {Scalar(a[0]), Scalar(a[1]), Scalar(a[2])};
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

Point operator*(const Point& a, Scalar b)
{
    return {a[0] * b, a[1] * b, a[2] * b};
}

template<ArithmeticMode Mode> Point scaled(const Point& a, int e)
{
    static_assert(Mode == kernel_mode);
    return {scaled<Mode>(a[0], e), scaled<Mode>(a[1], e), scaled<Mode>(a[2], e)};
}

template<ArithmeticMode Mode> Scalar dot(const Point& a, const Point& b)
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
        if (a[axis].value != b[axis].value)
            return false;
    return true;
}

template<ArithmeticMode Mode> bool nonzero(const Point& a)
{
    static_assert(Mode == kernel_mode);
    return std::any_of(a.begin(), a.end(), [](auto x) { return x.value != 0; });
}

template<ArithmeticMode Mode> int sign(Scalar a)
{
    static_assert(Mode == kernel_mode);
    if (a.value > 0)
        return 1;
    if (a.value < 0)
        return -1;
    if (a.zero())
        return 0;
    reject("nonfinite support/topology scalar.");
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
        const auto shift = Scalar(from.origin[axis]) - Scalar(to.origin[axis]);
        const auto lo = scaled<Mode>(scaled<Mode>(Scalar(box[axis].lo), from.exponent) + shift, -to.exponent);
        const auto hi = scaled<Mode>(scaled<Mode>(Scalar(box[axis].hi), from.exponent) + shift, -to.exponent);
        result[axis] = {lo.value, hi.value};
    }
    return result;
}

template<ArithmeticMode Mode> DynamicPoint converted_point(const DynamicPoint& input, ArithmeticMode mode)
{
    static_assert(Mode == kernel_mode);
    if (input.arithmetic_mode == ArithmeticMode::Exact || mode == ArithmeticMode::Exact)
        reject("Exact arithmetic is not a finite geometry mode.");
    if (input.arithmetic_mode == mode)
        return input;
    if (mode != ArithmeticMode::Ordinary)
        reject("ordinary point cannot be upgraded to a bounded certificate.");
    return DynamicPoint(input.value, {}, mode); // Explicit conversion drops provenance and radii.
}

template<ArithmeticMode Mode> Scalar raw_distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane)
{
    static_assert(Mode == kernel_mode);
    const auto coordinates = [](const Point& point)
    { return ordinary_predicates::Point{point[0].value, point[1].value, point[2].value}; };
    // Keep both original frame origins and local coordinates in the predicate.
    // A rounded world-coordinate conversion can erase a thin displacement
    // before the determinant or dot product has an opportunity to compensate it.
    const ordinary_predicates::FramedPoint point{coordinates(vertex.coordinates), frame.origin, frame.exponent};
    const ordinary_predicates::FramedPoint anchor{
        coordinates(plane->anchor), plane->frame.origin, plane->frame.exponent};
    const auto result = ordinary_predicates::plane_distance(point, anchor, coordinates(plane->normal), plane->offset.value);
    (void) ordinary_predicates::require_sign(result);
    return Scalar(result.value);
}

template<ArithmeticMode Mode> Scalar distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane)
{
    static_assert(Mode == kernel_mode);
    ++statistics.support_distances;
    if (incident<Mode>(vertex, plane))
        return Scalar{};
    for (const auto& key : plane->authorities)
        if (std::find(vertex.authorities.begin(), vertex.authorities.end(), key) != vertex.authorities.end())
            return Scalar{};
    if (plane->input)
        for (const auto& original : vertex.incident)
            if (original->input && matching_input<Mode>(*original->input, *plane->input))
                return Scalar{};
    return raw_distance<Mode>(vertex, frame, plane);
}

template<ArithmeticMode Mode> Scalar square_root(Scalar x)
{
    static_assert(Mode == kernel_mode);
    if (x.value < 0)
        reject("negative square root in ordinary geometry.");
    operation();
    return Scalar(std::sqrt(x.value));
}

template<ArithmeticMode Mode> void Accumulator<Mode>::add(Scalar term)
{
    static_assert(Mode == kernel_mode);
    ArithmeticPolicy<Mode>::scalar_operation();
    const auto x = term.value, next = sum + x;
    operation();
    correction += std::abs(sum) >= std::abs(x) ? (sum - next) + x : (x - next) + sum;
    finite(next);
    finite(correction);
    sum = next;
}

template<ArithmeticMode Mode> Scalar Accumulator<Mode>::result() const
{
    static_assert(Mode == kernel_mode);
    return Scalar(sum + correction);
}

template Scalar scalar_value<kernel_mode>(DynamicGeometryValue a);
template DynamicGeometryValue bounded<kernel_mode>(Scalar a);
template Point point<kernel_mode>(const DynamicPoint& a);
template Point point<kernel_mode>(GeometryPoint a);
template DynamicPoint bounded<kernel_mode>(const Point& a);
template Point scaled<kernel_mode>(const Point& a, int e);
template Scalar dot<kernel_mode>(const Point& a, const Point& b);
template Point cross<kernel_mode>(const Point& a, const Point& b);
template bool same<kernel_mode>(const Point& a, const Point& b);
template bool nonzero<kernel_mode>(const Point& a);
template int sign<kernel_mode>(Scalar a);
template Point transform<kernel_mode>(const Point& p, const Frame& from, const Frame& to);
template CoordinateBox transform<kernel_mode>(const CoordinateBox& box, const Frame& from, const Frame& to);
template DynamicPoint converted_point<kernel_mode>(const DynamicPoint& input, ArithmeticMode mode);
template Scalar raw_distance<kernel_mode>(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template Scalar distance<kernel_mode>(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template Scalar square_root<kernel_mode>(Scalar x);
template void Accumulator<kernel_mode>::add(Scalar term);
template Scalar Accumulator<kernel_mode>::result() const;
} // namespace ordinary_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
