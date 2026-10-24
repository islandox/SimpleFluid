/** @file ordinary/BoundedConvexGeometry.tcc
 * @brief Ordinary scalar geometry arithmetic and nominal-value conversion.
 */
namespace SimpleFluid::Meshes::bounded_geometry_detail
{
namespace ordinary_kernel SIMPLEFLUID_LOCAL
{

template<ArithmeticMode Mode> Scalar scalar_value(BoundedGeometryValue a)
{
    static_assert(Mode == kernel_mode);
    return Scalar(a.value);
}

template<ArithmeticMode Mode> BoundedGeometryValue bounded(Scalar a)
{
    static_assert(Mode == kernel_mode);
    return {a.value, 0, ArithmeticMode::Ordinary};
}

template<ArithmeticMode Mode> Point point(const BoundedPoint& a)
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

template<ArithmeticMode Mode> BoundedPoint bounded(const Point& a)
{
    static_assert(Mode == kernel_mode);
    BoundedPoint result;
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

template<ArithmeticMode Mode> BoundedPoint converted_point(const BoundedPoint& input, ArithmeticMode mode)
{
    static_assert(Mode == kernel_mode);
    if (input.arithmetic_mode == ArithmeticMode::Exact || mode == ArithmeticMode::Exact)
        reject("Exact arithmetic is not a finite geometry mode.");
    if (input.arithmetic_mode == mode)
        return input;
    if (mode != ArithmeticMode::Ordinary)
        reject("ordinary point cannot be upgraded to a bounded certificate.");
    return BoundedPoint(input.value, {}, mode); // Explicit conversion drops provenance and radii.
}

template<ArithmeticMode Mode> Scalar raw_distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane)
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

template Scalar scalar_value<kernel_mode>(BoundedGeometryValue a);
template BoundedGeometryValue bounded<kernel_mode>(Scalar a);
template Point point<kernel_mode>(const BoundedPoint& a);
template Point point<kernel_mode>(GeometryPoint a);
template BoundedPoint bounded<kernel_mode>(const Point& a);
template Point scaled<kernel_mode>(const Point& a, int e);
template Scalar dot<kernel_mode>(const Point& a, const Point& b);
template Point cross<kernel_mode>(const Point& a, const Point& b);
template bool same<kernel_mode>(const Point& a, const Point& b);
template bool nonzero<kernel_mode>(const Point& a);
template int sign<kernel_mode>(Scalar a);
template Point transform<kernel_mode>(const Point& p, const Frame& from, const Frame& to);
template CoordinateBox transform<kernel_mode>(const CoordinateBox& box, const Frame& from, const Frame& to);
template BoundedPoint converted_point<kernel_mode>(const BoundedPoint& input, ArithmeticMode mode);
template Scalar raw_distance<kernel_mode>(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template Scalar distance<kernel_mode>(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template Scalar square_root<kernel_mode>(Scalar x);
template void Accumulator<kernel_mode>::add(Scalar term);
template Scalar Accumulator<kernel_mode>::result() const;
} // namespace ordinary_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes::bounded_geometry_detail
