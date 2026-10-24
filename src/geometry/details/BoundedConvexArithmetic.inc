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
