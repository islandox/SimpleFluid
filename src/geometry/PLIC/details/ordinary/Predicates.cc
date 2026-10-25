/** @file Predicates.cc
 * @brief Fixed-width compensation and explicit discarded-error bounds.
 */
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Ordinary geometry predicates require strict floating-point compilation."
#endif
#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#endif
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <utility>

namespace SimpleFluid::PLIC::details::ordinary_predicates
{
namespace
{
using Limits = std::numeric_limits<long double>;
[[noreturn]] void unresolved(const char* message)
{
    ++statistics.rejections;
    throw GeometryError(GeometryErrorCode::UnresolvedArithmetic, message);
}
void charge(std::size_t count = 1)
{
    if (active_work)
    {
        if (count > active_work->operation_limit - std::min(active_work->operations, active_work->operation_limit))
        {
            ++statistics.rejections;
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary predicate scalar-work cap exhausted.");
        }
        active_work->operations += count;
    }
    statistics.floating_operations += count;
}
void environment()
{
    bool supported = Limits::is_iec559 && Limits::radix == 2 && Limits::has_denorm == std::denorm_present &&
                     std::fegetround() == FE_TONEAREST;
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__SSE2__)
    const unsigned csr = _mm_getcsr();
    supported = supported && (csr & ((1u << 15) | (1u << 6) | (3u << 13))) == 0 && (csr & (0x3fu << 7)) == (0x3fu << 7);
    if constexpr (Limits::digits == 64)
    {
        unsigned short control;
        __asm__ __volatile__("fnstcw %0" : "=m"(control));
        supported = supported && (control & 0x3fu) == 0x3fu && (control & (3u << 8)) == (3u << 8) &&
                    (control & (3u << 10)) == 0;
    }
#endif
    // Ordinary does not require the x86-only wider-interval backend. Check
    // gradual underflow in the arithmetic format actually used by this kernel.
    if (!supported)
    {
        ++statistics.rejections;
        throw GeometryError(GeometryErrorCode::InvalidInput,
            "Ordinary predicate requires a supported IEEE floating-point environment.");
    }
    volatile long double smallest_normal = Limits::min();
    volatile long double half = .5L;
    charge(2);
    const auto subnormal = smallest_normal * half;
    supported = supported && subnormal > 0 && subnormal + subnormal == smallest_normal;
    if (!supported)
    {
        ++statistics.rejections;
        throw GeometryError(GeometryErrorCode::InvalidInput,
            "Ordinary predicate requires round-to-nearest IEEE arithmetic with gradual underflow.");
    }
}
class ScopedPredicateEnvironment
{
public:
    ScopedPredicateEnvironment()
    {
        if (std::fegetenv(&saved_) != 0)
            throw GeometryError(
                GeometryErrorCode::InvalidInput, "Ordinary predicate cannot capture its FP environment.");
        if (std::fesetround(FE_TONEAREST) != 0)
        {
            (void) std::fesetenv(&saved_);
            throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary predicate cannot select round-to-nearest.");
        }
    }
    ~ScopedPredicateEnvironment() { (void) std::fesetenv(&saved_); }
    ScopedPredicateEnvironment(const ScopedPredicateEnvironment&) = delete;
    ScopedPredicateEnvironment& operator=(const ScopedPredicateEnvironment&) = delete;

private:
    // The standard floating-point environment belongs to the calling thread.
    // Saving it also preserves caller exception flags and control modes. Only
    // rounding is changed; environment() still rejects unsupported trap/FTZ/
    // precision modes before executing arithmetic. Strict compilation with
    // frounding-math and no contraction/reassociation keeps the error-free
    // transforms within this nearest-rounded scope; caller code is unaffected.
    std::fenv_t saved_;
};
void finite_input(long double x)
{
    if (!std::isfinite(x))
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary predicate received a nonfinite coordinate.");
}
long double finite(long double x)
{
    if (!std::isfinite(x))
        unresolved("Ordinary predicate arithmetic exceeds the finite range.");
    return x;
}
long double add(long double a, long double b)
{
    charge();
    return finite(a + b);
}
long double mul(long double a, long double b)
{
    charge();
    return finite(a * b);
}
long double scale(long double a, int exponent)
{
    charge();
    return finite(std::scalbn(a, exponent));
}
long double next_up(long double a)
{
    charge();
    return finite(std::nextafter(a, Limits::infinity()));
}
long double bound_add(long double a, long double b)
{
    return a == 0 ? b : b == 0 ? a : next_up(add(a, b));
}
long double bound_mul(long double a, long double b)
{
    return a == 0 || b == 0 ? 0 : next_up(mul(a, b));
}
long double gap(long double a)
{
    charge(2);
    const auto upper = finite(std::nextafter(a, Limits::infinity()));
    const auto lower = finite(std::nextafter(a, -Limits::infinity()));
    return std::max(add(upper, -a), add(a, -lower));
}
std::pair<long double, long double> two_sum(long double a, long double b)
{
    // Knuth TwoSum is an error-free transform with round-to-nearest and
    // gradual underflow, provided its intermediate sums do not overflow.
    const auto s = add(a, b), z = add(s, -a);
    return {s, add(add(a, -add(s, -z)), add(b, -z))};
}
struct Twofold
{
    long double high = 0, low = 0, error = 0;
    void append(long double x)
    {
        const auto [s, e] = two_sum(high, x);
        const auto [t, discarded] = two_sum(low, e);
        const auto [h, l] = two_sum(s, t);
        high = h;
        low = l;
        error = bound_add(error, std::abs(discarded));
    }
};
Twofold negate(Twofold a)
{
    return {-a.high, -a.low, a.error};
}
Twofold sum(Twofold a, const Twofold& b)
{
    a.append(b.high);
    a.append(b.low);
    a.error = bound_add(a.error, b.error);
    return a;
}
long double magnitude(const Twofold& a)
{
    return bound_add(std::abs(a.high), std::abs(a.low));
}
void product(Twofold& result, long double a, long double b)
{
    if (a == 0 || b == 0)
        return;
    const auto p = mul(a, b);
    charge();
    const auto r = finite(std::fma(a, b, -p));
    result.append(p);
    result.append(r);
    // FMA recovers the product residual exactly except possibly at underflow.
    // Each input has lattice exponent >= frexp_exponent - digits. If their
    // product lattice reaches below denorm_min, one denorm_min safely bounds
    // the only possible residual rounding loss (including a rounded zero).
    int ea, eb;
    charge(2);
    std::frexp(a, &ea);
    std::frexp(b, &eb);
    if (ea + eb - 2 * Limits::digits < Limits::min_exponent - Limits::digits)
        result.error = bound_add(result.error, Limits::denorm_min());
}
Twofold multiply(const Twofold& a, const Twofold& b)
{
    Twofold result;
    product(result, a.high, b.high);
    product(result, a.high, b.low);
    product(result, a.low, b.high);
    product(result, a.low, b.low);
    result.error =
        bound_add(result.error, bound_add(bound_add(bound_mul(magnitude(a), b.error), bound_mul(magnitude(b), a.error)),
                                    bound_mul(a.error, b.error)));
    return result;
}
Twofold scaled(Twofold a, int exponent)
{
    if (exponent == std::numeric_limits<int>::min())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary predicate scaling exponent is invalid.");
    Twofold result;
    for (const auto component : {a.high, a.low})
    {
        if (component == 0)
            continue;
        const auto shifted = scale(component, exponent);
        // Reverse scaling is exact unless scaling discarded significand bits.
        // Check it only when it cannot overflow; the original was finite.
        charge();
        const auto restored = std::scalbn(shifted, -exponent);
        if (restored != component)
            result.error = bound_add(result.error, gap(shifted));
        result.append(shifted);
    }
    if (a.error != 0)
        result.error = bound_add(result.error, next_up(scale(a.error, exponent)));
    return result;
}
Result finish(Twofold a, int exponent)
{
    a = scaled(a, exponent);
    const auto [value, residual] = two_sum(a.high, a.low);
    Result result{value, bound_add(a.error, std::abs(residual)), Sign::Unresolved};
    if (value > result.absolute_error)
        result.sign = Sign::Positive;
    else if (value < -result.absolute_error)
        result.sign = Sign::Negative;
    else if (value == 0 && result.absolute_error == 0)
        result.sign = Sign::Zero;
    return result;
}
using CompensatedPoint = std::array<Twofold, 3>;
template<std::size_t N> int coordinate_exponent(const std::array<Point, N>& points)
{
    long double extent = 0;
    for (const auto& p : points)
        for (const auto x : p)
        {
            finite_input(x);
            extent = std::max(extent, std::abs(x));
        }
    int exponent = 0;
    charge();
    std::frexp(extent, &exponent);
    return exponent;
}
template<std::size_t N> std::array<CompensatedPoint, N> normalized(const std::array<Point, N>& points, int exponent)
{
    std::array<CompensatedPoint, N> result;
    for (std::size_t i = 0; i < N; ++i)
        for (unsigned a = 0; a < 3; ++a)
            result[i][a] = scaled({points[i][a]}, -exponent);
    return result;
}
template<std::size_t N> int normalize_edges(std::array<CompensatedPoint, N>& edges)
{
    long double extent = 0;
    for (const auto& edge : edges)
        for (const auto& x : edge)
            extent = std::max(extent, magnitude(x));
    int exponent = 0;
    charge();
    std::frexp(extent, &exponent);
    for (auto& edge : edges)
        for (auto& x : edge)
            x = scaled(x, -exponent);
    return exponent;
}
} // namespace

void ResultAccumulator::add(const Result& term)
{
    const ScopedPredicateEnvironment scope;
    environment();
    finite_input(term.value);
    if (!std::isfinite(term.absolute_error) || term.absolute_error < 0)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary predicate contribution has an invalid error.");
    auto total = sum(Twofold{high_, low_, error_}, Twofold{term.value, 0, term.absolute_error});
    high_ = total.high;
    low_ = total.low;
    error_ = total.error;
}
Result ResultAccumulator::result(int binary_scale) const
{
    const ScopedPredicateEnvironment scope;
    environment();
    return finish({high_, low_, error_}, binary_scale);
}

Result orientation(const std::array<Point, 4>& points)
{
    const ScopedPredicateEnvironment scope;
    environment();
    const auto exponent = coordinate_exponent(points);
    const auto p = normalized(points, exponent);
    for (std::size_t i = 0; i < points.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (points[i] == points[j])
                return {}; // Identical represented input points have zero determinant.
    // A fixed, coordinate-based minimax criterion chooses the shortest worst
    // edge set once. It is independent of determinant sign and vertex numbering.
    std::size_t anchor = 0;
    long double best = Limits::infinity();
    for (std::size_t i = 0; i < 4; ++i)
    {
        long double extent = 0;
        for (std::size_t j = 0; j < 4; ++j)
            for (unsigned a = 0; a < 3; ++a)
                extent = std::max(extent, std::abs(add(p[j][a].high, -p[i][a].high)));
        if (extent < best || (extent == best && points[i] < points[anchor]))
        {
            anchor = i;
            best = extent;
        }
    }
    std::array<std::size_t, 4> order{anchor};
    for (std::size_t i = 0, next = 1; i < 4; ++i)
        if (i != anchor)
            order[next++] = i;
    std::sort(order.begin() + 1, order.end(), [&](auto a, auto b) { return points[a] < points[b]; });
    unsigned inversions = 0;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = i + 1; j < 4; ++j)
            inversions += order[i] > order[j];
    // Canonicalize the remaining edges too: every permutation now evaluates
    // identical products/sums, with only the final requested orientation sign.
    std::array<CompensatedPoint, 3> edges;
    for (unsigned row = 0; row < 3; ++row)
        for (unsigned a = 0; a < 3; ++a)
            edges[row][a] = sum(p[order[row + 1]][a], negate(p[anchor][a]));
    const auto edge_exponent = normalize_edges(edges);
    Twofold determinant;
    constexpr std::array<std::array<unsigned, 3>, 6> terms{
        {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {1, 0, 2}, {2, 1, 0}}};
    for (unsigned t = 0; t < terms.size(); ++t)
    {
        const auto& term = terms[t];
        auto value = multiply(multiply(edges[0][term[0]], edges[1][term[1]]), edges[2][term[2]]);
        determinant = sum(determinant, t < 3 ? value : negate(value));
    }
    if (inversions % 2)
        determinant = negate(determinant);
    return finish(determinant, 3 * (exponent + edge_exponent));
}
Result projected_orientation(const Point& a, const Point& b, const Point& c, unsigned drop)
{
    if (drop >= 3)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary predicate has an invalid projection axis.");
    const ScopedPredicateEnvironment scope;
    environment();
    const std::array<Point, 3> points{a, b, c};
    const auto exponent = coordinate_exponent(points);
    const auto p = normalized(points, exponent);
    std::array<CompensatedPoint, 2> edges;
    for (unsigned i = 0; i < 2; ++i)
        for (unsigned axis = 0; axis < 3; ++axis)
            edges[i][axis] = sum(p[i + 1][axis], negate(p[0][axis]));
    const auto edge_exponent = normalize_edges(edges);
    const auto j = (drop + 1) % 3, k = (drop + 2) % 3;
    const auto determinant = sum(multiply(edges[0][j], edges[1][k]), negate(multiply(edges[0][k], edges[1][j])));
    return finish(determinant, 2 * (exponent + edge_exponent));
}
Result plane_distance(const Point& point, const Point& normal, const Point& anchor, long double offset)
{
    const ScopedPredicateEnvironment scope;
    environment();
    finite_input(offset);
    const std::array<Point, 2> points{point, anchor};
    const auto exponent = coordinate_exponent(points);
    const auto p = normalized(points, exponent);
    Twofold result;
    for (unsigned a = 0; a < 3; ++a)
    {
        finite_input(normal[a]);
        result = sum(result, multiply(sum(p[0][a], negate(p[1][a])), Twofold{normal[a]}));
    }
    result = sum(scaled(result, exponent), Twofold{-offset});
    return finish(result, 0);
}
Result plane_distance(const FramedPoint& point, const FramedPoint& anchor, const Point& normal, long double offset)
{
    const ScopedPredicateEnvironment scope;
    environment();
    finite_input(offset);
    const auto difference_exponent = [](int first, int second)
    {
        const auto difference = static_cast<long long>(first) - static_cast<long long>(second);
        // scaled() also forms the negated exponent for its reverse check.
        if (difference <= std::numeric_limits<int>::min() || difference > std::numeric_limits<int>::max())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary predicate frame exponent is invalid.");
        return static_cast<int>(difference);
    };
    std::array<Point, 2> origins;
    for (unsigned a = 0; a < 3; ++a)
    {
        finite_input(point.local[a]);
        finite_input(anchor.local[a]);
        finite_input(normal[a]);
        origins[0][a] = point.origin[a];
        origins[1][a] = anchor.origin[a];
    }
    // Subtract normalized original origins before combining with local terms.
    // This retains cancellation without either a rounded global point or an
    // overflowing difference of opposite binary64 origins on narrow platforms.
    const auto origin_exponent = coordinate_exponent(origins);
    const auto normalized_origins = normalized(origins, origin_exponent);
    const auto local_scale = difference_exponent(point.exponent, anchor.exponent);
    const auto origin_scale = difference_exponent(origin_exponent, anchor.exponent);
    Twofold result;
    for (unsigned a = 0; a < 3; ++a)
    {
        const auto origin_difference =
            scaled(sum(normalized_origins[0][a], negate(normalized_origins[1][a])), origin_scale);
        const auto local_difference = sum(scaled(Twofold{point.local[a]}, local_scale), Twofold{-anchor.local[a]});
        const auto relative = sum(origin_difference, local_difference);
        result = sum(result, multiply(relative, Twofold{normal[a]}));
    }
    return finish(sum(result, Twofold{-offset}), 0);
}
int require_sign(const Result& result)
{
    if (result.sign == Sign::Unresolved)
        unresolved("Ordinary predicate sign remains unresolved after fixed-width compensation.");
    return static_cast<int>(result.sign);
}
} // namespace SimpleFluid::PLIC::details::ordinary_predicates
