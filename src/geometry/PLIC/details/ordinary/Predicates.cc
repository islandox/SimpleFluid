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
#include <vector>

namespace SimpleFluid::PLIC::details::ordinary_predicates
{
namespace
{
using Limits = std::numeric_limits<long double>;
struct AccountingContext
{
    Work* work;
    GeometryStatistics& stats;

    void charge(std::size_t count = 1) const
    {
        if (work)
        {
            if (count > work->operation_limit - std::min(work->operations, work->operation_limit))
            {
                ++stats.rejections;
                throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary predicate scalar-work cap exhausted.");
            }
            work->operations += count;
        }
        stats.floating_operations += count;
    }
};
[[noreturn]] void unresolved(const AccountingContext& context, const char* message)
{
    ++context.stats.rejections;
    throw GeometryError(GeometryErrorCode::UnresolvedArithmetic, message);
}
void environment(const AccountingContext& context)
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
        ++context.stats.rejections;
        throw GeometryError(GeometryErrorCode::InvalidInput,
            "Ordinary predicate requires a supported IEEE floating-point environment.");
    }
    volatile long double smallest_normal = Limits::min();
    volatile long double half = .5L;
    context.charge(2);
    const auto subnormal = smallest_normal * half;
    supported = supported && subnormal > 0 && subnormal + subnormal == smallest_normal;
    if (!supported)
    {
        ++context.stats.rejections;
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
long double finite(const AccountingContext& context, long double x)
{
    if (!std::isfinite(x))
        unresolved(context, "Ordinary predicate arithmetic exceeds the finite range.");
    return x;
}
long double add(const AccountingContext& context, long double a, long double b)
{
    context.charge();
    return finite(context, a + b);
}
long double mul(const AccountingContext& context, long double a, long double b)
{
    context.charge();
    return finite(context, a * b);
}
long double scale(const AccountingContext& context, long double a, int exponent)
{
    context.charge();
    return finite(context, std::scalbn(a, exponent));
}
long double next_up(const AccountingContext& context, long double a)
{
    context.charge();
    return finite(context, std::nextafter(a, Limits::infinity()));
}
long double bound_add(const AccountingContext& context, long double a, long double b)
{
    return a == 0 ? b : b == 0 ? a : next_up(context, add(context, a, b));
}
long double bound_mul(const AccountingContext& context, long double a, long double b)
{
    return a == 0 || b == 0 ? 0 : next_up(context, mul(context, a, b));
}
long double gap(const AccountingContext& context, long double a)
{
    context.charge(2);
    const auto upper = finite(context, std::nextafter(a, Limits::infinity()));
    const auto lower = finite(context, std::nextafter(a, -Limits::infinity()));
    return std::max(add(context, upper, -a), add(context, a, -lower));
}
std::pair<long double, long double> two_sum(const AccountingContext& context, long double a, long double b)
{
    // Knuth TwoSum is an error-free transform with round-to-nearest and
    // gradual underflow, provided its intermediate sums do not overflow.
    const auto s = add(context, a, b), z = add(context, s, -a);
    return {s, add(context, add(context, a, -add(context, s, -z)), add(context, b, -z))};
}
struct Twofold
{
    long double high = 0, low = 0, error = 0;
    void append(const AccountingContext& context, long double x)
    {
        const auto [s, e] = two_sum(context, high, x);
        const auto [t, discarded] = two_sum(context, low, e);
        const auto [h, l] = two_sum(context, s, t);
        high = h;
        low = l;
        error = bound_add(context, error, std::abs(discarded));
    }
};
Twofold negate(Twofold a)
{
    return {-a.high, -a.low, a.error};
}
Twofold sum(const AccountingContext& context, Twofold a, const Twofold& b)
{
    a.append(context, b.high);
    a.append(context, b.low);
    a.error = bound_add(context, a.error, b.error);
    return a;
}
Twofold add_term(const AccountingContext& context, Twofold total, const Result& term)
{
    finite_input(term.value);
    if (!std::isfinite(term.absolute_error) || term.absolute_error < 0)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary predicate contribution has an invalid error.");
    return sum(context, total, Twofold{term.value, 0, term.absolute_error});
}
long double magnitude(const AccountingContext& context, const Twofold& a)
{
    return bound_add(context, std::abs(a.high), std::abs(a.low));
}
void product(const AccountingContext& context, Twofold& result, long double a, long double b)
{
    if (a == 0 || b == 0)
        return;
    const auto p = mul(context, a, b);
    context.charge();
    const auto r = finite(context, std::fma(a, b, -p));
    result.append(context, p);
    result.append(context, r);
    // FMA recovers the product residual exactly except possibly at underflow.
    // Each input has lattice exponent >= frexp_exponent - digits. If their
    // product lattice reaches below denorm_min, one denorm_min safely bounds
    // the only possible residual rounding loss (including a rounded zero).
    int ea, eb;
    context.charge(2);
    std::frexp(a, &ea);
    std::frexp(b, &eb);
    if (ea + eb - 2 * Limits::digits < Limits::min_exponent - Limits::digits)
        result.error = bound_add(context, result.error, Limits::denorm_min());
}
Twofold multiply(const AccountingContext& context, const Twofold& a, const Twofold& b)
{
    Twofold result;
    product(context, result, a.high, b.high);
    product(context, result, a.high, b.low);
    product(context, result, a.low, b.high);
    product(context, result, a.low, b.low);
    result.error = bound_add(context, result.error,
        bound_add(context,
            bound_add(context, bound_mul(context, magnitude(context, a), b.error),
                bound_mul(context, magnitude(context, b), a.error)),
            bound_mul(context, a.error, b.error)));
    return result;
}
Twofold scaled(const AccountingContext& context, Twofold a, int exponent)
{
    if (exponent == std::numeric_limits<int>::min())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary predicate scaling exponent is invalid.");
    Twofold result;
    for (const auto component : {a.high, a.low})
    {
        if (component == 0)
            continue;
        const auto shifted = scale(context, component, exponent);
        // Reverse scaling is exact unless scaling discarded significand bits.
        // Check it only when it cannot overflow; the original was finite.
        context.charge();
        const auto restored = std::scalbn(shifted, -exponent);
        if (restored != component)
            result.error = bound_add(context, result.error, gap(context, shifted));
        result.append(context, shifted);
    }
    if (a.error != 0)
        result.error = bound_add(context, result.error, next_up(context, scale(context, a.error, exponent)));
    return result;
}
Result finish(const AccountingContext& context, Twofold a, int exponent)
{
    a = scaled(context, a, exponent);
    const auto [value, residual] = two_sum(context, a.high, a.low);
    Result result{value, bound_add(context, a.error, std::abs(residual)), Sign::Unresolved};
    if (value > result.absolute_error)
        result.sign = Sign::Positive;
    else if (value < -result.absolute_error)
        result.sign = Sign::Negative;
    else if (value == 0 && result.absolute_error == 0)
        result.sign = Sign::Zero;
    return result;
}
using CompensatedPoint = std::array<Twofold, 3>;
template<std::size_t N> int coordinate_exponent(const AccountingContext& context, const std::array<Point, N>& points)
{
    long double extent = 0;
    for (const auto& p : points)
        for (const auto x : p)
        {
            finite_input(x);
            extent = std::max(extent, std::abs(x));
        }
    int exponent = 0;
    context.charge();
    std::frexp(extent, &exponent);
    return exponent;
}
template<std::size_t N>
std::array<CompensatedPoint, N> normalized(
    const AccountingContext& context, const std::array<Point, N>& points, int exponent)
{
    std::array<CompensatedPoint, N> result;
    for (std::size_t i = 0; i < N; ++i)
        for (unsigned a = 0; a < 3; ++a)
            result[i][a] = scaled(context, {points[i][a]}, -exponent);
    return result;
}
template<std::size_t N> int normalize_edges(const AccountingContext& context, std::array<CompensatedPoint, N>& edges)
{
    long double extent = 0;
    for (const auto& edge : edges)
        for (const auto& x : edge)
            extent = std::max(extent, magnitude(context, x));
    int exponent = 0;
    context.charge();
    std::frexp(extent, &exponent);
    for (auto& edge : edges)
        for (auto& x : edge)
            x = scaled(context, x, -exponent);
    return exponent;
}

// Grow-expansion with zero elimination (TwoSum) keeps nonoverlapping,
// increasing-magnitude components. Its last component determines the exact
// sign. Every capacity increase is bounded and charged, including copies.
class Expansion
{
public:
    Expansion() = default;
    explicit Expansion(const AccountingContext& context, long double x)
    {
        if (x != 0)
            push(context, x);
    }
    Expansion(const Expansion&) = delete;
    [[nodiscard]] static Expansion clone(const AccountingContext& context, const Expansion& other)
    {
        Expansion result;
        result.reserve(context, other.values_.size());
        result.values_.assign(other.values_.begin(), other.values_.end());
        return result;
    }
    Expansion(Expansion&&) noexcept = default;
    Expansion& operator=(Expansion&&) noexcept = default;
    Expansion& operator=(const Expansion&) = delete;
    [[nodiscard]] bool empty() const { return values_.empty(); }
    [[nodiscard]] long double leading() const { return empty() ? 0 : values_.back(); }
    [[nodiscard]] const std::vector<long double>& values() const { return values_; }
    void append(const AccountingContext& context, long double x)
    {
        if (x == 0)
            return;
        std::size_t count = 0;
        for (const auto component : values_)
        {
            const auto [sum, residual] = two_sum(context, x, component);
            if (residual != 0)
                values_[count++] = residual;
            x = sum;
        }
        values_.resize(count);
        if (x != 0)
            push(context, x);
    }

private:
    void reserve(const AccountingContext& context, std::size_t count)
    {
        constexpr std::size_t capacity = 2048;
        if (count > capacity)
        {
            ++context.stats.rejections;
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Ordinary support expansion capacity exhausted.");
        }
        if (count <= values_.capacity())
            return;
        const auto requested = std::min(capacity, std::max(count, 2 * values_.capacity()));
        if (context.work)
            context.work->charge(requested - values_.capacity(), sizeof(long double));
        values_.reserve(requested);
    }
    void push(const AccountingContext& context, long double x)
    {
        reserve(context, values_.size() + 1);
        values_.push_back(x);
    }
    std::vector<long double> values_;
};
Expansion expansion_sum(const AccountingContext& context, Expansion a, const Expansion& b, int sign = 1)
{
    for (const auto x : b.values())
        a.append(context, sign < 0 ? -x : x);
    return a;
}
Expansion expansion_product(const AccountingContext& context, const Expansion& a, const Expansion& b)
{
    Expansion result;
    for (const auto x : a.values())
        for (const auto y : b.values())
        {
            int ex, ey;
            context.charge(2);
            std::frexp(x, &ex);
            std::frexp(y, &ey);
            // The product's exact binary lattice must fit, including its FMA
            // residual. A rounded-to-zero residual alone proves no contact.
            if (ex + ey - 2 * Limits::digits < Limits::min_exponent - Limits::digits)
                unresolved(context, "Ordinary support expansion product exceeds its exact underflow lattice.");
            const auto product = mul(context, x, y);
            context.charge();
            const auto residual = finite(context, std::fma(x, y, -product));
            result.append(context, product);
            result.append(context, residual);
        }
    return result;
}
Expansion expansion_scaled(const AccountingContext& context, const Expansion& a, int exponent)
{
    if (exponent == std::numeric_limits<int>::min())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary support scaling exponent is invalid.");
    Expansion result;
    for (const auto x : a.values())
    {
        const auto shifted = scale(context, x, exponent);
        context.charge();
        if (std::scalbn(shifted, -exponent) != x)
            unresolved(context, "Ordinary support expansion scaling loses represented input bits.");
        result.append(context, shifted);
    }
    return result;
}
int exponent_difference(int a, int b)
{
    const auto difference = static_cast<long long>(a) - b;
    if (difference <= std::numeric_limits<int>::min() || difference > std::numeric_limits<int>::max())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary support frame exponent is invalid.");
    return static_cast<int>(difference);
}
Expansion support_coefficient(const AccountingContext& context, const FramedPoint& point, const SupportRow& row)
{
    finite_input(row.offset);
    Expansion result(context, -row.offset);
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        finite_input(point.local[axis]);
        finite_input(point.origin[axis]);
        finite_input(row.anchor.local[axis]);
        finite_input(row.anchor.origin[axis]);
        finite_input(row.normal[axis]);
        int origin_exponent = 0;
        context.charge();
        std::frexp(std::max(std::abs(static_cast<long double>(point.origin[axis])),
                       std::abs(static_cast<long double>(row.anchor.origin[axis]))),
            &origin_exponent);
        auto origin = expansion_sum(context,
            expansion_scaled(context, Expansion(context, point.origin[axis]), -origin_exponent),
            expansion_scaled(context, Expansion(context, row.anchor.origin[axis]), -origin_exponent), -1);
        origin = expansion_scaled(context, origin, exponent_difference(origin_exponent, row.anchor.exponent));
        auto local = expansion_sum(context,
            expansion_scaled(context, Expansion(context, point.local[axis]),
                exponent_difference(point.exponent, row.anchor.exponent)),
            Expansion(context, row.anchor.local[axis]), -1);
        result = expansion_sum(context, std::move(result),
            expansion_product(
                context, expansion_sum(context, std::move(origin), local), Expansion(context, row.normal[axis])));
    }
    return result;
}
using ExpansionRow = std::array<Expansion, 4>;
struct NormalizedSupportRow
{
    ExpansionRow coefficients;
    int exponent = 0;
};
NormalizedSupportRow support_row(
    const AccountingContext& context, const std::array<FramedPoint, 4>& source, const SupportRow& row)
{
    if ((row.orientation != 1 && row.orientation != -1) || row.source_incidence > 15 || row.source_face > 4)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary support row has invalid structural metadata.");
    if (row.source_face == 4)
    {
        finite_input(row.offset);
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            finite_input(row.anchor.local[axis]);
            finite_input(row.anchor.origin[axis]);
            finite_input(row.normal[axis]);
        }
    }
    NormalizedSupportRow result;
    long double extent = 0;
    for (unsigned j = 0; j < 4; ++j)
    {
        if (row.source_face < 4)
            result.coefficients[j] = Expansion(context, j == row.source_face ? row.orientation : 0);
        else if ((row.source_incidence & (1u << j)) == 0)
        {
            result.coefficients[j] = support_coefficient(context, source[j], row);
            if (row.orientation < 0)
                result.coefficients[j] = expansion_sum(context, Expansion{}, result.coefficients[j], -1);
        }
        extent = std::max(extent, std::abs(result.coefficients[j].leading()));
    }
    context.charge();
    std::frexp(extent, &result.exponent);
    for (auto& coefficient : result.coefficients)
        coefficient = expansion_scaled(context, coefficient, -result.exponent);
    return result;
}
Expansion support_determinant(
    const AccountingContext& context, const std::array<NormalizedSupportRow, 3>& basis, const ExpansionRow& last)
{
    Expansion result;
    std::array<unsigned, 4> permutation{0, 1, 2, 3};
    do
    {
        unsigned inversions = 0;
        for (unsigned i = 0; i < 4; ++i)
            for (unsigned j = i + 1; j < 4; ++j)
                inversions += permutation[i] > permutation[j];
        auto term =
            expansion_product(context, basis[0].coefficients[permutation[0]], basis[1].coefficients[permutation[1]]);
        term = expansion_product(context, term, basis[2].coefficients[permutation[2]]);
        term = expansion_product(context, term, last[permutation[3]]);
        result = expansion_sum(context, std::move(result), term, inversions % 2 ? -1 : 1);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return result;
}
long double expansion_magnitude(const AccountingContext& context, const Expansion& value, bool tail_only = false)
{
    long double result = 0;
    const auto count = value.values().size() - (tail_only && !value.empty() ? 1 : 0);
    for (std::size_t i = 0; i < count; ++i)
        result = bound_add(context, result, std::abs(value.values()[i]));
    return result;
}
} // namespace

SupportEvaluation support_intersection_distance(
    const std::array<FramedPoint, 4>& source, const std::array<SupportRow, 3>& supports, const SupportRow& query)
{
    const ScopedPredicateEnvironment scope;
    const AccountingContext context{active_work, statistics};
    environment(context);
    for (const auto& point : source)
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            finite_input(point.local[axis]);
            finite_input(point.origin[axis]);
        }
    std::array<NormalizedSupportRow, 3> basis;
    for (unsigned i = 0; i < 3; ++i)
        basis[i] = support_row(context, source, supports[i]);
    ExpansionRow ones{Expansion(context, 1), Expansion(context, 1), Expansion(context, 1), Expansion(context, 1)};
    auto denominator = support_determinant(context, basis, ones);
    if (denominator.empty())
        return {{}, true};
    const auto q = support_row(context, source, query);
    auto numerator = support_determinant(context, basis, q.coefficients);
    if (numerator.empty())
        return {};
    const auto sign = (numerator.leading() > 0) == (denominator.leading() > 0) ? Sign::Positive : Sign::Negative;
    int ne, de;
    context.charge(2);
    std::frexp(numerator.leading(), &ne);
    std::frexp(denominator.leading(), &de);
    numerator = expansion_scaled(context, numerator, -ne);
    denominator = expansion_scaled(context, denominator, -de);
    context.charge();
    const auto candidate = finite(context, numerator.leading() / denominator.leading());
    const auto residual = expansion_sum(context, Expansion::clone(context, numerator),
        expansion_product(context, Expansion(context, candidate), denominator), -1);
    const auto tail = expansion_magnitude(context, denominator, true);
    const auto lower = add(context, std::abs(denominator.leading()), -tail);
    context.charge();
    const auto denominator_lower = std::nextafter(lower, 0.L);
    if (denominator_lower <= 0)
        unresolved(context, "Ordinary support expansion has no usable metric denominator bound.");
    auto error = expansion_magnitude(context, residual);
    if (error != 0)
    {
        context.charge();
        error = next_up(context, finite(context, error / denominator_lower));
    }
    const auto scale_exponent = static_cast<long long>(ne) - de + q.exponent;
    if (scale_exponent <= std::numeric_limits<int>::min() || scale_exponent > std::numeric_limits<int>::max())
        unresolved(context, "Ordinary support quotient exceeds its finite scaling range.");
    const auto exponent = static_cast<int>(scale_exponent);
    const auto value = scale(context, candidate, exponent);
    context.charge();
    if (value == 0 || std::scalbn(value, -exponent) != candidate)
        unresolved(context, "Ordinary support quotient loses represented metric bits.");
    if (error != 0)
        error = next_up(context, scale(context, error, exponent));
    return {{value, error, sign}, false};
}

void ResultAccumulator::add(const Result& term)
{
    const ScopedPredicateEnvironment scope;
    const AccountingContext context{active_work, statistics};
    environment(context);
    auto total = add_term(context, {high_, low_, error_}, term);
    high_ = total.high;
    low_ = total.low;
    error_ = total.error;
}
Result ResultAccumulator::result(int binary_scale) const
{
    const ScopedPredicateEnvironment scope;
    const AccountingContext context{active_work, statistics};
    environment(context);
    return finish(context, {high_, low_, error_}, binary_scale);
}

Result sum_results(std::span<const Result> terms, int binary_scale)
{
    const ScopedPredicateEnvironment scope;
    const AccountingContext context{active_work, statistics};
    environment(context);
    Twofold total;
    for (const auto& term : terms)
        total = add_term(context, total, term);
    return finish(context, total, binary_scale);
}

Result orientation(const std::array<Point, 4>& points)
{
    const ScopedPredicateEnvironment scope;
    const AccountingContext context{active_work, statistics};
    environment(context);
    const auto exponent = coordinate_exponent(context, points);
    const auto p = normalized(context, points, exponent);
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
                extent = std::max(extent, std::abs(add(context, p[j][a].high, -p[i][a].high)));
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
            edges[row][a] = sum(context, p[order[row + 1]][a], negate(p[anchor][a]));
    const auto edge_exponent = normalize_edges(context, edges);
    Twofold determinant;
    constexpr std::array<std::array<unsigned, 3>, 6> terms{
        {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {1, 0, 2}, {2, 1, 0}}};
    for (unsigned t = 0; t < terms.size(); ++t)
    {
        const auto& term = terms[t];
        auto value = multiply(context, multiply(context, edges[0][term[0]], edges[1][term[1]]), edges[2][term[2]]);
        determinant = sum(context, determinant, t < 3 ? value : negate(value));
    }
    if (inversions % 2)
        determinant = negate(determinant);
    return finish(context, determinant, 3 * (exponent + edge_exponent));
}
Result projected_orientation(const Point& a, const Point& b, const Point& c, unsigned drop)
{
    if (drop >= 3)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Ordinary predicate has an invalid projection axis.");
    const ScopedPredicateEnvironment scope;
    const AccountingContext context{active_work, statistics};
    environment(context);
    const std::array<Point, 3> points{a, b, c};
    const auto exponent = coordinate_exponent(context, points);
    const auto p = normalized(context, points, exponent);
    std::array<CompensatedPoint, 2> edges;
    for (unsigned i = 0; i < 2; ++i)
        for (unsigned axis = 0; axis < 3; ++axis)
            edges[i][axis] = sum(context, p[i + 1][axis], negate(p[0][axis]));
    const auto edge_exponent = normalize_edges(context, edges);
    const auto j = (drop + 1) % 3, k = (drop + 2) % 3;
    const auto determinant = sum(context, multiply(context, edges[0][j], edges[1][k]),
        negate(multiply(context, edges[0][k], edges[1][j])));
    return finish(context, determinant, 2 * (exponent + edge_exponent));
}
Result plane_distance(const Point& point, const Point& normal, const Point& anchor, long double offset)
{
    const ScopedPredicateEnvironment scope;
    const AccountingContext context{active_work, statistics};
    environment(context);
    finite_input(offset);
    const std::array<Point, 2> points{point, anchor};
    const auto exponent = coordinate_exponent(context, points);
    const auto p = normalized(context, points, exponent);
    Twofold result;
    for (unsigned a = 0; a < 3; ++a)
    {
        finite_input(normal[a]);
        result = sum(context, result, multiply(context, sum(context, p[0][a], negate(p[1][a])), Twofold{normal[a]}));
    }
    result = sum(context, scaled(context, result, exponent), Twofold{-offset});
    return finish(context, result, 0);
}
Result plane_distance(const FramedPoint& point, const FramedPoint& anchor, const Point& normal, long double offset)
{
    const ScopedPredicateEnvironment scope;
    const AccountingContext context{active_work, statistics};
    environment(context);
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
    const auto origin_exponent = coordinate_exponent(context, origins);
    const auto normalized_origins = normalized(context, origins, origin_exponent);
    const auto local_scale = difference_exponent(point.exponent, anchor.exponent);
    const auto origin_scale = difference_exponent(origin_exponent, anchor.exponent);
    Twofold result;
    for (unsigned a = 0; a < 3; ++a)
    {
        const auto origin_difference =
            scaled(context, sum(context, normalized_origins[0][a], negate(normalized_origins[1][a])), origin_scale);
        const auto local_difference =
            sum(context, scaled(context, Twofold{point.local[a]}, local_scale), Twofold{-anchor.local[a]});
        const auto relative = sum(context, origin_difference, local_difference);
        result = sum(context, result, multiply(context, relative, Twofold{normal[a]}));
    }
    return finish(context, sum(context, result, Twofold{-offset}), 0);
}
int require_sign(const Result& result)
{
    if (result.sign == Sign::Unresolved)
    {
        const AccountingContext context{active_work, statistics};
        unresolved(context, "Ordinary predicate sign remains unresolved after fixed-width compensation.");
    }
    return static_cast<int>(result.sign);
}
} // namespace SimpleFluid::PLIC::details::ordinary_predicates
