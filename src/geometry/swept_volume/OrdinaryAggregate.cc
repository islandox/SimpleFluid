#include "geometry/swept_volume/OrdinaryAggregate.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/SupportClassification.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>

namespace SimpleFluid::geometry::swept_volume
{
using namespace ::SimpleFluid::PLIC;
namespace
{
namespace predicates = ::SimpleFluid::PLIC::details::ordinary_predicates;
using V = std::array<long double, 3>;
using B = std::array<long double, 4>;
constexpr long double eps = std::numeric_limits<long double>::epsilon();
[[noreturn]] void fail(GeometryErrorCode code, const char* message)
{
    throw GeometryError(code, message);
}
/** Ordinary quality estimates assume full relative precision. A subnormal
 * result cannot satisfy that model, even when it remains positive. */
long double represented(long double value)
{
    if (!std::isfinite(value) || std::fpclassify(value) == FP_SUBNORMAL)
        fail(GeometryErrorCode::UnresolvedArithmetic, "Ordinary aggregate arithmetic lost normal relative precision.");
    return value;
}
long double sum_values(long double a, long double b)
{
    return represented(a + b);
}
long double product(long double a, long double b)
{
    const auto result = represented(a * b);
    if (!std::isfinite(result) || (a != 0 && b != 0 && result == 0))
        fail(GeometryErrorCode::UnresolvedArithmetic, "Ordinary aggregate product exceeds its represented range.");
    return result;
}
long double quotient(long double numerator, long double denominator)
{
    const auto result = represented(numerator / denominator);
    if (!std::isfinite(result) || (numerator != 0 && result == 0))
        fail(GeometryErrorCode::UnresolvedArithmetic, "Ordinary aggregate quotient exceeds its represented range.");
    return result;
}
long double dot(V a, V b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
class ScopedAggregateEnvironment
{
public:
    ScopedAggregateEnvironment()
    {
        if (std::fegetenv(&saved_) != 0)
            fail(GeometryErrorCode::InvalidInput, "Cannot capture aggregate FP environment.");
        try
        {
            if (std::fesetround(FE_TONEAREST) != 0)
                fail(GeometryErrorCode::InvalidInput, "Cannot select aggregate FP environment.");
            (void) predicates::orientation({V{}, V{1, 0, 0}, V{0, 1, 0}, V{0, 0, 1}});
        }
        catch (...)
        {
            std::fesetenv(&saved_);
            throw;
        }
    }
    ~ScopedAggregateEnvironment() { std::fesetenv(&saved_); }

private:
    std::fenv_t saved_{};
};
struct Sum
{
    long double value = 0, correction = 0, absolute = 0, error = 0;
    void add(long double x, long double input_error = 0)
    {
        represented(x);
        represented(input_error);
        const auto t = sum_values(value, x);
        const auto residual = std::abs(value) >= std::abs(x) ? sum_values(sum_values(value, -t), x)
                                                             : sum_values(sum_values(x, -t), value);
        correction = sum_values(correction, residual);
        value = t;
        absolute = sum_values(absolute, std::abs(x));
        error = sum_values(error, input_error);
    }
    long double get() const { return sum_values(value, correction); }
};
struct AggregateWork
{
    SweepLimits sweep;
    GeometryLimits geometry;
    OrdinaryAggregateStatistics stats;
    std::size_t bytes = 0, operation_limit = 0;
    OrdinaryAggregateContext* context = nullptr;
    std::size_t indirect_calls = 0, indirect_bytes = 0, indirect_reuses = 0;
    void sync() const
    {
        if (context)
        {
            context->bytes = bytes;
            context->pairs = stats.pair_tests;
            context->operations = stats.operations;
        }
    }
    void charge(std::size_t n)
    {
        const auto limit = std::min(sweep.byte_budget, geometry.byte_budget);
        if (n > limit || bytes > limit - n)
        {
            std::ostringstream message;
            message << "Ordinary aggregate byte budget exhausted: indirect_calls=" << indirect_calls
                    << " indirect_bytes=" << indirect_bytes << " indirect_reuses=" << indirect_reuses << '.';
            throw GeometryError(GeometryErrorCode::ResourceLimit, message.str());
        }
        bytes += n;
        stats.storage_bytes = bytes;
        sync();
    }
    void op(std::size_t n = 1)
    {
        if (!operation_limit)
            operation_limit = ::SimpleFluid::PLIC::details::Work(geometry).operation_limit;
        const auto limit = operation_limit;
        if (n > limit || stats.operations > limit - n)
            fail(GeometryErrorCode::ResourceLimit, "Ordinary aggregate work budget exhausted.");
        stats.operations += n;
        sync();
    }
    void pairs(std::size_t count)
    {
        const auto limit = std::min(sweep.maximum_pair_tests, geometry.maximum_pair_tests);
        if (count > limit - stats.pair_tests)
            fail(GeometryErrorCode::ResourceLimit, "Ordinary aggregate pair budget exhausted.");
        stats.pair_tests += count;
        sync();
    }
    void pair() { pairs(1); }
    void witness()
    {
        if (++stats.witnesses > sweep.maximum_atoms)
            fail(GeometryErrorCode::ResourceLimit, "Ordinary aggregate witness cap exhausted.");
    }
};
template<class Function> predicates::Result predicate(Function&& function, AggregateWork& work)
{
    const auto before = geometry_statistics().floating_operations;
    const auto result = function();
    work.op(geometry_statistics().floating_operations - before);
    represented(result.value);
    represented(result.absolute_error);
    return result;
}
// Construction evidence has no relative-error floor: exact represented
// operations retain zero error. Metric quality remains a separate admission.
long double error_sum(long double a, long double b)
{
    if (a == 0)
        return b;
    if (b == 0)
        return a;
    return represented(std::nextafter(sum_values(a, b), std::numeric_limits<long double>::infinity()));
}
long double error_product(long double a, long double b)
{
    if (a == 0 || b == 0)
        return 0;
    return represented(std::nextafter(product(a, b), std::numeric_limits<long double>::infinity()));
}
predicates::Result construction_sum(predicates::Result a, predicates::Result b, AggregateWork& work)
{
    return predicate(
        [&]
        {
            predicates::ResultAccumulator sum;
            sum.add(a);
            sum.add(b);
            return sum.result();
        },
        work);
}
long double product_residual(long double a, long double b, long double value)
{
    if (a == 0 || b == 0)
        return 0;
    int ea = 0, eb = 0;
    std::frexp(a, &ea);
    std::frexp(b, &eb);
    // An underflowed residual cannot establish exactness in this primitive.
    if (ea + eb - 2 * std::numeric_limits<long double>::digits <
        std::numeric_limits<long double>::min_exponent - std::numeric_limits<long double>::digits)
        fail(GeometryErrorCode::UnresolvedArithmetic, "Ordinary aggregate construction residual exceeds its range.");
    return represented(std::fma(a, b, -value));
}
predicates::Result construction_product(predicates::Result a, predicates::Result b, AggregateWork& work)
{
    work.op(16);
    const auto value = product(a.value, b.value);
    const auto error = error_sum(std::abs(product_residual(a.value, b.value, value)),
        error_sum(error_product(std::abs(a.value), b.absolute_error),
            error_sum(error_product(std::abs(b.value), a.absolute_error),
                error_product(a.absolute_error, b.absolute_error))));
    return {value, error, predicates::Sign::Unresolved};
}
predicates::Result construction_difference(predicates::Result a, predicates::Result b, AggregateWork& work)
{
    b.value = -b.value;
    return construction_sum(a, b, work);
}
predicates::Result construction_quotient(predicates::Result a, predicates::Result b, AggregateWork& work)
{
    work.op(20);
    if (!(std::abs(b.value) > b.absolute_error))
        fail(GeometryErrorCode::UnresolvedArithmetic, "Ordinary aggregate crossing denominator is unresolved.");
    const auto value = quotient(a.value, b.value);
    // Validate FMA's residual lattice just as for a product. Unlike the
    // product residual, this residual is a - value*b.
    (void) product_residual(value, b.value, product(value, b.value));
    const auto residual = represented(std::fma(-value, b.value, a.value));
    const auto numerator_error =
        error_sum(std::abs(residual), error_sum(a.absolute_error, error_product(std::abs(value), b.absolute_error)));
    if (numerator_error == 0)
        return {value, 0, predicates::Sign::Unresolved};
    const auto denominator = std::nextafter(std::abs(b.value) - b.absolute_error, 0.L);
    const auto error = represented(
        std::nextafter(quotient(numerator_error, denominator), std::numeric_limits<long double>::infinity()));
    return {value, error, predicates::Sign::Unresolved};
}
/** An immutable defining equation in the original tetrahedron's barycentric
 * frame. The catalog outlives every witness, including complementary copies. */
struct Support
{
    predicates::SupportRow expression;
    const void* identity = nullptr;
};
struct SupportKeyOrder
{
    bool operator()(const std::array<const Support*, 4>& a, const std::array<const Support*, 4>& b) const
    {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), std::less<const Support*>{});
    }
};
using DefiningCache = std::map<std::array<const Support*, 4>, predicates::SupportEvaluation, SupportKeyOrder>;
bool support_expression_less(const Support* a, const Support* b)
{
    const auto key = [](const Support* support)
    {
        const auto& row = support->expression;
        return std::tie(row.source_face, row.normal, row.anchor.origin, row.anchor.exponent, row.anchor.local,
            row.offset, row.source_incidence);
    };
    if (key(a) != key(b))
        return key(a) < key(b);
    // Equal represented expressions can still have distinct issued identities.
    return std::less<const Support*>{}(a, b);
}
struct Construction
{
    std::size_t first = 0, second = 0;
    const Support* support = nullptr;
    int orientation = 1;
};
struct Face
{
    std::vector<std::size_t> vertices;
    int source = -1;
};
struct Poly
{
    std::vector<B> vertices;
    std::vector<B> errors;
    std::vector<std::vector<const Support*>> supports;
    const OrdinaryAggregateTetra* source = nullptr;
    DefiningCache* defining_cache = nullptr;
    // The first four entries are immutable source leaves. Later entries
    // retain the edge/support expression, independent of rounded weights.
    std::vector<Construction> constructions;
    std::vector<Face> faces;
};
void charge_poly_points(const Poly& p, AggregateWork& work)
{
    work.charge(p.vertices.size() * (2 * sizeof(B) + sizeof(Construction)) +
                p.supports.size() * sizeof(std::vector<const Support*>));
    for (const auto& support : p.supports)
        work.charge(support.size() * sizeof(const Support*));
}
void charge_poly_copy(const Poly& p, AggregateWork& work)
{
    charge_poly_points(p, work);
    work.charge(sizeof(Poly) + p.faces.size() * sizeof(Face));
    for (const auto& face : p.faces)
        work.charge(face.vertices.size() * sizeof(std::size_t));
}
constexpr std::array<std::array<unsigned, 3>, 4> tetra_faces{{{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}}};
Poly tetra(const OrdinaryAggregateTetra& source, const std::array<Support, 4>& supports, DefiningCache& cache,
    AggregateWork& work)
{
    work.witness();
    work.charge(sizeof(Poly) + 8 * sizeof(B) + 4 * sizeof(Construction) + 4 * sizeof(std::vector<const Support*>) +
                4 * sizeof(Face) + 12 * sizeof(std::size_t) + 12 * sizeof(const Support*));
    Poly p;
    p.source = &source;
    p.defining_cache = &cache;
    p.vertices.resize(4);
    p.errors.resize(4);
    p.supports.resize(4);
    p.constructions.resize(4);
    for (unsigned i = 0; i < 4; ++i)
    {
        p.vertices[i][i] = 1;
        p.supports[i].reserve(3);
        for (unsigned j = 0; j < 4; ++j)
            if (i != j)
                p.supports[i].push_back(&supports[j]);
    }
    for (unsigned i = 0; i < 4; ++i)
        p.faces.push_back({{tetra_faces[i].begin(), tetra_faces[i].end()}, static_cast<int>(i)});
    return p;
}
std::array<unsigned, 3> axes(unsigned pivot)
{
    std::array<unsigned, 3> a{};
    unsigned j = 0;
    for (unsigned k = 0; k < 4; ++k)
        if (k != pivot)
            a[j++] = k;
    return a;
}
struct Distances
{
    B value{}, error{};
    const Support* support = nullptr;
    int orientation = 1;
};
predicates::Result distance(const B& b, const B& error, const Distances& d, AggregateWork& work)
{
    std::array<predicates::Result, 4> terms;
    for (unsigned k = 0; k < 4; ++k)
        terms[k] = construction_product({b[k], error[k]}, {d.value[k], d.error[k]}, work);
    return predicate(
        [&]
        {
            predicates::ResultAccumulator sum;
            for (const auto& term : terms)
                sum.add(term);
            return sum.result();
        },
        work);
}
bool resolved(const predicates::Result& result)
{
    return result.absolute_error == 0 || std::abs(result.value) > result.absolute_error;
}
/** Resolve from three independent defining equations, not from an ancestry of
 * rounded edge-distance estimates. Exactly singular bases may be replaced;
 * arithmetic/range/resource failures never establish contact. */
std::optional<predicates::Result> defining_distance(
    const Poly& p, std::size_t vertex, const Distances& d, AggregateWork& work)
{
    if (!p.source || !p.defining_cache || !d.support)
        fail(GeometryErrorCode::ProvenanceMismatch, "Aggregate defining-support source is missing.");
    std::array<predicates::FramedPoint, 4> source;
    for (unsigned i = 0; i < 4; ++i)
        source[i].local = p.source->vertices[i].value;
    auto query = d.support->expression;
    query.orientation = 1;
    const auto interpret = [&](predicates::SupportEvaluation evaluation) -> std::optional<predicates::Result>
    {
        auto result = evaluation.distance;
        if (result.sign == predicates::Sign::Unresolved)
            return std::nullopt;
        // Complementary queries share one evaluation, with identical error.
        result.value *= d.orientation;
        if (d.orientation < 0)
        {
            if (result.sign == predicates::Sign::Positive)
                result.sign = predicates::Sign::Negative;
            else if (result.sign == predicates::Sign::Negative)
                result.sign = predicates::Sign::Positive;
        }
        represented(result.value);
        represented(result.absolute_error);
        if (result.sign != predicates::Sign::Zero &&
            (result.value == 0 || (result.value > 0) != (result.sign == predicates::Sign::Positive)))
            fail(GeometryErrorCode::UnresolvedArithmetic,
                "Aggregate defining-support distance has no usable represented magnitude.");
        return result;
    };
    const auto& supports = p.supports.at(vertex);
    for (std::size_t a = 0; a < supports.size(); ++a)
        for (std::size_t b = a + 1; b < supports.size(); ++b)
            for (std::size_t c = b + 1; c < supports.size(); ++c)
            {
                work.pair();
                std::array<const Support*, 4> key{supports[a], supports[b], supports[c], d.support};
                std::sort(key.begin(), key.begin() + 3, support_expression_less);
                if (const auto found = p.defining_cache->find(key); found != p.defining_cache->end())
                {
                    ++work.indirect_reuses;
                    if (found->second.singular)
                        continue;
                    return interpret(found->second);
                }
                auto remaining = work.geometry;
                remaining.byte_budget = std::min(work.sweep.byte_budget, remaining.byte_budget) - work.bytes;
                if (!work.operation_limit)
                    work.operation_limit = ::SimpleFluid::PLIC::details::Work(work.geometry).operation_limit;
                remaining.maximum_operations = work.operation_limit - work.stats.operations;
                remaining.maximum_pair_tests =
                    std::min(work.sweep.maximum_pair_tests, remaining.maximum_pair_tests) - work.stats.pair_tests;
                const auto caller = ::SimpleFluid::PLIC::details::active_work;
                if (caller)
                {
                    remaining.byte_budget = std::min(remaining.byte_budget, caller->limits.byte_budget - caller->bytes);
                    remaining.maximum_operations =
                        std::min(remaining.maximum_operations, caller->operation_limit - caller->operations);
                    remaining.maximum_pair_tests =
                        std::min(remaining.maximum_pair_tests, caller->limits.maximum_pair_tests - caller->pairs);
                }
                if (!remaining.maximum_operations || !remaining.maximum_pair_tests || !remaining.byte_budget)
                    fail(GeometryErrorCode::ResourceLimit, "Aggregate defining-support work budget exhausted.");
                ::SimpleFluid::PLIC::details::Operation operation(remaining);
                // A nested Operation normally shares the caller. This fallback
                // needs its own remaining-budget guard and zero-based counters.
                const auto active = &operation.work;
                ::SimpleFluid::PLIC::details::active_work = active;
                const auto account = [&]
                {
                    work.indirect_bytes += active->bytes;
                    work.charge(active->bytes);
                    work.op(active->operations);
                    work.pairs(active->pairs);
                    if (caller)
                    {
                        caller->charge(active->bytes, 1);
                        caller->operations += active->operations;
                        caller->pairs += active->pairs;
                    }
                };
                predicates::SupportEvaluation evaluation;
                try
                {
                    ++work.indirect_calls;
                    evaluation = predicates::support_intersection_distance(
                        source, {key[0]->expression, key[1]->expression, key[2]->expression}, query);
                }
                catch (...)
                {
                    account();
                    throw;
                }
                account();
                work.charge(sizeof(DefiningCache::value_type) + 4 * sizeof(void*));
                p.defining_cache->emplace(key, evaluation);
                if (evaluation.singular)
                    continue;
                return interpret(evaluation);
            }
    return std::nullopt;
}
/** Complementary clipping retains barycentric placement at original source
 * supports. Children never undergo a metric or positivity admission. */
Poly clipped(const Poly& p, Distances d, AggregateWork& work)
{
    if (p.faces.empty())
        return {};
    work.witness();
    if (!d.support)
        fail(GeometryErrorCode::ProvenanceMismatch, "Ordinary aggregate cut has no immutable identity.");
    work.charge(p.vertices.size() * (sizeof(predicates::Result) + sizeof(bool)));
    std::vector<predicates::Result> distances(p.vertices.size());
    std::vector<bool> active(p.vertices.size(), false);
    for (const auto& f : p.faces)
        for (auto i : f.vertices)
            active[i] = true;
    bool inside = false, outside = false;
    for (std::size_t i = 0; i < p.vertices.size(); ++i)
    {
        if (!active[i])
            continue;
        if (std::find(p.supports[i].begin(), p.supports[i].end(), d.support) == p.supports[i].end())
            distances[i] = distance(p.vertices[i], p.errors[i], d, work);
        bool classified = resolved(distances[i]);
        if (!classified)
            if (const auto direct = defining_distance(p, i, d, work))
            {
                distances[i] = *direct;
                classified = true;
            }
        if (!classified)
        {
            std::ostringstream message;
            message.precision(std::numeric_limits<long double>::max_digits10);
            message << "Ordinary aggregate cut classification is unresolved: vertex=" << i
                    << " distance=" << distances[i].value << " estimate=" << distances[i].absolute_error << " weights=";
            for (auto b : p.vertices[i])
                message << b << ',';
            throw GeometryError(GeometryErrorCode::UnresolvedArithmetic, message.str());
        }
        inside |= distances[i].value < 0;
        outside |= distances[i].value > 0;
    }
    if (!inside && !outside)
        fail(GeometryErrorCode::UnresolvedArithmetic,
            "Ordinary aggregate full-dimensional witness collapsed onto a cut.");
    if (!inside)
        return {};
    if (!outside)
    {
        charge_poly_copy(p, work);
        auto retained = p;
        for (std::size_t i = 0; i < p.vertices.size(); ++i)
            if (active[i] && distances[i].value == 0 &&
                std::find(retained.supports[i].begin(), retained.supports[i].end(), d.support) ==
                    retained.supports[i].end())
            {
                retained.supports[i].push_back(d.support);
                work.charge(sizeof(const void*));
            }
        return retained;
    }
    Poly out;
    out.source = p.source;
    out.defining_cache = p.defining_cache;
    charge_poly_points(p, work);
    out.vertices = p.vertices;
    out.errors = p.errors;
    out.supports = p.supports;
    out.constructions = p.constructions;
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> intersections;
    std::set<std::size_t> cap;
    std::map<std::size_t, std::size_t> cap_ring;
    const auto add_cap = [&](std::size_t i)
    {
        if (cap.insert(i).second)
            work.charge(sizeof(std::size_t) + 4 * sizeof(void*));
    };
    for (const auto& face : p.faces)
    {
        Face f;
        f.source = face.source;
        auto previous = face.vertices.back();
        for (auto current : face.vertices)
        {
            work.op();
            const auto dp = distances[previous].value, dc = distances[current].value;
            if ((dp < 0 && dc > 0) || (dp > 0 && dc < 0))
            {
                const auto key = std::minmax(previous, current);
                auto [it, fresh] = intersections.try_emplace(key, out.vertices.size());
                if (fresh)
                {
                    work.charge(2 * sizeof(B) + sizeof(Construction) + sizeof(decltype(intersections)::value_type) +
                                4 * sizeof(void*));
                    const auto i = std::abs(dp) <= std::abs(dc) ? previous : current,
                               j = i == previous ? current : previous;
                    const auto denominator = construction_difference(distances[i], distances[j], work);
                    const auto t = construction_quotient(distances[i], denominator, work);
                    const auto complement = construction_difference({1, 0}, t, work);
                    B b{}, error{};
                    for (unsigned k = 0; k < 4; ++k)
                    {
                        const auto coordinate =
                            construction_sum(construction_product(complement, {p.vertices[i][k], p.errors[i][k]}, work),
                                construction_product(t, {p.vertices[j][k], p.errors[j][k]}, work), work);
                        b[k] = coordinate.value;
                        error[k] = coordinate.absolute_error;
                        if (!std::isfinite(b[k]) || !std::isfinite(error[k]))
                            fail(GeometryErrorCode::UnresolvedArithmetic,
                                "Ordinary aggregate interpolation estimate is nonfinite.");
                    }
                    out.vertices.push_back(b);
                    out.errors.push_back(error);
                    out.constructions.push_back({i, j, d.support, d.orientation});
                    std::vector<const Support*> supports{d.support};
                    for (auto id : p.supports[i])
                        if (std::find(p.supports[j].begin(), p.supports[j].end(), id) != p.supports[j].end() &&
                            id != d.support)
                            supports.push_back(id);
                    work.charge(sizeof(std::vector<const Support*>) + supports.size() * sizeof(const Support*));
                    out.supports.push_back(std::move(supports));
                    work.op(20);
                }
                f.vertices.push_back(it->second);
                add_cap(it->second);
            }
            if (dc <= 0)
            {
                f.vertices.push_back(current);
                if (dc == 0)
                {
                    add_cap(current);
                    if (std::find(out.supports[current].begin(), out.supports[current].end(), d.support) ==
                        out.supports[current].end())
                    {
                        out.supports[current].push_back(d.support);
                        work.charge(sizeof(const void*));
                    }
                }
            }
            previous = current;
        }
        if (f.vertices.size() >= 3)
        {
            for (std::size_t k = 0; k < f.vertices.size(); ++k)
            {
                const auto a = f.vertices[k], b = f.vertices[(k + 1) % f.vertices.size()];
                if (a != b && cap.contains(a) && cap.contains(b))
                    if (!cap_ring.emplace(b, a).second)
                        fail(GeometryErrorCode::UnresolvedArithmetic,
                            "Ordinary aggregate cap ring has ambiguous incidence.");
                    else
                        work.charge(sizeof(decltype(cap_ring)::value_type) + 4 * sizeof(void*));
            }
            work.charge(sizeof(Face) + f.vertices.size() * sizeof(std::size_t));
            out.faces.push_back(std::move(f));
        }
    }
    if (cap.size() < 3)
        fail(GeometryErrorCode::UnresolvedArithmetic, "Ordinary aggregate cut lost its retained boundary.");
    Face f;
    if (cap_ring.size() != cap.size())
        fail(GeometryErrorCode::UnresolvedArithmetic, "Ordinary aggregate cap ring is incomplete.");
    auto current = cap_ring.begin()->first;
    do
    {
        f.vertices.push_back(current);
        auto it = cap_ring.find(current);
        if (it == cap_ring.end())
            fail(GeometryErrorCode::UnresolvedArithmetic, "Ordinary aggregate cap ring is open.");
        current = it->second;
    } while (current != f.vertices.front() && f.vertices.size() <= cap_ring.size());
    if (f.vertices.size() != cap_ring.size())
        fail(GeometryErrorCode::UnresolvedArithmetic, "Ordinary aggregate cap ring is disconnected.");
    work.charge(sizeof(Face) + f.vertices.size() * sizeof(std::size_t));
    out.faces.push_back(std::move(f));
    if (out.vertices.size() > work.geometry.maximum_vertices || out.faces.size() > work.geometry.maximum_faces)
        fail(GeometryErrorCode::ResourceLimit, "Ordinary aggregate V/F cap exhausted.");
    return out;
}
struct Cut
{
    V normal{}, anchor{};
    std::array<double, 3> origin{};
    int exponent = 0, orientation = 1;
    long double offset = 0;
    std::vector<PlaneAuthority> authorities;
    std::vector<char> native_support;
    const void* identity = nullptr;
};
Distances source_distances(
    const OrdinaryAggregateTetra& t, const Support& support, int orientation, AggregateWork& work)
{
    Distances result;
    result.support = &support;
    result.orientation = orientation;
    const auto& expression = support.expression;
    for (unsigned i = 0; i < 4; ++i)
    {
        if (expression.source_incidence & (1u << i))
            continue;
        const auto distance = predicate(
            [&]
            {
                return predicates::plane_distance(predicates::FramedPoint{t.vertices[i].value, {}, 0},
                    expression.anchor, expression.normal, expression.offset);
            },
            work);
        result.value[i] = distance.value * orientation;
        result.error[i] = distance.absolute_error;
        work.op(64);
    }
    return result;
}
Cut phase_cut(const GeometryHalfspace& h)
{
    Cut c;
    c.identity = &h;
    for (unsigned k = 0; k < 3; ++k)
    {
        c.normal[k] = h.normal[k];
        c.anchor[k] = h.anchor[k];
    }
    c.offset = h.offset;
    if (!std::isfinite(dot(c.normal, c.normal)) || !(dot(c.normal, c.normal) > 0) || !std::isfinite(c.offset))
        fail(GeometryErrorCode::InvalidInput, "Invalid Ordinary aggregate phase plane.");
    return c;
}
std::vector<Cut> donor_cuts(const OrdinaryAggregateDonor& d, AggregateWork& work)
{
    const auto storage = ::SimpleFluid::PLIC::details::checked_geometry_storage<ArithmeticMode::Ordinary>(
        DynamicGeometryAccess::implementation(*d.geometry->dynamic_geometry()));
    std::vector<Cut> result;
    work.charge(storage->surfaces.size() * sizeof(Cut));
    for (const auto& surface : storage->surfaces)
    {
        Cut c;
        const auto& plane = *surface.plane;
        c.identity = surface.plane.get();
        for (unsigned k = 0; k < 3; ++k)
        {
            c.normal[k] = plane.normal[k].value;
            c.anchor[k] = plane.anchor[k].value;
            c.origin[k] = plane.frame.origin[k];
        }
        c.exponent = plane.frame.exponent;
        c.offset = plane.offset.value;
        c.orientation = surface.orientation;
        c.authorities = plane.authorities;
        c.native_support = plane.native_support;
        if (c.authorities.empty() && !c.native_support.empty())
            c.authorities.push_back(::SimpleFluid::PLIC::details::original_support_authority(c.native_support));
        work.charge(c.native_support.size());
        for (const auto& authority : c.authorities)
            work.charge(sizeof(PlaneAuthority) + authority.key.size() * sizeof(std::uint64_t));
        result.push_back(std::move(c));
    }
    return result;
}
/** Multilinear integration leaves tiny barycentric weights intact. Every
 * surviving exterior face uses the SAME physical pivot and power-of-two scale. */
std::array<predicates::Result, 4> integration_coefficients(
    const OrdinaryAggregateTetra& t, V pivot, long double scale, AggregateWork& work)
{
    const auto cube = product(product(scale, scale), scale);
    std::array<predicates::Result, 4> result;
    for (unsigned omitted = 0; omitted < 4; ++omitted)
    {
        const auto j = axes(omitted);
        const auto coefficient = predicate(
            [&]
            {
                return predicates::orientation(
                    {pivot, t.vertices[j[0]].value, t.vertices[j[1]].value, t.vertices[j[2]].value});
            },
            work);
        result[omitted] = {quotient(coefficient.value, cube), quotient(coefficient.absolute_error, cube)};
    }
    return result;
}
void integrate(const Poly& p, const std::array<predicates::Result, 4>& coefficients, const predicates::Result& jacobian,
    const std::array<bool, 4>& cancelled, Sum& sum, AggregateWork& work)
{
    const auto orientation = jacobian.value > 0 ? 1.L : -1.L;
    for (const auto& f : p.faces)
    {
        if (f.source >= 0 && cancelled[f.source])
            continue;
        const auto& a = p.vertices[f.vertices[0]];
        for (std::size_t triangle = 1; triangle + 1 < f.vertices.size(); ++triangle)
        {
            const auto& b = p.vertices[f.vertices[triangle]];
            const auto& c = p.vertices[f.vertices[triangle + 1]];
            const auto& ea = p.errors[f.vertices[0]];
            const auto& eb = p.errors[f.vertices[triangle]];
            const auto& ec = p.errors[f.vertices[triangle + 1]];
            for (unsigned omitted = 0; omitted < 4; ++omitted)
            {
                const auto j = axes(omitted);
                const auto weight = predicate(
                    [&]
                    {
                        return predicates::orientation({V{}, V{a[j[0]], a[j[1]], a[j[2]]}, V{b[j[0]], b[j[1]], b[j[2]]},
                            V{c[j[0]], c[j[1]], c[j[2]]}});
                    },
                    work);
                auto weight_error = weight.absolute_error;
                for (unsigned i = 0; i < 3; ++i)
                    for (unsigned k = 0; k < 3; ++k)
                        if (k != i)
                        {
                            const auto l = 3 - i - k;
                            const auto av = std::abs(a[j[i]]), bv = std::abs(b[j[k]]), cv = std::abs(c[j[l]]);
                            weight_error = sum_values(
                                weight_error, sum_values(sum_values(product(product(ea[j[i]], sum_values(bv, eb[j[k]])),
                                                                        sum_values(cv, ec[j[l]])),
                                                             product(product(av, eb[j[k]]), sum_values(cv, ec[j[l]]))),
                                                  product(product(av, bv), ec[j[l]])));
                        }
                const auto value = coefficients[omitted].value, error = coefficients[omitted].absolute_error;
                const auto term = quotient(product(product(orientation, value), weight.value), 6);
                sum.add(term, sum_values(quotient(sum_values(product(std::abs(value), weight_error),
                                                      product(error, sum_values(std::abs(weight.value), weight_error))),
                                             6),
                                  product(8 * eps, std::abs(term))));
                work.op(64);
            }
        }
    }
}
/** Reconstruct the same aggregate integral in immutable source domains.
 * Each witness uses its FULL closed chain, so its local chart/reference may
 * differ from its neighbor's. Domain additivity is equivalent to integration
 * of their identity-cancelled exterior boundary. No witness scalar is admitted
 * or published: every signed cone enters the final physical authority's Sum. */
void integrate_source_reference(
    const Poly& p, const predicates::Result& jacobian, long double scale, Sum& sum, AggregateWork& work)
{
    if (p.faces.empty())
        return;
    unsigned dominant = 0;
    std::size_t reference = p.faces.front().vertices.front();
    long double largest = -1;
    for (const auto& face : p.faces)
        for (auto vertex : face.vertices)
            for (unsigned axis = 0; axis < 4; ++axis)
                if (p.vertices[vertex][axis] > largest)
                {
                    largest = p.vertices[vertex][axis];
                    dominant = axis;
                    reference = vertex;
                }
    const auto chart = axes(dominant);
    const auto cube = product(product(scale, scale), scale);
    const auto jacobian_value = quotient(std::abs(jacobian.value), cube);
    const auto jacobian_error = quotient(jacobian.absolute_error, cube);
    const auto parity = dominant % 2 ? -1.L : 1.L;
    V q{};
    for (unsigned axis = 0; axis < 3; ++axis)
        q[axis] = p.vertices[reference][chart[axis]];
    for (const auto& face : p.faces)
        for (std::size_t triangle = 1; triangle + 1 < face.vertices.size(); ++triangle)
        {
            const std::array<std::size_t, 3> vertex{
                face.vertices[0], face.vertices[triangle], face.vertices[triangle + 1]};
            // Identical point identity proves this cone exactly zero, including
            // its correlated placement error. No magnitude screen is involved.
            if (std::find(vertex.begin(), vertex.end(), reference) != vertex.end())
                continue;
            std::array<V, 3> point{}, difference{}, error{};
            for (unsigned i = 0; i < 3; ++i)
                for (unsigned axis = 0; axis < 3; ++axis)
                {
                    const auto coordinate = chart[axis];
                    point[i][axis] = p.vertices[vertex[i]][coordinate];
                    const auto delta = std::abs(sum_values(point[i][axis], -q[axis]));
                    difference[i][axis] =
                        delta == 0 ? 0
                                   : represented(std::nextafter(delta, std::numeric_limits<long double>::infinity()));
                    error[i][axis] = sum_values(p.errors[vertex[i]][coordinate], p.errors[reference][coordinate]);
                }
            const auto determinant =
                predicate([&] { return predicates::orientation({q, point[0], point[1], point[2]}); }, work);
            auto determinant_error = determinant.absolute_error;
            for (unsigned i = 0; i < 3; ++i)
                for (unsigned j = 0; j < 3; ++j)
                    if (i != j)
                    {
                        const auto k = 3 - i - j;
                        const auto av = difference[0][i], bv = difference[1][j], cv = difference[2][k];
                        const auto ae = error[0][i], be = error[1][j], ce = error[2][k];
                        determinant_error = sum_values(determinant_error,
                            sum_values(sum_values(product(product(ae, sum_values(bv, be)), sum_values(cv, ce)),
                                           product(product(av, be), sum_values(cv, ce))),
                                product(product(av, bv), ce)));
                    }
            const auto term = quotient(product(product(parity, jacobian_value), determinant.value), 6);
            const auto term_error =
                quotient(sum_values(product(jacobian_value, determinant_error),
                             product(jacobian_error, sum_values(std::abs(determinant.value), determinant_error))),
                    6);
            sum.add(term, sum_values(term_error, product(8 * eps, std::abs(term))));
            work.op(192);
        }
}
} // namespace
struct OrdinaryAggregateSource::Impl
{
    std::vector<OrdinaryAggregateTetra> tetrahedra;
    std::vector<predicates::Result> orientations;
    std::vector<std::array<bool, 4>> cancelled;
    std::vector<std::array<std::pair<std::size_t, unsigned>, 4>> partners;
    SweepLimits sweep;
    GeometryLimits geometry;
    OrdinaryAggregateStatistics initial;
    long double signed_volume = 0;
};
OrdinaryAggregateSource::OrdinaryAggregateSource(std::span<const OrdinaryAggregateTetra> recipes, SweepLimits sweep,
    GeometryLimits geometry, OrdinaryAggregateContext* progress)
{
    OrdinaryAggregateContext local;
    auto& context = progress ? *progress : local;
    context = {};
    const ScopedAggregateEnvironment environment;
    if (geometry.arithmetic_mode != ArithmeticMode::Ordinary)
        fail(GeometryErrorCode::InvalidInput, "Ordinary aggregate requires Ordinary geometry limits.");
    if (recipes.size() > sweep.maximum_simplices)
        fail(GeometryErrorCode::ResourceLimit, "Ordinary aggregate source simplex cap exhausted.");
    AggregateWork work{sweep, geometry};
    work.context = &context;
    work.charge(sizeof(Impl) + recipes.size() * (sizeof(OrdinaryAggregateTetra) + sizeof(predicates::Result) +
                                                    sizeof(std::array<bool, 4>) +
                                                    sizeof(std::array<std::pair<std::size_t, unsigned>, 4>)));
    auto impl = std::make_shared<Impl>();
    impl->sweep = sweep;
    impl->geometry = geometry;
    impl->tetrahedra.assign(recipes.begin(), recipes.end());
    impl->orientations.reserve(recipes.size());
    impl->cancelled.resize(recipes.size());
    impl->partners.resize(recipes.size());
    std::map<std::uint64_t, Point<ArithmeticMode::Ordinary>> points;
    struct Incidence
    {
        std::size_t tetra;
        unsigned face;
        int orientation;
    };
    std::map<std::pair<int, std::array<std::uint64_t, 3>>, Incidence> boundaries;
    Sum volume;
    for (std::size_t i = 0; i < recipes.size(); ++i)
    {
        context.source = i;
        const auto& t = recipes[i];
        if (t.sign != 1 && t.sign != -1)
            fail(GeometryErrorCode::InvalidInput, "Invalid Ordinary aggregate stream sign.");
        for (unsigned k = 0; k < 4; ++k)
        {
            for (auto x : t.vertices[k].value)
                if (!std::isfinite(x))
                    fail(GeometryErrorCode::InvalidInput, "Nonfinite Ordinary aggregate source.");
            auto [it, fresh] = points.try_emplace(t.vertex_ids[k], t.vertices[k]);
            if (fresh)
                work.charge(
                    sizeof(decltype(points)::value_type) + 4 * sizeof(void*) + point_storage_bytes(t.vertices[k]));
            else if (it->second.value != t.vertices[k].value || it->second.supports() != t.vertices[k].supports())
                fail(GeometryErrorCode::ProvenanceMismatch,
                    "Ordinary aggregate source identity changed its immutable point.");
            for (unsigned j = 0; j < k; ++j)
                if (t.vertex_ids[k] == t.vertex_ids[j])
                    fail(GeometryErrorCode::InvalidTopology, "Repeated source tetra vertex identity.");
        }
        const auto orientation = predicate(
            [&]
            {
                return predicates::orientation(
                    {t.vertices[0].value, t.vertices[1].value, t.vertices[2].value, t.vertices[3].value});
            },
            work);
        const auto d = orientation.value;
        work.op(64);
        if (!std::isfinite(d) || orientation.sign == predicates::Sign::Unresolved || d == 0)
            fail(GeometryErrorCode::UnresolvedArithmetic, "Ordinary aggregate source orientation is unresolved.");
        impl->orientations.push_back(orientation);
        volume.add(product(t.sign, quotient(std::abs(d), 6)));
        for (unsigned f = 0; f < 4; ++f)
        {
            std::array<std::uint64_t, 3> ids{};
            for (unsigned k = 0; k < 3; ++k)
                ids[k] = t.vertex_ids[tetra_faces[f][k]];
            int parity = d > 0 ? 1 : -1;
            for (unsigned a = 0; a < 3; ++a)
                for (unsigned b = a + 1; b < 3; ++b)
                    if (ids[a] > ids[b])
                        parity = -parity;
            std::sort(ids.begin(), ids.end());
            auto [it, fresh] = boundaries.try_emplace({t.sign, ids}, Incidence{i, f, parity});
            work.pair();
            if (fresh)
                work.charge(sizeof(decltype(boundaries)::value_type) + 4 * sizeof(void*));
            else
            {
                if (it->second.orientation != -parity || impl->cancelled[it->second.tetra][it->second.face])
                    fail(GeometryErrorCode::InvalidTopology, "Ordinary aggregate source face has invalid incidence.");
                impl->cancelled[i][f] = impl->cancelled[it->second.tetra][it->second.face] = true;
                impl->partners[i][f] = {it->second.tetra, it->second.face};
                impl->partners[it->second.tetra][it->second.face] = {i, f};
                ++work.stats.cancelled_boundaries;
            }
        }
    }
    impl->signed_volume = volume.get();
    work.stats.source_pieces = recipes.size();
    impl->initial = work.stats;
    impl_ = std::move(impl);
}
std::size_t OrdinaryAggregateSource::storage_bytes() const noexcept
{
    return impl_->initial.storage_bytes;
}
OrdinaryAggregateResult OrdinaryAggregateSource::measure(
    std::span<const OrdinaryAggregateDonor> donors, OrdinaryAggregateContext* progress) const
{
    OrdinaryAggregateContext local;
    auto& context = progress ? *progress : local;
    context = {};
    context.stage = OrdinaryAggregateContext::Stage::DonorValidation;
    const ScopedAggregateEnvironment environment;
    AggregateWork work{impl_->sweep, impl_->geometry, impl_->initial, impl_->initial.storage_bytes};
    work.context = &context;
    work.sync();
    std::vector<ConvexGeometryPtr<ArithmeticMode::Ordinary>> geometries;
    std::vector<std::vector<Cut>> cuts;
    std::set<std::uint64_t> ids;
    work.charge(donors.size() * (sizeof(ConvexGeometryPtr<ArithmeticMode::Ordinary>) + sizeof(std::vector<Cut>) +
                                    sizeof(OrdinaryAggregateGroup) * 2));
    for (const auto& d : donors)
    {
        context.donor = d.id;
        if (!d.geometry || !ids.insert(d.id).second || bool(d.gas) == (d.pure_gas || d.pure_carrier) ||
            d.pure_gas && d.pure_carrier)
            fail(GeometryErrorCode::InvalidInput, "Invalid Ordinary aggregate donor declaration.");
        for (auto b : {d.bulk_budget, d.gas_budget, d.carrier_budget})
            if (!std::isfinite(b) || b < 0)
                fail(GeometryErrorCode::InvalidInput, "Invalid aggregate accounting budget.");
        geometries.push_back(d.geometry);
        cuts.push_back(donor_cuts(d, work));
    }
    // Distinct native cells can pin different StoredPlane owners for one
    // authenticated support. Unify ONLY issued support identities, never
    // numerical coplanarity, so complementary donor cuts retain dependency.
    std::map<const void*, const void*> parents;
    std::map<PlaneAuthority, const void*> authorities;
    std::map<std::vector<char>, const void*> native_supports;
    const auto root = [&](const void* id)
    {
        while (parents.at(id) != id)
            id = parents.at(id);
        return id;
    };
    for (auto& dc : cuts)
        for (auto& c : dc)
        {
            if (parents.try_emplace(c.identity, c.identity).second)
                work.charge(sizeof(decltype(parents)::value_type) + 4 * sizeof(void*));
            for (const auto& a : c.authorities)
            {
                auto [it, fresh] = authorities.try_emplace(a, c.identity);
                if (fresh)
                    work.charge(sizeof(decltype(authorities)::value_type) + a.key.size() * sizeof(std::uint64_t) +
                                4 * sizeof(void*));
                else
                    parents[root(c.identity)] = root(it->second);
            }
            if (!c.native_support.empty())
            {
                auto [it, fresh] = native_supports.try_emplace(c.native_support, c.identity);
                if (fresh)
                    work.charge(
                        sizeof(decltype(native_supports)::value_type) + c.native_support.size() + 4 * sizeof(void*));
                else
                    parents[root(c.identity)] = root(it->second);
            }
        }
    for (auto& dc : cuts)
        for (auto& c : dc)
            c.identity = root(c.identity);
    // One original expression per issued support identity, independent of donor
    // traversal order. Authority establishes equality; rounded coefficients do
    // not create an identity. Incidence orientation remains separate.
    std::map<const void*, Cut> canonical;
    const auto expression_order = [](const Cut& c)
    { return std::tie(c.normal, c.origin, c.exponent, c.anchor, c.offset); };
    for (const auto& dc : cuts)
        for (const auto& c : dc)
        {
            auto [it, fresh] = canonical.try_emplace(c.identity, c);
            auto& selected = it->second;
            if (fresh)
            {
                work.charge(sizeof(decltype(canonical)::value_type) + 4 * sizeof(void*) +
                            selected.native_support.capacity() +
                            selected.authorities.capacity() * sizeof(PlaneAuthority));
                for (const auto& authority : selected.authorities)
                    work.charge(authority.key.capacity() * sizeof(std::uint64_t));
            }
            else
            {
                if (expression_order(c) < expression_order(selected))
                {
                    selected.normal = c.normal;
                    selected.origin = c.origin;
                    selected.exponent = c.exponent;
                    selected.anchor = c.anchor;
                    selected.offset = c.offset;
                }
                for (const auto& authority : c.authorities)
                    if (std::find(selected.authorities.begin(), selected.authorities.end(), authority) ==
                        selected.authorities.end())
                    {
                        work.charge(2 * sizeof(PlaneAuthority) + authority.key.size() * sizeof(std::uint64_t));
                        selected.authorities.push_back(authority);
                    }
            }
            selected.orientation = 1;
        }
    // Source leaves and cut supports are immutable throughout this measure.
    // Share their predicate results across coverage, phase channels and the
    // reconstruction retry; only executed predicates consume floating work.
    work.charge(donors.size() * sizeof(Cut));
    std::vector<Cut> phase_cuts(donors.size());
    for (std::size_t di = 0; di < donors.size(); ++di)
        if (donors[di].gas)
            phase_cuts[di] = phase_cut(*donors[di].gas);
    work.charge(impl_->tetrahedra.size() *
                (sizeof(std::array<Support, 4>) + sizeof(std::map<const void*, Support>) + sizeof(DefiningCache)));
    std::vector<std::array<Support, 4>> source_faces(impl_->tetrahedra.size());
    std::vector<DefiningCache> defining_caches(impl_->tetrahedra.size());
    std::vector<std::map<const void*, Support>> support_catalogs(impl_->tetrahedra.size());
    for (auto& faces : source_faces)
        for (unsigned j = 0; j < 4; ++j)
        {
            faces[j].expression.source_face = j;
            faces[j].identity = &faces[j];
        }
    std::map<std::array<std::size_t, 3>, Distances> distance_cache;
    const auto source_values = [&](std::size_t ti, std::size_t di, std::size_t ci) -> const Distances&
    {
        const std::array key{ti, di, ci};
        const auto found = distance_cache.find(key);
        if (found != distance_cache.end())
            return found->second;
        const bool phase = ci == cuts[di].size();
        const auto& cut = phase ? phase_cuts[di] : cuts[di][ci];
        const auto& original = phase ? cut : canonical.at(cut.identity);
        auto& catalog = support_catalogs[ti];
        auto [position, fresh] = catalog.try_emplace(cut.identity);
        auto& support = position->second;
        if (fresh)
        {
            work.charge(sizeof(std::map<const void*, Support>::value_type) + 4 * sizeof(void*));
            support.identity = cut.identity;
            support.expression.anchor = {original.anchor, original.origin, original.exponent};
            support.expression.normal = original.normal;
            support.expression.offset = original.offset;
            for (unsigned i = 0; i < 4; ++i)
                if (has_any_support(impl_->tetrahedra[ti].vertices[i].dynamic(), original.authorities))
                    support.expression.source_incidence |= 1u << i;
        }
        unsigned pivot = 0;
        for (unsigned k = 1; k < 3; ++k)
            if (std::abs(original.normal[k]) > std::abs(original.normal[pivot]))
                pivot = k;
        if (original.normal[pivot] == 0 || cut.normal[pivot] == 0)
            fail(GeometryErrorCode::UnresolvedArithmetic, "Aggregate original support orientation is unresolved.");
        const auto orientation = cut.orientation * ((original.normal[pivot] > 0) == (cut.normal[pivot] > 0) ? 1 : -1);
        auto values = source_distances(impl_->tetrahedra[ti], support, orientation, work);
        work.charge(sizeof(decltype(distance_cache)::value_type) + 4 * sizeof(void*));
        return distance_cache.emplace(key, values).first->second;
    };
    context.source = context.donor = context.cut = OrdinaryAggregateContext::absent;
    // Existing donor admission authenticates disjoint interiors. Only its
    // constructor is used: no computational child is materialized/measured.
    {
        auto remaining = impl_->geometry;
        remaining.byte_budget = std::min(impl_->sweep.byte_budget, remaining.byte_budget) - work.bytes;
        remaining.maximum_pair_tests =
            std::min(impl_->sweep.maximum_pair_tests, remaining.maximum_pair_tests) - work.stats.pair_tests;
        if (!work.operation_limit)
            work.operation_limit = ::SimpleFluid::PLIC::details::Work(impl_->geometry).operation_limit;
        remaining.maximum_operations = work.operation_limit - work.stats.operations;
        if (!remaining.maximum_operations)
            fail(GeometryErrorCode::ResourceLimit, "Ordinary aggregate work budget exhausted before donor validation.");
        ::SimpleFluid::PLIC::details::Operation validation(remaining);
        const auto active = ::SimpleFluid::PLIC::details::active_work;
        const auto bytes_before = active->bytes, operations_before = active->operations, pairs_before = active->pairs;
        const auto aggregate_bytes = work.bytes, aggregate_operations = work.stats.operations,
                   aggregate_pairs = work.stats.pair_tests;
        try
        {
            const ConvexPartition<ArithmeticMode::Ordinary> disjoint(geometries, remaining);
            work.charge(active->bytes - bytes_before + sizeof(disjoint));
            work.op(active->operations - operations_before);
            work.pairs(active->pairs - pairs_before);
        }
        catch (...)
        {
            // Preserve partial provider work even when validation fails before
            // its normal accounting handoff. Do not mask the original cause.
            context.bytes = aggregate_bytes + active->bytes - bytes_before;
            context.operations = aggregate_operations + active->operations - operations_before;
            context.pairs = aggregate_pairs + active->pairs - pairs_before;
            throw;
        }
    }
    // Subtract every donor through retained complementary cuts. Every outside
    // child remains a witness until a later donor consumes it. No volume closure
    // tolerance substitutes for this structural coverage proof.
    context.stage = OrdinaryAggregateContext::Stage::Coverage;
    for (std::size_t ordinal = 0; ordinal < impl_->tetrahedra.size(); ++ordinal)
    {
        context.source = ordinal;
        for (unsigned retry = 0; retry < 2; ++retry)
        {
            context.retry = retry;
            try
            {
                std::vector<Poly> remainder{
                    tetra(impl_->tetrahedra[ordinal], source_faces[ordinal], defining_caches[ordinal], work)};
                for (std::size_t donor_index = 0; donor_index < cuts.size(); ++donor_index)
                {
                    const auto donor_ordinal = retry ? cuts.size() - 1 - donor_index : donor_index;
                    context.donor = donors[donor_ordinal].id;
                    const auto& dc = cuts[donor_ordinal];
                    std::vector<Poly> next;
                    for (auto& retained : remainder)
                    {
                        auto piece = std::move(retained);
                        for (std::size_t cut_index = 0; cut_index < dc.size(); ++cut_index)
                        {
                            if (piece.faces.empty())
                                break;
                            context.cut = retry ? dc.size() - 1 - cut_index : cut_index;
                            auto d = source_values(ordinal, donor_ordinal, context.cut);
                            auto opposite = d;
                            opposite.orientation = -opposite.orientation;
                            for (auto& x : opposite.value)
                                x = -x;
                            auto outside = clipped(piece, opposite, work);
                            if (!outside.faces.empty())
                                next.push_back(std::move(outside));
                            piece = clipped(piece, d, work);
                            work.pair();
                        }
                    }
                    remainder = std::move(next);
                    if (remainder.empty())
                        break;
                }
                if (!remainder.empty())
                    fail(GeometryErrorCode::InvalidTopology,
                        "Ordinary aggregate donor cover leaves an uncovered source remainder.");
                break;
            }
            catch (const GeometryError& error)
            {
                if (error.code() != GeometryErrorCode::UnresolvedArithmetic || retry)
                    throw;
                ++work.stats.retries;
            }
        }
    }
    context = {};
    context.stage = OrdinaryAggregateContext::Stage::Metric;
    work.sync();
    OrdinaryAggregateResult result;
    result.signed_source_volume = impl_->signed_volume;
    for (std::size_t di = 0; di < donors.size(); ++di)
        for (int sign : {-1, 1})
        {
            const auto& donor = donors[di];
            context.donor = donor.id;
            bool used = false;
            for (const auto& t : impl_->tetrahedra)
                used |= t.sign == sign;
            if (!used)
                continue;
            OrdinaryAggregateGroup accepted;
            std::string quality_detail;
            bool resolved = false;
            for (unsigned retry = 0; retry < 2 && !resolved; ++retry)
            {
                context.source = context.cut = OrdinaryAggregateContext::absent;
                context.retry = retry;
                if (retry)
                    ++work.stats.retries;
                try
                {
                    OrdinaryAggregateGroup group;
                    group.donor = donor.id;
                    group.sign = sign;
                    std::array<Sum, 3> sums;
                    std::array<bool, 3> positive{};
                    V pivot{};
                    long double scale = 0;
                    const auto& first = *std::find_if(impl_->tetrahedra.begin(), impl_->tetrahedra.end(),
                        [&](const auto& t) { return t.sign == sign; });
                    pivot = first.vertices[0].value;
                    auto closest = std::numeric_limits<long double>::infinity();
                    for (const auto& t : impl_->tetrahedra)
                        if (t.sign == sign)
                            for (const auto& p : t.vertices)
                            {
                                if (!retry)
                                {
                                    if (p.value < pivot)
                                        pivot = p.value;
                                }
                                else if (donor.gas)
                                {
                                    const auto c = phase_cut(*donor.gas);
                                    const auto distance = predicate([&]
                                        { return predicates::plane_distance(p.value, c.normal, c.anchor, c.offset); },
                                        work);
                                    const auto value = std::abs(distance.value);
                                    if (value < closest || (value == closest && p.value < pivot))
                                    {
                                        closest = value;
                                        pivot = p.value;
                                    }
                                }
                                else if (pivot < p.value)
                                    pivot = p.value;
                            }
                    for (const auto& t : impl_->tetrahedra)
                        if (t.sign == sign)
                            for (const auto& p : t.vertices)
                                for (unsigned k = 0; k < 3; ++k)
                                    scale = std::max(scale, std::abs(p.value[k] - pivot[k]));
                    int exponent = 0;
                    std::frexp(scale, &exponent);
                    scale = represented(std::ldexp(1.L, exponent));
                    struct Witness
                    {
                        std::size_t tetra;
                        Poly bulk, gas, carrier;
                        std::array<predicates::Result, 4> coefficients;
                    };
                    std::vector<Witness> witnesses;
                    work.charge(impl_->tetrahedra.size() * sizeof(Witness));
                    for (std::size_t ti = 0; ti < impl_->tetrahedra.size(); ++ti)
                    {
                        context.source = ti;
                        const auto& t = impl_->tetrahedra[ti];
                        if (t.sign != sign)
                            continue;
                        auto bulk = tetra(t, source_faces[ti], defining_caches[ti], work);
                        // Replay immutable supports in reverse order on the single
                        // reconstruction retry; rounded child coordinates are not inputs.
                        for (std::size_t k = 0; k < cuts[di].size() && !bulk.faces.empty(); ++k)
                        {
                            const auto j = retry ? cuts[di].size() - 1 - k : k;
                            context.cut = j;
                            bulk = clipped(bulk, source_values(ti, di, j), work);
                            work.pair();
                        }
                        if (bulk.faces.empty())
                            continue;
                        Witness witness;
                        witness.tetra = ti;
                        if (!retry)
                            witness.coefficients = integration_coefficients(t, pivot, scale, work);
                        if (!donor.pure_gas && !donor.pure_carrier)
                        {
                            context.cut = cuts[di].size(); // The phase support follows donor supports.
                            auto d = source_values(ti, di, cuts[di].size());
                            // Rebuild phase witnesses from immutable source
                            // supports on the existing retry. Applying the phase
                            // first retains its small endpoint weights before
                            // donor intersections introduce dependent crossings.
                            auto phase_source = retry ? tetra(t, source_faces[ti], defining_caches[ti], work) : Poly{};
                            const auto& phase_parent = retry ? phase_source : bulk;
                            witness.gas = clipped(phase_parent, d, work);
                            d.orientation = -d.orientation;
                            for (auto& x : d.value)
                                x = -x;
                            witness.carrier = clipped(phase_parent, d, work);
                            if (retry)
                                for (std::size_t k = 0; k < cuts[di].size(); ++k)
                                {
                                    const auto j = cuts[di].size() - 1 - k;
                                    context.cut = j;
                                    const auto cut = source_values(ti, di, j);
                                    if (!witness.gas.faces.empty())
                                        witness.gas = clipped(witness.gas, cut, work);
                                    if (!witness.carrier.faces.empty())
                                        witness.carrier = clipped(witness.carrier, cut, work);
                                    work.pair();
                                }
                        }
                        witness.bulk = std::move(bulk);
                        witnesses.push_back(std::move(witness));
                    }
                    for (unsigned channel = 0; channel < 3; ++channel)
                    {
                        const auto poly = [&](const Witness& w) -> const Poly&
                        {
                            if (channel == 0 || (channel == 1 && donor.pure_gas) ||
                                (channel == 2 && donor.pure_carrier))
                                return w.bulk;
                            return channel == 1 ? w.gas : w.carrier;
                        };
                        std::vector<std::array<bool, 4>> present(impl_->tetrahedra.size());
                        work.charge(present.size() * sizeof(std::array<bool, 4>));
                        for (const auto& w : witnesses)
                            for (const auto& f : poly(w).faces)
                                if (f.source >= 0)
                                    present[w.tetra][f.source] = true;
                        for (const auto& w : witnesses)
                        {
                            const auto& p = poly(w);
                            if (p.faces.empty())
                                continue;
                            positive[channel] = true;
                            std::array<bool, 4> cancelled{};
                            for (unsigned f = 0; f < 4; ++f)
                                if (impl_->cancelled[w.tetra][f])
                                {
                                    const auto [neighbor, face] = impl_->partners[w.tetra][f];
                                    cancelled[f] = present[neighbor][face];
                                }
                            context.source = w.tetra;
                            context.cut = OrdinaryAggregateContext::absent;
                            if (retry)
                                integrate_source_reference(p, impl_->orientations[w.tetra], scale, sums[channel], work);
                            else
                                integrate(
                                    p, w.coefficients, impl_->orientations[w.tetra], cancelled, sums[channel], work);
                        }
                    }
                    context.source = context.cut = OrdinaryAggregateContext::absent;
                    const auto cube = product(product(scale, scale), scale);
                    auto output =
                        std::array<long double*, 3>{&group.nominal.bulk, &group.nominal.gas, &group.nominal.carrier};
                    auto errors = std::array<long double*, 3>{&group.numerical_error_estimate.bulk,
                        &group.numerical_error_estimate.gas, &group.numerical_error_estimate.carrier};
                    const std::array<long double, 3> budgets{donor.bulk_budget, donor.gas_budget, donor.carrier_budget};
                    resolved = true;
                    for (unsigned k = 0; k < 3; ++k)
                    {
                        *output[k] = product(sums[k].get(), cube);
                        *errors[k] = product(sum_values(sums[k].error, product(256 * eps, sums[k].absolute)), cube);
                        const auto allowance = budgets[k];
                        const auto channel_resolved = std::isfinite(*output[k]) && std::isfinite(*errors[k]) &&
                                                      *output[k] >= 0 && *errors[k] <= allowance &&
                                                      (!positive[k] || *output[k] > *errors[k]);
                        resolved &= channel_resolved;
                        if (!channel_resolved)
                        {
                            std::ostringstream message;
                            message.precision(std::numeric_limits<long double>::max_digits10);
                            message << "donor=" << donor.id << " sign=" << sign << " retry=" << retry
                                    << " channel=" << k << " nominal=" << *output[k] << " estimate=" << *errors[k]
                                    << " allowance=" << allowance;
                            quality_detail = message.str();
                        }
                    }
                    group.bulk_positive = positive[0];
                    group.gas_positive = positive[1];
                    group.carrier_positive = positive[2];
                    const auto closure =
                        std::abs(sum_values(sum_values(group.nominal.gas, group.nominal.carrier), -group.nominal.bulk));
                    resolved &= closure <= donor.bulk_budget;
                    if (closure > donor.bulk_budget)
                    {
                        std::ostringstream message;
                        message.precision(std::numeric_limits<long double>::max_digits10);
                        message << "donor=" << donor.id << " sign=" << sign << " retry=" << retry
                                << " bulk closure=" << closure << " allowance=" << donor.bulk_budget;
                        quality_detail = message.str();
                    }
                    accepted = group;
                }
                catch (const GeometryError& error)
                {
                    if (error.code() != GeometryErrorCode::UnresolvedArithmetic)
                        throw;
                    resolved = false;
                    quality_detail = "donor=" + std::to_string(donor.id) + " sign=" + std::to_string(sign) +
                                     " retry=" + std::to_string(retry) + " cause=" + error.what();
                }
            }
            if (!resolved)
                throw GeometryError(GeometryErrorCode::UnresolvedArithmetic,
                    "Ordinary aggregate physical-phase quality remains unresolved after reconstruction: " +
                        quality_detail);
            if (accepted.bulk_positive)
                result.groups.push_back(accepted);
        }
    result.statistics = work.stats;
    return result;
}
} // namespace SimpleFluid::geometry::swept_volume
