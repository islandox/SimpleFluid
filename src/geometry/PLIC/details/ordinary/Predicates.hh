/** @file Predicates.hh
 * @brief Fixed-work compensated predicates on represented Ordinary coordinates.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include <array>
#include <span>

namespace SimpleFluid::PLIC::details
{
namespace ordinary_predicates SIMPLEFLUID_LOCAL
{
using Point = std::array<long double, 3>;
struct FramedPoint
{
    Point local{};
    std::array<double, 3> origin{};
    int exponent = 0;
};
enum class Sign
{
    Negative = -1,
    Zero = 0,
    Positive = 1,
    Unresolved = 2
};
/** Bounds only evaluation error of the supplied represented inputs. It does
 * not replace construction incidence or certify their geometric provenance.
 * Zero requires an exactly zero compensated result and zero discarded error.
 * Evaluation privately selects round-to-nearest and restores the caller's
 * thread-local floating-point environment on return or exception.
 */
struct Result
{
    long double value = 0, absolute_error = 0;
    Sign sign = Sign::Zero;
};
/** Preserve every signed contribution, including unresolved individual terms.
 * Only result() classifies the compensated total. Fixed stack storage contains
 * a twofold sum and its discarded/input error, never coordinate intervals. */
class ResultAccumulator
{
public:
    void add(const Result&);
    [[nodiscard]] Result result(int binary_scale = 0) const;

private:
    long double high_ = 0, low_ = 0, error_ = 0;
};
/** Sum a closed sequence of already computed contributions in input order.
 * Validation and addition are interleaved under one protected FP environment;
 * only the compensated total is classified. No caller arithmetic runs here. */
[[nodiscard]] Result sum_results(std::span<const Result>, int binary_scale = 0);
/** det(b-a,c-a,d-a). Uses one deterministic minimax-edge anchor, with
 * coordinate-lexicographic ties; input permutation parity is preserved. */
[[nodiscard]] Result orientation(const std::array<Point, 4>&);
/** Cyclic projection (drop+1, drop+2), hence the corresponding cross component. */
[[nodiscard]] Result projected_orientation(const Point&, const Point&, const Point&, unsigned drop);
/** normal . (point-anchor) - offset, evaluated without a rounded displacement. */
[[nodiscard]] Result plane_distance(const Point& point, const Point& normal, const Point& anchor, long double offset);
/** Evaluate in the anchor frame, retaining subtraction/scaling error before
 * any world-coordinate rounding. Binary64 origins are original input leaves.
 * Construction error of a supplied plane normal remains separate evidence. */
[[nodiscard]] Result plane_distance(
    const FramedPoint& point, const FramedPoint& anchor, const Point& normal, long double offset);
/** Original support expression in a source tetrahedron's barycentric chart.
 * source_face < 4 denotes lambda[source_face]; otherwise the framed plane is
 * normal.(point-anchor)-offset, in anchor-frame units. source_incidence is a
 * mask of independently proven original source contacts, not rounded zeros.
 * Orientation reverses the expression without changing support identity. */
struct SupportRow
{
    unsigned source_face = 4;
    FramedPoint anchor{};
    Point normal{};
    long double offset = 0;
    unsigned source_incidence = 0;
    int orientation = 1;
};
struct SupportEvaluation
{
    Result distance{};
    bool singular = false;
};
/** Bounded indirect predicate for three defining supports and an original
 * query. Coefficients and determinant signs are evaluated from input leaves,
 * with error-free floating expansions capped at 2048 components each. The
 * metric is the actual query distance, with a residual-derived absolute error.
 * An exactly singular basis is reported separately so another can be tried.
 * Unsupported finite range/underflow rejects with UnresolvedArithmetic;
 * expansion capacity, allocation and scalar work obey ResourceLimit guards.
 * No rational geometry or arithmetic-policy fallback is invoked. */
[[nodiscard]] SupportEvaluation support_intersection_distance(
    const std::array<FramedPoint, 4>& source, const std::array<SupportRow, 3>& basis, const SupportRow& query);
/** Require a resolved sign without reinterpreting an uncertain zero as contact. */
[[nodiscard]] int require_sign(const Result&);
} // namespace ordinary_predicates SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
