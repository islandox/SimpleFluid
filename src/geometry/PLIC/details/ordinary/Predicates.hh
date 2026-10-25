/** @file Predicates.hh
 * @brief Fixed-work compensated predicates on represented Ordinary coordinates.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include <array>

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
/** Require a resolved sign without reinterpreting an uncertain zero as contact. */
[[nodiscard]] int require_sign(const Result&);
} // namespace ordinary_predicates SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
