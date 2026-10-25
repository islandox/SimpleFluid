/** @file FiniteGeometryAccess.hh
 * @brief Static geometry access and explicit publication at runtime VoF boundaries.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include <map>
namespace SimpleFluid::VoF::finite_detail
{
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL PLIC::DynamicGeometryValue runtime_value(PLIC::GeometryValue<Mode> value)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        return {value.value, 0, Mode};
    else
        return {value.value, value.absolute_error, Mode};
}
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL PLIC::DynamicPhaseMeasures runtime_phases(const PLIC::PhaseMeasures<Mode>& value)
{
    return {runtime_value<Mode>(value.bulk), runtime_value<Mode>(value.gas), runtime_value<Mode>(value.carrier)};
}
template<ArithmeticMode Mode> struct SIMPLEFLUID_LOCAL PointArithmetic
{
    using Point = PLIC::Point<Mode>;
    using Math = Arithmetic<Mode>;
    static Point add(const Point& a, const Point& b) { return Point(Math::add(a.dynamic(), b.dynamic())); }
    static Point subtract(const Point& a, const Point& b) { return Point(Math::subtract(a.dynamic(), b.dynamic())); }
    static Point cross(const Point& a, const Point& b) { return Point(Math::cross(a.dynamic(), b.dynamic())); }
    static Point scale(const Point& a, Value factor) { return Point(Math::scale(a.dynamic(), factor)); }
    static Value dot(const Point& a, const Point& b) { return Math::dot(a.dynamic(), b.dynamic()); }
};
} // namespace SimpleFluid::VoF::finite_detail
