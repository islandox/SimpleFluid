/** @file CellGeometryConstraints.hh
 * @brief Explicit original-face constraints over immutable cell views.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
namespace SimpleFluid::PLIC
{
template<ArithmeticMode Mode> struct CellGeometryFaceConstraint
{
    CellGeometryView<Mode> geometry;
    std::size_t face = 0;
};
template<ArithmeticMode Mode>
[[nodiscard]] Point<Mode> backtrace_constrained(
    const Point<Mode>&, GeometryPoint, double, std::span<const CellGeometryFaceConstraint<Mode>>, GeometryLimits = {})
    requires(Mode != ArithmeticMode::Exact);
extern template Point<ArithmeticMode::Ordinary> backtrace_constrained(const Point<ArithmeticMode::Ordinary>&,
    GeometryPoint, double, std::span<const CellGeometryFaceConstraint<ArithmeticMode::Ordinary>>, GeometryLimits);
extern template Point<ArithmeticMode::BoundedCertification> backtrace_constrained(
    const Point<ArithmeticMode::BoundedCertification>&, GeometryPoint, double,
    std::span<const CellGeometryFaceConstraint<ArithmeticMode::BoundedCertification>>, GeometryLimits);
} // namespace SimpleFluid::PLIC
