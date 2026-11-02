/** @file LocalCoordinates.hh
 * @brief Backend-neutral synchronous numerical coordinates relative to origin().
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include <memory>
#include <type_traits>

namespace SimpleFluid::PLIC::details
{
/** Visits original vertex order under one arithmetic operation. Coordinates
 * are stack values valid only during the callback; callers must not retain
 * references. No point identity, export snapshot, or coordinate array is built.
 * Arithmetic and the caller floating-point environment match local_points(). */
template<ArithmeticMode Mode>
SIMPLEFLUID_LOCAL void visit_local_coordinates(const ConvexCellGeometry<Mode>&, void*,
    void (*)(void*, std::size_t, const PointCoordinates<Mode>&));
template<ArithmeticMode Mode, class Visitor>
void visit_local_coordinates(const ConvexCellGeometry<Mode>& geometry, Visitor&& visitor)
{
    using Callback = std::remove_reference_t<Visitor>;
    visit_local_coordinates<Mode>(geometry, const_cast<void*>(static_cast<const void*>(std::addressof(visitor))),
        [](void* context, std::size_t index, const PointCoordinates<Mode>& coordinates)
        { (*static_cast<Callback*>(context))(index, coordinates); });
}
/** Invocation-owned numeric table for consumers requiring indexed original
 * vertices. Its actual capacity uses the existing geometry operation budget. */
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<PointCoordinates<Mode>> materialize_local_coordinates(
    const ConvexCellGeometry<Mode>&);
extern template std::vector<PointCoordinates<ArithmeticMode::Ordinary>> materialize_local_coordinates(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>&);
extern template std::vector<PointCoordinates<ArithmeticMode::BoundedCertification>> materialize_local_coordinates(
    const ConvexCellGeometry<ArithmeticMode::BoundedCertification>&);
extern template void visit_local_coordinates<ArithmeticMode::Ordinary>(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>&, void*,
    void (*)(void*, std::size_t, const PointCoordinates<ArithmeticMode::Ordinary>&));
extern template void visit_local_coordinates<ArithmeticMode::BoundedCertification>(
    const ConvexCellGeometry<ArithmeticMode::BoundedCertification>&, void*,
    void (*)(void*, std::size_t, const PointCoordinates<ArithmeticMode::BoundedCertification>&));
} // namespace SimpleFluid::PLIC::details
