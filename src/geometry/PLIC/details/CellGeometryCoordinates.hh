/** @file CellGeometryCoordinates.hh
 * @brief Exact existing local-coordinate expression for owning geometry views.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include <memory>
#include <type_traits>
namespace SimpleFluid::PLIC::details
{
template<ArithmeticMode M>
    requires(M != ArithmeticMode::Exact)
SIMPLEFLUID_LOCAL void visit_local_coordinates(
    const CellGeometryView<M>&, void*, void (*)(void*, std::size_t, const PointCoordinates<M>&));
template<ArithmeticMode M, class Visitor>
    requires(M != ArithmeticMode::Exact)
void visit_local_coordinates(const CellGeometryView<M>& view, Visitor&& visitor)
{
    using Callback = std::remove_reference_t<Visitor>;
    visit_local_coordinates<M>(view, const_cast<void*>(static_cast<const void*>(std::addressof(visitor))),
        [](void* context, std::size_t index, const PointCoordinates<M>& coordinates)
        { (*static_cast<Callback*>(context))(index, coordinates); });
}
template<ArithmeticMode M>
    requires(M != ArithmeticMode::Exact)
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<PointCoordinates<M>> materialize_local_coordinates(
    const CellGeometryView<M>&);
/** Count numeric coordinate buffers and arithmetic in the same remaining
 * cumulative workspace as materialized shells; caller owns the returned vector. */
template<ArithmeticMode M>
    requires(M != ArithmeticMode::Exact)
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<PointCoordinates<M>> materialize_local_coordinates(
    const CellGeometryView<M>&, CellGeometryOperationWorkspace&);
extern template void visit_local_coordinates<ArithmeticMode::Ordinary>(
    const CellGeometryView<ArithmeticMode::Ordinary>&, void*,
    void (*)(void*, std::size_t, const PointCoordinates<ArithmeticMode::Ordinary>&));
extern template void visit_local_coordinates<ArithmeticMode::BoundedCertification>(
    const CellGeometryView<ArithmeticMode::BoundedCertification>&, void*,
    void (*)(void*, std::size_t, const PointCoordinates<ArithmeticMode::BoundedCertification>&));
extern template std::vector<PointCoordinates<ArithmeticMode::Ordinary>> materialize_local_coordinates(
    const CellGeometryView<ArithmeticMode::Ordinary>&);
extern template std::vector<PointCoordinates<ArithmeticMode::BoundedCertification>> materialize_local_coordinates(
    const CellGeometryView<ArithmeticMode::BoundedCertification>&);
extern template std::vector<PointCoordinates<ArithmeticMode::Ordinary>> materialize_local_coordinates(
    const CellGeometryView<ArithmeticMode::Ordinary>&, CellGeometryOperationWorkspace&);
extern template std::vector<PointCoordinates<ArithmeticMode::BoundedCertification>> materialize_local_coordinates(
    const CellGeometryView<ArithmeticMode::BoundedCertification>&, CellGeometryOperationWorkspace&);
} // namespace SimpleFluid::PLIC::details
