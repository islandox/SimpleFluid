/** @file cell_geometry_view_test_helpers.hh
 * @brief Explicit owning-shell access for geometry assertions and existing fixture inputs.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include <utility>
namespace SimpleFluid::PLIC::test
{
template<ArithmeticMode Mode> [[nodiscard]] CellGeometryView<Mode> view(ConvexGeometryPtr<Mode> geometry)
{ return CellGeometryView<Mode>::from_existing(std::move(geometry)); }
template<ArithmeticMode Mode> [[nodiscard]] CellGeometryView<Mode> view(CellGeometryView<Mode> geometry)
{ return geometry; }
template<ArithmeticMode Mode> [[nodiscard]] ConvexGeometryPtr<Mode> shell(const ConvexGeometryPtr<Mode>& geometry)
{ return geometry; }
template<ArithmeticMode Mode> [[nodiscard]] ConvexGeometryPtr<Mode> shell(const CellGeometryView<Mode>& geometry)
{
    if (!geometry) return {};
    if constexpr (Mode == ArithmeticMode::Exact) return geometry.existing_geometry();
    else
    {
        CellGeometryOperationWorkspace workspace;
        return materialize_cell(geometry, workspace);
    }
}
template<ArithmeticMode Mode> [[nodiscard]] std::pair<const void*,std::size_t> identity(const CellGeometryView<Mode>& geometry)
{
    if (!geometry) return {};
    if constexpr (Mode == ArithmeticMode::Ordinary)
        if (geometry.is_procedural())
        {
            const auto ref = details::NativeCellGeometrySourceAccess::source_cell(geometry);
            return {ref.source.get(),ref.cell_slot};
        }
    if constexpr (Mode == ArithmeticMode::Exact) return {geometry.existing_geometry().get(),0};
    else return {details::NativeCellGeometrySourceAccess::existing_geometry(geometry)->storage_identity(),0};
}
}
