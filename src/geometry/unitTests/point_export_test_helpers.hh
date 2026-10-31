/** @file point_export_test_helpers.hh
 * @brief Explicitly budgeted materialization for legacy finite geometry tests.
 */
#pragma once
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include <vector>

namespace SimpleFluid::PLIC::test
{
inline constexpr std::size_t point_export_test_budget = 64 * 1024 * 1024;
inline std::vector<DynamicPoint> exported_points(const DynamicConvexGeometry& geometry)
{
    GeometryAllocationLedger ledger;
    std::size_t used = 0;
    const auto publication = geometry.export_points(ledger, used, point_export_test_budget);
    const auto points = publication.points();
    return {points.begin(), points.end()};
}
template<ArithmeticMode Mode> std::vector<Point<Mode>> exported_points(const ConvexCellGeometry<Mode>& geometry)
{
    GeometryAllocationLedger ledger;
    std::size_t used = 0;
    const auto publication = geometry.export_points(ledger, used, point_export_test_budget);
    const auto points = publication.points();
    return {points.begin(), points.end()};
}
} // namespace SimpleFluid::PLIC::test
