/** @file plic_analytic_test_geometry.hh
 * @brief Independent analytic source certificates for finite-kernel unit tests.
 */
#pragma once
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include <bit>

namespace plic_test
{
/** Each input coordinate is a represented binary64 value, converted exactly to
 * the wider test scalar. A vertex lies by construction on three coordinate
 * planes. Keys name those analytic equations directly; they are never copied
 * from Ordinary construction lineage or recovered from rounded intersections.
 * The selected geometry kernel still validates the supplied oriented shell. */
template<SimpleFluid::ArithmeticMode Mode>
SimpleFluid::PLIC::ConvexGeometryPtr<Mode> axis_box(
    SimpleFluid::PLIC::GeometryPoint lower, SimpleFluid::PLIC::GeometryPoint upper)
{
    namespace PLIC = SimpleFluid::PLIC;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) || !(lower[axis] < upper[axis]))
            throw std::invalid_argument("Analytic box fixture requires finite positive spans.");
    std::array<PLIC::Point<Mode>, 8> vertices;
    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex)
    {
        PLIC::DynamicPoint point;
        point.arithmetic_mode = Mode;
        std::vector<PLIC::PlaneAuthority> supports;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const double coordinate = (vertex >> axis) & 1 ? upper[axis] : lower[axis];
            point.value[axis] = coordinate;
            supports.push_back(PLIC::PlaneAuthority::opaque({0x706c69635f746573ULL, axis,
                std::bit_cast<std::uint64_t>(coordinate == 0 ? 0. : coordinate)}));
        }
        PLIC::DynamicGeometryAccess::seal(point, std::move(supports));
        vertices[vertex] = PLIC::Point<Mode>(point);
    }
    return PLIC::ConvexCellGeometry<Mode>::make_points(vertices,
        {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}});
}
/** Unit coordinate tetrahedron with independently known incidences x=0, y=0,
 * z=0, and x+y+z=1. Its vertices and plane equations are exactly represented. */
template<SimpleFluid::ArithmeticMode Mode>
SimpleFluid::PLIC::ConvexGeometryPtr<Mode> unit_tetrahedron()
{
    namespace PLIC = SimpleFluid::PLIC;
    const std::array<std::array<long double, 3>, 4> coordinates{
        std::array<long double, 3>{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    std::array<PLIC::Point<Mode>, 4> vertices;
    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex)
    {
        PLIC::DynamicPoint point(coordinates[vertex], {}, Mode);
        std::vector<PLIC::PlaneAuthority> supports;
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (point.value[axis] == 0)
                supports.push_back(PLIC::PlaneAuthority::opaque({0x706c69635f746573ULL, axis, 0}));
        if (point.value[0] + point.value[1] + point.value[2] == 1)
            supports.push_back(PLIC::PlaneAuthority::opaque({0x706c69635f746573ULL, 3, 1}));
        PLIC::DynamicGeometryAccess::seal(point, std::move(supports));
        vertices[vertex] = PLIC::Point<Mode>(point);
    }
    return PLIC::ConvexCellGeometry<Mode>::make_points(vertices, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
}
} // namespace plic_test
