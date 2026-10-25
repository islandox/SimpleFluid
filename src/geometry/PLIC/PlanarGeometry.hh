/** @file PlanarGeometry.hh
 * @brief Mode-selected planar geometry and opaque topology identities.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/MeshUtils.hh"
#include "geometry/PLIC/ConvexCellGeometry.hh"
#include "geometry/PLIC/GeometryBackendTypes.hh"
#include "geometry/PLIC/GeometryMeasure.hh"
namespace SimpleFluid::PLIC
{
template<ArithmeticMode Mode> [[nodiscard]] SIMPLEFLUID_LOCAL std::vector<char> planar_support(const PlanarFace<Mode>&);
template<ArithmeticMode Mode> [[nodiscard]] SIMPLEFLUID_LOCAL int planar_orientation(const PlanarFace<Mode>&);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<PlanarFacePtr<Mode>> subtract_planar_faces(
    const PlanarFace<Mode>&, std::span<const PlanarFacePtr<Mode>>);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<char> displacement(GeometryPoint, GeometryPoint);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<char> displacement_difference(std::span<const char>, std::span<const char>);
template<ArithmeticMode Mode> [[nodiscard]] SIMPLEFLUID_LOCAL GeometryPoint displacement_display(std::span<const char>);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<char> point_translated(std::span<const char>, std::span<const char>);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<char> periodic_point(
    std::span<const char>, GeometryPoint, GeometryPoint, std::array<bool, 3>);
template<ArithmeticMode Mode> [[nodiscard]] SIMPLEFLUID_LOCAL GeometryPoint point_display(std::span<const char>);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL std::array<GeometryValue<Mode>, 3> numerical_point(std::span<const char>);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL GeometryMeasure<Mode> point_projection(
    std::span<const char>, GeometryPoint, GeometryPoint);
} // namespace SimpleFluid::PLIC
#include "geometry/PLIC/details/exact/PlanarGeometry.hh"
#include "geometry/PLIC/details/ordinary/PlanarGeometry.hh"
