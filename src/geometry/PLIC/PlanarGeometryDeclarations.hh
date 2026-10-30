/** @file PlanarGeometryDeclarations.hh
 * @brief Mode-selected planar geometry and opaque topology identities.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/MeshUtils.hh"
#include "geometry/PLIC/ConvexCellGeometry.hh"
#include "geometry/PLIC/GeometryBackendTypes.hh"
#include "geometry/PLIC/GeometryMeasure.hh"
#include "geometry/PLIC/GeometryPacketCodec.hh"
namespace SimpleFluid::PLIC
{
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL PlaneSupportIdentity<Mode> planar_support(const PlanarFace<Mode>&);
template<ArithmeticMode Mode> [[nodiscard]] SIMPLEFLUID_LOCAL int planar_orientation(const PlanarFace<Mode>&);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<PlanarFacePtr<Mode>> subtract_planar_faces(
    const PlanarFace<Mode>&, std::span<const PlanarFacePtr<Mode>>);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL DisplacementIdentity<Mode> displacement(GeometryPoint, GeometryPoint);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL DisplacementIdentity<Mode> displacement_difference(
    const DisplacementIdentity<Mode>&, const DisplacementIdentity<Mode>&);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL GeometryPoint displacement_display(const DisplacementIdentity<Mode>&);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL PointIdentity<Mode> point_translated(
    const PointIdentity<Mode>&, const DisplacementIdentity<Mode>&);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL PointIdentity<Mode> periodic_point(
    const PointIdentity<Mode>&, GeometryPoint, GeometryPoint, std::array<bool, 3>);
template<ArithmeticMode Mode> [[nodiscard]] SIMPLEFLUID_LOCAL GeometryPoint point_display(const PointIdentity<Mode>&);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL std::array<GeometryValue<Mode>, 3> numerical_point(const PointIdentity<Mode>&);
template<ArithmeticMode Mode>
[[nodiscard]] SIMPLEFLUID_LOCAL GeometryMeasure<Mode> point_projection(
    const PointIdentity<Mode>&, GeometryPoint, GeometryPoint);
template<ArithmeticMode Mode> [[nodiscard]] std::vector<char> canonical_face_key(const PlanarFace<Mode>&);
/** Shared identity owners retained by this face. Exact faces retain their
 * arithmetic polygon directly and have no separately cached identity owners. */
template<ArithmeticMode Mode, class Visitor>
void visit_identity_allocations(const PlanarFace<Mode>& face, Visitor&& visitor)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        visit_identity_allocations(planar_support<Mode>(face), std::forward<Visitor>(visitor));
}
} // namespace SimpleFluid::PLIC
