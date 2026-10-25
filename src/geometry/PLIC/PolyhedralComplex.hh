/** @file PolyhedralComplex.hh
 * @brief Mode-selected native atomic-face complexes and shared ancestry.
 */
#pragma once
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/mesh/AMR/AdaptiveHierarchy.hh"
#include <string>
namespace SimpleFluid::PLIC
{
struct PolyhedralSurfaceKey
{
    enum class Kind : std::uint64_t
    {
        Root,
        Cut
    };
    Kind kind = Kind::Root;
    std::uint64_t region = 0;
    ::SimpleFluid::Meshes::AMR::ConvexCellAncestry creator;
    std::uint64_t ordinal = 0;
    auto operator<=>(const PolyhedralSurfaceKey&) const = default;
};
struct PolyhedralSurfaceUse
{
    PolyhedralSurfaceKey key;
    int orientation = 1;
    bool exterior = true;
    /** Empty is a valid unnamed physical exterior; interior uses stay empty. */
    std::string boundary;
};
template<ArithmeticMode Mode> struct PolyhedralComplexCell
{
    ::SimpleFluid::Meshes::AMR::AdaptiveCellKey key;
    std::uint64_t gid = 0;
    ConvexGeometryPtr<Mode> geometry;
    std::vector<PolyhedralSurfaceUse> surfaces;
};

} // namespace SimpleFluid::PLIC
#include "geometry/PLIC/details/exact/PolyhedralComplex.hh"
#include "geometry/PLIC/details/ordinary/PolyhedralComplex.hh"
