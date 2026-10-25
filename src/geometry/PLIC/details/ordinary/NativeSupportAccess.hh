/** @file NativeSupportAccess.hh
 * @brief Private attachment seam for native provider construction ancestry.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/PlanarGeometry.hh"

namespace SimpleFluid::PLIC::details
{
/** Only native construction providers may attach identities established by
 * their immutable topology/recipe ancestry. These are not fitted plane IDs. */
class NativeSupportAccess
{
public:
    static ConvexGeometryPtr<ArithmeticMode::Ordinary> cell_supports(
        const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell, std::span<const std::vector<char>> supports)
    {
        return cell.with_native_supports(supports);
    }
    static PlanarFacePtr<ArithmeticMode::Ordinary> face_support(std::span<const std::array<long double, 3>> points,
        std::array<long double, 3> normal, std::vector<char> support)
    {
        return PlanarFace<ArithmeticMode::Ordinary>::from_retained_support(points, normal, std::move(support));
    }
};
} // namespace SimpleFluid::PLIC::details
