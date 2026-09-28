/** @file PredicateSigns2D.hh
 * @brief Exact signs of planar predicates on stored binary64 coordinates.
 */
#pragma once

#include "geometry/MeshUtils.hh"

namespace SimpleFluid::Meshes::detail
{
enum class PredicateSign : signed char
{
    Negative = -1,
    Zero = 0,
    Positive = 1
};

/** Positive for counterclockwise order; zero only for exact collinearity. */
PredicateSign orient2d_sign(const MeshUtils::Vec3& a,
                            const MeshUtils::Vec3& b,
                            const MeshUtils::Vec3& c);

/** Positive inside a CCW circle; reversing its winding reverses the sign. */
PredicateSign incircle_sign(const MeshUtils::Vec3& a,
                            const MeshUtils::Vec3& b,
                            const MeshUtils::Vec3& c,
                            const MeshUtils::Vec3& p);

/** Exact winding of a simple polygon in its supplied order. */
PredicateSign polygon_area_sign(const Arr<MeshUtils::Vec3>& polygon);
} // namespace SimpleFluid::Meshes::detail
