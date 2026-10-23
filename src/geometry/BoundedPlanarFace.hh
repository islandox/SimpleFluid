/** @file BoundedPlanarFace.hh
 * @brief Authenticated bounded projected overlap of coplanar native face loops.
 */
#pragma once
#include "geometry/BoundedConvexGeometry.hh"
namespace SimpleFluid::Meshes
{
struct BoundedPlanarFaceOverlap
{
    BoundedGeometryValue first_projected_area, second_projected_area, overlap_projected_area;
    /** Signed incidence transfer ratios. Each magnitude is overlap/its named
     * input area; sign changes when input loop windings are opposite. */
    BoundedGeometryValue overlap_over_first, overlap_over_second;
    /** Positive-area intersection loop, same winding as first. Sealed
     * provenance authenticates the original supports of this FP construction. */
    std::vector<BoundedPoint> overlap_vertices;
    int orientation = 0;
    unsigned dropped_axis = 0;
};
/** Both convex loops must retain sealed original support evidence for support.
 * A common projection cancels the normal metric in both area ratios. Transient
 * 2D clipping retains authenticated source edge incidences; ambiguous signs
 * reject, with no rational construction or measure fallback. Empty/contact
 * overlap is exactly zero only when topology or shared line authority proves it.
 * Returned area enclosures have physical area units; the caller applies its
 * unchanged physical accuracy budget and cumulative handoff resource caps. */
[[nodiscard]] BoundedPlanarFaceOverlap bounded_planar_face_overlap(std::span<const BoundedPoint> first,
    std::span<const BoundedPoint> second, const BoundedPlaneAuthority& support, BoundedGeometryLimits = {});
} // namespace SimpleFluid::Meshes
