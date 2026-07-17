/**
 * @file FrontalDelaunay2D.hh
 * @brief Frontal point placement and Delaunay triangulation in the XY plane.
 */

#pragma once

#include "geometry/MeshUtils.hh"

#include <array>
#include <string>

namespace SimpleFluid::Meshes
{

/**
 * @brief Generate triangular XY meshes with advancing-front point placement.
 *
 * Boundary points are fixed first.  Interior points advance inward from the
 * boundary at approximately the requested edge length, after which a
 * Bowyer-Watson Delaunay triangulation supplies the connectivity.  The
 * polygon overload currently accepts convex, counter-clockwise boundaries;
 * the disk overload additionally accepts prescribed radial fronts so a
 * cylinder can retain radial boundary-layer spacing.
 */
class FrontalDelaunay2D
{
public:
    using Vec3 = MeshUtils::Vec3;
    using Triangle = std::array<unsigned, 3>;

    struct BoundaryEdge
    {
        unsigned node0{};
        unsigned node1{};
        std::string batch_name = "side";
    };

    struct Result
    {
        Arr<Vec3> nodes;
        Arr<Triangle> triangles;
        Arr<BoundaryEdge> boundary_edges;
    };

    /**
     * @brief Mesh a convex, counter-clockwise XY polygon.
     *
     * Polygon edges are subdivided to respect @p target_edge_length before
     * the inward front is advanced.  Z coordinates in @p boundary are
     * ignored and generated nodes lie on z=0.
     */
    static Result triangulate(
        const Arr<Vec3>& boundary,
        real_t target_edge_length,
        const std::string& boundary_name = "side");

    /**
     * @brief Mesh a disk using circular fronts at prescribed radii.
     *
     * @p radial_edges must start at zero and increase strictly.  Its final
     * value is the disk radius.  Each nonzero radius becomes a point-placement
     * front, while the Delaunay step determines triangle connectivity.
     */
    static Result triangulate_disk(
        const ArrReal& radial_edges,
        real_t target_edge_length,
        const std::string& boundary_name = "radial");
};

} // namespace SimpleFluid::Meshes
