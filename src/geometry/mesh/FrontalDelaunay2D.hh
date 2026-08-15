/**
 * @file FrontalDelaunay2D.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Frontal point placement and Delaunay triangulation in the XY plane.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
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

    /** @brief Oriented boundary segment and its physical batch name. */
    struct BoundaryEdge
    {
        unsigned node0{};
        unsigned node1{};
        std::string batch_name = "side";
    };

    /** @brief Nodes, triangles, and boundary segments produced by meshing. */
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
     *
     * @param boundary Convex counter-clockwise polygon vertices.
     * @param target_edge_length Requested mesh spacing.
     * @param boundary_name Batch name assigned to polygon boundary segments.
     * @return Triangular mesh and boundary connectivity.
     * @throws std::invalid_argument If the boundary, spacing, or name is invalid.
     * @throws std::overflow_error If generated connectivity exceeds its ID type.
     * @throws std::runtime_error If triangulation invariants cannot be satisfied.
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
     *
     * @param radial_edges Strictly increasing radii beginning at zero.
     * @param target_edge_length Requested circumferential edge length.
     * @param boundary_name Batch name assigned to the outer circle.
     * @return Triangular disk mesh and boundary connectivity.
     * @throws std::invalid_argument If the radii, spacing, or name is invalid.
     * @throws std::overflow_error If generated connectivity exceeds its ID type.
     * @throws std::runtime_error If triangulation invariants cannot be satisfied.
     */
    static Result triangulate_disk(
        const ArrReal& radial_edges,
        real_t target_edge_length,
        const std::string& boundary_name = "radial");
};

} // namespace SimpleFluid::Meshes
