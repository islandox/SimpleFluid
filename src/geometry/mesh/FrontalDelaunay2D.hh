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
 * the annular overload preserves two supplied polygon loops for conforming
 * boundary-layer attachment. The disk overload additionally accepts
 * prescribed radial fronts so a cylinder can retain radial layer spacing.
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
     * @brief Mesh the region between two fixed convex polygon loops.
     *
     * Both loops must be strictly convex, counter-clockwise, and supplied
     * without repeating their first vertex.  The inner loop must lie strictly
     * inside the outer loop.  No boundary point is moved or added: this permits
     * one-to-one attachment to quadrilateral boundary layers.  Output nodes
     * begin with the outer loop, followed by the inner loop, followed by the
     * advancing-front interior points.  Their Z coordinates are zero.
     *
     * Bowyer-Watson triangulation, fixed-segment recovery, and unconstrained
     * edge legalization produce a constrained Delaunay triangulation.  Boundary
     * segments are oriented with the domain on their left: outer CCW, inner CW.
     * The requested spacing controls interior point placement; the supplied
     * boundary segment lengths remain unchanged.
     *
     * @throws std::invalid_argument If loops, spacing, or names are invalid.
     * @throws std::overflow_error If generated node IDs exceed their type.
     * @throws std::runtime_error If triangulation invariants cannot be satisfied.
     */
    static Result triangulate_annulus(
        const Arr<Vec3>& outer_boundary,
        const Arr<Vec3>& inner_boundary,
        real_t target_edge_length,
        const std::string& outer_name = "outer",
        const std::string& inner_name = "inner");

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
