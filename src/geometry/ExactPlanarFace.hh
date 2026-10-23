/** @file ExactPlanarFace.hh
 * @brief Authoritative immutable planar face certificates for nested native joins.
 */
#pragma once
#include "geometry/ConvexCellGeometry.hh"
#include "geometry/MeshUtils.hh"
namespace SimpleFluid::Meshes
{
class ExactPlanarFace;
class ExactGeometryMeasure;
using ExactPlanarFacePtr = std::shared_ptr<const ExactPlanarFace>;
/** Exact planar support and subtraction helpers; independent of AMR ancestry. */
[[nodiscard]] std::vector<char> exact_planar_support(const ExactPlanarFace&);
[[nodiscard]] int exact_planar_orientation(const ExactPlanarFace&);
[[nodiscard]] std::vector<ExactPlanarFacePtr> subtract_planar_faces(
    const ExactPlanarFace&, std::span<const ExactPlanarFacePtr>);
/** Sorted exact points on one nondegenerate line; no public rational types.
 * Query work is logarithmic plus returned segment points, never all-line fanout. */
[[nodiscard]] std::vector<char> exact_displacement(GeometryPoint from, GeometryPoint to);
[[nodiscard]] std::vector<char> exact_displacement_difference(
    std::span<const char> first, std::span<const char> second);
[[nodiscard]] GeometryPoint exact_displacement_display(std::span<const char> displacement);
[[nodiscard]] std::vector<char> exact_point_translated(std::span<const char> point, std::span<const char> displacement);
[[nodiscard]] std::vector<char> exact_periodic_point(
    std::span<const char> point, GeometryPoint lower, GeometryPoint upper, std::array<bool, 3> periodic);
[[nodiscard]] GeometryPoint exact_point_display(std::span<const char> identity);
/** Enclose a source authority point for a finite cache. Coordinate conversion
 * only; the returned values are not an exact construction certificate. */
[[nodiscard]] std::array<CertifiedGeometryValue, 3> numerical_point(std::span<const char> identity);
/** Exact signed normal dot (point-anchor), retaining rational support locations. */
[[nodiscard]] ExactGeometryMeasure exact_point_projection(
    std::span<const char> identity, GeometryPoint normal, GeometryPoint anchor);
class ExactEdgePointIndex
{
public:
    struct Impl;
    explicit ExactEdgePointIndex(std::span<const std::vector<char>> points);
    [[nodiscard]] std::vector<std::vector<char>> closed_segment(
        std::span<const char> first, std::span<const char> second) const;
    [[nodiscard]] std::size_t storage_bytes() const;

private:
    std::shared_ptr<const Impl> impl_;
};
class ExactPlanarFace
{
public:
    struct Impl;
    [[nodiscard]] static ExactPlanarFacePtr make(std::span<const GeometryPoint> vertices);
    [[nodiscard]] static ExactPlanarFacePtr from_native_face(const MultiRegionMesh&, std::uint64_t face);
    [[nodiscard]] CertifiedGeometryValue area() const;
    [[nodiscard]] MeshUtils::Vec3 centroid() const;
    [[nodiscard]] MeshUtils::Vec3 area_vector() const;
    /** Positive exact area ratio on one exact support plane; opposite winding
     * is allowed. Noncoplanar or degenerate support rejects, without sqrt. */
    [[nodiscard]] ExactGeometryMeasure coplanar_area_ratio(
        const ExactPlanarFace& denominator, std::size_t byte_budget = 64 * 1024 * 1024) const;
    /** Signed exact integral velocity dot n dA for represented velocity (m^3/s).
     * The opaque result may remain nonzero below the public scalar range. */
    [[nodiscard]] ExactGeometryMeasure area_vector_dot(
        GeometryPoint velocity, std::size_t byte_budget = 64 * 1024 * 1024) const;
    [[nodiscard]] ExactPlanarFacePtr reversed() const;
    [[nodiscard]] ExactPlanarFacePtr translated(GeometryPoint displacement) const;
    [[nodiscard]] ExactPlanarFacePtr translated_exact(std::span<const char> displacement) const;
    [[nodiscard]] ExactPlanarFacePtr oriented_like(GeometryPoint area_vector) const;
    [[nodiscard]] ExactPlanarFacePtr intersection(const ExactPlanarFace&) const;
    /** Lossless exact point and infinite-line certificates for sparse edge
     * authority. Identities do not depend on loop start or collinear inserts. */
    [[nodiscard]] std::vector<std::vector<char>> vertex_identities() const;
    [[nodiscard]] std::vector<std::vector<char>> edge_support_identities() const;
    [[nodiscard]] ExactPlanarFacePtr with_edge_vertices(std::span<const std::vector<char>> exact_points) const;
    [[nodiscard]] bool equivalent(const ExactPlanarFace&) const;
    [[nodiscard]] bool contains_exact_point(std::span<const char> identity) const;
    [[nodiscard]] bool cell_bounds_overlap(const ConvexCellGeometry&) const;
    [[nodiscard]] bool contains(const ExactPlanarFace&) const;
    [[nodiscard]] bool oppositely_oriented(const ExactPlanarFace&) const;
    [[nodiscard]] static ConvexGeometryPtr cell_shell(std::span<const ExactPlanarFacePtr> outward_faces);
    /** Opposite winding is accepted; every tile must be whole and contained.
     * Exact coplanarity, disjointness, area and first moments are mandatory. */
    void certify_partition(std::span<const ExactPlanarFacePtr> tiles) const;
    [[nodiscard]] std::vector<char> encode() const;
    [[nodiscard]] std::vector<char> encode_incidence() const;
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] static ExactPlanarFacePtr decode(std::span<const char>);

private:
    friend std::vector<char> exact_planar_support(const ExactPlanarFace&);
    friend int exact_planar_orientation(const ExactPlanarFace&);
    friend std::vector<ExactPlanarFacePtr> subtract_planar_faces(
        const ExactPlanarFace&, std::span<const ExactPlanarFacePtr>);
    friend class ConvexCellGeometry;
    friend class ExactSweepAccess;
    friend class ExactExtrudedConstruction;
    friend class ExactPolyhedralComplex;
    explicit ExactPlanarFace(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::Meshes
