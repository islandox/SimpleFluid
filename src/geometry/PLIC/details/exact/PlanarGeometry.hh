#pragma once
#include "geometry/PLIC/PlanarGeometryDeclarations.hh"
namespace SimpleFluid::PLIC
{
namespace MeshUtils = ::SimpleFluid::MeshUtils;
/** Exact planar support and subtraction helpers; independent of AMR ancestry. */
template<>
[[nodiscard]] SIMPLEFLUID_LOCAL PlaneSupportIdentity<ArithmeticMode::Exact> planar_support<ArithmeticMode::Exact>(
    const PlanarFace<ArithmeticMode::Exact>&);
template<>
[[nodiscard]] SIMPLEFLUID_LOCAL int planar_orientation<ArithmeticMode::Exact>(const PlanarFace<ArithmeticMode::Exact>&);
template<>
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<PlanarFacePtr<ArithmeticMode::Exact>>
subtract_planar_faces<ArithmeticMode::Exact>(
    const PlanarFace<ArithmeticMode::Exact>&, std::span<const PlanarFacePtr<ArithmeticMode::Exact>>);

template<>
[[nodiscard]] std::vector<char> canonical_face_key<ArithmeticMode::Exact>(const PlanarFace<ArithmeticMode::Exact>&);
/** Sorted exact points on one nondegenerate line; no public rational types.
 * Query work is logarithmic plus returned segment points, never all-line fanout. */
template<>
[[nodiscard]] SIMPLEFLUID_LOCAL DisplacementIdentity<ArithmeticMode::Exact> displacement<ArithmeticMode::Exact>(
    GeometryPoint from, GeometryPoint to);

template<>
[[nodiscard]] SIMPLEFLUID_LOCAL DisplacementIdentity<ArithmeticMode::Exact>
displacement_difference<ArithmeticMode::Exact>(const DisplacementIdentity<ArithmeticMode::Exact>& first,
    const DisplacementIdentity<ArithmeticMode::Exact>& second);
template<>
[[nodiscard]] SIMPLEFLUID_LOCAL GeometryPoint displacement_display<ArithmeticMode::Exact>(
    const DisplacementIdentity<ArithmeticMode::Exact>& displacement);
template<>
[[nodiscard]] SIMPLEFLUID_LOCAL PointIdentity<ArithmeticMode::Exact> point_translated<ArithmeticMode::Exact>(
    const PointIdentity<ArithmeticMode::Exact>& point, const DisplacementIdentity<ArithmeticMode::Exact>& displacement);
template<>
[[nodiscard]] SIMPLEFLUID_LOCAL PointIdentity<ArithmeticMode::Exact> periodic_point<ArithmeticMode::Exact>(
    const PointIdentity<ArithmeticMode::Exact>& point, GeometryPoint lower, GeometryPoint upper,
    std::array<bool, 3> periodic);
template<>
[[nodiscard]] SIMPLEFLUID_LOCAL GeometryPoint point_display<ArithmeticMode::Exact>(
    const PointIdentity<ArithmeticMode::Exact>& identity);

/** Enclose a source authority point for a finite cache. Coordinate conversion
 * only; the returned values are not an exact construction certificate. */
template<>
[[nodiscard]] SIMPLEFLUID_LOCAL std::array<GeometryValue<ArithmeticMode::Exact>, 3>
numerical_point<ArithmeticMode::Exact>(const PointIdentity<ArithmeticMode::Exact>& identity);

/** Exact signed normal dot (point-anchor), retaining rational support locations. */
template<>
[[nodiscard]] SIMPLEFLUID_LOCAL GeometryMeasure<ArithmeticMode::Exact> point_projection<ArithmeticMode::Exact>(
    const PointIdentity<ArithmeticMode::Exact>& identity, GeometryPoint normal, GeometryPoint anchor);
template<> class EdgePointIndex<ArithmeticMode::Exact>
{
public:
    struct Impl;
    explicit EdgePointIndex(std::span<const PointIdentity<ArithmeticMode::Exact>> points);
    [[nodiscard]] std::vector<PointIdentity<ArithmeticMode::Exact>> closed_segment(
        const PointIdentity<ArithmeticMode::Exact>& first, const PointIdentity<ArithmeticMode::Exact>& second) const;
    [[nodiscard]] std::size_t storage_bytes() const;

private:
    std::shared_ptr<const Impl> impl_;
};
template<> class PlanarFace<ArithmeticMode::Exact>
{
public:
    struct Impl;
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Exact> make(std::span<const GeometryPoint> vertices);
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Exact> from_geometry(
        const PlanarFace<ArithmeticMode::Ordinary>&);
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Exact> from_native_face(
        const MultiRegionMesh&, std::uint64_t face);
    [[nodiscard]] GeometryValue<ArithmeticMode::Exact> area() const;
    [[nodiscard]] MeshUtils::Vec3 centroid() const;
    [[nodiscard]] MeshUtils::Vec3 area_vector() const;
    /** Positive exact area ratio on one exact support plane; opposite winding
     * is allowed. Noncoplanar or degenerate support rejects, without sqrt. */
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Exact> coplanar_area_ratio(
        const PlanarFace<ArithmeticMode::Exact>& denominator, std::size_t byte_budget = 64 * 1024 * 1024) const;
    /** Signed exact integral velocity dot n dA for represented velocity (m^3/s).
     * The opaque result may remain nonzero below the public scalar range. */
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Exact> area_vector_dot(
        GeometryPoint velocity, std::size_t byte_budget = 64 * 1024 * 1024) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Exact> reversed() const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Exact> translated(GeometryPoint displacement) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Exact> translated(
        const DisplacementIdentity<ArithmeticMode::Exact>& displacement) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Exact> oriented_like(GeometryPoint area_vector) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Exact> intersection(const PlanarFace<ArithmeticMode::Exact>&) const;
    /** Lossless exact point and infinite-line certificates for sparse edge
     * authority. Identities do not depend on loop start or collinear inserts. */
    [[nodiscard]] std::vector<PointIdentity<ArithmeticMode::Exact>> vertex_identities() const;
    [[nodiscard]] std::vector<LineIdentity<ArithmeticMode::Exact>> edge_support_identities() const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Exact> with_edge_vertices(
        std::span<const PointIdentity<ArithmeticMode::Exact>> exact_points) const;
    [[nodiscard]] bool equivalent(const PlanarFace<ArithmeticMode::Exact>&) const;
    [[nodiscard]] bool contains_point(const PointIdentity<ArithmeticMode::Exact>& identity) const;
    [[nodiscard]] bool cell_bounds_overlap(const ConvexCellGeometry<ArithmeticMode::Exact>&) const;
    [[nodiscard]] bool contains(const PlanarFace<ArithmeticMode::Exact>&) const;
    [[nodiscard]] bool oppositely_oriented(const PlanarFace<ArithmeticMode::Exact>&) const;
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Exact> cell_shell(
        std::span<const PlanarFacePtr<ArithmeticMode::Exact>> outward_faces);
    /** Opposite winding is accepted; every tile must be whole and contained.
     * Exact coplanarity, disjointness, area and first moments are mandatory. */
    void certify_partition(std::span<const PlanarFacePtr<ArithmeticMode::Exact>> tiles) const;
    [[nodiscard]] std::vector<char> encode(std::size_t byte_budget = geometry_packet_default_budget) const;
    [[nodiscard]] std::vector<char> encode_incidence(std::size_t byte_budget = geometry_packet_default_budget) const;
    /** Own face buffers only; shared identity owners are visited separately. */
    [[nodiscard]] std::size_t exclusive_storage_bytes() const;
    /** Full standalone footprint, including retained identity ancestry. */
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Exact> decode(
        std::span<const char>, std::size_t byte_budget = geometry_packet_default_budget);

private:
    friend std::vector<char> canonical_face_key<ArithmeticMode::Exact>(const PlanarFace&);
    friend PlaneSupportIdentity<ArithmeticMode::Exact> planar_support<ArithmeticMode::Exact>(
        const PlanarFace<ArithmeticMode::Exact>&);
    friend int planar_orientation<ArithmeticMode::Exact>(const PlanarFace<ArithmeticMode::Exact>&);
    friend std::vector<PlanarFacePtr<ArithmeticMode::Exact>> subtract_planar_faces<ArithmeticMode::Exact>(
        const PlanarFace<ArithmeticMode::Exact>&, std::span<const PlanarFacePtr<ArithmeticMode::Exact>>);
    friend class ConvexCellGeometry<ArithmeticMode::Exact>;
    friend class ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>;
    friend class ExtrudedConstruction<ArithmeticMode::Exact>;
    friend class PolyhedralComplex<ArithmeticMode::Exact>;
    explicit PlanarFace(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::PLIC
