/** @file PlanarGeometry.hh
 * @brief Ordinary planar polygons with finite coordinate and support identities.
 */
#pragma once
#include "geometry/PLIC/PlanarGeometryDeclarations.hh"
namespace SimpleFluid::PLIC
{
namespace details
{
class NativeSupportAccess;
}
template<>
[[nodiscard]] PlaneSupportIdentity<ArithmeticMode::Ordinary> planar_support<ArithmeticMode::Ordinary>(
    const PlanarFace<ArithmeticMode::Ordinary>&);
template<> [[nodiscard]] int planar_orientation<ArithmeticMode::Ordinary>(const PlanarFace<ArithmeticMode::Ordinary>&);
template<>
[[nodiscard]] std::vector<PlanarFacePtr<ArithmeticMode::Ordinary>> subtract_planar_faces<ArithmeticMode::Ordinary>(
    const PlanarFace<ArithmeticMode::Ordinary>&, std::span<const PlanarFacePtr<ArithmeticMode::Ordinary>>);
template<>
[[nodiscard]] DisplacementIdentity<ArithmeticMode::Ordinary> displacement<ArithmeticMode::Ordinary>(
    GeometryPoint, GeometryPoint);
template<>
[[nodiscard]] DisplacementIdentity<ArithmeticMode::Ordinary> displacement_difference<ArithmeticMode::Ordinary>(
    const DisplacementIdentity<ArithmeticMode::Ordinary>&, const DisplacementIdentity<ArithmeticMode::Ordinary>&);
template<>
[[nodiscard]] GeometryPoint displacement_display<ArithmeticMode::Ordinary>(
    const DisplacementIdentity<ArithmeticMode::Ordinary>&);
template<>
[[nodiscard]] PointIdentity<ArithmeticMode::Ordinary> point_translated<ArithmeticMode::Ordinary>(
    const PointIdentity<ArithmeticMode::Ordinary>&, const DisplacementIdentity<ArithmeticMode::Ordinary>&);
template<>
[[nodiscard]] PointIdentity<ArithmeticMode::Ordinary> periodic_point<ArithmeticMode::Ordinary>(
    const PointIdentity<ArithmeticMode::Ordinary>&, GeometryPoint, GeometryPoint, std::array<bool, 3>);
template<>
[[nodiscard]] GeometryPoint point_display<ArithmeticMode::Ordinary>(const PointIdentity<ArithmeticMode::Ordinary>&);
template<>
[[nodiscard]] std::array<GeometryValue<ArithmeticMode::Ordinary>, 3> numerical_point<ArithmeticMode::Ordinary>(
    const PointIdentity<ArithmeticMode::Ordinary>&);
template<>
[[nodiscard]] GeometryMeasure<ArithmeticMode::Ordinary> point_projection<ArithmeticMode::Ordinary>(
    const PointIdentity<ArithmeticMode::Ordinary>&, GeometryPoint, GeometryPoint);

template<>
[[nodiscard]] std::vector<char> canonical_face_key<ArithmeticMode::Ordinary>(
    const PlanarFace<ArithmeticMode::Ordinary>&);
template<> class EdgePointIndex<ArithmeticMode::Ordinary>
{
public:
    struct Impl;
    explicit EdgePointIndex(std::span<const PointIdentity<ArithmeticMode::Ordinary>> points);
    [[nodiscard]] std::vector<PointIdentity<ArithmeticMode::Ordinary>> closed_segment(
        const PointIdentity<ArithmeticMode::Ordinary>& first,
        const PointIdentity<ArithmeticMode::Ordinary>& second) const;
    [[nodiscard]] std::size_t storage_bytes() const;

private:
    std::shared_ptr<const Impl> impl_;
};

template<> class PlanarFace<ArithmeticMode::Ordinary>
{
public:
    using WidePoint = std::array<long double, 3>;
    struct Impl;
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Ordinary> make(std::span<const GeometryPoint>);
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Ordinary> make_wide(std::span<const WidePoint>);
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Ordinary> from_native_face(
        const MultiRegionMesh&, std::uint64_t);
    [[nodiscard]] GeometryValue<ArithmeticMode::Ordinary> area() const;
    [[nodiscard]] ::SimpleFluid::MeshUtils::Vec3 centroid() const;
    [[nodiscard]] ::SimpleFluid::MeshUtils::Vec3 area_vector() const;
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Ordinary> coplanar_area_ratio(
        const PlanarFace&, std::size_t byte_budget = 64 * 1024 * 1024) const;
    [[nodiscard]] GeometryMeasure<ArithmeticMode::Ordinary> area_vector_dot(
        GeometryPoint, std::size_t byte_budget = 64 * 1024 * 1024) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> reversed() const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> translated(GeometryPoint) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> translated(
        const DisplacementIdentity<ArithmeticMode::Ordinary>&) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> oriented_like(GeometryPoint) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> intersection(const PlanarFace&) const;
    [[nodiscard]] std::vector<PointIdentity<ArithmeticMode::Ordinary>> vertex_identities() const;
    [[nodiscard]] std::vector<LineIdentity<ArithmeticMode::Ordinary>> edge_support_identities() const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> with_edge_vertices(
        std::span<const PointIdentity<ArithmeticMode::Ordinary>>) const;
    [[nodiscard]] bool equivalent(const PlanarFace&) const;
    [[nodiscard]] bool contains_point(const PointIdentity<ArithmeticMode::Ordinary>&) const;
    [[nodiscard]] bool cell_bounds_overlap(const ConvexCellGeometry<ArithmeticMode::Ordinary>&) const;
    [[nodiscard]] bool contains(const PlanarFace&) const;
    [[nodiscard]] bool oppositely_oriented(const PlanarFace&) const;
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Ordinary> cell_shell(
        std::span<const PlanarFacePtr<ArithmeticMode::Ordinary>>);
    /** Require disjoint containment and exact cancellation of represented
     * oriented boundary segments after shared-vertex subdivision. Independently
     * check area and local first-moment closure within the finite 8e-15 physical
     * budget. The metric allowance never admits a gap in the boundary chain.
     * Work is capped at 65536 tiles, 262144 pair tests, and 64 MiB. */
    void certify_partition(std::span<const PlanarFacePtr<ArithmeticMode::Ordinary>>) const;
    [[nodiscard]] std::vector<char> encode(std::size_t byte_budget = geometry_packet_default_budget) const;
    [[nodiscard]] std::vector<char> encode_incidence(std::size_t byte_budget = geometry_packet_default_budget) const;
    /** Own face buffers only; shared identity owners are visited separately. */
    [[nodiscard]] std::size_t exclusive_storage_bytes() const;
    /** Full standalone footprint, including retained identity ancestry. */
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Ordinary> decode(
        std::span<const char>, std::size_t byte_budget = geometry_packet_default_budget);
    [[nodiscard]] const std::vector<WidePoint>& points() const;

private:
    friend std::vector<char> canonical_face_key<ArithmeticMode::Ordinary>(const PlanarFace&);
    friend class details::NativeSupportAccess;
    friend PlaneSupportIdentity<ArithmeticMode::Ordinary> planar_support<ArithmeticMode::Ordinary>(const PlanarFace&);
    friend int planar_orientation<ArithmeticMode::Ordinary>(const PlanarFace&);
    friend std::vector<PlanarFacePtr<ArithmeticMode::Ordinary>> subtract_planar_faces<ArithmeticMode::Ordinary>(
        const PlanarFace&, std::span<const PlanarFacePtr<ArithmeticMode::Ordinary>>);
    template<ArithmeticMode> friend class ConvexCellGeometry;
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Ordinary> from_retained_support(
        std::span<const WidePoint>, WidePoint normal, PlaneSupportIdentity<ArithmeticMode::Ordinary> support);
    explicit PlanarFace(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::PLIC
