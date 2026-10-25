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
[[nodiscard]] std::vector<char> planar_support<ArithmeticMode::Ordinary>(const PlanarFace<ArithmeticMode::Ordinary>&);
template<> [[nodiscard]] int planar_orientation<ArithmeticMode::Ordinary>(const PlanarFace<ArithmeticMode::Ordinary>&);
template<>
[[nodiscard]] std::vector<PlanarFacePtr<ArithmeticMode::Ordinary>> subtract_planar_faces<ArithmeticMode::Ordinary>(
    const PlanarFace<ArithmeticMode::Ordinary>&, std::span<const PlanarFacePtr<ArithmeticMode::Ordinary>>);
template<> [[nodiscard]] std::vector<char> displacement<ArithmeticMode::Ordinary>(GeometryPoint, GeometryPoint);
template<>
[[nodiscard]] std::vector<char> displacement_difference<ArithmeticMode::Ordinary>(
    std::span<const char>, std::span<const char>);
template<> [[nodiscard]] GeometryPoint displacement_display<ArithmeticMode::Ordinary>(std::span<const char>);
template<>
[[nodiscard]] std::vector<char> point_translated<ArithmeticMode::Ordinary>(
    std::span<const char>, std::span<const char>);
template<>
[[nodiscard]] std::vector<char> periodic_point<ArithmeticMode::Ordinary>(
    std::span<const char>, GeometryPoint, GeometryPoint, std::array<bool, 3>);
template<> [[nodiscard]] GeometryPoint point_display<ArithmeticMode::Ordinary>(std::span<const char>);
template<>
[[nodiscard]] std::array<GeometryValue<ArithmeticMode::Ordinary>, 3> numerical_point<ArithmeticMode::Ordinary>(
    std::span<const char>);
template<>
[[nodiscard]] GeometryMeasure<ArithmeticMode::Ordinary> point_projection<ArithmeticMode::Ordinary>(
    std::span<const char>, GeometryPoint, GeometryPoint);

template<> class EdgePointIndex<ArithmeticMode::Ordinary>
{
public:
    struct Impl;
    explicit EdgePointIndex(std::span<const std::vector<char>> points);
    [[nodiscard]] std::vector<std::vector<char>> closed_segment(
        std::span<const char> first, std::span<const char> second) const;
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
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> translated_packet(std::span<const char>) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> oriented_like(GeometryPoint) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> intersection(const PlanarFace&) const;
    [[nodiscard]] std::vector<std::vector<char>> vertex_identities() const;
    [[nodiscard]] std::vector<std::vector<char>> edge_support_identities() const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Ordinary> with_edge_vertices(std::span<const std::vector<char>>) const;
    [[nodiscard]] bool equivalent(const PlanarFace&) const;
    [[nodiscard]] bool contains_point(std::span<const char>) const;
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
    [[nodiscard]] std::vector<char> encode() const;
    [[nodiscard]] std::vector<char> encode_incidence() const;
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Ordinary> decode(std::span<const char>);
    [[nodiscard]] const std::vector<WidePoint>& points() const;

private:
    friend class details::NativeSupportAccess;
    friend std::vector<char> planar_support<ArithmeticMode::Ordinary>(const PlanarFace&);
    friend int planar_orientation<ArithmeticMode::Ordinary>(const PlanarFace&);
    friend std::vector<PlanarFacePtr<ArithmeticMode::Ordinary>> subtract_planar_faces<ArithmeticMode::Ordinary>(
        const PlanarFace&, std::span<const PlanarFacePtr<ArithmeticMode::Ordinary>>);
    template<ArithmeticMode> friend class ConvexCellGeometry;
    [[nodiscard]] static PlanarFacePtr<ArithmeticMode::Ordinary> from_retained_support(
        std::span<const WidePoint>, WidePoint normal, std::vector<char> support);
    explicit PlanarFace(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::PLIC
