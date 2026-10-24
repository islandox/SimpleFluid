/** @file BoundedGeometryKernels.hh
 * @brief Compiled ordinary and certification kernels sharing immutable geometry storage.
 */
#pragma once
// Both policies reuse the same Impl, planes, vertices, and cache representation.
// Separate lexical namespaces bind operators on that shared storage to a fixed
// policy; templates cannot select a namespace during dependent operator lookup.
// Raw-storage arithmetic is deleted so an ADL fallback is a compile-time error.
// Kernel views are stack-only; every retained allocation is the existing facade.
#include "geometry/details/BoundedGeometryArithmetic.hh"

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
namespace ordinary_kernel SIMPLEFLUID_LOCAL
{
inline constexpr ArithmeticMode kernel_mode = ArithmeticMode::Ordinary;
using StorageInterval = bounded_geometry_detail::Interval;
using Interval = ::SimpleFluid::detail::directed_fp::Interval<ArithmeticPolicy<kernel_mode>>;
static_assert(sizeof(Interval) == sizeof(StorageInterval) && alignof(Interval) == alignof(StorageInterval));
constexpr bool ordinary()
{
    return kernel_mode == ArithmeticMode::Ordinary;
}
inline void finite(long double value)
{
    ArithmeticPolicy<kernel_mode>::finite(value);
}
inline Interval operator+(StorageInterval a, StorageInterval b)
{
    return Interval(a) + Interval(b);
}
inline Interval operator-(StorageInterval a, StorageInterval b)
{
    return Interval(a) - Interval(b);
}
inline Interval operator*(StorageInterval a, StorageInterval b)
{
    return Interval(a) * Interval(b);
}
inline Interval operator/(StorageInterval a, StorageInterval b)
{
    return Interval(a) / Interval(b);
}
inline Interval operator-(StorageInterval a)
{
    return -Interval(a);
}
template<ArithmeticMode Mode> Interval scaled(StorageInterval a, int e)
{
    static_assert(Mode == kernel_mode);
    return ::SimpleFluid::detail::directed_fp::scaled(Interval(a), e);
}
template<ArithmeticMode Mode> struct Accumulator
{
    static_assert(Mode == kernel_mode);
    Interval enclosure;
    long double sum = 0, correction = 0;
    void add(Interval term);
    Interval result() const;
};
template<ArithmeticMode Mode> struct NumericalWriter
{
    static_assert(Mode == kernel_mode);
    std::vector<char> bytes;
    std::size_t limit;
    explicit NumericalWriter(std::size_t budget) : limit(budget) {}
    void word(std::uint64_t x);
    void real(long double x);
    void interval(Interval x);
    void coordinates(const Point& p);
    void frame(const Frame& f);
    void authorities(const std::vector<BoundedPlaneAuthority>& keys);
};
template<ArithmeticMode Mode> struct NumericalReader
{
    static_assert(Mode == kernel_mode);
    std::span<const char> bytes;
    std::size_t cursor = 0;
    explicit NumericalReader(std::span<const char> input) : bytes(input) {}
    std::uint64_t word();
    std::size_t count(std::size_t maximum, std::size_t element);
    long double real();
    Interval interval();
    Point coordinates() { return {interval(), interval(), interval()}; }
    Frame frame();
    std::vector<BoundedPlaneAuthority> authorities(
        const std::set<BoundedPlaneAuthority>& registry, BoundedGeometryLimits limits);
};
template<ArithmeticMode Mode> class GeometryKernel : public BoundedConvexGeometry
{
    static_assert(Mode == kernel_mode);

public:
    using Impl = BoundedConvexGeometry::Impl;
    using BoundedConvexGeometry::arithmetic_mode;
    using BoundedConvexGeometry::faces;
    using BoundedConvexGeometry::impl_;
    using BoundedConvexGeometry::origin;
    using BoundedConvexGeometry::points;
    using BoundedConvexGeometry::storage_bytes;
    static std::vector<BoundedConvexGeometryPtr> validate_donors(
        std::span<const BoundedConvexGeometryPtr> donors, BoundedGeometryLimits limits);
    static BoundedPoint backtrace(const BoundedPoint& source, GeometryPoint velocity, double timestep,
        std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits);
    explicit GeometryKernel(std::shared_ptr<const Impl> impl) : BoundedConvexGeometry(std::move(impl)) {}
    static BoundedConvexGeometryPtr make(
        std::span<const GeometryPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits);
    static BoundedConvexGeometryPtr make_bounded(
        std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits);
    static BoundedConvexGeometryPtr make_bounded_local(GeometryPoint origin, std::span<const BoundedPoint> vertices,
        const std::vector<Face>& faces, BoundedGeometryLimits limits);
    static BoundedConvexGeometryPtr from_geometry(const ConvexCellGeometry& source, BoundedGeometryLimits limits);
    static BoundedConvexGeometryPtr box(GeometryPoint lower, GeometryPoint upper, BoundedGeometryLimits limits);
    static BoundedConvexGeometryPtr tetrahedron(
        const std::array<BoundedPoint, 4>& vertices, BoundedGeometryLimits limits);
    BoundedGeometryValue volume() const;
    BoundedGeometryMoments moments() const;
    BoundedGeometryMoments moments(GeometryPoint origin) const;
    GeometryPoint centroid() const;
    std::vector<BoundedPoint> local_points() const;
    std::array<BoundedGeometryValue, 3> axis_spans() const;
    BoundedSupport support(GeometryPoint normal, GeometryPoint anchor) const;
    BoundedClipResult clip(const GeometryHalfspace& input, bool measure_section) const;
    BoundedClipResult clip_by_face(const BoundedConvexGeometry& source, std::size_t face, bool measure_section) const;
    BoundedClipResult clip_impl(const std::shared_ptr<const Impl>& source, std::size_t plane_id, bool measure_section,
        bool construct_outside = true, bool partition_proof = false) const;
    BoundedPhaseMeasures phases(const GeometryHalfspace& gas) const;
    bool bounds_prove_disjoint_interiors(const BoundedConvexGeometry& other) const;
    BoundedConvexGeometryPtr intersect(const BoundedConvexGeometry& other) const;
    BoundedConvexGeometryPtr translated(GeometryPoint displacement) const;
    void certify_partition(std::span<const BoundedConvexGeometryPtr> pieces, double relative_budget) const;
    std::vector<char> encode(const BoundedGeometryPacketContext& context, std::size_t byte_budget) const;
    static BoundedConvexGeometryPtr decode(std::span<const char> packet, const BoundedGeometryPacketContext& expected,
        std::span<const BoundedConvexGeometryPtr> authenticated_sources, BoundedGeometryLimits limits);
};
template<ArithmeticMode Mode> class ValueKernel : public BoundedGeometryValue
{
    static_assert(Mode == kernel_mode);

public:
    using BoundedGeometryValue::absolute_error;
    using BoundedGeometryValue::arithmetic_mode;
    using BoundedGeometryValue::value;
    explicit ValueKernel(BoundedGeometryValue value) : BoundedGeometryValue(value) {}
    long double lower() const;
    long double upper() const;
    double as_double() const;
    bool relative_error_at_most(long double target, double tolerance) const;
};
template<ArithmeticMode Mode> class PartitionKernel
{
    static_assert(Mode == kernel_mode);
    const std::vector<BoundedConvexGeometryPtr>& donors_;
    const BoundedGeometryLimits& limits_;

public:
    explicit PartitionKernel(const BoundedConvexPartition& source) : donors_(source.donors_), limits_(source.limits_) {}
    std::vector<BoundedPartitionPiece> partition(const BoundedConvexGeometry& parent, std::size_t maximum_pieces) const;
};
template<ArithmeticMode Mode> long double directed_add(long double a, long double b, bool up);
template<ArithmeticMode Mode> long double directed_multiply(long double a, long double b, bool up);
template<ArithmeticMode Mode> long double directed_divide(long double a, long double b, bool up);
template<ArithmeticMode Mode> Interval enclosure(BoundedGeometryValue a);
template<ArithmeticMode Mode> BoundedGeometryValue bounded(Interval a);
template<ArithmeticMode Mode> Point point(const BoundedPoint& a);
template<ArithmeticMode Mode> Point point(GeometryPoint a);
template<ArithmeticMode Mode> BoundedPoint bounded(const Point& a);
Point operator+(const Point& a, const Point& b);
Point operator-(const Point& a, const Point& b);
Point operator*(const Point& a, Interval b);
template<ArithmeticMode Mode> Point scaled(const Point& a, int e);
template<ArithmeticMode Mode> Interval dot(const Point& a, const Point& b);
template<ArithmeticMode Mode> Point cross(const Point& a, const Point& b);
template<ArithmeticMode Mode> bool same(const Point& a, const Point& b);
template<ArithmeticMode Mode> bool nonzero(const Point& a);
template<ArithmeticMode Mode> int sign(Interval a);
template<ArithmeticMode Mode> Point transform(const Point& p, const Frame& from, const Frame& to);
template<ArithmeticMode Mode> CoordinateBox transform(const CoordinateBox& box, const Frame& from, const Frame& to);
template<ArithmeticMode Mode> Point interval_box(const CoordinateBox& box);
template<ArithmeticMode Mode> BoundedPoint converted_point(const BoundedPoint& input, ArithmeticMode mode);
template<ArithmeticMode Mode> Interval raw_distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template<ArithmeticMode Mode> Interval distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template<ArithmeticMode Mode> void tighten_incident_coordinates(Vertex& vertex, const Frame& frame);
template<ArithmeticMode Mode> Interval square_root(Interval x);
template<ArithmeticMode Mode> Integrals integrate_ordinary(const BoundedConvexGeometry::Impl& poly, bool moments);
template<ArithmeticMode Mode> Integrals integrate(const BoundedConvexGeometry::Impl& poly, bool moments);
template<ArithmeticMode Mode>
void finish(BoundedConvexGeometry::Impl& poly, bool partition_proof = false, bool positive_construction = false);
template<ArithmeticMode Mode>
void validate_connected_boundary(std::size_t vertex_count, const std::vector<Face>& faces);
template<ArithmeticMode Mode>
std::shared_ptr<BoundedConvexGeometry::Impl> import(GeometryPoint origin, std::span<const BoundedPoint> vertices,
    const std::vector<Face>& faces, BoundedGeometryLimits limits, bool authenticated_planarity,
    std::span<const std::shared_ptr<const BoundedPointProvenance>> identities = {});
template<ArithmeticMode Mode> BoundedPoint bounded_point_average(std::span<const BoundedPoint> vertices);
template<ArithmeticMode Mode>
BoundedPoint bounded_point_interpolate(const BoundedPoint& a, const BoundedPoint& b, long double fraction);
template<ArithmeticMode Mode> bool bounded_points_identical(const BoundedPoint& a, const BoundedPoint& b);
template<ArithmeticMode Mode> BoundedGeometryValue bounded_orientation(const std::array<BoundedPoint, 4>& vertices);
template<ArithmeticMode Mode> int matching_input(const GeometryHalfspace& a, const GeometryHalfspace& b);
template<ArithmeticMode Mode> bool incident(const Vertex& v, const PlanePtr& plane);
template<ArithmeticMode Mode>
Interval refine_triangle_distance(const Point& coordinates, const Frame& frame, const Vertex& a, const Vertex& b,
    const Vertex& c, const Frame& triangle_frame, const PlanePtr& plane, int orientation, Interval result);
template<ArithmeticMode Mode>
BoundedGeometrySection section(
    const BoundedConvexGeometry::Impl& poly, const std::vector<Vertex>& vertices, const std::vector<std::size_t>& loop);
template<ArithmeticMode Mode>
bool construction_contains(const std::shared_ptr<const Ancestry>& parent, std::shared_ptr<const Ancestry> child);
template<ArithmeticMode Mode>
bool construction_disjoint(std::shared_ptr<const Ancestry> a, std::shared_ptr<const Ancestry> b);
template<ArithmeticMode Mode>
bool separated_bounds(const BoundedConvexGeometry::Impl& a, const BoundedConvexGeometry::Impl& b);
template<ArithmeticMode Mode>
bool structural_cover(
    const std::shared_ptr<const Ancestry>& root, const std::vector<std::shared_ptr<const Ancestry>>& leaves);
template<ArithmeticMode Mode>
BoundedPoint source_point_fan(
    std::span<const ConvexCellGeometry* const> sources, std::span<const char> identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
BoundedPoint bounded_source_point(
    const ConvexCellGeometry& source, std::span<const char> identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
BoundedPoint bounded_source_point(
    std::span<const ConvexGeometryPtr> sources, std::span<const char> identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
BoundedPoint bounded_point_displaced(const BoundedPoint& source, const BoundedPoint& displacement);
template<ArithmeticMode Mode>
BoundedPoint bounded_backtrace_constrained(const BoundedPoint& source, GeometryPoint velocity, double timestep,
    std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits);
template<ArithmeticMode Mode> Interval planar_turn(const PlanarVertex& a, const PlanarVertex& b, const PlanarVertex& c);
template<ArithmeticMode Mode>
std::vector<BoundedPlaneAuthority> planar_common(
    const std::vector<BoundedPlaneAuthority>& first, const std::vector<BoundedPlaneAuthority>& second);
template<ArithmeticMode Mode> void planar_charge(const PlanarVertex& vertex);
template<ArithmeticMode Mode>
PlanarLine planar_line(const PlanarVertex& first, const PlanarVertex& second, const BoundedPlaneAuthority& primary);
template<ArithmeticMode Mode> Interval planar_distance(const PlanarVertex& vertex, const PlanarLine& edge);
template<ArithmeticMode Mode> Interval planar_area(const PlanarPolygon& polygon);
template<ArithmeticMode Mode>
void planar_convex(const PlanarPolygon& polygon, int winding, const BoundedPlaneAuthority& primary);
template<ArithmeticMode Mode> void planar_push(PlanarPolygon& polygon, PlanarVertex vertex);
template<ArithmeticMode Mode>
PlanarVertex planar_crossing(
    const PlanarVertex& a, const PlanarVertex& b, Interval da, Interval db, const PlanarLine& edge, std::size_t& token);
template<ArithmeticMode Mode>
bool planar_line_contact(const PlanarPolygon& polygon, const BoundedPlaneAuthority& primary);
template<ArithmeticMode Mode>
BoundedPlanarFaceOverlap bounded_planar_face_overlap(std::span<const BoundedPoint> first,
    std::span<const BoundedPoint> second, const BoundedPlaneAuthority& support, BoundedGeometryLimits limits);
} // namespace ordinary_kernel SIMPLEFLUID_LOCAL

} // namespace SimpleFluid::Meshes::bounded_geometry_detail

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
namespace certification_kernel SIMPLEFLUID_LOCAL
{
inline constexpr ArithmeticMode kernel_mode = ArithmeticMode::BoundedCertification;
using StorageInterval = bounded_geometry_detail::Interval;
using Interval = ::SimpleFluid::detail::directed_fp::Interval<ArithmeticPolicy<kernel_mode>>;
static_assert(sizeof(Interval) == sizeof(StorageInterval) && alignof(Interval) == alignof(StorageInterval));
constexpr bool ordinary()
{
    return kernel_mode == ArithmeticMode::Ordinary;
}
inline void finite(long double value)
{
    ArithmeticPolicy<kernel_mode>::finite(value);
}
inline Interval operator+(StorageInterval a, StorageInterval b)
{
    return Interval(a) + Interval(b);
}
inline Interval operator-(StorageInterval a, StorageInterval b)
{
    return Interval(a) - Interval(b);
}
inline Interval operator*(StorageInterval a, StorageInterval b)
{
    return Interval(a) * Interval(b);
}
inline Interval operator/(StorageInterval a, StorageInterval b)
{
    return Interval(a) / Interval(b);
}
inline Interval operator-(StorageInterval a)
{
    return -Interval(a);
}
template<ArithmeticMode Mode> Interval scaled(StorageInterval a, int e)
{
    static_assert(Mode == kernel_mode);
    return ::SimpleFluid::detail::directed_fp::scaled(Interval(a), e);
}
template<ArithmeticMode Mode> struct Accumulator
{
    static_assert(Mode == kernel_mode);
    Interval enclosure;
    long double sum = 0, correction = 0;
    void add(Interval term);
    Interval result() const;
};
template<ArithmeticMode Mode> struct NumericalWriter
{
    static_assert(Mode == kernel_mode);
    std::vector<char> bytes;
    std::size_t limit;
    explicit NumericalWriter(std::size_t budget) : limit(budget) {}
    void word(std::uint64_t x);
    void real(long double x);
    void interval(Interval x);
    void coordinates(const Point& p);
    void frame(const Frame& f);
    void authorities(const std::vector<BoundedPlaneAuthority>& keys);
};
template<ArithmeticMode Mode> struct NumericalReader
{
    static_assert(Mode == kernel_mode);
    std::span<const char> bytes;
    std::size_t cursor = 0;
    explicit NumericalReader(std::span<const char> input) : bytes(input) {}
    std::uint64_t word();
    std::size_t count(std::size_t maximum, std::size_t element);
    long double real();
    Interval interval();
    Point coordinates() { return {interval(), interval(), interval()}; }
    Frame frame();
    std::vector<BoundedPlaneAuthority> authorities(
        const std::set<BoundedPlaneAuthority>& registry, BoundedGeometryLimits limits);
};
template<ArithmeticMode Mode> class GeometryKernel : public BoundedConvexGeometry
{
    static_assert(Mode == kernel_mode);

public:
    using Impl = BoundedConvexGeometry::Impl;
    using BoundedConvexGeometry::arithmetic_mode;
    using BoundedConvexGeometry::faces;
    using BoundedConvexGeometry::impl_;
    using BoundedConvexGeometry::origin;
    using BoundedConvexGeometry::points;
    using BoundedConvexGeometry::storage_bytes;
    static std::vector<BoundedConvexGeometryPtr> validate_donors(
        std::span<const BoundedConvexGeometryPtr> donors, BoundedGeometryLimits limits);
    static BoundedPoint backtrace(const BoundedPoint& source, GeometryPoint velocity, double timestep,
        std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits);
    explicit GeometryKernel(std::shared_ptr<const Impl> impl) : BoundedConvexGeometry(std::move(impl)) {}
    static BoundedConvexGeometryPtr make(
        std::span<const GeometryPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits);
    static BoundedConvexGeometryPtr make_bounded(
        std::span<const BoundedPoint> vertices, const std::vector<Face>& faces, BoundedGeometryLimits limits);
    static BoundedConvexGeometryPtr make_bounded_local(GeometryPoint origin, std::span<const BoundedPoint> vertices,
        const std::vector<Face>& faces, BoundedGeometryLimits limits);
    static BoundedConvexGeometryPtr from_geometry(const ConvexCellGeometry& source, BoundedGeometryLimits limits);
    static BoundedConvexGeometryPtr box(GeometryPoint lower, GeometryPoint upper, BoundedGeometryLimits limits);
    static BoundedConvexGeometryPtr tetrahedron(
        const std::array<BoundedPoint, 4>& vertices, BoundedGeometryLimits limits);
    BoundedGeometryValue volume() const;
    BoundedGeometryMoments moments() const;
    BoundedGeometryMoments moments(GeometryPoint origin) const;
    GeometryPoint centroid() const;
    std::vector<BoundedPoint> local_points() const;
    std::array<BoundedGeometryValue, 3> axis_spans() const;
    BoundedSupport support(GeometryPoint normal, GeometryPoint anchor) const;
    BoundedClipResult clip(const GeometryHalfspace& input, bool measure_section) const;
    BoundedClipResult clip_by_face(const BoundedConvexGeometry& source, std::size_t face, bool measure_section) const;
    BoundedClipResult clip_impl(const std::shared_ptr<const Impl>& source, std::size_t plane_id, bool measure_section,
        bool construct_outside = true, bool partition_proof = false) const;
    BoundedPhaseMeasures phases(const GeometryHalfspace& gas) const;
    bool bounds_prove_disjoint_interiors(const BoundedConvexGeometry& other) const;
    BoundedConvexGeometryPtr intersect(const BoundedConvexGeometry& other) const;
    BoundedConvexGeometryPtr translated(GeometryPoint displacement) const;
    void certify_partition(std::span<const BoundedConvexGeometryPtr> pieces, double relative_budget) const;
    std::vector<char> encode(const BoundedGeometryPacketContext& context, std::size_t byte_budget) const;
    static BoundedConvexGeometryPtr decode(std::span<const char> packet, const BoundedGeometryPacketContext& expected,
        std::span<const BoundedConvexGeometryPtr> authenticated_sources, BoundedGeometryLimits limits);
};
template<ArithmeticMode Mode> class ValueKernel : public BoundedGeometryValue
{
    static_assert(Mode == kernel_mode);

public:
    using BoundedGeometryValue::absolute_error;
    using BoundedGeometryValue::arithmetic_mode;
    using BoundedGeometryValue::value;
    explicit ValueKernel(BoundedGeometryValue value) : BoundedGeometryValue(value) {}
    long double lower() const;
    long double upper() const;
    double as_double() const;
    bool relative_error_at_most(long double target, double tolerance) const;
};
template<ArithmeticMode Mode> class PartitionKernel
{
    static_assert(Mode == kernel_mode);
    const std::vector<BoundedConvexGeometryPtr>& donors_;
    const BoundedGeometryLimits& limits_;

public:
    explicit PartitionKernel(const BoundedConvexPartition& source) : donors_(source.donors_), limits_(source.limits_) {}
    std::vector<BoundedPartitionPiece> partition(const BoundedConvexGeometry& parent, std::size_t maximum_pieces) const;
};
template<ArithmeticMode Mode> long double directed_add(long double a, long double b, bool up);
template<ArithmeticMode Mode> long double directed_multiply(long double a, long double b, bool up);
template<ArithmeticMode Mode> long double directed_divide(long double a, long double b, bool up);
template<ArithmeticMode Mode> Interval enclosure(BoundedGeometryValue a);
template<ArithmeticMode Mode> BoundedGeometryValue bounded(Interval a);
template<ArithmeticMode Mode> Point point(const BoundedPoint& a);
template<ArithmeticMode Mode> Point point(GeometryPoint a);
template<ArithmeticMode Mode> BoundedPoint bounded(const Point& a);
Point operator+(const Point& a, const Point& b);
Point operator-(const Point& a, const Point& b);
Point operator*(const Point& a, Interval b);
template<ArithmeticMode Mode> Point scaled(const Point& a, int e);
template<ArithmeticMode Mode> Interval dot(const Point& a, const Point& b);
template<ArithmeticMode Mode> Point cross(const Point& a, const Point& b);
template<ArithmeticMode Mode> bool same(const Point& a, const Point& b);
template<ArithmeticMode Mode> bool nonzero(const Point& a);
template<ArithmeticMode Mode> int sign(Interval a);
template<ArithmeticMode Mode> Point transform(const Point& p, const Frame& from, const Frame& to);
template<ArithmeticMode Mode> CoordinateBox transform(const CoordinateBox& box, const Frame& from, const Frame& to);
template<ArithmeticMode Mode> Point interval_box(const CoordinateBox& box);
template<ArithmeticMode Mode> BoundedPoint converted_point(const BoundedPoint& input, ArithmeticMode mode);
template<ArithmeticMode Mode> Interval raw_distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template<ArithmeticMode Mode> Interval distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template<ArithmeticMode Mode> void tighten_incident_coordinates(Vertex& vertex, const Frame& frame);
template<ArithmeticMode Mode> Interval square_root(Interval x);
template<ArithmeticMode Mode> Integrals integrate_ordinary(const BoundedConvexGeometry::Impl& poly, bool moments);
template<ArithmeticMode Mode> Integrals integrate(const BoundedConvexGeometry::Impl& poly, bool moments);
template<ArithmeticMode Mode>
void finish(BoundedConvexGeometry::Impl& poly, bool partition_proof = false, bool positive_construction = false);
template<ArithmeticMode Mode>
void validate_connected_boundary(std::size_t vertex_count, const std::vector<Face>& faces);
template<ArithmeticMode Mode>
std::shared_ptr<BoundedConvexGeometry::Impl> import(GeometryPoint origin, std::span<const BoundedPoint> vertices,
    const std::vector<Face>& faces, BoundedGeometryLimits limits, bool authenticated_planarity,
    std::span<const std::shared_ptr<const BoundedPointProvenance>> identities = {});
template<ArithmeticMode Mode> BoundedPoint bounded_point_average(std::span<const BoundedPoint> vertices);
template<ArithmeticMode Mode>
BoundedPoint bounded_point_interpolate(const BoundedPoint& a, const BoundedPoint& b, long double fraction);
template<ArithmeticMode Mode> bool bounded_points_identical(const BoundedPoint& a, const BoundedPoint& b);
template<ArithmeticMode Mode> BoundedGeometryValue bounded_orientation(const std::array<BoundedPoint, 4>& vertices);
template<ArithmeticMode Mode> int matching_input(const GeometryHalfspace& a, const GeometryHalfspace& b);
template<ArithmeticMode Mode> bool incident(const Vertex& v, const PlanePtr& plane);
template<ArithmeticMode Mode>
Interval refine_triangle_distance(const Point& coordinates, const Frame& frame, const Vertex& a, const Vertex& b,
    const Vertex& c, const Frame& triangle_frame, const PlanePtr& plane, int orientation, Interval result);
template<ArithmeticMode Mode>
BoundedGeometrySection section(
    const BoundedConvexGeometry::Impl& poly, const std::vector<Vertex>& vertices, const std::vector<std::size_t>& loop);
template<ArithmeticMode Mode>
bool construction_contains(const std::shared_ptr<const Ancestry>& parent, std::shared_ptr<const Ancestry> child);
template<ArithmeticMode Mode>
bool construction_disjoint(std::shared_ptr<const Ancestry> a, std::shared_ptr<const Ancestry> b);
template<ArithmeticMode Mode>
bool separated_bounds(const BoundedConvexGeometry::Impl& a, const BoundedConvexGeometry::Impl& b);
template<ArithmeticMode Mode>
bool structural_cover(
    const std::shared_ptr<const Ancestry>& root, const std::vector<std::shared_ptr<const Ancestry>>& leaves);
template<ArithmeticMode Mode>
BoundedPoint source_point_fan(
    std::span<const ConvexCellGeometry* const> sources, std::span<const char> identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
BoundedPoint bounded_source_point(
    const ConvexCellGeometry& source, std::span<const char> identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
BoundedPoint bounded_source_point(
    std::span<const ConvexGeometryPtr> sources, std::span<const char> identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
BoundedPoint bounded_point_displaced(const BoundedPoint& source, const BoundedPoint& displacement);
template<ArithmeticMode Mode>
BoundedPoint bounded_backtrace_constrained(const BoundedPoint& source, GeometryPoint velocity, double timestep,
    std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits);
template<ArithmeticMode Mode> Interval planar_turn(const PlanarVertex& a, const PlanarVertex& b, const PlanarVertex& c);
template<ArithmeticMode Mode>
std::vector<BoundedPlaneAuthority> planar_common(
    const std::vector<BoundedPlaneAuthority>& first, const std::vector<BoundedPlaneAuthority>& second);
template<ArithmeticMode Mode> void planar_charge(const PlanarVertex& vertex);
template<ArithmeticMode Mode>
PlanarLine planar_line(const PlanarVertex& first, const PlanarVertex& second, const BoundedPlaneAuthority& primary);
template<ArithmeticMode Mode> Interval planar_distance(const PlanarVertex& vertex, const PlanarLine& edge);
template<ArithmeticMode Mode> Interval planar_area(const PlanarPolygon& polygon);
template<ArithmeticMode Mode>
void planar_convex(const PlanarPolygon& polygon, int winding, const BoundedPlaneAuthority& primary);
template<ArithmeticMode Mode> void planar_push(PlanarPolygon& polygon, PlanarVertex vertex);
template<ArithmeticMode Mode>
PlanarVertex planar_crossing(
    const PlanarVertex& a, const PlanarVertex& b, Interval da, Interval db, const PlanarLine& edge, std::size_t& token);
template<ArithmeticMode Mode>
bool planar_line_contact(const PlanarPolygon& polygon, const BoundedPlaneAuthority& primary);
template<ArithmeticMode Mode>
BoundedPlanarFaceOverlap bounded_planar_face_overlap(std::span<const BoundedPoint> first,
    std::span<const BoundedPoint> second, const BoundedPlaneAuthority& support, BoundedGeometryLimits limits);
} // namespace certification_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes::bounded_geometry_detail
