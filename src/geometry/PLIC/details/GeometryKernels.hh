/** @file GeometryKernels.hh
 * @brief Compiled ordinary and certification kernels with policy-specific immutable storage.
 */
#pragma once
// Separate lexical namespaces bind operators to each fixed arithmetic policy.
// Persisted scalar and geometry types follow that policy without layout punning.
// Raw-storage arithmetic is deleted so an ADL fallback is a compile-time error.
#include "geometry/PLIC/details/GeometryArithmetic.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"

namespace SimpleFluid::PLIC::details
{
namespace ordinary_kernel SIMPLEFLUID_LOCAL
{
inline constexpr ArithmeticMode kernel_mode = ArithmeticMode::Ordinary;
[[noreturn]] inline void reject(std::string_view reason)
{
    details::reject(kernel_mode, reason);
}
using StorageScalar = StoredValue<kernel_mode>;
using Point = StoredPoint<kernel_mode>;
using Plane = StoredPlane<kernel_mode>;
using PlanePtr = StoredPlanePtr<kernel_mode>;
using Vertex = StoredVertex<kernel_mode>;
using Surface = StoredSurface<kernel_mode>;
using Integrals = StoredIntegrals<kernel_mode>;
using MomentCache = StoredMomentCache<kernel_mode>;
using PlanarVertex = StoredPlanarVertex<kernel_mode>;
using PlanarPolygon = StoredPlanarPolygon<kernel_mode>;
using PlanarLine = StoredPlanarLine<kernel_mode>;
using StoredGeometry = GeometryStorage<kernel_mode>;
using Scalar = ::SimpleFluid::detail::ordinary_fp::Scalar<ArithmeticPolicy<kernel_mode>>;
static_assert(sizeof(Scalar) == sizeof(StorageScalar) && alignof(Scalar) == alignof(StorageScalar));
inline void finite(long double value)
{
    ArithmeticPolicy<kernel_mode>::finite(value);
}
inline Scalar operator+(StorageScalar a, StorageScalar b)
{
    return Scalar(a) + Scalar(b);
}
inline Scalar operator-(StorageScalar a, StorageScalar b)
{
    return Scalar(a) - Scalar(b);
}
inline Scalar operator*(StorageScalar a, StorageScalar b)
{
    return Scalar(a) * Scalar(b);
}
inline Scalar operator/(StorageScalar a, StorageScalar b)
{
    return Scalar(a) / Scalar(b);
}
inline Scalar operator-(StorageScalar a)
{
    return -Scalar(a);
}
template<ArithmeticMode Mode> Scalar scaled(StorageScalar a, int e)
{
    static_assert(Mode == kernel_mode);
    return ::SimpleFluid::detail::ordinary_fp::scaled(Scalar(a), e);
}
template<ArithmeticMode Mode> struct Accumulator
{
    static_assert(Mode == kernel_mode);
    long double sum = 0, correction = 0;
    void add(Scalar term);
    Scalar result() const;
};
template<ArithmeticMode Mode> struct NumericalWriter
{
    static_assert(Mode == kernel_mode);
    std::vector<char> bytes;
    std::size_t limit;
    explicit NumericalWriter(std::size_t budget) : limit(budget) {}
    void byte(std::uint8_t x);
    void tag(GeometryTag value) { byte(static_cast<std::uint8_t>(value)); }
    void word(std::uint64_t x);
    void real(long double x);
    void scalar(Scalar x);
    void coordinates(const Point& p);
    void frame(const Frame& f);
    void authorities(const std::vector<PlaneAuthority>& keys);
};
template<ArithmeticMode Mode> struct NumericalReader
{
    static_assert(Mode == kernel_mode);
    std::span<const char> bytes;
    std::size_t cursor = 0;
    explicit NumericalReader(std::span<const char> input) : bytes(input) {}
    std::uint8_t byte();
    GeometryTag tag() { return static_cast<GeometryTag>(byte()); }
    std::uint64_t word();
    std::size_t count(std::size_t maximum, std::size_t element);
    long double real();
    Scalar scalar();
    Point coordinates() { return {scalar(), scalar(), scalar()}; }
    Frame frame();
    std::vector<PlaneAuthority> authorities(const std::set<PlaneAuthority>& registry, GeometryLimits limits);
};
template<ArithmeticMode Mode> class GeometryKernel : public DynamicConvexGeometry
{
    static_assert(Mode == kernel_mode);

public:
    using Impl = StoredGeometry;
    using DynamicConvexGeometry::arithmetic_mode;
    using DynamicConvexGeometry::faces;
    std::shared_ptr<const Impl> impl_;
    using DynamicConvexGeometry::origin;
    using DynamicConvexGeometry::points;
    std::size_t storage_bytes() const { return DynamicConvexGeometry::storage_bytes<Mode>(); }
    static std::vector<DynamicConvexGeometryPtr> validate_donors(
        std::span<const DynamicConvexGeometryPtr> donors, GeometryLimits limits);
    static DynamicPoint backtrace(const DynamicPoint& source, GeometryPoint velocity, double timestep,
        std::span<const DynamicFaceConstraint> walls, GeometryLimits limits);
    explicit GeometryKernel(const std::shared_ptr<const DynamicConvexGeometry::Impl>& impl)
        : DynamicConvexGeometry(impl), impl_(checked_geometry_storage<Mode>(impl))
    {
    }
    static DynamicConvexGeometryPtr make(
        std::span<const GeometryPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits);
    static DynamicConvexGeometryPtr make_bounded(
        std::span<const DynamicPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits);
    static DynamicConvexGeometryPtr make_bounded_local(GeometryPoint origin, std::span<const DynamicPoint> vertices,
        const std::vector<Face>& faces, GeometryLimits limits);
    static DynamicConvexGeometryPtr from_snapshot(const GeometrySourceSnapshot& source, GeometryLimits limits);
    static DynamicConvexGeometryPtr box(GeometryPoint lower, GeometryPoint upper, GeometryLimits limits);
    static DynamicConvexGeometryPtr tetrahedron(const std::array<DynamicPoint, 4>& vertices, GeometryLimits limits);
    DynamicGeometryValue volume() const;
    DynamicGeometryMoments moments() const;
    DynamicGeometryMoments moments(GeometryPoint origin) const;
    GeometryPoint centroid() const;
    std::vector<DynamicPoint> local_points() const;
    std::array<DynamicGeometryValue, 3> axis_spans() const;
    DynamicSupport support(GeometryPoint normal, GeometryPoint anchor) const;
    DynamicClipResult clip(const GeometryHalfspace& input, bool measure_section) const;
    DynamicClipResult clip_by_face(const DynamicConvexGeometry& source, std::size_t face, bool measure_section) const;
    DynamicClipResult clip_impl(const std::shared_ptr<const DynamicConvexGeometry::Impl>& source, std::size_t plane_id,
        bool measure_section, bool construct_outside = true, bool partition_proof = false, int orientation_override = 0) const;
    DynamicPhaseMeasures phases(const GeometryHalfspace& gas) const;
    bool bounds_prove_disjoint_interiors(const DynamicConvexGeometry& other) const;
    DynamicConvexGeometryPtr intersect(const DynamicConvexGeometry& other) const;
    DynamicConvexGeometryPtr translated(GeometryPoint displacement) const;
    void certify_partition(std::span<const DynamicConvexGeometryPtr> pieces, double relative_budget) const;
    std::vector<char> encode(const GeometryPacketContext& context, std::size_t byte_budget) const;
    static DynamicConvexGeometryPtr decode(std::span<const char> packet, const GeometryPacketContext& expected,
        std::span<const DynamicConvexGeometryPtr> authenticated_sources, GeometryLimits limits);
};
template<ArithmeticMode Mode> class ValueKernel : public DynamicGeometryValue
{
    static_assert(Mode == kernel_mode);

public:
    using DynamicGeometryValue::absolute_error;
    using DynamicGeometryValue::arithmetic_mode;
    using DynamicGeometryValue::value;
    explicit ValueKernel(DynamicGeometryValue value) : DynamicGeometryValue(value) {}
    long double lower() const;
    long double upper() const;
    double as_double() const;
    bool relative_error_at_most(long double target, double tolerance) const;
};
template<ArithmeticMode Mode> class PartitionKernel
{
    static_assert(Mode == kernel_mode);
    const std::vector<DynamicConvexGeometryPtr>& donors_;
    const GeometryLimits& limits_;

public:
    PartitionKernel(const std::vector<DynamicConvexGeometryPtr>& donors, const GeometryLimits& limits)
        : donors_(donors), limits_(limits)
    {
    }
    std::vector<DynamicPartitionPiece> partition(const DynamicConvexGeometry& parent, std::size_t maximum_pieces) const;
};
template<ArithmeticMode Mode> Scalar scalar_value(DynamicGeometryValue a);
template<ArithmeticMode Mode> DynamicGeometryValue bounded(Scalar a);
template<ArithmeticMode Mode> Point point(const DynamicPoint& a);
template<ArithmeticMode Mode> Point point(GeometryPoint a);
template<ArithmeticMode Mode> DynamicPoint bounded(const Point& a);
Point operator+(const Point& a, const Point& b);
Point operator-(const Point& a, const Point& b);
Point operator*(const Point& a, Scalar b);
template<ArithmeticMode Mode> Point scaled(const Point& a, int e);
template<ArithmeticMode Mode> Scalar dot(const Point& a, const Point& b);
template<ArithmeticMode Mode> Point cross(const Point& a, const Point& b);
template<ArithmeticMode Mode> bool same(const Point& a, const Point& b);
template<ArithmeticMode Mode> bool nonzero(const Point& a);
template<ArithmeticMode Mode> int sign(Scalar a);
template<ArithmeticMode Mode> Point transform(const Point& p, const Frame& from, const Frame& to);
template<ArithmeticMode Mode> CoordinateBox transform(const CoordinateBox& box, const Frame& from, const Frame& to);
template<ArithmeticMode Mode> DynamicPoint converted_point(const DynamicPoint& input, ArithmeticMode mode);
template<ArithmeticMode Mode> Scalar raw_distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template<ArithmeticMode Mode> Scalar distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template<ArithmeticMode Mode> Scalar square_root(Scalar x);
std::optional<std::array<long double, 3>> source_point_error(const DynamicPoint& relative,
    const DynamicPointProvenance* original, const Frame& frame, const Point& stored, bool authenticated_shell);
std::optional<long double> source_normal_error(
    const std::array<Point, 3>& vertices, const std::array<std::array<long double, 3>, 3>& errors, const Point& normal);
bool independent_source_normals(const Plane& a, const Plane& b, const Plane& c);
// Zero means that the construction-aware determinant sign is unresolved.
int source_normal_orientation(const Plane& a, const Plane& b, const Plane& c);
bool recover_source_identity(Vertex& vertex, const Frame& frame, const StoredGeometry* parent = nullptr,
    const Vertex* edge_first = nullptr, const Vertex* edge_second = nullptr);
bool same_source_plane(const Plane& a, const Plane& b);
namespace source_witness
{
// Conservative evaluation-error bounds for the supplied represented operation.
// Unsupported environments or inconsistent expressions provide no witness.
long double sum_gap(long double a, long double b, long double result);
long double product_gap(long double a, long double b, long double result);
long double scale_gap(long double a, int exponent, long double result);
} // namespace source_witness

template<ArithmeticMode Mode> Integrals integrate_ordinary(const StoredGeometry& poly, bool moments);
template<ArithmeticMode Mode> Integrals integrate(const StoredGeometry& poly, bool moments);
// Private draft preparation: no measure is admitted or published here.
template<ArithmeticMode Mode> void prepare_finish(StoredGeometry& poly, bool partition_proof);
template<ArithmeticMode Mode>
void finish(StoredGeometry& poly, bool partition_proof = false, bool positive_construction = false);
template<ArithmeticMode Mode>
void validate_connected_boundary(std::size_t vertex_count, const std::vector<Face>& faces);
template<ArithmeticMode Mode>
std::shared_ptr<StoredGeometry> import(GeometryPoint origin, std::span<const DynamicPoint> vertices,
    const std::vector<Face>& faces, GeometryLimits limits, bool authenticated_planarity,
    std::span<const std::shared_ptr<const DynamicPointProvenance>> identities = {});
template<ArithmeticMode Mode> DynamicPoint dynamic_point_average(std::span<const DynamicPoint> vertices);
template<ArithmeticMode Mode>
DynamicPoint dynamic_point_interpolate(const DynamicPoint& a, const DynamicPoint& b, long double fraction);
template<ArithmeticMode Mode> bool dynamic_points_identical(const DynamicPoint& a, const DynamicPoint& b);
template<ArithmeticMode Mode> DynamicGeometryValue dynamic_orientation(const std::array<DynamicPoint, 4>& vertices);
template<ArithmeticMode Mode> int matching_input(const GeometryHalfspace& a, const GeometryHalfspace& b);
template<ArithmeticMode Mode> bool incident(const Vertex& v, const PlanePtr& plane);
template<ArithmeticMode Mode>
DynamicGeometrySection section(
    const StoredGeometry& poly, const std::vector<Vertex>& vertices, const std::vector<std::size_t>& loop);
template<ArithmeticMode Mode>
bool construction_contains(const std::shared_ptr<const Ancestry>& parent, std::shared_ptr<const Ancestry> child);
template<ArithmeticMode Mode>
bool construction_disjoint(std::shared_ptr<const Ancestry> a, std::shared_ptr<const Ancestry> b);
template<ArithmeticMode Mode>
bool separated_bounds(const DynamicConvexGeometry::Impl& a, const DynamicConvexGeometry::Impl& b);
template<ArithmeticMode Mode>
bool structural_cover(
    const std::shared_ptr<const Ancestry>& root, const std::vector<std::shared_ptr<const Ancestry>>& leaves);
template<ArithmeticMode Mode>
DynamicPoint source_point_fan(std::span<const ConvexCellGeometry<ArithmeticMode::Ordinary>* const> sources,
    const PointIdentity<ArithmeticMode::Ordinary>& identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
DynamicPoint dynamic_source_point(const ConvexCellGeometry<ArithmeticMode::Ordinary>& source,
    const PointIdentity<ArithmeticMode::Ordinary>& identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
DynamicPoint dynamic_source_point(std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>> sources,
    const PointIdentity<ArithmeticMode::Ordinary>& identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
DynamicPoint dynamic_point_displaced(const DynamicPoint& source, const DynamicPoint& displacement);
template<ArithmeticMode Mode>
DynamicPoint dynamic_backtrace_constrained(const DynamicPoint& source, GeometryPoint velocity, double timestep,
    std::span<const DynamicFaceConstraint> walls, GeometryLimits limits);
template<ArithmeticMode Mode> Scalar planar_turn(const PlanarVertex& a, const PlanarVertex& b, const PlanarVertex& c);
template<ArithmeticMode Mode>
std::vector<PlaneAuthority> planar_common(
    const std::vector<PlaneAuthority>& first, const std::vector<PlaneAuthority>& second);
template<ArithmeticMode Mode> void planar_charge(const PlanarVertex& vertex);
template<ArithmeticMode Mode>
PlanarLine planar_line(const PlanarVertex& first, const PlanarVertex& second, const PlaneAuthority& primary);
template<ArithmeticMode Mode> Scalar planar_distance(const PlanarVertex& vertex, const PlanarLine& edge);
template<ArithmeticMode Mode> Scalar planar_area(const PlanarPolygon& polygon);
template<ArithmeticMode Mode>
void planar_convex(const PlanarPolygon& polygon, int winding, const PlaneAuthority& primary);
template<ArithmeticMode Mode> void planar_push(PlanarPolygon& polygon, PlanarVertex vertex);
template<ArithmeticMode Mode>
PlanarVertex planar_crossing(
    const PlanarVertex& a, const PlanarVertex& b, Scalar da, Scalar db, const PlanarLine& edge, std::size_t& token);
template<ArithmeticMode Mode> bool planar_line_contact(const PlanarPolygon& polygon, const PlaneAuthority& primary);
template<ArithmeticMode Mode>
DynamicPlanarFaceOverlap dynamic_planar_face_overlap(std::span<const DynamicPoint> first,
    std::span<const DynamicPoint> second, const PlaneAuthority& support, GeometryLimits limits);
} // namespace ordinary_kernel SIMPLEFLUID_LOCAL

} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::details
{
namespace certification_kernel SIMPLEFLUID_LOCAL
{
inline constexpr ArithmeticMode kernel_mode = ArithmeticMode::BoundedCertification;
[[noreturn]] inline void reject(std::string_view reason)
{
    details::reject(kernel_mode, reason);
}
using StorageInterval = StoredValue<kernel_mode>;
using Point = StoredPoint<kernel_mode>;
using Plane = StoredPlane<kernel_mode>;
using PlanePtr = StoredPlanePtr<kernel_mode>;
using Vertex = StoredVertex<kernel_mode>;
using Surface = StoredSurface<kernel_mode>;
using Integrals = StoredIntegrals<kernel_mode>;
using MomentCache = StoredMomentCache<kernel_mode>;
using PlanarVertex = StoredPlanarVertex<kernel_mode>;
using PlanarPolygon = StoredPlanarPolygon<kernel_mode>;
using PlanarLine = StoredPlanarLine<kernel_mode>;
using StoredGeometry = GeometryStorage<kernel_mode>;
using Interval = ::SimpleFluid::detail::directed_fp::Interval<ArithmeticPolicy<kernel_mode>>;
static_assert(sizeof(Interval) == sizeof(StorageInterval) && alignof(Interval) == alignof(StorageInterval));
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
    void byte(std::uint8_t x);
    void tag(GeometryTag value) { byte(static_cast<std::uint8_t>(value)); }
    void word(std::uint64_t x);
    void real(long double x);
    void interval(Interval x);
    void coordinates(const Point& p);
    void frame(const Frame& f);
    void authorities(const std::vector<PlaneAuthority>& keys);
};
template<ArithmeticMode Mode> struct NumericalReader
{
    static_assert(Mode == kernel_mode);
    std::span<const char> bytes;
    std::size_t cursor = 0;
    explicit NumericalReader(std::span<const char> input) : bytes(input) {}
    std::uint8_t byte();
    GeometryTag tag() { return static_cast<GeometryTag>(byte()); }
    std::uint64_t word();
    std::size_t count(std::size_t maximum, std::size_t element);
    long double real();
    Interval interval();
    Point coordinates() { return {interval(), interval(), interval()}; }
    Frame frame();
    std::vector<PlaneAuthority> authorities(const std::set<PlaneAuthority>& registry, GeometryLimits limits);
};
template<ArithmeticMode Mode> class GeometryKernel : public DynamicConvexGeometry
{
    static_assert(Mode == kernel_mode);

public:
    using Impl = StoredGeometry;
    using DynamicConvexGeometry::arithmetic_mode;
    using DynamicConvexGeometry::faces;
    std::shared_ptr<const Impl> impl_;
    using DynamicConvexGeometry::origin;
    using DynamicConvexGeometry::points;
    std::size_t storage_bytes() const { return DynamicConvexGeometry::storage_bytes<Mode>(); }
    static std::vector<DynamicConvexGeometryPtr> validate_donors(
        std::span<const DynamicConvexGeometryPtr> donors, GeometryLimits limits);
    static DynamicPoint backtrace(const DynamicPoint& source, GeometryPoint velocity, double timestep,
        std::span<const DynamicFaceConstraint> walls, GeometryLimits limits);
    explicit GeometryKernel(const std::shared_ptr<const DynamicConvexGeometry::Impl>& impl)
        : DynamicConvexGeometry(impl), impl_(checked_geometry_storage<Mode>(impl))
    {
    }
    static DynamicConvexGeometryPtr make(
        std::span<const GeometryPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits);
    static DynamicConvexGeometryPtr make_bounded(
        std::span<const DynamicPoint> vertices, const std::vector<Face>& faces, GeometryLimits limits);
    static DynamicConvexGeometryPtr make_bounded_local(GeometryPoint origin, std::span<const DynamicPoint> vertices,
        const std::vector<Face>& faces, GeometryLimits limits);
    static DynamicConvexGeometryPtr from_snapshot(const GeometrySourceSnapshot& source, GeometryLimits limits);
    static DynamicConvexGeometryPtr box(GeometryPoint lower, GeometryPoint upper, GeometryLimits limits);
    static DynamicConvexGeometryPtr tetrahedron(const std::array<DynamicPoint, 4>& vertices, GeometryLimits limits);
    DynamicGeometryValue volume() const;
    DynamicGeometryMoments moments() const;
    DynamicGeometryMoments moments(GeometryPoint origin) const;
    GeometryPoint centroid() const;
    std::vector<DynamicPoint> local_points() const;
    std::array<DynamicGeometryValue, 3> axis_spans() const;
    DynamicSupport support(GeometryPoint normal, GeometryPoint anchor) const;
    DynamicClipResult clip(const GeometryHalfspace& input, bool measure_section) const;
    DynamicClipResult clip_by_face(const DynamicConvexGeometry& source, std::size_t face, bool measure_section) const;
    DynamicClipResult clip_impl(const std::shared_ptr<const DynamicConvexGeometry::Impl>& source, std::size_t plane_id,
        bool measure_section, bool construct_outside = true, bool partition_proof = false, int orientation_override = 0) const;
    DynamicPhaseMeasures phases(const GeometryHalfspace& gas) const;
    bool bounds_prove_disjoint_interiors(const DynamicConvexGeometry& other) const;
    DynamicConvexGeometryPtr intersect(const DynamicConvexGeometry& other) const;
    DynamicConvexGeometryPtr translated(GeometryPoint displacement) const;
    void certify_partition(std::span<const DynamicConvexGeometryPtr> pieces, double relative_budget) const;
    std::vector<char> encode(const GeometryPacketContext& context, std::size_t byte_budget) const;
    static DynamicConvexGeometryPtr decode(std::span<const char> packet, const GeometryPacketContext& expected,
        std::span<const DynamicConvexGeometryPtr> authenticated_sources, GeometryLimits limits);
};
template<ArithmeticMode Mode> class ValueKernel : public DynamicGeometryValue
{
    static_assert(Mode == kernel_mode);

public:
    using DynamicGeometryValue::absolute_error;
    using DynamicGeometryValue::arithmetic_mode;
    using DynamicGeometryValue::value;
    explicit ValueKernel(DynamicGeometryValue value) : DynamicGeometryValue(value) {}
    long double lower() const;
    long double upper() const;
    double as_double() const;
    bool relative_error_at_most(long double target, double tolerance) const;
};
template<ArithmeticMode Mode> class PartitionKernel
{
    static_assert(Mode == kernel_mode);
    const std::vector<DynamicConvexGeometryPtr>& donors_;
    const GeometryLimits& limits_;

public:
    PartitionKernel(const std::vector<DynamicConvexGeometryPtr>& donors, const GeometryLimits& limits)
        : donors_(donors), limits_(limits)
    {
    }
    std::vector<DynamicPartitionPiece> partition(const DynamicConvexGeometry& parent, std::size_t maximum_pieces) const;
};
template<ArithmeticMode Mode> long double directed_add(long double a, long double b, bool up);
template<ArithmeticMode Mode> long double directed_multiply(long double a, long double b, bool up);
template<ArithmeticMode Mode> long double directed_divide(long double a, long double b, bool up);
template<ArithmeticMode Mode> Interval enclosure(DynamicGeometryValue a);
template<ArithmeticMode Mode> DynamicGeometryValue bounded(Interval a);
template<ArithmeticMode Mode> Point point(const DynamicPoint& a);
template<ArithmeticMode Mode> Point point(GeometryPoint a);
template<ArithmeticMode Mode> DynamicPoint bounded(const Point& a);
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
template<ArithmeticMode Mode> DynamicPoint converted_point(const DynamicPoint& input, ArithmeticMode mode);
template<ArithmeticMode Mode> Interval raw_distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template<ArithmeticMode Mode> Interval distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
template<ArithmeticMode Mode> void tighten_incident_coordinates(Vertex& vertex, const Frame& frame);
template<ArithmeticMode Mode> Interval square_root(Interval x);
template<ArithmeticMode Mode> Integrals integrate(const StoredGeometry& poly, bool moments);
template<ArithmeticMode Mode>
void finish(StoredGeometry& poly, bool partition_proof = false, bool positive_construction = false);
template<ArithmeticMode Mode>
void validate_connected_boundary(std::size_t vertex_count, const std::vector<Face>& faces);
template<ArithmeticMode Mode>
std::shared_ptr<StoredGeometry> import(GeometryPoint origin, std::span<const DynamicPoint> vertices,
    const std::vector<Face>& faces, GeometryLimits limits, bool authenticated_planarity,
    std::span<const std::shared_ptr<const DynamicPointProvenance>> identities = {});
template<ArithmeticMode Mode> DynamicPoint dynamic_point_average(std::span<const DynamicPoint> vertices);
template<ArithmeticMode Mode>
DynamicPoint dynamic_point_interpolate(const DynamicPoint& a, const DynamicPoint& b, long double fraction);
template<ArithmeticMode Mode> bool dynamic_points_identical(const DynamicPoint& a, const DynamicPoint& b);
template<ArithmeticMode Mode> DynamicGeometryValue dynamic_orientation(const std::array<DynamicPoint, 4>& vertices);
template<ArithmeticMode Mode> int matching_input(const GeometryHalfspace& a, const GeometryHalfspace& b);
template<ArithmeticMode Mode> bool incident(const Vertex& v, const PlanePtr& plane);
template<ArithmeticMode Mode>
Interval refine_triangle_distance(const Point& coordinates, const Frame& frame, const Vertex& a, const Vertex& b,
    const Vertex& c, const Frame& triangle_frame, const PlanePtr& plane, int orientation, Interval result);
template<ArithmeticMode Mode>
DynamicGeometrySection section(
    const StoredGeometry& poly, const std::vector<Vertex>& vertices, const std::vector<std::size_t>& loop);
template<ArithmeticMode Mode>
bool construction_contains(const std::shared_ptr<const Ancestry>& parent, std::shared_ptr<const Ancestry> child);
template<ArithmeticMode Mode>
bool construction_disjoint(std::shared_ptr<const Ancestry> a, std::shared_ptr<const Ancestry> b);
template<ArithmeticMode Mode>
bool separated_bounds(const DynamicConvexGeometry::Impl& a, const DynamicConvexGeometry::Impl& b);
template<ArithmeticMode Mode>
bool structural_cover(
    const std::shared_ptr<const Ancestry>& root, const std::vector<std::shared_ptr<const Ancestry>>& leaves);
template<ArithmeticMode Mode>
DynamicPoint source_point_fan(std::span<const ConvexCellGeometry<ArithmeticMode::Ordinary>* const> sources,
    const PointIdentity<ArithmeticMode::Ordinary>& identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
DynamicPoint dynamic_source_point(const ConvexCellGeometry<ArithmeticMode::Ordinary>& source,
    const PointIdentity<ArithmeticMode::Ordinary>& identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
DynamicPoint dynamic_source_point(std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>> sources,
    const PointIdentity<ArithmeticMode::Ordinary>& identity, ArithmeticMode mode);
template<ArithmeticMode Mode>
DynamicPoint dynamic_point_displaced(const DynamicPoint& source, const DynamicPoint& displacement);
template<ArithmeticMode Mode>
DynamicPoint dynamic_backtrace_constrained(const DynamicPoint& source, GeometryPoint velocity, double timestep,
    std::span<const DynamicFaceConstraint> walls, GeometryLimits limits);
template<ArithmeticMode Mode> Interval planar_turn(const PlanarVertex& a, const PlanarVertex& b, const PlanarVertex& c);
template<ArithmeticMode Mode>
std::vector<PlaneAuthority> planar_common(
    const std::vector<PlaneAuthority>& first, const std::vector<PlaneAuthority>& second);
template<ArithmeticMode Mode> void planar_charge(const PlanarVertex& vertex);
template<ArithmeticMode Mode>
PlanarLine planar_line(const PlanarVertex& first, const PlanarVertex& second, const PlaneAuthority& primary);
template<ArithmeticMode Mode> Interval planar_distance(const PlanarVertex& vertex, const PlanarLine& edge);
template<ArithmeticMode Mode> Interval planar_area(const PlanarPolygon& polygon);
template<ArithmeticMode Mode>
void planar_convex(const PlanarPolygon& polygon, int winding, const PlaneAuthority& primary);
template<ArithmeticMode Mode> void planar_push(PlanarPolygon& polygon, PlanarVertex vertex);
template<ArithmeticMode Mode>
PlanarVertex planar_crossing(
    const PlanarVertex& a, const PlanarVertex& b, Interval da, Interval db, const PlanarLine& edge, std::size_t& token);
template<ArithmeticMode Mode> bool planar_line_contact(const PlanarPolygon& polygon, const PlaneAuthority& primary);
template<ArithmeticMode Mode>
DynamicPlanarFaceOverlap dynamic_planar_face_overlap(std::span<const DynamicPoint> first,
    std::span<const DynamicPoint> second, const PlaneAuthority& support, GeometryLimits limits);
} // namespace certification_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
