#pragma once
#include "SimpleFluidExport.hh"
#include "utils/BoundedInterval.hh"
/** @file BoundedConvexGeometry.cc
 * @brief Bounded wider-FP construction and structural convex partitions.
 */
#include "geometry/BoundedConvexGeometry.hh"
#include "geometry/BoundedPlanarFace.hh"
#include "geometry/ExactPlanarFace.hh"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace SimpleFluid::Meshes
{
namespace bounded_geometry_detail
{
}
using namespace bounded_geometry_detail;
struct SIMPLEFLUID_LOCAL BoundedPointProvenance
{
    std::array<long double, 3> value, absolute_error;
    std::vector<BoundedPlaneAuthority> supports;
};

class SIMPLEFLUID_LOCAL BoundedGeometryAccess
{
public:
    static std::size_t storage_bytes(const BoundedPoint& p);
    static bool same_source(const BoundedPoint& a, const BoundedPoint& b);
    static void seal(BoundedPoint& point, std::vector<BoundedPlaneAuthority> supports);
};
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{

namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
{
extern thread_local BoundedGeometryStatistics statistics;
[[noreturn]] void reject(const char* reason);

struct Work
{
    BoundedGeometryLimits limits;
    std::size_t operations = 0, bytes = 0, pairs = 0, operation_limit = 0;
    explicit Work(BoundedGeometryLimits requested);
    void operation();
    void charge(std::size_t count, std::size_t size);
    void pair();
};
extern thread_local Work* active_work;

struct Operation
{
    Work work;
    Work* previous;
    explicit Operation(BoundedGeometryLimits limits);
    ~Operation() { active_work = previous; }
};
void operation();
void interval_operation();
void finite(long double value);

using ::SimpleFluid::detail::directed_fp::infinity;
struct ScalarPolicy
{
    static bool ordinary() { return false; }
    static void interval_operation() { bounded_geometry_detail::interval_operation(); }
    static void operation() { bounded_geometry_detail::operation(); }
    static void finite(long double value) { bounded_geometry_detail::finite(value); }
    [[noreturn]] static void reject(const char* reason) { bounded_geometry_detail::reject(reason); }
};
long double directed_add(long double a, long double b, bool up);
long double directed_multiply(long double a, long double b, bool up);
long double directed_divide(long double a, long double b, bool up);

using Interval = ::SimpleFluid::detail::directed_fp::Interval<ScalarPolicy>;
using ::SimpleFluid::detail::directed_fp::scaled;

Interval enclosure(BoundedGeometryValue a);
BoundedGeometryValue bounded(Interval a);

using Point = std::array<Interval, 3>;
Point point(const BoundedPoint& a);
Point point(GeometryPoint a);
BoundedPoint bounded(const Point& a);
Point operator+(const Point& a, const Point& b);
Point operator-(const Point& a, const Point& b);
Point operator*(const Point& a, Interval b);
Point scaled(const Point& a, int e);
Interval dot(const Point& a, const Point& b);
Point cross(const Point& a, const Point& b);
bool same(const Point& a, const Point& b);
bool nonzero(const Point& a);
int sign(Interval a);

struct Frame
{
    GeometryPoint origin{};
    int exponent = 0;
};
Point transform(const Point& p, const Frame& from, const Frame& to);

struct Plane
{
    Frame frame;
    Point normal, anchor;
    Interval offset;
    std::optional<GeometryHalfspace> input;
    std::vector<BoundedPlaneAuthority> authorities;
};

using PlanePtr = std::shared_ptr<const Plane>;

struct Vertex
{
    Point coordinates;
    std::vector<PlanePtr> incident;
    std::vector<BoundedPlaneAuthority> authorities;
};

using Face = std::vector<std::size_t>;

struct Split;

struct Ancestry
{
    std::shared_ptr<const Ancestry> parent;
    std::shared_ptr<const Split> split;
    bool outside = false;
};

struct Split
{
    std::shared_ptr<const Ancestry> parent;
};

struct Surface
{
    PlanePtr plane;
    int orientation = 1;
};

int matching_input(const GeometryHalfspace&, const GeometryHalfspace&);
Interval distance(const Vertex& vertex, const Frame& frame, const PlanePtr& plane);
Interval square_root(Interval x);

struct Accumulator
{
    Interval enclosure;
    long double sum = 0, correction = 0;
    void add(Interval term);
    Interval result() const;
};

struct Integrals
{
    Interval volume;
    Point first;
};

struct MomentCache
{
    std::once_flag prepared;
    std::optional<Integrals> integrals;
};
} // namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{

struct SIMPLEFLUID_LOCAL BoundedConvexGeometry::Impl
{
    Frame frame;
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    std::vector<Surface> surfaces;
    std::vector<BoundedPoint> displayed;
    BoundedGeometryLimits limits;
    Interval volume;
    std::shared_ptr<const Ancestry> ancestry = std::make_shared<Ancestry>();
    std::shared_ptr<MomentCache> moment_cache = std::make_shared<MomentCache>();
};
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{

namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
{
Integrals integrate(const BoundedConvexGeometry::Impl& poly, bool moments);
void finish(BoundedConvexGeometry::Impl& poly);
void validate_connected_boundary(std::size_t vertex_count, const std::vector<Face>& faces);
std::shared_ptr<BoundedConvexGeometry::Impl> import(GeometryPoint origin, std::span<const BoundedPoint> vertices,
    const std::vector<Face>& faces, BoundedGeometryLimits limits, bool authenticated_planarity);
} // namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{

std::vector<BoundedPlaneAuthority> bounded_common_supports(std::span<const BoundedPoint> points);
BoundedPoint bounded_point_average(std::span<const BoundedPoint> vertices);
BoundedPoint bounded_point_interpolate(const BoundedPoint& a, const BoundedPoint& b, long double fraction);
bool bounded_geometry_environment();
BoundedGeometryStatistics bounded_geometry_statistics();
void reset_bounded_geometry_statistics();
void bounded_record_ledger_operations(std::uint64_t count);
std::size_t bounded_point_storage_bytes(const BoundedPoint& p);
bool bounded_points_identical(const BoundedPoint& a, const BoundedPoint& b);
BoundedGeometryValue bounded_orientation(const std::array<BoundedPoint, 4>& vertices);
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{

namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
{
int matching_input(const GeometryHalfspace& a, const GeometryHalfspace& b);
bool incident(const Vertex& v, const PlanePtr& plane);
BoundedGeometrySection section(
    const BoundedConvexGeometry::Impl& poly, const std::vector<Vertex>& vertices, const std::vector<std::size_t>& loop);
bool structural_cover(
    const std::shared_ptr<const Ancestry>& root, const std::vector<std::shared_ptr<const Ancestry>>& leaves);
BoundedPoint source_point_fan(std::span<const ConvexCellGeometry* const> sources, std::span<const char> identity);
} // namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{

BoundedPoint bounded_source_point(const ConvexCellGeometry& source, std::span<const char> identity);
BoundedPoint bounded_source_point(std::span<const ConvexGeometryPtr> sources, std::span<const char> identity);
BoundedPoint bounded_point_displaced(const BoundedPoint& source, const BoundedPoint& displacement);
BoundedPoint bounded_backtrace_constrained(const BoundedPoint& source, GeometryPoint velocity, double timestep,
    std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits);
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{

namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
{
struct NumericalWriter
{
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

struct NumericalReader
{
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
} // namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{

const void* bounded_geometry_statistics_domain();
}
namespace SimpleFluid::Meshes
{

namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
{
struct PlanarVertex
{
    std::array<Interval, 2> coordinates;
    Point physical;
    std::vector<BoundedPlaneAuthority> supports;
    std::size_t token = 0;
};

using PlanarPolygon = std::vector<PlanarVertex>;

struct PlanarLine
{
    PlanarVertex first, second;
    std::vector<BoundedPlaneAuthority> supports;
};
Interval planar_turn(const PlanarVertex& a, const PlanarVertex& b, const PlanarVertex& c);
std::vector<BoundedPlaneAuthority> planar_common(
    const std::vector<BoundedPlaneAuthority>& first, const std::vector<BoundedPlaneAuthority>& second);
void planar_charge(const PlanarVertex& vertex);
PlanarLine planar_line(const PlanarVertex& first, const PlanarVertex& second, const BoundedPlaneAuthority& primary);
Interval planar_distance(const PlanarVertex& vertex, const PlanarLine& edge);
Interval planar_area(const PlanarPolygon& polygon);
void planar_convex(const PlanarPolygon& polygon, int winding, const BoundedPlaneAuthority& primary);
void planar_push(PlanarPolygon& polygon, PlanarVertex vertex);
PlanarVertex planar_crossing(
    const PlanarVertex& a, const PlanarVertex& b, Interval da, Interval db, const PlanarLine& edge, std::size_t& token);
bool planar_line_contact(const PlanarPolygon& polygon, const BoundedPlaneAuthority& primary);
} // namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{

BoundedPlanarFaceOverlap bounded_planar_face_overlap(std::span<const BoundedPoint> first,
    std::span<const BoundedPoint> second, const BoundedPlaneAuthority& support, BoundedGeometryLimits limits);
}
