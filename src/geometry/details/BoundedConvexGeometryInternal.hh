#pragma once
#include "SimpleFluidExport.hh"
#include "utils/BoundedInterval.hh"
/** @file BoundedConvexGeometryInternal.hh
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
    ArithmeticMode arithmetic_mode;
};

class SIMPLEFLUID_LOCAL BoundedGeometryAccess
{
public:
    static std::shared_ptr<const BoundedPointProvenance> identity(const BoundedPoint& p);
    static std::size_t identity_bytes(const std::shared_ptr<const BoundedPointProvenance>& p);
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
    void charge(std::size_t count, std::size_t size);
    void pair();
};
extern thread_local Work* active_work;
BoundedGeometryLimits mode_limits(ArithmeticMode mode);
void require_mode(ArithmeticMode a, ArithmeticMode b);

struct Operation
{
    Work work;
    Work* previous;
    explicit Operation(BoundedGeometryLimits limits);
    ~Operation() { active_work = previous; }
};
void operation();

using ::SimpleFluid::detail::directed_fp::infinity;
// Persisted intervals carry coordinates only. Arithmetic must first bind a
// fixed policy; deleted hooks turn accidental storage arithmetic into an error.
struct StoragePolicy
{
    static bool ordinary() = delete;
    static void interval_operation() = delete;
    static void operation() = delete;
    static void finite(long double value)
    {
        if (!std::isfinite(value))
            bounded_geometry_detail::reject("arithmetic range unresolved (overflow or subnormal wider-FP result).");
    }
    [[noreturn]] static void reject(const char* reason) { bounded_geometry_detail::reject(reason); }
};

using Interval = ::SimpleFluid::detail::directed_fp::Interval<StoragePolicy>;

using Point = std::array<Interval, 3>;

struct Frame
{
    GeometryPoint origin{};
    int exponent = 0;
};

// Spatial coordinate ranges are not uncertainty scalars: keep their full extent
// even when ordinary arithmetic uses only each point's nominal coordinates.
struct CoordinateRange
{
    long double lo = 0, hi = 0;
};

using CoordinateBox = std::array<CoordinateRange, 3>;

struct Plane
{
    Frame frame;
    Point normal, anchor;
    Interval offset;
    std::optional<GeometryHalfspace> input;
    std::vector<BoundedPlaneAuthority> authorities;
    std::vector<std::weak_ptr<const BoundedPointProvenance>> source_vertices;
};

using PlanePtr = std::shared_ptr<const Plane>;

struct Vertex
{
    Point coordinates;
    std::vector<PlanePtr> incident;
    std::vector<BoundedPlaneAuthority> authorities;
    std::shared_ptr<const BoundedPointProvenance> source_identity;
};

using Face = std::vector<std::size_t>;

struct Split;

struct Ancestry
{
    std::shared_ptr<const Ancestry> parent;
    std::shared_ptr<const Split> split;
    bool outside = false;
    std::size_t depth = 0;
};

struct Split
{
    std::shared_ptr<const Ancestry> parent;
};

struct PartitionCertificate
{
    std::shared_ptr<const Ancestry> parent;
    std::vector<std::weak_ptr<const Ancestry>> members;
};

struct Surface
{
    PlanePtr plane;
    int orientation = 1;
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
    GeometryPoint reference_origin{};
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    std::vector<Surface> surfaces;
    std::vector<BoundedPoint> displayed;
    BoundedGeometryLimits limits;
    CoordinateBox bounds;
    Interval volume;
    bool partition_proof_only = false;
    std::shared_ptr<const Ancestry> ancestry = std::make_shared<Ancestry>();
    std::shared_ptr<const PartitionCertificate> partition_certificate;
    std::shared_ptr<MomentCache> moment_cache = std::make_shared<MomentCache>();
};
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

BoundedPoint bounded_source_point(
    const ConvexCellGeometry& source, std::span<const char> identity, ArithmeticMode mode);
BoundedPoint bounded_source_point(
    std::span<const ConvexGeometryPtr> sources, std::span<const char> identity, ArithmeticMode mode);
BoundedPoint bounded_point_displaced(const BoundedPoint& source, const BoundedPoint& displacement);
BoundedPoint bounded_backtrace_constrained(const BoundedPoint& source, GeometryPoint velocity, double timestep,
    std::span<const BoundedFaceConstraint> walls, BoundedGeometryLimits limits);
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
} // namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes
{

BoundedPlanarFaceOverlap bounded_planar_face_overlap(std::span<const BoundedPoint> first,
    std::span<const BoundedPoint> second, const BoundedPlaneAuthority& support, BoundedGeometryLimits limits);
}
