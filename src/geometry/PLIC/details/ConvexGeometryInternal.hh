#pragma once
#include "SimpleFluidExport.hh"
#include "utils/BoundedInterval.hh"
#include "utils/OrdinaryScalar.hh"
#include <type_traits>
/** @file ConvexGeometryInternal.hh
 * @brief Policy-specific finite construction storage and structural convex partitions.
 */
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/PlanarFace.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
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
#include <string_view>
#include <utility>
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace SimpleFluid::PLIC
{
namespace details
{
}
using namespace details;
struct SIMPLEFLUID_LOCAL DynamicPointProvenance
{
    std::array<long double, 3> value, absolute_error;
    std::vector<PlaneAuthority> supports;
    ArithmeticMode arithmetic_mode;
    /** Only explicit global-coordinate source constructors issue this marker.
     * Rounded clipping/interpolation outputs do not acquire it implicitly. */
    bool represented_native_source = false;
};

class SIMPLEFLUID_LOCAL DynamicGeometryAccess
{
public:
    static std::shared_ptr<const DynamicPointProvenance> identity(const DynamicPoint& p);
    static std::size_t identity_bytes(const std::shared_ptr<const DynamicPointProvenance>& p);
    static std::size_t storage_bytes(const DynamicPoint& p);
    static bool same_source(const DynamicPoint& a, const DynamicPoint& b);
    static void seal(DynamicPoint& point, std::vector<PlaneAuthority> supports);
    static void seal_represented_source(DynamicPoint& point);
    static bool same_represented_source(
        const std::shared_ptr<const DynamicPointProvenance>&, const std::shared_ptr<const DynamicPointProvenance>&);
};
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

namespace details SIMPLEFLUID_LOCAL
{
extern thread_local GeometryStatistics statistics;
[[noreturn]] void reject(std::string_view reason);
[[noreturn]] void reject(ArithmeticMode mode, std::string_view reason);

struct Work
{
    GeometryLimits limits;
    std::size_t operations = 0, bytes = 0, pairs = 0, operation_limit = 0;
    explicit Work(GeometryLimits requested);
    void charge(std::size_t count, std::size_t size);
    void pair();
};
extern thread_local Work* active_work;
struct GeometrySourceSnapshot
{
    NumericalConvexShell<ArithmeticMode::Ordinary> shell;
    std::size_t preparation_bytes = 0, preparation_operations = 0, preparation_pairs = 0;
    const Work* charged_work = nullptr;
};
GeometrySourceSnapshot prepare_source_geometry(const ConvexCellGeometry<ArithmeticMode::Ordinary>&, GeometryLimits);
void charge_source_preparation(const GeometrySourceSnapshot&);

GeometryLimits mode_limits(ArithmeticMode mode);
void require_mode(ArithmeticMode a, ArithmeticMode b);

struct Operation
{
    Work work;
    Work* previous;
    explicit Operation(GeometryLimits limits);
    ~Operation() { active_work = previous; }
};
void operation();

using ::SimpleFluid::detail::directed_fp::infinity;
// Persisted intervals carry coordinates only. Arithmetic must first bind a
// fixed policy; deleted hooks turn accidental storage arithmetic into an error.
template<ArithmeticMode Mode> struct StoragePolicy
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    static constexpr bool ordinary() { return Mode == ArithmeticMode::Ordinary; }
    static void interval_operation() = delete;
    static void scalar_operation() = delete;
    static void operation() = delete;
    static void finite(long double value)
    {
        if (!std::isfinite(value))
            details::reject(Mode, "nonfinite arithmetic result.");
    }
    [[noreturn]] static void reject(const char* reason) { details::reject(Mode, reason); }
};

template<ArithmeticMode Mode>
using StoredValue = std::conditional_t<Mode == ArithmeticMode::Ordinary,
    ::SimpleFluid::detail::ordinary_fp::Scalar<StoragePolicy<Mode>>,
    ::SimpleFluid::detail::directed_fp::Interval<StoragePolicy<Mode>>>;

template<ArithmeticMode Mode> using StoredPoint = std::array<StoredValue<Mode>, 3>;

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

template<ArithmeticMode Mode> struct PlaneNormalWitness
{
};
template<> struct PlaneNormalWitness<ArithmeticMode::Ordinary>
{
    // Optional absolute component error for the original constructed normal.
    // This supports source-identity rank proofs, not geometric admission.
    std::optional<long double> normal_error;
    std::vector<char> native_support; ///< Retained source support identity, separate from numerical coefficients.
};

template<ArithmeticMode Mode> struct StoredPlane : PlaneNormalWitness<Mode>
{
    Frame frame;
    StoredPoint<Mode> normal, anchor;
    StoredValue<Mode> offset;
    std::optional<GeometryHalfspace> input;
    std::vector<PlaneAuthority> authorities;
    std::vector<std::weak_ptr<const DynamicPointProvenance>> source_vertices;
};

template<ArithmeticMode Mode> using StoredPlanePtr = std::shared_ptr<const StoredPlane<Mode>>;

template<ArithmeticMode Mode> struct StoredVertex
{
    StoredPoint<Mode> coordinates;
    std::vector<StoredPlanePtr<Mode>> incident;
    std::vector<PlaneAuthority> authorities;
    std::shared_ptr<const DynamicPointProvenance> source_identity;
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

template<ArithmeticMode Mode> struct StoredSurface
{
    StoredPlanePtr<Mode> plane;
    int orientation = 1;
};

template<ArithmeticMode Mode> struct StoredIntegrals
{
    StoredValue<Mode> volume;
    StoredPoint<Mode> first;
};

template<ArithmeticMode Mode> struct StoredMomentCache
{
    std::once_flag prepared;
    std::optional<StoredIntegrals<Mode>> integrals;
};
} // namespace details SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

struct SIMPLEFLUID_LOCAL DynamicConvexGeometry::Impl
{
    Frame frame;
    GeometryPoint reference_origin{};
    std::vector<Face> faces;
    std::vector<DynamicPoint> displayed;
    GeometryLimits limits;
    CoordinateBox bounds;
    bool partition_proof_only = false;
    std::shared_ptr<const Ancestry> ancestry = std::make_shared<Ancestry>();
    std::shared_ptr<const PartitionCertificate> partition_certificate;
};

namespace details
{
template<ArithmeticMode Mode> struct GeometryStorage : DynamicConvexGeometry::Impl
{
    GeometryStorage() { limits.arithmetic_mode = Mode; }
    std::vector<StoredVertex<Mode>> vertices;
    std::vector<StoredSurface<Mode>> surfaces;
    StoredValue<Mode> volume;
    std::shared_ptr<StoredMomentCache<Mode>> moment_cache = std::make_shared<StoredMomentCache<Mode>>();
};

template<ArithmeticMode Mode>
std::shared_ptr<const GeometryStorage<Mode>> checked_geometry_storage(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>& source)
{
    require_mode(source->limits.arithmetic_mode, Mode);
    return std::static_pointer_cast<const GeometryStorage<Mode>>(source);
}
} // namespace details

} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

std::vector<PlaneAuthority> dynamic_common_supports(std::span<const DynamicPoint> points);
DynamicPoint dynamic_point_average(std::span<const DynamicPoint> vertices);
DynamicPoint dynamic_point_interpolate(const DynamicPoint& a, const DynamicPoint& b, long double fraction);
bool certification_environment();
GeometryStatistics geometry_statistics();
void reset_geometry_statistics();
void record_ledger_operations(std::uint64_t count);
std::size_t dynamic_point_storage_bytes(const DynamicPoint& p);
bool dynamic_points_identical(const DynamicPoint& a, const DynamicPoint& b);
DynamicGeometryValue dynamic_orientation(const std::array<DynamicPoint, 4>& vertices);
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

DynamicPoint dynamic_source_point(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>& source, std::span<const char> identity, ArithmeticMode mode);
DynamicPoint dynamic_source_point(std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>> sources,
    std::span<const char> identity, ArithmeticMode mode);
DynamicPoint dynamic_point_displaced(const DynamicPoint& source, const DynamicPoint& displacement);
DynamicPoint dynamic_backtrace_constrained(const DynamicPoint& source, GeometryPoint velocity, double timestep,
    std::span<const DynamicFaceConstraint> walls, GeometryLimits limits);
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

const void* geometry_statistics_domain();
}
namespace SimpleFluid::PLIC
{

namespace details SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode> struct StoredPlanarVertex
{
    std::array<StoredValue<Mode>, 2> coordinates;
    StoredPoint<Mode> physical;
    std::vector<PlaneAuthority> supports;
    std::size_t token = 0;
};

template<ArithmeticMode Mode> using StoredPlanarPolygon = std::vector<StoredPlanarVertex<Mode>>;

template<ArithmeticMode Mode> struct StoredPlanarLine
{
    StoredPlanarVertex<Mode> first, second;
    std::vector<PlaneAuthority> supports;
};
} // namespace details SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

DynamicPlanarFaceOverlap dynamic_planar_face_overlap(std::span<const DynamicPoint> first,
    std::span<const DynamicPoint> second, const PlaneAuthority& support, GeometryLimits limits);
}
