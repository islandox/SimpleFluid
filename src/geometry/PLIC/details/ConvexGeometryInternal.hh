#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "utils/BoundedInterval.hh"
#include "utils/OrdinaryScalar.hh"
#include <type_traits>
/** @file ConvexGeometryInternal.hh
 * @brief Policy-specific finite construction storage and structural convex partitions.
 */
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/PlanarFace.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/OriginalSupportAuthority.hh"
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
    /** Explicit represented global-input identity in the selected finite mode.
     * This is not native support, ownership or trial authority. Rounded or
     * uncertain construction outputs do not acquire it implicitly. */
    bool represented_native_source = false;
    bool constructed = false;
    bool numerical_only = false;
    std::shared_ptr<const DynamicPointProvenance> source_origin;
    std::vector<std::array<std::shared_ptr<const DynamicPointProvenance>, 2>> source_edges;
};

class SIMPLEFLUID_LOCAL DynamicGeometryAccess
{
public:
    static std::shared_ptr<const DynamicPointProvenance> identity(const DynamicPoint& p);
    static std::size_t identity_bytes(
        const std::shared_ptr<const DynamicPointProvenance>& p, bool include_support_payloads = true);
    static void retain_supports(
        const std::shared_ptr<const DynamicPointProvenance>&, GeometryAllocationLedger&, std::size_t&, std::size_t);
    static std::size_t storage_bytes(const DynamicPoint& p, bool include_support_payloads = true);
    static bool same_source(const DynamicPoint& a, const DynamicPoint& b);
    static void seal(DynamicPoint& point, std::vector<PlaneAuthority> supports);
    static void seal_represented_source(DynamicPoint& point);
    static void seal_represented_source(DynamicPoint&, std::vector<PlaneAuthority>);
    static bool same_represented_source(
        const std::shared_ptr<const DynamicPointProvenance>&, const std::shared_ptr<const DynamicPointProvenance>&);
    /** Borrow an immutable ancestor owner slot. The named input owner must
     * remain alive and unchanged while the reference is used. Rvalues cannot
     * pin that lifetime. Use original_source when retaining ownership. */
    static const std::shared_ptr<const DynamicPointProvenance>& borrowed_original_source(
        const std::shared_ptr<const DynamicPointProvenance>&);
    static const std::shared_ptr<const DynamicPointProvenance>& borrowed_original_source(
        std::shared_ptr<const DynamicPointProvenance>&&) = delete;
    static const std::shared_ptr<const DynamicPointProvenance>& borrowed_original_source(
        const std::shared_ptr<const DynamicPointProvenance>&&) = delete;
    static std::shared_ptr<const DynamicPointProvenance> original_source(
        const std::shared_ptr<const DynamicPointProvenance>&);
    static void seal_export(DynamicPoint&, const std::shared_ptr<const DynamicPointProvenance>&,
        std::vector<PlaneAuthority>, bool numerical_only = false);
    static void seal_generated(DynamicPoint&,
        std::vector<std::array<std::shared_ptr<const DynamicPointProvenance>, 2>>,
        std::vector<PlaneAuthority> = {}, bool numerical_only = false);
    static void seal_numerical(DynamicPoint&);
    static const std::shared_ptr<const DynamicConvexGeometry::Impl>& implementation(const DynamicConvexGeometry&);
    static DynamicConvexGeometryPtr geometry(std::shared_ptr<const DynamicConvexGeometry::Impl>);
};
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

namespace details SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode>
[[nodiscard]] PlaneAuthority original_support_authority(const PlaneSupportIdentity<Mode>& identity)
{
    return OriginalSupportAuthorityAccess::issue(identity);
}
extern constinit thread_local GeometryStatistics statistics;
[[noreturn]] void reject(std::string_view reason);
[[noreturn]] void reject(ArithmeticMode mode, std::string_view reason);

struct Work
{
    GeometryLimits limits;
    std::size_t operations = 0, bytes = 0, pairs = 0, operation_limit = 0;
    GeometryAllocationLedger allocations;
    explicit Work(GeometryLimits requested);
    template<class Identity> void retain_identity(const Identity& identity)
    {
        allocations.retain(identity, bytes, limits.byte_budget);
    }
    template<ArithmeticMode Mode> void retain_geometry(const DynamicConvexGeometryPtr& geometry)
    {
        allocations.retain<Mode>(geometry, bytes, limits.byte_budget);
    }
    void charge(std::size_t count, std::size_t size);
    void pair();
};
extern constinit thread_local Work* active_work;
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

struct GeometryConstruction;
struct GeometryClipConstruction
{
    std::shared_ptr<const GeometryConstruction> parent;
    std::shared_ptr<const void> plane;
    std::vector<std::shared_ptr<const DynamicPointProvenance>> retained_points;
    ArithmeticMode mode = ArithmeticMode::Ordinary;
    int orientation = 1;
    bool measure_section = false, construct_outside = true, partition_proof = false;
};
struct GeometryConstruction
{
    enum class Kind { Source, Clip, Rebase };
    Kind kind = Kind::Source;
    std::shared_ptr<const DynamicConvexGeometry::Impl> source;
    std::vector<std::shared_ptr<const DynamicPointProvenance>> retained_points;
    std::shared_ptr<const GeometryClipConstruction> clip;
    bool outside = false;
    std::shared_ptr<const GeometryConstruction> parent;
    GeometryHalfspace rebase_input;
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
    std::optional<PlaneSupportIdentity<ArithmeticMode::Ordinary>>
        native_support; ///< Retained source support identity, separate from numerical coefficients.
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
    bool numerical_only = false;
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
    bool numerical_only = false;
    std::shared_ptr<const Ancestry> ancestry = std::make_shared<Ancestry>();
    std::shared_ptr<const PartitionCertificate> partition_certificate;
    std::shared_ptr<const GeometryConstruction> construction;
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
template<ArithmeticMode Mode>
std::shared_ptr<const GeometryConstruction> construction_source(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&);
std::size_t construction_storage_bytes(const std::shared_ptr<const GeometryConstruction>&);
template<ArithmeticMode Mode>
std::size_t geometry_allocation_footprint(const std::shared_ptr<const DynamicConvexGeometry::Impl>&);
template<ArithmeticMode Mode> bool construction_edge_incident(const StoredVertex<Mode>&, const StoredPlanePtr<Mode>&);
template<ArithmeticMode Mode>
void preserve_crossing_incidence(
    StoredVertex<Mode>&, const StoredVertex<Mode>&, const StoredVertex<Mode>&, const Frame&, bool numerical_only);
} // namespace details

} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

std::vector<PlaneAuthority> dynamic_common_supports(std::span<const DynamicPoint> points);
bool certification_environment();
GeometryStatistics geometry_statistics();
void reset_geometry_statistics();
void record_ledger_operations(std::uint64_t count);
std::size_t dynamic_point_storage_bytes(const DynamicPoint& p);
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

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

}
