#pragma once
#include "utils/export_helper.hh"
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
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/PlanarFace.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/OriginalSupportAuthority.hh"
#include "geometry/PLIC/details/PointSnapshot.hh"
#include <algorithm>
#include <atomic>
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
#include <variant>
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace SimpleFluid::PLIC
{
using namespace details;
namespace details
{
struct NativeGeometryBody;
template<class Visitor> void visit_provenance_children(PointProvenanceView point, Visitor&& visitor)
{
    if (const auto origin = point.source_origin())
        visitor(origin);
    for (const auto edge : point.source_edges())
    {
        visitor(edge[0]);
        visitor(edge[1]);
    }
}
} // namespace details

class SIMPLEFLUID_LOCAL DynamicGeometryAccess
{
public:
    static PointProvenanceRef identity(const DynamicPoint&);
    static std::size_t identity_bytes(const PointProvenanceRef&, bool include_support_payloads = true);
    static std::size_t identity_bytes(PointProvenanceView, bool include_support_payloads = true);
    static void retain_supports(PointProvenanceView, GeometryAllocationLedger&, std::size_t&, std::size_t);
    static void retain_supports(
        const PointProvenanceRef& point, GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
    {
        retain_supports(point.view(), ledger, used, limit);
    }
    static std::size_t storage_bytes(const DynamicPoint&, bool include_support_payloads = true);
    static bool same_source(const DynamicPoint&, const DynamicPoint&);
    static void seal(DynamicPoint&, std::vector<PlaneAuthority>);
    static void seal_represented_source(DynamicPoint&);
    static void seal_represented_source(DynamicPoint&, std::vector<PlaneAuthority>);
    static bool same_represented_source(PointProvenanceView, PointProvenanceView);
    static bool same_represented_source(const PointProvenanceRef& a, const PointProvenanceRef& b)
    {
        return same_represented_source(a.view(), b.view());
    }
    static PointProvenanceView borrowed_original_source(PointProvenanceView);
    static PointProvenanceView borrowed_original_source(const PointProvenanceRef& point)
    {
        return borrowed_original_source(point.view());
    }
    static PointProvenanceView borrowed_original_source(PointProvenanceRef&&) = delete;
    static PointProvenanceView borrowed_original_source(const PointProvenanceRef&&) = delete;
    static PointProvenanceRef original_source(const PointProvenanceRef& point)
    {
        return borrowed_original_source(point).owning();
    }
    static PointProvenanceRef original_source(PointProvenanceView point)
    {
        return borrowed_original_source(point).owning();
    }
    static void seal_export(DynamicPoint&, PointProvenanceView, std::vector<PlaneAuthority>, bool numerical = false);
    static void seal_export(DynamicPoint& point, const PointProvenanceRef& source, std::vector<PlaneAuthority> supports,
        bool numerical = false)
    {
        seal_export(point, source.view(), std::move(supports), numerical);
    }
    static void seal_generated(DynamicPoint&, std::vector<std::array<PointProvenanceView, 2>>,
        std::vector<PlaneAuthority> = {}, bool numerical = false);
    static void seal_numerical(DynamicPoint&);
    static void attach_identity(DynamicPoint&, PointProvenanceRef);
    template<ArithmeticMode Mode>
    static PointSnapshot<Mode> capture_snapshot(
        std::span<const Point<Mode>>, GeometryAllocationLedger&, std::size_t&, std::size_t);
    template<ArithmeticMode Mode>
    static const std::shared_ptr<const PointSnapshotPayload<Mode>>& snapshot_payload(const PointSnapshot<Mode>& points)
    {
        return points.payload_;
    }
    static PointAuthorityRange boundary_keys(std::span<const PlaneAuthority> keys)
    {
        return PointAuthorityRange(keys.data(), 0, keys.size(), nullptr);
    }
    template<ArithmeticMode Mode>
    static AdmittedPointIncidence<Mode> admit_incidence(std::span<const PlaneAuthority> keys)
    {
        return AdmittedPointIncidence<Mode>(PointAuthorityRange(keys.data(), 0, keys.size(), nullptr));
    }
    template<ArithmeticMode Mode> static AdmittedPointIncidence<Mode> admit_incidence(PointAuthorityRange keys)
    {
        return AdmittedPointIncidence<Mode>(keys);
    }
    template<ArithmeticMode Mode>
    static PointExport<Mode> export_typed(const std::shared_ptr<const DynamicConvexGeometry::Impl>&,
        GeometryAllocationLedger&, std::size_t&, std::size_t);
    template<ArithmeticMode Mode>
    static DynamicPointExport export_dynamic(const std::shared_ptr<const DynamicConvexGeometry::Impl>&,
        GeometryAllocationLedger&, std::size_t&, std::size_t);
    template<ArithmeticMode Mode, bool Typed>
    static auto export_impl(const std::shared_ptr<const DynamicConvexGeometry::Impl>&, GeometryAllocationLedger&,
        std::size_t&, std::size_t);

    /** Private construction handoff: every new binding owner is already charged before allocation. */
    template<ArithmeticMode Mode>
    static void retain_prepaid_native_binding(
        const ConvexGeometryPtr<Mode>&, GeometryAllocationLedger&, std::size_t&, std::size_t);
    static const std::shared_ptr<const DynamicConvexGeometry::Impl>& implementation(const DynamicConvexGeometry&);
    static DynamicConvexGeometryPtr geometry(std::shared_ptr<const DynamicConvexGeometry::Impl>);
    template<ArithmeticMode Mode>
    static void retain_geometry_storage(const std::shared_ptr<const DynamicConvexGeometry::Impl>&,
        GeometryAllocationLedger&, std::size_t&, std::size_t);
    static void retain_native_body(const std::shared_ptr<const details::NativeGeometryBody>&,
        GeometryAllocationLedger&, std::size_t&, std::size_t);
};

namespace SIMPLEFLUID_LOCAL details
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
    std::shared_ptr<const PointSnapshotContext<ArithmeticMode::Ordinary>> point_context;
    std::vector<StoredPointRef<ArithmeticMode::Ordinary>> original_points;
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
template<ArithmeticMode Mode> struct StoredMomentCache;
struct NativeGeometryConstruction
{
    std::shared_ptr<const CartesianCellSource> source;
    std::size_t cell_slot = 0;
    bool primed = false;
    GeometryLimits limits;
    std::shared_ptr<StoredMomentCache<ArithmeticMode::Ordinary>> moments;
    std::shared_ptr<const NativeGeometryBody> body;
};
struct GeometryClipConstruction
{
    std::shared_ptr<const GeometryConstruction> parent;
    /** Retention/proof input only; it is not a geometric parent or wire DAG node. */
    std::shared_ptr<const GeometryConstruction> native_plane_source;
    std::shared_ptr<const void> plane;
    std::shared_ptr<const PointSnapshotContextBase> point_context;
    std::vector<DynamicStoredPointRef> retained_points;
    ArithmeticMode mode = ArithmeticMode::Ordinary;
    int orientation = 1;
    bool measure_section = false, construct_outside = true, partition_proof = false;
};
struct GeometryConstruction
{
    enum class Kind
    {
        Source,
        Clip,
        Rebase
    };
    struct Source
    {
        using Payload = std::variant<std::shared_ptr<const DynamicConvexGeometry::Impl>,
            std::unique_ptr<const NativeGeometryConstruction>>;
        Payload payload;
        explicit Source(std::shared_ptr<const DynamicConvexGeometry::Impl> value) : payload(std::move(value)) {}
        explicit Source(std::unique_ptr<const NativeGeometryConstruction> value) : payload(std::move(value)) {}
        [[nodiscard]] const NativeGeometryConstruction* native() const
        {
            const auto* value = std::get_if<std::unique_ptr<const NativeGeometryConstruction>>(&payload);
            return value ? value->get() : nullptr;
        }
        [[nodiscard]] std::shared_ptr<const DynamicConvexGeometry::Impl> geometry() const
        {
            const auto* value = std::get_if<std::shared_ptr<const DynamicConvexGeometry::Impl>>(&payload);
            return value ? *value : nullptr;
        }
    };
    struct Clip
    {
        std::shared_ptr<const GeometryClipConstruction> clip;
        bool outside = false;
    };
    struct Rebase
    {
        std::shared_ptr<const GeometryConstruction> parent;
        GeometryHalfspace input;
    };
    using Payload = std::variant<Source, Clip, std::unique_ptr<const Rebase>>;
    explicit GeometryConstruction(Source value) : payload(std::move(value)) {}
    explicit GeometryConstruction(Clip value) : payload(std::move(value)) {}
    explicit GeometryConstruction(std::unique_ptr<const Rebase> value) : payload(std::move(value)) {}
    [[nodiscard]] Kind kind() const { return static_cast<Kind>(payload.index()); }
    [[nodiscard]] const Source& source() const { return std::get<Source>(payload); }
    [[nodiscard]] const Clip& clip() const { return std::get<Clip>(payload); }
    [[nodiscard]] const Rebase& rebase() const { return *std::get<std::unique_ptr<const Rebase>>(payload); }
    const Payload payload;
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
    LocalAuthorityRange<Mode> authorities;
    const PointSnapshotContext<Mode>* point_context = nullptr;
    std::vector<StoredPointRef<Mode>> source_vertices;
};

template<ArithmeticMode Mode> using StoredPlanePtr = std::shared_ptr<const StoredPlane<Mode>>;

/** Mutable draft support storage, frozen only against an exact source sequence.
 * Reads never detach. Shared storage is immutable; append stages a deep copy. */
template<ArithmeticMode Mode> struct StoredVertex
{
    StoredPoint<Mode> coordinates;
    std::vector<StoredPlanePtr<Mode>> incident;
    LocalAuthorityRange<Mode> authorities;
    StoredPointRef<Mode> source_identity;
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

template<ArithmeticMode Mode> struct FrozenTopology;
template<ArithmeticMode Mode> class IncidentPlaneRange
{
    // A transient draft view borrows its vector object. Published views borrow
    // topology + row number; neither representation owns incidence storage.
    const void* source_ = nullptr;
    std::size_t row_ = std::numeric_limits<std::size_t>::max();

public:
    using value_type = StoredPlanePtr<Mode>;
    using iterator = ValueRangeIterator<IncidentPlaneRange>;
    IncidentPlaneRange() = default;
    explicit IncidentPlaneRange(const std::vector<StoredPlanePtr<Mode>>& planes) : source_(&planes) {}
    IncidentPlaneRange(const FrozenTopology<Mode>* topology, std::size_t row) : source_(topology), row_(row) {}
    std::size_t size() const noexcept;
    bool empty() const noexcept { return size() == 0; }
    const StoredPlanePtr<Mode>& operator[](std::size_t i) const;
    const StoredPlanePtr<Mode>& front() const { return (*this)[0]; }
    iterator begin() const { return {*this, 0}; }
    iterator end() const { return {*this, static_cast<std::ptrdiff_t>(size())}; }
};
/** Immutable coordinate/provenance payload; incidence only borrows the shared topology. */
template<ArithmeticMode Mode> struct FrozenVertex
{
    StoredPoint<Mode> coordinates;
    IncidentPlaneRange<Mode> incident;
    LocalAuthorityRange<Mode> authorities;
    StoredPointRef<Mode> source_identity;
    bool numerical_only = false;
    FrozenVertex() = default;
    FrozenVertex(const StoredVertex<Mode>& vertex)
        : coordinates(vertex.coordinates), incident(vertex.incident), authorities(vertex.authorities),
          source_identity(vertex.source_identity), numerical_only(vertex.numerical_only)
    {
    }
};

struct TopologyCounts
{
    std::size_t vertices = 0, faces = 0, planes = 0, face_slots = 0, incident_slots = 0;
};
inline std::size_t topology_sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Frozen topology offset overflows.");
    return a + b;
}
inline std::size_t topology_product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Frozen topology allocation size overflows.");
    return a * b;
}
inline bool compact_topology(const TopologyCounts& counts, bool force_wide = false)
{
    (void) topology_sum(counts.vertices, 1);
    (void) topology_sum(counts.faces, 1);
    const auto max = std::numeric_limits<std::uint32_t>::max();
    return !force_wide && counts.vertices <= max && counts.faces <= max && counts.planes <= max &&
           counts.face_slots <= max && counts.incident_slots <= max;
}
template<class Index> struct TopologyArrays
{
    std::vector<Index> face_offsets, face_vertices, incident_offsets, incident_planes, surface_planes;
    std::vector<int> surface_orientations;
    std::size_t bytes() const
    {
        auto count = topology_sum(face_offsets.capacity(), face_vertices.capacity());
        count = topology_sum(count, incident_offsets.capacity());
        count = topology_sum(count, incident_planes.capacity());
        count = topology_sum(count, surface_planes.capacity());
        return topology_sum(
            topology_product(count, sizeof(Index)), topology_product(surface_orientations.capacity(), sizeof(int)));
    }
};
template<ArithmeticMode Mode> struct FrozenTopology
{
    std::variant<TopologyArrays<std::uint32_t>, TopologyArrays<std::uint64_t>> arrays;
    std::vector<StoredPlanePtr<Mode>> planes;
    FaceRange faces() const
    {
        return std::visit(
            [](const auto& a)
            {
                return FaceRange(a.face_offsets.data(), a.face_vertices.data(), a.face_offsets.size() - 1,
                    sizeof(typename std::remove_cvref_t<decltype(a.face_offsets)>::value_type));
            },
            arrays);
    }
    IncidentPlaneRange<Mode> incidence(std::size_t vertex) const { return {this, vertex}; }
    std::size_t incident_count(std::size_t vertex) const
    {
        return std::visit([&](const auto& a)
            { return static_cast<std::size_t>(a.incident_offsets[vertex + 1] - a.incident_offsets[vertex]); }, arrays);
    }
    const StoredPlanePtr<Mode>& incident_plane(std::size_t vertex, std::size_t index) const
    {
        return std::visit([&](const auto& a) -> const StoredPlanePtr<Mode>&
            { return planes[a.incident_planes[a.incident_offsets[vertex] + index]]; }, arrays);
    }
    StoredSurface<Mode> surface(std::size_t face) const
    {
        return std::visit([&](const auto& a)
            { return StoredSurface<Mode>{planes[a.surface_planes[face]], a.surface_orientations[face]}; }, arrays);
    }
    std::size_t bytes() const
    {
        return topology_sum(
            sizeof(FrozenTopology), topology_sum(topology_product(planes.capacity(), sizeof(StoredPlanePtr<Mode>)),
                                        std::visit([](const auto& a) { return a.bytes(); }, arrays)));
    }
};
template<ArithmeticMode Mode> std::size_t IncidentPlaneRange<Mode>::size() const noexcept
{
    if (!source_)
        return 0;
    return row_ == std::numeric_limits<std::size_t>::max()
               ? static_cast<const std::vector<StoredPlanePtr<Mode>>*>(source_)->size()
               : static_cast<const FrozenTopology<Mode>*>(source_)->incident_count(row_);
}
template<ArithmeticMode Mode> const StoredPlanePtr<Mode>& IncidentPlaneRange<Mode>::operator[](std::size_t i) const
{
    return row_ == std::numeric_limits<std::size_t>::max()
               ? (*static_cast<const std::vector<StoredPlanePtr<Mode>>*>(source_))[i]
               : static_cast<const FrozenTopology<Mode>*>(source_)->incident_plane(row_, i);
}
template<ArithmeticMode Mode> class SurfaceRange
{
    const FrozenTopology<Mode>* topology_ = nullptr;

public:
    using value_type = StoredSurface<Mode>;
    using iterator = ValueRangeIterator<SurfaceRange>;
    SurfaceRange() = default;
    explicit SurfaceRange(const FrozenTopology<Mode>* topology) : topology_(topology) {}
    std::size_t size() const { return topology_ ? topology_->faces().size() : 0; }
    bool empty() const { return size() == 0; }
    value_type operator[](std::size_t i) const { return topology_->surface(i); }
    value_type at(std::size_t i) const
    {
        if (i >= size())
            throw std::out_of_range("surface index");
        return (*this)[i];
    }
    iterator begin() const { return {*this, 0}; }
    iterator end() const { return {*this, static_cast<std::ptrdiff_t>(size())}; }
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
    std::atomic<bool> ready{false};
};
} // namespace SIMPLEFLUID_LOCAL details

struct SIMPLEFLUID_LOCAL DynamicPointExportPayload
{
    ArithmeticMode mode = ArithmeticMode::Ordinary;
    std::vector<DynamicPoint> points;
    std::shared_ptr<const PointSnapshotContextBase> point_context;
};
template<ArithmeticMode Mode> struct PointExportPayload
{
    std::vector<Point<Mode>> points;
    std::shared_ptr<const PointSnapshotContext<Mode>> point_context;
};
template<ArithmeticMode Mode> struct PointSnapshotPayload
{
    std::shared_ptr<const PointSnapshotContext<Mode>> context;
    std::vector<StoredPointRef<Mode>> refs;
};
struct SIMPLEFLUID_LOCAL PointExportCache
{
    std::mutex mutex;
    std::fenv_t environment{}; ///< Arithmetic environment of the validated global-coordinate publication.

    std::weak_ptr<const DynamicPointExportPayload> dynamic_root;
    std::weak_ptr<const PointExportPayload<ArithmeticMode::Ordinary>> ordinary_root;
    std::weak_ptr<const PointExportPayload<ArithmeticMode::BoundedCertification>> bounded_root;
    std::weak_ptr<const PointSnapshotContextBase> provenance_root;
    std::vector<DynamicStoredPointRef> vertex_provenance;
};

struct SIMPLEFLUID_LOCAL DynamicConvexGeometry::Impl
{
    Impl() = default;
    explicit Impl(std::shared_ptr<const Ancestry> source_ancestry) : ancestry(std::move(source_ancestry)) {}
    Frame frame;
    GeometryPoint reference_origin{};
    FaceRange faces;
    std::shared_ptr<PointExportCache> point_export_cache;
    std::shared_ptr<const Impl> export_source;
    std::shared_ptr<const GeometryConstruction> native_export_source;
    bool export_seal = false;
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
template<ArithmeticMode Mode> struct GeometryMetadata : DynamicConvexGeometry::Impl
{
    GeometryMetadata() { limits.arithmetic_mode = Mode; }
    explicit GeometryMetadata(std::shared_ptr<const Ancestry> source_ancestry)
        : DynamicConvexGeometry::Impl(std::move(source_ancestry))
    {
        limits.arithmetic_mode = Mode;
    }
    std::shared_ptr<const PointSnapshotContext<Mode>> point_context;
    StoredValue<Mode> volume;
    std::shared_ptr<StoredMomentCache<Mode>> moment_cache;
};
/** Normal construction owns its array. Evaluation contexts may borrow an
 * immutable original array; copying a context retains the established deep
 * copy contract used by rebase and other mutation paths. */
template<ArithmeticMode Mode> class FrozenVertexBuffer
{
public:
    using value_type = FrozenVertex<Mode>;
    FrozenVertexBuffer() = default;
    explicit FrozenVertexBuffer(std::vector<value_type>&& values) : owned_(std::move(values)) {}
    FrozenVertexBuffer(const FrozenVertexBuffer& other)
    {
        if (!other.empty())
            owned_.assign(other.begin(), other.end());
    }
    FrozenVertexBuffer(FrozenVertexBuffer&&) noexcept = default;
    FrozenVertexBuffer& operator=(FrozenVertexBuffer&&) noexcept = default;
    FrozenVertexBuffer& operator=(const FrozenVertexBuffer& other)
    {
        if (this != &other)
        {
            require_owned();
            owned_.assign(other.begin(), other.end());
        }
        return *this;
    }
    void borrow(std::span<const value_type> values, std::size_t capacity)
    {
        if (is_borrowed_ || !owned_.empty() || owned_.capacity() || capacity < values.size())
            throw GeometryError(GeometryErrorCode::InvalidTopology, "Borrowed vertices require an empty binding.");
        borrowed_ = values;
        borrowed_capacity_ = capacity;
        is_borrowed_ = true;
    }
    [[nodiscard]] bool borrowed() const noexcept { return is_borrowed_; }
    [[nodiscard]] std::size_t size() const noexcept { return is_borrowed_ ? borrowed_.size() : owned_.size(); }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] std::size_t capacity() const noexcept
    { return is_borrowed_ ? borrowed_capacity_ : owned_.capacity(); }
    [[nodiscard]] std::size_t owned_capacity() const noexcept { return owned_.capacity(); }
    [[nodiscard]] const value_type* data() const noexcept { return is_borrowed_ ? borrowed_.data() : owned_.data(); }
    [[nodiscard]] const value_type* begin() const noexcept { return data(); }
    [[nodiscard]] const value_type* end() const noexcept { return empty() ? data() : data() + size(); }
    [[nodiscard]] const value_type& operator[](std::size_t index) const noexcept { return data()[index]; }
    [[nodiscard]] const value_type& at(std::size_t index) const
    {
        if (index >= size())
            throw std::out_of_range("frozen vertex index");
        return data()[index];
    }
    [[nodiscard]] const value_type& front() const { return at(0); }
    [[nodiscard]] value_type* begin() { require_owned(); return owned_.data(); }
    [[nodiscard]] value_type* end()
    { require_owned(); return owned_.empty() ? owned_.data() : owned_.data() + owned_.size(); }
    [[nodiscard]] value_type& operator[](std::size_t index) { require_owned(); return owned_[index]; }
    [[nodiscard]] value_type& at(std::size_t index) { require_owned(); return owned_.at(index); }
    [[nodiscard]] value_type& front() { require_owned(); return owned_.front(); }
    void reserve(std::size_t count) { require_owned(); owned_.reserve(count); }
    void resize(std::size_t count) { require_owned(); owned_.resize(count); }
    template<class Iterator> void assign(Iterator first, Iterator last)
    { require_owned(); owned_.assign(first, last); }
private:
    void require_owned() const
    {
        if (is_borrowed_)
            throw GeometryError(GeometryErrorCode::InvalidTopology, "Evaluation vertices are immutable borrows.");
    }
    std::vector<value_type> owned_;
    std::span<const value_type> borrowed_;
    std::size_t borrowed_capacity_ = 0;
    bool is_borrowed_ = false;
};
template<ArithmeticMode Mode> struct GeometryStorage : GeometryMetadata<Mode>
{
    std::shared_ptr<const FrozenTopology<Mode>> topology;
    FrozenVertexBuffer<Mode> vertices;
    using VertexOwner = std::variant<std::monostate, std::shared_ptr<const DynamicConvexGeometry::Impl>,
        std::shared_ptr<const NativeGeometryBody>>;
    VertexOwner vertex_owner;
    SurfaceRange<Mode> surfaces;
    GeometryStorage() = default;
    GeometryStorage(const GeometryStorage& other)
        : GeometryMetadata<Mode>(other), topology(other.topology), vertices(other.vertices), surfaces(topology.get())
    {
    }
    explicit GeometryStorage(std::shared_ptr<const Ancestry> source_ancestry)
        : GeometryMetadata<Mode>(std::move(source_ancestry))
    {
    }
    GeometryStorage(GeometryMetadata<Mode>&& metadata, std::shared_ptr<const FrozenTopology<Mode>> owner,
        std::vector<FrozenVertex<Mode>>&& payload)
        : GeometryMetadata<Mode>(std::move(metadata)), topology(std::move(owner)), vertices(std::move(payload)),
          surfaces(topology.get())
    {
        this->faces = topology->faces();
    }
};
/** Validated original mesh-source representation. It owns original numerical
 * facts and certificates without retaining a published geometry Impl/wrapper. */
struct NativeGeometryBody
{
    Frame frame;
    GeometryPoint reference_origin{};
    CoordinateBox bounds;
    std::shared_ptr<const FrozenTopology<ArithmeticMode::Ordinary>> topology;
    std::vector<FrozenVertex<ArithmeticMode::Ordinary>> vertices;
    std::shared_ptr<const PointSnapshotContext<ArithmeticMode::Ordinary>> point_context;
    StoredValue<ArithmeticMode::Ordinary> volume;
    std::shared_ptr<PointExportCache> export_cache;
    std::shared_ptr<const Ancestry> ancestry;
    bool export_seal = false;
};
/** Mutable construction only. No draft instance is ever published. */
template<ArithmeticMode Mode> struct GeometryDraft : GeometryMetadata<Mode>
{
    std::vector<Face> faces;
    std::vector<StoredVertex<Mode>> vertices;
    std::vector<StoredSurface<Mode>> surfaces;
    // Set only by fully reserved draft_geometry(); its translation caller changes values, never capacities.
    bool buffers_prepaid = false;
    bool planes_prepaid = false;
    GeometryDraft() = default;
    explicit GeometryDraft(const GeometryMetadata<Mode>& metadata) : GeometryMetadata<Mode>(metadata) {}
};

#include "geometry/PLIC/details/FrozenTopology.ipp"

template<ArithmeticMode Mode>
std::shared_ptr<const GeometryStorage<Mode>> checked_geometry_storage(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>& source)
{
    require_mode(source->limits.arithmetic_mode, Mode);
    return std::static_pointer_cast<const GeometryStorage<Mode>>(source);
}
/** A cut pins one plane and its provenance without pretending to be a published shell. */
template<ArithmeticMode Mode> struct CutPlaneDescriptor
{
    std::shared_ptr<const GeometryStorage<Mode>> owner;
    StoredSurface<Mode> support;
    GeometryLimits limits;
    Frame frame;
    FaceRange faces;
    std::span<const FrozenVertex<Mode>> vertices;
    std::shared_ptr<const PointSnapshotContext<Mode>> point_context;
    bool numerical_only = false;
    CutPlaneDescriptor(StoredPlanePtr<Mode> plane, int orientation, GeometryLimits requested)
        : support{std::move(plane), orientation}, limits(requested), frame(support.plane->frame)
    {
        if (support.plane->point_context)
        {
            require_mode(support.plane->point_context->mode, Mode);
            point_context = std::static_pointer_cast<const PointSnapshotContext<Mode>>(
                support.plane->point_context->shared_from_this());
        }
    }
    CutPlaneDescriptor(std::shared_ptr<const GeometryStorage<Mode>> source, std::size_t face)
        : owner(std::move(source)), support(owner->surfaces.at(face)), limits(owner->limits), frame(owner->frame),
          faces(owner->faces), vertices(owner->vertices), point_context(owner->point_context),
          numerical_only(owner->numerical_only)
    {
    }
};
template<ArithmeticMode Mode>
std::shared_ptr<const GeometryConstruction> construction_source(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&);
std::shared_ptr<const GeometryConstruction> construction_node(GeometryConstruction::Source);
std::shared_ptr<const GeometryConstruction> construction_native(NativeGeometryConstruction);
std::shared_ptr<const GeometryConstruction> retained_native_plane_source(
    std::shared_ptr<const GeometryConstruction>, const void* plane);
bool same_native_construction(const GeometryConstruction&, const GeometryConstruction&);
ConvexGeometryPtr<ArithmeticMode::Ordinary> materialize_native_construction(
    const std::shared_ptr<const GeometryConstruction>&);
void retain_rebase_export(const std::shared_ptr<const DynamicConvexGeometry::Impl>&,
    DynamicConvexGeometry::Impl&);
std::shared_ptr<const GeometryConstruction> construction_node(GeometryConstruction::Clip);
std::shared_ptr<const GeometryConstruction> construction_rebase(
    std::shared_ptr<const GeometryConstruction>, const GeometryHalfspace&);
std::size_t construction_storage_bytes(const std::shared_ptr<const GeometryConstruction>&);
template<ArithmeticMode Mode>
std::size_t geometry_allocation_footprint(const std::shared_ptr<const DynamicConvexGeometry::Impl>&);
template<ArithmeticMode Mode>
bool construction_edge_incident(
    const FrozenVertex<Mode>&, const StoredPlanePtr<Mode>&, const PointSnapshotContext<Mode>*);
template<ArithmeticMode Mode>
void preserve_crossing_incidence(StoredVertex<Mode>&, const FrozenVertex<Mode>&, const FrozenVertex<Mode>&,
    const Frame&, bool numerical_only, PointSnapshotBuilder<Mode>&);
} // namespace details

namespace SIMPLEFLUID_LOCAL details
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
} // namespace SIMPLEFLUID_LOCAL details
} // namespace SimpleFluid::PLIC
