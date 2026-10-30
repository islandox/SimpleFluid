/** @file ConvexGeometry.hh
 * @brief Mode-typed finite convex geometry and explicit runtime adapters.
 */
#pragma once
#include "geometry/PLIC/ConvexCellGeometry.hh"
#include "geometry/PLIC/GeometryBackendTypes.hh"
#include "geometry/PLIC/GeometryPacketCodec.hh"
#include "utils/ArithmeticMode.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace SimpleFluid::PLIC
{
template<ArithmeticMode> class ConstructionRegistry;
namespace details
{
class NativeSupportAccess;
class OriginalSupportAuthorityAccess;
namespace ordinary_kernel
{
template<ArithmeticMode> class GeometryKernel;
template<ArithmeticMode> class PartitionKernel;
} // namespace ordinary_kernel
namespace certification_kernel
{
template<ArithmeticMode> class GeometryKernel;
template<ArithmeticMode> class PartitionKernel;
} // namespace certification_kernel
} // namespace details
/** Explicitly selected finite arithmetic value. In BoundedCertification the
 * fields enclose the original construction in physical units; zero-containing
 * intervals establish neither zero nor positivity. In Ordinary only value is
 * meaningful: absolute_error is untracked, and certificate queries reject.
 */
struct DynamicGeometryValue
{
    long double value = 0, absolute_error = 0;
    /** Ordinary values have no tracked error bound; the error field is unused. */
    ArithmeticMode arithmetic_mode = ArithmeticMode::BoundedCertification;
    [[nodiscard]] long double lower() const;
    [[nodiscard]] long double upper() const;
    [[nodiscard]] double as_double() const;
    /** Compiled fixed-mode query/publication helpers. Tags must match Mode;
     * certificate queries retain their Ordinary-mode rejection contract. */
    template<ArithmeticMode Mode> [[nodiscard]] long double lower() const;
    template<ArithmeticMode Mode> [[nodiscard]] long double upper() const;
    template<ArithmeticMode Mode> [[nodiscard]] double as_double() const;
    [[nodiscard]] bool relative_error_at_most(long double target, double tolerance) const;
};
/** Collision-free opaque original geometric support identity. Only immutable
 * source authority export issues these keys; they are never geometric numbers
 * or ownership/revision credentials. The caller authenticates those separately. */
/** Opaque source-support sidecar. Public authority words remain the identity;
 * only immutable source issuance/authenticated imports may seal this value. */
class OriginalSupportAttachment
{
public:
    OriginalSupportAttachment() = default;

private:
    friend class details::OriginalSupportAuthorityAccess;
    std::optional<PlaneSupportIdentity<ArithmeticMode::Ordinary>> support_;
};
struct PlaneAuthority
{
    std::vector<std::uint64_t> key;
    OriginalSupportAttachment original_support;
    bool operator==(const PlaneAuthority& other) const { return key == other.key; }
    auto operator<=>(const PlaneAuthority& other) const { return key <=> other.key; }
};
struct DynamicPointProvenance;
class DynamicGeometryAccess;
struct DynamicPoint
{
    std::array<long double, 3> value{}, absolute_error{};
    ArithmeticMode arithmetic_mode = ArithmeticMode::BoundedCertification;
    DynamicPoint() = default;
    DynamicPoint(std::array<long double, 3> coordinates, std::array<long double, 3> error = {},
        ArithmeticMode mode = ArithmeticMode::BoundedCertification)
        : value(coordinates), absolute_error(error), arithmetic_mode(mode)
    {
    }
    /** Changing coordinates/error invalidates the sealed source provenance. */
    [[nodiscard]] const std::vector<PlaneAuthority>& supports() const;

private:
    friend class DynamicGeometryAccess;
    template<ArithmeticMode> friend class Point;
    std::shared_ptr<const DynamicPointProvenance> provenance_;
};
[[nodiscard]] std::vector<PlaneAuthority> dynamic_common_supports(std::span<const DynamicPoint>);
struct DynamicSupport
{
    DynamicGeometryValue lower, upper;
};
struct DynamicGeometryMoments
{
    DynamicGeometryValue volume;
    GeometryPoint origin{};
    std::array<DynamicGeometryValue, 3> first_moment;
};
struct DynamicGeometrySection
{
    std::vector<DynamicPoint> vertices;
    DynamicGeometryValue area;
    DynamicPoint centroid;
};
/** Exact transaction pins supplied by the authenticated finite transport owner.
 * The owner defines the four words (publication/policy, geometry, partition,
 * trial); all words are compared, never inferred from numerical coordinates. */
struct GeometryPacketContext
{
    std::array<std::uint64_t, 4> publication{};
    bool operator==(const GeometryPacketContext&) const = default;
};
struct GeometryLimits
{
    std::size_t maximum_vertices = 4096, maximum_faces = 4096;
    /** Zero selects the checked geometry guard derived from V/F caps. Explicit
     * trajectory limits count interval scalar operations (one add/multiply),
     * matching TrajectoryLimits arithmetic operations, not endpoint FLOPs. */
    std::size_t maximum_operations = 0, maximum_pair_tests = 262144;
    std::size_t byte_budget = 64 * 1024 * 1024;
    ArithmeticMode arithmetic_mode = ArithmeticMode::BoundedCertification;
};
/** Thread-local counters; source authority export is counted separately. */
struct GeometryStatistics
{
    std::uint64_t source_imports = 0, constructions = 0, intersections = 0, support_distances = 0;
    std::uint64_t volume_evaluations = 0, moment_evaluations = 0, partition_checks = 0;
    /** Endpoint primitive invocations, not a hardware FLOP count. */
    std::uint64_t structural_partition_reuses = 0, floating_operations = 0, interval_operations = 0;
    /** Value arithmetic/accumulation helper events, not hardware FLOPs. */
    std::uint64_t ledger_operations = 0;
    std::uint64_t wider_routes = 0, rejections = 0;
    /** Runtime backend selections; fixed-mode kernel operations leave this unchanged. */
    std::uint64_t runtime_dispatches = 0;
    std::uint64_t exact_predicates = 0, rational_constructions = 0, rational_measures = 0, rational_ledgers = 0;
};
[[nodiscard]] GeometryStatistics geometry_statistics();
/** Calling image/thread identity; use only for duplicate-domain suppression. */
[[nodiscard]] const void* geometry_statistics_domain();
void reset_geometry_statistics();
void record_ledger_operations(std::uint64_t count = 1);
[[nodiscard]] bool certification_environment();
/** Retained point allocation, including sealed provenance/support keys. */
[[nodiscard]] std::size_t dynamic_point_storage_bytes(const DynamicPoint&);

class DynamicConvexGeometry;
using DynamicConvexGeometryPtr = std::shared_ptr<const DynamicConvexGeometry>;
struct DynamicFaceConstraint
{
    DynamicConvexGeometryPtr geometry;
    std::size_t face = 0;
};
struct DynamicClipResult
{
    DynamicConvexGeometryPtr inside, outside;
    DynamicGeometrySection section;
};
struct DynamicPhaseMeasures
{
    DynamicGeometryValue bulk, gas, carrier;
};

/** Immutable finite storage carrier at the runtime state boundary. Numerical
 * operations are available only through ConvexCellGeometry<Mode>; this handle
 * never selects an arithmetic kernel. */
class DynamicConvexGeometry
{
public:
    struct Impl;
    [[nodiscard]] ArithmeticMode arithmetic_mode() const noexcept;
    [[nodiscard]] GeometryPoint origin() const;
    [[nodiscard]] const std::vector<DynamicPoint>& points() const;
    [[nodiscard]] const std::vector<std::vector<std::size_t>>& faces() const;
    [[nodiscard]] const void* storage_identity() const noexcept { return impl_.get(); }
    template<ArithmeticMode Mode> [[nodiscard]] std::size_t storage_bytes() const;

private:
    template<ArithmeticMode> friend class details::ordinary_kernel::GeometryKernel;
    template<ArithmeticMode> friend class details::certification_kernel::GeometryKernel;
    template<ArithmeticMode> friend class details::ordinary_kernel::PartitionKernel;
    template<ArithmeticMode> friend class details::certification_kernel::PartitionKernel;
    template<ArithmeticMode> friend class ConvexCellGeometry;
    friend class DynamicGeometryAccess;
    explicit DynamicConvexGeometry(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
};

struct DynamicPartitionPiece
{
    std::size_t donor = 0;
    DynamicConvexGeometryPtr geometry;
};
/** A finite point's numerical fields follow its arithmetic policy. Original
 * source support evidence is retained separately and invalidated by mutation. */
template<ArithmeticMode Mode> struct PointCoordinates;
template<> struct PointCoordinates<ArithmeticMode::Ordinary>
{
    std::array<long double, 3> value{};
};
template<> struct PointCoordinates<ArithmeticMode::BoundedCertification>
{
    std::array<long double, 3> value{}, absolute_error{};
};
template<ArithmeticMode Mode> class Point : public PointCoordinates<Mode>
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);

public:
    static constexpr ArithmeticMode arithmetic_mode = Mode;
    Point() = default;
    explicit Point(std::array<long double, 3> coordinates) { this->value = coordinates; }
    Point(std::array<long double, 3> coordinates, std::array<long double, 3> error)
        requires(Mode == ArithmeticMode::BoundedCertification)
    {
        this->value = coordinates;
        this->absolute_error = error;
    }
    /** Checked conversion at a runtime policy boundary, preserving source identity. */
    explicit Point(const DynamicPoint&);
    [[nodiscard]] DynamicPoint dynamic() const;
    [[nodiscard]] const std::vector<PlaneAuthority>& supports() const { return dynamic().supports(); }

private:
    std::shared_ptr<const DynamicPointProvenance> provenance_;
};

template<ArithmeticMode Mode> struct Section
{
    std::vector<Point<Mode>> vertices;
    GeometryValue<Mode> area;
    Point<Mode> centroid;
};
template<ArithmeticMode Mode> struct Support
{
    GeometryValue<Mode> lower, upper;
};
template<ArithmeticMode Mode> struct ClipResult
{
    ConvexGeometryPtr<Mode> inside, outside;
    Section<Mode> section;
};
template<ArithmeticMode Mode> struct PhaseMeasures
{
    GeometryValue<Mode> bulk, gas, carrier;
};
template<ArithmeticMode Mode> struct FaceConstraint
{
    ConvexGeometryPtr<Mode> geometry;
    std::size_t face = 0;
};
template<ArithmeticMode Mode> struct PartitionPiece
{
    std::size_t donor = 0;
    ConvexGeometryPtr<Mode> geometry;
};

/** Statically selected finite geometry. Each object retains the matching
 * scalar/interval storage kernel. Runtime adapters are explicit; cross-mode
 * geometry operations are not part of this interface. Exact has a separate
 * specialization declared in ConvexCellGeometry.hh. */
template<ArithmeticMode Mode> class ConvexCellGeometry
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);

public:
    [[nodiscard]] static constexpr ArithmeticMode arithmetic_mode() noexcept { return Mode; }
    [[nodiscard]] static ConvexGeometryPtr<Mode> make(
        std::span<const GeometryPoint>, const std::vector<std::vector<std::size_t>>&, GeometryLimits = {});
    [[nodiscard]] static ConvexGeometryPtr<Mode> make_points(
        std::span<const Point<Mode>>, const std::vector<std::vector<std::size_t>>&, GeometryLimits = {});
    [[nodiscard]] static ConvexGeometryPtr<Mode> make_local(GeometryPoint origin, std::span<const Point<Mode>>,
        const std::vector<std::vector<std::size_t>>&, GeometryLimits = {});
    [[nodiscard]] static ConvexGeometryPtr<Mode> from_geometry(
        const ConvexCellGeometry<ArithmeticMode::Ordinary>&, GeometryLimits = {});
    [[nodiscard]] static ConvexGeometryPtr<Mode> box(GeometryPoint, GeometryPoint, GeometryLimits = {});
    [[nodiscard]] static ConvexGeometryPtr<Mode> tetrahedron(const std::array<Point<Mode>, 4>&, GeometryLimits = {});
    [[nodiscard]] static NativeConvexShell native_cell_shell(const MultiRegionMesh&, std::uint64_t)
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] static ConvexGeometryPtr<Mode> from_native_cell(const MultiRegionMesh&, std::uint64_t)
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] static ConvexGeometryPtr<Mode> from_native_cell_atomic(const MultiRegionMesh&, std::uint64_t)
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] static ConvexGeometryPtr<Mode> from_native_shell(
        std::span<const GeometryPoint>, const std::vector<std::vector<std::size_t>>&)
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] static ConvexGeometryPtr<Mode> extruded_polygon(
        std::span<const std::array<double, 2>>, double lower_z, double upper_z)
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] NumericalConvexShell<Mode> numerical_shell() const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] std::size_t num_faces() const noexcept { return geometry_->faces().size(); }
    [[nodiscard]] PlanarFacePtr<Mode> face_geometry(std::size_t) const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] std::vector<ConvexOctantPiece<Mode>> split_aabb_midpoints(double minimum_scaled_volume = 0) const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] bool quality_above(double minimum_scaled_volume) const;
    [[nodiscard]] bool axis_span_exceeds(const ConvexCellGeometry&, std::size_t axis, double ratio) const;
    [[nodiscard]] bool equivalent(const ConvexCellGeometry&) const;
    [[nodiscard]] std::vector<char> encode_certificate(std::size_t byte_budget = geometry_packet_default_budget) const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] static ConvexGeometryPtr<Mode> decode_certificate(
        std::span<const char>, std::size_t byte_budget = geometry_packet_default_budget)
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] GeometryValue<Mode> volume() const;
    [[nodiscard]] GeometryMoments<Mode> moments() const { return moments(origin()); }
    [[nodiscard]] GeometryMoments<Mode> moments(GeometryPoint origin) const;
    [[nodiscard]] GeometryPoint centroid() const;
    [[nodiscard]] GeometryPoint origin() const { return geometry_->origin(); }
    [[nodiscard]] std::vector<Point<Mode>> points() const;
    [[nodiscard]] std::vector<Point<Mode>> local_points() const;
    [[nodiscard]] const std::vector<std::vector<std::size_t>>& faces() const { return geometry_->faces(); }
    [[nodiscard]] std::array<GeometryValue<Mode>, 3> axis_spans() const;
    [[nodiscard]] Support<Mode> support(GeometryPoint normal, GeometryPoint anchor) const;
    [[nodiscard]] ClipResult<Mode> clip(const GeometryHalfspace&, bool measure_section = true) const;
    [[nodiscard]] ClipResult<Mode> clip_by_face(
        const ConvexCellGeometry&, std::size_t face, bool measure_section = false) const;
    [[nodiscard]] PhaseMeasures<Mode> phases(const GeometryHalfspace&) const;
    [[nodiscard]] bool bounds_prove_disjoint_interiors(const ConvexCellGeometry& other) const;
    [[nodiscard]] ConvexGeometryPtr<Mode> intersect(const ConvexCellGeometry&) const;
    [[nodiscard]] ConvexGeometryPtr<Mode> translated(GeometryPoint displacement) const;
    void certify_partition(std::span<const ConvexGeometryPtr<Mode>>, double relative_budget = 8e-15) const;
    /** Prove a finite transport fragment cover using retained complementary splits,
     * donor-batch certificates and the selected kernel's physical closure checks.
     * Native AMR shell partitions retain the separate certify_partition contract. */
    void certify_fragment_partition(std::span<const ConvexGeometryPtr<Mode>>, double relative_budget = 8e-15) const;
    [[nodiscard]] std::vector<char> encode(
        const GeometryPacketContext& context, std::size_t byte_budget = 64 * 1024 * 1024) const;
    [[nodiscard]] static ConvexGeometryPtr<Mode> decode(std::span<const char>, const GeometryPacketContext&,
        std::span<const ConvexGeometryPtr<Mode>> authenticated_sources, GeometryLimits = {});
    /** Version 4 preserves authenticated construction incidence by replay. */
    [[nodiscard]] std::vector<char> encode(
        const ConstructionRegistry<Mode>&, std::size_t byte_budget = 64 * 1024 * 1024) const;
    [[nodiscard]] static ConvexGeometryPtr<Mode> decode(
        std::span<const char>, const ConstructionRegistry<Mode>&, GeometryLimits = {});
    [[nodiscard]] const void* storage_identity() const noexcept { return geometry_->storage_identity(); }
    /** Retained payload plus both typed and runtime-adapter wrappers; shared
     * control blocks and allocator overhead are excluded. */
    [[nodiscard]] std::size_t storage_bytes() const
    {
        return sizeof(*this) + sizeof(DynamicConvexGeometry) + geometry_->template storage_bytes<Mode>();
    }
    /** The only runtime policy crossing. A foreign-mode object rejects. */
    [[nodiscard]] static ConvexGeometryPtr<Mode> from_dynamic(DynamicConvexGeometryPtr);
    [[nodiscard]] const DynamicConvexGeometryPtr& dynamic_geometry() const noexcept { return geometry_; }

private:
    template<ArithmeticMode> friend class PlanarFace;
    friend class details::NativeSupportAccess;
    friend std::vector<char> canonical_cell_key(const ConvexCellGeometry<ArithmeticMode::Ordinary>&);
    [[nodiscard]] ConvexGeometryPtr<Mode> with_native_supports(
        std::span<const PlaneSupportIdentity<ArithmeticMode::Ordinary>>) const
        requires(Mode == ArithmeticMode::Ordinary);
    explicit ConvexCellGeometry(DynamicConvexGeometryPtr);
    DynamicConvexGeometryPtr geometry_;
};

template<ArithmeticMode Mode> class ConvexPartition
{
public:
    explicit ConvexPartition(std::span<const ConvexGeometryPtr<Mode>>, GeometryLimits = {});
    [[nodiscard]] std::vector<PartitionPiece<Mode>> partition(
        const ConvexCellGeometry<Mode>&, std::size_t maximum_pieces = 65536) const;
    [[nodiscard]] std::size_t storage_bytes() const noexcept
    {
        return sizeof(*this) + donors_.capacity() * sizeof(DynamicConvexGeometryPtr);
    }

private:
    std::vector<DynamicConvexGeometryPtr> donors_;
    GeometryLimits limits_;
};

template<ArithmeticMode Mode> [[nodiscard]] std::vector<PlaneAuthority> common_supports(std::span<const Point<Mode>>);
template<ArithmeticMode Mode> [[nodiscard]] std::size_t point_storage_bytes(const Point<Mode>& point)
{
    return dynamic_point_storage_bytes(point.dynamic());
}
template<ArithmeticMode Mode> [[nodiscard]] Point<Mode> point_average(std::span<const Point<Mode>>);
template<ArithmeticMode Mode>
[[nodiscard]] Point<Mode> point_interpolate(const Point<Mode>&, const Point<Mode>&, long double fraction);
template<ArithmeticMode Mode> [[nodiscard]] GeometryValue<Mode> orientation(const std::array<Point<Mode>, 4>&);
template<ArithmeticMode Mode> [[nodiscard]] bool points_identical(const Point<Mode>&, const Point<Mode>&);
template<ArithmeticMode Mode> [[nodiscard]] Point<Mode> point_displaced(const Point<Mode>&, const Point<Mode>&);
template<ArithmeticMode Mode>
[[nodiscard]] Point<Mode> source_point(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>&, const PointIdentity<ArithmeticMode::Ordinary>& point_identity);
template<ArithmeticMode Mode>
[[nodiscard]] Point<Mode> source_point(std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>>,
    const PointIdentity<ArithmeticMode::Ordinary>& point_identity);
template<ArithmeticMode Mode>
[[nodiscard]] Point<Mode> backtrace_constrained(const Point<Mode>&, GeometryPoint velocity, double timestep,
    std::span<const FaceConstraint<Mode>>, GeometryLimits = {});

extern template class Point<ArithmeticMode::Ordinary>;
extern template class Point<ArithmeticMode::BoundedCertification>;
extern template class ConvexCellGeometry<ArithmeticMode::Ordinary>;
extern template class ConvexCellGeometry<ArithmeticMode::BoundedCertification>;
extern template class ConvexPartition<ArithmeticMode::Ordinary>;
extern template class ConvexPartition<ArithmeticMode::BoundedCertification>;

} // namespace SimpleFluid::PLIC
