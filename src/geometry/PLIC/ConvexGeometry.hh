/** @file ConvexGeometry.hh
 * @brief Mode-typed finite convex geometry and explicit runtime adapters.
 */
#pragma once
#include "geometry/PLIC/ConvexCellGeometry.hh"
#include "geometry/PLIC/GeometryBackendTypes.hh"
#include "utils/ArithmeticMode.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::PLIC
{
namespace details
{
class NativeSupportAccess;
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
/** Collision-free opaque original support identity. Only immutable source
 * authority export can issue these keys; they are never geometric numbers. */
struct PlaneAuthority
{
    std::vector<std::uint64_t> key;
    bool operator==(const PlaneAuthority&) const = default;
    auto operator<=>(const PlaneAuthority&) const = default;
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
[[nodiscard]] DynamicPoint dynamic_point_average(std::span<const DynamicPoint>);
[[nodiscard]] DynamicPoint dynamic_point_interpolate(const DynamicPoint&, const DynamicPoint&, long double fraction);
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
     * matching ExactTrajectoryLimits arithmetic operations, not endpoint FLOPs. */
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
    std::uint64_t exact_predicates = 0, rational_constructions = 0, rational_measures = 0, rational_ledgers = 0;
};
[[nodiscard]] GeometryStatistics geometry_statistics();
/** Calling image/thread identity; use only for duplicate-domain suppression. */
[[nodiscard]] const void* geometry_statistics_domain();
void reset_geometry_statistics();
void record_ledger_operations(std::uint64_t count = 1);
[[nodiscard]] bool certification_environment();
/** Six times the oriented tetrahedron volume, retaining input uncertainty. */
[[nodiscard]] DynamicGeometryValue dynamic_orientation(const std::array<DynamicPoint, 4>&);
[[nodiscard]] bool dynamic_points_identical(const DynamicPoint&, const DynamicPoint&);
/** Retained point allocation, including sealed provenance/support keys. */
[[nodiscard]] std::size_t dynamic_point_storage_bytes(const DynamicPoint&);
/** Bootstrap-only source coordinate/support export, separate from step work. */
[[nodiscard]] DynamicPoint dynamic_source_point(const ConvexCellGeometry<ArithmeticMode::Ordinary>&,
    std::span<const char> point_identity, ArithmeticMode = ArithmeticMode::BoundedCertification);
/** One bootstrap coordinate export plus the union of true supports over the
 * complete authenticated incident source-cell fan (including hanging nodes). */
[[nodiscard]] DynamicPoint dynamic_source_point(
    std::span<const ConvexGeometryPtr<ArithmeticMode::Ordinary>> incident_sources, std::span<const char> point_identity,
    ArithmeticMode = ArithmeticMode::BoundedCertification);
[[nodiscard]] DynamicPoint dynamic_point_displaced(const DynamicPoint&, const DynamicPoint& displacement);

class DynamicConvexGeometry;
class DynamicConvexPartition;
using DynamicConvexGeometryPtr = std::shared_ptr<const DynamicConvexGeometry>;
struct DynamicFaceConstraint
{
    DynamicConvexGeometryPtr geometry;
    std::size_t face = 0;
};
/** Authenticate every wall incidence, then construct motion in the common
 * tangent space using certified cross/triple-product rank. Ambiguous rank
 * rejects; nearly parallel walls are never merged by an angular tolerance. */
[[nodiscard]] DynamicPoint dynamic_backtrace_constrained(const DynamicPoint&, GeometryPoint velocity, double timestep,
    std::span<const DynamicFaceConstraint>, GeometryLimits = {});
struct DynamicClipResult
{
    DynamicConvexGeometryPtr inside, outside;
    DynamicGeometrySection section;
};
struct DynamicPhaseMeasures
{
    DynamicGeometryValue bulk, gas, carrier;
};

/** Immutable local, power-of-two-scaled convex shell. Ordinary uses nominal FP
 * arithmetic. BoundedCertification uses directed enclosures and rejects
 * unresolved certification. Exact is rejected; there is no automatic fallback.
 * Caps apply to each operation; callers also charge aggregate trial work.
 */
class DynamicConvexGeometry
{
public:
    struct Impl;
    [[nodiscard]] ArithmeticMode arithmetic_mode() const noexcept;
    [[nodiscard]] static DynamicConvexGeometryPtr make(
        std::span<const GeometryPoint>, const std::vector<std::vector<std::size_t>>&, GeometryLimits = {});
    /** Global coordinates retain valid sealed input-point identities locally;
     * equal coordinate/error tuples alone never establish point identity. */
    [[nodiscard]] static DynamicConvexGeometryPtr make_bounded(
        std::span<const DynamicPoint>, const std::vector<std::vector<std::size_t>>&, GeometryLimits = {});
    [[nodiscard]] static DynamicConvexGeometryPtr make_bounded_local(GeometryPoint origin,
        std::span<const DynamicPoint> relative_vertices, const std::vector<std::vector<std::size_t>>&,
        GeometryLimits = {});
    /** Explicit source-authority export, once per immutable source: coordinate
     * enclosures only, never exact volumes or moments. */
    [[nodiscard]] static DynamicConvexGeometryPtr from_geometry(
        const ConvexCellGeometry<ArithmeticMode::Ordinary>&, GeometryLimits = {});
    [[nodiscard]] static DynamicConvexGeometryPtr box(GeometryPoint, GeometryPoint, GeometryLimits = {});
    /** Null only for proven zero determinant; an uncertain determinant rejects. */
    [[nodiscard]] static DynamicConvexGeometryPtr tetrahedron(const std::array<DynamicPoint, 4>&, GeometryLimits = {});
    /** A strict-sign clipping construction proves positive interior even when
     * its quantitative volume enclosure includes zero. Such uncertainty is
     * retained; division, centroid and packet export still require a positive
     * numerical lower bound. Independent imports retain strict admission. */
    [[nodiscard]] DynamicGeometryValue volume() const;
    [[nodiscard]] DynamicGeometryMoments moments() const;
    [[nodiscard]] DynamicGeometryMoments moments(GeometryPoint origin) const;
    [[nodiscard]] GeometryPoint centroid() const;
    /** Stable default moment reference. Numerical clipping frames may rebase
     * privately without moving the physical geometry or this reference. */
    [[nodiscard]] GeometryPoint origin() const;
    [[nodiscard]] const std::vector<DynamicPoint>& points() const;
    /** Coordinates relative to origin(); global support authority is omitted. */
    [[nodiscard]] std::vector<DynamicPoint> local_points() const;
    [[nodiscard]] const std::vector<std::vector<std::size_t>>& faces() const;
    [[nodiscard]] std::array<DynamicGeometryValue, 3> axis_spans() const;
    [[nodiscard]] DynamicSupport support(GeometryPoint normal, GeometryPoint anchor) const;
    [[nodiscard]] DynamicClipResult clip(const GeometryHalfspace&, bool measure_section = true) const;
    [[nodiscard]] DynamicClipResult clip_by_face(
        const DynamicConvexGeometry&, std::size_t face, bool measure_section = false) const;
    [[nodiscard]] DynamicPhaseMeasures phases(const GeometryHalfspace&) const;
    /** Test nominal spatial extents in Ordinary; in BoundedCertification true
     * certifies disjoint interiors with outward extents. False is inconclusive.
     * Face/edge/point contacts may still exist.
     * This certificate concerns volume/moments, not surface integrals. */
    [[nodiscard]] bool bounds_prove_disjoint_interiors(const DynamicConvexGeometry&) const;
    /** Construct and certify only the requested intersection. Unused outside
     * pieces are not measured; complementary clip APIs still certify both.
     * Outward local bounds reject separated interiors before and during clips,
     * including zero-volume contacts. Overlapping bounds supply no certificate. */
    [[nodiscard]] DynamicConvexGeometryPtr intersect(const DynamicConvexGeometry&) const;
    [[nodiscard]] DynamicConvexGeometryPtr translated(GeometryPoint displacement) const;
    /** Containment/disjointness precede completeness. Private subset ancestry
     * and opposite branches replace geometric tests when they prove the relation.
     * Authenticated complete
     * complement ancestry proves exact geometric closure without subtracting
     * dependent interval measures. Individual measure enclosures are unchanged.
     * Other partitions use the unchanged physical closure budget or prove an
     * empty remainder by bounded complementary subtraction after containment
     * and disjointness checks. Private remainder witnesses retain every strict
     * positive-interior child without measuring it; they cannot supply public
     * measures or packets. All proof work shares the original resource caps. */
    void certify_partition(std::span<const DynamicConvexGeometryPtr>, double relative_budget) const;
    /** Numerical certificate for an already authenticated owner/publication.
     * Homogeneous IEEE wider-FP format only; decode does not authenticate MPI
     * routing, source ownership, epochs or a physical acceptance budget. */
    [[nodiscard]] std::vector<char> encode(
        const GeometryPacketContext&, std::size_t byte_budget = 64 * 1024 * 1024) const;
    [[nodiscard]] static DynamicConvexGeometryPtr decode(std::span<const char>, const GeometryPacketContext&,
        std::span<const DynamicConvexGeometryPtr> authenticated_sources, GeometryLimits = {});
    /** Opaque immutable allocation identity for alias-aware payload accounting. */
    [[nodiscard]] const void* storage_identity() const noexcept { return impl_.get(); }
    [[nodiscard]] std::size_t storage_bytes() const;

private:
    template<ArithmeticMode> friend class details::ordinary_kernel::GeometryKernel;
    template<ArithmeticMode> friend class details::certification_kernel::GeometryKernel;
    template<ArithmeticMode> friend class details::ordinary_kernel::PartitionKernel;
    template<ArithmeticMode> friend class details::certification_kernel::PartitionKernel;
    friend class DynamicConvexPartition;
    template<ArithmeticMode> friend class ConvexCellGeometry;
    friend DynamicPoint dynamic_backtrace_constrained(
        const DynamicPoint&, GeometryPoint, double, std::span<const DynamicFaceConstraint>, GeometryLimits);
    explicit DynamicConvexGeometry(std::shared_ptr<const Impl>);
    [[nodiscard]] DynamicClipResult clip_impl(const std::shared_ptr<const Impl>& source, std::size_t plane,
        bool measure_section, bool construct_outside = true, bool partition_proof = false) const;
    std::shared_ptr<const Impl> impl_;
};

struct DynamicPartitionPiece
{
    std::size_t donor = 0;
    DynamicConvexGeometryPtr geometry;
};
/** Immutable, validated disjoint donor set. Holds the actual geometry objects,
 * not caller-supplied flags or IDs. Partitioning constructs a shared complement
 * tree, then measures one parent/donor intersection per used donor and binds
 * the complete output set to a private certificate. Temporary subdivisions
 * supply no transfer measures. Any uncovered remainder rejects.
 * The certificate is local; numerical packet decoding does not recreate it. */
class DynamicConvexPartition
{
public:
    explicit DynamicConvexPartition(std::span<const DynamicConvexGeometryPtr>, GeometryLimits = {});
    [[nodiscard]] std::vector<DynamicPartitionPiece> partition(
        const DynamicConvexGeometry&, std::size_t maximum_pieces = 65536) const;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;

private:
    template<ArithmeticMode> friend class details::ordinary_kernel::GeometryKernel;
    template<ArithmeticMode> friend class details::certification_kernel::GeometryKernel;
    template<ArithmeticMode> friend class details::ordinary_kernel::PartitionKernel;
    template<ArithmeticMode> friend class details::certification_kernel::PartitionKernel;
    template<ArithmeticMode> friend class ConvexPartition;
    std::vector<DynamicConvexGeometryPtr> donors_;
    GeometryLimits limits_;
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
    [[nodiscard]] std::vector<char> encode_certificate() const
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] static ConvexGeometryPtr<Mode> decode_certificate(std::span<const char>)
        requires(Mode == ArithmeticMode::Ordinary);
    [[nodiscard]] GeometryValue<Mode> volume() const;
    [[nodiscard]] GeometryMoments<Mode> moments() const { return moments(origin()); }
    [[nodiscard]] GeometryMoments<Mode> moments(GeometryPoint origin) const;
    [[nodiscard]] GeometryPoint centroid() const { return geometry_->centroid(); }
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
    [[nodiscard]] bool bounds_prove_disjoint_interiors(const ConvexCellGeometry& other) const
    {
        return geometry_->bounds_prove_disjoint_interiors(*other.geometry_);
    }
    [[nodiscard]] ConvexGeometryPtr<Mode> intersect(const ConvexCellGeometry&) const;
    [[nodiscard]] ConvexGeometryPtr<Mode> translated(GeometryPoint displacement) const;
    void certify_partition(std::span<const ConvexGeometryPtr<Mode>>, double relative_budget = 8e-15) const;
    [[nodiscard]] std::vector<char> encode(
        const GeometryPacketContext& context, std::size_t byte_budget = 64 * 1024 * 1024) const
    {
        return geometry_->encode(context, byte_budget);
    }
    [[nodiscard]] static ConvexGeometryPtr<Mode> decode(std::span<const char>, const GeometryPacketContext&,
        std::span<const ConvexGeometryPtr<Mode>> authenticated_sources, GeometryLimits = {});
    [[nodiscard]] const void* storage_identity() const noexcept { return geometry_->storage_identity(); }
    /** Retained payload plus both typed and runtime-adapter wrappers; shared
     * control blocks and allocator overhead are excluded. */
    [[nodiscard]] std::size_t storage_bytes() const
    {
        return sizeof(*this) + sizeof(DynamicConvexGeometry) + geometry_->storage_bytes();
    }
    /** The only runtime policy crossing. A foreign-mode object rejects. */
    [[nodiscard]] static ConvexGeometryPtr<Mode> from_dynamic(DynamicConvexGeometryPtr);
    [[nodiscard]] const DynamicConvexGeometryPtr& dynamic_geometry() const noexcept { return geometry_; }

private:
    template<ArithmeticMode> friend class PlanarFace;
    friend class details::NativeSupportAccess;
    [[nodiscard]] ConvexGeometryPtr<Mode> with_native_supports(std::span<const std::vector<char>>) const
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
    [[nodiscard]] std::size_t storage_bytes() const noexcept { return partition_.storage_bytes(); }

private:
    DynamicConvexPartition partition_;
};

template<ArithmeticMode Mode> [[nodiscard]] Point<Mode> point_average(std::span<const Point<Mode>>);
template<ArithmeticMode Mode>
[[nodiscard]] Point<Mode> point_interpolate(const Point<Mode>&, const Point<Mode>&, long double fraction);
template<ArithmeticMode Mode> [[nodiscard]] GeometryValue<Mode> orientation(const std::array<Point<Mode>, 4>&);
template<ArithmeticMode Mode> [[nodiscard]] bool points_identical(const Point<Mode>&, const Point<Mode>&);
template<ArithmeticMode Mode> [[nodiscard]] Point<Mode> point_displaced(const Point<Mode>&, const Point<Mode>&);
template<ArithmeticMode Mode>
[[nodiscard]] Point<Mode> source_point(
    const ConvexCellGeometry<ArithmeticMode::Ordinary>&, std::span<const char> point_identity);
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
