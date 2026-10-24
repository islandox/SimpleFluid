/** @file BoundedConvexGeometry.hh
 * @brief Separate bounded floating-point convex geometry for finite VoF.
 */
#pragma once
#include "geometry/ConvexCellGeometry.hh"
#include "utils/ArithmeticMode.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::Meshes
{
namespace bounded_geometry_detail
{
namespace ordinary_kernel
{
template<ArithmeticMode> class GeometryKernel;
template<ArithmeticMode> class PartitionKernel;
}
namespace certification_kernel
{
template<ArithmeticMode> class GeometryKernel;
template<ArithmeticMode> class PartitionKernel;
}
}
/** Explicitly selected finite arithmetic value. In BoundedCertification the
 * fields enclose the original construction in physical units; zero-containing
 * intervals establish neither zero nor positivity. In Ordinary only value is
 * meaningful: absolute_error is untracked, and certificate queries reject.
 */
struct BoundedGeometryValue
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
struct BoundedPlaneAuthority
{
    std::vector<std::uint64_t> key;
    bool operator==(const BoundedPlaneAuthority&) const = default;
    auto operator<=>(const BoundedPlaneAuthority&) const = default;
};
struct BoundedPointProvenance;
class BoundedGeometryAccess;
struct BoundedPoint
{
    std::array<long double, 3> value{}, absolute_error{};
    ArithmeticMode arithmetic_mode = ArithmeticMode::BoundedCertification;
    BoundedPoint() = default;
    BoundedPoint(std::array<long double, 3> coordinates, std::array<long double, 3> error = {},
        ArithmeticMode mode = ArithmeticMode::BoundedCertification)
        : value(coordinates), absolute_error(error), arithmetic_mode(mode)
    {
    }
    /** Changing coordinates/error invalidates the sealed source provenance. */
    [[nodiscard]] const std::vector<BoundedPlaneAuthority>& supports() const;

private:
    friend class BoundedGeometryAccess;
    std::shared_ptr<const BoundedPointProvenance> provenance_;
};
[[nodiscard]] std::vector<BoundedPlaneAuthority> bounded_common_supports(std::span<const BoundedPoint>);
[[nodiscard]] BoundedPoint bounded_point_average(std::span<const BoundedPoint>);
[[nodiscard]] BoundedPoint bounded_point_interpolate(const BoundedPoint&, const BoundedPoint&, long double fraction);
struct BoundedSupport
{
    BoundedGeometryValue lower, upper;
};
struct BoundedGeometryMoments
{
    BoundedGeometryValue volume;
    GeometryPoint origin{};
    std::array<BoundedGeometryValue, 3> first_moment;
};
struct BoundedGeometrySection
{
    std::vector<BoundedPoint> vertices;
    BoundedGeometryValue area;
    BoundedPoint centroid;
};
/** Exact transaction pins supplied by the authenticated finite transport owner.
 * The owner defines the four words (publication/policy, geometry, partition,
 * trial); all words are compared, never inferred from numerical coordinates. */
struct BoundedGeometryPacketContext
{
    std::array<std::uint64_t, 4> publication{};
    bool operator==(const BoundedGeometryPacketContext&) const = default;
};
struct BoundedGeometryLimits
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
struct BoundedGeometryStatistics
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
[[nodiscard]] BoundedGeometryStatistics bounded_geometry_statistics();
/** Calling image/thread identity; use only for duplicate-domain suppression. */
[[nodiscard]] const void* bounded_geometry_statistics_domain();
void reset_bounded_geometry_statistics();
void bounded_record_ledger_operations(std::uint64_t count = 1);
[[nodiscard]] bool bounded_geometry_environment();
/** Six times the oriented tetrahedron volume, retaining input uncertainty. */
[[nodiscard]] BoundedGeometryValue bounded_orientation(const std::array<BoundedPoint, 4>&);
[[nodiscard]] bool bounded_points_identical(const BoundedPoint&, const BoundedPoint&);
/** Retained point allocation, including sealed provenance/support keys. */
[[nodiscard]] std::size_t bounded_point_storage_bytes(const BoundedPoint&);
/** Bootstrap-only source coordinate/support export, separate from step work. */
[[nodiscard]] BoundedPoint bounded_source_point(const ConvexCellGeometry&, std::span<const char> point_identity,
    ArithmeticMode = ArithmeticMode::BoundedCertification);
/** One bootstrap coordinate export plus the union of true supports over the
 * complete authenticated incident source-cell fan (including hanging nodes). */
[[nodiscard]] BoundedPoint bounded_source_point(
    std::span<const ConvexGeometryPtr> incident_sources, std::span<const char> point_identity,
    ArithmeticMode = ArithmeticMode::BoundedCertification);
[[nodiscard]] BoundedPoint bounded_point_displaced(const BoundedPoint&, const BoundedPoint& displacement);

class BoundedConvexGeometry;
class BoundedConvexPartition;
using BoundedConvexGeometryPtr = std::shared_ptr<const BoundedConvexGeometry>;
struct BoundedFaceConstraint
{
    BoundedConvexGeometryPtr geometry;
    std::size_t face = 0;
};
/** Authenticate every wall incidence, then construct motion in the common
 * tangent space using certified cross/triple-product rank. Ambiguous rank
 * rejects; nearly parallel walls are never merged by an angular tolerance. */
[[nodiscard]] BoundedPoint bounded_backtrace_constrained(const BoundedPoint&, GeometryPoint velocity, double timestep,
    std::span<const BoundedFaceConstraint>, BoundedGeometryLimits = {});
struct BoundedClipResult
{
    BoundedConvexGeometryPtr inside, outside;
    BoundedGeometrySection section;
};
struct BoundedPhaseMeasures
{
    BoundedGeometryValue bulk, gas, carrier;
};

/** Immutable local, power-of-two-scaled convex shell. Ordinary uses nominal FP
 * arithmetic. BoundedCertification uses directed enclosures and rejects
 * unresolved certification. Exact is rejected; there is no automatic fallback.
 * Caps apply to each operation; callers also charge aggregate trial work.
 */
class BoundedConvexGeometry
{
public:
    struct Impl;
    [[nodiscard]] ArithmeticMode arithmetic_mode() const noexcept;
    [[nodiscard]] static BoundedConvexGeometryPtr make(
        std::span<const GeometryPoint>, const std::vector<std::vector<std::size_t>>&, BoundedGeometryLimits = {});
    /** Global coordinates retain valid sealed input-point identities locally;
     * equal coordinate/error tuples alone never establish point identity. */
    [[nodiscard]] static BoundedConvexGeometryPtr make_bounded(
        std::span<const BoundedPoint>, const std::vector<std::vector<std::size_t>>&, BoundedGeometryLimits = {});
    [[nodiscard]] static BoundedConvexGeometryPtr make_bounded_local(GeometryPoint origin,
        std::span<const BoundedPoint> relative_vertices, const std::vector<std::vector<std::size_t>>&,
        BoundedGeometryLimits = {});
    /** Explicit source-authority export, once per immutable source: coordinate
     * enclosures only, never exact volumes or moments. */
    [[nodiscard]] static BoundedConvexGeometryPtr from_geometry(const ConvexCellGeometry&, BoundedGeometryLimits = {});
    [[nodiscard]] static BoundedConvexGeometryPtr box(GeometryPoint, GeometryPoint, BoundedGeometryLimits = {});
    /** Null only for proven zero determinant; an uncertain determinant rejects. */
    [[nodiscard]] static BoundedConvexGeometryPtr tetrahedron(
        const std::array<BoundedPoint, 4>&, BoundedGeometryLimits = {});
    /** A strict-sign clipping construction proves positive interior even when
     * its quantitative volume enclosure includes zero. Such uncertainty is
     * retained; division, centroid and packet export still require a positive
     * numerical lower bound. Independent imports retain strict admission. */
    [[nodiscard]] BoundedGeometryValue volume() const;
    [[nodiscard]] BoundedGeometryMoments moments() const;
    [[nodiscard]] BoundedGeometryMoments moments(GeometryPoint origin) const;
    [[nodiscard]] GeometryPoint centroid() const;
    /** Stable default moment reference. Numerical clipping frames may rebase
     * privately without moving the physical geometry or this reference. */
    [[nodiscard]] GeometryPoint origin() const;
    [[nodiscard]] const std::vector<BoundedPoint>& points() const;
    /** Coordinates relative to origin(); global support authority is omitted. */
    [[nodiscard]] std::vector<BoundedPoint> local_points() const;
    [[nodiscard]] const std::vector<std::vector<std::size_t>>& faces() const;
    [[nodiscard]] std::array<BoundedGeometryValue, 3> axis_spans() const;
    [[nodiscard]] BoundedSupport support(GeometryPoint normal, GeometryPoint anchor) const;
    [[nodiscard]] BoundedClipResult clip(const GeometryHalfspace&, bool measure_section = true) const;
    [[nodiscard]] BoundedClipResult clip_by_face(
        const BoundedConvexGeometry&, std::size_t face, bool measure_section = false) const;
    [[nodiscard]] BoundedPhaseMeasures phases(const GeometryHalfspace&) const;
    /** Test nominal spatial extents in Ordinary; in BoundedCertification true
     * certifies disjoint interiors with outward extents. False is inconclusive.
     * Face/edge/point contacts may still exist.
     * This certificate concerns volume/moments, not surface integrals. */
    [[nodiscard]] bool bounds_prove_disjoint_interiors(const BoundedConvexGeometry&) const;
    /** Construct and certify only the requested intersection. Unused outside
     * pieces are not measured; complementary clip APIs still certify both.
     * Outward local bounds reject separated interiors before and during clips,
     * including zero-volume contacts. Overlapping bounds supply no certificate. */
    [[nodiscard]] BoundedConvexGeometryPtr intersect(const BoundedConvexGeometry&) const;
    [[nodiscard]] BoundedConvexGeometryPtr translated(GeometryPoint displacement) const;
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
    void certify_partition(std::span<const BoundedConvexGeometryPtr>, double relative_budget) const;
    /** Numerical certificate for an already authenticated owner/publication.
     * Homogeneous IEEE wider-FP format only; decode does not authenticate MPI
     * routing, source ownership, epochs or a physical acceptance budget. */
    [[nodiscard]] std::vector<char> encode(
        const BoundedGeometryPacketContext&, std::size_t byte_budget = 64 * 1024 * 1024) const;
    [[nodiscard]] static BoundedConvexGeometryPtr decode(std::span<const char>, const BoundedGeometryPacketContext&,
        std::span<const BoundedConvexGeometryPtr> authenticated_sources, BoundedGeometryLimits = {});
    /** Opaque immutable allocation identity for alias-aware payload accounting. */
    [[nodiscard]] const void* storage_identity() const noexcept { return impl_.get(); }
    [[nodiscard]] std::size_t storage_bytes() const;

private:
    template<ArithmeticMode> friend class bounded_geometry_detail::ordinary_kernel::GeometryKernel;
    template<ArithmeticMode> friend class bounded_geometry_detail::certification_kernel::GeometryKernel;
    template<ArithmeticMode> friend class bounded_geometry_detail::ordinary_kernel::PartitionKernel;
    template<ArithmeticMode> friend class bounded_geometry_detail::certification_kernel::PartitionKernel;
    friend class BoundedConvexPartition;
    friend BoundedPoint bounded_backtrace_constrained(
        const BoundedPoint&, GeometryPoint, double, std::span<const BoundedFaceConstraint>, BoundedGeometryLimits);
    explicit BoundedConvexGeometry(std::shared_ptr<const Impl>);
    [[nodiscard]] BoundedClipResult clip_impl(
        const std::shared_ptr<const Impl>& source, std::size_t plane, bool measure_section,
        bool construct_outside = true, bool partition_proof = false) const;
    std::shared_ptr<const Impl> impl_;
};

struct BoundedPartitionPiece
{
    std::size_t donor = 0;
    BoundedConvexGeometryPtr geometry;
};
/** Immutable, validated disjoint donor set. Holds the actual geometry objects,
 * not caller-supplied flags or IDs. Partitioning constructs a shared complement
 * tree, then measures one parent/donor intersection per used donor and binds
 * the complete output set to a private certificate. Temporary subdivisions
 * supply no transfer measures. Any uncovered remainder rejects.
 * The certificate is local; numerical packet decoding does not recreate it. */
class BoundedConvexPartition
{
public:
    explicit BoundedConvexPartition(std::span<const BoundedConvexGeometryPtr>, BoundedGeometryLimits = {});
    [[nodiscard]] std::vector<BoundedPartitionPiece> partition(
        const BoundedConvexGeometry&, std::size_t maximum_pieces = 65536) const;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;

private:
    template<ArithmeticMode> friend class bounded_geometry_detail::ordinary_kernel::GeometryKernel;
    template<ArithmeticMode> friend class bounded_geometry_detail::certification_kernel::GeometryKernel;
    template<ArithmeticMode> friend class bounded_geometry_detail::ordinary_kernel::PartitionKernel;
    template<ArithmeticMode> friend class bounded_geometry_detail::certification_kernel::PartitionKernel;
    std::vector<BoundedConvexGeometryPtr> donors_;
    BoundedGeometryLimits limits_;
};
} // namespace SimpleFluid::Meshes
