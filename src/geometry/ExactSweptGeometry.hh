/** @file ExactSweptGeometry.hh
 * @brief Shared exact trajectories, oriented flux chains and certified donor partitions.
 */
#pragma once
#include "geometry/ExactPlanarFace.hh"
#include <cstdint>
namespace SimpleFluid::Meshes
{
/** Exact rational scalar arithmetic retained independently of output rounding.
 * Geometry producers use volume units (m^3); derived inventory, momentum and
 * residual consumers retain their own physical units. This type does not check
 * dimensions. Conversion has the ConvexCellGeometry finite/range contract. */
class ExactGeometryMeasure
{
public:
    struct Impl;
    ExactGeometryMeasure();
    [[nodiscard]] static ExactGeometryMeasure from_value(double);
    [[nodiscard]] static ExactGeometryMeasure volume(const ConvexCellGeometry&);
    [[nodiscard]] CertifiedGeometryValue certified() const;
    /** Diagnostic enclosure, including nonzero residuals below output range.
     * exact_nonzero survives an interval containing zero; publication through
     * as_double() still rejects disappearance of a known nonzero measure. */
    [[nodiscard]] CertifiedGeometryValue enclosure() const;
    /** Exact relative acceptance; never forms tolerance*reference in floating
     * arithmetic. A zero reference admits only exact equality. */
    [[nodiscard]] bool relative_error_at_most(const ExactGeometryMeasure& reference, double tolerance) const;
    /** Approximate interpolation aid only, never an acceptance predicate.
     * Positive rational operands are normalized before taking a floating log. */
    [[nodiscard]] double log_ratio(const ExactGeometryMeasure& denominator) const;
    [[nodiscard]] std::vector<char> encode(std::size_t byte_budget = 1024 * 1024) const;
    [[nodiscard]] static ExactGeometryMeasure decode(std::span<const char>, std::size_t byte_budget = 1024 * 1024);
    [[nodiscard]] int sign() const;
    [[nodiscard]] int compare(const ExactGeometryMeasure&) const;
    [[nodiscard]] ExactGeometryMeasure operator+(const ExactGeometryMeasure&) const;
    [[nodiscard]] ExactGeometryMeasure operator-(const ExactGeometryMeasure&) const;
    [[nodiscard]] ExactGeometryMeasure scaled(double) const;
    /** Exact arithmetic with a retained-result byte limit and a conservative
     * integer bit-count preflight; temporary allocator peak is separate. */
    [[nodiscard]] ExactGeometryMeasure divided_by(
        const ExactGeometryMeasure&, std::size_t byte_budget = 1024 * 1024) const;
    [[nodiscard]] ExactGeometryMeasure squared(std::size_t byte_budget = 1024 * 1024) const;
    /** Fresh exact copy without arithmetic allocation slack. The explicit lease
     * covers simultaneous source and destination capacities; source is unchanged.
     * No represented-value conversion or change to other measure factories. */
    [[nodiscard]] ExactGeometryMeasure compacted(std::size_t operation_byte_budget) const;
    /** Truncate toward zero to at most significant_bits binary significand bits
     * at the exact value's own exponent. Nonzero signs survive below binary64
     * range. This is a proposed representation only: callers must independently
     * admit its exact discarded difference against their original certificate.
     * The lease covers source, result and conservative integer workspace;
     * resource failures throw length_error, distinct from malformed inputs. */
    [[nodiscard]] ExactGeometryMeasure rounded_dyadic(
        unsigned significant_bits, std::size_t operation_byte_budget) const;
    /** Capacity estimate, including the opaque rational allocation; excludes
     * allocator/control-block overhead. Deduplicate shared storage_identity(). */
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] const void* storage_identity() const noexcept { return impl_.get(); }

private:
    friend class ExactSweepAccess;
    friend class ExactSweptFace;
    friend class ExactDonorExport;
    explicit ExactGeometryMeasure(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
};
struct ExactPhaseMeasures;
/** Exact internal convex transport fragment. Unlike a native cell, a fragment
 * may be below the metric-output range. All shell/convexity/incidence checks
 * remain exact; narrow only after exact per-donor/recipient aggregation. */
class ExactConvexFragment
{
public:
    struct Impl;
    [[nodiscard]] static std::shared_ptr<const ExactConvexFragment> from_geometry(const ConvexCellGeometry&);
    [[nodiscard]] static std::shared_ptr<const ExactConvexFragment> decode(
        std::span<const char>, std::size_t byte_budget = 64 * 1024 * 1024);
    [[nodiscard]] std::vector<char> encode(std::size_t byte_budget = 64 * 1024 * 1024) const;
    [[nodiscard]] ExactGeometryMeasure volume() const;
    [[nodiscard]] ExactPhaseMeasures phases(const GeometryHalfspace& gas) const;
    [[nodiscard]] std::shared_ptr<const ExactConvexFragment> translated_exact(std::span<const char>) const;
    /** Explicitly enforce the ordinary ConvexCellGeometry output contract.
     * Individual checked metric/narrowing queries may still reject. */
    [[nodiscard]] ConvexGeometryPtr as_geometry() const;
    /** Exact arrays and rational capacities, excluding allocator/control blocks. */
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] const void* storage_identity() const noexcept { return impl_.get(); }

private:
    friend class ExactSweepAccess;
    explicit ExactConvexFragment(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
};
using ExactConvexFragmentPtr = std::shared_ptr<const ExactConvexFragment>;
struct ExactSweepLimits
{
    std::size_t maximum_simplices = 4096, maximum_atoms = 65536;
    std::size_t maximum_pair_tests = 262144;
    /** Cumulative admitted construction capacities, not process/allocator peak.
     * Exact clipping temporaries and allocator overhead are separate. */
    std::size_t byte_budget = 64 * 1024 * 1024;
};
/** Rank-local exact trajectory construction limits. Arithmetic capacities are
 * cumulative admitted intermediates, not allocator/process peak memory. */
struct ExactTrajectoryLimits
{
    std::size_t maximum_constraints = 256, maximum_operations = 65536;
    std::size_t maximum_integer_bits = 1024 * 1024, byte_budget = 64 * 1024 * 1024;
};
struct ExactNodeTrajectory
{
    std::vector<char> point, departure;
    /** A shared backward trajectory. Multiplication/subtraction remain exact
     * on represented inputs; no globally rounded departure node is constructed. */
    [[nodiscard]] static ExactNodeTrajectory backtrace(
        std::span<const char> point, GeometryPoint velocity, double timestep);
    /** Orthogonally project the represented displacement into the exact common
     * wall tangent space. The source must lie in every closed wall polygon.
     * Callers authenticate and complete the incident-wall catalog; this local
     * geometry primitive cannot establish native boundary ownership. Exactly
     * dependent constraints are redundant; nearly parallel constraints remain.
     * Empty walls preserve ordinary backtrace, and rank-three corners stay fixed. */
    [[nodiscard]] static ExactNodeTrajectory backtrace_constrained(std::span<const char> point, GeometryPoint velocity,
        double timestep, std::span<const ExactPlanarFacePtr> walls, ExactTrajectoryLimits = {});
    /** Exact squared physical displacement (m^2), including below-output-range
     * components. Compare directly with an exact CFL length-squared bound. */
    [[nodiscard]] ExactGeometryMeasure displacement_squared(ExactTrajectoryLimits = {}) const;
};
struct SignedSweepSimplex
{
    int sign = 1;
    ExactConvexFragmentPtr geometry;
};
struct ExactSweepDonor
{
    std::uint64_t id = 0;
    ConvexGeometryPtr geometry;
};
struct ExactSweepIntersection
{
    std::uint64_t donor = 0;
    int sign = 1;
    ExactConvexFragmentPtr geometry;
};
struct ExactPhaseMeasures
{
    ExactGeometryMeasure bulk, gas, carrier;
};
/** One canonical face, its shared edge trajectories and private interior
 * correction. Signed simplices intentionally permit crossed fluxes and exact
 * cancellation; they are not independent face-normal prisms. */
class ExactSweptFace
{
public:
    struct Impl;
    [[nodiscard]] static std::shared_ptr<const ExactSweptFace> make(
        const ExactPlanarFace&, std::span<const ExactNodeTrajectory>, double projected_volume, ExactSweepLimits = {});
    [[nodiscard]] static std::shared_ptr<const ExactSweptFace> make(const ExactPlanarFace&,
        std::span<const ExactNodeTrajectory>, const ExactGeometryMeasure& projected_volume, ExactSweepLimits = {});
    [[nodiscard]] const std::vector<SignedSweepSimplex>& simplices() const noexcept;
    [[nodiscard]] ExactPlanarFacePtr origin_face() const noexcept;
    [[nodiscard]] const std::vector<ExactNodeTrajectory>& trajectories() const noexcept;
    [[nodiscard]] ExactGeometryMeasure volume() const;
    [[nodiscard]] ExactGeometryMeasure uncorrected_volume() const;
    [[nodiscard]] std::vector<char> corrected_departure_center() const;
    /** Exact point triples for side triangles. Adjacent incident faces receive
     * identical unoriented triangles when they supply identical node traces. */
    [[nodiscard]] const std::vector<std::array<std::vector<char>, 3>>& edge_time_triangles() const noexcept;
    /** Every positive simplex is covered exactly by disjoint actual donors.
     * Missing donors or overlapping donor interiors reject before return. */
    [[nodiscard]] std::vector<ExactSweepIntersection> partition(
        std::span<const ExactSweepDonor>, ExactSweepLimits = {}) const;
    /** The supplied plane applies to every contribution in the span; callers
     * with donor-specific PLIC planes group/filter by authenticated donor. */
    [[nodiscard]] static ExactPhaseMeasures phases(
        std::span<const ExactSweepIntersection>, const GeometryHalfspace& gas);

private:
    friend class ExactDonorExport;
    explicit ExactSweptFace(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
};
struct OrientedFaceSweep
{
    std::shared_ptr<const ExactSweptFace> face;
    int outward_sign = 1;
};
/** Certify the signed sum of a cell's incident outward sweeps on its actual
 * donor geometry. Exact arrangement atoms must have export multiplicity0/1;
 * negative or repeated export rejects. Complements are retained independently,
 * never computed by subtracting rounded nearly equal volumes. */
class ExactDonorExport
{
public:
    struct Impl;
    [[nodiscard]] static ExactDonorExport make(
        const ConvexCellGeometry&, std::span<const OrientedFaceSweep>, ExactSweepLimits = {});
    [[nodiscard]] ExactGeometryMeasure exported_volume() const;
    [[nodiscard]] ExactGeometryMeasure retained_volume() const;
    [[nodiscard]] ExactPhaseMeasures exported_phases(const GeometryHalfspace& gas) const;
    [[nodiscard]] ExactPhaseMeasures retained_phases(const GeometryHalfspace& gas) const;
    [[nodiscard]] const std::vector<ExactConvexFragmentPtr>& exported_atoms() const noexcept;
    [[nodiscard]] const std::vector<ExactConvexFragmentPtr>& retained_atoms() const noexcept;

private:
    explicit ExactDonorExport(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
};
struct ExactSweepRecipient
{
    std::uint64_t id = 0;
    ConvexGeometryPtr geometry;
    std::vector<OrientedFaceSweep> outward_faces;
};
struct ExactTransportAtom
{
    std::uint64_t donor = 0, recipient = 0;
    ExactConvexFragmentPtr geometry;
};
/** Closed-cell pullback: indicator(target) - sum(oriented face chains).
 * Every donor arrangement atom must belong to exactly one recipient with
 * coefficient1. Exact donor and native recipient capacities must agree; no
 * tolerance hides divergence in independently rounded projected face fluxes. */
class ExactTransportPartition
{
public:
    [[nodiscard]] static ExactTransportPartition make(
        std::span<const ExactSweepDonor>, std::span<const ExactSweepRecipient>, ExactSweepLimits = {});
    /** Target-owner proof on a bounded donor halo. Unassigned donor regions
     * may belong to remote targets; donor owners must certify_donor_cover over
     * their authenticated incoming positive atoms before global publication. */
    [[nodiscard]] static ExactTransportPartition local(
        std::span<const ExactSweepDonor>, std::span<const ExactSweepRecipient>, ExactSweepLimits = {});
    static void certify_donor_cover(const ExactSweepDonor&, std::span<const ExactTransportAtom>, ExactSweepLimits = {});
    [[nodiscard]] const std::vector<ExactTransportAtom>& atoms() const noexcept { return atoms_; }
    /** Informational proof-path diagnostic; no policy or tolerance switch. */
    [[nodiscard]] bool used_departure_fan() const noexcept { return used_departure_fan_; }
    [[nodiscard]] static ExactPhaseMeasures phases(std::span<const ExactTransportAtom>, const GeometryHalfspace& gas);

private:
    ExactTransportPartition() = default;
    std::vector<ExactTransportAtom> atoms_;
    bool used_departure_fan_ = false;
};
} // namespace SimpleFluid::Meshes
