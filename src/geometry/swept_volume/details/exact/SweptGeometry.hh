#pragma once
#include "geometry/swept_volume/SweptGeometryTypes.hh"
namespace SimpleFluid::geometry::swept_volume
{
/** Selecting this specialization explicitly imports a represented finite source
 * into the optional exact construction. Native catalogs remain ordinary. */
template<> struct SweepDonor<ArithmeticMode::Exact>
{
    std::uint64_t id = 0;
    PLIC::ConvexGeometryPtr<ArithmeticMode::Exact> geometry;
    SweepDonor() = default;
    SweepDonor(std::uint64_t source, PLIC::ConvexGeometryPtr<ArithmeticMode::Exact> cell)
        : id(source), geometry(std::move(cell))
    {
    }
    SweepDonor(std::uint64_t, const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>&);
};
template<> struct SweepRecipient<ArithmeticMode::Exact>
{
    std::uint64_t id = 0;
    PLIC::ConvexGeometryPtr<ArithmeticMode::Exact> geometry;
    std::vector<OrientedFaceSweep<ArithmeticMode::Exact>> outward_faces;
    SweepRecipient() = default;
    SweepRecipient(std::uint64_t target, PLIC::ConvexGeometryPtr<ArithmeticMode::Exact> cell,
        std::vector<OrientedFaceSweep<ArithmeticMode::Exact>> faces)
        : id(target), geometry(std::move(cell)), outward_faces(std::move(faces))
    {
    }
    SweepRecipient(std::uint64_t, const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>&,
        std::vector<OrientedFaceSweep<ArithmeticMode::Exact>>);
};
/** Exact internal convex transport fragment. Unlike a native cell, a fragment
 * may be below the metric-output range. All shell/convexity/incidence checks
 * remain exact; narrow only after exact per-donor/recipient aggregation. */
template<> class ConvexFragment<ArithmeticMode::Exact>
{
public:
    struct Impl;
    [[nodiscard]] static std::shared_ptr<const ConvexFragment<ArithmeticMode::Exact>> from_geometry(
        const PLIC::ConvexCellGeometry<ArithmeticMode::Exact>&);
    [[nodiscard]] static std::shared_ptr<const ConvexFragment<ArithmeticMode::Exact>> from_geometry(
        const PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>&);
    [[nodiscard]] static std::shared_ptr<const ConvexFragment<ArithmeticMode::Exact>> decode(
        std::span<const char>, std::size_t byte_budget = 64 * 1024 * 1024);
    [[nodiscard]] std::vector<char> encode(std::size_t byte_budget = 64 * 1024 * 1024) const;
    [[nodiscard]] PLIC::GeometryMeasure<ArithmeticMode::Exact> volume() const;
    [[nodiscard]] PLIC::PhaseMeasures<ArithmeticMode::Exact> phases(const PLIC::GeometryHalfspace& gas) const;
    [[nodiscard]] std::shared_ptr<const ConvexFragment<ArithmeticMode::Exact>> translated(
        const PLIC::DisplacementIdentity<ArithmeticMode::Exact>&) const;
    /** Explicitly enforce the ordinary PLIC::ConvexCellGeometry<ArithmeticMode::Exact> output contract.
     * Individual checked metric/narrowing queries may still reject. */
    [[nodiscard]] PLIC::ConvexGeometryPtr<ArithmeticMode::Exact> as_geometry() const;
    /** Exact arrays and rational capacities, excluding allocator/control blocks. */
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] const void* storage_identity() const noexcept { return impl_.get(); }

private:
    friend class SweepAccess<ArithmeticMode::Exact>;
    explicit ConvexFragment(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
};
/** Rank-local exact trajectory construction limits. Arithmetic capacities are
 * cumulative admitted intermediates, not allocator/process peak memory. */
template<> struct NodeTrajectory<ArithmeticMode::Exact>
{
    PLIC::PointIdentity<ArithmeticMode::Exact> point, departure;
    /** A shared backward trajectory. Multiplication/subtraction remain exact
     * on represented inputs; no globally rounded departure node is constructed. */
    [[nodiscard]] static NodeTrajectory<ArithmeticMode::Exact> backtrace(
        const PLIC::PointIdentity<ArithmeticMode::Exact>& point, PLIC::GeometryPoint velocity, double timestep);
    /** Orthogonally project the represented displacement into the exact common
     * wall tangent space. The source must lie in every closed wall polygon.
     * Callers authenticate and complete the incident-wall catalog; this local
     * geometry primitive cannot establish native boundary ownership. Exactly
     * dependent constraints are redundant; nearly parallel constraints remain.
     * Empty walls preserve ordinary backtrace, and rank-three corners stay fixed. */
    [[nodiscard]] static NodeTrajectory<ArithmeticMode::Exact> backtrace_constrained(
        const PLIC::PointIdentity<ArithmeticMode::Exact>& point, PLIC::GeometryPoint velocity, double timestep,
        std::span<const PLIC::PlanarFacePtr<ArithmeticMode::Exact>> walls, TrajectoryLimits = {});
    /** Exact squared physical displacement (m^2), including below-output-range
     * components. Compare directly with an exact CFL length-squared bound. */
    [[nodiscard]] PLIC::GeometryMeasure<ArithmeticMode::Exact> displacement_squared(TrajectoryLimits = {}) const;
};

/** One canonical face, its shared edge trajectories and private interior
 * correction. Signed simplices intentionally permit crossed fluxes and exact
 * cancellation; they are not independent face-normal prisms. */
template<> class SweptFace<ArithmeticMode::Exact>
{
public:
    struct Impl;
    [[nodiscard]] static std::shared_ptr<const SweptFace<ArithmeticMode::Exact>> make(
        const PLIC::PlanarFace<ArithmeticMode::Exact>&, std::span<const NodeTrajectory<ArithmeticMode::Exact>>,
        double projected_volume, SweepLimits = {});
    [[nodiscard]] static std::shared_ptr<const SweptFace<ArithmeticMode::Exact>> make(
        const PLIC::PlanarFace<ArithmeticMode::Exact>&, std::span<const NodeTrajectory<ArithmeticMode::Exact>>,
        const PLIC::GeometryMeasure<ArithmeticMode::Exact>& projected_volume, SweepLimits = {});
    [[nodiscard]] const std::vector<SignedSweepSimplex<ArithmeticMode::Exact>>& simplices() const noexcept;
    [[nodiscard]] PLIC::PlanarFacePtr<ArithmeticMode::Exact> origin_face() const noexcept;
    [[nodiscard]] const std::vector<NodeTrajectory<ArithmeticMode::Exact>>& trajectories() const noexcept;
    [[nodiscard]] PLIC::GeometryMeasure<ArithmeticMode::Exact> volume() const;
    [[nodiscard]] PLIC::GeometryMeasure<ArithmeticMode::Exact> uncorrected_volume() const;
    [[nodiscard]] PLIC::PointIdentity<ArithmeticMode::Exact> corrected_departure_center() const;
    /** Exact point triples for side triangles. Adjacent incident faces receive
     * identical unoriented triangles when they supply identical node traces. */
    [[nodiscard]] const std::vector<std::array<PLIC::PointIdentity<ArithmeticMode::Exact>, 3>>&
    edge_time_triangles() const noexcept;
    /** Every positive simplex is covered exactly by disjoint actual donors.
     * Missing donors or overlapping donor interiors reject before return. */
    [[nodiscard]] std::vector<SweepIntersection<ArithmeticMode::Exact>> partition(
        std::span<const SweepDonor<ArithmeticMode::Exact>>, SweepLimits = {}) const;
    /** The supplied plane applies to every contribution in the span; callers
     * with donor-specific PLIC planes group/filter by authenticated donor. */
    [[nodiscard]] static PLIC::PhaseMeasures<ArithmeticMode::Exact> phases(
        std::span<const SweepIntersection<ArithmeticMode::Exact>>, const PLIC::GeometryHalfspace& gas);

private:
    friend class DonorExport<ArithmeticMode::Exact>;
    explicit SweptFace(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
};

/** Certify the signed sum of a cell's incident outward sweeps on its actual
 * donor geometry. Exact arrangement atoms must have export multiplicity0/1;
 * negative or repeated export rejects. Complements are retained independently,
 * never computed by subtracting rounded nearly equal volumes. */
template<> class DonorExport<ArithmeticMode::Exact>
{
public:
    struct Impl;
    [[nodiscard]] static DonorExport<ArithmeticMode::Exact> make(const PLIC::ConvexCellGeometry<ArithmeticMode::Exact>&,
        std::span<const OrientedFaceSweep<ArithmeticMode::Exact>>, SweepLimits = {});
    [[nodiscard]] static DonorExport<ArithmeticMode::Exact> make(const PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>&,
        std::span<const OrientedFaceSweep<ArithmeticMode::Exact>>, SweepLimits = {});
    [[nodiscard]] PLIC::GeometryMeasure<ArithmeticMode::Exact> exported_volume() const;
    [[nodiscard]] PLIC::GeometryMeasure<ArithmeticMode::Exact> retained_volume() const;
    [[nodiscard]] PLIC::PhaseMeasures<ArithmeticMode::Exact> exported_phases(const PLIC::GeometryHalfspace& gas) const;
    [[nodiscard]] PLIC::PhaseMeasures<ArithmeticMode::Exact> retained_phases(const PLIC::GeometryHalfspace& gas) const;
    [[nodiscard]] const std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>>& exported_atoms() const noexcept;
    [[nodiscard]] const std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>>& retained_atoms() const noexcept;

private:
    explicit DonorExport(std::shared_ptr<const Impl>);
    std::shared_ptr<const Impl> impl_;
};

/** Closed-cell pullback: indicator(target) - sum(oriented face chains).
 * Every donor arrangement atom must belong to exactly one recipient with
 * coefficient1. Exact donor and native recipient capacities must agree; no
 * tolerance hides divergence in independently rounded projected face fluxes. */
template<> class TransportPartition<ArithmeticMode::Exact>
{
public:
    [[nodiscard]] static TransportPartition<ArithmeticMode::Exact> make(
        std::span<const SweepDonor<ArithmeticMode::Exact>>, std::span<const SweepRecipient<ArithmeticMode::Exact>>,
        SweepLimits = {});
    /** Target-owner proof on a bounded donor halo. Unassigned donor regions
     * may belong to remote targets; donor owners must certify_donor_cover over
     * their authenticated incoming positive atoms before global publication. */
    [[nodiscard]] static TransportPartition<ArithmeticMode::Exact> local(
        std::span<const SweepDonor<ArithmeticMode::Exact>>, std::span<const SweepRecipient<ArithmeticMode::Exact>>,
        SweepLimits = {});
    static void certify_donor_cover(const SweepDonor<ArithmeticMode::Exact>&,
        std::span<const TransportAtom<ArithmeticMode::Exact>>, SweepLimits = {});
    [[nodiscard]] const std::vector<TransportAtom<ArithmeticMode::Exact>>& atoms() const noexcept { return atoms_; }
    /** Informational proof-path diagnostic; no policy or tolerance switch. */
    [[nodiscard]] bool used_departure_fan() const noexcept { return used_departure_fan_; }
    [[nodiscard]] static PLIC::PhaseMeasures<ArithmeticMode::Exact> phases(
        std::span<const TransportAtom<ArithmeticMode::Exact>>, const PLIC::GeometryHalfspace& gas);

private:
    TransportPartition() = default;
    std::vector<TransportAtom<ArithmeticMode::Exact>> atoms_;
    bool used_departure_fan_ = false;
};
} // namespace SimpleFluid::geometry::swept_volume
