/** @file GeneralVoFSolver.hh
 * @brief Source-free geometric two-phase flow on certified native polyhedral cells.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "FVM/GeneralVoFBalance.hh"
#include "geometry/GeometryStatisticsSnapshot.hh"
#include "solvers/VoFSolver.hh"
namespace SimpleFluid
{
/** General native flow controls. Physical tolerances retain their original
 * extensive/face equations. Phase cleanup is configured on the supplied authority.
 */
struct GeneralVoFSolverOptions
{
    VoFSolverOptions flow;
    VoF::GeneralVoFSurfaceOptions capillary;
    int maximum_momentum_iterations = 3000;
    double momentum_relative_tolerance = 1e-12;
    /** Momentum exact Input/Scratch/Result buffers and retained payloads use
     * cumulative admission, with shared opaque storage deduplicated per ledger.
     * Nested face binding conservatively reserves its own input accounting;
     * aliases already retained by the outer solve may therefore reduce usable
     * headroom. Momentum backend/Krylov allocations and allocator overhead are excluded.
     * Projection/forest reserve cumulative payload slots, explicit buffers and
     * bounded sparse packet arenas before arithmetic/communication. Conservative
     * per-scalar slot admission can reject before all byte capacity is consumed.
     * Ordinary compatible projection additionally bounds a conservative
     * matrix/Krylov payload estimate within the same rank-local cap; opaque
     * backend allocator overhead remains outside that estimate.
     * Resource exhaustion rejects; it never changes physical tolerances.
     */
    std::size_t byte_budget = 64 * 1024 * 1024;
};
struct GeneralVoFMomentumDiagnostics
{
    /** Original raw transport equation, before declared phase relabeling. */
    double physical_relative_residual = 0;
    int implicit_iterations = 0, defect_corrections = 0;
    /** Global signed kg m/s added by this step's phase relabeling, at the
     * implicit transported velocity before forces/projection. Kept separate
     * from arithmetic uncertainty and the raw transport physical residual. */
    long double cleanup_momentum = 0, accumulated_cleanup_momentum = 0;
    double cleanup_identity_relative_residual = 0;
};
/** Synchronous live ownership operation. Forecasts remain one-step-lag observations;
 * timings are maximum-rank wall time, native bytes exclude phase/solver fields. */
struct GeneralVoFRebalanceResult
{
    VoF::GeneralVoFBalanceProposal proposal;
    bool applied = false;
    double proposal_seconds = 0, migration_seconds = 0;
    std::uint64_t local_native_application_bytes = 0, global_native_application_bytes = 0;
};
/** Independent general-mesh backend; Cartesian specialized solvers are retained.
 *
 * The supplied immutable phase authority owns native source geometry and
 * independent gas/carrier volumes. Its arithmetic mode selects Ordinary
 * (default), BoundedCertification, or the experimental Exact path. Exact retains
 * exact phase history and remains unqualified, with known memory/performance
 * gaps. The solver pins the source mode for geometry, phase, mass and momentum.
 * Ordinary uses represented-value physical acceptance; only the opt-in bounded
 * mode requires interval certificates. Independent accumulated arithmetic
 * history follows phase transfers in bounded mode. Neither FP mode selects a
 * rational numerical construction, measure or ledger fallback.
 * All construction, initialization, timestep and
 * advance operations are collective, including empty-owner ranks. Pressure,
 * viscosity and curvature use actual physical geometry. Source-free compatible
 * momentum uses positive authenticated donor transfers and the original raw face
 * donor equation; material-volume sources, species sources and drift are absent.
 * Explicit phase cleanup is a separately budgeted relabeling: raw momentum
 * transport passes its original residual before the actual mass correction
 * adds momentum at the implicit transported velocity. Cleaned mass supplies
 * density, force acceleration and projection; cleanup momentum is reported
 * separately from the physical transport residual and arithmetic uncertainty.
 *
 * A failed step leaves the complete accepted publication and clock unchanged.
 * Source-free occupancy, reconstruction/support, donor-capacity, physical linear
 * residual and conditioning failures reject. Only the authority's explicit
 * cleanup policy can relabel phase inventory. References are invalidated by successful mutation.
 */
class SIMPLEFLUID_SOLVERS_EXPORT GeneralVoFSolver
{
public:
    /** This library image's calling-thread counters. Call every observed
     * image's bridge and deduplicate snapshots by their separate domain IDs. */
    static void enable_geometry_statistics(bool enabled);
    static void reset_geometry_statistics();
    [[nodiscard]] static Meshes::GeometryStatisticsSnapshot geometry_statistics();
    explicit GeneralVoFSolver(
        std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial material = {}, GeneralVoFSolverOptions options = {});
    ~GeneralVoFSolver();
    GeneralVoFSolver(GeneralVoFSolver&&) noexcept;
    GeneralVoFSolver& operator=(GeneralVoFSolver&&) noexcept;
    GeneralVoFSolver(const GeneralVoFSolver&) = delete;
    GeneralVoFSolver& operator=(const GeneralVoFSolver&) = delete;
    /** Owned cell velocities [m/s]; optional canonical overlap normal velocities.
     * Omitted cell values are reconstructed from supplied faces; omitting both
     * arrays selects zero flow. Supplied cell/face values retain their native order.
     * Ordinary projects the physical cell momentum into compatible closed
     * canonical rates and cell moments. Supplied faces are a predictor estimate,
     * not independent prescribed forcing when cell velocities are supplied.
     */
    void initialize(
        const std::vector<VoF::Vector>& owned_velocity = {}, const std::vector<double>& overlap_face_velocity = {});
    [[nodiscard]] VoFStepDiagnostics step(double timestep);
    [[nodiscard]] double stable_timestep() const;
    /** Conservative fixed-clock adaptation of phases and physical momentum.
     * Source keys are persistent hierarchy keys; rejected plans preserve all
     * accepted state. Geometry-dependent operators are rebuilt on the candidate.
     * Enabled phase cleanup rejects this coupled momentum/pressure remap.
     */
    void regrid(
        std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen = {});
    /** Ownership-only native migration in current owned-key order. Ordinary
     * preserves accepted keyed fields and cleanup accounting without arithmetic
     * recovery. Other modes retain their existing coupled-remap restrictions. */
    void repartition(std::span<const int> destinations);
    /** Collectively forecast from this exact accepted state's last complete
     * captured Ordinary interface-local step and apply an accepted proposal.
     * Rejection preserves the publication. Success clears partition-bound
     * prepared work; another accepted captured step is required before reuse. */
    [[nodiscard]] GeneralVoFRebalanceResult rebalance(const VoF::GeneralVoFBalanceOptions& options = {});
    [[nodiscard]] double time() const noexcept;
    [[nodiscard]] const std::shared_ptr<const VoF::GeneralVoF>& phases() const noexcept;
    [[nodiscard]] const std::shared_ptr<const MeshHandle<>>& mesh() const noexcept;
    /** Bounded geometric fitting/CSF label view, distinct from independent
     * physical gas/carrier extensive inventories in phases(). */
    [[nodiscard]] const std::vector<double>& gas_fraction() const noexcept;
    [[nodiscard]] const std::array<GeneralVoFMomentumDiagnostics, 3>& momentum_transport_diagnostics() const noexcept;
    /** Accepted face-stream step and attached reusable momentum preparation.
     * Empty before advancement or for the legacy swept backend. */
    [[nodiscard]] const std::shared_ptr<const VoF::GeneralVoFPreparedTransport>& prepared_transport() const noexcept;
    [[nodiscard]] const std::vector<VoF::Vector>& velocity() const noexcept;
    [[nodiscard]] const std::vector<double>& face_velocity() const noexcept;
    [[nodiscard]] const std::vector<double>& pressure() const noexcept;
    [[nodiscard]] const std::vector<double>& curvature() const noexcept;
    [[nodiscard]] const VoF::GeneralVoFCapillaryDiagnostics& capillary_diagnostics() const noexcept;
    [[nodiscard]] const VoFStepDiagnostics& diagnostics() const noexcept;

private:
    struct SIMPLEFLUID_SOLVERS_LOCAL Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid
