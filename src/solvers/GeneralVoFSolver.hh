/** @file GeneralVoFSolver.hh
 * @brief Source-free geometric two-phase flow on certified native polyhedral cells.
 */
#pragma once
#include "geometry/GeometryStatisticsSnapshot.hh"
#include "FVM/GeneralVoF.hh"
#include "solvers/VoFSolver.hh"
namespace SimpleFluid
{
/** General native flow controls. All physical tolerances retain their original
 * extensive/face equations; no volume or inventory floor is introduced.
 */
struct GeneralVoFSolverOptions
{
    VoFSolverOptions flow;
    int maximum_momentum_iterations = 3000;
    double momentum_relative_tolerance = 1e-12;
    /** Momentum exact Input/Scratch/Result buffers and retained payloads use
     * cumulative admission, with shared opaque storage deduplicated per ledger.
     * Nested face binding conservatively reserves its own input accounting;
     * aliases already retained by the outer solve may therefore reduce usable
     * headroom. Backend/Krylov allocations and allocator overhead are excluded.
     * Projection/forest reserve cumulative payload slots, explicit buffers and
     * bounded sparse packet arenas before arithmetic/communication. Conservative
     * per-scalar slot admission can reject before all byte capacity is consumed.
     * Resource exhaustion rejects; it never changes physical tolerances.
     */
    std::size_t byte_budget = 64 * 1024 * 1024;
};
struct GeneralVoFMomentumDiagnostics
{
    double physical_relative_residual = 0;
    int implicit_iterations = 0, defect_corrections = 0;
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
 *
 * A failed step leaves the complete accepted publication and clock unchanged.
 * Source-free occupancy, reconstruction/support, donor-capacity, physical linear
 * residual and conditioning failures reject; they never normalize a cell or
 * invent a material source. References are invalidated by successful mutation.
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
     */
    void initialize(
        const std::vector<VoF::Vector>& owned_velocity = {}, const std::vector<double>& overlap_face_velocity = {});
    [[nodiscard]] VoFStepDiagnostics step(double timestep);
    [[nodiscard]] double stable_timestep() const;
    /** Conservative fixed-clock adaptation of phases and physical momentum.
     * Source keys are persistent hierarchy keys; rejected plans preserve all
     * accepted state. Geometry-dependent operators are rebuilt on the candidate.
     */
    void regrid(
        std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen = {});
    /** Ownership-only conservative native migration in current owned-key order. */
    void repartition(std::span<const int> destinations);
    [[nodiscard]] double time() const noexcept;
    [[nodiscard]] const std::shared_ptr<const VoF::GeneralVoF>& phases() const noexcept;
    [[nodiscard]] const std::shared_ptr<const MeshHandle<>>& mesh() const noexcept;
    /** Bounded geometric fitting/CSF label view, distinct from independent
     * physical gas/carrier extensive inventories in phases(). */
    [[nodiscard]] const std::vector<double>& gas_fraction() const noexcept;
    [[nodiscard]] const std::array<GeneralVoFMomentumDiagnostics, 3>& momentum_transport_diagnostics() const noexcept;
    [[nodiscard]] const std::vector<VoF::Vector>& velocity() const noexcept;
    [[nodiscard]] const std::vector<double>& face_velocity() const noexcept;
    [[nodiscard]] const std::vector<double>& pressure() const noexcept;
    [[nodiscard]] const std::vector<double>& curvature() const noexcept;
    [[nodiscard]] const VoFStepDiagnostics& diagnostics() const noexcept;

private:
    struct SIMPLEFLUID_SOLVERS_LOCAL Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid
