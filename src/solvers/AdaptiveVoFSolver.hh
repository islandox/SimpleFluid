/** @file AdaptiveVoFSolver.hh
 * @brief Distributed geometric two-phase flow on immutable adaptive Cartesian meshes.
 */
#pragma once

#include "FVM/AdaptiveVoF.hh"
#include "solvers/VoFLowMach.hh"
#include "solvers/VoFSolver.hh"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace SimpleFluid
{

/**
 * Native distributed adaptive counterpart of the fixed-grid VoFSolver.
 *
 * Cell arrays contain owned cells; face arrays contain overlap faces with
 * canonical owner-outward normal velocity in m/s. All state-changing methods,
 * timestep queries, checkpoint operations, and construction are collective.
 * Levels advance synchronously with one timestep. No global solution array
 * or finest-level background grid is constructed. Momentum uses final mixture
 * mass and the integrated phase/species mass ledger in a conservative implicit
 * upwind equation, shared by incompressible and low-Mach transport.
 *
 * Regridding replaces the immutable hierarchy and every dependent field/map
 * together after conservative remapping and projection succeed. Checkpoints
 * freeze topology until explicitly released. Rejected steps/regrids retain
 * the complete previous accepted state and all passive extensive inventories.
 * The source-free isothermal phase, gravity, viscosity and surface-tension
 * assumptions are those of VoFSolver; dispersed-bubble kinetics are absent.
 * Accessor references are invalidated by successful mutations or restore,
 * including inventory registration. Independently copied mesh shared pointers
 * retain their immutable mesh; array and field references do not retain state.
 */
class SIMPLEFLUID_SOLVERS_EXPORT AdaptiveVoFSolver
{
public:
    using Hierarchy = Meshes::CartesianAMR;
    using Handle = MeshHandle<>;
    using ScalarField = VoFSolver::ScalarField;
    using VectorField = VoFSolver::VectorField;
    using Checkpoint = std::uint64_t;

    explicit AdaptiveVoFSolver(
        std::shared_ptr<const Hierarchy> hierarchy, VoFMaterial material = {}, VoFSolverOptions options = {});
    ~AdaptiveVoFSolver();
    AdaptiveVoFSolver(AdaptiveVoFSolver&&) noexcept;
    AdaptiveVoFSolver& operator=(AdaptiveVoFSolver&&) noexcept;
    AdaptiveVoFSolver(const AdaptiveVoFSolver&) = delete;
    AdaptiveVoFSolver& operator=(const AdaptiveVoFSolver&) = delete;

    /** Reset flow state and clear registered inventories; project initial flux. */
    void initialize(const std::vector<double>& owned_fraction, const std::vector<VoF::Vector>& owned_velocity = {});
    /** Initial normal face velocities use native overlap ordering. */
    void initialize_faces(const std::vector<double>& owned_fraction, const std::vector<double>& overlap_face_velocity);
    [[nodiscard]] VoFStepDiagnostics step(double dt);
    [[nodiscard]] double stable_timestep() const;

    /** Opt into EOS/energy/phase-inventory evolution; existing constructors remain source-free. */
    void initialize_low_mach(VoF::LowMachOptions options, const VoF::LowMachInitialState& initial);
    [[nodiscard]] bool low_mach_enabled() const noexcept;
    [[nodiscard]] const VoF::LowMachState& low_mach_state() const;
    [[nodiscard]] const VoF::LowMachDiagnostics& low_mach_diagnostics() const;
    [[nodiscard]] const VoF::LowMachOptions& low_mach_options() const;

    class SIMPLEFLUID_SOLVERS_EXPORT LowMachTrial
    {
    public:
        [[nodiscard]] const VoF::LowMachState& state() const;
        [[nodiscard]] const VoF::LowMachDiagnostics& diagnostics() const;
        [[nodiscard]] const VoF::LowMachTransport& transport() const;
        [[nodiscard]] const std::vector<double>& face_velocity() const;
        [[nodiscard]] const std::vector<VoF::Vector>& velocity() const;
        [[nodiscard]] const std::vector<double>& mechanical_pressure() const;
        [[nodiscard]] const std::vector<double>& gas_fraction() const;
        [[nodiscard]] bool is_representation_change() const;
        [[nodiscard]] const VoF::RepresentationDiagnostics& representation_diagnostics() const;

    private:
        friend class AdaptiveVoFSolver;
        explicit LowMachTrial(std::shared_ptr<const void> candidate);
        std::shared_ptr<const void> candidate_;
    };

    /** Replays from accepted state; no fields, clock, or ledgers are published. */
    [[nodiscard]] LowMachTrial preview_low_mach_step(
        double dt, const VoF::LowMachSources& sources, const VoF::LowMachState* thermodynamic_guess = nullptr);
    /** Commit only a trial from the unchanged accepted state and current hierarchy. */
    void commit_low_mach_trial(const LowMachTrial& trial);

    /** Fixed-clock conservative conversion; commit with commit_low_mach_trial.
     * Carries phase pressure/volume/energy histories and defers physical relaxation.
     * Nonempty changes reject generic registered passives and unrelaxed prior
     * conversions. Regrid/reinitialization require physical relaxation first.
     * An empty parcel request is an exact fixed-clock identity trial.
     */
    [[nodiscard]] LowMachTrial preview_representation_change(const VoF::RepresentationChangeRequest& request);
    /** Accepted-state/control revision, independent of hierarchy topology, partition and geometry. */
    [[nodiscard]] std::uint64_t state_revision() const noexcept;
    [[nodiscard]] bool pending_representation_relaxation() const noexcept;

    /** Conservative adaptation; supplied stencil/motion/wake/promotion guards constrain coarsening. */
    void regrid(const std::vector<std::size_t>& refine, const std::vector<Meshes::AMRBlock>& coarsen = {},
        VoF::RefinementOptions protection = {}, VoF::RefinementHints hints = {});
    /** Apply a plan only to its exact source observation and, for solver-issued plans, unchanged
     * accepted state; recheck current interface guards before conservative adaptation. */
    void regrid(const VoF::RefinementPlan& plan);
    [[nodiscard]] VoF::RefinementPlan refinement_plan(
        VoF::RefinementOptions options = {}, VoF::RefinementHints hints = {}) const;

    /** Register an owned-cell extensive scalar; it is transported and remapped with its carrier.
     * Low-Mach mode accepts geometric carriers; legacy constant-density MixtureMass is rejected.
     * LiquidVolume denotes the unresolved carrier, including its dispersed gas.
     */
    void register_inventory(VoF::ConservedQuantity inventory);
    [[nodiscard]] const VoF::ConservedQuantity& inventory(const std::string& name) const;
    [[nodiscard]] const std::vector<VoF::ConservedQuantity>& inventories() const noexcept;

    /** One checkpoint may be active. Restore keeps it active for further replay. */
    [[nodiscard]] Checkpoint checkpoint();
    void restore(Checkpoint checkpoint);
    void release_checkpoint(Checkpoint checkpoint);
    [[nodiscard]] bool has_checkpoint() const noexcept;

    [[nodiscard]] const std::shared_ptr<const Hierarchy>& hierarchy() const noexcept;
    [[nodiscard]] const SP<const Handle>& mesh() const noexcept;
    [[nodiscard]] std::uint64_t topology_generation() const noexcept;
    [[nodiscard]] std::uint64_t partition_generation() const noexcept;
    [[nodiscard]] const VoFMaterial& material() const noexcept;
    [[nodiscard]] const VoFSolverOptions& options() const noexcept;
    [[nodiscard]] double time() const noexcept;
    [[nodiscard]] const VoFStepDiagnostics& diagnostics() const noexcept;
    [[nodiscard]] const std::vector<double>& gas_fraction() const noexcept;
    [[nodiscard]] const std::vector<VoF::Vector>& velocity() const noexcept;
    [[nodiscard]] const std::vector<double>& pressure() const noexcept;
    [[nodiscard]] const std::vector<double>& curvature() const noexcept;
    [[nodiscard]] const std::vector<double>& face_velocity() const noexcept;
    [[nodiscard]] const ScalarField& fraction_field() const noexcept;
    [[nodiscard]] const ScalarField& pressure_field() const noexcept;
    [[nodiscard]] const VectorField& velocity_field() const noexcept;

private:
    struct SIMPLEFLUID_SOLVERS_LOCAL Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace SimpleFluid
