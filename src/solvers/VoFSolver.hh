/**
 * @file VoFSolver.hh
 * @brief Fixed-grid, geometric volume-of-fluid solver for two incompressible phases.
 */
#pragma once

#include "FVM/GeometricVoF.hh"
#include "SimpleFluidExport.hh"
#include "fields/FieldStored.hh"

#include <memory>
#include <vector>

namespace SimpleFluid
{

/** SI material properties; f=1 denotes gas and f=0 denotes liquid. */
struct VoFMaterial
{
    double liquid_density = 1000.0;
    double gas_density = 0.1;
    double liquid_viscosity = 1.0e-3;
    double gas_viscosity = 1.0e-5;
    double surface_tension = 0.07;
};

/** Controls for the serial uniform Cartesian baseline. */
struct VoFSolverOptions
{
    VoF::Vector gravity{0.0, 0.0, 0.0};
    double max_courant = 0.35;
    double capillary_safety = 0.5;
    double viscous_safety = 0.8;
    double pressure_relative_tolerance = 1.0e-11; ///< Dimensionless physical pressure-residual limit.
    double pressure_absolute_tolerance = 1.0e-13; ///< Absolute max|div(u)|/dt limit [1/s2].
    double divergence_tolerance = 1.0e-8;         ///< Absolute face-flux divergence limit [1/s].
    int max_pressure_iterations = 3000;           ///< Shared budget across residual corrections.
};

/** Accepted-state budgets; masses are kg, momentum kg m/s, energy J. */
struct VoFStepDiagnostics
{
    double time = 0.0;
    double gas_volume = 0.0;
    double gas_mass = 0.0;
    double liquid_mass = 0.0;
    VoF::Vector momentum{0.0, 0.0, 0.0};
    double kinetic_energy = 0.0;
    double maximum_velocity = 0.0;
    double maximum_divergence = 0.0; ///< max|div(u_face)| [1/s].
    double pressure_residual = 0.0;  ///< max|div(u_face)|/dt [1/s2], not a pressure in Pa.
    int pressure_iterations = 0;
};

/**
 * @brief Geometric VoF with conservative mass/momentum transport and projection.
 *
 * Source-free, isothermal, constant-property incompressible liquid and gas.
 * The fixed uniform grid supports periodic boundaries and impermeable slip
 * walls. All ranks reject a multi-rank communicator before mesh construction;
 * this baseline does not silently replicate a distributed physical solution.
 *
 * Native MeshHandle/FieldStored snapshots expose accepted output. Geometric
 * face velocities supply the authoritative divergence-free transport flux.
 * Cell momentum uses conservative implicit upwind transport with final mixture
 * mass storage and exactly the geometric gas/liquid face masses. This preserves
 * passive velocity bounds in the force-free divergence-free limit without a
 * stale accepted-old donor state; it adds first-order implicit diffusion.
 * Pressure and surface tension share one face gradient and mobility stencil.
 * Native fields and read-only array views are published together only after
 * all final physical residual and finite-value gates pass. Exceptions leave the accepted
 * state, time, diagnostics, and face flux unchanged.
 *
 * Accessor references remain valid until the next successful initialize/step.
 * No thermal/EOS coupling, phase change, turbulence, moving mesh, contact angle,
 * refinement, or dispersed-bubble exchange is provided by this first stage.
 */
class SIMPLEFLUID_SOLVERS_EXPORT VoFSolver
{
public:
    using ScalarField = FieldStored<ScalarCellFieldDescriptor<>>;
    using VectorField = FieldStored<VectorCellFieldDescriptor<>>;
    using Handle = MeshHandle<>;

    explicit VoFSolver(VoF::UniformGrid grid, VoFMaterial material = {}, VoFSolverOptions options = {});
    ~VoFSolver();
    VoFSolver(VoFSolver&&) noexcept;
    VoFSolver& operator=(VoFSolver&&) noexcept;
    VoFSolver(const VoFSolver&) = delete;
    VoFSolver& operator=(const VoFSolver&) = delete;

    /** Set fractions and cell velocities; project initial face flux without capillarity. */
    void initialize(const std::vector<double>& gas_fraction, const std::vector<VoF::Vector>& cell_velocity = {});
    /** Set fractions and face velocities; project initial flux without capillarity. */
    void initialize(const std::vector<double>& gas_fraction, const VoF::FaceField& face_velocity);

    /** Advance by a positive finite dt no larger than stable_timestep(). */
    [[nodiscard]] VoFStepDiagnostics step(double dt);
    /** Explicit advection, capillary, and viscous limit; infinity for a motionless inviscid single phase. */
    [[nodiscard]] double stable_timestep() const;
    [[nodiscard]] double time() const noexcept;
    [[nodiscard]] const VoFStepDiagnostics& diagnostics() const noexcept;
    [[nodiscard]] const VoF::UniformGrid& grid() const noexcept;
    [[nodiscard]] const VoFMaterial& material() const noexcept;
    [[nodiscard]] const VoFSolverOptions& options() const noexcept;
    [[nodiscard]] const SP<const Handle>& mesh() const noexcept;
    [[nodiscard]] const std::vector<double>& gas_fraction() const noexcept;
    [[nodiscard]] const std::vector<VoF::Vector>& velocity() const noexcept;
    [[nodiscard]] const std::vector<double>& pressure() const noexcept;
    [[nodiscard]] const std::vector<double>& curvature() const noexcept;
    [[nodiscard]] const VoF::FaceField& face_velocity() const noexcept;
    [[nodiscard]] const ScalarField& fraction_field() const noexcept;
    [[nodiscard]] const ScalarField& pressure_field() const noexcept;
    [[nodiscard]] const VectorField& velocity_field() const noexcept;

private:
    struct SIMPLEFLUID_SOLVERS_LOCAL Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace SimpleFluid
