/** @file LowMachPressureEquation.hh
 * @brief Native variable-density low-Mach pressure correction on orthogonal mapped meshes.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "fields/MeshFieldTraits.hh"
#include "geometry/MeshHandle.hh"
#include "solvers/BelosLinearSolver.hh"
#include <memory>
#include <span>

namespace SimpleFluid::Reacting
{
class SIMPLEFLUID_SOLVERS_EXPORT LowMachPressureEquation
{
public:
    using Pack = DefaultTpetraTypes;
    using Mesh = MeshHandle<Pack>;
    using Traits = MeshFieldTraits<Pack, Mesh>;
    using ScalarField = Traits::scalar_cell_type;
    using VectorField = Traits::vector_cell_type;
    using FaceField = Traits::scalar_face_type;
    struct Result
    {
        LinearSolveStatistics linear;
        double maximum_continuity_residual =
            0; ///< Cell-integrated m3/s; material flux for project_material_with_eos().
        double global_compatibility_residual = 0;
        double maximum_linearized_continuity_residual = 0; ///< Includes chi*delta_p/dt, integrated m3/s.
        double continuity_scale = 0; ///< Original physical residual normalization, before EOS linearization.
        double thermodynamic_pressure_increment = 0; ///< Removed uniform mechanical gauge; add this to p0.
    };
    explicit LowMachPressureEquation(SP<const Mesh> mesh);
    ~LowMachPressureEquation();
    LowMachPressureEquation(const LowMachPressureEquation&) = delete;
    LowMachPressureEquation& operator=(const LowMachPressureEquation&) = delete;
    /** Fixed geometry and closed physical boundaries; predictor must carry exactly zero physical boundary flux.
     * Geometry changes after construction reject before using the retained gradient cache.
     * All faces use dt*harmonic(1/rho)*A/d with canonical owner signs. The nullspace is gauged globally.
     * Inputs must share this native mesh. All ranks participate, including empty owners.
     */
    Result project(const ScalarField& density, double time_step, std::span<const double> integrated_volume_target,
        const FaceField& predictor_flux, FaceField& corrected_flux, ScalarField& mechanical_pressure,
        VectorField& velocity, const LinearSolverOptions& linear = {}, double continuity_tolerance = 1e-10);

    /** Newton pressure step with actual chi=-dV/dp at fixed transported/source packets and Vadv, m3/Pa.
     * Solves (A+diag(chi/dt))*delta_p = target-div(predictor). Each chi must be finite and nonnegative.
     * The linearized continuity residual is enforced here; the caller must converge the returned physical
     * continuity residual and nonlinear EOS occupancy before acceptance. No physical tolerance is relaxed.
     * With positive total chi the compatible summed equation fixes the chi-weighted uniform pressure mode.
     * Re-gauging mechanical pressure preserves absolute p only when the caller adds the returned increment to p0.
     * Zero total chi recovers project() exactly. All ranks call, including owners with empty chi spans.
     */
    Result project_with_eos(const ScalarField& density, double time_step,
        std::span<const double> integrated_volume_target, std::span<const double> volume_susceptibility,
        const FaceField& predictor_flux, FaceField& corrected_flux, ScalarField& mechanical_pressure,
        VectorField& velocity, const LinearSolverOptions& linear = {}, double continuity_tolerance = 1e-10);

    /** Project phase-weighted absolute material flux while returning the absolute carrier flux.
     * Frozen theta=sum(upwind material fractions) and S=sum(fraction*phase slip flux) give
     * Phi_material=theta*phi_carrier+S. Mobility is theta*dt*harmonic(1/rho)*A/d.
     * The cell-integrated target and returned continuity residuals refer to Phi_material, m3/s. theta is
     * finite/nonnegative; it is never normalized to one. Exact zero theta requires S=0 and introduces no
     * division/floor: the carrier still updates by its unweighted momentum pressure gradient. If any interior theta is
     * zero, every cell must have chi>0 (empty owners are neutral). Other possibly disconnected pressure graphs reject
     * explicitly; this sufficient anchoring gate adds no diagonal floors or invented material mobility.
     * EOS linearization, p0 gauge transfer and nonlinear acceptance obligations match project_with_eos().
     */
    Result project_material_with_eos(const ScalarField& density, double time_step,
        std::span<const double> integrated_material_volume_target, std::span<const double> volume_susceptibility,
        const FaceField& material_carrier_fraction, const FaceField& slip_material_flux,
        const FaceField& predictor_carrier_flux, FaceField& corrected_carrier_flux, ScalarField& mechanical_pressure,
        VectorField& velocity, const LinearSolverOptions& linear = {}, double continuity_tolerance = 1e-10);

private:
    SIMPLEFLUID_SOLVERS_LOCAL Result project_impl(const ScalarField& density, double time_step,
        std::span<const double> integrated_volume_target, std::span<const double> volume_susceptibility,
        bool linearize_eos, const FaceField* material_carrier_fraction, const FaceField* slip_material_flux,
        const FaceField& predictor_flux, FaceField& corrected_flux, ScalarField& mechanical_pressure,
        VectorField& velocity, const LinearSolverOptions& linear, double continuity_tolerance);
    struct SIMPLEFLUID_SOLVERS_LOCAL Implementation;
    std::unique_ptr<Implementation> d;
};
} // namespace SimpleFluid::Reacting
