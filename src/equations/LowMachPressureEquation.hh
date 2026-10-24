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
        double maximum_continuity_residual = 0; ///< Integrated m3/s.
        double global_compatibility_residual = 0;
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

private:
    struct SIMPLEFLUID_SOLVERS_LOCAL Implementation;
    std::unique_ptr<Implementation> d;
};
} // namespace SimpleFluid::Reacting
