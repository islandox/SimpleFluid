/** @file CoupledNonlinearProblem.hh
 * @brief Immutable-step native residual and analytic Newton action.
 */
#pragma once

#include "SimpleFluidExport.hh"
#include "FVM/BoundaryCache.hh"
#include "dataclass/TpetraTypes.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/BoussinesqModel.hh"
#include "equations/TimeStepperOptions.hh"
#include "equations/VolumeContinuityTarget.hh"
#include "fields/MeshFieldTraits.hh"
#include "solvers/NonlinearSolverOptions.hh"
#include "solvers/NoxNonlinearSolver.hh"

#include <memory>

namespace SimpleFluid
{
struct CoupledNonlinearProblemStatistics
{
    size_t residual_evaluations = 0;
    size_t linearizations = 0;
};

/**
 * Fixed orthogonal-mesh backward-Euler momentum with constant viscosity or
 * frozen physical Boussinesq material/source coefficients, and a coupled
 * pressure problem. Physical boundaries prescribe velocity (including
 * NoSlip) or use axis-aligned Slip, with Neumann pressure; mesh periodic
 * interfaces are supported.
 * All histories and boundary values are copied at construction. Callbacks
 * own the private problem context and never write accepted solver fields.
 * Instances and retained Jacobian actions are sequential-use objects.
 */
class SIMPLEFLUID_SOLVERS_EXPORT CoupledNonlinearProblem
{
public:
    using pack_type = DefaultTpetraTypes;
    using mesh_type = MeshHandle<pack_type>;
    using field_traits = MeshFieldTraits<pack_type, mesh_type>;
    using field_type = field_traits::scalar_cell_type;
    using velocity_field_type = field_traits::vector_cell_type;
    using face_flux_field_type = field_traits::scalar_face_type;
    using vector_type = pack_type::vector_type;
    using continuity_target_type = VolumeContinuityTarget<pack_type, mesh_type>;
    using material_type = MaterialPropertyFields<pack_type, mesh_type>;
    using boundary_cache_type = FVM::FieldStoredBoundaryCache<pack_type, mesh_type>;

    /** Inputs are borrowed only during construction, then deeply copied.
     * Physical diffusion, density/thermal buoyancy, turbulent pressure and
     * accepted-velocity transpose stress retain the native split treatment.
     */
    struct FrozenBoussinesqInput
    {
        const field_type& temperature;
        const material_type& material;
        bool density_feedback_enabled = false;
        const field_type* effective_dynamic_viscosity = nullptr;
        const velocity_field_type* turbulent_kinetic_energy_gradient = nullptr;
        const boundary_cache_type* boundary_dynamic_viscosity = nullptr;
    };

    CoupledNonlinearProblem(SP<const mesh_type> mesh, const velocity_field_type& accepted_velocity,
        const field_type& physical_pressure, const BoundaryConditionSet& boundaries,
        const TimeStepperOptions& time_options, const NonlinearSolverOptions& nonlinear_options,
        double reference_density = 1.0, const continuity_target_type* continuity_target = nullptr);
    CoupledNonlinearProblem(SP<const mesh_type> mesh, const velocity_field_type& accepted_velocity,
        const field_type& physical_pressure, const BoundaryConditionSet& boundaries,
        const TimeStepperOptions& time_options, const NonlinearSolverOptions& nonlinear_options,
        double reference_density, const continuity_target_type* continuity_target,
        const FrozenBoussinesqInput* boussinesq);
    ~CoupledNonlinearProblem();

    NonlinearCallbacks callbacks();
    /**
     * Pack accepted velocity and the private all-Neumann pressure initial
     * guess q_c = (p_c - p_g) / reference_density. The gauge is the native
     * coupled system's minimum global cell ID, wherever it is owned.
     * Accepted pressure remains unchanged; residual callbacks accept arbitrary
     * q and continue to enforce their explicit q_g = 0 algebraic row.
     */
    Teuchos::RCP<vector_type> pack_initial() const;
    /** Physical all-cell continuity from the last successful residual or commit. */
    VolumeContinuityResiduals<double> continuity() const;
    CoupledNonlinearProblemStatistics statistics() const;

    /** Publish a finite candidate after caller convergence/continuity checks. */
    void commit(const vector_type& state, velocity_field_type& velocity, field_type& physical_pressure,
        face_flux_field_type& flux);

private:
    struct Impl;
    std::shared_ptr<Impl> d_impl;
};
} // namespace SimpleFluid
