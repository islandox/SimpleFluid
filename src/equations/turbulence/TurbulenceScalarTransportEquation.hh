/**
 * @file TurbulenceScalarTransportEquation.hh
 * @brief Positive scalar transport used by two-equation turbulence models.
 */

#pragma once

#include "FVM/TransportSystem.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/EquationValidation.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "solvers/BelosLinearSolver.hh"

#include <functional>

namespace SimpleFluid
{

/**
 * @brief Semi-implicit transport equation for a positive turbulence scalar.
 *
 * The equation advances a density-divided scalar with unit storage and
 * advection weights, variable cell diffusivity, an explicit source, and a
 * linearized implicit sink. The accepted field is replaced only after a
 * converged candidate has been validated and floored.
 *
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template <TpetraTypePack Pack = DefaultTpetraTypes> class TurbulenceScalarTransportEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using face_field_type = FaceField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_provider_type = std::function<scalar_type(local_ordinal_type)>;

    /**
     * @brief Construct an equation on @p mesh with scalar boundary data.
     *
     * Missing boundary names use homogeneous Neumann conditions.
     */
    explicit TurbulenceScalarTransportEquation(SP<const mesh_type> mesh,
                                               BoundaryConditionMap boundary_conditions = {});

    /**
     * @brief Advance one accepted scalar state through a transport solve.
     *
     * The explicit source and implicit sink have units of scalar/time and
     * 1/time, respectively. Both must be finite and non-negative in every
     * owned cell. The solved candidate is clamped to @p positive_floor before
     * it atomically replaces @p state.
     *
     * @return Linear-solver convergence statistics.
     * @throws std::invalid_argument for invalid inputs.
     * @throws std::runtime_error if the solve fails or returns non-finite data.
     */
    LinearSolveStatistics
    advance(const field_type& old_state, const face_field_type& face_fluxes, scalar_type time_step,
            const field_type& effective_diffusivity, field_type& state,
            const scalar_provider_type& explicit_source, const scalar_provider_type& implicit_sink,
            scalar_type positive_floor,
            FVM::NonOrthogonalTreatment treatment = FVM::NonOrthogonalTreatment::Explicit,
            const LinearSolverOptions& linear_options = {}) const;

    /** @brief Return the configured scalar boundary conditions. */
    const BoundaryConditionMap& boundary_conditions() const noexcept
    {
        return d_boundary_conditions;
    }

private:
    SP<const mesh_type> d_mesh;
    BoundaryConditionMap d_boundary_conditions;
    field_type d_unit_weight;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_transport_matrix;
    mutable bool d_cached_graph_supports_non_orthogonal_correction = false;
    mutable BelosLinearSolver<Pack> d_linear_solver;
};

} // namespace SimpleFluid
