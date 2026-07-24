/**
 * @file PressureProjectionEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Pressure projection equation for the Boussinesq solver.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/EquationValidation.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "FVM/Operators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
class FluidSolver;

namespace detail
{

/**
 * @brief Test-only accessor for pressure-projection implementation state.
 * @tparam Pack Tpetra type pack used by the equation.
 */
template<TpetraTypePack Pack>
struct PressureProjectionEquationTestAccess;

} // namespace detail

/**
 * @brief Build the default linear-solver options for pressure projection.
 * @return Options selecting the MueLu preconditioner.
 */
inline LinearSolverOptions pressure_projection_linear_solver_options()
{
    LinearSolverOptions options;
    options.preconditioner = LinearPreconditioner::MueLu;
    return options;
}

/**
 * @brief Pressure-projection solve used to correct transient velocity fields.
 *
 * The class keeps the legacy pressure reset and the pressure-correction
 * projection used by the Boussinesq solver separated from orchestration.
 * Projection unknowns are normalized internally, while returned pressure
 * corrections are physical gauge pressures in Pa.
 *
 * @tparam Pack Tpetra type pack used for matrix/vector storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class PressureProjectionEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using face_flux_field_type = FaceField<Pack>;
    using face_velocity_field_type = VectorFaceField<Pack>;
    using face_flux_workspace_type =
        FVM::PressureWeightedFaceFluxWorkspace<Pack>;
    using map_type = typename Pack::map_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using source_type = std::function<scalar_type(local_ordinal_type)>;
    /** @brief Pressure-correction solve and continuity statistics. */
    struct ProjectionResult
    {
        scalar_type pressure_correction = {}; ///< L2 norm of the Pa update.
        scalar_type continuity = {}; ///< L2 continuity residual after projection.
        LinearSolveStatistics linear_solve;
    };

    explicit PressureProjectionEquation(
        SP<const mesh_type> mesh,
        LinearSolverOptions linear_options =
            pressure_projection_linear_solver_options(),
        BoundaryConditionMap pressure_boundary_conditions = {});

    void set_linear_solver_options(LinearSolverOptions options)
    {
        d_linear_options = options;
    }

    const LinearSolverOptions& linear_solver_options() const noexcept
    {
        return d_linear_options;
    }

    void rebuild_matrix() const;

    void solve(field_type& pressure);

    ProjectionResult project(
        field_type& pressure,
        scalar_type time_step,
        scalar_type reference_density,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        velocity_field_type& velocity);

    /**
     * @brief Project velocity while accumulating a physical pressure field.
     *
     * The predictor face flux is reconstructed from @p pressure before the
     * homogeneous correction is solved.  On return, @p pressure includes the
     * correction stored in @p pressure_correction, and
     * corrected_face_fluxes() is consistent with both updated fields.
     *
     * @param[in,out] pressure Accumulated physical gauge pressure in Pa.
     * @param[out] pressure_correction Physical pressure correction in Pa.
     * @param time_step Positive time-step size.
     * @param reference_density Positive pressure-normalization density.
     * @param velocity_boundary_cache Cached velocity boundary conditions.
     * @param[in,out] velocity Pressure-corrected cell velocity.
     * @return Correction, continuity, and linear-solve statistics.
     */
    ProjectionResult project(
        field_type& pressure,
        field_type& pressure_correction,
        scalar_type time_step,
        scalar_type reference_density,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        velocity_field_type& velocity);

    ProjectionResult project(
        field_type& pressure,
        scalar_type time_step,
        scalar_type reference_density,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        velocity_field_type& velocity,
        const source_type& right_hand_source);

    /**
     * @brief Return the total-pressure-consistent flux from the latest project().
     *
     * The reference remains valid until the next project() call or equation
     * destruction.
     */
    const face_flux_field_type& corrected_face_fluxes() const noexcept
    {
        return d_cached_face_fluxes;
    }

private:
    friend struct detail::PressureProjectionEquationTestAccess<Pack>;
    friend class FluidSolver<Pack>;

    static Teuchos::RCP<const map_type> require_owned_cell_map(
        const SP<const mesh_type>& mesh);

    /**
     * @brief Project using the final face flux cached by the preceding
     *        accumulated-pressure projection as this predictor.
     *
     * This is restricted to FluidSolver's adjacent PISO correctors: pressure
     * and velocity must be unchanged since the preceding project() returned.
     */
    ProjectionResult project_reusing_cached_predictor(
        field_type& pressure,
        field_type& pressure_correction,
        scalar_type time_step,
        scalar_type reference_density,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        velocity_field_type& velocity);

    ProjectionResult project_impl(
        field_type& pressure_correction,
        scalar_type time_step,
        scalar_type reference_density,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        velocity_field_type& velocity,
        const source_type& right_hand_source,
        field_type* accumulated_pressure,
        bool reuse_cached_predictor_flux = false);

    SP<const mesh_type> d_mesh;
    LinearSolverOptions d_linear_options;
    BoundaryConditionMap d_pressure_boundary_conditions;
    BoundaryConditionMap d_pressure_correction_boundary_conditions;
    mutable std::optional<typename Pack::global_ordinal_type>
        d_pressure_gauge_gid;
    mutable face_flux_field_type d_cached_face_fluxes;
    mutable face_flux_workspace_type d_face_flux_workspace;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_pressure_matrix;
    mutable Teuchos::RCP<typename Pack::vector_type> d_cached_rhs;
    bool d_cached_predictor_flux_valid = false;
    std::size_t d_cached_predictor_flux_reuse_count = 0;
    BelosLinearSolver<Pack> d_linear_solver;
};

} // namespace SimpleFluid
