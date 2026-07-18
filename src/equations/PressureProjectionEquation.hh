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

#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{

namespace detail
{

template<TpetraTypePack Pack>
struct PressureProjectionEquationTestAccess;

} // namespace detail

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
    struct ProjectionResult
    {
        scalar_type pressure_correction = {}; ///< L2 norm of the Pa update.
        scalar_type continuity = {};
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

    ProjectionResult project(
        field_type& pressure,
        scalar_type time_step,
        scalar_type reference_density,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        velocity_field_type& velocity,
        const source_type& right_hand_source);

private:
    friend struct detail::PressureProjectionEquationTestAccess<Pack>;

    static Teuchos::RCP<const map_type> require_owned_cell_map(
        const SP<const mesh_type>& mesh);

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
    BelosLinearSolver<Pack> d_linear_solver;
};

} // namespace SimpleFluid
