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
#include "FVM/FvmOperators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Pressure-projection solve used to correct transient velocity fields.
 *
 * The class keeps the legacy pressure reset and the pressure-correction
 * projection used by the Boussinesq solver separated from orchestration.
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
    using map_type = typename Pack::map_type;
    using scalar_type = typename Pack::scalar_type;

    explicit PressureProjectionEquation(
        SP<const mesh_type> mesh,
        LinearSolverOptions linear_options = {});

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

    void project(field_type& pressure,
                 scalar_type time_step,
                 const FvmOperators::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
                 velocity_field_type& velocity);

private:
    static Teuchos::RCP<const map_type> require_owned_cell_map(
        const SP<const mesh_type>& mesh);

    SP<const mesh_type> d_mesh;
    LinearSolverOptions d_linear_options;
    mutable face_velocity_field_type d_cached_face_velocity;
    mutable face_flux_field_type d_cached_face_fluxes;
    mutable std::vector<typename mesh_type::Vec3> d_cached_gradients;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_pressure_matrix;
    mutable Teuchos::RCP<typename Pack::vector_type> d_cached_rhs;
};

} // namespace SimpleFluid
