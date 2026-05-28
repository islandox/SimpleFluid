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

#include "equations/BoundaryConditions.hh"
#include "equations/EquationValidation.hh"
#include "fields/CellField.hh"
#include "fields/VectorCellField.hh"
#include "operators/FvmOperators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <stdexcept>
#include <utility>

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

    void solve(field_type& pressure);

    void project(field_type& pressure,
                 scalar_type time_step,
                 const BoundaryConditionSet& boundary_conditions,
                 velocity_field_type& velocity);

private:
    static Teuchos::RCP<const map_type> require_owned_cell_map(
        const SP<const mesh_type>& mesh);

    SP<const mesh_type> d_mesh;
    LinearSolverOptions d_linear_options;
};

/**
 * @brief Construct a pressure projection equation with mesh and solver options.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the assembled mesh.
 * @param linear_options Belos linear solver configuration.
 * @throws std::invalid_argument if the mesh is null.
 * @throws std::runtime_error if the mesh has no owned-cell map.
 */
template<TpetraTypePack Pack>
PressureProjectionEquation<Pack>::PressureProjectionEquation(
    SP<const mesh_type> mesh,
    LinearSolverOptions linear_options)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "PressureProjectionEquation")),
      d_linear_options(linear_options)
{
    require_owned_cell_map(d_mesh);
}

/**
 * @brief Retrieve and validate the owned-cell map from the mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the mesh.
 * @return RCP to the owned-cell Tpetra map.
 * @throws std::runtime_error if the mesh has no owned-cell map.
 */
template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::require_owned_cell_map(
    const SP<const mesh_type>& mesh) -> Teuchos::RCP<const map_type>
{
    auto map = mesh->owned_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error(
            "PressureProjectionEquation requires an assembled mesh with an owned-cell map.");
    }

    return map;
}

/**
 * @brief Apply the legacy identity pressure reset.
 *
 * @tparam Pack Tpetra type pack.
 * @param pressure Pressure field updated by the projection solve.
 */
template<TpetraTypePack Pack>
void PressureProjectionEquation<Pack>::solve(field_type& pressure)
{
    EquationValidation::require_mesh_match(*d_mesh, pressure,
                                           "PressureProjectionEquation");

    pressure.owned_data().putScalar(0.0);
    pressure.sync_ghosts();
}

/**
 * @brief Solve pressure correction and project all velocity components.
 *
 * @tparam Pack Tpetra type pack.
 * @param pressure Pressure-correction field.
 * @param time_step Time-step size.
 * @param boundary_conditions Velocity boundary conditions for face fluxes.
 * @param velocity Velocity predictor, overwritten with projected velocity.
 */
template<TpetraTypePack Pack>
void PressureProjectionEquation<Pack>::project(
    field_type& pressure,
    scalar_type time_step,
    const BoundaryConditionSet& boundary_conditions,
    velocity_field_type& velocity)
{
    EquationValidation::require_mesh_match(*d_mesh, pressure,
                                           "PressureProjectionEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity,
                                           "PressureProjectionEquation");
    if (time_step <= 0.0)
    {
        throw std::invalid_argument("PressureProjectionEquation requires a positive time step.");
    }
    if (d_mesh->num_owned_cells() == 0)
    {
        return;
    }

    const auto face_fluxes = FvmOperators::face_fluxes(
        *d_mesh, velocity, &boundary_conditions);
    const auto gauge_gid = d_mesh->owned_cell_global_ids().front();
    auto matrix = FvmOperators::pressure_poisson_matrix<Pack>(*d_mesh, gauge_gid);
    typename Pack::vector_type rhs(d_mesh->owned_cell_map(), true);

    for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<typename Pack::local_ordinal_type>(owned);
        const auto row_gid = d_mesh->cell_global_id(cell_lid);
        const auto rhs_value = row_gid == gauge_gid
                             ? scalar_type{}
                             : -FvmOperators::cell_flux_balance<Pack>(
                                   *d_mesh, face_fluxes, cell_lid) / time_step;
        rhs.replaceLocalValue(cell_lid, rhs_value);
    }

    pressure.owned_data().putScalar(0.0);
    Teuchos::RCP<const typename Pack::matrix_type> const_matrix = matrix;
    if (!solve_linear_system<Pack>(const_matrix, rhs, pressure.owned_data(),
                                   d_linear_options))
    {
        throw std::runtime_error("PressureProjectionEquation projection solve did not converge.");
    }
    pressure.sync_ghosts();

    const auto gradients = FvmOperators::cell_gradient(pressure);
    for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<typename Pack::local_ordinal_type>(owned);
        const auto& gradient = gradients[owned];
        const auto corrected = velocity.value(cell_lid) - gradient * time_step;
        velocity.set_owned_value(cell_lid, corrected);
    }

    velocity.sync_ghosts();
}

} // namespace SimpleFluid
