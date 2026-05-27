/**
 * @file PressureProjectionEquation.hh
 * @brief Pressure projection equation for the Boussinesq solver.
 */
#pragma once

#include "fields/CellField.hh"
#include "operators/FvmOperators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <stdexcept>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Pressure-projection solve used to keep the transient flow update bounded.
 *
 * This class owns the matrix and right-hand-side cache for the projection
 * equation. The current implementation preserves the solver's identity
 * projection behavior while separating the equation from solver orchestration.
 *
 * @tparam Pack Tpetra type pack used for matrix/vector storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class PressureProjectionEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using map_type = typename Pack::map_type;
    using matrix_type = typename Pack::matrix_type;
    using vector_type = typename Pack::vector_type;

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

private:
    static Teuchos::RCP<const map_type> require_owned_cell_map(
        const SP<const mesh_type>& mesh);

    SP<const mesh_type> d_mesh;
    LinearSolverOptions d_linear_options;
    Teuchos::RCP<matrix_type> d_pressure_matrix;
    vector_type d_pressure_rhs;
};

template<TpetraTypePack Pack>
PressureProjectionEquation<Pack>::PressureProjectionEquation(
    SP<const mesh_type> mesh,
    LinearSolverOptions linear_options)
    : d_mesh(std::move(mesh)),
      d_linear_options(linear_options),
      d_pressure_rhs(require_owned_cell_map(d_mesh), true)
{
}

template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::require_owned_cell_map(
    const SP<const mesh_type>& mesh) -> Teuchos::RCP<const map_type>
{
    if (!mesh)
    {
        throw std::invalid_argument("PressureProjectionEquation requires a non-null mesh.");
    }

    auto map = mesh->owned_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error(
            "PressureProjectionEquation requires an assembled mesh with an owned-cell map.");
    }

    return map;
}

/**
 * @brief Solve the pressure projection system.
 *
 * @tparam Pack Tpetra type pack.
 * @param pressure Pressure field updated by the projection solve.
 * @throws std::runtime_error if the linear solver does not converge.
 */
template<TpetraTypePack Pack>
void PressureProjectionEquation<Pack>::solve(field_type& pressure)
{
    if (&pressure.mesh() != d_mesh.get())
    {
        throw std::invalid_argument("PressureProjectionEquation field mesh mismatch.");
    }

    if (d_pressure_matrix == Teuchos::null)
    {
        d_pressure_matrix =
            FvmOperators::identity_matrix<Pack>(d_mesh->owned_cell_map());
    }

    d_pressure_rhs.putScalar(0.0);
    pressure.owned_data().putScalar(0.0);

    auto pressure_operator =
        Teuchos::rcp_implicit_cast<const typename Pack::operator_type>(
            d_pressure_matrix);
    const bool converged =
        solve_linear_system<Pack>(pressure_operator, d_pressure_rhs,
                                  pressure.owned_data(), d_linear_options);
    if (!converged)
    {
        throw std::runtime_error("Pressure projection equation did not converge.");
    }

    pressure.sync_ghosts();
}

} // namespace SimpleFluid
