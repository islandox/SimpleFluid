/**
 * @file PressureProjectionEquation.hh
 * @brief Pressure projection equation for the Boussinesq solver.
 */
#pragma once

#include "equations/EquationValidation.hh"
#include "fields/CellField.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <stdexcept>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Pressure-projection solve used to keep the transient flow update bounded.
 *
 * The current implementation preserves the solver's identity projection
 * behavior while separating the equation from solver orchestration.
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
};

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
 * @brief Apply the current identity pressure projection.
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

} // namespace SimpleFluid
