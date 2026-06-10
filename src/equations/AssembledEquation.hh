/**
 * @file AssembledEquation.hh
 * @brief Solvable matrix/RHS representation of a generic equation.
 */

#pragma once

#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <stdexcept>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Linear system assembled for one stored field.
 *
 * Owns the matrix and right-hand side while sharing the solution field.
 * solve() refreshes overlap storage after updating owned solution values.
 */
template<class StoredField, TpetraTypePack Pack = DefaultTpetraTypes>
class AssembledEquation
{
public:
    using field_type = StoredField;
    using matrix_type = typename Pack::matrix_type;
    using rhs_type = typename field_type::storage_type;

    /**
     * @brief Construct a solvable equation from assembled linear algebra.
     * @throws std::invalid_argument If any required object is null.
     */
    AssembledEquation(SP<field_type> solution,
                      Teuchos::RCP<matrix_type> matrix,
                      Teuchos::RCP<rhs_type> rhs,
                      LinearSolverOptions options = {})
        : d_solution(std::move(solution)),
          d_matrix(std::move(matrix)),
          d_rhs(std::move(rhs)),
          d_options(options)
    {
        if (!d_solution || d_matrix.is_null() || d_rhs.is_null())
        {
            throw std::invalid_argument(
                "AssembledEquation requires a solution, matrix, and RHS.");
        }
    }

    const std::string& name() const noexcept
    {
        return d_solution->name();
    }

    field_type& solution() noexcept { return *d_solution; }
    const field_type& solution() const noexcept { return *d_solution; }

    Teuchos::RCP<matrix_type> matrix() noexcept { return d_matrix; }
    Teuchos::RCP<const matrix_type> matrix() const noexcept
    {
        return d_matrix;
    }

    rhs_type& rhs() noexcept { return *d_rhs; }
    const rhs_type& rhs() const noexcept { return *d_rhs; }

    void set_linear_solver_options(LinearSolverOptions options)
    {
        d_options = options;
    }

    /**
     * @brief Solve the linear system into the solution field.
     * @return Whether the configured linear solver converged.
     */
    bool solve()
    {
        Teuchos::RCP<const matrix_type> matrix = d_matrix;
        const auto converged = d_solver.solve(
            matrix,
            *d_rhs,
            d_solution->owned_data(),
            d_options);
        d_solution->sync_ghosts();
        return converged;
    }

private:
    SP<field_type> d_solution;
    Teuchos::RCP<matrix_type> d_matrix;
    Teuchos::RCP<rhs_type> d_rhs;
    LinearSolverOptions d_options;
    BelosLinearSolver<Pack> d_solver;
};

} // namespace SimpleFluid
