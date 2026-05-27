/**
 * @file BelosLinearSolver.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Small Belos/Tpetra linear-solver wrapper.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "dataclass/TpetraTypes.hh"

#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosTypes.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace SimpleFluid
{

/**
 * @brief Configuration options for the Belos linear solver.
 */
struct LinearSolverOptions
{
    int max_iterations = 200;
    real_t tolerance = 1.0e-10;
    int verbosity = Belos::Errors + Belos::Warnings;
};

/**
 * @brief Solve a linear system using Belos GMRES with the given operator.
 *
 * @tparam Pack Tpetra type pack.
 * @param matrix Tpetra operator representing the system matrix.
 * @param rhs Right-hand side vector.
 * @param solution On input, initial guess; on output, the solution.
 * @param options Solver convergence and verbosity options.
 * @return true if the solver converged, false otherwise.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
bool solve_linear_system(const Teuchos::RCP<const typename Pack::operator_type>& matrix,
                         const typename Pack::vector_type& rhs,
                         typename Pack::vector_type& solution,
                         const LinearSolverOptions& options = {})
{
    using scalar_type = typename Pack::scalar_type;
    using multi_vector_type = typename Pack::multi_vector_type;
    using operator_type = typename Pack::operator_type;
    using problem_type = Belos::LinearProblem<scalar_type, multi_vector_type, operator_type>;
    using solver_type = Belos::PseudoBlockGmresSolMgr<scalar_type,
                                                      multi_vector_type,
                                                      operator_type>;

    auto x = Teuchos::rcp_implicit_cast<multi_vector_type>(
        Teuchos::rcpFromRef(solution));
    auto b = Teuchos::rcp_implicit_cast<const multi_vector_type>(
        Teuchos::rcpFromRef(rhs));

    auto problem = Teuchos::rcp(new problem_type(matrix, x, b));
    if (!problem->setProblem())
    {
        return false;
    }

    auto params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set("Maximum Iterations", options.max_iterations);
    params->set("Convergence Tolerance", options.tolerance);
    params->set("Verbosity", options.verbosity);

    solver_type solver(problem, params);
    return solver.solve() == Belos::Converged;
}

/**
 * @brief Solve a linear system using Belos GMRES with a Tpetra CrsMatrix.
 *
 * This overload wraps the matrix in an operator and delegates to the
 * operator-based solve.
 *
 * @tparam Pack Tpetra type pack.
 * @param matrix Tpetra CRS matrix.
 * @param rhs Right-hand side vector.
 * @param solution On input, initial guess; on output, the solution.
 * @param options Solver convergence and verbosity options.
 * @return true if the solver converged, false otherwise.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
bool solve_linear_system(const Teuchos::RCP<const typename Pack::matrix_type>& matrix,
                         const typename Pack::vector_type& rhs,
                         typename Pack::vector_type& solution,
                         const LinearSolverOptions& options = {})
{
    auto op = Teuchos::rcp_implicit_cast<const typename Pack::operator_type>(matrix);
    return solve_linear_system<Pack>(op, rhs, solution, options);
}

} // namespace SimpleFluid
