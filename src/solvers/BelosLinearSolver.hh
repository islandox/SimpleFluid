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
 * @brief Reusable Belos GMRES solver for systems with stable maps.
 *
 * The solver manager and its Krylov workspace are retained between solves.
 * A new manager is created only when the operator domain or range map changes.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class BelosLinearSolver
{
public:
    using scalar_type = typename Pack::scalar_type;
    using multi_vector_type = typename Pack::multi_vector_type;
    using vector_type = typename Pack::vector_type;
    using operator_type = typename Pack::operator_type;
    using problem_type =
        Belos::LinearProblem<scalar_type, multi_vector_type, operator_type>;
    using solver_type =
        Belos::PseudoBlockGmresSolMgr<
            scalar_type, multi_vector_type, operator_type>;

    bool solve(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options = {})
    {
        auto x = Teuchos::rcpFromRef(solution);
        auto b = Teuchos::rcpFromRef(rhs);

        if (!has_compatible_maps(matrix))
        {
            d_problem = Teuchos::rcp(
                new problem_type(matrix, x, b));
            d_parameters = Teuchos::rcp(new Teuchos::ParameterList());
            configure(options);
            if (!d_problem->setProblem())
            {
                return false;
            }
            d_solver = Teuchos::rcp(
                new solver_type(d_problem, d_parameters));
        }
        else
        {
            d_problem->setOperator(matrix);
            d_problem->setLHS(x);
            d_problem->setRHS(b);
            configure(options);
            d_solver->setParameters(d_parameters);
            if (!d_problem->setProblem())
            {
                return false;
            }
            d_solver->reset(Belos::Problem);
        }

        return d_solver->solve() == Belos::Converged;
    }

    bool solve(
        const Teuchos::RCP<const operator_type>& matrix,
        const vector_type& rhs,
        vector_type& solution,
        const LinearSolverOptions& options = {})
    {
        auto x = Teuchos::rcp_implicit_cast<multi_vector_type>(
            Teuchos::rcpFromRef(solution));
        auto b = Teuchos::rcp_implicit_cast<const multi_vector_type>(
            Teuchos::rcpFromRef(rhs));
        return solve(matrix, *b, *x, options);
    }

private:
    bool has_compatible_maps(
        const Teuchos::RCP<const operator_type>& matrix) const
    {
        if (matrix.is_null() || d_problem.is_null()
            || d_problem->getOperator().is_null()
            || d_solver.is_null())
        {
            return false;
        }

        const auto previous = d_problem->getOperator();
        return previous->getDomainMap()->isSameAs(*matrix->getDomainMap())
            && previous->getRangeMap()->isSameAs(*matrix->getRangeMap());
    }

    void configure(const LinearSolverOptions& options)
    {
        d_parameters->set(
            "Maximum Iterations", options.max_iterations);
        d_parameters->set(
            "Convergence Tolerance", options.tolerance);
        d_parameters->set("Verbosity", options.verbosity);
    }

    Teuchos::RCP<problem_type> d_problem;
    Teuchos::RCP<Teuchos::ParameterList> d_parameters;
    Teuchos::RCP<solver_type> d_solver;
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
                         const typename Pack::multi_vector_type& rhs,
                         typename Pack::multi_vector_type& solution,
                         const LinearSolverOptions& options = {})
{
    BelosLinearSolver<Pack> solver;
    return solver.solve(matrix, rhs, solution, options);
}

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
    using multi_vector_type = typename Pack::multi_vector_type;

    auto x = Teuchos::rcp_implicit_cast<multi_vector_type>(
        Teuchos::rcpFromRef(solution));
    auto b = Teuchos::rcp_implicit_cast<const multi_vector_type>(
        Teuchos::rcpFromRef(rhs));

    return solve_linear_system<Pack>(matrix, *b, *x, options);
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
                         const typename Pack::multi_vector_type& rhs,
                         typename Pack::multi_vector_type& solution,
                         const LinearSolverOptions& options = {})
{
    auto op = Teuchos::rcp_implicit_cast<const typename Pack::operator_type>(matrix);
    return solve_linear_system<Pack>(op, rhs, solution, options);
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
