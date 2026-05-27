/**
 * @file BelosLinearSolver.hh
 * @brief Small Belos/Tpetra linear-solver wrapper.
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

struct LinearSolverOptions
{
    int max_iterations = 200;
    real_t tolerance = 1.0e-10;
    int verbosity = Belos::Errors + Belos::Warnings;
};

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
