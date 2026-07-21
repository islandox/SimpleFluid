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
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace SimpleFluid
{

/** @brief Available right-preconditioning strategies for Belos solves. */
enum class LinearPreconditioner
{
    None,
    MueLu
};

/**
 * @brief Convert a preconditioner selection to its configuration name.
 *
 * @param preconditioner Preconditioner selection to convert.
 * @return Stable human-readable name.
 * @throws std::invalid_argument if @p preconditioner is not recognized.
 */
inline std::string_view to_string(LinearPreconditioner preconditioner)
{
    switch (preconditioner)
    {
        case LinearPreconditioner::None:  return "none";
        case LinearPreconditioner::MueLu: return "MueLu";
    }

    throw std::invalid_argument("Unknown LinearPreconditioner value.");
}

/** @brief Convergence statistics for one linear solve. */
struct LinearSolveStatistics
{
    bool converged = false;
    int iterations = 0;
    real_t achieved_tolerance = {};
};

/** @brief Aggregate convergence statistics across several linear solves. */
struct LinearSolveSummary
{
    bool converged = true;
    int solves = 0;
    int iterations = 0;
    real_t achieved_tolerance = {};

    void add(const LinearSolveStatistics& statistics)
    {
        converged = converged && statistics.converged;
        ++solves;
        iterations += statistics.iterations;
        achieved_tolerance =
            std::max(achieved_tolerance, statistics.achieved_tolerance);
    }
};

/**
 * @brief Configuration options for the Belos linear solver.
 */
struct LinearSolverOptions
{
    int max_iterations = 200;
    real_t tolerance = 1.0e-10;
    int verbosity = Belos::Errors + Belos::Warnings;
    LinearPreconditioner preconditioner = LinearPreconditioner::None;
    /**
     * @brief Reuse a preconditioner for consecutive solves with the exact
     *        same operator object.
     *
     * This is opt-in because an operator may be updated in place without its
     * identity changing. Enable reuse only when the operator's numerical
     * values remain unchanged between solves. A different operator object
     * always triggers a rebuild.
     */
    bool reuse_preconditioner = false;
};

namespace detail
{

/**
 * @brief Test-only access to retained Belos solver state.
 *
 * @tparam Pack Tpetra type pack used by the solver under test.
 */
template<TpetraTypePack Pack>
struct BelosLinearSolverTestAccess;

} // namespace detail

/**
 * @brief Reusable Belos GMRES solver for systems with stable maps.
 *
 * The solver manager and its Krylov workspace are retained between solves.
 * A new manager is created only when the operator domain or range map changes.
 *
 * @tparam Pack Tpetra type pack defining vectors, maps, and operators.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class BelosLinearSolver
{
public:
    using scalar_type = typename Pack::scalar_type;
    using multi_vector_type = typename Pack::multi_vector_type;
    using vector_type = typename Pack::vector_type;
    using operator_type = typename Pack::operator_type;
    using matrix_type = typename Pack::matrix_type;
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
        return solve_with_statistics(
            matrix, rhs, solution, options).converged;
    }

    LinearSolveStatistics solve_with_statistics(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options = {})
    {
        auto x = Teuchos::rcpFromRef(solution);
        auto b = Teuchos::rcpFromRef(rhs);

        if (!has_compatible_maps(matrix))
        {
            // A newly created Belos problem must never inherit a hierarchy
            // prepared for the previous problem, even if an operator address
            // were to be recycled.
            invalidate_preconditioner();
            d_problem = Teuchos::rcp(
                new problem_type(matrix, x, b));
            d_parameters = Teuchos::rcp(new Teuchos::ParameterList());
            configure(options);
            configure_preconditioner(matrix, options);
            if (!d_problem->setProblem())
            {
                return {};
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
            configure_preconditioner(matrix, options);
            if (!d_problem->setProblem())
            {
                return {};
            }
            d_solver->reset(Belos::Problem);
        }

        const auto converged =
            d_solver->solve() == Belos::Converged;
        return {
            converged,
            d_solver->getNumIters(),
            d_solver->achievedTol()};
    }

    bool solve(
        const Teuchos::RCP<const operator_type>& matrix,
        const vector_type& rhs,
        vector_type& solution,
        const LinearSolverOptions& options = {})
    {
        return solve_with_statistics(
            matrix, rhs, solution, options).converged;
    }

    LinearSolveStatistics solve_with_statistics(
        const Teuchos::RCP<const operator_type>& matrix,
        const vector_type& rhs,
        vector_type& solution,
        const LinearSolverOptions& options = {})
    {
        auto x = Teuchos::rcp_implicit_cast<multi_vector_type>(
            Teuchos::rcpFromRef(solution));
        auto b = Teuchos::rcp_implicit_cast<const multi_vector_type>(
            Teuchos::rcpFromRef(rhs));
        return solve_with_statistics(matrix, *b, *x, options);
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

    void invalidate_preconditioner()
    {
        d_preconditioner = Teuchos::null;
        d_preconditioner_operator = Teuchos::null;
        if (!d_problem.is_null())
        {
            d_problem->setRightPrec(Teuchos::null);
        }
    }

    void configure_preconditioner(
        const Teuchos::RCP<const operator_type>& matrix,
        const LinearSolverOptions& options)
    {
        if (options.preconditioner == LinearPreconditioner::None)
        {
            invalidate_preconditioner();
            return;
        }
        if (options.preconditioner != LinearPreconditioner::MueLu)
        {
            throw std::invalid_argument(
                "BelosLinearSolver received an unknown preconditioner.");
        }

        const auto crs_matrix =
            Teuchos::rcp_dynamic_cast<const matrix_type>(matrix, false);
        if (crs_matrix.is_null())
        {
            throw std::invalid_argument(
                "MueLu preconditioning requires a Tpetra::CrsMatrix operator.");
        }

        const auto can_reuse =
            options.reuse_preconditioner
            && !d_preconditioner.is_null()
            && !d_preconditioner_operator.is_null()
            && d_preconditioner_operator.getRawPtr() == matrix.getRawPtr();
        if (can_reuse)
        {
            d_problem->setRightPrec(d_preconditioner);
            return;
        }

        Teuchos::ParameterList parameters;
        parameters.set("verbosity", "none");
        parameters.set("coarse: max size", 64);
        parameters.set("smoother: type", "RELAXATION");
        parameters.sublist("smoother: params").set(
            "relaxation: type", "Jacobi");
        parameters.set("coarse: type", "RELAXATION");
        parameters.sublist("coarse: params").set(
            "relaxation: type", "Jacobi");
        parameters.sublist("coarse: params").set(
            "relaxation: sweeps", 4);

        auto mutable_matrix =
            Teuchos::rcp_const_cast<matrix_type>(crs_matrix);
        Teuchos::RCP<operator_type> mutable_operator = mutable_matrix;
        d_preconditioner =
            MueLu::CreateTpetraPreconditioner(
                mutable_operator, parameters);
        d_preconditioner_operator = matrix;
        ++d_preconditioner_setup_count;
        d_problem->setRightPrec(d_preconditioner);
    }

    friend struct detail::BelosLinearSolverTestAccess<Pack>;

    Teuchos::RCP<problem_type> d_problem;
    Teuchos::RCP<Teuchos::ParameterList> d_parameters;
    Teuchos::RCP<solver_type> d_solver;
    Teuchos::RCP<const operator_type> d_preconditioner;
    Teuchos::RCP<const operator_type> d_preconditioner_operator;
    std::size_t d_preconditioner_setup_count = 0;
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
