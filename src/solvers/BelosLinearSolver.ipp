/**
 * @file BelosLinearSolver.ipp
 * @brief Inline member implementations for BelosLinearSolver.
 */
#pragma once

#include <algorithm>

namespace SimpleFluid
{

inline void LinearSolveSummary::add(const LinearSolveStatistics& statistics)
{
    converged = converged && statistics.converged;
    ++solves;
    iterations += statistics.iterations;
    achieved_tolerance =
        std::max(achieved_tolerance, statistics.achieved_tolerance);
}


template <TpetraTypePack Pack>
void BelosLinearSolver<Pack>::validate_options(const LinearSolverOptions& options)
{
    if (options.max_iterations <= 0)
    {
        throw std::invalid_argument(
            "BelosLinearSolver requires positive maximum iterations.");
    }
    if (!std::isfinite(options.tolerance)
        || options.tolerance <= real_t{})
    {
        throw std::invalid_argument(
            "BelosLinearSolver requires a positive finite tolerance.");
    }
    static_cast<void>(to_string(options.backend));
    static_cast<void>(to_string(options.preconditioner));
    if (options.backend == LinearSolverBackend::Cg
        && (options.preconditioner == LinearPreconditioner::ILU0
            || options.preconditioner == LinearPreconditioner::ILUT
            || options.preconditioner == LinearPreconditioner::GaussSeidel))
    {
        throw std::invalid_argument(
            "CG requires a symmetric positive-definite preconditioner; "
            "ILU0, ILUT, and forward Gauss-Seidel are not supported.");
    }
}

template<TpetraTypePack Pack>
inline LinearSolveStatistics BelosLinearSolver<Pack>::solve_with_statistics(
    const Teuchos::RCP<const operator_type>& matrix,
    const vector_type& rhs,
    vector_type& solution,
    const LinearSolverOptions& options,
    LinearResidualScaling residual_scaling,
    const Teuchos::RCP<const matrix_type>& preconditioning_matrix)
{
    auto x = Teuchos::rcp_implicit_cast<multi_vector_type>(
        Teuchos::rcpFromRef(solution));
    auto b = Teuchos::rcp_implicit_cast<const multi_vector_type>(
        Teuchos::rcpFromRef(rhs));
    return solve_with_statistics(
        matrix, *b, *x, options, residual_scaling, preconditioning_matrix);
}

} // namespace SimpleFluid
