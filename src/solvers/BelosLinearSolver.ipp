/**
 * @file BelosLinearSolver.ipp
 * @brief Inline member implementations for BelosLinearSolver.
 */
#pragma once

#include "solvers/BelosLinearSolver.hh"

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
