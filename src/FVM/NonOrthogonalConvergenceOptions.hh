/** @file NonOrthogonalConvergenceOptions.hh @brief Steady implicit correction convergence controls. */
#pragma once

#include "solvers/BelosLinearSolver.hh"

namespace SimpleFluid::FVM
{

/**
 * @brief Convergence of lagged partition gradients in an implicit steady solve.
 *
 * These controls apply to Implicit treatment only. Explicit and Hybrid retain
 * their fixed number of correction sweeps. The linear solver tolerance should
 * be tighter than residual_tolerance to leave room for the correction error.
 */
struct NonOrthogonalConvergenceOptions
{
    /** Maximum number of assembled linear solves, including the first solve. */
    int max_iterations = 40;
    /** Global infinity-norm update divided by max(1, ||solution||_infinity). */
    real_t update_tolerance = 1.0e-10;
    /** Fresh full-equation defect / physical source-and-boundary RHS norm (absolute if that norm is zero). */
    real_t residual_tolerance = 1.0e-10;
};

} // namespace SimpleFluid::FVM
