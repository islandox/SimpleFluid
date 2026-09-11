#pragma once

#include "SimpleFluidExport.hh"
#include "dataclass/TpetraTypes.hh"
#include "solvers/NonlinearSolverOptions.hh"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace SimpleFluid
{
struct LinearSolverOptions;

/** A retained linearization generation; residual trials must not mutate it. */
struct NonlinearLinearization
{
    Teuchos::RCP<const DefaultTpetraTypes::operator_type> jacobian;
    Teuchos::RCP<const DefaultTpetraTypes::operator_type> right_preconditioner;
};

/** Native physical equations presented to the private NOX/Thyra adapter.
 * state_scale is Dx, residual_scale is Dr in Fhat(y) = Dr F(Dx y).
 * Null scales mean identity. Callbacks must be collective on map's communicator.
 * Return false for an inadmissible residual trial, consistently on all ranks.
 * A linearization and its preconditioner remain immutable while retained.
 */
struct NonlinearCallbacks
{
    Teuchos::RCP<const DefaultTpetraTypes::map_type> map;
    Teuchos::RCP<const DefaultTpetraTypes::vector_type> state_scale;
    Teuchos::RCP<const DefaultTpetraTypes::vector_type> residual_scale;
    std::function<bool(const DefaultTpetraTypes::vector_type&, DefaultTpetraTypes::vector_type&)> residual;
    std::function<NonlinearLinearization(const DefaultTpetraTypes::vector_type&)> linearize;
    /** Optional collective physical acceptance check, evaluated only after
     * algebraic convergence. It may update private scratch but must not publish
     * accepted fields. A false result continues nonlinear iteration.
     */
    std::function<bool(const DefaultTpetraTypes::vector_type&)> convergence_gate;
};

struct NonlinearSolveResult
{
    bool converged = false;
    int nonlinear_iterations = 0;
    int linear_solves = 0;
    int krylov_iterations = 0;
    int residual_evaluations = 0;
    int line_search_rejections = 0;
    double scaled_residual = 0.0;
    double achieved_linear_tolerance = 0.0;
    /** Wall durations are the maximum local elapsed duration across ranks.
     * Phase durations include nested native work, so do not interpret their
     * sum as an exclusive breakdown of total_seconds.
     */
    double total_seconds = 0.0;
    double residual_seconds = 0.0;
    double linearization_seconds = 0.0;
    double linear_solve_seconds = 0.0;
    std::string reason;
};

/** Cumulative workspace creation counters for a reusable solver instance. */
struct NonlinearSolverCacheStatistics
{
    std::size_t vector_cache_builds = 0;  // map-dependent native vector bundles
    std::size_t linear_solver_builds = 0; // Belos solver manager creations
    std::size_t linearization_generations = 0;
};

/** Owning, sequential-use nonlinear solver with private NOX/Thyra machinery.
 * Owns callback closures, retained linearizations, physical/scaled scratch and
 * Krylov workspaces. Every solve resets nonlinear validity and convergence
 * references; only allocations are reused. Failed solves leave input unchanged.
 */
class SIMPLEFLUID_SOLVERS_EXPORT NOXNonlinearSolver
{
public:
    using vector_type = DefaultTpetraTypes::vector_type;

    explicit NOXNonlinearSolver(NonlinearCallbacks callbacks);
    ~NOXNonlinearSolver();
    NOXNonlinearSolver(const NOXNonlinearSolver&) = delete;
    NOXNonlinearSolver& operator=(const NOXNonlinearSolver&) = delete;
    NOXNonlinearSolver(NOXNonlinearSolver&&) noexcept;
    NOXNonlinearSolver& operator=(NOXNonlinearSolver&&) noexcept;

    /** Replace owned callbacks collectively, invalidating all numerical state.
     * Compatible maps retain native vector and Krylov allocations. A different
     * map on a congruent communicator rebuilds the corresponding workspaces.
     * Invalid replacement metadata leaves the existing callbacks unchanged.
     */
    void set_callbacks(NonlinearCallbacks callbacks);
    NonlinearSolveResult solve(
        vector_type& physical_x, const NonlinearSolverOptions& nonlinear, const LinearSolverOptions& linear);
    NonlinearSolverCacheStatistics cache_statistics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> d_impl;
};

/** One-shot compatibility helper. Reuse NOXNonlinearSolver across physical steps
 * to retain allocations. Algebraic component and optional physical gates must
 * both pass before this updates physical_x.
 */
SIMPLEFLUID_SOLVERS_EXPORT NonlinearSolveResult solve_nox(const NonlinearCallbacks& callbacks,
    DefaultTpetraTypes::vector_type& physical_x, const NonlinearSolverOptions& nonlinear,
    const LinearSolverOptions& linear);
} // namespace SimpleFluid
