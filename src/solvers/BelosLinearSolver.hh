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
#include "solvers/DICPreconditioner.hh"

#include <BelosBiCGStabSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosSolverManager.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosTypes.hpp>
#include <Ifpack2_Factory.hpp>
#include <Ifpack2_Preconditioner.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace SimpleFluid
{

/** @brief Krylov algorithms available through the Belos wrapper. */
enum class LinearSolverBackend
{
    Gmres,
    Cg,
    BiCGStab
};

/**
 * @brief Convert a linear-solver backend to its stable configuration name.
 */
inline std::string_view to_string(LinearSolverBackend backend)
{
    switch (backend)
    {
        case LinearSolverBackend::Gmres:     return "gmres";
        case LinearSolverBackend::Cg:        return "cg";
        case LinearSolverBackend::BiCGStab:  return "bicgstab";
    }

    throw std::invalid_argument("Unknown LinearSolverBackend value.");
}

/**
 * @brief Parse a supported linear-solver backend name.
 * @throws std::invalid_argument if @p value is unknown.
 */
inline LinearSolverBackend parse_linear_solver_backend(
    std::string_view value)
{
    if (value == "gmres" || value == "GMRES"
        || value == "pseudoBlockGMRES")
    {
        return LinearSolverBackend::Gmres;
    }
    if (value == "cg" || value == "CG" || value == "pcg" || value == "PCG"
        || value == "pseudoBlockCG")
    {
        return LinearSolverBackend::Cg;
    }
    if (value == "bicgstab" || value == "BiCGStab"
        || value == "BICGSTAB")
    {
        return LinearSolverBackend::BiCGStab;
    }
    throw std::invalid_argument(
        "Unknown linear-solver backend '" + std::string(value) + "'.");
}

/** @brief Available inverse preconditioners for Belos solves. */
enum class LinearPreconditioner
{
    None,
    MueLu,
    Jacobi,
    ILU0,
    ILUT,
    DIC,
    GaussSeidel,
    SymmetricGaussSeidel
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
        case LinearPreconditioner::None:    return "none";
        case LinearPreconditioner::MueLu:   return "MueLu";
        case LinearPreconditioner::Jacobi:  return "jacobi";
        case LinearPreconditioner::ILU0:    return "ilu0";
        case LinearPreconditioner::ILUT:    return "ilut";
        case LinearPreconditioner::DIC:     return "dic";
        case LinearPreconditioner::GaussSeidel: return "gaussSeidel";
        case LinearPreconditioner::SymmetricGaussSeidel:
            return "symmetricGaussSeidel";
    }

    throw std::invalid_argument("Unknown LinearPreconditioner value.");
}

/**
 * @brief Parse a supported preconditioner name.
 * @throws std::invalid_argument if @p value is unknown.
 */
inline LinearPreconditioner parse_linear_preconditioner(
    std::string_view value)
{
    if (value == "none" || value == "off")
        return LinearPreconditioner::None;
    if (value == "MueLu" || value == "muelu")
        return LinearPreconditioner::MueLu;
    if (value == "jacobi" || value == "Jacobi")
        return LinearPreconditioner::Jacobi;
    if (value == "ilu0" || value == "ILU0"
        || value == "riluk" || value == "RILUK")
    {
        return LinearPreconditioner::ILU0;
    }
    if (value == "ilut" || value == "ILUT")
        return LinearPreconditioner::ILUT;
    if (value == "dic" || value == "DIC")
        return LinearPreconditioner::DIC;
    if (value == "gaussSeidel" || value == "GaussSeidel"
        || value == "gs" || value == "GS")
        return LinearPreconditioner::GaussSeidel;
    if (value == "symmetricGaussSeidel" || value == "SymmetricGaussSeidel"
        || value == "symGaussSeidel" || value == "sgs" || value == "SGS")
        return LinearPreconditioner::SymmetricGaussSeidel;
    throw std::invalid_argument(
        "Unknown linear preconditioner '" + std::string(value) + "'.");
}

/**
 * @brief Optional explicit-residual scaling for one linear solve.
 *
 * The true residual is divided by the larger of the current RHS norm and
 * @p rhs_norm_floor.  A zero floor preserves the usual RHS-relative metric.
 */
struct LinearResidualScaling
{
    real_t rhs_norm_floor = {};
};

/** @brief Convergence statistics for one linear solve. */
struct LinearSolveStatistics
{
    bool converged = false;
    /** Includes refinement; BiCGStab also sums independently solved columns. */
    int iterations = 0;
    real_t achieved_tolerance = {};
    real_t rhs_norm = {};
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
    /**
     * CG is intended only when both the operator and configured
     * preconditioner preserve a symmetric positive-definite system. ILU0,
     * ILUT, and forward Gauss-Seidel are rejected with CG. GMRES and BiCGStab
     * support the nonsymmetric transport and gauge-fixed pressure operators
     * assembled by the general FVM path.
     */
    LinearSolverBackend backend = LinearSolverBackend::Gmres;
    /**
     * DIC requires a symmetric matrix with positive incomplete pivots; its
     * distributed factor retains cross-rank couplings using halo exchanges.
     * Gauss-Seidel variants use one unit-damped sweep from a zero guess;
     * under MPI they relax within each rank, with Jacobi between ranks.
     */
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
 * @brief Reusable Belos solver for systems with stable maps.
 *
 * The selected solver manager and its Krylov workspace are retained between
 * compatible solves. A new manager is created when maps or the selected
 * backend change. Convergence is scaled by the RHS norm so a good transient
 * initial guess does not make the requested absolute residual unattainable.
 *
 * @tparam Pack Tpetra type pack defining vectors, maps, and operators.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class BelosLinearSolver
{
public:
    using scalar_type = typename Pack::scalar_type;
    using magnitude_type =
        typename Teuchos::ScalarTraits<scalar_type>::magnitudeType;
    using multi_vector_type = typename Pack::multi_vector_type;
    using vector_type = typename Pack::vector_type;
    using operator_type = typename Pack::operator_type;
    using matrix_type = typename Pack::matrix_type;
    using problem_type =
        Belos::LinearProblem<scalar_type, multi_vector_type, operator_type>;
    using solver_type =
        Belos::SolverManager<
            scalar_type, multi_vector_type, operator_type>;
    using ifpack2_preconditioner_type =
        Ifpack2::Preconditioner<
            scalar_type,
            typename Pack::local_ordinal_type,
            typename Pack::global_ordinal_type,
            typename Pack::node_type>;

    /**
     * @brief Invalidate numeric factors after changing the current operator in place.
     *
     * Operator identity alone cannot detect changed coefficients. Call collectively
     * before the next solve whenever any rank changes values; compatible Krylov
     * storage remains available, but the preconditioner is rebuilt from new values.
     * The cumulative preconditioner setup count is retained.
     */
    void notify_operator_values_changed()
    {
        invalidate_preconditioner();
    }

    /** @brief Release all operator-dependent state, including compatible Krylov storage. */
    void reset()
    {
        if (!d_problem.is_null())
        {
            d_problem->setRightPrec(Teuchos::null);
        }
        d_problem = Teuchos::null;
        d_parameters = Teuchos::null;
        d_solver = Teuchos::null;
        d_backend.reset();
        d_residual_workspace = Teuchos::null;
        d_bicgstab_rhs_workspace = Teuchos::null;
        d_bicgstab_solution_workspace = Teuchos::null;
        d_preconditioner = Teuchos::null;
        d_preconditioner_operator = Teuchos::null;
        d_preconditioner_kind.reset();
        d_muelu_factory = Teuchos::null;
        d_muelu_map = Teuchos::null;
    }

    bool solve(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options = {})
    {
        return solve_with_statistics(
            matrix, rhs, solution, options).converged;
    }

    /**
     * @param solution Initial guesses replaced by computed solutions.
     * @return Convergence flag, iteration count, and achieved tolerance.
     */
    LinearSolveStatistics solve_with_statistics(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options = {},
        LinearResidualScaling residual_scaling = {})
    {
        return solve_impl(matrix, rhs, solution, options, residual_scaling, false);
    }

    /**
     * @brief Solve after explicitly replacing every initial-guess column by zero.
     *
     * Establishing the zero guess here makes r0=b exact without an operator
     * application or the warm-start screening norms. The final true-residual
     * check and bounded refinement use the same contract as ordinary solves.
     */
    LinearSolveStatistics solve_from_zero_with_statistics(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options = {},
        LinearResidualScaling residual_scaling = {})
    {
        return solve_impl(matrix, rhs, solution, options, residual_scaling, true);
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

    /**
     * @param solution Initial guess replaced by the computed solution.
     * @return Convergence flag, iteration count, and achieved tolerance.
     */
    LinearSolveStatistics solve_with_statistics(
        const Teuchos::RCP<const operator_type>& matrix,
        const vector_type& rhs,
        vector_type& solution,
        const LinearSolverOptions& options = {},
        LinearResidualScaling residual_scaling = {})
    {
        auto x = Teuchos::rcp_implicit_cast<multi_vector_type>(
            Teuchos::rcpFromRef(solution));
        auto b = Teuchos::rcp_implicit_cast<const multi_vector_type>(
            Teuchos::rcpFromRef(rhs));
        return solve_with_statistics(
            matrix, *b, *x, options, residual_scaling);
    }

    /** @brief Validate local option values before callers agree across ranks. */
    static void validate_options(const LinearSolverOptions& options)
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

private:
    LinearSolveStatistics solve_impl(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options,
        LinearResidualScaling residual_scaling,
        bool zero_initial_guess)
    {
        validate_options(options);
        validate_residual_scaling(residual_scaling);
        if (options.backend == LinearSolverBackend::BiCGStab
            && rhs.getNumVectors() > 1)
        {
            return solve_bicgstab_columns(
                matrix, rhs, solution, options, residual_scaling, zero_initial_guess);
        }
        if (matrix.is_null())
        {
            throw std::invalid_argument(
                "BelosLinearSolver requires a non-null operator.");
        }
        if (zero_initial_guess)
        {
            if (rhs.getNumVectors() != solution.getNumVectors()
                || !matrix->getDomainMap()->isSameAs(*solution.getMap())
                || !matrix->getRangeMap()->isSameAs(*rhs.getMap()))
            {
                throw std::invalid_argument(
                    "BelosLinearSolver requires matching operator maps and RHS/solution vector counts.");
            }
            solution.putScalar(scalar_type{});
            residual_workspace(rhs).update(scalar_type{1}, rhs, scalar_type{});
        }
        else
            prepare_initial_guess(matrix, rhs, solution);
        auto x = Teuchos::rcpFromRef(solution);
        auto b = Teuchos::rcpFromRef(rhs);

        const bool rebuild_solver =
            !has_compatible_maps(matrix)
            || !d_backend.has_value()
            || *d_backend != options.backend;
        if (rebuild_solver)
        {
            // A newly created Belos problem must never inherit a hierarchy
            // prepared for the previous problem, even if an operator address
            // were to be recycled.
            invalidate_preconditioner();
            d_solver = Teuchos::null;
            d_backend.reset();
            d_problem = Teuchos::rcp(
                new problem_type(matrix, x, b));
            d_parameters = Teuchos::rcp(new Teuchos::ParameterList());
            configure(options);
            configure_preconditioner(matrix, options);
            if (!set_problem_with_prepared_residual())
            {
                return {};
            }
            d_solver = create_solver(options.backend);
            d_backend = options.backend;
        }
        else
        {
            d_problem->setOperator(matrix);
            d_problem->setLHS(x);
            d_problem->setRHS(b);
            configure(options);
            d_solver->setParameters(d_parameters);
            configure_preconditioner(matrix, options);
            if (!set_problem_with_prepared_residual())
            {
                return {};
            }
        }

        auto return_status = d_solver->solve();
        int iterations = d_solver->getNumIters();
        real_t rhs_norm{};
        bool accurate_residual = false;
        // The ceiling identifies a floating-point residual gap, not an
        // acceptance tolerance. Dense pressure systems can exceed a small
        // multiple of a 1e-14 target while remaining near roundoff.
        const auto gmres_gap_ceiling = std::max(real_t{8} * options.tolerance,
            static_cast<real_t>(std::sqrt(std::numeric_limits<magnitude_type>::epsilon())));
        const auto checked_residual = [&] {
            if (accurate_residual && accurate_crs_residual(matrix, rhs, solution))
                return residual_relative_norm(rhs, residual_scaling, &rhs_norm);
            const auto standard = true_relative_residual(matrix, rhs, solution, residual_scaling, &rhs_norm);
            if (options.backend == LinearSolverBackend::Gmres
                && std::isfinite(standard) && standard > options.tolerance
                && standard <= gmres_gap_ceiling && accurate_crs_residual(matrix, rhs, solution))
            {
                accurate_residual = true; // use the same operator evaluation for later defects/checks
                const auto checked = residual_relative_norm(rhs, residual_scaling, &rhs_norm);
                if (!d_reported_accurate_residual)
                {
                    if (rhs.getMap()->getComm()->getRank() == 0 && (options.verbosity & Belos::Warnings))
                    {
                        std::ostringstream message;
                        message.precision(std::numeric_limits<real_t>::max_digits10);
                        message << "BelosLinearSolver CRS residual precision check: standard=" << standard
                                << " compensated=" << checked << " tolerance=" << options.tolerance << '\n';
                        std::cerr << message.str();
                    }
                    d_reported_accurate_residual = true;
                }
                return checked;
            }
            return standard;
        };
        auto achieved_tolerance = checked_residual();

        // Recurrence residuals can pass while b - A*x is still just above the
        // requested tolerance. GMRES may instead report loss of accuracy (LOA)
        // and stop early with the same small residual gap. CG/BiCGStab restart
        // from the recomputed residual; GMRES solves a zero-start defect
        // correction so its explicit check does not subtract the full A*x
        // inside every Krylov convergence test. The gap bounds select eligible
        // retries only; final acceptance remains <= tolerance.
        // An ordinary unconverged solve without LOA is never retried here.
        // Keep both the original iteration budget and a separate restart cap:
        // a zero-iteration or stagnating retry must still terminate. The
        // preconditioner and original RHS scaling remain unchanged.
        auto refinement_options = options;
        const int maximum_refinements = options.backend == LinearSolverBackend::Gmres ? 4 : 2;
        std::array<real_t, 5> refinement_residuals{achieved_tolerance};
        size_t refinement_count = 1;
        for (int refinement = 0;
             refinement < maximum_refinements
                 && (return_status == Belos::Converged
                     || (options.backend == LinearSolverBackend::Gmres
                         && d_solver->isLOADetected()))
                 && std::isfinite(achieved_tolerance)
                 && achieved_tolerance > options.tolerance
                 && (options.backend == LinearSolverBackend::Gmres
                     ? achieved_tolerance <= gmres_gap_ceiling
                     : achieved_tolerance / options.tolerance <= real_t{2})
                 && iterations < options.max_iterations;
             ++refinement)
        {
            refinement_options.max_iterations =
                options.max_iterations - iterations;
            if (options.backend == LinearSolverBackend::Gmres)
            {
                auto defect = Teuchos::rcp(new multi_vector_type(rhs.getMap(), rhs.getNumVectors()));
                defect->update(scalar_type{1}, residual_workspace(rhs), scalar_type{});
                auto correction = Teuchos::rcp(new multi_vector_type(solution.getMap(), solution.getNumVectors()));
                correction->putScalar(scalar_type{});
                // This target is relative to the defect RHS, not the original
                // RHS. It requests at most half the original absolute target
                // for every column; only the original final residual accepts.
                refinement_options.tolerance = std::min(real_t{0.1},
                    real_t{0.5} / (achieved_tolerance / options.tolerance));
                const auto restore_original_vectors = [&] {
                    d_problem->setLHS(x);
                    d_problem->setRHS(b);
                    d_problem->setInitResVec(d_residual_workspace);
                    d_problem->setInitPrecResVec(Teuchos::null);
                    // Clear any current-system views of the correction and
                    // defect, including if a manager threw before setCurrLS.
                    // The prepared original residual avoids another SpMV.
                    (void)d_problem->setProblem();
                };
                try
                {
                    configure(refinement_options);
                    d_solver->setParameters(d_parameters);
                    d_problem->setLHS(correction);
                    d_problem->setRHS(defect);
                    d_problem->setInitResVec(defect); // zero-start residual is exactly r
                    d_problem->setInitPrecResVec(Teuchos::null);
                    if (!d_problem->setProblem())
                    {
                        restore_original_vectors();
                        break;
                    }
                    return_status = d_solver->solve();
                    iterations += d_solver->getNumIters();
                }
                catch (...)
                {
                    restore_original_vectors();
                    throw;
                }
                restore_original_vectors();
                solution.update(scalar_type{1}, *correction, scalar_type{1});
            }
            else
            {
                refinement_options.tolerance *= real_t{0.1};
                if (refinement_options.tolerance <= real_t{})
                    break;
                configure(refinement_options);
                d_solver->setParameters(d_parameters);
                if (!set_problem_with_prepared_residual())
                    break;
                return_status = d_solver->solve();
                iterations += d_solver->getNumIters();
            }
            achieved_tolerance = checked_residual();
            refinement_residuals[refinement_count++] = achieved_tolerance;
        }

        const bool converged = std::isfinite(achieved_tolerance)
            && achieved_tolerance <= options.tolerance;
        // Belos::Errors is zero and denotes diagnostics that are always printed.
        if (!converged)
        {
            if (options.backend == LinearSolverBackend::Gmres && refinement_count > 1
                && rhs.getMap()->getComm()->getRank() == 0 && (options.verbosity & Belos::Warnings))
            {
                std::ostringstream message;
                message.precision(std::numeric_limits<real_t>::max_digits10);
                message << "GMRES defect residuals at original RHS scaling:";
                for (size_t i = 0; i < refinement_count; ++i) message << ' ' << refinement_residuals[i];
                std::cerr << message.str() << '\n';
            }
            report_failed_solve(
                rhs, options, residual_scaling, return_status, iterations);
        }
        return {
            converged,
            iterations,
            achieved_tolerance,
            rhs_norm};
    }

    /**
     * @brief Solve BiCGStab columns in owned contiguous storage.
     *
     * Trilinos 17.2 Tpetra single-column subviews of a nonconstant-stride
     * multivector use whichVectors[0] + index rather than whichVectors[index].
     * BiCGStab's per-column axpy reaches that path after RHS deflation; e.g.
     * surviving columns [0,2] cause an update intended for 2 to overwrite 1.
     * Independent owning columns avoid nested deflated views entirely.
     * Each RHS retains its tolerance and iteration limit. A preconditioner
     * can safely be reused within this call because its matrix is unchanged.
     */
    LinearSolveStatistics solve_bicgstab_columns(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs, multi_vector_type& solution,
        const LinearSolverOptions& options,
        LinearResidualScaling residual_scaling,
        bool zero_initial_guess)
    {
        if (rhs.getNumVectors() != solution.getNumVectors())
        {
            throw std::invalid_argument(
                "BiCGStab requires matching RHS and solution vector counts.");
        }
        if (d_bicgstab_rhs_workspace.is_null()
            || !d_bicgstab_rhs_workspace->getMap()->isSameAs(*rhs.getMap()))
        {
            d_bicgstab_rhs_workspace = Teuchos::rcp(
                new multi_vector_type(rhs.getMap(), 1, false));
        }
        if (d_bicgstab_solution_workspace.is_null()
            || !d_bicgstab_solution_workspace->getMap()->isSameAs(*solution.getMap()))
        {
            d_bicgstab_solution_workspace = Teuchos::rcp(
                new multi_vector_type(solution.getMap(), 1, false));
        }

        LinearSolveStatistics combined;
        combined.converged = true;
        auto column_options = options;
        for (std::size_t column = 0; column < rhs.getNumVectors(); ++column)
        {
            {
                const auto source = rhs.getData(column);
                auto destination = d_bicgstab_rhs_workspace->getDataNonConst(0);
                std::copy(source.begin(), source.end(), destination.begin());
            }
            if (!zero_initial_guess)
            {
                const auto source = solution.getData(column);
                auto destination = d_bicgstab_solution_workspace->getDataNonConst(0);
                std::copy(source.begin(), source.end(), destination.begin());
            }
            const auto statistics = solve_impl(
                matrix, *d_bicgstab_rhs_workspace, *d_bicgstab_solution_workspace,
                column_options, residual_scaling, zero_initial_guess);
            {
                const auto source = d_bicgstab_solution_workspace->getData(0);
                auto destination = solution.getDataNonConst(column);
                std::copy(source.begin(), source.end(), destination.begin());
            }
            combined.converged = combined.converged && statistics.converged;
            combined.iterations += statistics.iterations;
            combined.achieved_tolerance = std::max(
                combined.achieved_tolerance, statistics.achieved_tolerance);
            combined.rhs_norm = std::max(combined.rhs_norm, statistics.rhs_norm);
            column_options.reuse_preconditioner = true;
        }
        return combined;
    }

    /**
     * Hand off the residual already prepared for the current RHS and guess.
     * Belos stores this owning RCP without copying its multivector. The scratch
     * storage remains alive for the complete solve; only after solve() returns
     * does the true-residual check overwrite it. Each solve and refinement
     * refreshes both residual pointers before setProblem(), including after a
     * RHS/map/column-count change. Refinement reuses the just-checked b-A*x.
     */
    bool set_problem_with_prepared_residual()
    {
        d_problem->setInitResVec(d_residual_workspace);
        d_problem->setInitPrecResVec(Teuchos::null);
        return d_problem->setProblem();
    }

    /** @brief Report the true residual per RHS after an unsuccessful solve. */
    void report_failed_solve(
        const multi_vector_type& rhs,
        const LinearSolverOptions& options,
        LinearResidualScaling residual_scaling,
        Belos::ReturnType return_status,
        int iterations) const
    {
        Teuchos::Array<magnitude_type> rhs_norms(rhs.getNumVectors());
        Teuchos::Array<magnitude_type> residual_norms(rhs.getNumVectors());
        rhs.norm2(rhs_norms());
        residual_workspace(rhs).norm2(residual_norms());
        if (rhs.getMap()->getComm()->getRank() != 0)
            return;

        std::ostringstream message;
        message.precision(std::numeric_limits<real_t>::max_digits10);
        message << "BelosLinearSolver " << to_string(options.backend) << '/'
                << to_string(options.preconditioner)
                << " failed explicit residual check: Belos="
                << (return_status == Belos::Converged ? "converged" : "unconverged")
                << " iterations=" << iterations
                << " tolerance=" << options.tolerance << '\n';
        for (std::size_t column = 0; column < rhs.getNumVectors(); ++column)
        {
            const auto denominator = std::max(rhs_norms[column],
                static_cast<magnitude_type>(residual_scaling.rhs_norm_floor));
            const auto scaled = denominator > magnitude_type{}
                ? residual_norms[column] / denominator : residual_norms[column];
            message << "  RHS " << column << ": norm=" << rhs_norms[column]
                    << " residual=" << residual_norms[column]
                    << " scaled=" << scaled << '\n';
        }
        std::cerr << message.str();
    }

    static void validate_residual_scaling(
        const LinearResidualScaling& residual_scaling)
    {
        if (!std::isfinite(residual_scaling.rhs_norm_floor)
            || residual_scaling.rhs_norm_floor < real_t{})
        {
            throw std::invalid_argument(
                "BelosLinearSolver requires a finite non-negative RHS "
                "norm floor.");
        }
    }

    Teuchos::RCP<solver_type> create_solver(
        LinearSolverBackend backend) const
    {
        switch (backend)
        {
            case LinearSolverBackend::Gmres:
                return Teuchos::rcp(
                    new Belos::PseudoBlockGmresSolMgr<
                        scalar_type, multi_vector_type, operator_type>(
                            d_problem, d_parameters));
            case LinearSolverBackend::Cg:
                return Teuchos::rcp(
                    new Belos::PseudoBlockCGSolMgr<
                        scalar_type, multi_vector_type, operator_type>(
                            d_problem, d_parameters));
            case LinearSolverBackend::BiCGStab:
                return Teuchos::rcp(
                    new Belos::BiCGStabSolMgr<
                        scalar_type, multi_vector_type, operator_type>(
                            d_problem, d_parameters));
        }

        throw std::invalid_argument(
            "BelosLinearSolver received an unknown backend.");
    }

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
        d_parameters = Teuchos::rcp(new Teuchos::ParameterList());
        d_parameters->set(
            "Maximum Iterations", options.max_iterations);
        d_parameters->set(
            "Convergence Tolerance", options.tolerance);
        d_parameters->set("Verbosity", options.verbosity);
        d_parameters->set(
            "Implicit Residual Scaling", "Norm of RHS");
        if (options.backend == LinearSolverBackend::Gmres)
        {
            d_parameters->set(
                "Explicit Residual Scaling", "Norm of RHS");
        }
    }

    void invalidate_preconditioner()
    {
        d_preconditioner = Teuchos::null;
        d_preconditioner_operator = Teuchos::null;
        d_preconditioner_kind.reset();
        if (!d_problem.is_null())
        {
            d_problem->setRightPrec(Teuchos::null);
        }
    }

    /**
     * @brief Build, reuse, or remove the configured right preconditioner.
     * @throws std::invalid_argument If the selection is unknown or a
     *         matrix-based preconditioner is requested for a non-CRS operator.
     */
    void configure_preconditioner(
        const Teuchos::RCP<const operator_type>& matrix,
        const LinearSolverOptions& options)
    {
        if (options.preconditioner == LinearPreconditioner::None)
        {
            invalidate_preconditioner();
            return;
        }

        const auto crs_matrix =
            Teuchos::rcp_dynamic_cast<const matrix_type>(matrix, false);
        if (crs_matrix.is_null())
        {
            throw std::invalid_argument(
                std::string(to_string(options.preconditioner))
                + " preconditioning requires a Tpetra::CrsMatrix operator.");
        }

        const auto can_reuse =
            options.reuse_preconditioner
            && !d_preconditioner.is_null()
            && !d_preconditioner_operator.is_null()
            && d_preconditioner_operator.getRawPtr() == matrix.getRawPtr()
            && d_preconditioner_kind.has_value()
            && *d_preconditioner_kind == options.preconditioner;
        if (can_reuse)
        {
            d_problem->setRightPrec(d_preconditioner);
            return;
        }

        switch (options.preconditioner)
        {
            case LinearPreconditioner::DIC:
                d_preconditioner = Teuchos::rcp(
                    new detail::DICPreconditioner<Pack>(*crs_matrix));
                break;
            case LinearPreconditioner::GaussSeidel:
            case LinearPreconditioner::SymmetricGaussSeidel:
                d_preconditioner = create_ifpack2_preconditioner(
                    crs_matrix, "RELAXATION",
                    [&options](Teuchos::ParameterList& parameters)
                    {
                        parameters.set("relaxation: type",
                            options.preconditioner
                                    == LinearPreconditioner::GaussSeidel
                                ? "Gauss-Seidel" : "Symmetric Gauss-Seidel");
                        parameters.set("relaxation: sweeps", 1);
                        parameters.set("relaxation: damping factor", scalar_type{1});
                        parameters.set("relaxation: zero starting solution", true);
                    });
                break;
            case LinearPreconditioner::Jacobi:
                d_preconditioner =
                    create_ifpack2_preconditioner(
                        crs_matrix, "RELAXATION",
                        [](Teuchos::ParameterList& parameters)
                        {
                            parameters.set(
                                "relaxation: type", "Jacobi");
                            parameters.set(
                                "relaxation: sweeps", 1);
                            parameters.set(
                                "relaxation: damping factor",
                                scalar_type{1});
                            parameters.set(
                                "relaxation: zero starting solution",
                                true);
                        });
                break;
            case LinearPreconditioner::ILU0:
                d_preconditioner =
                    create_ifpack2_preconditioner(
                        crs_matrix, "RILUK",
                        [](Teuchos::ParameterList& parameters)
                        {
                            parameters.set(
                                "fact: iluk level-of-fill", 0);
                        });
                break;
            case LinearPreconditioner::ILUT:
                d_preconditioner =
                    create_ifpack2_preconditioner(
                        crs_matrix, "ILUT",
                        [](Teuchos::ParameterList& parameters)
                        {
                            parameters.set(
                                "fact: ilut level-of-fill", 1.0);
                            parameters.set(
                                "fact: drop tolerance",
                                magnitude_type{1.0e-4});
                        });
                break;
            case LinearPreconditioner::MueLu:
            {
                // The policy is fixed, but parsing it builds a sizeable factory
                // tree. Retain that configuration for the exact same map/comm;
                // every numeric setup still gets a fresh hierarchy and factors.
                if (d_muelu_factory.is_null() ||
                    d_muelu_map.get() != crs_matrix->getDomainMap().get())
                {
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
                    d_muelu_factory = Teuchos::rcp(new muelu_factory_type(
                        parameters, crs_matrix->getDomainMap()->getComm()));
                    d_muelu_map = crs_matrix->getDomainMap();
                    ++d_muelu_configuration_count;
                }

                auto mutable_matrix =
                    Teuchos::rcp_const_cast<matrix_type>(crs_matrix);
                auto xpetra_matrix = Xpetra::toXpetra(mutable_matrix);
                auto hierarchy = d_muelu_factory->CreateHierarchy(xpetra_matrix->getObjectLabel());
                hierarchy->setlib(xpetra_matrix->getDomainMap()->lib());
                hierarchy->GetLevel(0)->Set("A", xpetra_matrix);
                hierarchy->SetProcRankVerbose(xpetra_matrix->getDomainMap()->getComm()->getRank());
                d_muelu_factory->SetupHierarchy(*hierarchy);
                d_preconditioner = Teuchos::rcp(new MueLu::TpetraOperator<scalar_type,
                    typename Pack::local_ordinal_type, typename Pack::global_ordinal_type,
                    typename Pack::node_type>(hierarchy));
                break;
            }
            case LinearPreconditioner::None:
                throw std::logic_error(
                    "BelosLinearSolver reached disabled preconditioner setup.");
            default:
                throw std::invalid_argument(
                    "BelosLinearSolver received an unknown preconditioner.");
        }

        d_preconditioner_operator = matrix;
        d_preconditioner_kind = options.preconditioner;
        ++d_preconditioner_setup_count;
        // Belos CG applies this inverse to the residual in its PCG recurrence
        // and applies the original A to search directions. BiCGStab requires
        // right attachment; retaining it also avoids preconditioned norm scaling.
        d_problem->setRightPrec(d_preconditioner);
    }

    template<class Configure>
    Teuchos::RCP<const operator_type> create_ifpack2_preconditioner(
        const Teuchos::RCP<const matrix_type>& matrix,
        const std::string& factory_name,
        Configure&& configure_parameters) const
    {
        Teuchos::ParameterList parameters;
        std::forward<Configure>(configure_parameters)(parameters);
        auto preconditioner =
            Ifpack2::Factory::create(factory_name, matrix);
        preconditioner->setParameters(parameters);
        preconditioner->initialize();
        preconditioner->compute();
        return Teuchos::rcp_implicit_cast<const operator_type>(
            preconditioner);
    }

    /** @brief Recompute the stored real CRS operator with extended accumulation.
     * This does not replace the diagonal, reconstruct fluxes, or change scaling.
     * Other operators/map layouts retain their existing apply-based residual.
     */
    bool accurate_crs_residual(const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs, const multi_vector_type& solution) const
    {
        if constexpr (!std::is_floating_point_v<scalar_type>) return false;
        else
        {
            const auto crs = Teuchos::rcp_dynamic_cast<const matrix_type>(matrix);
            const auto comm = rhs.getMap()->getComm();
            int local_supported = !crs.is_null() && crs->isFillComplete();
            int supported = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_supported, &supported);
            if (!supported) return false;
            if (!crs->getRowMap()->isSameAs(*rhs.getMap())
                || !crs->getRangeMap()->isSameAs(*rhs.getMap())
                || !crs->getDomainMap()->isSameAs(*solution.getMap())) return false;
            const auto importer = crs->getCrsGraph()->getImporter();
            const int has_importer = !importer.is_null();
            int minimum_importer = 0, maximum_importer = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &has_importer, &minimum_importer);
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &has_importer, &maximum_importer);
            if (minimum_importer != maximum_importer) return false;
            Teuchos::RCP<multi_vector_type> imported;
            const multi_vector_type* column_solution = &solution;
            if (has_importer)
            {
                imported = Teuchos::rcp(new multi_vector_type(crs->getColMap(), solution.getNumVectors()));
                imported->doImport(solution, *importer, Tpetra::INSERT);
                column_solution = imported.get();
            }
            else if (!crs->getColMap()->isSameAs(*solution.getMap())) return false;
            auto& residual = residual_workspace(rhs);
            const auto local = crs->getLocalMatrixHost();
            for (size_t column = 0; column < rhs.getNumVectors(); ++column)
            {
                // Logical-column access also handles nonconstant-stride views.
                const auto x_values = column_solution->getData(column);
                const auto b_values = rhs.getData(column);
                auto r_values = residual.getDataNonConst(column);
                for (size_t row = 0; row < rhs.getLocalLength(); ++row)
                {
                    long double sum = b_values[row], correction = 0;
                    for (auto entry = local.graph.row_map(row); entry < local.graph.row_map(row + 1); ++entry)
                    {
                        const auto term = -static_cast<long double>(local.values(entry))
                            * static_cast<long double>(x_values[local.graph.entries(entry)]);
                        const auto next = sum + term;
                        correction += std::abs(sum) >= std::abs(term)
                            ? (sum - next) + term : (term - next) + sum;
                        sum = next;
                    }
                    r_values[row] = static_cast<scalar_type>(sum + correction);
                }
            }
            return true;
        }
    }

    real_t true_relative_residual(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        const multi_vector_type& solution,
        LinearResidualScaling residual_scaling = {},
        real_t* maximum_rhs_norm = nullptr) const
    {
        validate_residual_scaling(residual_scaling);
        auto& residual = residual_workspace(rhs);
        residual.update(
            scalar_type{1}, rhs, scalar_type{0});
        matrix->apply(
            solution, residual, Teuchos::NO_TRANS,
            scalar_type{-1}, scalar_type{1});

        return residual_relative_norm(rhs, residual_scaling, maximum_rhs_norm);
    }

    real_t residual_relative_norm(const multi_vector_type& rhs,
        LinearResidualScaling residual_scaling = {}, real_t* maximum_rhs_norm = nullptr) const
    {
        validate_residual_scaling(residual_scaling);
        const auto& residual = residual_workspace(rhs);

        Teuchos::Array<magnitude_type> rhs_norms(
            rhs.getNumVectors());
        Teuchos::Array<magnitude_type> residual_norms(
            residual.getNumVectors());
        rhs.norm2(rhs_norms());
        residual.norm2(residual_norms());

        magnitude_type maximum{};
        magnitude_type largest_rhs_norm{};
        for (std::size_t column = 0;
             column < rhs.getNumVectors();
             ++column)
        {
            const auto rhs_norm = rhs_norms[column];
            const auto residual_norm = residual_norms[column];
            if (!std::isfinite(rhs_norm)
                || !std::isfinite(residual_norm))
            {
                if (maximum_rhs_norm != nullptr)
                {
                    *maximum_rhs_norm =
                        std::numeric_limits<real_t>::infinity();
                }
                return std::numeric_limits<real_t>::infinity();
            }
            largest_rhs_norm =
                std::max(largest_rhs_norm, rhs_norm);
            const auto denominator = std::max(
                rhs_norm,
                static_cast<magnitude_type>(
                    residual_scaling.rhs_norm_floor));
            const auto scaled =
                denominator > magnitude_type{}
                    ? residual_norm / denominator
                    : residual_norm;
            if (!std::isfinite(scaled))
            {
                return std::numeric_limits<real_t>::infinity();
            }
            maximum = std::max(maximum, scaled);
        }
        if (maximum_rhs_norm != nullptr)
        {
            *maximum_rhs_norm =
                static_cast<real_t>(largest_rhs_norm);
        }
        return static_cast<real_t>(maximum);
    }

    /**
     * @brief Retain only initial-guess columns whose residual is no worse
     *        than the corresponding zero guess.
     *
     * A previous transient state is normally helpful, but a discontinuous
     * coefficient or source change can make it catastrophically worse than
     * zero.  Comparing global residual norms prevents that pathological case
     * without weakening RHS-scaled convergence.
     */
    void prepare_initial_guess(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution) const
    {
        auto& residual = residual_workspace(rhs);
        residual.update(
            scalar_type{1}, rhs, scalar_type{0});
        matrix->apply(
            solution, residual, Teuchos::NO_TRANS,
            scalar_type{-1}, scalar_type{1});

        Teuchos::Array<magnitude_type> rhs_norms(
            rhs.getNumVectors());
        Teuchos::Array<magnitude_type> residual_norms(
            residual.getNumVectors());
        rhs.norm2(rhs_norms());
        residual.norm2(residual_norms());

        for (std::size_t column = 0;
             column < rhs.getNumVectors();
             ++column)
        {
            if (std::isfinite(residual_norms[column])
                && residual_norms[column] <= rhs_norms[column])
            {
                continue;
            }
            auto values = solution.getDataNonConst(column);
            std::fill(values.begin(), values.end(), scalar_type{});
            // The rejected column now has a zero guess, so its residual is b.
            const auto source = rhs.getData(column);
            auto destination = residual.getDataNonConst(column);
            std::copy(source.begin(), source.end(), destination.begin());
        }
    }

    /**
     * @brief Return retained scratch storage for residual checks.
     *
     * Initial-residual preparation and the final true-residual check execute
     * around every solve. This owned multivector also supplies Belos' initial
     * residual, avoiding its duplicate operator application. Storage is rebuilt
     * when the map or column count changes.
     */
    multi_vector_type& residual_workspace(
        const multi_vector_type& rhs) const
    {
        if (d_residual_workspace.is_null()
            || d_residual_workspace->getNumVectors()
                   != rhs.getNumVectors()
            || !d_residual_workspace->getMap()->isSameAs(
                   *rhs.getMap()))
        {
            d_residual_workspace = Teuchos::rcp(
                new multi_vector_type(
                    rhs.getMap(), rhs.getNumVectors(), false));
        }
        return *d_residual_workspace;
    }

    friend struct detail::BelosLinearSolverTestAccess<Pack>;

    Teuchos::RCP<problem_type> d_problem;
    Teuchos::RCP<Teuchos::ParameterList> d_parameters;
    Teuchos::RCP<solver_type> d_solver;
    std::optional<LinearSolverBackend> d_backend;
    mutable Teuchos::RCP<multi_vector_type> d_residual_workspace;
    bool d_reported_accurate_residual = false;
    Teuchos::RCP<multi_vector_type> d_bicgstab_rhs_workspace;
    Teuchos::RCP<multi_vector_type> d_bicgstab_solution_workspace;
    Teuchos::RCP<const operator_type> d_preconditioner;
    Teuchos::RCP<const operator_type> d_preconditioner_operator;
    std::optional<LinearPreconditioner> d_preconditioner_kind;
    std::size_t d_preconditioner_setup_count = 0;
    using muelu_factory_type = MueLu::ParameterListInterpreter<scalar_type,
        typename Pack::local_ordinal_type, typename Pack::global_ordinal_type,
        typename Pack::node_type>;
    Teuchos::RCP<muelu_factory_type> d_muelu_factory;
    Teuchos::RCP<const typename Pack::map_type> d_muelu_map;
    std::size_t d_muelu_configuration_count = 0;
};

/**
 * @brief Solve a linear system using the selected Belos backend.
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
 * @brief Solve a linear system using the selected Belos backend.
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
 * @brief Solve a linear system using the selected Belos backend.
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
 * @brief Solve a linear system using the selected Belos backend.
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
