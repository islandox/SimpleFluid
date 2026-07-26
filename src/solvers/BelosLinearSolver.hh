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
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <stdexcept>
#include <string_view>
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
    if (value == "cg" || value == "CG"
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

/** @brief Available right-preconditioning strategies for Belos solves. */
enum class LinearPreconditioner
{
    None,
    MueLu,
    Jacobi,
    ILU0,
    ILUT
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
    throw std::invalid_argument(
        "Unknown linear preconditioner '" + std::string(value) + "'.");
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
    /**
     * CG is intended only when both the operator and configured
     * preconditioner preserve a symmetric positive-definite system. ILU0 and
     * ILUT are not generally CG-compatible. GMRES and BiCGStab support the
     * nonsymmetric transport and gauge-fixed pressure operators assembled by
     * the general FVM path.
     */
    LinearSolverBackend backend = LinearSolverBackend::Gmres;
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
        const LinearSolverOptions& options = {})
    {
        validate_options(options);
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
            if (!d_problem->setProblem())
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
            if (!d_problem->setProblem())
            {
                return {};
            }
        }

        static_cast<void>(d_solver->solve());
        const auto achieved_tolerance =
            true_relative_residual(matrix, rhs, solution);
        return {
            std::isfinite(achieved_tolerance)
                && achieved_tolerance <= options.tolerance,
            d_solver->getNumIters(),
            achieved_tolerance};
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
        const LinearSolverOptions& options = {})
    {
        auto x = Teuchos::rcp_implicit_cast<multi_vector_type>(
            Teuchos::rcpFromRef(solution));
        auto b = Teuchos::rcp_implicit_cast<const multi_vector_type>(
            Teuchos::rcpFromRef(rhs));
        return solve_with_statistics(matrix, *b, *x, options);
    }

private:
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
                Teuchos::RCP<operator_type> mutable_operator =
                    mutable_matrix;
                d_preconditioner =
                    MueLu::CreateTpetraPreconditioner(
                        mutable_operator, parameters);
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

    real_t true_relative_residual(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        const multi_vector_type& solution) const
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

        magnitude_type maximum{};
        for (std::size_t column = 0;
             column < rhs.getNumVectors();
             ++column)
        {
            const auto rhs_norm = rhs_norms[column];
            const auto residual_norm = residual_norms[column];
            if (!std::isfinite(rhs_norm)
                || !std::isfinite(residual_norm))
            {
                return std::numeric_limits<real_t>::infinity();
            }
            const auto scaled =
                rhs_norm > magnitude_type{}
                    ? residual_norm / rhs_norm
                    : residual_norm;
            if (!std::isfinite(scaled))
            {
                return std::numeric_limits<real_t>::infinity();
            }
            maximum = std::max(maximum, scaled);
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
        if (matrix.is_null())
        {
            throw std::invalid_argument(
                "BelosLinearSolver requires a non-null operator.");
        }

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
        }
    }

    /**
     * @brief Return retained scratch storage for residual checks.
     *
     * Warm-start screening and the final true-residual check execute around
     * every solve. Retaining this multivector avoids adding a heap allocation
     * to each equation solve while still rebuilding it when the map or column
     * count changes.
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
    Teuchos::RCP<const operator_type> d_preconditioner;
    Teuchos::RCP<const operator_type> d_preconditioner_operator;
    std::optional<LinearPreconditioner> d_preconditioner_kind;
    std::size_t d_preconditioner_setup_count = 0;
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
