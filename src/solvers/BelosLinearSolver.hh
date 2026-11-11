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
#include <BelosSolverManager.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosTypes.hpp>
#include <Ifpack2_Preconditioner.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

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
    SymmetricGaussSeidel,
    /** Symmetric smoothed aggregation with SGS smoothing and an exact coarse solve. */
    MueLuSymmetric
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
        case LinearPreconditioner::MueLuSymmetric: return "MueLuSymmetric";
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
    if (value == "MueLuSymmetric")
        return LinearPreconditioner::MueLuSymmetric;
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

    void add(const LinearSolveStatistics& statistics);
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
     *        same preconditioning matrix object.
     *
     * This is opt-in because a matrix may be updated in place without its
     * identity changing. Enable reuse only when the preconditioning matrix's
     * numerical values remain unchanged between solves. A different matrix
     * object always triggers a rebuild.
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
class SIMPLEFLUID_SOLVERS_EXPORT BelosLinearSolver
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
    void reset();

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
     * @param matrix Linear operator to solve with.
     * @param rhs Right-hand-side multivector.
     * @param solution Initial guesses replaced by computed solutions.
     * @param options Linear solver and preconditioner controls.
     * @param residual_scaling Optional component scaling for residual checks.
     * @param preconditioning_matrix Optional CRS matrix used to build the preconditioner.
     * @return Convergence flag, iteration count, and achieved tolerance.
     */
    LinearSolveStatistics solve_with_statistics(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options = {},
        LinearResidualScaling residual_scaling = {},
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix = Teuchos::null)
    {
        return solve_impl(matrix, rhs, solution, options, residual_scaling, false,
            preconditioning_matrix);
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
        LinearResidualScaling residual_scaling = {},
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix = Teuchos::null)
    {
        return solve_impl(matrix, rhs, solution, options, residual_scaling, true,
            preconditioning_matrix);
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
     * @param matrix Linear operator to solve with.
     * @param rhs Right-hand-side vector.
     * @param solution Initial guess replaced by the computed solution.
     * @param options Linear solver and preconditioner controls.
     * @param residual_scaling Optional component scaling for residual checks.
     * @param preconditioning_matrix Optional CRS matrix used to build the preconditioner.
     * @return Convergence flag, iteration count, and achieved tolerance.
     */
    LinearSolveStatistics solve_with_statistics(
        const Teuchos::RCP<const operator_type>& matrix,
        const vector_type& rhs,
        vector_type& solution,
        const LinearSolverOptions& options = {},
        LinearResidualScaling residual_scaling = {},
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix = Teuchos::null);

    /** @brief Validate local option values before callers agree across ranks. */
    static inline void validate_options(const LinearSolverOptions& options);

private:
    LinearSolveStatistics solve_impl(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options,
        LinearResidualScaling residual_scaling,
        bool zero_initial_guess,
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix);

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
        bool zero_initial_guess,
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix);

    /**
     * Hand off the residual already prepared for the current RHS and guess.
     * Belos stores this owning RCP without copying its multivector. The scratch
     * storage remains alive for the complete solve; only after solve() returns
     * does the true-residual check overwrite it. Each solve and refinement
     * refreshes both residual pointers before setProblem(), including after a
     * RHS/map/column-count change. Refinement reuses the just-checked b-A*x.
     */
    bool set_problem_with_prepared_residual();

    /** @brief Report the true residual per RHS after an unsuccessful solve. */
    void report_failed_solve(
        const multi_vector_type& rhs,
        const LinearSolverOptions& options,
        LinearResidualScaling residual_scaling,
        Belos::ReturnType return_status,
        int iterations) const;

    static void validate_residual_scaling(
        const LinearResidualScaling& residual_scaling);

    Teuchos::RCP<solver_type> create_solver(LinearSolverBackend backend) const;

    bool has_compatible_maps(
        const Teuchos::RCP<const operator_type>& matrix) const;

    void configure(const LinearSolverOptions& options);

    void invalidate_preconditioner();

    /**
     * @brief Build, reuse, or remove the configured right preconditioner.
     * @throws std::invalid_argument If the selection is unknown or a
     *         matrix-based preconditioner is requested for a non-CRS operator.
     */
    void configure_preconditioner(
        const Teuchos::RCP<const operator_type>& matrix,
        const LinearSolverOptions& options,
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix);

    template<class Configure>
    Teuchos::RCP<const operator_type> create_ifpack2_preconditioner(
        const Teuchos::RCP<const matrix_type>& matrix,
        const std::string& factory_name,
        Configure&& configure_parameters) const;

    /** @brief Recompute the stored real CRS operator with extended accumulation.
     * This does not replace the diagonal, reconstruct fluxes, or change scaling.
     * Other operators/map layouts retain their existing apply-based residual.
     */
    bool accurate_crs_residual(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        const multi_vector_type& solution) const;

    real_t true_relative_residual(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        const multi_vector_type& solution,
        LinearResidualScaling residual_scaling = {},
        real_t* maximum_rhs_norm = nullptr) const;

    real_t residual_relative_norm(
        const multi_vector_type& rhs,
        LinearResidualScaling residual_scaling = {},
        real_t* maximum_rhs_norm = nullptr) const;

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
        multi_vector_type& solution) const;

    /**
     * @brief Return retained scratch storage for residual checks.
     *
     * Initial-residual preparation and the final true-residual check execute
     * around every solve. This owned multivector also supplies Belos' initial
     * residual, avoiding its duplicate operator application. Storage is rebuilt
     * when the map or column count changes.
     */
    multi_vector_type& residual_workspace(
        const multi_vector_type& rhs) const;

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
    std::optional<LinearPreconditioner> d_muelu_kind;
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

// Compiled for the default pack. A non-default pack must include
// "solvers/BelosLinearSolver.tcc" in the translation unit that instantiates it.
extern template class BelosLinearSolver<DefaultTpetraTypes>;

} // namespace SimpleFluid

#include "solvers/BelosLinearSolver.ipp"
