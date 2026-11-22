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
#include <vector>

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

/** @brief Execution strategy actually used for a linear solve. */
enum class MultiRhsExecutionMode
{
    SingleRhs,
    BatchedKrylov,
    ColumnSerial
};

/** @brief Convert a multi-RHS execution mode to its stable diagnostic name. */
inline std::string_view to_string(MultiRhsExecutionMode mode)
{
    switch (mode)
    {
        case MultiRhsExecutionMode::SingleRhs:     return "single_rhs";
        case MultiRhsExecutionMode::BatchedKrylov: return "batched_krylov";
        case MultiRhsExecutionMode::ColumnSerial:  return "column_serial";
    }

    throw std::invalid_argument("Unknown MultiRhsExecutionMode value.");
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

/** @brief Timing breakdown for one linear solve invocation. */
struct LinearSolveTimings
{
    double preconditioner_setup_seconds = 0.0;
    double solve_seconds = 0.0;
    double residual_check_seconds = 0.0;
    double pack_unpack_seconds = 0.0;
};

/** @brief Per-column convergence and residual metrics on the original (A, B). */
struct ColumnLinearSolveStatistics
{
    bool converged = false;
    bool skipped_zero = false;
    real_t achieved_tolerance = {};
    real_t true_residual_norm = {};
    real_t rhs_norm = {};
    int refinement_passes = 0;
    /**
     * @brief Krylov iterations attributable to this column.
     *
     * Sums each pass's scalar-equivalent work: for a batched pass this is the
     * iteration at which the column first met that pass's acceptance bound,
     * not the manager-wide iteration count that also serves slower columns.
     * Initial passes and GMRES defect-correction passes are recorded and
     * accumulated separately, so a column is never charged another column's
     * iterations.
     */
    int iterations = 0;
};

/** @brief Convergence statistics for one linear solve. */
struct LinearSolveStatistics
{
    bool converged = false;
    /** Includes refinement; column-serial CG/BiCGStab sum independent columns. */
    int iterations = 0;
    real_t achieved_tolerance = {};
    real_t rhs_norm = {};
    MultiRhsExecutionMode execution_mode = MultiRhsExecutionMode::SingleRhs;
    int batch_krylov_iterations = 0;
    int column_serial_iterations = 0;
    std::size_t preconditioner_builds = 0;
    std::size_t operator_apply_calls = 0;
    std::size_t operator_column_applies = 0;
    std::size_t max_batch_width = 0;
    LinearSolveTimings timings = {};
    std::vector<ColumnLinearSolveStatistics> column_statistics = {};
};

/** @brief Per-column residual acceptance override for multi-RHS solves. */
struct ColumnResidualPolicy
{
    std::optional<real_t> tolerance = {};
    std::optional<real_t> absolute_tolerance = {};
    std::optional<real_t> rhs_norm_floor = {};
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
    /**
     * @brief Absolute residual acceptance bound for the true residual norm.
     *
     * Zero (the default) keeps the pure RHS-relative stopping test. When
     * positive, the solve also accepts ||b - A*x|| <= absolute_tolerance
     * and a zero initial guess with ||b|| <= absolute_tolerance returns
     * the zero solution immediately without entering Krylov. This gives
     * cold-start and compatibility-projected systems a safe short-circuit
     * that never overwrites a warm solution: the zero-guess contract is
     * the incremental/correction form only.
     */
    real_t absolute_tolerance = {};
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
    /**
     * @brief Maximum active RHS columns solved together per Krylov batch.
     *
     * Zero (default) solves all active columns in a single batch for
     * PseudoBlockGMRES. PseudoBlockCG is currently restricted to column-serial
     * execution so each column keeps an independent, exact iteration budget;
     * BiCGStab is likewise column-serial. Positive values chunk active GMRES
     * columns into contiguous batches of at most @p max_batch_width.
     */
    int max_batch_width = 0;
};

/** @brief Options for multi-RHS solves with per-column policies and active masks. */
struct MultiRhsSolveOptions
{
    LinearSolverOptions solver = {};
    LinearResidualScaling default_residual_scaling = {};
    std::vector<ColumnResidualPolicy> column_policies = {};
    /**
     * @brief Globally consistent active-column mask (empty = all columns active).
     *
     * Inactive columns are collectively skipped without entering Krylov and
     * their solution columns are explicitly zeroed.
     */
    std::vector<bool> active_columns = {};
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

    /**
     * @brief Solve a multi-RHS system with optional per-column policies,
     *        active-column masking, and batch-width chunking.
     */
    LinearSolveStatistics solve_multi_rhs_with_statistics(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const MultiRhsSolveOptions& options = {},
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix = Teuchos::null)
    {
        return solve_multi_rhs_impl(
            matrix, rhs, solution, options, false, preconditioning_matrix);
    }

    /**
     * @brief Solve a multi-RHS system from an explicit zero initial guess.
     */
    LinearSolveStatistics solve_multi_rhs_from_zero_with_statistics(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const MultiRhsSolveOptions& options = {},
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix = Teuchos::null)
    {
        return solve_multi_rhs_impl(
            matrix, rhs, solution, options, true, preconditioning_matrix);
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

    LinearSolveStatistics solve_multi_rhs_impl(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const MultiRhsSolveOptions& options,
        bool zero_initial_guess,
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix);

    LinearSolveStatistics solve_contiguous_batch(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options,
        LinearResidualScaling residual_scaling,
        const std::vector<ColumnResidualPolicy>& column_policies,
        bool zero_initial_guess,
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix);

    /**
     * @brief Solve active columns in independent owned storage.
     *
     * Trilinos 17.2 Tpetra single-column subviews of a nonconstant-stride
     * multivector use whichVectors[0] + index rather than whichVectors[index].
     * BiCGStab's per-column axpy reaches that path after RHS deflation; e.g.
     * surviving columns [0,2] cause an update intended for 2 to overwrite 1.
     * Independent owning columns avoid nested deflated views entirely.
     *
     * PseudoBlockCG uses this path as an explicit interim because its installed
     * manager exposes no per-iteration status hook. Column-serial execution
     * keeps each column's Krylov work and repair budget independent; the
     * reported execution mode is ColumnSerial, not BatchedKrylov.
     *
     * Each RHS retains its tolerance and iteration limit. A preconditioner
     * can safely be reused within this call because its matrix is unchanged.
     */
    LinearSolveStatistics solve_column_serial(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs, multi_vector_type& solution,
        const LinearSolverOptions& options,
        LinearResidualScaling residual_scaling,
        const std::vector<ColumnResidualPolicy>& column_policies,
        const std::vector<std::size_t>& active_indices,
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

    void configure(const LinearSolverOptions& options, std::size_t batch_columns);

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

    Teuchos::RCP<const operator_type> d_raw_operator;
    Teuchos::RCP<problem_type> d_problem;
    Teuchos::RCP<Teuchos::ParameterList> d_parameters;
    Teuchos::RCP<solver_type> d_solver;
    std::optional<LinearSolverBackend> d_backend;
    mutable Teuchos::RCP<multi_vector_type> d_residual_workspace;
    bool d_reported_accurate_residual = false;
    // Retained across solver reuse: a Belos manager keeps the debug status test
    // installed at its first solve, so the recorder must outlive that solve and
    // keep pointing at valid per-pass scratch.
    std::vector<int> d_column_convergence_iterations;
    std::vector<magnitude_type> d_column_convergence_bounds;
    // Local column -> caller column for the current pass. Initial solves are the
    // identity; packed defect-correction solves map their narrow block back to
    // the original batch columns.
    std::vector<int> d_column_convergence_columns;
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
