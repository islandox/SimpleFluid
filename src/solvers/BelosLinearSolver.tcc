/**
 * @file BelosLinearSolver.tcc
 * @brief Out-of-line template implementations for BelosLinearSolver.
 */
#pragma once

#include "solvers/BelosLinearSolver.hh"
#include "solvers/DICPreconditioner.hh"
#include "solvers/ScaledLinearOperator.hh"

#include <BelosBiCGStabSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <Ifpack2_Factory.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace SimpleFluid
{

namespace
{

template<TpetraTypePack Pack>
class CountingLinearOperator final : public Pack::operator_type
{
public:
    using scalar_type = typename Pack::scalar_type;
    using multi_vector_type = typename Pack::multi_vector_type;
    using operator_type = typename Pack::operator_type;
    using map_type = typename Pack::map_type;

    explicit CountingLinearOperator(Teuchos::RCP<const operator_type> target)
        : d_target(std::move(target))
    {
    }

    void set_target(Teuchos::RCP<const operator_type> target)
    {
        d_target = std::move(target);
    }

    Teuchos::RCP<const map_type> getDomainMap() const override
    {
        return d_target->getDomainMap();
    }

    Teuchos::RCP<const map_type> getRangeMap() const override
    {
        return d_target->getRangeMap();
    }

    bool hasTransposeApply() const override
    {
        return d_target->hasTransposeApply();
    }

    void apply(
        const multi_vector_type& input,
        multi_vector_type& output,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
        scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const override
    {
        ++d_calls;
        const auto width = static_cast<std::size_t>(input.getNumVectors());
        d_column_applies += width;
        d_max_batch_width = std::max(d_max_batch_width, width);
        d_target->apply(input, output, mode, alpha, beta);
    }

    std::size_t calls() const noexcept { return d_calls; }
    std::size_t column_applies() const noexcept { return d_column_applies; }
    std::size_t max_batch_width() const noexcept { return d_max_batch_width; }

private:
    Teuchos::RCP<const operator_type> d_target;
    mutable std::size_t d_calls = 0;
    mutable std::size_t d_column_applies = 0;
    mutable std::size_t d_max_batch_width = 0;
};

double elapsed_seconds(const std::chrono::steady_clock::time_point start) noexcept
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
}

} // namespace

/**
* @brief  Solve a linear system with the configured Belos solver and preconditioner.
* @param matrix Tpetra operator for the linear system.
* @param rhs Right-hand side vector.
* @param solution On input, initial guess; on output, the computed solution.
* @param options Linear solver and preconditioner controls.
* @param residual_scaling Optional component scaling for residual checks.
* @param zero_initial_guess If true, replace the initial guess with zero before solving.
* @param preconditioning_matrix Optional CRS matrix used to build the preconditioner.
* @return Convergence flag, iteration count, and achieved tolerance.
* @throws std::invalid_argument If solver options are invalid or a matrix-based
*  preconditioner is requested for a non-CRS operator.
*/
template <TpetraTypePack Pack>
LinearSolveStatistics BelosLinearSolver<Pack>::solve_impl(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options,
        LinearResidualScaling residual_scaling,
        bool zero_initial_guess,
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix)
{
    return solve_multi_rhs_impl(
        matrix,
        rhs,
        solution,
        MultiRhsSolveOptions{options, residual_scaling, {}, {}},
        zero_initial_guess,
        preconditioning_matrix);
}

template <TpetraTypePack Pack>
LinearSolveStatistics BelosLinearSolver<Pack>::solve_multi_rhs_impl(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const MultiRhsSolveOptions& options,
        bool zero_initial_guess,
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix)
{
    validate_options(options.solver);
    validate_residual_scaling(options.default_residual_scaling);
    if (matrix.is_null())
    {
        throw std::invalid_argument(
            "BelosLinearSolver requires a non-null operator.");
    }
    const std::size_t num_cols = rhs.getNumVectors();
    if (num_cols == 0)
    {
        throw std::invalid_argument(
            "BelosLinearSolver requires at least one RHS column.");
    }
    if (num_cols != solution.getNumVectors()
        || !matrix->getDomainMap()->isSameAs(*solution.getMap())
        || !matrix->getRangeMap()->isSameAs(*rhs.getMap()))
    {
        throw std::invalid_argument(
            "BelosLinearSolver requires matching operator maps and RHS/solution vector counts.");
    }
    if (!preconditioning_matrix.is_null()
        && options.solver.preconditioner != LinearPreconditioner::None
        && (!preconditioning_matrix->getDomainMap()->isSameAs(*matrix->getDomainMap())
            || !preconditioning_matrix->getRangeMap()->isSameAs(*matrix->getRangeMap())
            || !preconditioning_matrix->getDomainMap()->isSameAs(*solution.getMap())
            || !preconditioning_matrix->getRangeMap()->isSameAs(*rhs.getMap())))
    {
        throw std::invalid_argument(
            "BelosLinearSolver requires matching operator, preconditioning matrix, and vector maps.");
    }
    if (!options.column_policies.empty() && options.column_policies.size() != num_cols)
    {
        throw std::invalid_argument(
            "BelosLinearSolver requires column_policies to be empty or match RHS column count.");
    }
    for (const auto& policy : options.column_policies)
    {
        if (policy.tolerance.has_value()
            && (!std::isfinite(*policy.tolerance) || *policy.tolerance <= real_t{}))
        {
            throw std::invalid_argument(
                "BelosLinearSolver requires positive finite per-column tolerance.");
        }
        if (policy.absolute_tolerance.has_value()
            && (!std::isfinite(*policy.absolute_tolerance) || *policy.absolute_tolerance < real_t{}))
        {
            throw std::invalid_argument(
                "BelosLinearSolver requires non-negative finite per-column absolute tolerance.");
        }
        if (policy.rhs_norm_floor.has_value()
            && (!std::isfinite(*policy.rhs_norm_floor) || *policy.rhs_norm_floor < real_t{}))
        {
            throw std::invalid_argument(
                "BelosLinearSolver requires non-negative finite per-column RHS norm floor.");
        }
    }

    const auto comm = rhs.getMap()->getComm();
    const int has_mask = options.active_columns.empty() ? 0 : 1;
    std::vector<int> local_mask_state(num_cols + 2, 0);
    local_mask_state[0] = has_mask;
    local_mask_state[1] = static_cast<int>(options.active_columns.size());
    if (has_mask != 0)
    {
        if (options.active_columns.size() != num_cols)
        {
            local_mask_state[0] = 2;
        }
        else
        {
            for (std::size_t c = 0; c < num_cols; ++c)
            {
                local_mask_state[2 + c] = options.active_columns[c] ? 1 : 0;
            }
        }
    }
    if (comm->getSize() > 1)
    {
        std::vector<int> min_state(local_mask_state.size(), 0);
        std::vector<int> max_state(local_mask_state.size(), 0);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN,
            static_cast<int>(local_mask_state.size()), local_mask_state.data(), min_state.data());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX,
            static_cast<int>(local_mask_state.size()), local_mask_state.data(), max_state.data());
        if (max_state[0] == 2 || min_state != max_state)
        {
            throw std::invalid_argument(
                "BelosLinearSolver requires every rank to supply the exact same active_columns mask.");
        }
    }
    else if (local_mask_state[0] == 2)
    {
        throw std::invalid_argument(
            "BelosLinearSolver requires active_columns to be empty or match RHS column count.");
    }

    std::vector<std::size_t> active_indices;
    active_indices.reserve(num_cols);
    std::vector<ColumnLinearSolveStatistics> all_column_stats(num_cols);
    for (std::size_t c = 0; c < num_cols; ++c)
    {
        const bool active = options.active_columns.empty() || options.active_columns[c];
        if (active)
        {
            active_indices.push_back(c);
        }
        else
        {
            auto sol_col = solution.getDataNonConst(c);
            std::fill(sol_col.begin(), sol_col.end(), scalar_type{});
            all_column_stats[c] = ColumnLinearSolveStatistics{
                true, true, real_t{}, real_t{}, real_t{}, 0};
        }
    }

    if (!options.solver.reuse_preconditioner)
    {
        invalidate_preconditioner();
    }

    if (active_indices.empty())
    {
        LinearSolveStatistics skipped;
        skipped.converged = true;
        skipped.execution_mode =
            (options.solver.backend == LinearSolverBackend::BiCGStab && num_cols > 1)
                ? MultiRhsExecutionMode::ColumnSerial
                : MultiRhsExecutionMode::SingleRhs;
        skipped.column_statistics = std::move(all_column_stats);
        return skipped;
    }

    if (options.solver.backend == LinearSolverBackend::BiCGStab && num_cols > 1)
    {
        return solve_bicgstab_columns(
            matrix, rhs, solution, options.solver, options.default_residual_scaling,
            options.column_policies, active_indices, zero_initial_guess,
            preconditioning_matrix);
    }

    const std::size_t active_count = active_indices.size();
    const std::size_t chunk_width = options.solver.max_batch_width > 0
        ? std::min(active_count, static_cast<std::size_t>(options.solver.max_batch_width))
        : active_count;

    if (active_count == num_cols && chunk_width == num_cols)
    {
        auto stats = solve_contiguous_batch(
            matrix, rhs, solution, options.solver, options.default_residual_scaling,
            options.column_policies, zero_initial_guess, preconditioning_matrix);
        if (num_cols == 1)
        {
            stats.execution_mode = MultiRhsExecutionMode::SingleRhs;
        }
        else
        {
            stats.execution_mode = MultiRhsExecutionMode::BatchedKrylov;
            stats.batch_krylov_iterations = stats.iterations;
            stats.column_serial_iterations = 0;
        }
        return stats;
    }

    LinearSolveStatistics combined;
    combined.converged = true;
    combined.execution_mode = num_cols == 1
        ? MultiRhsExecutionMode::SingleRhs
        : (chunk_width > 1 ? MultiRhsExecutionMode::BatchedKrylov
                           : MultiRhsExecutionMode::ColumnSerial);

    auto chunk_options = options.solver;
    Teuchos::RCP<multi_vector_type> rhs_chunk;
    Teuchos::RCP<multi_vector_type> sol_chunk;
    for (std::size_t offset = 0; offset < active_count; offset += chunk_width)
    {
        const std::size_t this_width = std::min(chunk_width, active_count - offset);
        const auto pack_start = std::chrono::steady_clock::now();
        if (rhs_chunk.is_null() || rhs_chunk->getNumVectors() != this_width)
        {
            rhs_chunk = Teuchos::rcp(new multi_vector_type(rhs.getMap(), this_width, false));
            sol_chunk = Teuchos::rcp(new multi_vector_type(solution.getMap(), this_width, false));
        }
        std::vector<ColumnResidualPolicy> chunk_policies;
        if (!options.column_policies.empty())
        {
            chunk_policies.reserve(this_width);
        }
        for (std::size_t local_col = 0; local_col < this_width; ++local_col)
        {
            const std::size_t global_col = active_indices[offset + local_col];
            const auto rhs_src = rhs.getData(global_col);
            auto rhs_dst = rhs_chunk->getDataNonConst(local_col);
            std::copy(rhs_src.begin(), rhs_src.end(), rhs_dst.begin());
            if (!zero_initial_guess)
            {
                const auto sol_src = solution.getData(global_col);
                auto sol_dst = sol_chunk->getDataNonConst(local_col);
                std::copy(sol_src.begin(), sol_src.end(), sol_dst.begin());
            }
            if (!options.column_policies.empty())
            {
                chunk_policies.push_back(options.column_policies[global_col]);
            }
        }
        combined.timings.pack_unpack_seconds += elapsed_seconds(pack_start);

        const auto chunk_stats = solve_contiguous_batch(
            matrix, *rhs_chunk, *sol_chunk, chunk_options,
            options.default_residual_scaling, chunk_policies,
            zero_initial_guess, preconditioning_matrix);

        const auto unpack_start = std::chrono::steady_clock::now();
        for (std::size_t local_col = 0; local_col < this_width; ++local_col)
        {
            const std::size_t global_col = active_indices[offset + local_col];
            const auto sol_src = sol_chunk->getData(local_col);
            auto sol_dst = solution.getDataNonConst(global_col);
            std::copy(sol_src.begin(), sol_src.end(), sol_dst.begin());
            if (local_col < chunk_stats.column_statistics.size())
            {
                all_column_stats[global_col] = chunk_stats.column_statistics[local_col];
            }
        }
        combined.timings.pack_unpack_seconds += elapsed_seconds(unpack_start);

        combined.converged = combined.converged && chunk_stats.converged;
        combined.iterations += chunk_stats.iterations;
        if (combined.execution_mode == MultiRhsExecutionMode::BatchedKrylov)
        {
            combined.batch_krylov_iterations += chunk_stats.iterations;
        }
        else if (combined.execution_mode == MultiRhsExecutionMode::ColumnSerial)
        {
            combined.column_serial_iterations += chunk_stats.iterations;
        }
        combined.achieved_tolerance = std::max(
            combined.achieved_tolerance, chunk_stats.achieved_tolerance);
        combined.rhs_norm = std::max(combined.rhs_norm, chunk_stats.rhs_norm);
        combined.operator_apply_calls += chunk_stats.operator_apply_calls;
        combined.operator_column_applies += chunk_stats.operator_column_applies;
        combined.max_batch_width = std::max(
            combined.max_batch_width, chunk_stats.max_batch_width);
        combined.timings.preconditioner_setup_seconds +=
            chunk_stats.timings.preconditioner_setup_seconds;
        combined.timings.solve_seconds += chunk_stats.timings.solve_seconds;
        combined.timings.residual_check_seconds +=
            chunk_stats.timings.residual_check_seconds;
        if (options.solver.preconditioner == LinearPreconditioner::None
            || !d_preconditioner.is_null())
        {
            chunk_options.reuse_preconditioner = true;
        }
    }

    combined.column_statistics = std::move(all_column_stats);
    return combined;
}

template <TpetraTypePack Pack>
LinearSolveStatistics BelosLinearSolver<Pack>::solve_contiguous_batch(
        const Teuchos::RCP<const operator_type>& matrix,
        const multi_vector_type& rhs,
        multi_vector_type& solution,
        const LinearSolverOptions& options,
        LinearResidualScaling residual_scaling,
        const std::vector<ColumnResidualPolicy>& column_policies,
        bool zero_initial_guess,
        const Teuchos::RCP<const matrix_type>& preconditioning_matrix)
{
    const std::size_t batch_cols = rhs.getNumVectors();
    auto effective_tol = [&](std::size_t col) -> real_t
    {
        return (!column_policies.empty() && column_policies[col].tolerance.has_value())
            ? *column_policies[col].tolerance
            : options.tolerance;
    };
    auto effective_abs_tol = [&](std::size_t col) -> magnitude_type
    {
        return static_cast<magnitude_type>(
            (!column_policies.empty() && column_policies[col].absolute_tolerance.has_value())
                ? *column_policies[col].absolute_tolerance
                : options.absolute_tolerance);
    };
    auto effective_floor = [&](std::size_t col) -> magnitude_type
    {
        return static_cast<magnitude_type>(
            (!column_policies.empty() && column_policies[col].rhs_norm_floor.has_value())
                ? *column_policies[col].rhs_norm_floor
                : residual_scaling.rhs_norm_floor);
    };

    real_t manager_tolerance = options.tolerance;
    for (std::size_t col = 0; col < batch_cols; ++col)
    {
        manager_tolerance = std::min(manager_tolerance, effective_tol(col));
    }

    // Combined per-column acceptance: every RHS column must independently
    // satisfy its own absolute bound or RHS-relative criterion on (A_g, B_g),
    // and any nonfinite column norm rejects the solve.
    const auto columns_accept_combined =
        [&](const Teuchos::ArrayView<const magnitude_type> rhs_norms,
            const Teuchos::ArrayView<const magnitude_type> residual_norms)
    {
        bool converged = true;
        for (std::size_t column = 0; column < rhs_norms.size(); ++column)
        {
            if (!std::isfinite(rhs_norms[column])
                || !std::isfinite(residual_norms[column]))
            {
                return false;
            }
            const auto bound = std::max(
                effective_abs_tol(column),
                static_cast<magnitude_type>(effective_tol(column))
                    * std::max(rhs_norms[column], effective_floor(column)));
            converged = converged && residual_norms[column] <= bound;
        }
        return converged;
    };

    auto counting_op = Teuchos::rcp(new CountingLinearOperator<Pack>(matrix));
    auto counting_op_base = Teuchos::rcp_implicit_cast<const operator_type>(counting_op);
    LinearSolveTimings timings;

    if (zero_initial_guess)
    {
        solution.putScalar(scalar_type{});
        residual_workspace(rhs).update(scalar_type{1}, rhs, scalar_type{});
        bool all_have_abs_tol = true;
        for (std::size_t col = 0; col < batch_cols; ++col)
        {
            if (effective_abs_tol(col) <= magnitude_type{})
            {
                all_have_abs_tol = false;
                break;
            }
        }
        if (all_have_abs_tol)
        {
            const auto check_start = std::chrono::steady_clock::now();
            Teuchos::Array<magnitude_type> zero_rhs_norms(batch_cols);
            rhs.norm2(zero_rhs_norms());
            magnitude_type zero_rhs_norm{};
            bool all_zero_accepted = true;
            for (std::size_t column = 0; column < batch_cols; ++column)
            {
                if (!std::isfinite(zero_rhs_norms[column])
                    || zero_rhs_norms[column] > effective_abs_tol(column))
                {
                    all_zero_accepted = false;
                    break;
                }
                zero_rhs_norm = std::max(zero_rhs_norm, zero_rhs_norms[column]);
            }
            timings.residual_check_seconds += elapsed_seconds(check_start);
            if (all_zero_accepted)
            {
                LinearSolveStatistics short_circuit;
                short_circuit.converged = true;
                short_circuit.iterations = 0;
                short_circuit.achieved_tolerance = real_t{};
                short_circuit.rhs_norm = static_cast<real_t>(zero_rhs_norm);
                short_circuit.timings = timings;
                short_circuit.column_statistics.reserve(batch_cols);
                for (std::size_t col = 0; col < batch_cols; ++col)
                {
                    short_circuit.column_statistics.push_back(ColumnLinearSolveStatistics{
                        true,
                        true,
                        real_t{},
                        static_cast<real_t>(zero_rhs_norms[col]),
                        static_cast<real_t>(zero_rhs_norms[col]),
                        0});
                }
                return short_circuit;
            }
        }
    }
    else
    {
        const auto prep_start = std::chrono::steady_clock::now();
        prepare_initial_guess(counting_op_base, rhs, solution);
        timings.residual_check_seconds += elapsed_seconds(prep_start);
    }

    auto x = Teuchos::rcpFromRef(solution);
    auto b = Teuchos::rcpFromRef(rhs);

    auto belos_options = options;
    belos_options.tolerance = manager_tolerance;

    const bool rebuild_solver =
        !has_compatible_maps(matrix)
        || !d_backend.has_value()
        || *d_backend != options.backend;
    if (rebuild_solver)
    {
        invalidate_preconditioner();
        d_solver = Teuchos::null;
        d_backend.reset();
        d_raw_operator = matrix;
        d_problem = Teuchos::rcp(
            new problem_type(counting_op_base, x, b));
        configure(belos_options, batch_cols);
        const auto prec_start = std::chrono::steady_clock::now();
        configure_preconditioner(matrix, options, preconditioning_matrix);
        timings.preconditioner_setup_seconds += elapsed_seconds(prec_start);
        if (!set_problem_with_prepared_residual())
        {
            return {};
        }
        d_solver = create_solver(options.backend);
        d_backend = options.backend;
    }
    else
    {
        d_raw_operator = matrix;
        d_problem->setOperator(counting_op_base);
        d_problem->setLHS(x);
        d_problem->setRHS(b);
        configure(belos_options, batch_cols);
        d_solver->setParameters(d_parameters);
        const auto prec_start = std::chrono::steady_clock::now();
        configure_preconditioner(matrix, options, preconditioning_matrix);
        timings.preconditioner_setup_seconds += elapsed_seconds(prec_start);
        if (!set_problem_with_prepared_residual())
        {
            return {};
        }
    }

    const auto solve_start = std::chrono::steady_clock::now();
    auto return_status = d_solver->solve();
    timings.solve_seconds += elapsed_seconds(solve_start);
    int iterations = d_solver->getNumIters();
    real_t rhs_norm{};
    bool accurate_residual = false;
    const auto evaluate_column_norms = [&](
        Teuchos::Array<magnitude_type>& b_norms,
        Teuchos::Array<magnitude_type>& r_norms,
        real_t* out_max_rhs,
        real_t* out_max_rel,
        real_t* out_unaccepted_ratio,
        bool* out_unaccepted_within_gap,
        bool* out_all_accepted)
    {
        rhs.norm2(b_norms());
        residual_workspace(rhs).norm2(r_norms());
        magnitude_type max_b{};
        magnitude_type max_rel{};
        magnitude_type worst_ratio{};
        bool all_accepted = true;
        bool within_gap = true;
        bool any_nonfinite = false;
        for (std::size_t col = 0; col < batch_cols; ++col)
        {
            if (!std::isfinite(b_norms[col]) || !std::isfinite(r_norms[col]))
            {
                any_nonfinite = true;
                all_accepted = false;
                within_gap = false;
                continue;
            }
            max_b = std::max(max_b, b_norms[col]);
            const auto denom = std::max(b_norms[col], effective_floor(col));
            const auto rel = denom > magnitude_type{} ? r_norms[col] / denom : r_norms[col];
            if (!std::isfinite(rel))
            {
                any_nonfinite = true;
                all_accepted = false;
                within_gap = false;
                continue;
            }
            max_rel = std::max(max_rel, rel);
            const auto bound = std::max(
                effective_abs_tol(col),
                static_cast<magnitude_type>(effective_tol(col)) * denom);
            const bool accepted = r_norms[col] <= bound;
            if (!accepted)
            {
                all_accepted = false;
                const auto col_tol = static_cast<magnitude_type>(effective_tol(col));
                const auto ratio = rel / col_tol;
                worst_ratio = std::max(worst_ratio, ratio);
                const auto col_gap = options.backend == LinearSolverBackend::Gmres
                    ? std::max(magnitude_type{8} * col_tol,
                        static_cast<magnitude_type>(
                            std::sqrt(std::numeric_limits<magnitude_type>::epsilon())))
                    : magnitude_type{2} * col_tol;
                if (rel > col_gap)
                {
                    within_gap = false;
                }
            }
        }
        if (out_max_rhs != nullptr)
        {
            *out_max_rhs = any_nonfinite
                ? std::numeric_limits<real_t>::infinity()
                : static_cast<real_t>(max_b);
        }
        if (out_max_rel != nullptr)
        {
            *out_max_rel = any_nonfinite
                ? std::numeric_limits<real_t>::infinity()
                : static_cast<real_t>(max_rel);
        }
        if (out_unaccepted_ratio != nullptr)
        {
            *out_unaccepted_ratio = any_nonfinite
                ? std::numeric_limits<real_t>::infinity()
                : static_cast<real_t>(worst_ratio);
        }
        if (out_unaccepted_within_gap != nullptr)
        {
            *out_unaccepted_within_gap = !any_nonfinite && within_gap;
        }
        if (out_all_accepted != nullptr)
        {
            *out_all_accepted = !any_nonfinite && all_accepted;
        }
    };

    Teuchos::Array<magnitude_type> precheck_rhs_norms(batch_cols);
    Teuchos::Array<magnitude_type> precheck_residual_norms(batch_cols);
    real_t unaccepted_ratio{};
    bool unaccepted_within_gap = false;
    bool per_column_accepted = false;

    const auto checked_residual = [&] {
        const auto check_start = std::chrono::steady_clock::now();
        real_t max_rel{};
        if (accurate_residual && accurate_crs_residual(matrix, rhs, solution))
        {
            evaluate_column_norms(
                precheck_rhs_norms, precheck_residual_norms,
                &rhs_norm, &max_rel, &unaccepted_ratio,
                &unaccepted_within_gap, &per_column_accepted);
            timings.residual_check_seconds += elapsed_seconds(check_start);
            return max_rel;
        }
        (void)true_relative_residual(
            counting_op_base, rhs, solution, residual_scaling, &rhs_norm);
        evaluate_column_norms(
            precheck_rhs_norms, precheck_residual_norms,
            &rhs_norm, &max_rel, &unaccepted_ratio,
            &unaccepted_within_gap, &per_column_accepted);
        if (options.backend == LinearSolverBackend::Gmres
            && !per_column_accepted && unaccepted_within_gap
            && unaccepted_ratio > real_t{1}
            && accurate_crs_residual(matrix, rhs, solution))
        {
            accurate_residual = true;
            const auto standard = max_rel;
            evaluate_column_norms(
                precheck_rhs_norms, precheck_residual_norms,
                &rhs_norm, &max_rel, &unaccepted_ratio,
                &unaccepted_within_gap, &per_column_accepted);
            if (!d_reported_accurate_residual)
            {
                if (rhs.getMap()->getComm()->getRank() == 0 && (options.verbosity & Belos::Warnings))
                {
                    std::ostringstream message;
                    message.precision(std::numeric_limits<real_t>::max_digits10);
                    message << "BelosLinearSolver CRS residual precision check: standard=" << standard
                            << " compensated=" << max_rel << " tolerance=" << manager_tolerance << '\n';
                    std::cerr << message.str();
                }
                d_reported_accurate_residual = true;
            }
            timings.residual_check_seconds += elapsed_seconds(check_start);
            return max_rel;
        }
        timings.residual_check_seconds += elapsed_seconds(check_start);
        return max_rel;
    };

    auto achieved_tolerance = checked_residual();
    bool residual_workspace_current = true;

    auto refinement_options = belos_options;
    const int maximum_refinements = options.backend == LinearSolverBackend::Gmres ? 4 : 2;
    std::array<real_t, 5> refinement_residuals{achieved_tolerance};
    size_t refinement_count = 1;
    std::vector<int> column_refinements(batch_cols, 0);

    auto column_needs_refinement = [&](std::size_t col)
    {
        const auto bound = std::max(
            effective_abs_tol(col),
            static_cast<magnitude_type>(effective_tol(col))
                * std::max(precheck_rhs_norms[col], effective_floor(col)));
        return !std::isfinite(precheck_rhs_norms[col])
            || !std::isfinite(precheck_residual_norms[col])
            || precheck_residual_norms[col] > bound;
    };

    for (int refinement = 0;
            refinement < maximum_refinements
                && !per_column_accepted
                && (return_status == Belos::Converged
                    || (options.backend == LinearSolverBackend::Gmres
                        && d_solver->isLOADetected()))
                && std::isfinite(unaccepted_ratio)
                && unaccepted_ratio > real_t{1}
                && unaccepted_within_gap
                && iterations < options.max_iterations;
            ++refinement)
    {
        for (std::size_t col = 0; col < batch_cols; ++col)
        {
            if (column_needs_refinement(col))
            {
                ++column_refinements[col];
            }
        }
        refinement_options.max_iterations =
            options.max_iterations - iterations;
        if (options.backend == LinearSolverBackend::Gmres)
        {
            auto defect = Teuchos::rcp(new multi_vector_type(rhs.getMap(), batch_cols));
            defect->update(scalar_type{1}, residual_workspace(rhs), scalar_type{});
            auto correction = Teuchos::rcp(new multi_vector_type(solution.getMap(), batch_cols));
            correction->putScalar(scalar_type{});
            refinement_options.tolerance = std::min(real_t{0.1},
                real_t{0.5} / std::max(unaccepted_ratio, real_t{1}));
            const auto restore_original_vectors = [&] {
                d_problem->setLHS(x);
                d_problem->setRHS(b);
                d_problem->setInitResVec(d_residual_workspace);
                d_problem->setInitPrecResVec(Teuchos::null);
                (void)d_problem->setProblem();
            };
            try
            {
                configure(refinement_options, batch_cols);
                d_solver->setParameters(d_parameters);
                d_problem->setLHS(correction);
                d_problem->setRHS(defect);
                d_problem->setInitResVec(defect);
                d_problem->setInitPrecResVec(Teuchos::null);
                if (!d_problem->setProblem())
                {
                    restore_original_vectors();
                    residual_workspace_current = false;
                    break;
                }
                const auto ref_solve_start = std::chrono::steady_clock::now();
                return_status = d_solver->solve();
                timings.solve_seconds += elapsed_seconds(ref_solve_start);
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
            configure(refinement_options, batch_cols);
            d_solver->setParameters(d_parameters);
            if (!set_problem_with_prepared_residual())
            {
                residual_workspace_current = false;
                break;
            }
            const auto ref_solve_start = std::chrono::steady_clock::now();
            return_status = d_solver->solve();
            timings.solve_seconds += elapsed_seconds(ref_solve_start);
            iterations += d_solver->getNumIters();
        }
        achieved_tolerance = checked_residual();
        residual_workspace_current = true;
        refinement_residuals[refinement_count++] = achieved_tolerance;
    }

    if (!residual_workspace_current)
    {
        const auto check_start = std::chrono::steady_clock::now();
        (void)true_relative_residual(
            counting_op_base, rhs, solution, residual_scaling, &rhs_norm);
        rhs.norm2(precheck_rhs_norms());
        residual_workspace(rhs).norm2(precheck_residual_norms());
        timings.residual_check_seconds += elapsed_seconds(check_start);
    }

    std::vector<ColumnLinearSolveStatistics> col_stats(batch_cols);
    bool converged = true;
    real_t max_col_achieved{};
    real_t max_col_rhs{};
    for (std::size_t col = 0; col < batch_cols; ++col)
    {
        const auto b_norm = precheck_rhs_norms[col];
        const auto r_norm = precheck_residual_norms[col];
        const bool finite = std::isfinite(b_norm) && std::isfinite(r_norm);
        const auto denom = std::max(b_norm, effective_floor(col));
        const auto col_achieved = !finite
            ? std::numeric_limits<real_t>::infinity()
            : static_cast<real_t>(denom > magnitude_type{} ? r_norm / denom : r_norm);
        const auto bound = std::max(
            effective_abs_tol(col),
            static_cast<magnitude_type>(effective_tol(col)) * denom);
        const bool col_conv = finite && std::isfinite(col_achieved) && r_norm <= bound;
        converged = converged && col_conv;
        if (!std::isfinite(col_achieved))
        {
            max_col_achieved = std::numeric_limits<real_t>::infinity();
        }
        else if (std::isfinite(max_col_achieved))
        {
            max_col_achieved = std::max(max_col_achieved, col_achieved);
        }
        if (finite)
        {
            max_col_rhs = std::max(max_col_rhs, static_cast<real_t>(b_norm));
        }
        else
        {
            max_col_rhs = std::numeric_limits<real_t>::infinity();
        }
        col_stats[col] = ColumnLinearSolveStatistics{
            col_conv,
            false,
            col_achieved,
            static_cast<real_t>(r_norm),
            static_cast<real_t>(b_norm),
            column_refinements[col]};
    }
    achieved_tolerance = column_policies.empty() ? achieved_tolerance : max_col_achieved;
    if (column_policies.empty())
    {
        // Ensure per-column floor/relative policy still governs achieved_tolerance
        // if any column had nonfinite residual.
        if (!std::isfinite(max_col_achieved))
        {
            achieved_tolerance = std::numeric_limits<real_t>::infinity();
        }
        rhs_norm = std::isfinite(max_col_rhs) ? max_col_rhs : rhs_norm;
    }
    else
    {
        rhs_norm = max_col_rhs;
    }

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
            rhs, belos_options, residual_scaling, return_status, iterations);
    }
    LinearSolveStatistics result;
    result.converged = converged;
    result.iterations = iterations;
    result.achieved_tolerance = achieved_tolerance;
    result.rhs_norm = rhs_norm;
    result.execution_mode = batch_cols > 1
        ? MultiRhsExecutionMode::BatchedKrylov
        : MultiRhsExecutionMode::SingleRhs;
    result.batch_krylov_iterations = batch_cols > 1 ? iterations : 0;
    result.column_serial_iterations = 0;
    result.operator_apply_calls = counting_op->calls();
    result.operator_column_applies = counting_op->column_applies();
    result.max_batch_width = counting_op->max_batch_width();
    result.timings = timings;
    result.column_statistics = std::move(col_stats);
    return result;
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
template <TpetraTypePack Pack>
LinearSolveStatistics BelosLinearSolver<Pack>::solve_bicgstab_columns(
    const Teuchos::RCP<const operator_type>& matrix,
    const multi_vector_type& rhs, multi_vector_type& solution,
    const LinearSolverOptions& options,
    LinearResidualScaling residual_scaling,
    const std::vector<ColumnResidualPolicy>& column_policies,
    const std::vector<std::size_t>& active_indices,
    bool zero_initial_guess,
    const Teuchos::RCP<const matrix_type>& preconditioning_matrix)
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

    const std::size_t num_cols = rhs.getNumVectors();
    LinearSolveStatistics combined;
    combined.converged = true;
    combined.execution_mode = MultiRhsExecutionMode::ColumnSerial;
    combined.column_statistics.assign(
        num_cols, ColumnLinearSolveStatistics{true, true, real_t{}, real_t{}, real_t{}, 0});

    auto column_options = options;
    for (const std::size_t column : active_indices)
    {
        const auto pack_start = std::chrono::steady_clock::now();
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
        combined.timings.pack_unpack_seconds += elapsed_seconds(pack_start);

        std::vector<ColumnResidualPolicy> single_policy;
        if (!column_policies.empty())
        {
            single_policy.push_back(column_policies[column]);
        }
        const auto statistics = solve_contiguous_batch(
            matrix, *d_bicgstab_rhs_workspace, *d_bicgstab_solution_workspace,
            column_options, residual_scaling, single_policy, zero_initial_guess,
            preconditioning_matrix);

        const auto unpack_start = std::chrono::steady_clock::now();
        {
            const auto source = d_bicgstab_solution_workspace->getData(0);
            auto destination = solution.getDataNonConst(column);
            std::copy(source.begin(), source.end(), destination.begin());
        }
        combined.timings.pack_unpack_seconds += elapsed_seconds(unpack_start);

        combined.converged = combined.converged && statistics.converged;
        combined.iterations += statistics.iterations;
        combined.column_serial_iterations += statistics.iterations;
        combined.achieved_tolerance = std::max(
            combined.achieved_tolerance, statistics.achieved_tolerance);
        combined.rhs_norm = std::max(combined.rhs_norm, statistics.rhs_norm);
        combined.operator_apply_calls += statistics.operator_apply_calls;
        combined.operator_column_applies += statistics.operator_column_applies;
        combined.max_batch_width = std::max(combined.max_batch_width, statistics.max_batch_width);
        combined.timings.preconditioner_setup_seconds +=
            statistics.timings.preconditioner_setup_seconds;
        combined.timings.solve_seconds += statistics.timings.solve_seconds;
        combined.timings.residual_check_seconds +=
            statistics.timings.residual_check_seconds;
        if (!statistics.column_statistics.empty())
        {
            combined.column_statistics[column] = statistics.column_statistics[0];
        }
        if (options.preconditioner == LinearPreconditioner::None
            || !d_preconditioner.is_null())
        {
            column_options.reuse_preconditioner = true;
        }
    }
    return combined;
}


/**
* @brief Build, reuse, or remove the configured right preconditioner.
* @throws std::invalid_argument If the selection is unknown or a
*         matrix-based preconditioner is requested for a non-CRS operator.
*/
template <TpetraTypePack Pack>
void BelosLinearSolver<Pack>::configure_preconditioner(
    const Teuchos::RCP<const operator_type>& matrix,
    const LinearSolverOptions& options,
    const Teuchos::RCP<const matrix_type>& preconditioning_matrix)
{
    if (options.preconditioner == LinearPreconditioner::None)
    {
        invalidate_preconditioner();
        return;
    }

    const auto crs_matrix = preconditioning_matrix.is_null()
        ? Teuchos::rcp_dynamic_cast<const matrix_type>(matrix, false)
        : preconditioning_matrix;
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
        && d_preconditioner_operator.getRawPtr() == crs_matrix.getRawPtr()
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
        case LinearPreconditioner::MueLuSymmetric:
        {
            // The policy is fixed, but parsing it builds a sizeable factory
            // tree. Retain that configuration for the exact same map/comm;
            // every numeric setup still gets a fresh hierarchy and factors.
            if (d_muelu_factory.is_null() ||
                d_muelu_map.get() != crs_matrix->getDomainMap().get() ||
                d_muelu_kind != options.preconditioner)
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
                if (options.preconditioner == LinearPreconditioner::MueLuSymmetric)
                {
                    parameters.set("multigrid algorithm", "sa");
                    parameters.set("problem: symmetric", true);
                    parameters.set("transpose: use implicit", true);
                    parameters.set("smoother: pre or post", "both");
                    parameters.sublist("smoother: params").set(
                        "relaxation: type", "Symmetric Gauss-Seidel");
                    parameters.sublist("smoother: params").set("relaxation: sweeps", 1);
                    parameters.sublist("smoother: params").set("relaxation: damping factor", scalar_type{1});
                    parameters.set("coarse: type", "KLU2");
                    parameters.remove("coarse: params");
                }
                d_muelu_factory = Teuchos::rcp(new muelu_factory_type(
                    parameters, crs_matrix->getDomainMap()->getComm()));
                d_muelu_map = crs_matrix->getDomainMap();
                d_muelu_kind = options.preconditioner;
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

    d_preconditioner_operator = crs_matrix;
    d_preconditioner_kind = options.preconditioner;
    ++d_preconditioner_setup_count;
    // Belos CG applies this inverse to the residual in its PCG recurrence
    // and applies the original A to search directions. BiCGStab requires
    // right attachment; retaining it also avoids preconditioned norm scaling.
    d_problem->setRightPrec(d_preconditioner);
}

/**
 * @brief Creates an IFPACK2 preconditioner.
 *
 * @tparam Pack
 * @tparam Configure
 * @param matrix
 * @param factory_name
 * @param configure_parameters
 * @return Teuchos::RCP<const typename BelosLinearSolver<Pack>::operator_type>
 */
template <TpetraTypePack Pack>
template<class Configure>
Teuchos::RCP<const typename BelosLinearSolver<Pack>::operator_type>
BelosLinearSolver<Pack>::create_ifpack2_preconditioner(
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

template <TpetraTypePack Pack>
Teuchos::RCP<typename BelosLinearSolver<Pack>::solver_type>
BelosLinearSolver<Pack>::create_solver(
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

template <TpetraTypePack Pack>
void BelosLinearSolver<Pack>::reset()
{
    if (!d_problem.is_null())
    {
        d_problem->setRightPrec(Teuchos::null);
    }
    d_raw_operator = Teuchos::null;
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
    d_muelu_kind.reset();
}

template <TpetraTypePack Pack>
bool BelosLinearSolver<Pack>::set_problem_with_prepared_residual()
{
    d_problem->setInitResVec(d_residual_workspace);
    d_problem->setInitPrecResVec(Teuchos::null);
    return d_problem->setProblem();
}

template <TpetraTypePack Pack>
void BelosLinearSolver<Pack>::report_failed_solve(
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

template <TpetraTypePack Pack>
void BelosLinearSolver<Pack>::validate_residual_scaling(
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

template <TpetraTypePack Pack>
bool BelosLinearSolver<Pack>::has_compatible_maps(
    const Teuchos::RCP<const operator_type>& matrix) const
{
    if (matrix.is_null() || d_problem.is_null()
        || d_problem->getOperator().is_null()
        || d_solver.is_null())
    {
        return false;
    }

    const auto previous = d_raw_operator.is_null()
        ? d_problem->getOperator()
        : d_raw_operator;
    return previous->getDomainMap()->isSameAs(*matrix->getDomainMap())
        && previous->getRangeMap()->isSameAs(*matrix->getRangeMap());
}

template <TpetraTypePack Pack>
void BelosLinearSolver<Pack>::configure(
    const LinearSolverOptions& options,
    std::size_t batch_columns)
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
        // Always explicitly set Block Size on every solve and refinement call:
        // Trilinos 17.2 PseudoBlockGmresSolMgr defaults Block Size to 1 and
        // mutates blockSize_ during deflated/smaller solves.
        d_parameters->set(
            "Block Size",
            static_cast<int>(std::max<std::size_t>(1, batch_columns)));
    }
}

template <TpetraTypePack Pack>
void BelosLinearSolver<Pack>::invalidate_preconditioner()
{
    d_preconditioner = Teuchos::null;
    d_preconditioner_operator = Teuchos::null;
    d_preconditioner_kind.reset();
    if (!d_problem.is_null())
    {
        d_problem->setRightPrec(Teuchos::null);
    }
}

template <TpetraTypePack Pack>
bool BelosLinearSolver<Pack>::accurate_crs_residual(const Teuchos::RCP<const operator_type>& matrix,
    const multi_vector_type& rhs, const multi_vector_type& solution) const
{
    if constexpr (!std::is_floating_point_v<scalar_type>) return false;
    else
    {
        // A scaled CRS wrapper defeats the plain matrix dynamic cast, so
        // unwrap it explicitly: the compensated row loop then runs on the
        // wrapped matrix with the scale applied once per term, matching
        // Tpetra's alpha*(A*x) convention.
        const auto scaled = Teuchos::rcp_dynamic_cast<
            const detail::ScaledLinearOperator<Pack>>(matrix, false);
        const auto crs = scaled.is_null()
            ? Teuchos::rcp_dynamic_cast<const matrix_type>(matrix)
            : scaled->scaled_crs_matrix();
        const scalar_type crs_scale = scaled.is_null()
            ? scalar_type{1} : scaled->crs_scale();
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
                    const auto term = -static_cast<long double>(crs_scale)
                        * (static_cast<long double>(local.values(entry))
                            * static_cast<long double>(x_values[local.graph.entries(entry)]));
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

template <TpetraTypePack Pack>
real_t BelosLinearSolver<Pack>::true_relative_residual(
    const Teuchos::RCP<const operator_type>& matrix,
    const multi_vector_type& rhs,
    const multi_vector_type& solution,
    LinearResidualScaling residual_scaling,
    real_t* maximum_rhs_norm) const
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

template <TpetraTypePack Pack>
real_t BelosLinearSolver<Pack>::residual_relative_norm(const multi_vector_type& rhs,
    LinearResidualScaling residual_scaling, real_t* maximum_rhs_norm) const
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

template <TpetraTypePack Pack>
void BelosLinearSolver<Pack>::prepare_initial_guess(
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

template <TpetraTypePack Pack>
typename BelosLinearSolver<Pack>::multi_vector_type& BelosLinearSolver<Pack>::residual_workspace(
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
} // namespace SimpleFluid
