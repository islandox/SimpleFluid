/**
 * @file SteadyStateSearch.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Non-template adaptive steady-state controller implementations.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "solvers/SteadyStateSearch.hh"
#include "solvers/SteadyStateSearch.tcc"
#include "dataclass/TpetraTypes.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{

// Explicit instantiation of the default Tpetra type pack.

template class SteadyStateFieldMonitor<DefaultTpetraTypes>;

template void SteadyStateProgressStream::write<real_t>(const SteadyStateStepStatistics<real_t>& statistics,
                                           const FluidStepStatistics<real_t>& solver_statistics);
template void SteadyStateProgressStream::write_retry<real_t>(int iteration, int retry,
                                           int maximum_retries, real_t time, real_t time_step,
                                           real_t next_time_step, std::string_view reason);

namespace
{

bool finite_positive(real_t value) { return std::isfinite(value) && value > 0.0; }

bool finite_non_negative(real_t value) { return std::isfinite(value) && value >= 0.0; }

} // namespace

void validate_steady_state_search_options(const SteadyStateSearchOptions& options,
                                          real_t initial_time_step)
{
    if (options.maximum_steps < 1)
    {
        throw std::invalid_argument("Steady-state search requires at least one maximum step.");
    }
    if (options.minimum_steps < 1 || options.minimum_steps > options.maximum_steps)
    {
        throw std::invalid_argument("Steady-state minimum steps must lie in [1, maximum_steps].");
    }
    if (options.required_consecutive_steps < 1 ||
        options.required_consecutive_steps > options.maximum_steps)
    {
        throw std::invalid_argument(
            "Steady-state consecutive steps must lie in [1, maximum_steps].");
    }
    if (options.minimum_steps > options.maximum_steps - options.required_consecutive_steps + 1)
    {
        throw std::invalid_argument("Steady-state minimum and consecutive-step controls do not fit "
                                    "within maximum_steps.");
    }
    if (options.maximum_retries_per_step < 0)
    {
        throw std::invalid_argument("Steady-state maximum retries per step cannot be negative.");
    }
    if (options.rejection_recovery_steps < 0)
    {
        throw std::invalid_argument("Steady-state rejection recovery steps cannot be negative.");
    }
    if (!finite_positive(options.relative_update_tolerance))
    {
        throw std::invalid_argument("Steady-state relative update tolerance must be finite and "
                                    "positive.");
    }
    if (!finite_positive(options.minimum_time_step) ||
        !finite_positive(options.maximum_time_step) ||
        options.minimum_time_step > options.maximum_time_step)
    {
        throw std::invalid_argument("Steady-state time-step bounds must be finite, positive, and "
                                    "ordered.");
    }
    if (!finite_positive(initial_time_step) || initial_time_step < options.minimum_time_step ||
        initial_time_step > options.maximum_time_step)
    {
        throw std::invalid_argument("Initial steady-state time step must lie within its configured "
                                    "bounds.");
    }
    if (!finite_positive(options.target_courant_number))
    {
        throw std::invalid_argument(
            "Steady-state target Courant number must be finite and positive.");
    }
    if (!std::isfinite(options.time_step_growth_factor) || options.time_step_growth_factor <= 1.0)
    {
        throw std::invalid_argument(
            "Steady-state time-step growth factor must be finite and greater "
            "than one.");
    }
    if (!std::isfinite(options.time_step_reduction_factor) ||
        options.time_step_reduction_factor <= 0.0 || options.time_step_reduction_factor >= 1.0)
    {
        throw std::invalid_argument("Steady-state time-step reduction factor must lie in (0, 1).");
    }
    if (!std::isfinite(options.rejection_time_step_safety_factor) ||
        options.rejection_time_step_safety_factor <= 0.0 ||
        options.rejection_time_step_safety_factor >= 1.0)
    {
        throw std::invalid_argument("Steady-state rejection safety factor must lie in (0, 1).");
    }
    if (!finite_positive(options.update_scales.velocity) ||
        !finite_positive(options.update_scales.temperature) ||
        !finite_positive(options.update_scales.turbulence))
    {
        throw std::invalid_argument(
            "Steady-state field update scales must be finite and positive.");
    }
}

AdaptiveSteadyStateController::AdaptiveSteadyStateController(SteadyStateSearchOptions options,
                                                             real_t initial_time_step)
    : d_options(std::move(options)),
      d_rejection_time_step_ceiling(std::numeric_limits<real_t>::infinity())
{
    validate_steady_state_search_options(d_options, initial_time_step);
}

const SteadyStateSearchOptions& AdaptiveSteadyStateController::options() const noexcept
{
    return d_options;
}

SteadyStateStepStatistics<real_t>
AdaptiveSteadyStateController::observe(real_t time, real_t time_step, real_t maximum_courant_number,
                                       SteadyStateUpdateRates<real_t> update_rates,
                                       bool solver_converged)
{
    if (!finite_non_negative(time) || !finite_positive(time_step) ||
        !finite_non_negative(maximum_courant_number) ||
        !finite_non_negative(update_rates.velocity) ||
        !finite_non_negative(update_rates.temperature) ||
        !finite_non_negative(update_rates.turbulence))
    {
        throw std::invalid_argument("Steady-state observation requires finite non-negative "
                                    "diagnostics and a positive time step.");
    }

    ++d_iterations;
    const bool below_tolerance = update_rates.maximum() <= d_options.relative_update_tolerance;
    if (solver_converged && d_iterations >= d_options.minimum_steps && below_tolerance)
    {
        ++d_consecutive_converged_steps;
    }
    else
    {
        d_consecutive_converged_steps = 0;
    }

    auto next_time_step = adapted_time_step(time_step, maximum_courant_number, solver_converged);
    if (d_rejection_recovery_steps_remaining > 0)
    {
        next_time_step = std::min(next_time_step, time_step);
        --d_rejection_recovery_steps_remaining;
    }
    next_time_step = std::min(next_time_step, d_rejection_time_step_ceiling);

    return {d_iterations,
            d_options.maximum_steps,
            d_consecutive_converged_steps,
            d_options.required_consecutive_steps,
            time,
            time_step,
            next_time_step,
            maximum_courant_number,
            update_rates,
            solver_converged,
            d_consecutive_converged_steps >= d_options.required_consecutive_steps};
}

real_t AdaptiveSteadyStateController::rejected_time_step(real_t time_step)
{
    if (!finite_positive(time_step))
    {
        throw std::invalid_argument("Rejected steady-state time step must be finite and positive.");
    }
    d_rejection_recovery_steps_remaining = d_options.rejection_recovery_steps;
    d_rejection_time_step_ceiling =
        std::min(d_rejection_time_step_ceiling,
                 std::clamp(time_step * d_options.rejection_time_step_safety_factor,
                            d_options.minimum_time_step, d_options.maximum_time_step));

    return std::min(std::clamp(time_step * d_options.time_step_reduction_factor,
                               d_options.minimum_time_step, d_options.maximum_time_step),
                    d_rejection_time_step_ceiling);
}

real_t AdaptiveSteadyStateController::adapted_time_step(real_t current_time_step,
                                                        real_t maximum_courant_number,
                                                        bool solver_converged) const
{
    real_t factor = d_options.time_step_reduction_factor;
    if (solver_converged)
    {
        factor = d_options.time_step_growth_factor;
        if (maximum_courant_number > 0.0)
        {
            factor =
                std::clamp(d_options.target_courant_number / maximum_courant_number,
                           d_options.time_step_reduction_factor, d_options.time_step_growth_factor);
        }
    }
    return std::clamp(current_time_step * factor, d_options.minimum_time_step,
                      d_options.maximum_time_step);
}

SteadyStateProgressStream::SteadyStateProgressStream(std::ostream& output) : d_output(output) {}

} // namespace SimpleFluid
