/**
 * @file SteadyStateSearch.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Adaptive pseudo-transient controls and physical-update monitoring.
 * @version 0.1
 * @date 2026-07-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "dataclass/typedefs.hh"
#include "equations/PressureVelocityCoupling.hh"
#include "fields/CellField.hh"
#include "fields/VectorCellField.hh"
#include "geometry/Mesh.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Absolute scales used when a monitored field is initially zero.
 *
 * Once a field's volume-weighted RMS exceeds its floor, update rates are
 * normalized by the field itself. Temperature is measured relative to the
 * monitor's reference temperature rather than its absolute thermodynamic
 * value.
 */
struct SteadyStateUpdateScales
{
    real_t velocity = 1.0e-6;    ///< Velocity scale floor [m/s].
    real_t temperature = 1.0;    ///< Temperature-difference scale floor [K].
    real_t turbulence = 1.0e-12; ///< Generic turbulence-field scale floor.
};

/**
 * @brief Controls for an adaptive pseudo-transient steady-state search.
 *
 * A steady state is accepted only after every monitored relative update rate
 * remains below `relative_update_tolerance` for
 * `required_consecutive_steps`. The time step is independently adjusted
 * toward `target_courant_number`, bounded by the configured minimum,
 * maximum, and per-step growth/reduction factors.
 */
struct SteadyStateSearchOptions
{
    int maximum_steps = 1000;
    int minimum_steps = 20;
    int required_consecutive_steps = 5;
    int maximum_retries_per_step = 4;
    real_t relative_update_tolerance = 1.0e-4;
    real_t minimum_time_step = 1.0e-6;
    real_t maximum_time_step = 5.0e-2;
    real_t target_courant_number = 0.8;
    real_t time_step_growth_factor = 1.5;
    real_t time_step_reduction_factor = 0.5;
    int rejection_recovery_steps = 5;
    /**
     * @brief Persistent ceiling learned from a rejected step.
     *
     * After a failed attempt at dt, later growth is capped at this fraction
     * of dt so the controller does not repeatedly rediscover the same
     * unstable step size.
     */
    real_t rejection_time_step_safety_factor = 0.9;
    SteadyStateUpdateScales update_scales;
};

/** @brief Validate steady-state controls against the initial pseudo-time step.
 */
inline void validate_steady_state_search_options(const SteadyStateSearchOptions& options,
                                                 real_t initial_time_step)
{
    const auto finite_positive = [](real_t value) { return std::isfinite(value) && value > 0.0; };

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
        throw std::invalid_argument(
            "Steady-state rejection recovery steps cannot be negative.");
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
        throw std::invalid_argument(
            "Steady-state rejection safety factor must lie in (0, 1).");
    }
    if (!finite_positive(options.update_scales.velocity) ||
        !finite_positive(options.update_scales.temperature) ||
        !finite_positive(options.update_scales.turbulence))
    {
        throw std::invalid_argument(
            "Steady-state field update scales must be finite and positive.");
    }
}

/** @brief Normalized physical-field change rates from one accepted step. */
template<class Scalar> struct SteadyStateUpdateRates
{
    Scalar velocity = {};
    Scalar temperature = {};
    Scalar turbulence = {};

    Scalar maximum() const noexcept { return std::max({velocity, temperature, turbulence}); }
};

/** @brief Decision and diagnostics produced after one pseudo-transient step. */
template<class Scalar> struct SteadyStateStepStatistics
{
    int iteration = 0;
    int maximum_steps = 0;
    int consecutive_converged_steps = 0;
    int required_consecutive_steps = 0;
    Scalar time = {};
    Scalar time_step = {};
    Scalar next_time_step = {};
    Scalar maximum_courant_number = {};
    SteadyStateUpdateRates<Scalar> update_rates;
    bool solver_converged = false;
    bool steady = false;
};

/**
 * @brief Adapt pseudo-time steps and enforce sustained steady convergence.
 */
class AdaptiveSteadyStateController
{
public:
    explicit AdaptiveSteadyStateController(SteadyStateSearchOptions options,
                                           real_t initial_time_step)
        : d_options(std::move(options))
    {
        validate_steady_state_search_options(d_options, initial_time_step);
    }

    const SteadyStateSearchOptions& options() const noexcept { return d_options; }

    /**
     * @brief Observe one accepted solver step.
     *
     * @param time New pseudo-time after the step.
     * @param time_step Time step used for this observation.
     * @param maximum_courant_number Maximum cell Courant number for the step.
     * @param update_rates Physical relative update rates.
     * @param solver_converged Whether every linear solve reported convergence.
     */
    SteadyStateStepStatistics<real_t> observe(real_t time, real_t time_step,
                                              real_t maximum_courant_number,
                                              SteadyStateUpdateRates<real_t> update_rates,
                                              bool solver_converged)
    {
        const auto finite_non_negative = [](real_t value)
        { return std::isfinite(value) && value >= 0.0; };
        if (!std::isfinite(time) || time < 0.0 || !std::isfinite(time_step) || time_step <= 0.0 ||
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

        auto next_time_step =
            adapted_time_step(time_step, maximum_courant_number, solver_converged);
        if (d_rejection_recovery_steps_remaining > 0)
        {
            next_time_step = std::min(next_time_step, time_step);
            --d_rejection_recovery_steps_remaining;
        }
        next_time_step = std::min(
            next_time_step, d_rejection_time_step_ceiling);
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

    /**
     * @brief Reduce a rejected pseudo-time step without accepting an iteration.
     */
    real_t rejected_time_step(real_t time_step)
    {
        if (!std::isfinite(time_step) || time_step <= 0.0)
        {
            throw std::invalid_argument(
                "Rejected steady-state time step must be finite and positive.");
        }
        d_rejection_recovery_steps_remaining =
            d_options.rejection_recovery_steps;
        d_rejection_time_step_ceiling = std::min(
            d_rejection_time_step_ceiling,
            std::clamp(
                time_step
                    * d_options.rejection_time_step_safety_factor,
                d_options.minimum_time_step,
                d_options.maximum_time_step));
        return std::min(
            std::clamp(
                time_step * d_options.time_step_reduction_factor,
                d_options.minimum_time_step,
                d_options.maximum_time_step),
            d_rejection_time_step_ceiling);
    }

private:
    real_t adapted_time_step(real_t current_time_step, real_t maximum_courant_number,
                             bool solver_converged) const
    {
        real_t factor = d_options.time_step_reduction_factor;
        if (solver_converged)
        {
            factor = d_options.time_step_growth_factor;
            if (maximum_courant_number > 0.0)
            {
                factor = std::clamp(d_options.target_courant_number / maximum_courant_number,
                                    d_options.time_step_reduction_factor,
                                    d_options.time_step_growth_factor);
            }
        }
        return std::clamp(current_time_step * factor, d_options.minimum_time_step,
                          d_options.maximum_time_step);
    }

    SteadyStateSearchOptions d_options;
    int d_iterations = 0;
    int d_consecutive_converged_steps = 0;
    int d_rejection_recovery_steps_remaining = 0;
    real_t d_rejection_time_step_ceiling =
        std::numeric_limits<real_t>::infinity();
};

/**
 * @brief Monitor actual volume-weighted field changes across solver steps.
 *
 * The monitor owns rank-local snapshots and performs one collective reduction
 * per observation. Additional scalar fields are intended for transported
 * turbulence variables such as k and epsilon/omega. Their addresses and mesh
 * association must remain valid until the search ends.
 *
 * @tparam Pack Tpetra type pack used by the monitored fields.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes> class SteadyStateFieldMonitor
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    SteadyStateFieldMonitor(SP<const mesh_type> mesh, scalar_type reference_temperature,
                            SteadyStateUpdateScales scales = {})
        : d_mesh(require_mesh(std::move(mesh))), d_reference_temperature(reference_temperature),
          d_scales(scales)
    {
        if (!std::isfinite(reference_temperature))
        {
            throw std::invalid_argument("Steady-state reference temperature must be finite.");
        }
        const SteadyStateSearchOptions validation_options{.update_scales = d_scales};
        validate_steady_state_search_options(validation_options,
                                             validation_options.minimum_time_step);
    }

    /**
     * @brief Capture the initial state before the first search step.
     */
    void initialize(const velocity_field_type& velocity, const field_type& temperature,
                    std::vector<const field_type*> additional_scalar_fields = {})
    {
        require_field_mesh(velocity, "velocity");
        require_field_mesh(temperature, "temperature");
        for (const auto* field : additional_scalar_fields)
        {
            if (field == nullptr)
            {
                throw std::invalid_argument("Steady-state additional field cannot be null.");
            }
            require_field_mesh(*field, "additional scalar field");
        }

        d_velocity = &velocity;
        d_temperature = &temperature;
        d_additional_scalar_fields = std::move(additional_scalar_fields);
        capture_current_state();
        d_initialized = true;
    }

    /**
     * @brief Compare the current fields with the preceding accepted state.
     *
     * Rates have inverse-time units:
     * `rms(current - previous) / (time_step * max(rms(current), scale))`.
     * The current state becomes the reference for the next observation.
     */
    SteadyStateUpdateRates<scalar_type> observe(scalar_type time_step)
    {
        if (!d_initialized)
        {
            throw std::logic_error("Steady-state field monitor must be initialized first.");
        }
        if (!std::isfinite(time_step) || time_step <= scalar_type{})
        {
            throw std::invalid_argument(
                "Steady-state field monitor requires a finite positive time "
                "step.");
        }

        const size_t field_count = d_additional_scalar_fields.size();
        std::vector<scalar_type> local_sums(5 + 2 * field_count, scalar_type{});
        const auto velocity_values = d_velocity->owned_read_view();
        const auto temperature_values = d_temperature->owned_read_view();
        const auto& volumes = d_mesh->host_views().cell_geometry.volume;

        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            const auto volume = static_cast<scalar_type>(volumes[owned]);
            local_sums[0] += volume;

            scalar_type velocity_delta_squared{};
            scalar_type velocity_state_squared{};
            for (size_t component = 0; component < 3; ++component)
            {
                const auto current = velocity_values(cell, component);
                const auto previous = d_previous_velocity[3 * owned + component];
                const auto delta = current - previous;
                velocity_delta_squared += delta * delta;
                velocity_state_squared += current * current;
                d_previous_velocity[3 * owned + component] = current;
            }
            local_sums[1] += velocity_delta_squared * volume;
            local_sums[2] += velocity_state_squared * volume;

            const auto current_temperature = temperature_values(cell, 0);
            const auto temperature_delta = current_temperature - d_previous_temperature[owned];
            const auto relative_temperature = current_temperature - d_reference_temperature;
            local_sums[3] += temperature_delta * temperature_delta * volume;
            local_sums[4] += relative_temperature * relative_temperature * volume;
            d_previous_temperature[owned] = current_temperature;
        }

        for (size_t field_id = 0; field_id < field_count; ++field_id)
        {
            const auto values = d_additional_scalar_fields[field_id]->owned_read_view();
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                const auto current = values(cell, 0);
                const auto delta = current - d_previous_additional_fields[field_id][owned];
                const auto volume = static_cast<scalar_type>(volumes[owned]);
                local_sums[5 + 2 * field_id] += delta * delta * volume;
                local_sums[6 + 2 * field_id] += current * current * volume;
                d_previous_additional_fields[field_id][owned] = current;
            }
        }

        std::vector<scalar_type> global_sums(local_sums.size(), scalar_type{});
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
                           static_cast<int>(local_sums.size()), local_sums.data(),
                           global_sums.data());
        if (!std::isfinite(global_sums[0]) || global_sums[0] <= scalar_type{})
        {
            throw std::runtime_error("Steady-state monitor found a non-positive global volume.");
        }

        const auto relative_rate =
            [&](scalar_type delta_squared, scalar_type state_squared, real_t scale)
        {
            using std::sqrt;
            const auto delta_rms = sqrt(delta_squared / global_sums[0]);
            const auto state_rms = sqrt(state_squared / global_sums[0]);
            const auto denominator =
                time_step * std::max(state_rms, static_cast<scalar_type>(scale));
            return delta_rms / denominator;
        };

        SteadyStateUpdateRates<scalar_type> rates;
        rates.velocity = relative_rate(global_sums[1], global_sums[2], d_scales.velocity);
        rates.temperature = relative_rate(global_sums[3], global_sums[4], d_scales.temperature);
        for (size_t field_id = 0; field_id < field_count; ++field_id)
        {
            rates.turbulence =
                std::max(rates.turbulence,
                         relative_rate(global_sums[5 + 2 * field_id], global_sums[6 + 2 * field_id],
                                       d_scales.turbulence));
        }
        if (!std::isfinite(rates.velocity) || !std::isfinite(rates.temperature) ||
            !std::isfinite(rates.turbulence))
        {
            throw std::runtime_error("Steady-state monitor produced a non-finite update rate.");
        }
        return rates;
    }

private:
    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument("Steady-state field monitor requires a non-null mesh.");
        }
        return mesh;
    }

    template<class Field> void require_field_mesh(const Field& field, const char* name) const
    {
        if (field.mesh_ptr().get() != d_mesh.get())
        {
            throw std::invalid_argument(std::string("Steady-state ") + name +
                                        " belongs to a different mesh.");
        }
    }

    void capture_current_state()
    {
        const auto owned_cells = d_mesh->num_owned_cells();
        d_previous_velocity.assign(3 * owned_cells, scalar_type{});
        d_previous_temperature.assign(owned_cells, scalar_type{});
        d_previous_additional_fields.assign(d_additional_scalar_fields.size(),
                                            std::vector<scalar_type>(owned_cells, scalar_type{}));

        const auto velocity_values = d_velocity->owned_read_view();
        const auto temperature_values = d_temperature->owned_read_view();
        for (size_t owned = 0; owned < owned_cells; ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            for (size_t component = 0; component < 3; ++component)
            {
                d_previous_velocity[3 * owned + component] = velocity_values(cell, component);
            }
            d_previous_temperature[owned] = temperature_values(cell, 0);
        }
        for (size_t field_id = 0; field_id < d_additional_scalar_fields.size(); ++field_id)
        {
            const auto values = d_additional_scalar_fields[field_id]->owned_read_view();
            for (size_t owned = 0; owned < owned_cells; ++owned)
            {
                d_previous_additional_fields[field_id][owned] =
                    values(static_cast<local_ordinal_type>(owned), 0);
            }
        }
    }

    SP<const mesh_type> d_mesh;
    scalar_type d_reference_temperature;
    SteadyStateUpdateScales d_scales;
    const velocity_field_type* d_velocity = nullptr;
    const field_type* d_temperature = nullptr;
    std::vector<const field_type*> d_additional_scalar_fields;
    std::vector<scalar_type> d_previous_velocity;
    std::vector<scalar_type> d_previous_temperature;
    std::vector<std::vector<scalar_type>> d_previous_additional_fields;
    bool d_initialized = false;
};

/** @brief Format one adaptive steady-state search iteration. */
class SteadyStateProgressLineFormatter
{
public:
    template<class Scalar>
    std::string format(const SteadyStateStepStatistics<Scalar>& statistics,
                       const FluidStepStatistics<Scalar>& solver_statistics) const
    {
        std::ostringstream line;
        line << std::scientific << std::setprecision(6) << "steady_step=" << statistics.iteration
             << '/' << statistics.maximum_steps << " time=" << statistics.time
             << " dt=" << statistics.time_step << " next_dt=" << statistics.next_time_step
             << " max_Co=" << statistics.maximum_courant_number
             << " update_rates(U=" << statistics.update_rates.velocity
             << ",T=" << statistics.update_rates.temperature
             << ",turbulence=" << statistics.update_rates.turbulence
             << ",max=" << statistics.update_rates.maximum()
             << ") steady_window=" << statistics.consecutive_converged_steps << '/'
             << statistics.required_consecutive_steps
             << " steady=" << (statistics.steady ? "yes" : "no")
             << " linear_converged=" << (statistics.solver_converged ? "yes" : "no")
             << " krylov_iterations=" << solver_statistics.krylov_iterations
             << " linear_tolerance=" << solver_statistics.achieved_tolerance;
        return line.str();
    }

    template<class Scalar>
    std::string format_retry(int iteration, int retry, int maximum_retries, Scalar time,
                             Scalar time_step, Scalar next_time_step, std::string_view reason) const
    {
        std::ostringstream line;
        line << std::scientific << std::setprecision(6) << "steady_step=" << iteration
             << " rejected=yes retry=" << retry << '/' << maximum_retries << " time=" << time
             << " dt=" << time_step << " next_dt=" << next_time_step << " reason=\"" << reason
             << '"';
        return line.str();
    }
};

/** @brief Flush one steady-state progress line at a time. */
class SteadyStateProgressStream
{
public:
    explicit SteadyStateProgressStream(std::ostream& output) : d_output(output) {}

    template<class Scalar>
    void write(const SteadyStateStepStatistics<Scalar>& statistics,
               const FluidStepStatistics<Scalar>& solver_statistics)
    {
        d_output << d_formatter.format(statistics, solver_statistics) << std::endl;
    }

    template<class Scalar>
    void write_retry(int iteration, int retry, int maximum_retries, Scalar time, Scalar time_step,
                     Scalar next_time_step, std::string_view reason)
    {
        d_output << d_formatter.format_retry(iteration, retry, maximum_retries, time, time_step,
                                             next_time_step, reason)
                 << std::endl;
    }

private:
    std::ostream& d_output;
    SteadyStateProgressLineFormatter d_formatter;
};

} // namespace SimpleFluid
