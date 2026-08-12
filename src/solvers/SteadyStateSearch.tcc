/**
 * @file SteadyStateSearch.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementations for adaptive steady-state monitoring.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "solvers/SteadyStateSearch.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{

template <TpetraTypePack Pack>
SteadyStateFieldMonitor<Pack>::SteadyStateFieldMonitor(SP<const mesh_type> mesh,
                                                       scalar_type reference_temperature,
                                                       SteadyStateUpdateScales scales)
    : d_mesh(require_mesh(std::move(mesh))), d_reference_temperature(reference_temperature),
      d_scales(scales)
{
    if (!std::isfinite(reference_temperature))
    {
        throw std::invalid_argument("Steady-state reference temperature must be finite.");
    }
    const SteadyStateSearchOptions validation_options{.update_scales = d_scales};
    validate_steady_state_search_options(validation_options, validation_options.minimum_time_step);
}

template <TpetraTypePack Pack>
void SteadyStateFieldMonitor<Pack>::initialize(
    const velocity_field_type& velocity, const field_type& temperature,
    std::vector<const field_type*> additional_scalar_fields)
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

template <TpetraTypePack Pack>
SteadyStateUpdateRates<typename Pack::scalar_type>
SteadyStateFieldMonitor<Pack>::observe(scalar_type time_step)
{
    if (!d_initialized)
    {
        throw std::logic_error("Steady-state field monitor must be initialized first.");
    }
    if (!std::isfinite(time_step) || time_step <= scalar_type{})
    {
        throw std::invalid_argument("Steady-state field monitor requires a finite positive time "
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
                       static_cast<int>(local_sums.size()), local_sums.data(), global_sums.data());
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
        const auto denominator = time_step * std::max(state_rms, static_cast<scalar_type>(scale));
        return delta_rms / denominator;
    };

    SteadyStateUpdateRates<scalar_type> rates;
    rates.velocity = relative_rate(global_sums[1], global_sums[2], d_scales.velocity);
    rates.temperature = relative_rate(global_sums[3], global_sums[4], d_scales.temperature);
    for (size_t field_id = 0; field_id < field_count; ++field_id)
    {
        rates.turbulence = std::max(rates.turbulence, relative_rate(global_sums[5 + 2 * field_id],
                                                                    global_sums[6 + 2 * field_id],
                                                                    d_scales.turbulence));
    }
    if (!std::isfinite(rates.velocity) || !std::isfinite(rates.temperature) ||
        !std::isfinite(rates.turbulence))
    {
        throw std::runtime_error("Steady-state monitor produced a non-finite update rate.");
    }
    return rates;
}

template <TpetraTypePack Pack>
SP<const typename SteadyStateFieldMonitor<Pack>::mesh_type>
SteadyStateFieldMonitor<Pack>::require_mesh(SP<const mesh_type> mesh)
{
    if (!mesh)
    {
        throw std::invalid_argument("Steady-state field monitor requires a non-null mesh.");
    }
    return mesh;
}

template <TpetraTypePack Pack>
template <class Field>
void SteadyStateFieldMonitor<Pack>::require_field_mesh(const Field& field, const char* name) const
{
    if (field.mesh_ptr().get() != d_mesh.get())
    {
        throw std::invalid_argument(std::string("Steady-state ") + name +
                                    " belongs to a different mesh.");
    }
}

template <TpetraTypePack Pack>
void SteadyStateFieldMonitor<Pack>::capture_current_state()
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

template <class Scalar>
std::string
SteadyStateProgressLineFormatter<Scalar>::format(const SteadyStateStepStatistics<Scalar>& statistics,
                                         const FluidStepStatistics<Scalar>& solver_statistics)
{
    std::ostringstream line;
    line << std::scientific << std::setprecision(6) << "steady_step=" << statistics.iteration << '/'
         << statistics.maximum_steps << " time=" << statistics.time
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

template <class Scalar>
std::string SteadyStateProgressLineFormatter<Scalar>::format_retry(int iteration, int retry,
                                                           int maximum_retries, Scalar time,
                                                           Scalar time_step, Scalar next_time_step,
                                                           std::string_view reason)
{
    std::ostringstream line;
    line << std::scientific << std::setprecision(6) << "steady_step=" << iteration
         << " rejected=yes retry=" << retry << '/' << maximum_retries << " time=" << time
         << " dt=" << time_step << " next_dt=" << next_time_step << " reason=\"" << reason << '"';
    return line.str();
}

template <class Scalar>
void SteadyStateProgressStream::write(const SteadyStateStepStatistics<Scalar>& statistics,
                                      const FluidStepStatistics<Scalar>& solver_statistics)
{
    d_output << SteadyStateProgressLineFormatter<Scalar>::format(statistics, solver_statistics) << std::endl;
}

template <class Scalar>
void SteadyStateProgressStream::write_retry(int iteration, int retry, int maximum_retries,
                                            Scalar time, Scalar time_step, Scalar next_time_step,
                                            std::string_view reason)
{
    d_output << SteadyStateProgressLineFormatter<Scalar>::format_retry
                            (iteration, retry, maximum_retries,
                             time, time_step,
                             next_time_step, reason)
             << std::endl;
}

} // namespace SimpleFluid
