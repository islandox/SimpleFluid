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

#include "SimpleFluidExport.hh"
#include "dataclass/typedefs.hh"
#include "equations/PressureVelocityCoupling.hh"
#include "fields/CellField.hh"
#include "fields/VectorCellField.hh"
#include "geometry/Mesh.hh"

#include <iosfwd>
#include <string>
#include <string_view>
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

/**
 * @brief Validate steady-state controls against the initial pseudo-time step.
 */
SIMPLEFLUID_SOLVERS_EXPORT void
validate_steady_state_search_options(
    const SteadyStateSearchOptions& options,
    real_t initial_time_step);

/** @brief Normalized physical-field change rates from one accepted step. */
template <class Scalar>
struct SteadyStateUpdateRates
{
    Scalar velocity = {};
    Scalar temperature = {};
    Scalar turbulence = {};

    Scalar maximum() const noexcept {
        return std::max({velocity, temperature, turbulence});
    }
};

/** @brief Decision and diagnostics produced after one pseudo-transient step. */
template <class Scalar>
struct SteadyStateStepStatistics
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
class SIMPLEFLUID_SOLVERS_EXPORT AdaptiveSteadyStateController
{
public:
    explicit AdaptiveSteadyStateController(SteadyStateSearchOptions options,
                                           real_t initial_time_step);

    const SteadyStateSearchOptions& options() const noexcept;

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
                                              bool solver_converged);

    /**
     * @brief Reduce a rejected pseudo-time step without accepting an iteration.
     */
    real_t rejected_time_step(real_t time_step);

private:
    SIMPLEFLUID_SOLVERS_LOCAL
    real_t adapted_time_step(real_t current_time_step, real_t maximum_courant_number,
                             bool solver_converged) const;

    SteadyStateSearchOptions d_options;
    int d_iterations = 0;
    int d_consecutive_converged_steps = 0;
    int d_rejection_recovery_steps_remaining = 0;
    real_t d_rejection_time_step_ceiling;
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
template <TpetraTypePack Pack = DefaultTpetraTypes>
class SIMPLEFLUID_SOLVERS_EXPORT SteadyStateFieldMonitor
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using velocity_field_type = VectorCellField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    SteadyStateFieldMonitor(SP<const mesh_type> mesh, scalar_type reference_temperature,
                            SteadyStateUpdateScales scales = {});

    /**
     * @brief Capture the initial state before the first search step.
     */
    void initialize(const velocity_field_type& velocity, const field_type& temperature,
                    std::vector<const field_type*> additional_scalar_fields = {});

    /**
     * @brief Compare the current fields with the preceding accepted state.
     *
     * Rates have inverse-time units:
     * `rms(current - previous) / (time_step * max(rms(current), scale))`.
     * The current state becomes the reference for the next observation.
     */
    SteadyStateUpdateRates<scalar_type> observe(scalar_type time_step);

private:
    SIMPLEFLUID_SOLVERS_LOCAL
    static SP<const mesh_type> require_mesh(SP<const mesh_type> mesh);

    template <class Field>
    SIMPLEFLUID_SOLVERS_LOCAL
    void require_field_mesh(const Field& field, const char* name) const;

    SIMPLEFLUID_SOLVERS_LOCAL
    void capture_current_state();

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
template <class Scalar>
class SIMPLEFLUID_SOLVERS_EXPORT SteadyStateProgressLineFormatter
{
public:
    static std::string format(const SteadyStateStepStatistics<Scalar>& statistics,
                       const FluidStepStatistics<Scalar>& solver_statistics);

    static std::string format_retry(int iteration, int retry, int maximum_retries, Scalar time,
                             Scalar time_step, Scalar next_time_step,
                             std::string_view reason);
};

/** @brief Flush one steady-state progress line at a time. */
class SIMPLEFLUID_SOLVERS_EXPORT SteadyStateProgressStream
{
public:
    explicit SteadyStateProgressStream(std::ostream& output);

    template <class Scalar>
    void write(const SteadyStateStepStatistics<Scalar>& statistics,
               const FluidStepStatistics<Scalar>& solver_statistics);

    template <class Scalar>
    void write_retry(int iteration, int retry, int maximum_retries, Scalar time, Scalar time_step,
                     Scalar next_time_step, std::string_view reason);

private:
    std::ostream& d_output;
};

extern template class SteadyStateFieldMonitor<DefaultTpetraTypes>;
extern template class SteadyStateProgressLineFormatter<real_t>;
extern template void SteadyStateProgressStream::write<real_t>(
    const SteadyStateStepStatistics<real_t>& statistics,
    const FluidStepStatistics<real_t>& solver_statistics);
extern template void SteadyStateProgressStream::write_retry<real_t>(
    int iteration, int retry, int maximum_retries,
    real_t time, real_t time_step, real_t next_time_step,
    std::string_view reason);

} // namespace SimpleFluid
