/**
 * @file SolverProgress.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Reusable formatting and streaming for solver progress lines.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/PressureVelocityCoupling.hh"

#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

namespace SimpleFluid
{

/** @brief Format one physical-step convergence summary. */
class ProgressLineFormatter
{
public:
    /**
     * @brief Format one physical-step convergence summary.
     *
     * @tparam Scalar Scalar type used for time and residual values.
     * @param step Current absolute step index.
     * @param total_steps Final step index for this run.
     * @param time Current physical time.
     * @param statistics Aggregated convergence statistics.
     * @return Single-line progress message.
     */
    template<class Scalar>
    std::string format(
        int step,
        int total_steps,
        Scalar time,
        const FluidStepStatistics<Scalar>& statistics) const
    {
        std::ostringstream line;
        line << std::scientific << std::setprecision(6)
             << "step=" << step << '/' << total_steps
             << " time=" << time
             << " converged=" << (statistics.converged ? "yes" : "no")
             << " nonlinear=" << statistics.nonlinear_iterations
             << " linear_solves=" << statistics.linear_solves
             << " krylov_iterations=" << statistics.krylov_iterations
             << " linear_tolerance=" << statistics.achieved_tolerance
             << " residuals(momentum=" << statistics.momentum
             << ",pressure=" << statistics.pressure
             << ",temperature=" << statistics.temperature
             << ",continuity=" << statistics.continuity << ')';
        return line.str();
    }
};

/**
 * @brief Wrap an output stream and flush one formatted progress line at a time.
 */
class ProgressStream
{
public:
    /**
     * @brief Construct a progress sink around an existing output stream.
     *
     * @param output Stream that receives formatted progress lines.
     */
    explicit ProgressStream(std::ostream& output)
        : d_output(output)
    {
    }

    /**
     * @brief Format, write, and flush one progress line.
     *
     * @tparam Scalar Scalar type used for time and residual values.
     * @param step Current absolute step index.
     * @param total_steps Final step index for this run.
     * @param time Current physical time.
     * @param statistics Aggregated convergence statistics.
     */
    template<class Scalar>
    void write(
        int step,
        int total_steps,
        Scalar time,
        const FluidStepStatistics<Scalar>& statistics)
    {
        d_output << d_formatter.format(
            step, total_steps, time, statistics) << std::endl;
    }

private:
    std::ostream& d_output;
    ProgressLineFormatter d_formatter;
};

} // namespace SimpleFluid
