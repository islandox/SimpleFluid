/**
 * @file SolverProgress.hh
 * @brief Reusable formatting and streaming for solver progress lines.
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
    explicit ProgressStream(std::ostream& output)
        : d_output(output)
    {
    }

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
