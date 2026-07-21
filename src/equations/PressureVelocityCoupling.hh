/**
 * @file equations/PressureVelocityCoupling.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Runtime switches and residuals for pressure-velocity coupling.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "solvers/BelosLinearSolver.hh"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace SimpleFluid
{

/**
 * @brief Available pressure-velocity coupling drivers.
 */
enum class PressureVelocityCoupling
{
    SIMPLE,
    PISO,
    PIMPLE,
    CoupledKrylov
};

/**
 * @brief Parse the pressure-velocity coupling switch used by input files.
 */
inline PressureVelocityCoupling
pressure_velocity_coupling_from_string(std::string_view value)
{
    if (value == "SIMPLE" || value == "simple")
    {
        return PressureVelocityCoupling::SIMPLE;
    }
    if (value == "PISO" || value == "piso")
    {
        return PressureVelocityCoupling::PISO;
    }
    if (value == "PIMPLE" || value == "pimple")
    {
        return PressureVelocityCoupling::PIMPLE;
    }
    if (value == "coupledKrylov" || value == "CoupledKrylov"
        || value == "coupledkrylov")
    {
        return PressureVelocityCoupling::CoupledKrylov;
    }

    throw std::invalid_argument(
        "Unknown pressure-velocity coupling; expected SIMPLE, PISO, PIMPLE, "
        "or coupledKrylov.");
}

/**
 * @brief Return the input-file spelling for a pressure-velocity coupling mode.
 */
inline std::string_view
to_string(PressureVelocityCoupling coupling)
{
    switch (coupling)
    {
        case PressureVelocityCoupling::SIMPLE: return "SIMPLE";
        case PressureVelocityCoupling::PISO:   return "PISO";
        case PressureVelocityCoupling::PIMPLE: return "PIMPLE";
        case PressureVelocityCoupling::CoupledKrylov:
            return "coupledKrylov";
    }

    throw std::invalid_argument("Unknown PressureVelocityCoupling value.");
}

/**
 * @brief Last-step residual norms and linear statistics for pressure-velocity coupling.
 * @tparam Scalar Floating-point scalar type used for residuals.
 */
template<class Scalar>
struct PressureVelocityResiduals
{
    Scalar momentum = {};
    Scalar pressure = {};
    Scalar continuity = {};
    Scalar achieved_tolerance = {};
    int linear_iterations = 0;
};

/**
 * @brief Aggregated physical-step diagnostics and convergence state.
 * @tparam Scalar Floating-point scalar type used for residuals.
 */
template<class Scalar>
struct FluidStepStatistics
{
    bool converged = true;
    int nonlinear_iterations = 0;
    int linear_solves = 0;
    int krylov_iterations = 0;
    Scalar achieved_tolerance = {};
    Scalar momentum = {};
    Scalar pressure = {};
    Scalar temperature = {};
    Scalar continuity = {};

    void add(const LinearSolveStatistics& statistics)
    {
        converged = converged && statistics.converged;
        ++linear_solves;
        krylov_iterations += statistics.iterations;
        achieved_tolerance =
            std::max(achieved_tolerance, statistics.achieved_tolerance);
    }

    void add(const LinearSolveSummary& summary)
    {
        converged = converged && summary.converged;
        linear_solves += summary.solves;
        krylov_iterations += summary.iterations;
        achieved_tolerance =
            std::max(achieved_tolerance, summary.achieved_tolerance);
    }
};

template<class Scalar>
using BoussinesqStepStatistics = FluidStepStatistics<Scalar>;

} // namespace SimpleFluid
