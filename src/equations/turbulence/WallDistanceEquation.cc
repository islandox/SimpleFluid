/**
 * @file WallDistanceEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit instantiation of distributed Poisson wall distance.
 * @version 0.1
 * @date 2026-07-24
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "WallDistanceEquation.hh"
#include "WallDistanceEquation.tcc"

#include <cmath>
#include <stdexcept>

namespace SimpleFluid
{

void validate_wall_distance_equation_options(
    const WallDistanceEquationOptions& options)
{
    switch (options.non_orthogonal_treatment)
    {
        case FVM::NonOrthogonalTreatment::Explicit:
        case FVM::NonOrthogonalTreatment::Implicit:
        case FVM::NonOrthogonalTreatment::Hybrid:
            break;
        default:
            throw std::invalid_argument(
                "Poisson wall-distance non-orthogonal treatment is invalid.");
    }
    if (options.non_orthogonal_correctors < 0)
    {
        throw std::invalid_argument(
            "Poisson wall-distance non-orthogonal correctors cannot be "
            "negative.");
    }
    if (options.linear_solver.max_iterations <= 0
        || !std::isfinite(options.linear_solver.tolerance)
        || options.linear_solver.tolerance <= real_t{})
    {
        throw std::invalid_argument(
            "Poisson wall-distance linear solver requires positive "
            "iterations and tolerance.");
    }
    static_cast<void>(to_string(options.linear_solver.backend));
    static_cast<void>(to_string(options.linear_solver.preconditioner));
}

template class PoissonWallDistanceEquation<DefaultTpetraTypes>;
}
