/**
 * @file BoussinesqSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for BoussinesqSolver and its ALE path.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "solvers/BoussinesqSolver.hh"
#include "solvers/BoussinesqSolver.tcc"

namespace SimpleFluid
{
template class BoussinesqSolver<DefaultTpetraTypes>;
}
