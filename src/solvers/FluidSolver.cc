/**
 * @file FluidSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for FluidSolver.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "FluidSolver.hh"
#include "FluidSolver.tcc"

namespace SimpleFluid
{
template class FluidSolver<DefaultTpetraTypes>;
}
