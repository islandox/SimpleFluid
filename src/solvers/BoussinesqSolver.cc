/**
 * @file BoussinesqSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for BoussinesqSolver.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "cmake/StandardHeaders.hh"

#if defined(SIMPLEFLUID_USE_CXX_MODULES)
import SimpleFluid.Solvers;
#else
#include "BoussinesqSolver.hh"
#endif

#include "BoussinesqSolver.tcc"

namespace SimpleFluid
{
template class BoussinesqSolver<DefaultTpetraTypes>;
}
