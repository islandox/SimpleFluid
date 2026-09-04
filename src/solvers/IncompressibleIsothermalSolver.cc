/**
 * @file IncompressibleIsothermalSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for IncompressibleIsothermalSolver.
 * @version 0.1
 * @date 2026-08-20
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "IncompressibleIsothermalSolver.hh"
#include "IncompressibleIsothermalSolver.tcc"

namespace SimpleFluid
{
template class IncompressibleIsothermalSolver<DefaultTpetraTypes>;
}
