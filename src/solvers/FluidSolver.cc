/**
 * @file FluidSolver.cc
 * @brief Explicit template instantiation for FluidSolver.
 */

#include "FluidSolver.hh"
#include "FluidSolver.tcc"

namespace SimpleFluid
{
template class FluidSolver<DefaultTpetraTypes>;
}
