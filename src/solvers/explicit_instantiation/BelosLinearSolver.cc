/**
 * @file BelosLinearSolver.cc
 * @brief Explicit template instantiation for the Belos linear solver wrapper.
 */
#include "solvers/BelosLinearSolver.hh"
#include "solvers/BelosLinearSolver.tcc"

namespace SimpleFluid
{
template class BelosLinearSolver<DefaultTpetraTypes>;
}
