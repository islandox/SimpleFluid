/**
 * @file BelosLinearSolver.cc
 * @brief Explicit default-pack instantiation for the Belos solver wrapper.
 */

#include "solvers/BelosLinearSolver.hh"

namespace SimpleFluid
{

template class BelosLinearSolver<DefaultTpetraTypes>;

} // namespace SimpleFluid
