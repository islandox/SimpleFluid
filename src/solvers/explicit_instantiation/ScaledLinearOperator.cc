/**
 * @file ScaledLinearOperator.cc
 * @brief Solver-owned explicit scaled-operator instantiation.
 */

#include "solvers/ScaledLinearOperator.hh"
#include "solvers/ScaledLinearOperator.tcc"

namespace SimpleFluid::detail
{
template class ScaledLinearOperator<DefaultTpetraTypes>;
} // namespace SimpleFluid::detail
