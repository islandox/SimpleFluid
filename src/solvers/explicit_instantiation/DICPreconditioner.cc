/**
 * @file DICPreconditioner.cc
 * @brief  Solver-owned explicit DIC instantiation.
 */

#include "solvers/DICPreconditioner.tcc"

namespace SimpleFluid::detail
{
template class DICPreconditioner<DefaultTpetraTypes>;
} // namespace SimpleFluid::detail
