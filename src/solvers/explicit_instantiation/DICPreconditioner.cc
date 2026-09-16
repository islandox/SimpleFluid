/** @file DICPreconditioner.cc @brief Explicit instantiations for DICPreconditioner. */

#include "solvers/DICPreconditioner.tcc"

namespace SimpleFluid::detail
{
template class DICPreconditioner<DefaultTpetraTypes>;
} // namespace SimpleFluid::detail
