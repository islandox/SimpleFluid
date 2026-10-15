/**
 * @file MultiphysicsSolverCoupler.cc
 * @brief Compile the supported multiphysics coupler specialization.
 */
#include "solvers/MultiphysicsSolverCoupler.tcc"

namespace SimpleFluid
{
template class MultiphysicsSolverCoupler<DefaultTpetraTypes>;
}
