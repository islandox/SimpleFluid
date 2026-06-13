/**
 * @file IncompressibleMomentumEquation.cc
 * @brief Explicit template instantiation for IncompressibleMomentumEquation.
 */

#include "IncompressibleMomentumEquation.hh"
#include "IncompressibleMomentumEquation.tcc"

namespace SimpleFluid
{
template class IncompressibleMomentumEquation<DefaultTpetraTypes>;
}
