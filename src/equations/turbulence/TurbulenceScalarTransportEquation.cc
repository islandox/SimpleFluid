/**
 * @file TurbulenceScalarTransportEquation.cc
 * @brief Explicit instantiation of positive turbulence scalar transport.
 */

#include "TurbulenceScalarTransportEquation.hh"
#include "TurbulenceScalarTransportEquation.tcc"

namespace SimpleFluid
{
template class TurbulenceScalarTransportEquation<DefaultTpetraTypes>;
}
