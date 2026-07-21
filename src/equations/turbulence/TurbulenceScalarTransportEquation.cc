/**
 * @file TurbulenceScalarTransportEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit instantiation of positive turbulence scalar transport.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "TurbulenceScalarTransportEquation.hh"
#include "TurbulenceScalarTransportEquation.tcc"

namespace SimpleFluid
{
template class TurbulenceScalarTransportEquation<DefaultTpetraTypes>;
}
