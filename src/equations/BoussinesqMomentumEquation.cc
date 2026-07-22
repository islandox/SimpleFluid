/**
 * @file BoussinesqMomentumEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for BoussinesqMomentumEquation.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "cmake/StandardHeaders.hh"

#if defined(SIMPLEFLUID_USE_CXX_MODULES)
import SimpleFluid.Equations;
#else
#include "BoussinesqMomentumEquation.hh"
#endif

#include "BoussinesqMomentumEquation.tcc"

namespace SimpleFluid
{
template class BoussinesqMomentumEquation<DefaultTpetraTypes>;
}
