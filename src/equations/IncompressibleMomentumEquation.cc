/**
 * @file IncompressibleMomentumEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for IncompressibleMomentumEquation.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "cmake/StandardHeaders.hh"

#if defined(SIMPLEFLUID_USE_CXX_MODULES)
import SimpleFluid.Equations;
#else
#include "IncompressibleMomentumEquation.hh"
#endif

#include "IncompressibleMomentumEquation.tcc"

namespace SimpleFluid
{
template class IncompressibleMomentumEquation<DefaultTpetraTypes>;
}
