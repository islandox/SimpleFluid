/**
 * @file PressureProjectionEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for PressureProjectionEquation.
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
#include "PressureProjectionEquation.hh"
#endif

#include "PressureProjectionEquation.tcc"

namespace SimpleFluid
{
template class PressureProjectionEquation<DefaultTpetraTypes>;
}
