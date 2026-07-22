/**
 * @file TemperatureDiffusionEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for TemperatureDiffusionEquation.
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
#include "TemperatureDiffusionEquation.hh"
#endif

#include "TemperatureDiffusionEquation.tcc"

namespace SimpleFluid
{
template class TemperatureDiffusionEquation<DefaultTpetraTypes>;
}
