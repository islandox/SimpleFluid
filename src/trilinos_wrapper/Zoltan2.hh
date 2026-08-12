/**
 * @file Zoltan2.hh
 * @brief Selects the module or textual Zoltan2 API used by SimpleFluid.
 */
#pragma once

#if defined(SIMPLEFLUID_USE_TRILINOS_MODULES)
import "trilinos_wrapper/TrilinosHeaderUnit.hh";
#else
#include "trilinos_wrapper/detail/Zoltan2Headers.hh"
#endif
