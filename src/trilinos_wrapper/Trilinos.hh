/**
 * @file Trilinos.hh
 * @brief Aggregate module or textual Trilinos API used by SimpleFluid.
 *
 * This compatibility facade consumes TrilinosHeaderUnit.hh. The header-unit
 * payload deliberately does not include this facade or the package wrappers.
 */
#pragma once

#if defined(SIMPLEFLUID_USE_TRILINOS_MODULES)
import "trilinos_wrapper/TrilinosHeaderUnit.hh";
#else
#include "trilinos_wrapper/TrilinosHeaderUnit.hh"
#endif
