/**
 * @file LinearSolvers.hh
 * @brief Selects the module or textual Trilinos linear-solver API.
 */
#pragma once

#if defined(SIMPLEFLUID_USE_TRILINOS_MODULES)
import "trilinos_wrapper/TrilinosHeaderUnit.hh";
#else
#include "trilinos_wrapper/detail/LinearSolverHeaders.hh"
#endif
