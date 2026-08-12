/**
 * @file Kokkos.hh
 * @brief Selects the module or textual Kokkos API used by SimpleFluid.
 */
#pragma once

#if defined(SIMPLEFLUID_USE_TRILINOS_MODULES)
import "trilinos_wrapper/TrilinosHeaderUnit.hh";
#else
#include "trilinos_wrapper/detail/KokkosHeaders.hh"
#endif
