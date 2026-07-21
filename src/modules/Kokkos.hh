/**
 * @file Kokkos.hh
 * @brief Header and module boundary for the Kokkos API used by SimpleFluid.
 */
#pragma once

#if defined(SIMPLEFLUID_USE_TRILINOS_MODULES)
import "modules/Trilinos.hh";
#else
#include <Kokkos_Core.hpp>
#endif
