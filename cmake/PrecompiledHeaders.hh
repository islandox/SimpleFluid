/**
 * @file PrecompiledHeaders.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Stable headers shared by most SimpleFluid translation units.
 * @version 0.1
 * @date 2026-06-03
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "cmake/StandardHeaders.hh"

#if defined(__ELF__) && defined(_LIBCPP_VERSION)
// Static Trilinos is linked into both SimpleFluidSolvers and its consumers.
// libc++ requires its polymorphic template types to keep non-unique RTTI for
// cross-image dynamic_cast, while SimpleFluid implementation symbols remain
// hidden by the surrounding target visibility policy.
#  pragma GCC visibility push(default)
#endif

#include "modules/LinearSolvers.hh"
#include "modules/Kokkos.hh"
#include "modules/Teuchos.hh"
#include "modules/Tpetra.hh"

#if defined(__ELF__) && defined(_LIBCPP_VERSION)
#  pragma GCC visibility pop
#endif

#include "modules/STK.hh"
