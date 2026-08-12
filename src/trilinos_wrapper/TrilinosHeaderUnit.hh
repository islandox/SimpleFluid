/**
 * @file TrilinosHeaderUnit.hh
 * @brief Unconditional payload for the shared Trilinos header unit.
 *
 * This header contains only textual third-party declarations. Consumer-facing
 * wrappers may import it, but it must not include those wrappers in return.
 */
#pragma once

#include "trilinos_wrapper/detail/KokkosHeaders.hh"
#include "trilinos_wrapper/detail/TeuchosHeaders.hh"
#include "trilinos_wrapper/detail/TpetraHeaders.hh"
#include "trilinos_wrapper/detail/Zoltan2Headers.hh"
#include "trilinos_wrapper/detail/STKHeaders.hh"
#include "trilinos_wrapper/detail/LinearSolverHeaders.hh"
