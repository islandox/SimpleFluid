/**
 * @file Zoltan2.hh
 * @brief Header and module boundary for the Zoltan2 API used by SimpleFluid.
 */
#pragma once

#if defined(SIMPLEFLUID_USE_TRILINOS_MODULES)
import "modules/Trilinos.hh";
#else
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_TpetraRowGraphAdapter.hpp>
#endif
