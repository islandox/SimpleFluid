/** @file LowMachQbmmControllerRestart.hh
 * @brief Private diagnostic payload shared by native and controller QBMM restart entry points.
 */
#pragma once

#include "solvers/ReactingMixtureCoupler.hh"

namespace SimpleFluid::Reacting
{
struct LowMachMixtureSolver::ControllerRestartDiagnostics
{
    size_t reaction_count = 0;
    const ReactingMixtureCoupler::Diagnostics* saved = nullptr;
    ReactingMixtureCoupler::Diagnostics* restored = nullptr;
};
} // namespace SimpleFluid::Reacting
