/**
 * @file TurbulenceEquations.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Convenience include for turbulence closures and scalar transport.
 *
 * Closure classes evaluate coefficients and local PDE terms. The scalar
 * transport class assembles one positive transported variable. Runtime model
 * ownership and momentum/temperature coupling live in TurbulenceModel.hh.
 * TurbulenceWallTreatment.hh provides the policy-based wall data used by the
 * runtime model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "equations/turbulence/BSLKOmegaEquation.hh"
#include "equations/turbulence/RNGKEpsilonEquation.hh"
#include "equations/turbulence/RealizableKEpsilonEquation.hh"
#include "equations/turbulence/SSTKOmegaEquation.hh"
#include "equations/turbulence/StandardKEpsilonEquation.hh"
#include "equations/turbulence/StandardKOmegaEquation.hh"
#include "equations/turbulence/TurbulenceScalarTransportEquation.hh"
#include "equations/turbulence/TurbulenceWallTreatment.hh"
