/**
 * @file TurbulenceEquations.hh
 * @brief Convenience include for turbulence closures and scalar transport.
 *
 * Closure classes evaluate coefficients and local PDE terms. The scalar
 * transport class assembles one positive transported variable. Runtime model
 * ownership and momentum/temperature coupling live in TurbulenceModel.hh.
 * Wall functions are not provided by this layer.
 */

#pragma once

#include "equations/turbulence/BSLKOmegaEquation.hh"
#include "equations/turbulence/RNGKEpsilonEquation.hh"
#include "equations/turbulence/RealizableKEpsilonEquation.hh"
#include "equations/turbulence/SSTKOmegaEquation.hh"
#include "equations/turbulence/StandardKEpsilonEquation.hh"
#include "equations/turbulence/StandardKOmegaEquation.hh"
#include "equations/turbulence/TurbulenceScalarTransportEquation.hh"
