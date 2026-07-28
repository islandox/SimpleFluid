/**
 * @file FVM/FaceCoefficientInterpolation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Interior-face interpolation choices for transport coefficients.
 * @version 0.1
 * @date 2026-07-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

namespace SimpleFluid::FVM
{

/**
 * @brief Interpolation used for positive cell-centered diffusion coefficients.
 *
 * Harmonic interpolation preserves resistance across discontinuous material
 * interfaces. Linear interpolation matches OpenFOAM's default `linear`
 * interpolation scheme for smoothly varying effective RANS properties.
 */
enum class FaceCoefficientInterpolation
{
    Harmonic,
    Linear
};

} // namespace SimpleFluid::FVM
