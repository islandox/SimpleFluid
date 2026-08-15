/**
 * @file FVM/CellGradientScheme.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Cell-gradient reconstruction choices.
 * @version 0.1
 * @date 2026-07-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

namespace SimpleFluid::FVM
{

/** @brief Reconstruction used for cell-centered gradients. */
enum class CellGradientScheme
{
    LeastSquares,
    GaussLinear
};

} // namespace SimpleFluid::FVM
