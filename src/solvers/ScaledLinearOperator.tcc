/**
 * @file ScaledLinearOperator.tcc
 * @brief Out-of-line key function for the scaled CRS operator wrapper.
 */
#pragma once

#include "solvers/ScaledLinearOperator.hh"

namespace SimpleFluid::detail
{

template<TpetraTypePack Pack>
ScaledLinearOperator<Pack>::~ScaledLinearOperator() = default;

} // namespace SimpleFluid::detail
