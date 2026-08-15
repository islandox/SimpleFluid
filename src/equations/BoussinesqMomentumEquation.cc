/**
 * @file BoussinesqMomentumEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for BoussinesqMomentumEquation.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "BoussinesqMomentumEquation.hh"
#include "BoussinesqMomentumEquation.tcc"

namespace SimpleFluid
{
template class BoussinesqMomentumEquation<DefaultTpetraTypes>;
}
