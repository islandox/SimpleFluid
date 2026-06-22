/**
 * @file IncompressibleMomentumEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for IncompressibleMomentumEquation.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "IncompressibleMomentumEquation.hh"
#include "IncompressibleMomentumEquation.tcc"

namespace SimpleFluid
{
template class IncompressibleMomentumEquation<DefaultTpetraTypes>;
}
