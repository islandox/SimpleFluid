/**
 * @file TemperatureDiffusionEquation.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit template instantiation for TemperatureDiffusionEquation.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "equations/TemperatureDiffusionEquation.hh"
#include "equations/TemperatureDiffusionEquation.tcc"

namespace SimpleFluid
{
template class TemperatureDiffusionEquation<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class TemperatureDiffusionEquation<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
