/**
 * @file LiquidMassInventory.cc
 * @brief  Explicit template instantiations.
 */
#include "utils/export_helper.hh"

#include "solvers/LiquidMassInventory.tcc"

namespace SimpleFluid
{
template class LiquidMassInventory<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
