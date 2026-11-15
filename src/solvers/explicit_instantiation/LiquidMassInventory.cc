/**
 * @file LiquidMassInventory.cc
 * @brief  Explicit template instantiations.
 */
#include "SimpleFluidExport.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "solvers/LiquidMassInventory.tcc"

namespace SimpleFluid
{
template class LiquidMassInventory<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class LiquidMassInventory<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid

SIMPLEFLUID_LEGACY_API_END
