/** @file LiquidMassInventory.cc @brief Explicit template instantiations. */
#include "solvers/LiquidMassInventory.tcc"

namespace SimpleFluid
{
template class LiquidMassInventory<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class LiquidMassInventory<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
