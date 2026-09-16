/** @file CellGradientCache.cc @brief Explicit template instantiations. */
#include "FVM/CellGradientCache.tcc"

namespace SimpleFluid::FVM
{
template class CellGradientCache<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class CellGradientCache<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid::FVM
