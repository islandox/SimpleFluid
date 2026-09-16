/** @file FissionPowerSource.cc @brief Explicit template instantiations. */
#include "equations/FissionPowerSource.tcc"

namespace SimpleFluid
{
template class FissionPowerSource<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class FissionPowerSource<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
