/** @file BoilingSourceModel.cc @brief Explicit template instantiations. */
#include "equations/BoilingSourceModel.tcc"

namespace SimpleFluid
{
template class BoilingSourceModel<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class BoilingSourceModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
