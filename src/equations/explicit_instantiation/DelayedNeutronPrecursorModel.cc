/** @file DelayedNeutronPrecursorModel.cc @brief Explicit template instantiations. */
#include "equations/DelayedNeutronPrecursorModel.tcc"

namespace SimpleFluid
{
template class DelayedNeutronPrecursorModel<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class DelayedNeutronPrecursorModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
