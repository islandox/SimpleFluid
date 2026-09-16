/** @file ScalarVoidFractionModel.cc @brief Explicit template instantiations. */
#include "equations/ScalarVoidFractionModel.tcc"

namespace SimpleFluid
{
template class ScalarVoidFractionModel<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class ScalarVoidFractionModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
