/**
 * @file ScalarVoidFractionModel.cc
 * @brief  Explicit template instantiations.
 */
#include "utils/export_helper.hh"

#include "equations/ScalarVoidFractionModel.tcc"

namespace SimpleFluid
{
template class ScalarVoidFractionModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
