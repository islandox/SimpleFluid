/**
 * @file ScalarVoidFractionModel.cc
 * @brief  Explicit template instantiations.
 */
#include "utils/export_helper.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "equations/ScalarVoidFractionModel.tcc"

namespace SimpleFluid
{
template class ScalarVoidFractionModel<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class ScalarVoidFractionModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid

SIMPLEFLUID_LEGACY_API_END
