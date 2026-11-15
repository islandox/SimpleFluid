/**
 * @file BoilingSourceModel.cc
 * @brief  Explicit template instantiations.
 */
#include "utils/export_helper.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "equations/BoilingSourceModel.tcc"

namespace SimpleFluid
{
template class BoilingSourceModel<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class BoilingSourceModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid

SIMPLEFLUID_LEGACY_API_END
