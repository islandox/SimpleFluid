/**
 * @file PrescribedHeatSource.cc
 * @brief  Explicit template instantiations.
 */
#include "utils/export_helper.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "equations/PrescribedHeatSource.tcc"

namespace SimpleFluid
{
template class PrescribedHeatSource<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class PrescribedHeatSource<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid

SIMPLEFLUID_LEGACY_API_END
