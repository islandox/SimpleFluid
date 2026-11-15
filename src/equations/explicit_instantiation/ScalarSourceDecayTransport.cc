/**
 * @file ScalarSourceDecayTransport.cc
 * @brief  Explicit template instantiations.
 */
#include "SimpleFluidExport.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "equations/ScalarSourceDecayTransport.tcc"

namespace SimpleFluid
{
template class ScalarSourceDecayTransport<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class ScalarSourceDecayTransport<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid

SIMPLEFLUID_LEGACY_API_END
