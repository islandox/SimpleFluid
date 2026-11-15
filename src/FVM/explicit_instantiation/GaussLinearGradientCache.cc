/**
 * @file GaussLinearGradientCache.cc
 * @brief  Explicit instantiations for GaussLinearGradientCache.
 */
#include "SimpleFluidExport.hh"
SIMPLEFLUID_LEGACY_API_BEGIN


#include "FVM/GaussLinearGradientCache.tcc"

namespace SimpleFluid::FVM
{
template class GaussLinearGradientCache<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class GaussLinearGradientCache<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid::FVM

SIMPLEFLUID_LEGACY_API_END
