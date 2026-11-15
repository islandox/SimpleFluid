/** @file VariableDensityMomentumEquation.cc
 * @brief Supported conservative momentum equation explicit instantiations.
 */
#include "SimpleFluidExport.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "equations/VariableDensityMomentumEquation.tcc"
namespace SimpleFluid
{
template class VariableDensityMomentumEquation<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class VariableDensityMomentumEquation<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid

SIMPLEFLUID_LEGACY_API_END
