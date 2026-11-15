/** @file VariableDensityMomentumEquation.cc
 * @brief Supported conservative momentum equation explicit instantiations.
 */
#include "utils/export_helper.hh"

#include "equations/VariableDensityMomentumEquation.tcc"
namespace SimpleFluid
{
template class VariableDensityMomentumEquation<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
