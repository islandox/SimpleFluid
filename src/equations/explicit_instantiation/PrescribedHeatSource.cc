/**
 * @file PrescribedHeatSource.cc
 * @brief  Explicit template instantiations.
 */
#include "utils/export_helper.hh"

#include "equations/PrescribedHeatSource.tcc"

namespace SimpleFluid
{
template class PrescribedHeatSource<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
