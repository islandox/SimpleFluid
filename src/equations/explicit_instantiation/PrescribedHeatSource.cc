/** @file PrescribedHeatSource.cc @brief Explicit template instantiations. */
#include "equations/PrescribedHeatSource.tcc"

namespace SimpleFluid
{
template class PrescribedHeatSource<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class PrescribedHeatSource<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
