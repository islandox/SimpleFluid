/** @file VolumeContinuityModel.cc @brief Explicit template instantiations. */
#include "solvers/VolumeContinuityModel.tcc"

namespace SimpleFluid
{
template class VolumeContinuityModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
