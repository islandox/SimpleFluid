/**
 * @file BoilingSourceModel.cc
 * @brief  Explicit template instantiations.
 */
#include "utils/export_helper.hh"

#include "equations/BoilingSourceModel.tcc"

namespace SimpleFluid
{
template class BoilingSourceModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
