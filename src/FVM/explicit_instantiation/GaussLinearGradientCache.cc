/**
 * @file GaussLinearGradientCache.cc
 * @brief  Explicit instantiations for GaussLinearGradientCache.
 */
#include "utils/export_helper.hh"

#include "FVM/GaussLinearGradientCache.tcc"

namespace SimpleFluid::FVM
{
template class GaussLinearGradientCache<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid::FVM
