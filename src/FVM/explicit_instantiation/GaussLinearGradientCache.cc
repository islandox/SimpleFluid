/** @file GaussLinearGradientCache.cc @brief Explicit instantiations for GaussLinearGradientCache. */

#include "FVM/GaussLinearGradientCache.tcc"

namespace SimpleFluid::FVM
{
template class GaussLinearGradientCache<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class GaussLinearGradientCache<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid::FVM
