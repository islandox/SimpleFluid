/** @brief Compiled gradient-cache support for solid scalar operators. */
#include "FVM/CellGradientCache.tcc"
#include "geometry/SolidSubdomain.hh"

namespace SimpleFluid::FVM
{
template class CellGradientCache<DefaultTpetraTypes, SolidSubdomain<DefaultTpetraTypes>>;
} // namespace SimpleFluid::FVM
