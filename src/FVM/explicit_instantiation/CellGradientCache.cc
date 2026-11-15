/**
 * @file CellGradientCache.cc
 * @brief  Explicit template instantiations.
 */
#include "SimpleFluidExport.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "FVM/CellGradientCache.tcc"

namespace SimpleFluid::diagnostics
{
std::atomic<int>& pressure_gradient_build_depth() noexcept
{
    static std::atomic<int> depth{0};
    return depth;
}
} // namespace SimpleFluid::diagnostics

namespace SimpleFluid::FVM
{
template class CellGradientCache<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
template class CellGradientCache<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid::FVM

SIMPLEFLUID_LEGACY_API_END
