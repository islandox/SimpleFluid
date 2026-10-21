/** @file DistributedProductKernel.hh
 * @brief Detached independent base/axis factor kernels for native products.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
namespace SimpleFluid::Meshes::AMR
{
[[nodiscard]] std::shared_ptr<const AdaptiveProviderKernel> product_adaptive_kernel();
[[nodiscard]] AdaptiveLeaf product_support_root(const AdaptiveLeaf&);
}
