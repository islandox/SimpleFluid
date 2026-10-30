/** @file DistributedAdaptiveNativeRetention.hh
 * @brief Private policy for publication-only procedural draft retention.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
inline bool compact_native_geometry(const AdaptiveLeaf& source, bool compact_procedural) noexcept
{
    return compact_procedural && !source.geometry && source.native_recipe &&
           (source.provider == AdaptiveProviderKind::Cartesian || source.provider == AdaptiveProviderKind::Product);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
