#pragma once
#include <cstdint>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** General contact storage: identity pools, immutable point arrays, pages,
 * fact directories and allocation credits. These roles do not encode topology
 * or parallel membership semantics. */
enum class BasicContactPayloadRole : std::uint8_t
{
    PointArrayObject,
    PointArrayPhysical,
    PointArrayNormalized,
    PointArrayPhysicalRefs,
    PointArrayNormalizedRefs,
    IdentityPoolObject,
    IdentityBySlot,
    IdentityPageEntries,
    IdentityHashBuckets,
    IdentityOrderedValues,
    IdentityPageOwnerBuckets,
    IdentityPageCreditObject,
    IdentityPageObject,
    IdentityPageValues,
    IdentityDynamicBytes,
    IdentityCellPath,
    PageViewObject,
    PageViewDirectory,
    PageViewCaptureValues,
    FactDirectoryObject,
    FactPages,
    FactEntries,
    FactFreeEntries,
    FactFreePages,
    FactBuckets,
    FactPageIndexValues,
    FallbackCreditObject,
    FallbackBodyObject,
    FallbackValues,
    DeferredOwnerChargesObject,
    DeferredOwnerChargeNodes,
    IdentityPersistentRoot,
    IdentityPersistentIndex
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
