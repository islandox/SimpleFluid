#pragma once
#include <cstdint>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Parallel support mechanisms: ownership and support membership, groups,
 * fan bindings, halo edits and exchange completion. These identify storage for
 * communication and association rather than a geometric construction. */
enum class ParallelSupportPayloadRole : std::uint8_t
{
    GroupObject,
    GroupSlots,
    GroupFreeSlots,
    GroupBuckets,
    ReaderObject,
    HaloGroupEditPositions,
    HaloFanEdits,
    HaloRemovedBindings,
    HaloAppendedLocal,
    HaloAppendedRemote,
    BindingCatalogObject,
    BindingEntries,
    BindingFreeSlots,
    BindingBuckets,
    FanOwnerObject,
    FanViews,
    FanBlockObject,
    FanLocal,
    FanRemote,
    FanSlices,
    FanOwnerIndexBuckets,
    PhysicalOwnerIndexObject,
    PhysicalOwnerBuckets,
    OwnedMembershipNodes,
    OwnedMembershipCell,
    OwnedMembershipMembers,
    OwnedMembershipReverseNodes,
    OwnedMembershipSpill,
    OwnedMembershipValues,
    OwnedMembershipRetiredNodes,
    OwnedMembershipLeases,
    SupportRegistryObject,
    SupportRegistryGroupNodes,
    SupportRegistryDirtyNodes,
    SupportMembershipSpill,
    SupportMembershipValues,
    IncidenceAuthorityCountNodes,
    IncidenceResidentGenerationNodes,
    PooledLineCompletionProofObject,
    PooledLineCompletionKeys,
    PooledLineCompletionResults,
    PooledLineCompletionObject,
    HaloResidencyCounterObject,
    HaloControllerObject,
    HaloBudgetObject,
    HaloExternalLeaseCounter
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
