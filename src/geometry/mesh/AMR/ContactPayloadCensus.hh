#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
enum class ContactPayloadRole : std::uint8_t
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
    SourcePointSnapshotObject,
    SourcePointComponents,
    SourcePointSelectors,
    SourcePointLookup,
    SourcePointExceptions,
    SourcePointLeases,
    SourcePointViewObject,
    SourcePointDirectoryObject,
    CartesianSourceObject,
    CartesianSourceDomain,
    CartesianSourceIdentity,
    CartesianSourceRanges,
    CartesianSourceAxes,
    CartesianSourceAxisValues,
    CartesianSourceRevisions,
    CartesianSourceKernel,
    CartesianInventoryObject,
    CartesianInventoryBatches,
    CartesianInventoryRanges,
    CartesianInventoryIdentity,
    CartesianInventoryLeases,
    CartesianManifestObject,
    CartesianManifestSources,
    CartesianManifestIdentity,
    CartesianManifestRanges,
    CartesianManifestLeases,
    CartesianArenaObject,
    CartesianArenaEntries,
    CartesianOwnerCreditObject,
    CartesianLineCatalogObject,
    CartesianLineCatalogImpl,
    CartesianLineCatalogMaps,
    CartesianLineAxisEntries,
    CartesianLineCatalogLeases,
    CartesianLineRecipeObject,
    CartesianLineRecipeImpl,
    CartesianLineBytes,
    CartesianLineFamilies,
    CartesianLinePrototype,
    CartesianLineMask,
    CartesianLineRanges,
    CartesianLineResultObject,
    CartesianLineResultPoints,
    CartesianControlCreditObject,
    SourcePointValueBlockObject,
    IncidenceSessionObject,
    IncidenceSessionImpl,
    OwnedMembershipNodes,
    OwnedMembershipCell,
    OwnedMembershipMembers,
    OwnedMembershipReverseNodes,
    OwnedMembershipSpill,
    OwnedMembershipValues,
    OwnedMembershipRetiredNodes,
    OwnedMembershipLeases,
    LineRegistryObject,
    LineRegistryNodes,
    LineRegistryDirtyNodes,
    LineRegistryTouchedNodes,
    LineBucketObject,
    LineSubscriberValues,
    LineRunValues,
    LinePatternValues,
    LineSpillObject,
    LineResultValues,
    LineSourceOwners,
    LineSourceLeases,
    TilePageDirectory,
    TilePageCreditObject,
    TilePageObject,
    TilePageRecords,
    TilePageRings,
    TileSourceCreditObject,
    TileSourceLeaseValues,
    SupportRegistryObject,
    SupportRegistryGroupNodes,
    SupportRegistryDirtyNodes,
    SupportMembershipSpill,
    SupportMembershipValues,
    CellPointsObject,
    CellPointCounts,
    CellPointsBefore,
    OwnerRowsObject,
    OwnerRowsCellNodes,
    OwnerRowsDirtyNodes,
    TileIndexObject,
    TileIndexPageDirectory,
    TileIndexCompactDescriptors,
    TileIndexWideDescriptors,
    TileIndexPageObject,
    DeferredOwnerChargesObject,
    DeferredOwnerChargeNodes,
    CertifiedIncidenceLoopObject,
    CertifiedIncidenceLoopParts,
    CertifiedIncidenceLoopLeases,
    IncidenceAuthorityCountNodes,
    IncidenceResidentGenerationNodes,
    IncidenceLayerHistoryObject,
    IncidenceLayerHistoryEntries,
    IncidenceLayerObject,
    IncidenceLayerInput,
    IncidenceLayerOutput,
    IncidenceLayerEdges,
    IncidenceLayerEdgeResults,
    ContactIncidenceRowObject,
    ContactIncidenceRowDomain,
    ContactSnapshotObject,
    ContactSnapshotStorage,
    ContactSlotBranch,
    ContactSlotLeaf,
    ContactRowGroupObject,
    ContactRowGroupSlots,
    ContactOwnedIdentityObject,
    ContactOwnedIdentityDomain,
    ContactOwnerUsageNode,
    ContactBodyObject,
    ContactTraceFaceSlots,
    ContactTraceStem,
    ContactTraceInventoryObject,
    CartesianExplicitCellHeaders,
    CartesianExplicitMemberHeaders,
    CartesianExplicitSupportKeys,
    IncidenceLayerEndpointFacts,
    IncidenceLayerLeases,
    IncidenceSessionLayerSourceLeases,
    CertifiedIncidenceResultPartsObject,
    PooledLineCompletionProofObject,
    PooledLineCompletionKeys,
    PooledLineCompletionResults,
    CertifiedIncidenceResultValueObject,
    CertifiedIncidenceResultValueMask,
    PooledLineCompletionObject,
    IncidenceLayerCompletions,
    CartesianLinePrototypeEntries,
    SubfaceSnapshotObject,
    SubfaceIndexRootObject,
    SubfaceIndexPageObject,
    SubfaceIndexChildren,
    SubfacePieceOwnerIndexRows,
    SubfaceProvenanceOwnerIndexRows,
    SubfacePieces,
    SubfaceSelectors,
    SubfacePiecePageObject,
    SubfacePieceRecords,
    SubfacePointRanges,
    SubfaceProvenancePageObject,
    SubfaceProvenanceRecords,
    SubfaceMembers,
    SubfaceGroups,
    SubfaceForeignKeys,
    SubfaceBoundaryNames,
    SubfaceBoundaryBytes,
    SubfaceShifts,
    SubfaceRingObject,
    SubfaceRingSlots,
    IdentityPersistentRoot,
    IdentityPersistentIndex
};
enum class ContactPayloadKind : std::uint8_t
{
    Object,
    Buffer,
    NodeValue
};
struct ContactPayloadIdentity
{
    const void* address;
    ContactPayloadRole role;
    bool operator==(const ContactPayloadIdentity&) const = default;
};
/** Actual, known C++ object/buffer payload only. No admission ledger or 128-byte
 * allowance enters this record. Unknown allocator, shared-control and tree-node
 * overhead is explicitly excluded. Identity is role-qualified; embedded objects
 * are omitted and accounted only by their containing object. */
struct ContactPayloadAllocation
{
    ContactPayloadIdentity identity;
    const void* owner;
    ContactPayloadKind kind;
    std::size_t object_size, element_size, live_count, capacity, payload_bytes;
    bool allocator_overhead_excluded = true;
    bool control_overhead_excluded = false;
    bool node_overhead_excluded = false;
    bool opaque_element_payload_excluded = false;
};
template<class Visitor>
void contact_payload_object(
    Visitor& visitor, const void* object, ContactPayloadRole role, std::size_t bytes, bool shared_control = false)
{
    visitor(ContactPayloadAllocation{
        {object, role}, object, ContactPayloadKind::Object, bytes, bytes, 1, 1, bytes, true, shared_control, false});
}
template<class Visitor, class T>
void contact_payload_buffer(Visitor& visitor, const void* owner, ContactPayloadRole role, const std::vector<T>& values,
    std::size_t live, bool opaque_element_payload = false)
{
    if (values.capacity())
        visitor(ContactPayloadAllocation{{values.data(), role}, owner, ContactPayloadKind::Buffer, 0, sizeof(T), live,
            values.capacity(), sizeof(T) * values.capacity(), true, false, false, opaque_element_payload});
}
template<class Visitor, class T>
void contact_payload_buffer(Visitor& visitor, const void* owner, ContactPayloadRole role, const std::vector<T>& values)
{
    contact_payload_buffer(visitor, owner, role, values, values.size());
}
template<class Visitor, class Pair>
void contact_payload_node_value(Visitor& visitor, const void* owner, ContactPayloadRole role, const Pair& value)
{
    visitor(ContactPayloadAllocation{{&value, role}, owner, ContactPayloadKind::NodeValue, sizeof(Pair), sizeof(Pair),
        1, 1, sizeof(Pair), true, false, true});
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
