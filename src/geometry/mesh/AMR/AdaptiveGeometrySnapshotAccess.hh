/** @file AdaptiveGeometrySnapshotAccess.hh
 * @brief Private mesh acquisition from an already authenticated native publication draft.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
#include <cstddef>
#include <memory>
#include <span>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Only native-publication consumers may call this seam, after the owner fetch
 * and authority phases have produced the complete draft. Supplied row addresses
 * must belong to that draft. This is not a provider-input admission API. */
class GeometrySnapshotAccess
{
public:
    [[nodiscard]] static std::shared_ptr<const AdaptiveGeometrySnapshot> native_draft(
        const DistributedAdaptiveHierarchy&, const AdaptiveNativeDraft&, std::span<const AdaptiveLeaf* const>,
        const HierarchyObservation&, ArithmeticMode, GeometrySnapshotPurpose, std::size_t& used, std::size_t limit,
        PLIC::GeometryAllocationLedger&, PLIC::GeometryAllocationLedger&);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
