/** @file DistributedAdaptivePackets.hh
 * @brief Bounded batch transport sharing procedural and saved-parent recipe parts.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
namespace SimpleFluid::geometry_detail
{
class NativeConstructionAdmission;
struct NativeConstructorStorageOwner;
}
namespace SimpleFluid::Meshes::AMR
{
struct AdaptiveBatch
{
    std::vector<AdaptiveLeaf> leaves;
    std::vector<AdaptiveFamily> families;
    std::size_t shared_part_bytes = 0;
};
/** One dictionary covers active leaves, saved parents, native descriptor views
 * and their shared construction parts. Complete bytes,
 * never hashes or pointer values, authenticate dictionary contents. The budget
 * bounds the wire packet and dictionary payload/metadata before insertion.
 * Decoded provider objects, caller-owned records and allocator overhead are
 * reported by their owning state, rather than as a codec peak-memory claim. */
[[nodiscard]] AdaptiveBytes encode_adaptive_batch(
    std::span<const AdaptiveLeaf>, std::span<const AdaptiveFamily>, std::size_t byte_budget);
[[nodiscard]] AdaptiveBatch decode_adaptive_batch(std::span<const char>, std::size_t byte_budget);
namespace adaptive_detail
{
/** Encode the existing dictionary and records directly in the caller's
 * admitted packet storage. The selected source has already been validated. */
[[nodiscard]] SIMPLEFLUID_LOCAL AdaptiveBytes encode_adaptive_batch_admitted(
    std::span<const AdaptiveLeaf>, std::span<const AdaptiveFamily>,
    const geometry_detail::NativeConstructorStorageOwner&);
/** Construct the actual explicit cell and planar owners in an admitted domain.
 * A synchronous caller can inspect the received inventory before its staged
 * row buffers and decoder metadata release their admission. */
using AdmittedBatchVisit = void (*)(void*, const AdaptiveBatch&);
[[nodiscard]] SIMPLEFLUID_LOCAL AdaptiveBatch decode_adaptive_batch_admitted_geometry(
    std::span<const char>, std::size_t byte_budget, const geometry_detail::NativeConstructionAdmission&,
    void* context = nullptr, AdmittedBatchVisit = nullptr);
}
} // namespace SimpleFluid::Meshes::AMR
