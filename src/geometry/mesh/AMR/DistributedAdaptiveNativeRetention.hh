/** @file DistributedAdaptiveNativeRetention.hh
 * @brief Private policy for publication-only procedural draft retention.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
inline bool compact_native_geometry(const AdaptiveLeaf& source, bool compact_procedural) noexcept
{
    return compact_procedural && !source.geometry && source.native_recipe &&
           (source.provider == AdaptiveProviderKind::Cartesian || source.provider == AdaptiveProviderKind::Product);
}
/** Local owned-record admission; append accounts already materialized input,
 * while append_bounded also leases the opted-in provider's construction
 * envelope. Authority, bindings, halo/publication and process memory are separate.
 * Capacity/payload and tracker bytes follow the hierarchy storage scope:
 * allocator and shared control-block overhead are excluded. */
struct CompactNativeOwnedStats
{
    std::size_t leaves = 0, retained_bytes = 0, live_peak_bytes = 0, cumulative_bytes = 0;
    std::size_t serialized_bytes = 0, transient_geometry_peak_bytes = 0, encoding_scratch_peak_bytes = 0;
    std::size_t materialization_peak_bytes = 0;
};
/** Destination and all appended immutable owners must outlive this ledger.
 * No temporary full geometry enters its persistent owner registry. Failure
 * poisons admission; discard the ledger and destination as one transaction. */
class CompactNativeOwnedBudget
{
public:
    explicit CompactNativeOwnedBudget(std::size_t byte_limit);
    ~CompactNativeOwnedBudget();
    CompactNativeOwnedBudget(const CompactNativeOwnedBudget&) = delete;
    CompactNativeOwnedBudget& operator=(const CompactNativeOwnedBudget&) = delete;
    void append(std::vector<AdaptiveLeaf>& destination, AdaptiveLeaf transient, bool discard_geometry);
    /** Return false without materialization when the actual provider has no
     * proved construction envelope. The caller retains its legacy guard. */
    [[nodiscard]] bool append_bounded(
        std::vector<AdaptiveLeaf>& destination, const AdaptiveLeaf& source, const AdaptiveProviderKernel&);
    [[nodiscard]] CompactNativeOwnedStats statistics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
