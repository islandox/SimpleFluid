/** Hierarchy-backed, privately staged native publication transaction. */
#pragma once
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/mesh/AMR/NativeSourceCoverage.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include <optional>
namespace SimpleFluid::Meshes::AMR { struct AdaptiveGeometrySnapshotCache; }
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** No operation/controller reference escapes into a persistent native owner. */
class SIMPLEFLUID_LOCAL NativePublicationOperation final
{
public:
    explicit NativePublicationOperation(const DistributedAdaptiveHierarchy&);
    ~NativePublicationOperation();
    NativePublicationOperation(const NativePublicationOperation&) = delete;
    NativePublicationOperation& operator=(const NativePublicationOperation&) = delete;
    NativePublicationOperation(NativePublicationOperation&&) = delete;
    [[nodiscard]] const geometry_detail::NativeConstructionAdmission& admission() const noexcept;
    [[nodiscard]] PLIC::details::NativeGeometryAccounting& geometry_accounting() noexcept;
    [[nodiscard]] const NativePublicationInputs& inputs() const noexcept { return inputs_; }
    [[nodiscard]] const std::shared_ptr<const NativePublicationRoot>& previous_native_root() const noexcept
    { return accepted_native_.root(); }
    [[nodiscard]] NativeSourceCoverageTransaction& source_coverage_transaction() noexcept { return *coverage_tx_; }
    void prepare_source_coverage();
    void stage_source(NativeSourceOwnerStore);
    void stage_native_result(std::shared_ptr<const NativePublicationRoot>);
    /** All throwing work, including lease growth and revision-token acquisition,
     * is finished before the caller's final collective agreement. */
    void prepare_commit(NativePublicationSession&);
    void commit(NativePublicationSession&) noexcept;
private:
    const DistributedAdaptiveHierarchy* hierarchy_;
    NativePublicationInputs inputs_;
    std::shared_ptr<const void> source_data_;
    std::shared_ptr<AdaptiveGeometrySnapshotCache> cache_;
    PLIC::details::GeometryOwnerUnion accepted_union_, staged_union_;
    NativeSourceOwnerStore accepted_source_, staged_source_;
    NativeGeometryRootStore accepted_native_, staged_native_;
    SourceAccountingPartition partition_;
    std::optional<PLIC::details::SealedSourcePrefix> prefix_;
    std::optional<geometry_detail::NativeConstructionAdmission> admission_;
    std::optional<PLIC::details::NativeGeometryAccounting> accounting_;
    std::optional<NativeSourceCoverageTransaction> coverage_tx_;
    std::optional<ContactExternalResidencyLease> growth_;
    std::optional<ContactExternalResidencyLease::PreparedTransfer> transfer_;
    std::uint64_t cache_revision_ = 0, publication_revision_ = 0;
    bool source_staged_ = false, native_staged_ = false, token_ = false, committed_ = false;
};
[[nodiscard]] inline NativePublicationOperation begin_native_operation(const DistributedAdaptiveHierarchy& h)
{
    return NativePublicationOperation(h);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
