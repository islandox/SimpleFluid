/** @file AdaptivePipelineAudit.hh
 * @brief Internal opt-in synchronous allocation-phase and operation diagnostics. */
#pragma once
#include <cstddef>

namespace SimpleFluid::PLIC::details { class NativeGeometryAccounting; }

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
enum class PipelinePhase : unsigned
{
    Candidate,
    Transport,
    Directory,
    NativePublication
};
/** Event units are individual call sites, not inferred primitive constructions.
 * PLIC geometry_statistics supplies actual nested construction counts. ActiveLeafVisit
 * covers Data::visit_active only; RequiredWireRecord covers leased lookup keys. */
enum class PipelineWork : unsigned
{
    ActiveLeafVisit,
    RequiredWireRecord,
    NumericOutput,
    CellGeometryRequest,
    SurfaceGeometryRequest,
    PhysicalShellReconstruction,
    NativeMaterializationRequest,
    NativeFaceCertificateRequest,
    OrderedPageCreated,
    OrderedPageAdmittedBytes,
    LeafPageCreated,
    LeafRowDereference,
    WireSentPackets,
    WireReceivedPackets,
    WireSentBytes,
    WireReceivedBytes,
    WireEncodedLeaves,
    WireEncodedFamilies,
    WireDecodedLeaves,
    WireDecodedFamilies,
    WireEncodedProductFrames,
    WireDecodedProductFrames,
    Count
};
using PipelineWorkObserver = void (*)(void*, PipelineWork, std::size_t) noexcept;
using PipelinePhaseObserver = void (*)(void*, PipelinePhase) noexcept;
enum class PipelineNativeOwnerStage { Output, Source, Union };
struct PipelineNativeOwnerAdmission
{
    std::size_t live, peak, cumulative, limit;
    std::size_t index_spent, index_limit;
};
using PipelineNativeOwnerObserver =
    void (*)(void*, PipelineNativeOwnerStage, bool entering, PipelineNativeOwnerAdmission) noexcept;
struct PipelineObserver
{
    void* context = nullptr;
    PipelinePhaseObserver enter = nullptr;
    PipelineWorkObserver work = nullptr;
    PipelineNativeOwnerObserver native_owner = nullptr;
};
/** Registration belongs to the calling thread. Nested registrations and phases
 * restore their predecessors during stack unwinding. No callback may allocate. */
[[nodiscard]] PipelineObserver set_pipeline_observer(PipelineObserver) noexcept;
[[nodiscard]] PipelinePhase switch_pipeline_phase(PipelinePhase) noexcept;
void record_pipeline_work(PipelineWork, std::size_t count = 1) noexcept;
class PipelinePhaseScope
{
public:
    explicit PipelinePhaseScope(PipelinePhase phase) noexcept : previous_(switch_pipeline_phase(phase)) {}
    ~PipelinePhaseScope() { (void) switch_pipeline_phase(previous_); }

private:
    PipelinePhase previous_;
};

/** Samples the operation's live counter (including its prefix) and the
 * independent cumulative index cursor at the boundaries of owner staging. */
class PipelineNativeOwnerScope
{
public:
    PipelineNativeOwnerScope(PipelineNativeOwnerStage, PLIC::details::NativeGeometryAccounting&) noexcept;
    PipelineNativeOwnerScope(const PipelineNativeOwnerScope&) = delete;
    PipelineNativeOwnerScope& operator=(const PipelineNativeOwnerScope&) = delete;
    ~PipelineNativeOwnerScope();

private:
    void observe(bool entering) const noexcept;
    PipelineNativeOwnerStage stage_;
    PLIC::details::NativeGeometryAccounting& accounting_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
