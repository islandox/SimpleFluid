/** @file AdaptivePipelineAudit.hh
 * @brief Internal opt-in synchronous allocation-phase and operation diagnostics. */
#pragma once
#include <cstddef>

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
struct PipelineObserver
{
    void* context = nullptr;
    PipelinePhaseObserver enter = nullptr;
    PipelineWorkObserver work = nullptr;
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
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
