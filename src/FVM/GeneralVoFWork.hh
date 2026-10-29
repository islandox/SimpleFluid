/** @file GeneralVoFWork.hh
 * @brief Optional source-pinned measurements for phase partition forecasts.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include "utils/ArithmeticMode.hh"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace SimpleFluid::VoF
{
enum class GeneralVoFTransportBackend;
/** Executed geometry counters; fit_iterations below is the returned angular-fit diagnostic. */
struct GeneralVoFWorkMetrics
{
    double seconds = 0;
    std::uint64_t floating_operations = 0, volume_evaluations = 0, intersections = 0, constructions = 0;
};
struct GeneralVoFWorkCell
{
    Meshes::AMR::AdaptiveCellKey key;
    std::uint64_t revision = 0;
    int owner = -1;
    GeneralVoFWorkMetrics reconstruction;
    std::size_t calls = 0, mixed_fits = 0, retained_returns = 0, pure_returns = 0, zero_phase_returns = 0;
    unsigned fit_iterations = 0;
    bool mixed = false;
    /** Per-request complete encoded envelopes, also measured for local-only fitted donors.
     * Zero means no reconstruction/model is available, not zero transfer cost. */
    std::size_t request_payload_bytes = 0, reply_payload_bytes = 0;
};
struct GeneralVoFWorkDonor
{
    Meshes::AMR::AdaptiveCellKey key;
    std::uint64_t revision = 0;
    int owner = -1;
};
struct GeneralVoFWorkFace
{
    Meshes::AMR::AdaptiveBytes identity;
    GeneralVoFWorkDonor first, second;
    bool geometric = false;
    GeneralVoFWorkMetrics setup, aggregate;
    /** All candidates actually requested for this geometric face, including local donors. */
    std::vector<GeneralVoFWorkDonor> donors;
};
struct GeneralVoFWorkReceiver
{
    GeneralVoFWorkDonor donor;
    int requester = -1;
    bool mixed = false;
    GeneralVoFWorkMetrics remeasurement;
    std::size_t request_payload_bytes = 0, reply_payload_bytes = 0;
};
/** Immutable once bound to a prepared ledger. Keys/revisions include zero-work owned rows.
 * No cache of planes or phase geometry is retained. Times exclude communication/collectives. */
struct GeneralVoFWorkSample
{
    std::shared_ptr<const void> source_identity, mesh_identity, topology_identity;
    Meshes::AMR::HierarchyObservation observation;
    const void* communicator = nullptr;
    std::uint64_t source_control = 0, time_stage = 0;
    double timestep = 0;
    ArithmeticMode arithmetic_mode = ArithmeticMode::Ordinary;
    GeneralVoFTransportBackend backend{};
    bool fully_deduplicated = true;
    std::vector<GeneralVoFWorkCell> cells;
    std::vector<GeneralVoFWorkFace> faces;
    std::vector<GeneralVoFWorkReceiver> receivers;
    /** Actual retained allocation including vector capacities and deep adaptive keys. */
    std::size_t retained_bytes = 0;
};
} // namespace SimpleFluid::VoF
