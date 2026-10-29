/** @file DistributedAdaptiveClosure.hh
 * @brief Sparse exact closed-contact physical and comparable-trace requirements.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
namespace SimpleFluid::Meshes::AMR
{
/** Rank-local construction diagnostics, including a rejected prefix. Geometry
 * records are encoded once per owned leaf and transmitted at most once to each
 * actual point authority. Bytes are application estimates, distinct from RSS. */
struct AdaptiveClosureStats
{
    std::size_t cells = 0, faces = 0, contacts = 0, geometry_records = 0, trace_records = 0;
    std::size_t transmitted_geometry_records = 0, decoded_geometry_records = 0;
    std::size_t unique_record_bytes = 0, replicated_record_bytes = 0, decoded_geometry_bytes = 0;
    std::size_t cumulative_bytes = 0, live_peak_bytes = 0, input_retained_bytes = 0, retained_geometry_bytes = 0,
                work = 0;
};
struct AdaptiveClosureOptions
{
    std::size_t byte_budget = 64 * 1024 * 1024, contact_budget = 1024 * 1024, tile_budget = 262144;
    double maximum_span_ratio = 2;
    unsigned maximum_trace_jump = 1;
    AdaptiveClosureStats* statistics = nullptr; ///< Optional local output; not collective option semantics.
};
/** Return only requirements for cells owned by this rank. Complete canonical
 * faces and propagated edge points establish closed face/edge/vertex contacts;
 * coincident independent domains remain separate. The operation is collective
 * and leaves every source record unchanged on rejection. */
[[nodiscard]] std::vector<AdaptiveMark> required_adaptive_refinements(std::span<const AdaptiveLeaf> owned,
    const AdaptiveProviderKernel&, Teuchos::RCP<const Teuchos::Comm<int>>, AdaptiveClosureOptions = {});
} // namespace SimpleFluid::Meshes::AMR
