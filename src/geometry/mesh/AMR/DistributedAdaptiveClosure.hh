/** @file DistributedAdaptiveClosure.hh
 * @brief Sparse exact closed-contact physical and comparable-trace requirements.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
namespace SimpleFluid::Meshes::AMR
{
/** Rank-local construction diagnostics, including a rejected prefix. The spatial
 * route prepares owning span/trace records locally and fetches each matched remote
 * record once per owner/key/revision and session. It does not send full records
 * to point-hash authorities. geometry_records/trace_records count local preparation;
 * transmitted/decoded counters describe actual remote record traffic.
 * unique_record_bytes is local record header/key/domain admission; trace and
 * identity owners also enter live_peak_bytes. replicated_record_bytes counts
 * serialized remote records. Bytes are application estimates, not RSS.
 * Work counts performed unique span-pair and trace comparisons, rather than
 * repeated logical point incidences; paging never resets that counter. */
struct AdaptiveClosureStats
{
    std::size_t cells = 0, faces = 0, contacts = 0, geometry_records = 0, trace_records = 0;
    std::size_t transmitted_geometry_records = 0, decoded_geometry_records = 0;
    std::size_t unique_record_bytes = 0, replicated_record_bytes = 0;
    std::size_t decoded_geometry_bytes = 0; ///< Retained axis spans; trace faces enter live byte accounting separately.
    std::size_t cumulative_bytes = 0, live_peak_bytes = 0, input_retained_bytes = 0;
    std::size_t retained_geometry_bytes = 0; ///< Persistent request-phase source geometry and ledger admission.
    /** Maximum per-record admitted geometry/trace scratch estimate, including
     * certificate and decoded-shell overlap. Excludes the returned record and
     * existing nested encoder/writer-growth scratch outside this estimate. */
    std::size_t transient_geometry_peak_bytes = 0;
    std::size_t work = 0;
};
struct AdaptiveClosureOptions
{
    std::size_t byte_budget = 64 * 1024 * 1024, contact_budget = 1024 * 1024, tile_budget = 262144;
    double maximum_span_ratio = 2;
    unsigned maximum_trace_jump = 1;
    AdaptiveClosureStats* statistics = nullptr; ///< Optional local output; not collective option semantics.
};
/** Return requirements for owned cells using exact point matches with spatial
 * neighbors. One canonical support/edge bootstrap establishes complete incidence;
 * closed spatial covers discover candidates and complete owner replies establish
 * the halo. Physical point identities remain distinct from normalized keys.
 * Coincident independent domains remain separate. The operation is collective,
 * including empty owners, and leaves every source record unchanged on rejection. */
[[nodiscard]] std::vector<AdaptiveMark> required_adaptive_refinements(std::span<const AdaptiveLeaf> owned,
    const AdaptiveProviderKernel&, Teuchos::RCP<const Teuchos::Comm<int>>, AdaptiveClosureOptions = {});
} // namespace SimpleFluid::Meshes::AMR
