/** @file DistributedAdaptiveClosure.hh
 * @brief Sparse exact closed-contact physical and comparable-trace requirements.
 */
#pragma once
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
namespace SimpleFluid::Meshes::AMR
{
struct AdaptiveClosureOptions
{
    std::size_t byte_budget = 64 * 1024 * 1024, contact_budget = 1024 * 1024, tile_budget = 262144;
    double maximum_span_ratio = 2;
    unsigned maximum_trace_jump = 1;
};
/** Return only requirements for cells owned by this rank. Complete canonical
 * faces and propagated edge points establish closed face/edge/vertex contacts;
 * coincident independent domains remain separate. The operation is collective
 * and leaves every source record unchanged on rejection. */
[[nodiscard]] std::vector<AdaptiveMark> required_adaptive_refinements(std::span<const AdaptiveLeaf> owned,
    const AdaptiveProviderKernel&, Teuchos::RCP<const Teuchos::Comm<int>>, AdaptiveClosureOptions = {});
} // namespace SimpleFluid::Meshes::AMR
