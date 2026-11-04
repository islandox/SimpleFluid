#pragma once
#include "PooledLineRegistry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include <functional>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
using NextPooledLineEdit = std::function<bool(PooledLineEdit&)>;
using PooledLineReply = std::function<void(std::size_t line, const PooledLineSubscriber&)>;
/** Private fault-injection seam, applied after changed-query counts are minted.
 * The callback may alter only this page; the candidate is discarded on failure. */
using PooledLineBeforeSend = std::function<void(unsigned layer, std::uint64_t wave, std::vector<OwnerPacket>&)>;
/** Stream authenticated per-layer subscriptions, then recompute complete
 * dirty line queries. All ranks participate even with no local edits. */
void exchange_pooled_line_edits(PooledLineRegistry&, IncidencePools&, const NextPooledLineEdit&, unsigned layer,
    ContactHaloContext, Teuchos::RCP<const Teuchos::Comm<int>>, Budget&, std::size_t page_bytes, const PooledLineReply&,
    const PooledLineBeforeSend& = {});
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
