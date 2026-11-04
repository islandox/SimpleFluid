/** @file SupportGroupFetch.hh
 * @brief Authenticated, page-bounded complete support member fetch.
 */
#pragma once
#include "geometry/mesh/AMR/CompleteSupportOverlay.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIncidence.hh"

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class PooledSupportRegistry;
struct RegistryMember;
/** Borrowed occurrences and members exist only for this callback. The callback
 * may publish only candidate-private data and must admit every retained owner. */
using CompleteGroupEmit =
    std::function<void(std::span<const char>, std::span<const RegistryMember>, std::span<const SupportOccurrence>)>;
/** Called on an actual member owner for every authenticated request. Duplicates
 * are permitted; a receiver building a delivery manifest must deduplicate. */
using CompleteGroupExpected = std::function<void(std::span<const char>)>;
using CompleteGroupFlush = std::function<void()>;
/** Every rank participates in every byte-bounded page, including empty ranks.
 * Multiple groups share a page; groups are emitted complete, in key order, and
 * members retain registry arrival order. Per-group decoded geometry dies before
 * the next callback. Flush is collective and occurs once per nonempty global page. */
void fetch_dirty_support_groups(const PooledSupportRegistry&, const OwnedLeafLookup&, ContactHaloContext,
    Teuchos::RCP<const Teuchos::Comm<int>>, Budget&, IdentityBudget&, std::size_t page_members,
    const CompleteGroupExpected&, const CompleteGroupEmit&, const CompleteGroupFlush&);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
