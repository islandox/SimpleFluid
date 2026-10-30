/** @file SparseOwnerExchange.hh
 * @brief Budgeted variable-length peer packets without payload-width padding.
 */
#pragma once
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <cstddef>
#include <span>
#include <vector>
namespace SimpleFluid
{
struct OwnerPacket
{
    int peer = -1; ///< Destination on input; authenticated sender on output.
    std::vector<char> bytes;
};
struct OwnerExchangeStats
{
    std::size_t sent_packets = 0, received_packets = 0, sent_bytes = 0, received_bytes = 0, peak_scratch_bytes = 0;
};
struct OwnerExchange
{
    std::vector<OwnerPacket> packets;
    OwnerExchangeStats stats;
};
/** Collective, including empty ranks. One packed stream per actual peer uses a
 * Tpetra Distributor; packet sizes never pad another packet's payload. The byte
 * budget bounds the maximum live phase of packed buffers, output and explicit
 * framing/route metadata before allocations. Packed sends are destroyed after
 * transfer and before decoding; received bytes coexist with decoded output.
 * Stats count these buffers, not MPI/allocator/control-block RSS. Borrowed input
 * is excluded; diagnostics report its packet range and payload capacities.
 * Duplicate peers and zero-byte records are supported; decoding is collective.
 * A call rejects packed byte/record counts above INT_MAX before payload exchange;
 * callers requiring larger transfers must schedule bounded collective rounds. */
[[nodiscard]] OwnerExchange exchange_owner_packets(
    std::span<const OwnerPacket>, Teuchos::RCP<const Teuchos::Comm<int>>, std::size_t byte_budget);
/** Consumes packet storage. Collective rounds containing only local deliveries
 * move their authenticated packets directly, admitting actual vector/payload
 * capacities before transfer. Remote rounds borrow the input through the
 * transport above, excluding its retained capacities from the transport scratch
 * budget, and release consumed source storage before returning. */
[[nodiscard]] OwnerExchange consume_owner_packets(
    std::vector<OwnerPacket>&&, Teuchos::RCP<const Teuchos::Comm<int>>, std::size_t byte_budget);
} // namespace SimpleFluid
