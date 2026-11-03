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
    /// Live explicit owner capacities; borrowed source packets and Distributor,
    /// MPI, allocator bookkeeping/control blocks are outside this scratch scope.
    std::size_t planning_peak_bytes = 0, transfer_peak_bytes = 0, decode_peak_bytes = 0;
    std::size_t transport_rounds = 0, round_byte_limit = 0, largest_received_record = 0;
    std::size_t receive_block_capacity = 0, block_metadata_capacity = 0;
    bool segmented_receive = false;
};
struct OwnerExchange
{
    std::vector<OwnerPacket> packets;
    OwnerExchangeStats stats;
};
/** Collective, including empty ranks. One packed stream per actual peer uses a
 * Tpetra Distributor; packet sizes never pad another packet's payload. The byte
 * budget bounds the maximum live phase of packed buffers, output and explicit
 * framing/route metadata before allocations. Packed sends and export-only
 * grouping/route metadata are destroyed after transfer and before decoding.
 * Large exchanges use synchronized cyclic peer rounds with a total outgoing
 * and incoming block limit per rank (at most 16 MiB each), and retain exact
 * receive blocks in Distributor stream order. Complete inner framing is audited
 * collectively without decoded allocations; admission accounts for the next
 * whole packet while its source blocks remain live, and decoding retires
 * exhausted block owners. Small/tight cases retain the contiguous path when
 * its admitted peak is cheaper or block metadata would break a passing cap.
 * Stats count these buffers, not MPI/allocator/control-block RSS. Borrowed input
 * is excluded; diagnostics report its packet range and payload capacities.
 * Duplicate peers and zero-byte records are supported; decoding is collective.
 * A call still rejects total packed byte/record counts above INT_MAX before
 * payload exchange; internal rounds do not relax that admission contract. */
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
