/** @file SparseOwnerExchangeFraming.hh
 * @brief Private allocation-free owner-stream framing audit and traversal.
 */
#pragma once
#include "utils/export_helper.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace SimpleFluid::owner_exchange_detail
{
struct StreamAudit
{
    std::size_t records = 0, payload_bytes = 0, largest_payload = 0;
};
/** One stream's exact receive owners, in wire order. Blocks may divide either
 * a length word or a payload; they never change the framing protocol. */
struct OwnerStreamBlocks
{
    std::vector<std::vector<char>> blocks;
};
[[nodiscard]] SIMPLEFLUID_LOCAL StreamAudit walk_owner_block_streams(std::span<const OwnerStreamBlocks>,
    std::span<const std::size_t> stream_sizes, std::span<const std::uint64_t> headers);
/** Private validation seam: a nondefault limit requires the segmented path.
 * The limit is a maximum; collective transfer/decode admission may halve it. */
[[nodiscard]] SIMPLEFLUID_LOCAL OwnerExchange exchange_owner_packets_with_round_limit(std::span<const OwnerPacket>,
    Teuchos::RCP<const Teuchos::Comm<int>>, std::size_t byte_budget, std::size_t round_limit);
/** Collective audit/admission precedes every output allocation. Retained
 * metadata counts the caller's live route/header/block-owner capacities and
 * excludes the output packet array, which this helper admits itself. The
 * observer is a private test seam, called immediately before each nonempty
 * payload allocation. Successful decode releases exhausted raw block owners. */
[[nodiscard]] SIMPLEFLUID_LOCAL OwnerExchange decode_owner_block_streams(std::span<OwnerStreamBlocks>,
    std::span<const std::size_t> stream_sizes, std::span<const std::uint64_t> headers, std::span<const int> senders,
    Teuchos::RCP<const Teuchos::Comm<int>>, std::size_t byte_budget, std::size_t retained_metadata,
    void* observer_context = nullptr, void (*before_payload_allocation)(void*, std::size_t) = nullptr);
/** Headers contain (record count, framed byte count) for each stream. A null
 * visitor validates every complete stream without allocating output payloads.
 * Decoding calls the same traversal only after that full audit succeeds on all
 * ranks. Visitor spans borrow the packed bytes for the duration of the call. */
[[nodiscard]] SIMPLEFLUID_LOCAL StreamAudit walk_owner_streams(std::span<const char> packed,
    std::span<const std::size_t> stream_sizes, std::span<const std::uint64_t> headers, void* context = nullptr,
    void (*visitor)(void*, std::size_t stream, std::span<const char>) = nullptr);
} // namespace SimpleFluid::owner_exchange_detail
