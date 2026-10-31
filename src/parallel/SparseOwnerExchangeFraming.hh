/** @file SparseOwnerExchangeFraming.hh
 * @brief Private allocation-free owner-stream framing audit and traversal.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include <cstddef>
#include <cstdint>
#include <span>

namespace SimpleFluid::owner_exchange_detail
{
struct StreamAudit
{
    std::size_t records = 0, payload_bytes = 0;
};
/** Headers contain (record count, framed byte count) for each stream. A null
 * visitor validates every complete stream without allocating output payloads.
 * Decoding calls the same traversal only after that full audit succeeds on all
 * ranks. Visitor spans borrow the packed bytes for the duration of the call. */
[[nodiscard]] SIMPLEFLUID_LOCAL StreamAudit walk_owner_streams(
    std::span<const char> packed, std::span<const std::size_t> stream_sizes,
    std::span<const std::uint64_t> headers, void* context = nullptr,
    void (*visitor)(void*, std::size_t stream, std::span<const char>) = nullptr);
} // namespace SimpleFluid::owner_exchange_detail
