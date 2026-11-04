/** @file BudgetedOwnerExchange.hh
 * @brief Owner-packet exchange charged against an enclosing live arena.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Outgoing packet and vector capacities must already be charged to budget.
 * On success those owners are destroyed and their charge is replaced by the
 * actual received packet capacities. The result remains charged until retire. */
inline OwnerExchange consume_budgeted_owner_packets(std::vector<OwnerPacket>&& outgoing,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, Budget& budget, std::size_t outgoing_charge)
{
    const auto resident = budget.used;
    auto incoming = consume_owner_packets(std::move(outgoing), comm, budget.limit - budget.used);
    collective(*comm, [&]
        {
            if (incoming.stats.peak_scratch_bytes > budget.limit - resident || outgoing_charge > budget.used ||
                incoming.stats.peak_scratch_bytes > std::numeric_limits<std::size_t>::max() - budget.cumulative)
                throw std::invalid_argument("Owner transport exceeds its live byte credit.");
            budget.peak = std::max(budget.peak, resident + incoming.stats.peak_scratch_bytes);
            budget.cumulative += incoming.stats.peak_scratch_bytes;
            budget.used -= outgoing_charge;
            budget.add(incoming.packets.capacity() * sizeof(OwnerPacket));
            for (const auto& packet : incoming.packets)
                budget.add(packet.bytes.capacity());
        });
    return incoming;
}
inline void retire_budgeted_owner_packets(OwnerExchange& exchange, Budget& budget)
{
    std::size_t bytes = exchange.packets.capacity() * sizeof(OwnerPacket);
    for (const auto& packet : exchange.packets)
    {
        if (packet.bytes.capacity() > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::invalid_argument("Owner packet retirement size overflows.");
        bytes += packet.bytes.capacity();
    }
    {
        OwnerExchange empty;
        std::swap(exchange, empty);
    }
    budget.used -= bytes;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
