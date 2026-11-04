#pragma once
#include "CompleteLineQuery.hh"
#include "IncidencePools.hh"
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct PooledLineSubscriber
{
    using Slot = std::size_t;
    enum class ResultState : std::uint8_t
    {
        Unset,
        ForwardEndpoints,
        Explicit
    };
    Slot support = 0, first = 0, last = 0;
    int resident = -1, support_authority = -1;
    std::uint64_t emission = 0;
    std::uint8_t image = 0;
    ResultState result_state = ResultState::Unset;
    std::uint64_t edge = 0;
    // Registry consumers use the accessors: a cached forward-endpoint answer
    // owns no vector. Decoded wire replies retain their explicit vector.
    std::vector<Slot> result;
    [[nodiscard]] bool has_result() const noexcept { return result_state != ResultState::Unset; }
    [[nodiscard]] std::size_t result_size() const noexcept
    {
        return result_state == ResultState::ForwardEndpoints ? 2 : result.size();
    }
    [[nodiscard]] Slot result_at(std::size_t index) const
    {
        if (result_state == ResultState::ForwardEndpoints)
        {
            if (index >= 2)
                throw std::out_of_range("Invalid implicit line endpoint index.");
            return index ? last : first;
        }
        return result.at(index);
    }
};
/** The result view is borrowed for the callback only. */
using PooledLineChange = std::function<void(std::size_t, std::span<const std::size_t>)>;
/** Full-identity order, independent of pool slot numbering. */
[[nodiscard]] bool line_subscriber_before(
    const PooledLineSubscriber&, const PooledLineSubscriber&, const IncidencePools&);
/** Evaluate one complete line and notify only actual changed exact query
 * results. The caller admits all inputs, retained outputs and scratch; ordinary
 * point identities have inline payloads and are retained by the typed pool. */
void query_pooled_line(std::span<PooledLineSubscriber>, IncidencePools&, const LineCharge&, const PooledLineChange&);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
