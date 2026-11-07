/** @file ContactAllocationScope.hh
 * @brief Construction identity for contact allocation directories.
 */
#pragma once
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
inline std::uint64_t next_contact_allocation_scope()
{
    static std::atomic<std::uint64_t> next{1};
    auto value = next.load(std::memory_order_relaxed);
    for (;;)
    {
        // The maximum value belongs to the incidence tile-body credit family.
        if (value == std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("Contact allocation scope exhausted.");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
            return value;
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
