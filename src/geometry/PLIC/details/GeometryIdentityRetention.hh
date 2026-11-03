/** Private pure capacity inspection, without identity serialization or arithmetic. */
#pragma once
#include "geometry/PLIC/details/GeometryIdentityBacking.hh"
#include <cstddef>
#include <limits>

namespace SimpleFluid::PLIC::details
{
struct IdentityRetentionBytes
{
    std::size_t metadata = 0, populated = 0, unused = 0;
    bool complete = true, overflow = false;

    /** Inline storage belongs to metadata. Only separately allocated buffers
     * contribute populated and unused bytes; allocator overhead is excluded. */
    void buffer(std::size_t size, std::size_t capacity, std::size_t width) noexcept
    {
        if (size > capacity)
        {
            complete = false;
            return;
        }
        accumulate(populated, size, width);
        accumulate(unused, capacity - size, width);
    }

private:
    void accumulate(std::size_t& total, std::size_t count, std::size_t width) noexcept
    {
        if (width && count > (std::numeric_limits<std::size_t>::max() - total) / width)
        {
            overflow = true;
            return;
        }
        total += count * width;
    }
};
[[nodiscard]] SIMPLEFLUID_LOCAL IdentityRetentionBytes ordinary_identity_retention_bytes(
    const GeometryIdentityNativePayload&) noexcept;
[[nodiscard]] SIMPLEFLUID_LOCAL IdentityRetentionBytes exact_identity_retention_bytes(
    const GeometryIdentityNativePayload&) noexcept;
} // namespace SimpleFluid::PLIC::details
