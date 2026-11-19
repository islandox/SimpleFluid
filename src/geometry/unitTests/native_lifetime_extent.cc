#include "native_lifetime_extent.hh"
#include "pipeline_allocation_trace.hh"

namespace pipeline_trace
{
bool raw_extent_live(const void* address, std::size_t bytes) noexcept
{
    const Guard guard;
    for (const auto& entry : table)
        if (entry.live && entry.address == address && entry.bytes == bytes)
            return true;
    return false;
}
} // namespace pipeline_trace
