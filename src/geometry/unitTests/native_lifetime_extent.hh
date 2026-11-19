#pragma once
#include <cstddef>

namespace pipeline_trace
{
bool raw_extent_live(const void* address, std::size_t bytes) noexcept;
}
