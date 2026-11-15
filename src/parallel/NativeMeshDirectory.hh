/** @file NativeMeshDirectory.hh
 *  @brief Stable routing for sparse native source and algebraic identifiers.
 */
#pragma once

#include <cstdint>

namespace SimpleFluid::native_partition_detail
{
/** Mix all source-ID bits before reducing to a directory rank (SplitMix64). */
inline std::uint64_t directory_hash(std::uint64_t id) noexcept
{
    id += UINT64_C(0x9e3779b97f4a7c15);
    id = (id ^ (id >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    id = (id ^ (id >> 27)) * UINT64_C(0x94d049bb133111eb);
    return id ^ (id >> 31);
}

/** @pre ranks is a positive communicator size. */
inline int directory_rank(std::uint64_t id, int ranks) noexcept
{
    return static_cast<int>(directory_hash(id) % static_cast<std::uint64_t>(ranks));
}
} // namespace SimpleFluid::native_partition_detail
