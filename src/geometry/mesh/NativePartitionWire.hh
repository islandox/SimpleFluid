/** @file NativePartitionWire.hh
 * @brief Native-layout entry header for internal multi-region partition packets.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace SimpleFluid::Meshes::partition_wire
{
enum class Family : std::uint8_t
{
    MultiRegionPartition = 2
};
inline constexpr std::uint8_t version = 10;
struct Header
{
    Family family;
    std::uint8_t version;
    std::uint64_t region_count;
};
static_assert(std::is_standard_layout_v<Header> && std::is_trivially_copyable_v<Header>);
} // namespace SimpleFluid::Meshes::partition_wire
