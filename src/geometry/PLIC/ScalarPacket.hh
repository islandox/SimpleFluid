/** @file ScalarPacket.hh
 * @brief Lossless native scalar words shared by geometry and bounded transports.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include <array>
#include <cstdint>
#include <limits>
#include <span>
namespace SimpleFluid::PLIC
{
/** Sign, signed binary exponent, and 32-bit significand limbs. Callers framing
 * these words must authenticate native precision and exponent range. This is
 * a scalar representation only; it grants no geometric source provenance. */
using ScalarPacketWords = std::array<std::uint64_t, 2 + (std::numeric_limits<long double>::digits + 31) / 32>;
[[nodiscard]] SIMPLEFLUID_LOCAL ScalarPacketWords scalar_packet(long double);
[[nodiscard]] SIMPLEFLUID_LOCAL long double scalar_from_packet(std::span<const std::uint64_t>);
} // namespace SimpleFluid::PLIC
