/** @file GeometryPacketCodec.hh
 * @brief Versioned geometry wire envelopes, separate from canonical keys.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/GeometryCodecTags.hh"
#include "utils/ArithmeticMode.hh"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>
namespace SimpleFluid::PLIC
{
/** One family signature (8 bytes), then version/kind/mode bytes, finite
 * profile (8 bytes) and body length (8 bytes), all words little endian.
 * Framing validates metadata only; owning decoders validate bodies. */
enum class GeometryPacketKind : std::uint8_t
{
    PointIdentity = 1,
    LineIdentity = 2,
    PlaneSupportIdentity = 3,
    DisplacementIdentity = 4,
    PlanarFace = 16,
    PlanarFaceIncidence = 17,
    CellCertificate = 18,
    ConvexFragment = 19
};
inline constexpr std::size_t geometry_packet_header_bytes = 3 * sizeof(std::uint64_t) + 3;
inline constexpr std::size_t geometry_packet_default_budget = 64 * 1024 * 1024;
inline constexpr std::uint64_t geometry_packet_magic = 0x534647454f4d5631ULL;
inline constexpr std::uint8_t geometry_packet_version = 2;
[[nodiscard]] constexpr std::uint64_t geometry_packet_profile(ArithmeticMode mode) noexcept
{
    if (mode == ArithmeticMode::Exact)
        return 0;
    using Profile = std::numeric_limits<long double>;
    return std::uint64_t(Profile::radix) | (std::uint64_t(Profile::digits) << 8) |
           (std::uint64_t(std::uint16_t(Profile::min_exponent)) << 16) |
           (std::uint64_t(std::uint16_t(Profile::max_exponent)) << 32);
}
/** The body borrows the caller's packet storage. */
struct GeometryPacketView
{
    GeometryPacketKind kind;
    std::span<const char> body;
};
[[nodiscard]] SIMPLEFLUID_LOCAL std::vector<char> encode_geometry_packet(
    GeometryPacketKind, ArithmeticMode, std::span<const char>, std::size_t = geometry_packet_default_budget);
[[nodiscard]] SIMPLEFLUID_LOCAL GeometryPacketView decode_geometry_packet(
    std::span<const char>, ArithmeticMode, std::size_t = geometry_packet_default_budget);
} // namespace SimpleFluid::PLIC
