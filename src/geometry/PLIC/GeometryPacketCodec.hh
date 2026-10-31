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
#include <type_traits>
#include <vector>
namespace SimpleFluid::PLIC
{
/** Framing validates metadata only; owning decoders validate bodies. */
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
inline constexpr std::size_t geometry_packet_default_budget = 64 * 1024 * 1024;
inline constexpr std::uint8_t geometry_packet_version = 3;
/** The serialized header follows this natural C++ layout, including zeroed
 * inter-member and tail padding. Endpoints require compatible header layouts.
 * Body length remains little endian; arbitrary input addresses need not align. */
struct GeometryPacketHeader
{
    GeometryPacketFamily family;
    std::uint8_t version;
    GeometryPacketKind kind;
    ArithmeticMode mode;
    GeometryNumericProfile profile;
    std::uint64_t body_length;
};
static_assert(sizeof(ArithmeticMode) == 1);
static_assert(std::is_standard_layout_v<GeometryPacketHeader> && std::is_trivially_copyable_v<GeometryPacketHeader>);
inline constexpr std::size_t geometry_packet_header_bytes = sizeof(GeometryPacketHeader);

/** An unknown format is never silently mapped to the closest known profile. */
[[nodiscard]] constexpr GeometryNumericProfile geometry_numeric_profile(
    int radix, int digits, int minimum_exponent, int maximum_exponent) noexcept
{
    if (radix == 2 && digits == 53 && minimum_exponent == -1021 && maximum_exponent == 1024)
        return GeometryNumericProfile::Binary64;
    if (radix == 2 && digits == 64 && minimum_exponent == -16381 && maximum_exponent == 16384)
        return GeometryNumericProfile::X87Binary80;
    if (radix == 2 && digits == 113 && minimum_exponent == -16381 && maximum_exponent == 16384)
        return GeometryNumericProfile::Binary128;
    return GeometryNumericProfile::Unsupported;
}
[[nodiscard]] constexpr GeometryNumericProfile geometry_packet_profile(ArithmeticMode mode) noexcept
{
    if (mode == ArithmeticMode::Exact)
        return GeometryNumericProfile::ExactRational;
    if (mode != ArithmeticMode::Ordinary)
        return GeometryNumericProfile::Unsupported;
    using Profile = std::numeric_limits<long double>;
    return geometry_numeric_profile(Profile::radix, Profile::digits, Profile::min_exponent, Profile::max_exponent);
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
