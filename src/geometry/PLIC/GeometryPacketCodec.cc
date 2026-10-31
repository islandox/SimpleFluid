#include "geometry/PLIC/GeometryPacketCodec.hh"
#include <array>
#include <bit>
#include <cstring>
#include <stdexcept>
namespace SimpleFluid::PLIC
{
namespace
{
using Header = GeometryPacketHeader;
constexpr auto header_fields = []
{
    std::array<bool, sizeof(Header)> fields{};
    const auto mark = [&](std::size_t offset, std::size_t size)
    {
        for (std::size_t i = offset; i < offset + size; ++i)
            fields[i] = true;
    };
    mark(offsetof(Header, family), sizeof(Header::family));
    mark(offsetof(Header, version), sizeof(Header::version));
    mark(offsetof(Header, kind), sizeof(Header::kind));
    mark(offsetof(Header, mode), sizeof(Header::mode));
    mark(offsetof(Header, profile), sizeof(Header::profile));
    mark(offsetof(Header, body_length), sizeof(Header::body_length));
    return fields;
}();
std::uint64_t little_endian(std::uint64_t value) noexcept
{
    static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big);
    if constexpr (std::endian::native == std::endian::big)
        return std::byteswap(value);
    return value;
}
bool supported(GeometryPacketKind kind)
{
    switch (kind)
    {
        case GeometryPacketKind::PointIdentity:
        case GeometryPacketKind::LineIdentity:
        case GeometryPacketKind::PlaneSupportIdentity:
        case GeometryPacketKind::DisplacementIdentity:
        case GeometryPacketKind::PlanarFace:
        case GeometryPacketKind::PlanarFaceIncidence:
        case GeometryPacketKind::CellCertificate:
        case GeometryPacketKind::ConvexFragment:
            return true;
    }
    return false;
}
void mode_check(ArithmeticMode mode)
{
    if (mode != ArithmeticMode::Ordinary && mode != ArithmeticMode::Exact)
        throw std::invalid_argument("Unsupported geometry packet arithmetic mode.");
}
GeometryNumericProfile supported_profile(ArithmeticMode mode)
{
    mode_check(mode);
    const auto profile = geometry_packet_profile(mode);
    if (profile == GeometryNumericProfile::Unsupported)
        throw std::invalid_argument("Unsupported finite numeric profile for geometry packets.");
    return profile;
}
} // namespace
std::vector<char> encode_geometry_packet(
    GeometryPacketKind kind, ArithmeticMode mode, std::span<const char> body, std::size_t budget)
{
    const auto profile = supported_profile(mode);
    if (!supported(kind) || budget < geometry_packet_header_bytes ||
        body.size() > budget - geometry_packet_header_bytes)
        throw std::invalid_argument("Geometry packet exceeds its kind or byte budget.");
    std::vector<char> bytes;
    bytes.reserve(geometry_packet_header_bytes + body.size());
    // Initialize serialized storage, not a struct whose padding may be
    // indeterminate after member assignments. Only member bytes are overwritten.
    bytes.resize(geometry_packet_header_bytes, 0);
    bytes[offsetof(Header, family)] = static_cast<char>(GeometryPacketFamily::Geometry);
    bytes[offsetof(Header, version)] = static_cast<char>(geometry_packet_version);
    bytes[offsetof(Header, kind)] = static_cast<char>(kind);
    bytes[offsetof(Header, mode)] = static_cast<char>(mode);
    bytes[offsetof(Header, profile)] = static_cast<char>(profile);
    const auto length = little_endian(static_cast<std::uint64_t>(body.size()));
    std::memcpy(bytes.data() + offsetof(Header, body_length), &length, sizeof(length));
    bytes.insert(bytes.end(), body.begin(), body.end());
    return bytes;
}
GeometryPacketView decode_geometry_packet(std::span<const char> bytes, ArithmeticMode mode, std::size_t budget)
{
    const auto expected_profile = supported_profile(mode);
    if (bytes.size() > budget || bytes.size() < geometry_packet_header_bytes)
        throw std::invalid_argument("Geometry packet has a truncated header or exceeds its byte budget.");
    const auto byte = [&](std::size_t offset) { return static_cast<std::uint8_t>(bytes[offset]); };
    if (byte(offsetof(Header, family)) != static_cast<std::uint8_t>(GeometryPacketFamily::Geometry) ||
        byte(offsetof(Header, version)) != geometry_packet_version)
        throw std::invalid_argument("Unknown geometry packet family or version.");
    const auto kind = static_cast<GeometryPacketKind>(byte(offsetof(Header, kind)));
    if (!supported(kind) || byte(offsetof(Header, mode)) != static_cast<std::uint8_t>(mode) ||
        byte(offsetof(Header, profile)) != static_cast<std::uint8_t>(expected_profile))
        throw std::invalid_argument("Foreign geometry packet kind, arithmetic mode or finite profile.");
    for (std::size_t i = 0; i < header_fields.size(); ++i)
        if (!header_fields[i] && bytes[i] != 0)
            throw std::invalid_argument("Nonzero geometry packet header padding.");
    // Variable-length enclosing records may provide an unaligned input span.
    std::uint64_t stored_length;
    std::memcpy(&stored_length, bytes.data() + offsetof(Header, body_length), sizeof(stored_length));
    const auto length = little_endian(stored_length);
    if (length != bytes.size() - geometry_packet_header_bytes)
        throw std::invalid_argument("Geometry packet has a truncated body or trailing bytes.");
    return {kind, bytes.subspan(geometry_packet_header_bytes)};
}
} // namespace SimpleFluid::PLIC
