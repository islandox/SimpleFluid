#include "geometry/PLIC/GeometryPacketCodec.hh"
#include <stdexcept>
namespace SimpleFluid::PLIC
{
namespace
{
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
} // namespace
std::vector<char> encode_geometry_packet(
    GeometryPacketKind kind, ArithmeticMode mode, std::span<const char> body, std::size_t budget)
{
    mode_check(mode);
    if (!supported(kind) || budget < geometry_packet_header_bytes ||
        body.size() > budget - geometry_packet_header_bytes)
        throw std::invalid_argument("Geometry packet exceeds its kind or byte budget.");
    std::vector<char> bytes;
    bytes.reserve(geometry_packet_header_bytes + body.size());
    const auto word = [&](std::uint64_t value)
    {
        for (unsigned i = 0; i < 8; ++i)
            bytes.push_back(static_cast<char>(value >> (8 * i)));
    };
    word(geometry_packet_magic);
    bytes.push_back(static_cast<char>(geometry_packet_version));
    bytes.push_back(static_cast<char>(kind));
    bytes.push_back(static_cast<char>(mode));
    word(geometry_packet_profile(mode));
    word(body.size());
    bytes.insert(bytes.end(), body.begin(), body.end());
    return bytes;
}
GeometryPacketView decode_geometry_packet(std::span<const char> bytes, ArithmeticMode mode, std::size_t budget)
{
    mode_check(mode);
    if (bytes.size() > budget || bytes.size() < geometry_packet_header_bytes)
        throw std::invalid_argument("Geometry packet has a truncated header or exceeds its byte budget.");
    std::size_t cursor = 0;
    const auto word = [&]()
    {
        std::uint64_t value = 0;
        for (unsigned i = 0; i < 8; ++i)
            value |= std::uint64_t(static_cast<unsigned char>(bytes[cursor++])) << (8 * i);
        return value;
    };
    const auto byte = [&]() { return static_cast<std::uint8_t>(bytes[cursor++]); };
    if (word() != geometry_packet_magic || byte() != geometry_packet_version)
        throw std::invalid_argument("Unknown geometry packet magic or version.");
    const auto kind = static_cast<GeometryPacketKind>(byte());
    if (!supported(kind) || byte() != static_cast<std::uint8_t>(mode) || word() != geometry_packet_profile(mode))
        throw std::invalid_argument("Foreign geometry packet kind, arithmetic mode or finite profile.");
    const auto length = word();
    if (length != bytes.size() - geometry_packet_header_bytes)
        throw std::invalid_argument("Geometry packet has a truncated body or trailing bytes.");
    return {kind, bytes.subspan(geometry_packet_header_bytes)};
}
} // namespace SimpleFluid::PLIC
