/** @file GeometryWireProtocol.hh
 * @brief Versioned entry stamps for phase records carrying geometry packets.
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace SimpleFluid::VoF::geometry_wire
{
enum class Family : std::uint8_t
{
    VoFGeometry = 3
};
inline constexpr std::uint8_t version = 3;
enum class Record : std::uint8_t
{
    NativeShellRequest = 1,
    NativeShellReply,
    ExactTileRequest,
    ExactTileReply,
    ExactFaceOutput,
    ExactControls,
    NodeUnion,
    NodePublication,
    ExactAtomRoute
};
struct Header
{
    Family family;
    std::uint8_t version;
    Record record;
};
static_assert(std::is_standard_layout_v<Header> && std::is_trivially_copyable_v<Header>);
inline constexpr std::size_t header_bytes = sizeof(Header);
template<class Writer> void write_header(Writer& writer, Record record)
{
    std::array<std::uint8_t, header_bytes> bytes{};
    bytes[offsetof(Header, family)] = static_cast<std::uint8_t>(Family::VoFGeometry);
    bytes[offsetof(Header, version)] = version;
    bytes[offsetof(Header, record)] = static_cast<std::uint8_t>(record);
    for (const auto byte : bytes)
        writer.byte(byte);
}
template<class Reader> void read_header(Reader& reader, Record record)
{
    std::array<std::uint8_t, header_bytes> bytes{};
    for (auto& byte : bytes)
        byte = reader.byte();
    if (bytes[offsetof(Header, family)] != static_cast<std::uint8_t>(Family::VoFGeometry) ||
        bytes[offsetof(Header, version)] != version ||
        bytes[offsetof(Header, record)] != static_cast<std::uint8_t>(record))
        throw std::invalid_argument("VoF geometry record family, version or kind differs.");
    for (std::size_t i = 0; i < header_bytes; ++i)
        if (i != offsetof(Header, family) && i != offsetof(Header, version) && i != offsetof(Header, record) && bytes[i])
            throw std::invalid_argument("Nonzero VoF geometry record header padding.");
}
} // namespace SimpleFluid::VoF::geometry_wire
